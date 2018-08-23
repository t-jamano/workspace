from Models import *
from vae import *


class AdversarialAutoEncoder(VariationalAutoEncoder):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), keep_rate_word_dropout=0.5, enableWasserstein=False):

        self.enableWasserstein = enableWasserstein
        self.dis_loss = "binary_crossentropy" if not self.enableWasserstein else self.wasserstein_loss

        VariationalAutoEncoder.__init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=optimizer, keep_rate_word_dropout=keep_rate_word_dropout)

        self.build()


    def build(self):

        self.ae, self.gs_encoder, self.encoder = self.build_ae()
        self.discriminator = self.build_gs_discriminator()
        self.discriminator.compile(optimizer=Adam(), loss=self.dis_loss, metrics=['accuracy'])
        self.discriminator.trainable = False

        inputs = self.ae.inputs
        rec_pred = self.ae(inputs)
        aae_penalty = self.discriminator(self.gs_encoder(inputs[0]))
            
        self.model = Model(inputs, [rec_pred, aae_penalty])
        self.model.compile(optimizer=Adam(), loss=["sparse_categorical_crossentropy", self.dis_loss], loss_weights=[1, 1e-1])
        
        
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    def build_ae(self):

        encoder_inputs = Input(shape=(self.max_len,))
        self.encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="q_embedding_layer",
                                        mask_zero=True,
                                        trainable=True)

        self.encoder_lstm = GRU(self.hidden_dim, name="q_gru")
        # self.encoder_lstm = GlobalAveragePooling1D()
        norm = BatchNormalization()



        x = self.encoder_embedding(encoder_inputs)
        self.state = norm(self.encoder_lstm(x))

        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)

        state_mean = self.mean(self.state)
        state_var = self.var(self.state)

        state_z = Lambda(self.sampling, name="kl")([state_mean, state_var])


        decoder_inputs = Input(shape=(self.max_len,), name="dec_input")

        self.latent2hidden = Dense(self.hidden_dim)
        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True)
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")
        # self.decoder_embedding = Embedding(self.nb_words,
        #                                 self.embedding_matrix.shape[-1],
        #                                 weights=[self.embedding_matrix],
        #                                 input_length=self.max_len,
        #                                 name="dec_embedding",
        #                                 mask_zero=True,
        #                                 trainable=True)

        x = self.encoder_embedding(decoder_inputs)
        decoder_outputs = self.decoder_lstm(x, initial_state=self.latent2hidden(state_z))
        rec_outputs = self.decoder_dense(decoder_outputs)

        # Encoder's output : state, state_mean
        return Model([encoder_inputs, decoder_inputs], rec_outputs), Model(encoder_inputs, state_z), Model(encoder_inputs, self.state)


    
    def build_gs_discriminator(self):
        
        inputs = Input((self.latent_dim,))
        
        dense1 = Dense(self.hidden_dim)
        dense2 = Dense(self.latent_dim)
        dense3 = Dense(1, activation="sigmoid" if not self.enableWasserstein else "linear")

        # outputs = dense3(dense2(dense1(inputs)))
        # outputs = dense3(dense2(inputs))
        outputs = dense3(inputs)

        
        return Model(inputs, outputs)

    def get_training_data(self, path, train_data, bpe_dict):
        q_enc_inputs = np.load('%sdata/train_data/%s.q.npy' % (path,train_data))
        q_dec_inputs = np.load('%sdata/train_data/%s.q.di.npy' % (path,train_data))
        q_dec_outputs = np.load('%sdata/train_data/%s.q.do.npy' % (path,train_data))
        num = len(q_enc_inputs)
        
        valid = np.ones(num)
        fake = np.zeros(num) if not self.enableWasserstein else -valid
        
        idx = np.arange(num)
        shuffle(idx)

        x_train = [q_enc_inputs[idx], self.word_dropout(q_dec_inputs[idx], bpe_dict['<unk>'])]
        y_train = [np.expand_dims(q_dec_outputs[idx], axis=-1), valid]

        return x_train, y_train, valid, fake

    def train(self, path, train_data, batch_size, bpe_dict, test_set):

        TRAINING_RATIO = 5

        max_val_loss = float("inf") 
        x_train, y_train, valid, fake = self.get_training_data(path, train_data, bpe_dict)

        for epoch in range(100):

            # minibatches_size = batch_size * TRAINING_RATIO

            # for i in range(int(x_train[0].shape[0] // (batch_size * TRAINING_RATIO))):

            #     x_ = [k[i * minibatches_size:(i + 1) * minibatches_size] for k in x_train]
            #     y_ = [k[i * minibatches_size:(i + 1) * minibatches_size] for k in y_train]

            #     for j in range(TRAINING_RATIO):
            #         x__ = [k[j * batch_size:(j + 1) * batch_size] for k in x_]
            #         y__ = [k[j * batch_size:(j + 1) * batch_size] for k in y_]

            #         hist = self.model.train_on_batch(x__, y__)
            hist = self.model.fit(x_train, y_train, batch_size=batch_size, verbose=1, shuffle=False, validation_split=0.2)
            latent_fake = self.gs_encoder.predict(x_[0])
            latent_real = np.random.normal(size=(len(x_[0]), self.latent_dim))

            dis_x = np.concatenate([latent_fake, latent_real])
            dis_y = np.concatenate([fake[:len(x_train[0])], valid[:len(x_[0])]])

            dis_hist = self.discriminator.train_on_batch(dis_x, dis_y)
            # d_loss = hist.history['loss']
            # d_acc = hist.history['acc']

            # d_loss_real = self.discriminator.fit(latent_real, valid, batch_size=batch_size, verbose=1)
            # d_loss_fake = self.discriminator.fit(latent_fake, fake, batch_size=batch_size, verbose=1)
            # d_loss = 0.5 * np.add(d_loss_real.history['loss'], d_loss_fake.history['loss'])
            # d_acc = 0.5 * np.add(d_loss_real.history['acc'], d_loss_fake.history['acc'])
            # print(d_loss, d_acc)

            # hist = self.model.fit(x_train, y_train,
            #                                 shuffle=True,
            #                                 verbose=1,
            #                                 batch_size=batch_size,
            #                                 validation_split=0.2,
            #                                 callbacks=[EarlyStopping()]
            #                                 )
            # if max_val_loss < hist.history["val_loss"][0]:
            #     may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = evaluate(self.encoder, test_set)
            #     print(may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc)
            #     break
            # else:
            #     max_val_loss = hist.history["val_loss"][0]
            # may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = evaluate(self.encoder, test_set)
            # print(epoch, may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc)
            # generate_reconstruct_query(self.model, bpe, [x_train[0][:5], x_train[1][:5]])

    def name(self):
        return "aae" if not self.enableWasserstein else "wae"

