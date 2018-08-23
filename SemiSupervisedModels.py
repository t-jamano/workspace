from Models import *

class SS_PR():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), keep_rate_word_dropout=0.5, mode=1,enableWasserstein=False, enableGRU=False, enableS2S=False):
        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.enableWasserstein = enableWasserstein
        self.enableS2S = enableS2S
        self.num_negatives = 1
        self.mode = mode
        self.enableGRU=enableGRU
        self.keep_rate_word_dropout = keep_rate_word_dropout

        
        self.hidden_dim = self.dim[0]
        self.latent_dim = self.dim[1]

        self.dis_loss = "binary_crossentropy" if not self.enableWasserstein else self.wasserstein_loss
        self.gan_loss = "binary_crossentropy" if not self.enableGRU else "sparse_categorical_crossentropy"
        
        self.build()


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    def build(self):

        q_inputs = Input(shape=(self.max_len,))
        pd_inputs = Input(shape=(self.max_len,))
        nd_inputs = Input(shape=(self.max_len,))

        
        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        doc_encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        encoder_gru = GRU(self.hidden_dim)
        doc_encoder_gru = GRU(self.hidden_dim)



        hidden2latent = Dense(self.latent_dim)
        doc_hidden2latent = Dense(self.latent_dim)


        query_sem = hidden2latent(encoder_gru(encoder_embedding(q_inputs)))
        pos_doc_sem = doc_hidden2latent(doc_encoder_gru(doc_encoder_embedding(pd_inputs)))
        neg_doc_sem = doc_hidden2latent(doc_encoder_gru(doc_encoder_embedding(nd_inputs)))

        self.encoder = Model(q_inputs, query_sem)
        self.pd_encoder = Model(pd_inputs, pos_doc_sem)
        self.nd_encoder = Model(nd_inputs, neg_doc_sem)

        # DSSM model
        weight = np.array([1]).reshape(1, 1, 1)
        conv = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])
        R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
        R_Q_D_ns = dot([query_sem, neg_doc_sem], axes = 1, normalize = True) # See equation (4).
        concat_Rs = concatenate([R_Q_D_p, R_Q_D_ns])
        concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)
        with_gamma = conv(concat_Rs) # See equation (5).
        with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)
        pairwise_pred = Activation("softmax", name="pair")(with_gamma) # See equation (5).
        
        self.model = Model([q_inputs, pd_inputs, nd_inputs], pairwise_pred)
        self.model.compile(optimizer=self.optimizer, loss=["categorical_crossentropy"])


    
        mean = Dense(self.latent_dim)
        var = Dense(self.latent_dim)

        doc_mean = Dense(self.latent_dim)
        doc_var = Dense(self.latent_dim)

        
        query_sem_z = Lambda(self.sampling)([mean(query_sem), var(query_sem)])
        pos_doc_sem_z = Lambda(self.sampling)([doc_mean(pos_doc_sem), doc_var(pos_doc_sem)])
        neg_doc_sem_z = Lambda(self.sampling)([doc_mean(neg_doc_sem), doc_var(neg_doc_sem)])

        self.q_gs_encoder = Model(q_inputs, query_sem_z)
        self.pd_gs_encoder = Model(pd_inputs, pos_doc_sem_z)
        self.nd_gs_encoder = Model(nd_inputs, neg_doc_sem_z)

        
        dec_q_inputs = Input(shape=(self.max_len,))
        dec_pd_inputs = Input(shape=(self.max_len,))
        dec_nd_inputs = Input(shape=(self.max_len,))

        latent2hidden = Dense(self.hidden_dim)
        doc_latent2hidden = Dense(self.hidden_dim)

        decoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)
        doc_decoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        decoder_gru = GRU(self.hidden_dim, return_sequences=True)
        decoder_dense = Dense(self.nb_words, activation='softmax')

        doc_decoder_gru = GRU(self.hidden_dim, return_sequences=True)
        doc_decoder_dense = Dense(self.nb_words, activation='softmax')


        q_rec_pred = decoder_dense(decoder_gru(decoder_embedding(dec_q_inputs), initial_state=latent2hidden(query_sem_z)))
        pd_rec_pred = doc_decoder_dense(doc_decoder_gru(doc_decoder_embedding(dec_pd_inputs), initial_state=doc_latent2hidden(pos_doc_sem_z)))
        nd_rec_pred = doc_decoder_dense(doc_decoder_gru(doc_decoder_embedding(dec_nd_inputs), initial_state=doc_latent2hidden(neg_doc_sem_z)))

        

        # Generate Discriminators
        self.discriminator = self.build_gs_discriminator()
        self.discriminator.compile(optimizer=Adam(), loss=self.dis_loss, metrics=['accuracy'])
        self.discriminator.trainable = False

        self.doc_discriminator = self.build_gs_discriminator()
        self.doc_discriminator.compile(optimizer=Adam(), loss=self.dis_loss, metrics=['accuracy'])
        self.doc_discriminator.trainable = False



        q_aae_penalty = self.discriminator(self.q_gs_encoder(q_inputs))
        self.q_ae = Model([q_inputs, dec_q_inputs], [q_rec_pred, q_aae_penalty])
        self.q_ae.compile(optimizer=self.optimizer, loss=["sparse_categorical_crossentropy", self.dis_loss] , loss_weights=[1e-3, 1e-7])
        
        # self.pd_ae = Model([pd_inputs, dec_pd_inputs], pd_rec_pred)
        # self.nd_ae = Model([nd_inputs, dec_nd_inputs], nd_rec_pred)


    def build_gs_discriminator(self):
        
        inputs = Input((self.latent_dim,), name="gs_dis_input")
        
        dense1 = Dense(self.hidden_dim)
        dense2 = Dense(self.latent_dim)
        dense3 = Dense(1, activation="sigmoid" if not self.enableWasserstein else "linear")

        outputs = dense3(dense2(dense1(inputs)))
        
        return Model(inputs, outputs)
    
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                  stddev=1)
        return z_mean + K.exp(z_log_var / 2) * epsilon 

    def name(self):
        if self.enableGRU:
            return "ss_aae" if not self.enableWasserstein else "ss_wae"
        # else:
            # return "bow_pr_avg_aae_m%d" % self.mode if not self.enableWasserstein else "bow_pr_avg_wae_m%d" % self.mode


    def word_dropout(self, x, unk_token):
        np.random.seed(0)
        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

        return x_