from Models import *

class DSSM_AAE_PR():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), kl_rate=0.01, keep_rate_word_dropout=0.5, enableKL=True, enableCond=False, enableWasserstein=False, mode=1, enableSemi=False, limit=1):

        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.keep_rate_word_dropout = keep_rate_word_dropout
        self.kl_rate = kl_rate
        self.enableKL = enableKL
        self.enableCond = enableCond
        self.num_negatives = 1
        self.hidden_dim = self.dim[0]
        self.latent_dim = self.dim[1]
        self.mode = mode
        self.enableSemi = enableSemi
        self.enableWasserstein = enableWasserstein
        self.limit = limit

        # if self.mode == 1:
        #     self.adversarial_optimizer = AdversarialOptimizerSimultaneous()
        # elif self.mode == 2:
        #     self.adversarial_optimizer = AdversarialOptimizerAlternating()
        # elif self.mode == 3:
        #     self.adversarial_optimizer = AdversarialOptimizerScheduled([0,0,0,0,0,1])

        self.adversarial_optimizer = AdversarialOptimizerAlternating()

        self.dis_loss = "binary_crossentropy" if not self.enableWasserstein else self.wasserstein_loss
        self.build()

    def build(self):

        query_inputs = Input(shape=(self.max_len,))
        doc_inputs = Input(shape=(self.max_len,))
        neg_doc_inputs = Input(shape=(self.max_len,))


        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=False,
                                        name="q_embedding",
                                        trainable=True)

        doc_encoder_embedding = encoder_embedding


        encoder_lstm = Bidirectional(GRU(self.hidden_dim, return_sequences=True), name='q_gru')
        doc_encoder_lstm = encoder_lstm

        dense = Dense(self.latent_dim, activation="tanh", name="q_dense")


        state = dense(GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(query_inputs))))
        doc_state = dense(GlobalMaxPooling1D()(doc_encoder_lstm(doc_encoder_embedding(doc_inputs))))
        neg_doc_state = dense(GlobalMaxPooling1D()(doc_encoder_lstm(doc_encoder_embedding(neg_doc_inputs))))



        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)


        self.state_mean = self.mean(state)
        self.state_var = self.var(state)

        state_z = Lambda(self.sampling)([self.state_mean, self.state_var])



        self.pos_doc_state_mean = self.mean(doc_state)
        self.pos_doc_state_var = self.var(doc_state)
        pos_doc_state_z = Lambda(self.sampling)([self.pos_doc_state_mean, self.pos_doc_state_var])

        self.neg_doc_state_mean = self.mean(neg_doc_state)
        self.neg_doc_state_var = self.var(neg_doc_state)
        neg_doc_state_z = Lambda(self.sampling)([self.neg_doc_state_mean, self.neg_doc_state_var])

        query_sem = state
        pos_doc_sem = doc_state
        neg_doc_sem = neg_doc_state

                # DSSM model
        weight = np.array([1]).reshape(1, 1, 1)
        conv = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])
        R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
        R_Q_D_ns = dot([query_sem, neg_doc_sem], axes = 1, normalize = True) # See equation (4).
        concat_Rs = concatenate([R_Q_D_p, R_Q_D_ns])
        concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)
        with_gamma = conv(concat_Rs) # See equation (5).
        with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)
        pairwise_pred = Activation("softmax")(with_gamma) # See equation (5).

        # Adversarial
        self.discriminator = self.build_discriminator()
        gs_latents = normal_latent_sampling((self.latent_dim,))(query_inputs)
        # pos_doc_gs_latents = normal_latent_sampling((self.latent_dim,))(doc_inputs)
        # neg_doc_gs_latents = normal_latent_sampling((self.latent_dim,))(neg_doc_inputs)        


        query_real = self.discriminator(gs_latents)
        query_fake = self.discriminator(state_z)

        
        self.pr_discriminator = self.build_pr_discriminator()

        query_pr_latent = self.pr_discriminator(query_sem)
        pos_doc_pr_latent = self.pr_discriminator(pos_doc_sem)
        neg_doc_pr_latent = self.pr_discriminator(neg_doc_sem)


        pr_weight = np.array([1]).reshape(1, 1, 1)
        pr_conv = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [pr_weight])
        pr_R_Q_D_p = dot([query_pr_latent, pos_doc_pr_latent], axes = 1, normalize = True) # See equation (4).
        pr_R_Q_D_ns = dot([query_pr_latent, neg_doc_pr_latent], axes = 1, normalize = True) # See equation (4).
        pr_concat_Rs = concatenate([pr_R_Q_D_p, pr_R_Q_D_ns])
        pr_concat_Rs = Reshape((self.num_negatives + 1, 1))(pr_concat_Rs)
        pr_with_gamma = conv(pr_concat_Rs) # See equation (5).
        pr_with_gamma = Reshape((self.num_negatives + 1, ))(pr_with_gamma)
        pr_pairwise_pred = Activation("softmax")(pr_with_gamma) # See equation (5).

        

        decoder_inputs = Input(shape=(self.max_len,))
        decoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        trainable=True)
        # pos_doc_decoder_inputs = Input(shape=(self.max_len,))
        # neg_doc_decoder_inputs = Input(shape=(self.max_len,))


        self.latent2hidden = Dense(self.hidden_dim)
        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru")
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")

        # self.doc_latent2hidden = Dense(self.hidden_dim)
        # self.doc_decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru2")
        # self.doc_decoder_dense = Dense(self.nb_words, activation='softmax', name="rec2")
        

        concat = merge([decoder_embedding(decoder_inputs), RepeatVector(self.max_len)(self.latent2hidden(state_z))], mode="concat")
        rec_outputs = self.decoder_dense(self.decoder_lstm(concat))


        pr_concat = merge([decoder_embedding(decoder_inputs), RepeatVector(self.max_len)(self.latent2hidden(query_pr_latent))], mode="concat")
        pr_rec_outputs = self.decoder_dense(self.decoder_lstm(pr_concat))

        # pos_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(pos_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(pos_doc_state_z)))
        # neg_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(neg_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(neg_doc_state_z)))

        # self.ae = Model([query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs, pos_doc_decoder_inputs, neg_doc_decoder_inputs], [rec_outputs, pos_doc_rec_outputs, neg_doc_rec_outputs, pairwise_pred])

        rec_loss = "sparse_categorical_crossentropy"
        pair_loss = "categorical_crossentropy"

        model_loss_weights = [{"loss_weights": {"qfake": 1e-2, "qreal": 1e-2, "qpred": 1e-1, "prreal": -1e-2, "prfake": 1e-2, "pair": 1}}, {"loss_weights": {"qfake": 1e-2, "qreal": 1e-2, "qpred": 1e-1, "prreal": 1e-2, "prfake": -1e-2, "pair": 1}}]

        self.ae = Model([query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs], [rec_outputs, pairwise_pred])

        inputs = self.ae.inputs
        outputs = fix_names([rec_outputs, pairwise_pred, query_fake, query_real, pr_rec_outputs, pr_pairwise_pred], ["qpred", "pair","qfake","qreal","prfake","prreal"])
        
        self.aae = Model(inputs, outputs)

        generative_params = self.ae.trainable_weights
        discriminative_params = self.discriminator.trainable_weights
        # self.aae.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.vae_loss, self.vae_loss, "categorical_crossentropy"], loss_weights=[1e-4, 1e-4, 1e-4, 1])        
        self.model = AdversarialModel(base_model=self.aae, player_params=[generative_params, discriminative_params], player_names=["generator", "discriminator"])        

        self.model.adversarial_compile(adversarial_optimizer=self.adversarial_optimizer, player_optimizers=[self.optimizer, self.optimizer], loss={"qfake": self.dis_loss, "qreal": self.dis_loss, "qpred": rec_loss,"pair":pair_loss, "prfake": rec_loss, "prreal":pair_loss}, player_compile_kwargs=model_loss_weights)
        
        self.encoder = Model(query_inputs, state)

    def name(self):
        return "dssm_pra" if not self.enableWasserstein else "dssm_prw"
    
    def word_dropout(self, x, unk_token):
        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

            return x_

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_discriminator(self):
        
        inputs = Input((self.latent_dim,), name="gs_dis_input")
        
        dense1 = Dense(self.hidden_dim)
        dense2 = Dense(self.latent_dim)
        dense3 = Dense(1, activation="sigmoid" if not self.enableWasserstein else "linear")

        outputs = dense3(inputs)
        # outputs = dense3(dense2(dense1(inputs)))


        return Model(inputs, outputs)

    def build_pr_discriminator(self):
        
        inputs = Input((self.latent_dim,), name="gs_dis_input")
        
        dense1 = Dense(self.hidden_dim)
        dense2 = Dense(self.latent_dim)

        outputs = dense2(inputs)
        
        return Model(inputs, outputs)

    def sampling(self, args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                      stddev=1)
            return z_mean + K.exp(z_log_var / 2) * epsilon

