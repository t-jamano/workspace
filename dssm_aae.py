from Models import *

class DSSM_AAE():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), keep_rate_word_dropout=0.5, enableWasserstein=False, mode=1, enableSemi=False, limit=1):

        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.keep_rate_word_dropout = keep_rate_word_dropout
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

        dense = Dense(self.hidden_dim, activation="tanh", name="q_dense")


        state = dense(GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(query_inputs))))
        doc_state = dense(GlobalMaxPooling1D()(doc_encoder_lstm(doc_encoder_embedding(doc_inputs))))
        neg_doc_state = dense(GlobalMaxPooling1D()(doc_encoder_lstm(doc_encoder_embedding(neg_doc_inputs))))



        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)


        self.state_mean = self.mean(state)
        self.state_var = self.var(state)

        state_z = Lambda(self.sampling)([self.state_mean, self.state_var])


        # self.pos_doc_state_mean = self.mean(doc_state)
        # self.pos_doc_state_var = self.var(doc_state)
        # pos_doc_state_z = Lambda(self.sampling)([self.pos_doc_state_mean, self.pos_doc_state_var])

        # self.neg_doc_state_mean = self.mean(neg_doc_state)
        # self.neg_doc_state_var = self.var(neg_doc_state)
        # neg_doc_state_z = Lambda(self.sampling)([self.neg_doc_state_mean, self.neg_doc_state_var])


        
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

        # pos_doc_real = self.discriminator(pos_doc_gs_latents)
        # pos_doc_fake = self.discriminator(pos_doc_state_z)

        # neg_doc_real = self.discriminator(neg_doc_gs_latents)
        # neg_doc_fake = self.discriminator(neg_doc_state_z)
        

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
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="rec", use_bias=False, weights=[self.embedding_matrix.T])

        # self.doc_latent2hidden = Dense(self.hidden_dim)
        # self.doc_decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru2")
        # self.doc_decoder_dense = Dense(self.nb_words, activation='softmax', name="rec2")
        
        rec_outputs = self.decoder_dense(self.decoder_lstm(decoder_embedding(decoder_inputs) , initial_state=self.latent2hidden(state_z)))
        # pos_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(pos_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(pos_doc_state_z)))
        # neg_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(neg_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(neg_doc_state_z)))

        # self.ae = Model([query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs, pos_doc_decoder_inputs, neg_doc_decoder_inputs], [rec_outputs, pos_doc_rec_outputs, neg_doc_rec_outputs, pairwise_pred])

        if self.enableSemi:
            semi_outputs = fix_names([rec_outputs, query_fake, query_real], ["qpred", "qfake", "qreal"])
            self.semi_aae = Model([query_inputs, decoder_inputs], semi_outputs)

            semi_generative_params = self.semi_aae.trainable_weights
            semi_discriminative_params = self.discriminator.trainable_weights
            self.semi_model = AdversarialModel(base_model=self.semi_aae, player_params=[semi_generative_params, semi_discriminative_params], player_names=["generator", "discriminator"])        
            rec_loss = "sparse_categorical_crossentropy"
            # original 1e-4, 1e-2
            # try 1e-2, 1e-1
            # try 1e-1, 0.5
            self.semi_model.adversarial_compile(adversarial_optimizer=self.adversarial_optimizer, player_optimizers=[self.optimizer, self.optimizer], loss={"qfake": self.dis_loss, "qreal": self.dis_loss, "qpred": rec_loss}, player_compile_kwargs=[{"loss_weights": {"qfake": 1e-4, "qreal": 1e-4, "qpred": 1e-2}}] * 2)
            self.model = Model([query_inputs, doc_inputs, neg_doc_inputs], [pairwise_pred])
            self.model.compile(optimizer=self.optimizer, loss=["categorical_crossentropy"])  
        else:
            self.ae = Model([query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs], [rec_outputs, pairwise_pred])

            inputs = self.ae.inputs
            # outputs = fix_names([rec_outputs, pos_doc_rec_outputs, neg_doc_rec_outputs, pairwise_pred, query_fake, query_real, pos_doc_fake, pos_doc_real, neg_doc_fake, neg_doc_real], ["qpred","dpred","ndpred","pair","qfake","qreal","dfake","dreal","ndfake","ndreal"])
            outputs = fix_names([rec_outputs, pairwise_pred, query_fake, query_real], ["qpred", "pair","qfake","qreal"])
            
            self.aae = Model(inputs, outputs)

            generative_params = self.ae.trainable_weights
            discriminative_params = self.discriminator.trainable_weights
            # self.aae.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.vae_loss, self.vae_loss, "categorical_crossentropy"], loss_weights=[1e-4, 1e-4, 1e-4, 1])        
            self.model = AdversarialModel(base_model=self.aae, player_params=[generative_params, discriminative_params], player_names=["generator", "discriminator"])        
            rec_loss = "sparse_categorical_crossentropy"
            pair_loss = "categorical_crossentropy"
            self.model.adversarial_compile(adversarial_optimizer=self.adversarial_optimizer, player_optimizers=[self.optimizer, self.optimizer], loss={"qfake": self.dis_loss, "qreal": self.dis_loss, "qpred": rec_loss,"pair":pair_loss}, player_compile_kwargs=[{"loss_weights": {"qfake": 1e-4, "qreal": 1e-4, "qpred": 1e-2, "pair": 1}}] * 2)
        
        # if self.mode == 1:
        #     self.encoder = Model(query_inputs, state)
        # elif self.mode == 2:
        #     self.encoder = Model(query_inputs, self.state_mean)
        # elif self.mode == 3:
        #     self.encoder = Model(query_inputs, state_z)

        self.encoder = Model(query_inputs, state)



    def name(self):
        if self.enableSemi:
            return "dssm_aae_s_%d" % self.limit if not self.enableWasserstein else "dssm_wae_s_%d" % self.limit
        return "dssm_aae_s2s" if not self.enableWasserstein else "dssm_wae"
    
    def word_dropout(self, x, unk_token):
        # np.random.seed(0)
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
        
        return Model(inputs, outputs)

    def sampling(self, args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                      stddev=1)
            return z_mean + K.exp(z_log_var / 2) * epsilon



class DSSM_AAE_SS():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), keep_rate_word_dropout=0.5, enableWasserstein=False, mode=1, enableSemi=False, limit=1):

        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.keep_rate_word_dropout = keep_rate_word_dropout
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
        query_inputs2 = Input(shape=(self.max_len,))
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

        dense = Dense(self.hidden_dim, activation="tanh", name="q_dense")


        state = dense(GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(query_inputs))))
        doc_state = dense(GlobalMaxPooling1D()(doc_encoder_lstm(doc_encoder_embedding(doc_inputs))))
        neg_doc_state = dense(GlobalMaxPooling1D()(doc_encoder_lstm(doc_encoder_embedding(neg_doc_inputs))))

        state2 = dense(GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(query_inputs2))))



        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)


        self.state_mean = self.mean(state2)
        self.state_var = self.var(state2)

        state_z = Lambda(self.sampling)([self.state_mean, self.state_var])


        # self.pos_doc_state_mean = self.mean(doc_state)
        # self.pos_doc_state_var = self.var(doc_state)
        # pos_doc_state_z = Lambda(self.sampling)([self.pos_doc_state_mean, self.pos_doc_state_var])

        # self.neg_doc_state_mean = self.mean(neg_doc_state)
        # self.neg_doc_state_var = self.var(neg_doc_state)
        # neg_doc_state_z = Lambda(self.sampling)([self.neg_doc_state_mean, self.neg_doc_state_var])


        
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

        # pos_doc_real = self.discriminator(pos_doc_gs_latents)
        # pos_doc_fake = self.discriminator(pos_doc_state_z)

        # neg_doc_real = self.discriminator(neg_doc_gs_latents)
        # neg_doc_fake = self.discriminator(neg_doc_state_z)
        

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
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="rec", use_bias=False, weights=[self.embedding_matrix.T])

        # self.doc_latent2hidden = Dense(self.hidden_dim)
        # self.doc_decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru2")
        # self.doc_decoder_dense = Dense(self.nb_words, activation='softmax', name="rec2")
        
        rec_outputs = self.decoder_dense(self.decoder_lstm(decoder_embedding(decoder_inputs) , initial_state=self.latent2hidden(state_z)))
        # pos_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(pos_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(pos_doc_state_z)))
        # neg_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(neg_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(neg_doc_state_z)))

        # self.ae = Model([query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs, pos_doc_decoder_inputs, neg_doc_decoder_inputs], [rec_outputs, pos_doc_rec_outputs, neg_doc_rec_outputs, pairwise_pred])

        self.ae = Model([query_inputs, doc_inputs, neg_doc_inputs, query_inputs2, decoder_inputs], [rec_outputs, pairwise_pred])

        inputs = self.ae.inputs
        # outputs = fix_names([rec_outputs, pos_doc_rec_outputs, neg_doc_rec_outputs, pairwise_pred, query_fake, query_real, pos_doc_fake, pos_doc_real, neg_doc_fake, neg_doc_real], ["qpred","dpred","ndpred","pair","qfake","qreal","dfake","dreal","ndfake","ndreal"])
        outputs = fix_names([rec_outputs, pairwise_pred, query_fake, query_real], ["qpred", "pair","qfake","qreal"])
        
        self.aae = Model(inputs, outputs)

        generative_params = self.ae.trainable_weights
        discriminative_params = self.discriminator.trainable_weights
        # self.aae.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.vae_loss, self.vae_loss, "categorical_crossentropy"], loss_weights=[1e-4, 1e-4, 1e-4, 1])        
        self.model = AdversarialModel(base_model=self.aae, player_params=[generative_params, discriminative_params], player_names=["generator", "discriminator"])        
        rec_loss = "sparse_categorical_crossentropy"
        pair_loss = "categorical_crossentropy"
        self.model.adversarial_compile(adversarial_optimizer=self.adversarial_optimizer, player_optimizers=[self.optimizer, self.optimizer], loss={"qfake": self.dis_loss, "qreal": self.dis_loss, "qpred": rec_loss,"pair":pair_loss}, player_compile_kwargs=[{"loss_weights": {"qfake": 1e-4, "qreal": 1e-4, "qpred": 1e-2, "pair": 1}}] * 2)
        
        self.encoder = Model(query_inputs, state)
        self.dssm = Model([query_inputs, doc_inputs, neg_doc_inputs], [pairwise_pred])
        self.dssm.compile(optimizer=self.optimizer, loss=["categorical_crossentropy"])  



    def name(self):
        return "dssm_aae_ss" if not self.enableWasserstein else "dssm_wae_ss"
    
    def word_dropout(self, x, unk_token):
        # np.random.seed(0)
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
        
        return Model(inputs, outputs)

    def sampling(self, args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                      stddev=1)
            return z_mean + K.exp(z_log_var / 2) * epsilon




class Binary_AAE():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), kl_rate=0.01, keep_rate_word_dropout=0.5, enableKL=True, enableCond=False, enableWasserstein=False, mode=1, enableSemi=False):

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

        if self.mode == 1:
            self.adversarial_optimizer = AdversarialOptimizerSimultaneous()
        elif self.mode == 2:
            self.adversarial_optimizer = AdversarialOptimizerAlternating()
        elif self.mode == 3:
            self.adversarial_optimizer = AdversarialOptimizerScheduled([0,0,0,0,0,1])

        self.dis_loss = "binary_crossentropy" if not self.enableWasserstein else self.wasserstein_loss
        self.build()

    def build(self):

        query_inputs = Input(shape=(self.max_len,))
        doc_inputs = Input(shape=(self.max_len,))


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


        state = GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(query_inputs)))
        doc_state = GlobalMaxPooling1D()(doc_encoder_lstm(doc_encoder_embedding(doc_inputs)))



        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)


        self.state_mean = self.mean(state)
        self.state_var = self.var(state)

        state_z = Lambda(self.sampling)([self.state_mean, self.state_var])
        # if self.mode == 1:
        query_sem = state
        pos_doc_sem = doc_state

        cos = Flatten()(merge([query_sem, pos_doc_sem], mode="cos"))


        # Adversarial
        self.discriminator = self.build_discriminator()
        gs_latents = normal_latent_sampling((self.latent_dim,))(query_inputs)
        # pos_doc_gs_latents = normal_latent_sampling((self.latent_dim,))(doc_inputs)
        # neg_doc_gs_latents = normal_latent_sampling((self.latent_dim,))(neg_doc_inputs)

        query_real = self.discriminator(gs_latents)
        query_fake = self.discriminator(state_z)

        # pos_doc_real = self.discriminator(pos_doc_gs_latents)
        # pos_doc_fake = self.discriminator(pos_doc_state_z)

        # neg_doc_real = self.discriminator(neg_doc_gs_latents)
        # neg_doc_fake = self.discriminator(neg_doc_state_z)
        

        decoder_inputs = Input(shape=(self.max_len,))
        # pos_doc_decoder_inputs = Input(shape=(self.max_len,))
        # neg_doc_decoder_inputs = Input(shape=(self.max_len,))


        self.latent2hidden = Dense(self.hidden_dim)
        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru")
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")

        # self.doc_latent2hidden = Dense(self.hidden_dim)
        # self.doc_decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru2")
        # self.doc_decoder_dense = Dense(self.nb_words, activation='softmax', name="rec2")
        
        rec_outputs = self.decoder_dense(self.decoder_lstm(encoder_embedding(decoder_inputs) , initial_state=self.latent2hidden(state_z)))
        # pos_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(pos_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(pos_doc_state_z)))
        # neg_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(neg_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(neg_doc_state_z)))

        # self.ae = Model([query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs, pos_doc_decoder_inputs, neg_doc_decoder_inputs], [rec_outputs, pos_doc_rec_outputs, neg_doc_rec_outputs, pairwise_pred])

        
        self.ae = Model([query_inputs, doc_inputs, decoder_inputs], [rec_outputs, cos])

        inputs = self.ae.inputs
        # outputs = fix_names([rec_outputs, pos_doc_rec_outputs, neg_doc_rec_outputs, pairwise_pred, query_fake, query_real, pos_doc_fake, pos_doc_real, neg_doc_fake, neg_doc_real], ["qpred","dpred","ndpred","pair","qfake","qreal","dfake","dreal","ndfake","ndreal"])
        outputs = fix_names([rec_outputs, cos, query_fake, query_real], ["qpred", "pair","qfake","qreal"])
        
        self.aae = Model(inputs, outputs)

        generative_params = self.ae.trainable_weights
        discriminative_params = self.discriminator.trainable_weights
        # self.aae.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.vae_loss, self.vae_loss, "categorical_crossentropy"], loss_weights=[1e-4, 1e-4, 1e-4, 1])        
        self.model = AdversarialModel(base_model=self.aae, player_params=[generative_params, discriminative_params], player_names=["generator", "discriminator"])        
        rec_loss = "sparse_categorical_crossentropy"
        pair_loss = "binary_crossentropy"
        self.model.adversarial_compile(adversarial_optimizer=self.adversarial_optimizer, player_optimizers=[self.optimizer, self.optimizer], loss={"qfake": self.dis_loss, "qreal": self.dis_loss, "qpred": rec_loss,"pair":pair_loss}, player_compile_kwargs=[{"loss_weights": {"qfake": 1e-4, "qreal": 1e-4, "qpred": 1e-2, "pair": 1}}] * 2)
        self.encoder = Model(query_inputs, state)

    def name(self):
        return "binary_aae" if not self.enableWasserstein else "binary_wae"
    
    def word_dropout(self, x, unk_token):
        # np.random.seed(0)
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
        
        return Model(inputs, outputs)

    def sampling(self, args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                      stddev=1)
            return z_mean + K.exp(z_log_var / 2) * epsilon


class DSSM_AAE2():
    
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

        dense = Dense(self.hidden_dim, activation="tanh", name="q_dense")


        state = dense(GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(query_inputs))))
        doc_state = dense(GlobalMaxPooling1D()(doc_encoder_lstm(doc_encoder_embedding(doc_inputs))))
        neg_doc_state = dense(GlobalMaxPooling1D()(doc_encoder_lstm(doc_encoder_embedding(neg_doc_inputs))))



        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)


        self.state_mean = self.mean(state)
        self.state_var = self.var(state)

        state_z = Lambda(self.sampling)([self.state_mean, self.state_var])


        # test




        self.pos_doc_state_mean = self.mean(doc_state)
        self.pos_doc_state_var = self.var(doc_state)
        pos_doc_state_z = Lambda(self.sampling)([self.pos_doc_state_mean, self.pos_doc_state_var])

        self.neg_doc_state_mean = self.mean(neg_doc_state)
        self.neg_doc_state_var = self.var(neg_doc_state)
        neg_doc_state_z = Lambda(self.sampling)([self.neg_doc_state_mean, self.neg_doc_state_var])


        if self.mode == 1:
            query_sem = state_z
            pos_doc_sem = pos_doc_state_z
            neg_doc_sem = neg_doc_state_z
        elif self.mode == 2:
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

        # pos_doc_real = self.discriminator(pos_doc_gs_latents)
        # pos_doc_fake = self.discriminator(pos_doc_state_z)

        # neg_doc_real = self.discriminator(neg_doc_gs_latents)
        # neg_doc_fake = self.discriminator(neg_doc_state_z)
        

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
        
        rec_outputs = self.decoder_dense(self.decoder_lstm(decoder_embedding(decoder_inputs) , initial_state=self.latent2hidden(state_z)))
        # pos_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(pos_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(pos_doc_state_z)))
        # neg_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(neg_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(neg_doc_state_z)))

        # self.ae = Model([query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs, pos_doc_decoder_inputs, neg_doc_decoder_inputs], [rec_outputs, pos_doc_rec_outputs, neg_doc_rec_outputs, pairwise_pred])

        if self.enableSemi:
            semi_outputs = fix_names([rec_outputs, query_fake, query_real], ["qpred", "qfake", "qreal"])
            self.semi_aae = Model([query_inputs, decoder_inputs], semi_outputs)

            semi_generative_params = self.semi_aae.trainable_weights
            semi_discriminative_params = self.discriminator.trainable_weights
            self.semi_model = AdversarialModel(base_model=self.semi_aae, player_params=[semi_generative_params, semi_discriminative_params], player_names=["generator", "discriminator"])        
            rec_loss = "sparse_categorical_crossentropy"
            self.semi_model.adversarial_compile(adversarial_optimizer=self.adversarial_optimizer, player_optimizers=[self.optimizer, self.optimizer], loss={"qfake": self.dis_loss, "qreal": self.dis_loss, "qpred": rec_loss}, player_compile_kwargs=[{"loss_weights": {"qfake": 1e-4, "qreal": 1e-4, "qpred": 1e-4}}] * 2)
            self.model = Model([query_inputs, doc_inputs, neg_doc_inputs], [pairwise_pred])
            self.model.compile(optimizer=self.optimizer, loss=["categorical_crossentropy"])  
        else:
            self.ae = Model([query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs], [rec_outputs, pairwise_pred])

            inputs = self.ae.inputs
            # outputs = fix_names([rec_outputs, pos_doc_rec_outputs, neg_doc_rec_outputs, pairwise_pred, query_fake, query_real, pos_doc_fake, pos_doc_real, neg_doc_fake, neg_doc_real], ["qpred","dpred","ndpred","pair","qfake","qreal","dfake","dreal","ndfake","ndreal"])
            outputs = fix_names([rec_outputs, pairwise_pred, query_fake, query_real], ["qpred", "pair","qfake","qreal"])
            
            self.aae = Model(inputs, outputs)

            generative_params = self.ae.trainable_weights
            discriminative_params = self.discriminator.trainable_weights
            # self.aae.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.vae_loss, self.vae_loss, "categorical_crossentropy"], loss_weights=[1e-4, 1e-4, 1e-4, 1])        
            self.model = AdversarialModel(base_model=self.aae, player_params=[generative_params, discriminative_params], player_names=["generator", "discriminator"])        
            rec_loss = "sparse_categorical_crossentropy"
            pair_loss = "categorical_crossentropy"
            self.model.adversarial_compile(adversarial_optimizer=self.adversarial_optimizer, player_optimizers=[self.optimizer, self.optimizer], loss={"qfake": self.dis_loss, "qreal": self.dis_loss, "qpred": self.vae_loss, "pair": pair_loss}, player_compile_kwargs=[{"loss_weights": {"qfake": 1e-2, "qreal": 1e-2, "qpred": 1, "pair": 1}}] * 2)
        
        self.encoder = Model(query_inputs, state)


    def vae_loss(self, x, x_decoded_onehot):
        xent_loss = objectives.sparse_categorical_crossentropy(x, x_decoded_onehot) 
        return 0 * xent_loss

    def name(self):
        if self.enableSemi:
            return "dssm_aae_s_%d" % self.limit if not self.enableWasserstein else "dssm_wae_s_%d" % self.limit
        return "dssm_aae" if not self.enableWasserstein else "dssm_wae"
    
    def word_dropout(self, x, unk_token):
        # np.random.seed(0)
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
        
        return Model(inputs, outputs)

    def sampling(self, args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                      stddev=1)
            return z_mean + K.exp(z_log_var / 2) * epsilon