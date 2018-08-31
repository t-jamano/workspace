from Models import *

class DSSM_VAE():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), kl_rate=0.01, keep_rate_word_dropout=0.5, enableKL=True, enableCond=False, mode=1, enableGRU=True, enableSemi=False, enableSeparate=False, limit=1):

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
        self.enableGRU = enableGRU
        self.enableSemi = enableSemi
        self.enableSeparate = enableSeparate
        self.limit = limit
        self.build()

    def build(self):

        query_inputs = Input(shape=(self.max_len,))
        doc_inputs = Input(shape=(self.max_len,))
        neg_doc_inputs = Input(shape=(self.max_len,))

        label_inputs = Input(shape=(1,))
        self.kl_inputs = Input(shape=(1,))

        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=False,
                                        name="q_embedding",
                                        trainable=True)

        # doc_encoder_embedding = Embedding(self.nb_words,
        #                                 self.embedding_matrix.shape[-1],
        #                                 weights=[self.embedding_matrix],
        #                                 input_length=self.max_len,
        #                                 mask_zero=False,
        #                                 trainable=True) if self.enableSeparate else encoder_embedding


        encoder_lstm = Bidirectional(GRU(self.hidden_dim, return_sequences=True), name='q_gru')
        dense = Dense(self.hidden_dim, activation="tanh", name="q_dense")

        # doc_encoder_lstm = Bidirectional(GRU(self.hidden_dim, return_sequences=True), name='q_gru') if self.enableSeparate else encoder_lstm

        state = dense(GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(query_inputs))))
        doc_state = dense(GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(doc_inputs))))
        neg_doc_state = dense(GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(neg_doc_inputs))))



        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)


        self.state_mean = self.mean(state)
        self.state_var = self.var(state)

        self.pos_doc_state_mean = self.mean(doc_state)
        self.pos_doc_state_var = self.var(doc_state)

        self.neg_doc_state_mean = self.mean(neg_doc_state)
        self.neg_doc_state_var = self.var(neg_doc_state)


        state_z = Lambda(self.sampling)([self.state_mean, self.state_var])
        pos_doc_state_z = Lambda(self.sampling)([self.pos_doc_state_mean, self.pos_doc_state_var])
        neg_doc_state_z = Lambda(self.sampling)([self.neg_doc_state_mean, self.neg_doc_state_var])

        # if self.mode == 1:
        #     query_sem = state
        #     pos_doc_sem = doc_state
        #     neg_doc_sem = neg_doc_state
        # elif self.mode == 2:
        #     query_sem = self.state_mean
        #     pos_doc_sem = self.pos_doc_state_mean
        #     neg_doc_sem = self.neg_doc_state_mean
        # elif self.mode == 3:
        #     query_sem = state_z
        #     pos_doc_sem = pos_doc_state_z
        #     neg_doc_sem = neg_doc_state_z
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
        pairwise_pred = Activation("softmax", name="pair")(with_gamma) # See equation (5).
        

        decoder_inputs = Input(shape=(self.max_len,))
        # pos_doc_decoder_inputs = Input(shape=(self.max_len,))
        # neg_doc_decoder_inputs = Input(shape=(self.max_len,))


        self.latent2hidden = Dense(self.hidden_dim, activation="tanh")
        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru")
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="q_rec")

        # self.doc_latent2hidden = Dense(self.hidden_dim) if self.enableSeparate else self.latent2hidden
        # self.doc_decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru2") if self.enableSeparate else self.decoder_lstm
        # self.doc_decoder_dense = Dense(self.nb_words, activation='softmax', name="d_rec") if self.enableSeparate else self.decoder_dense
        
        rec_outputs = self.decoder_dense(self.decoder_lstm(encoder_embedding(decoder_inputs) , initial_state=self.latent2hidden(state_z)))
        # pos_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(pos_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(pos_doc_state_z)))
        # neg_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(neg_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(neg_doc_state_z)))

        if self.enableSemi:
            # self.ae = Model([query_inputs, doc_inputs, decoder_inputs, pos_doc_decoder_inputs], [rec_outputs, pos_doc_rec_outputs])
            self.semi_model = Model([query_inputs, decoder_inputs] if not self.enableKL else [query_inputs, decoder_inputs, self.kl_inputs], [rec_outputs])
            self.model = Model([query_inputs, doc_inputs, neg_doc_inputs], [pairwise_pred])
            # self.ae.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.pos_doc_vae_loss])        
            self.semi_model.compile(optimizer=self.optimizer, loss=[self.vae_loss], loss_weights=[1e-2])        
            self.model.compile(optimizer=self.optimizer, loss=["categorical_crossentropy"])        

        else:
            # inputs = [query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs, pos_doc_decoder_inputs, neg_doc_decoder_inputs]
            inputs = [query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs] if not self.enableKL else [query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs, self.kl_inputs]
            self.model = Model(inputs, [rec_outputs, pairwise_pred])
            # self.model = Model(inputs, [rec_outputs, pos_doc_rec_outputs, neg_doc_rec_outputs, pairwise_pred])
            self.model.compile(optimizer=self.optimizer, loss=[self.vae_loss, "categorical_crossentropy"], loss_weights=[1e-2, 1])        
            # self.model.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.pos_doc_vae_loss, self.neg_doc_vae_loss, "categorical_crossentropy"], loss_weights=[1e-2, 1e-2, 1e-2, 1])        
        
        

        self.encoder = Model(query_inputs, state)

    def vae_loss(self, x, x_decoded_onehot):
        xent_loss = objectives.sparse_categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + self.state_var - K.square(self.state_mean) - K.exp(self.state_var))
        loss = K.mean(xent_loss + kl_loss) if not self.enableKL else xent_loss + (self.kl_inputs * kl_loss)
        return loss

    def pos_doc_vae_loss(self, x, x_decoded_onehot):
        xent_loss = objectives.sparse_categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + self.pos_doc_state_var - K.square(self.pos_doc_state_mean) - K.exp(self.pos_doc_state_var))
        loss = xent_loss + kl_loss
        return loss

    def neg_doc_vae_loss(self, x, x_decoded_onehot):
        xent_loss = objectives.sparse_categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + self.neg_doc_state_var - K.square(self.neg_doc_state_mean) - K.exp(self.neg_doc_state_var))
        loss = xent_loss + kl_loss
        return loss


    def kl_loss(self, y_true, y_pred):
	    kl_loss = - 0.5 * K.mean(1 + self.state_var - K.square(self.state_mean) - K.exp(self.state_var))
	    return kl_loss

    def pos_doc_kl_loss(self, y_true, y_pred):
        kl_loss = - 0.5 * K.mean(1 + self.pos_doc_state_var - K.square(self.pos_doc_state_mean) - K.exp(self.pos_doc_state_var))
        return kl_loss

    def neg_doc_kl_loss(self, y_true, y_pred):
        kl_loss = - 0.5 * K.mean(1 + self.neg_doc_state_var - K.square(self.neg_doc_state_mean) - K.exp(self.neg_doc_state_var))
        return kl_loss

    def rec_loss(self, y_true, y_pred):
    	return objectives.sparse_categorical_crossentropy(y_true, y_pred)

    def name(self):
        if self.enableSemi and self.enableKL:
            return "dssm_vae_s_kl_%d" % self.limit
        elif self.enableSemi:
            return "dssm_vae_s_%d" % self.limit
        elif self.enableKL:
            return "dssm_vae_kl"
        return "dssm_vae"
    
    def word_dropout(self, x, unk_token):
        # np.random.seed(0)
        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

            return x_

    def sampling(self, args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                      stddev=1)
            return z_mean + K.exp(z_log_var / 2) * epsilon


class DSSM_VAE_PR(DSSM_VAE):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), kl_rate=0.01, keep_rate_word_dropout=0.5, enableKL=True, enableCond=False, mode=1):
        DSSM_VAE.__init__(self, nb_words, max_len, embedding_matrix, dim, optimizer, kl_rate, keep_rate_word_dropout, enableKL, enableCond, mode)

        self.build()

    def build(self):

        query_inputs = Input(shape=(self.max_len,))
        doc_inputs = Input(shape=(self.max_len,))
        neg_doc_inputs = Input(shape=(self.max_len,))

        label_inputs = Input(shape=(1,))
        kl_inputs = Input(shape=(1,))

        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        name="q_embedding_layer",
                                        trainable=True)

        doc_encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        pair_encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        pair_doc_encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)


        encoder_lstm = GRU(self.hidden_dim, name="q_gru")
        doc_encoder_lstm = GRU(self.hidden_dim)

        pair_encoder_lstm = GRU(self.hidden_dim)
        pair_doc_encoder_lstm = GRU(self.hidden_dim)


        state = encoder_lstm(encoder_embedding(query_inputs))
        doc_state = doc_encoder_lstm(doc_encoder_embedding(doc_inputs))
        neg_doc_state = doc_encoder_lstm(doc_encoder_embedding(neg_doc_inputs))


        pair_state = pair_encoder_lstm(pair_encoder_embedding(query_inputs))
        pair_doc_state = pair_doc_encoder_lstm(pair_doc_encoder_embedding(doc_inputs))
        pair_neg_doc_state = pair_doc_encoder_lstm(pair_doc_encoder_embedding(neg_doc_inputs))


        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)
        self.doc_mean = Dense(self.latent_dim)
        self.doc_var = Dense(self.latent_dim)


        self.state_mean = self.mean(state)
        self.state_var = self.var(state)

        self.pos_doc_state_mean = self.doc_mean(doc_state)
        self.pos_doc_state_var = self.doc_var(doc_state)

        self.neg_doc_state_mean = self.doc_mean(neg_doc_state)
        self.neg_doc_state_var = self.doc_var(neg_doc_state)


        state_z = Lambda(self.sampling)([self.state_mean, self.state_var])
        pos_doc_state_z = Lambda(self.sampling)([self.pos_doc_state_mean, self.pos_doc_state_var])
        neg_doc_state_z = Lambda(self.sampling)([self.neg_doc_state_mean, self.neg_doc_state_var])

        if self.mode == 1:
            query_sem = merge([state, pair_state], mode="concat")
            pos_doc_sem = merge([doc_state, pair_doc_state], mode="concat")
            neg_doc_sem = merge([neg_doc_state, pair_neg_doc_state], mode="concat")
        elif self.mode == 2:
            query_sem = merge([self.state_mean, pair_state], mode="concat")
            pos_doc_sem = merge([self.pos_doc_state_mean, pair_doc_state], mode="concat")
            neg_doc_sem = merge([self.neg_doc_state_mean, pair_neg_doc_state], mode="concat")
        elif self.mode == 3:
            query_sem = merge([state_z, pair_state], mode="concat")
            pos_doc_sem = merge([pos_doc_state_z, pair_doc_state], mode="concat")
            neg_doc_sem = merge([neg_doc_state_z, pair_neg_doc_state], mode="concat")


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
        

        decoder_inputs = Input(shape=(self.max_len,))
        pos_doc_decoder_inputs = Input(shape=(self.max_len,))
        neg_doc_decoder_inputs = Input(shape=(self.max_len,))


        self.latent2hidden = Dense(self.hidden_dim)
        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru")
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")

        self.doc_latent2hidden = Dense(self.hidden_dim)
        self.doc_decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru2")
        self.doc_decoder_dense = Dense(self.nb_words, activation='softmax', name="rec2")
        
        rec_outputs = self.decoder_dense(self.decoder_lstm(encoder_embedding(decoder_inputs) , initial_state=self.latent2hidden(state_z)))
        pos_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(pos_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(pos_doc_state_z)))
        neg_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(neg_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(neg_doc_state_z)))


        inputs = [query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs, pos_doc_decoder_inputs, neg_doc_decoder_inputs]
        self.model = Model(inputs, [rec_outputs, pos_doc_rec_outputs, neg_doc_rec_outputs, pairwise_pred])
        self.model.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.pos_doc_vae_loss, self.neg_doc_vae_loss, "categorical_crossentropy"], loss_weights=[1e-5, 1e-5, 1e-5, 1])        
                
        self.encoder = Model(query_inputs, state)


    def name(self):
 
        return "dssm_vae_pr"
    

class DSSM_AE():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), kl_rate=0.01, keep_rate_word_dropout=0.5, enableKL=True, enableCond=False, mode=1, enableGRU=True, enableSemi=False, enableSeparate=False):

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
        self.enableGRU = enableGRU
        self.enableSemi = enableSemi
        self.enableSeparate = enableSeparate
        self.build()

    def build(self):

        query_inputs = Input(shape=(self.max_len,))
        doc_inputs = Input(shape=(self.max_len,))
        neg_doc_inputs = Input(shape=(self.max_len,))

        label_inputs = Input(shape=(1,))
        kl_inputs = Input(shape=(1,))

        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True if self.enableGRU else False,
                                        name="q_embedding",
                                        trainable=True)

        doc_encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True if self.enableGRU else False,
                                        trainable=True) if self.enableSeparate else encoder_embedding


        if self.enableGRU:
            encoder_lstm = GRU(self.hidden_dim, name="q_gru", return_state=True)
            doc_encoder_lstm = GRU(self.hidden_dim, name="d_gru", return_state=True)
        else:
            encoder_lstm = GlobalAveragePooling1D()
            doc_encoder_lstm = GlobalAveragePooling1D()

        mean = Dense(self.latent_dim)
        _, state = encoder_lstm(encoder_embedding(query_inputs))
        _, doc_state = doc_encoder_lstm(doc_encoder_embedding(doc_inputs))
        _, neg_doc_state = doc_encoder_lstm(doc_encoder_embedding(neg_doc_inputs))


        state_z = mean(state)
        pos_doc_state_z = mean(doc_state)
        neg_doc_state_z = mean(neg_doc_state)

        query_sem = state_z
        pos_doc_sem = pos_doc_state_z
        neg_doc_sem = neg_doc_state_z


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
        

        decoder_inputs = Input(shape=(self.max_len,))
        pos_doc_decoder_inputs = Input(shape=(self.max_len,))
        neg_doc_decoder_inputs = Input(shape=(self.max_len,))


        self.latent2hidden = Dense(self.hidden_dim)
        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru")
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="q_rec")

        self.doc_latent2hidden = Dense(self.hidden_dim) if self.enableSeparate else self.latent2hidden
        self.doc_decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru2") if self.enableSeparate else self.decoder_lstm
        self.doc_decoder_dense = Dense(self.nb_words, activation='softmax', name="d_rec") if self.enableSeparate else self.decoder_dense
        
        rec_outputs = self.decoder_dense(self.decoder_lstm(encoder_embedding(decoder_inputs) , initial_state=self.latent2hidden(state_z)))
        pos_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(pos_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(pos_doc_state_z)))
        neg_doc_rec_outputs = self.doc_decoder_dense(self.doc_decoder_lstm(doc_encoder_embedding(neg_doc_decoder_inputs) , initial_state=self.doc_latent2hidden(neg_doc_state_z)))

        if self.enableSemi:
            self.semi_model = Model([query_inputs, doc_inputs, decoder_inputs, pos_doc_decoder_inputs], [rec_outputs, pos_doc_rec_outputs])
            self.model = Model([query_inputs, doc_inputs, neg_doc_inputs], [pairwise_pred])
            self.semi_model.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.vae_loss])        
            self.model.compile(optimizer=self.optimizer, loss=["categorical_crossentropy"])        

        else:
            inputs = [query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs, pos_doc_decoder_inputs, neg_doc_decoder_inputs]
            self.model = Model(inputs, [rec_outputs, pos_doc_rec_outputs, neg_doc_rec_outputs, pairwise_pred])
            # self.model.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.vae_loss, self.vae_loss, "categorical_crossentropy"], loss_weights=[1e-2, 1e-2, 1e-2, 1])        
            self.model.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.vae_loss, self.vae_loss, "categorical_crossentropy"], loss_weights=[0.05,0.05,-0.05, 2])        
        

        self.encoder = Model(query_inputs, state_z)

    def vae_loss(self, x, x_decoded_onehot):
        xent_loss = objectives.sparse_categorical_crossentropy(x, x_decoded_onehot)
        return xent_loss

    def name(self):
        if self.enableSemi and self.enableGRU:
            return "dssm_ae_s"
        elif self.enableSemi and not self.enableGRU:
            return "dssm_ae_avg_s"
        elif not self.enableGRU:
            return "dssm_ae"
        else:
            return "dssm_ae_avg"
    
    def word_dropout(self, x, unk_token):
        np.random.seed(0)
        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

            return x_
