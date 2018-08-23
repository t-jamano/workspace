from Models import *

class DSSM_VAE2():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), kl_rate=0.01, keep_rate_word_dropout=0.5, enableKL=True, enableCond=False):

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
                                        name="query_embedding",
                                        trainable=True)

        doc_encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        norm = BatchNormalization()


        encoder_lstm = GRU(self.hidden_dim, name="query_gru")
        doc_encoder_lstm = GRU(self.hidden_dim)


        state = norm(encoder_lstm(encoder_embedding(query_inputs)))
        doc_state = norm(doc_encoder_lstm(doc_encoder_embedding(doc_inputs)))
        neg_doc_state = norm(doc_encoder_lstm(doc_encoder_embedding(neg_doc_inputs)))



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
        
        # self.dssm = Model([q_inputs, pd_inputs, nd_inputs], pairwise_pred)
        # self.dssm.compile(optimizer=self.optimizer, loss=["categorical_crossentropy"])


        

        state_z = Lambda(self.sampling)([self.state_mean, self.state_var])
        pos_doc_state_z = Lambda(self.sampling)([self.pos_doc_state_mean, self.pos_doc_state_var])
        neg_doc_state_z = Lambda(self.sampling)([self.neg_doc_state_mean, self.neg_doc_state_var])



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


        inputs = [query_inputs, doc_inputs, neg_doc_inputs, decoder_inputs]
        self.model = Model(inputs, [rec_outputs, pairwise_pred])
        self.model.compile(optimizer=self.optimizer, loss=[self.vae_loss, "categorical_crossentropy"], loss_weights=[1e-4, 1])        
                
        self.encoder = Model(query_inputs, state)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def vae_loss(self, x, x_decoded_onehot):
	    xent_loss = objectives.sparse_categorical_crossentropy(x, x_decoded_onehot)
	    kl_loss = - 0.5 * K.mean(1 + self.state_var - K.square(self.state_mean) - K.exp(self.state_var))
	    loss = xent_loss + kl_loss
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
 
    	return "dssm_vae2"
    
    def word_dropout(self, x, unk_token):
        np.random.seed(0)
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