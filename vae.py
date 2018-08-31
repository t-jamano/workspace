from Models import *

class VariationalAutoEncoder():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), kl_rate=0.01, keep_rate_word_dropout=0.5, enableKL=True, enableCond=False, mode=1, enableGRU=True, enableSample=False, enablePair=False):

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
        self.enableSample = enableSample
        self.enablePair = enablePair
        self.build()

    def build(self):

        query_inputs = Input(shape=(self.max_len,))


        self.kl_inputs = Input(shape=(1,))

        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=False if self.enableGRU else False,
                                        name="q_embedding",
                                        trainable=True)

        encoder_lstm = Bidirectional(GRU(self.hidden_dim, return_sequences=True), name='q_gru') if self.enableGRU else GlobalMaxPooling1D()
        dense = Dense(self.latent_dim, activation="tanh", name="q_dense")

        state = dense(GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(query_inputs))))


        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)

        self.state_mean = self.mean(state)
        self.state_var = self.var(state)


        self.state_z = Lambda(self.sampling)([self.state_mean, self.state_var])

        decoder_inputs = Input(shape=(self.max_len,))

        self.latent2hidden = Dense(self.hidden_dim)
        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru")

        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="q_rec")

        dec_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="q_dec_embedding",
                                        mask_zero=True,
                                        trainable=True)

        sample_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        # dec_embed = encoder_embedding(decoder_inputs) if self.mode == 1 else decoder_embedding(decoder_inputs)
        
        
        if self.enableSample :

            dec_embed = dec_embedding(decoder_inputs) 
            sample_inputs = Input(shape=(self.max_len, 100,))
            embed_samples = sample_embedding(sample_inputs)
            rec_outputs = dot([self.decoder_lstm(dec_embed, initial_state=self.latent2hidden(self.state_z)), embed_samples], axes=-1)
            rec_outputs = Activation("softmax")(rec_outputs)
            inputs = [neg_query_inputs, decoder_inputs, sample_inputs]
            outputs = [rec_outputs]

        else:

            rec_outputs = self.decoder_dense(self.decoder_lstm(dec_embedding(decoder_inputs), initial_state=self.latent2hidden(self.state_z)))
            inputs = [query_inputs, decoder_inputs] if not self.enableKL else [query_inputs, decoder_inputs, self.kl_inputs]
            outputs = [rec_outputs]

            if self.enablePair:
                neg_decoder_inputs = Input(shape=(self.max_len,))
                neg_rec_outputs = self.decoder_dense(self.decoder_lstm(dec_embedding(neg_decoder_inputs), initial_state=self.latent2hidden(self.state_z)))
                inputs = [query_inputs, decoder_inputs, neg_decoder_inputs]
                outputs = [rec_outputs, neg_rec_outputs]


        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.neg_vae_loss] if self.enablePair else [self.vae_loss], metrics=[self.rec_loss, self.kl_loss] if not self.enableSample and not self.enablePair else None)    
        self.encoder = Model(query_inputs, state)

    def vae_loss(self, x, x_decoded_onehot):
        xent_loss = objectives.sparse_categorical_crossentropy(x, x_decoded_onehot) if not self.enableSample else objectives.categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + self.state_var - K.square(self.state_mean) - K.exp(self.state_var))
        loss = K.mean(xent_loss + kl_loss) if not self.enableKL else xent_loss + (self.kl_inputs * kl_loss)
        return loss

    def neg_vae_loss(self, x, x_decoded_onehot):
        xent_loss = objectives.sparse_categorical_crossentropy(x, x_decoded_onehot) if not self.enableSample else objectives.categorical_crossentropy(x, x_decoded_onehot)
        loss = xent_loss
        return -loss

    def kl_loss(self, y_true, y_pred):
        kl_loss = - 0.5 * K.mean(1 + self.state_var - K.square(self.state_mean) - K.exp(self.state_var))
        return kl_loss

    def rec_loss(self, y_true, y_pred):
        return objectives.sparse_categorical_crossentropy(y_true, y_pred) if not self.enableSample else objectives.categorical_crossentropy(y_true, y_pred)

    def name(self):
        if self.enablePair:
            return "vae_pair"
        elif self.enableSample:
            return "vae_fast"
        elif self.enableGRU:
            return "vae" if not self.enableKL else "vae_kl"
    
    def word_dropout(self, x, unk_token):
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


class KATE():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), comp_topk=2, ctype="kcomp"):

        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.hidden_dim = self.dim[0]
        self.latent_dim = self.dim[1]
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.build()

    def build(self):

        query_inputs = Input(shape=(self.max_len,))

        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="q_embedding",
                                        trainable=True)

        hidden_layer = Dense(self.hidden_dim, kernel_initializer='glorot_normal', activation="tanh")


        state = hidden_layer(GlobalMaxPooling1D()(encoder_embedding(query_inputs)))


        self.mean = Dense(self.latent_dim, kernel_initializer='glorot_normal')
        self.var = Dense(self.latent_dim, kernel_initializer='glorot_normal')

        self.state_mean = self.mean(state)
        self.state_var = self.var(state)

        self.z_mean = KCompetitive(self.comp_topk, self.ctype)(self.state_mean)

        self.state_z = Lambda(self.sampling)([self.z_mean, self.state_var])

        self.latent2hidden = Dense(self.hidden_dim, kernel_initializer='glorot_normal', activation="tanh")

        self.decoder_dense = Dense(self.nb_words, activation='sigmoid', name="q_rec")

        rec_outputs = self.decoder_dense(self.latent2hidden(self.state_z))        
        # self.model = Model(inputs, [rec_outputs, rec_outputs])
        self.model = Model(query_inputs, [rec_outputs])

        # self.model.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.neg_vae_loss])    
        self.model.compile(optimizer=self.optimizer, loss=[self.vae_loss])    

        self.encoder = Model(query_inputs, state)

    def vae_loss(self, x, x_decoded_onehot):
        xent_loss = K.mean(K.binary_crossentropy(x, x_decoded_onehot))
        kl_loss = - 0.5 * K.mean(1 + self.state_var - K.square(self.state_mean) - K.exp(self.state_var), axis=-1)
        loss = xent_loss + kl_loss
        return loss

    def name(self):
        return "kate"

    def sampling(self, args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                      stddev=1)
            return z_mean + K.exp(z_log_var / 2) * epsilon



class VariationalAutoEncoder2():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), kl_rate=0.01, keep_rate_word_dropout=0.5, enableKL=True, enableCond=False, mode=1, enableGRU=True, enableSample=False):

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
        self.enableSample = enableSample
        self.build()

    def build(self):

        query_inputs = Input(shape=(self.max_len,))

        self.kl_inputs = Input(shape=(1,))

        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="q_embedding",
                                        trainable=True)

        encoder_lstm = Bidirectional(GRU(self.hidden_dim, return_sequences=True), name='q_gru') if self.enableGRU else GlobalMaxPooling1D()

        state = GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(query_inputs)))


        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)

        self.state_mean = self.mean(state)
        self.state_var = self.var(state)


        state_z = Lambda(self.sampling)([self.state_mean, self.state_var])

        decoder_inputs = Input(shape=(self.max_len,))

        self.latent2hidden = Dense(self.hidden_dim)
        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru")

        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="q_rec")

        dec_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="q_dec_embedding",
                                        mask_zero=True,
                                        trainable=True)

        dec_embed = dec_embedding(decoder_inputs) 
        
        if not self.enableSample and not self.enablePair:
            rec_outputs = self.decoder_dense(self.decoder_lstm(RepeatVector(self.max_len)(self.latent2hidden(state_z))))
            inputs = [query_inputs] if not self.enableKL else [query_inputs, self.kl_inputs]
        if self.enablePair:
            neg_query_inputs = Input(shape=(self.max_len,))
            neg_decoder_inputs = Input(shape=(self.max_len,))

            neg_state = GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(neg_query_inputs)))
            self.neg_state_mean = self.mean(neg_state)
            self.neg_state_var = self.var(neg_state)
            neg_state_z = Lambda(self.sampling)([self.neg_state_mean, self.neg_state_var])

        else:
            sample_inputs = Input(shape=(self.max_len, 100,))
            embed_samples = dec_embedding(sample_inputs)
            rec_outputs = dot([self.decoder_lstm(dec_embed, initial_state=self.latent2hidden(state_z)), embed_samples], axes=-1)
            rec_outputs = Activation("softmax")(rec_outputs)
            inputs = [query_inputs, decoder_inputs, sample_inputs] if not self.enableKL else [query_inputs, decoder_inputs, sample_inputs, self.kl_inputs]
        
        # self.model = Model(inputs, [rec_outputs, rec_outputs])

        if self.enablePair:
            self.model = Model(inputs, [rec_outputs, neg_rec_outputs])
            self.model.compile(optimizer=self.optimizer, loss=[self.vae_loss, self.neg_vae_loss])    
        else:
            self.model = Model(inputs, [rec_outputs])
            self.model.compile(optimizer=self.optimizer, loss=[self.vae_loss])    

        self.encoder = Model(query_inputs, self.state_mean)

    def vae_loss(self, x, x_decoded_onehot):
        xent_loss = objectives.sparse_categorical_crossentropy(x, x_decoded_onehot) if not self.enableSample else objectives.categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + self.state_var - K.square(self.state_mean) - K.exp(self.state_var))
        loss = K.mean(xent_loss + kl_loss) if not self.enableKL else xent_loss + (self.kl_inputs * kl_loss)
        return loss

    def neg_vae_loss(self, x, x_decoded_onehot):
        xent_loss = objectives.sparse_categorical_crossentropy(x, x_decoded_onehot)
        loss = xent_loss
        return -loss

    def kl_loss(self, y_true, y_pred):
        kl_loss = - 0.5 * K.mean(1 + self.state_var - K.square(self.state_mean) - K.exp(self.state_var))
        return kl_loss

    def rec_loss(self, y_true, y_pred):
        return objectives.sparse_categorical_crossentropy(y_true, y_pred) if not self.enableSample else objectives.categorical_crossentropy(y_true, y_pred)

    def name(self):
        if not self.enableGRU:
            return "vae_max"
        elif self.enableSample:
            return "vae_fast"
        elif self.enableGRU:
            return "vae_no_dec_inp" if not self.enableKL else "vae_kl"
    
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

