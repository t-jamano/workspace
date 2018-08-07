from Models import *


class AdversarialAutoencoder(object):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, comp_topk=None, ctype=None, epsilon_std=1.0, optimizer=Adadelta(lr=2.), kl_weight=0, PoolMode="max", trainMode=1):
        self.dim = dim
        self.embedding_matrix = embedding_matrix
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.max_len = max_len
        self.nb_words = nb_words
        self.optimizer = optimizer
        self.kl_weight = kl_weight
        self.enableKL = True if kl_weight == 0 else False
        self.PoolMode = PoolMode
        self.trainMode = trainMode
        
        
        self.model, self.train_encoder, self.encoder, self.discriminator = self.build()
        
        if self.trainMode == 1:
            for layer in self.discriminator.layers:
                layer.trainable=False
        
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        
        

        self.model.compile(loss=['sparse_categorical_crossentropy', "binary_crossentropy"],
            loss_weights=[0.9, 0.1],
            optimizer=optimizer)

    def name(self):
        return "aae_%s_%d" % (self.PoolMode, self.trainMode)

    def build(self):
        
        e_input = Input(shape=(self.max_len,), name="enc_input")
        e_input2 = Input(shape=(self.max_len,), name="enc_input2")

        emb = Embedding(self.nb_words,
                            self.embedding_matrix.shape[-1],
                            weights=[self.embedding_matrix],
                            input_length=self.max_len,
                            mask_zero=True,
                            trainable=True)
        
        rnn = GRU(self.dim[0], name='lstm_1', return_sequences=False, return_state=True)

        h = emb(e_input)
        _, h = rnn(h)
        mu = Dense(self.dim[1])
        log_var = Dense(self.dim[1])
        h_mu = mu(h)
        h_log_var = log_var(h)
        z = merge([h_mu, h_log_var],
                mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
                output_shape=lambda p: p[0])
        
        
        h2 = emb(e_input2)
        _, h2 = rnn(h2)
        h2_mu = mu(h2)
        
        
        d_input = Input(shape=(self.max_len,), name="dec_input")
        
        d_latent2hidden = Dense(self.dim[0], activation='relu')
        d_lstm = GRU(self.dim[0], return_sequences=True)
        dec_embedding_layer = Embedding(self.nb_words,
                                    self.embedding_matrix.shape[-1],
#                                     weights=[self.embedding_matrix],
                                    input_length=self.max_len,
                                    mask_zero=True,
                                    trainable=True)

        softmax_layer = Dense(self.nb_words, activation="softmax")
        d_output2vocab = TimeDistributed(softmax_layer, name="rec")
        
        
        h_z = d_latent2hidden(z)
        
        d_embed_input = dec_embedding_layer(d_input)
        outputs = d_lstm(d_embed_input, initial_state=[h_z])
        x_ = d_output2vocab(outputs)
        
        
        
        

        z_input = Input(shape=(self.dim[1],))
        h1 = Dense(200, input_dim=self.dim[1])
        fc = Dense(1, activation="sigmoid", name="dis")

        pred = fc(h1(z_input))
        
        discriminator = Model(z_input, pred)
        
        
        
        gan_pred = fc(h1(z))
        
        train_encoder = Model(e_input, z)
        encoder = Model(e_input, h_mu)
        
        vae = Model([e_input, d_input], [x_, gan_pred])
        
        return vae, train_encoder, encoder, discriminator


class VAE_DSSM2(object):
    

    def __init__(self, nb_words, max_len, embedding_matrix, dim, comp_topk=None, ctype=None, epsilon_std=1.0, optimizer=Adadelta(lr=2.), kl_weight=0, PoolMode="max", FocusMode="fair", mode=1):
        self.dim = dim
        self.embedding_matrix = embedding_matrix
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.max_len = max_len
        self.nb_words = nb_words
        self.optimizer = optimizer
        self.kl_weight = kl_weight
        self.num_negatives = 1
        self.PoolMode = PoolMode
        self.FocusMode = FocusMode
        self.mode = mode
        if self.FocusMode == "fair":
            self.loss_weights = [1] * 5
        elif self.FocusMode == "pair":
            self.loss_weights = [0.001] * 4 + [0.999]
        elif self.FocusMode == "rec":
            self.loss_weights = [1] * 4 + [0.1]

        self.build()

    def build(self):
        
        query = Input(shape = (self.max_len,))
        pos_doc = Input(shape = (self.max_len,))
        neg_docs = [Input(shape = (self.max_len,)) for j in range(self.num_negatives)]
        
        q_emb_layer = Embedding(self.nb_words,
                            self.embedding_matrix.shape[-1],
                            weights=[self.embedding_matrix],
                            input_length=self.max_len,
                            trainable=True)
        # No need for now
        d_emb_layer = Embedding(self.nb_words,
                            self.embedding_matrix.shape[-1],
                            weights=[self.embedding_matrix],
                            input_length=self.max_len,
                            trainable=True) if self. mode == 2 else q_emb_layer



        Pooling = GlobalMaxPooling1D() if self.PoolMode == "max" else GlobalAveragePooling1D()
        
        query_sem = Pooling(q_emb_layer(query))
        pos_doc_sem = Pooling(d_emb_layer(pos_doc))
        neg_doc_sems = [Pooling(d_emb_layer(neg_doc)) for neg_doc in neg_docs]
        
#       DSSM
        R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
        R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True) for neg_doc_sem in neg_doc_sems] # See equation (4).
        concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
        concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)
        weight = np.array([1]).reshape(1, 1, 1)
        with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
        with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)
        pairwise_pred = Activation("softmax", name="pair")(with_gamma) # See equation (5).
        
        mean = Dense(self.dim[1])
        var = Dense(self.dim[1])

        self.q_mean = mean(query_sem)
        self.q_log_var = var(query_sem)
        self.pd_mean = mean(pos_doc_sem)
        self.pd_log_var = var(pos_doc_sem)
        # only support 1 negative samples
        self.nd_mean = mean(neg_doc_sems[0])
        self.nd_log_var = var(neg_doc_sems[0])
        
        if self.comp_topk != None:
            # TODO
            kcomp_layer = KCompetitive(self.comp_topk, self.ctype)
            self.q_mean_k = kcomp_layer(self.q_mean)
            self.pd_mean_k = kcomp_layer(self.pd_mean)
            self.nd_mean_k = kcomp_layer(self.nd_mean)

            encoded_q = Lambda(self.sampling, output_shape=(self.dim[1],))([self.q_mean_k, self.q_log_var])
            encoded_pd = Lambda(self.sampling, output_shape=(self.dim[1],))([self.pd_mean_k, self.pd_log_var])
            encoded_nd = Lambda(self.sampling, output_shape=(self.dim[1],))([self.nd_mean_k, self.nd_log_var])
        else:
            encoded_q = Lambda(self.sampling, output_shape=(self.dim[1],))([self.q_mean, self.q_log_var])
            encoded_pd = Lambda(self.sampling, output_shape=(self.dim[1],))([self.pd_mean, self.pd_log_var])
            encoded_nd = Lambda(self.sampling, output_shape=(self.dim[1],))([self.nd_mean, self.nd_log_var])


        repeated_context = RepeatVector(self.max_len)
        decoder_h = Dense(self.dim[1], kernel_initializer='glorot_normal', activation="tanh")
        
        softmax_layer = Dense(self.nb_words, activation='softmax')
        decoder_mean_q = TimeDistributed(softmax_layer, name="rec_q")#softmax is applied in the seq2seqloss by tf
        decoder_mean_pd = TimeDistributed(softmax_layer, name="rec_pd")
        decoder_mean_nd = TimeDistributed(softmax_layer, name="rec_nd")
        decoder_mean_kl = TimeDistributed(softmax_layer, name="kl")

        q_decoded = decoder_h(repeated_context(encoded_q))
        dec_q = decoder_mean_q(q_decoded)
        
        pd_decoded = decoder_h(repeated_context(encoded_pd))
        dec_pd = decoder_mean_pd(pd_decoded)
        
        nd_decoded = decoder_h(repeated_context(encoded_nd))
        dec_nd = decoder_mean_nd(nd_decoded)


        dec_kl = decoder_mean_kl(q_decoded)
        

        self.model = Model([query, pos_doc] + neg_docs, [dec_q, dec_pd, dec_nd, dec_kl, pairwise_pred])
        self.model.compile(optimizer=self.optimizer, loss= 3 * ["sparse_categorical_crossentropy"] + [self.kl_loss, "categorical_crossentropy"], loss_weights=self.loss_weights)
        self.encoder = Model(outputs=self.q_mean, inputs=query)


    def kl_loss(self, x, x_):

        q_kl_loss = - 0.5 * K.sum(1 + self.q_log_var - K.square(self.q_mean) - K.exp(self.q_log_var), axis=-1) 
        pd_kl_loss = - 0.5 * K.sum(1 + self.pd_log_var - K.square(self.pd_mean) - K.exp(self.pd_log_var), axis=-1) 
        nd_kl_loss = - 0.5 * K.sum(1 + self.nd_log_var - K.square(self.nd_mean) - K.exp(self.nd_log_var), axis=-1) 

        return K.mean(self.kl_weight * (q_kl_loss + pd_kl_loss + nd_kl_loss))

    def name(self):
        return "vae_dssm2_%s_%s_%d" % (self.PoolMode, self.FocusMode, self.mode) if self.comp_topk == None else "kate_dssm2_%s_%s_%d" % (self.PoolMode, self.FocusMode, self.mode)

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon


class VAE_LSTM(object):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, comp_topk=None, ctype=None, epsilon_std=1.0, optimizer=Adadelta(lr=2.), enableKL=False):
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.enableKL = enableKL

        
        act = 'tanh'
        input_layer = Input(shape=(self.max_len,))
        self.kl_input_layer = Input(shape=(1,))

        embedding_layer = Embedding(nb_words,
                            embedding_matrix.shape[-1],
                            weights=[self.embedding_matrix],
                            input_length=max_len,
                            mask_zero=True,
                            trainable=True)

        bilstm = GRU(self.dim[0], return_state=False)

        h1 = embedding_layer(input_layer)
        h1 = bilstm(h1)

        self.z_mean = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)
        self.z_log_var = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)

        if self.comp_topk != None:
            self.z_mean_k = KCompetitive(self.comp_topk, self.ctype)(self.z_mean)
            encoded = Lambda(self.sampling, output_shape=(self.dim[1],))([self.z_mean_k, self.z_log_var])
        else:
            encoded = Lambda(self.sampling, output_shape=(self.dim[1],))([self.z_mean, self.z_log_var])

        decoder_h = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        decoder_mean = Dense(self.nb_words, activation='softmax')

        h_decoded = decoder_h(encoded)
        h_decoded = RepeatVector(self.max_len)(h_decoded)
        h_decoded = GRU(self.dim[0], return_sequences=True)(h_decoded)
        x_decoded_mean = TimeDistributed(decoder_mean, name='rec')(h_decoded)
        x_decoded_kl = TimeDistributed(decoder_mean, name='kl')(h_decoded)

        self.model = Model(outputs=[x_decoded_mean, x_decoded_kl], inputs=[input_layer, self.kl_input_layer] if self.enableKL else [input_layer])
        self.model.compile(optimizer=self.optimizer, loss=["sparse_categorical_crossentropy", self.kl_loss])
        self.encoder = Model(outputs=self.z_mean, inputs=input_layer)


    def name(self):
        name = "kate_lstm_k%d" % self.comp_topk if self.ctype != None else "vae_lstm"
        return name + "_kl" if self.enableKL else name

    def kl_loss(self, x, x_):
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)

        return self.kl_input_layer * kl_loss if self.enableKL else kl_loss

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon


class BOW_VAE(object):
    

    def __init__(self, input_size, max_len, embedding_matrix, dim, comp_topk=None, ctype=None, epsilon_std=1.0, optimizer=Adadelta(lr=2.), kl_weight=0, PoolMode="max"):
        self.input_size = input_size
        self.dim = dim
        self.embedding_matrix = embedding_matrix
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.max_len = max_len
        self.nb_words = input_size
        self.optimizer = optimizer
        self.kl_weight = kl_weight
        self.enableKL = True if kl_weight == 0 else False
        self.PoolMode = PoolMode
        self.build()

    def build(self):
        act = 'tanh'
        input_layer = Input(shape=(self.max_len,))
        emb_layer = Embedding(self.nb_words,
                            self.embedding_matrix.shape[-1],
                            weights=[self.embedding_matrix],
                            input_length=self.max_len,
                            trainable=True)

        hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        
        Pooling = GlobalMaxPooling1D() if self.PoolMode == "max" else GlobalAveragePooling1D()

        h1 = Pooling(emb_layer(input_layer))
        
        h1 = hidden_layer1(input_layer)

        self.z_mean = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)
        self.z_log_var = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)

        if self.comp_topk != None:
            self.z_mean_k = KCompetitive(self.comp_topk, self.ctype)(self.z_mean)
            encoded = Lambda(self.sampling, output_shape=(self.dim[1],))([self.z_mean_k, self.z_log_var])
        else:
            encoded = Lambda(self.sampling, output_shape=(self.dim[1],))([self.z_mean, self.z_log_var])


        decoder_h = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)

        hidden2vocab = Dense(self.nb_words, activation='sigmoid', name="rec")
        h_decoded = decoder_h(encoded)
        h_decoded = hidden2vocab(h_decoded)
     
        self.model = Model(input_layer, [h_decoded, h_decoded])
        self.model.compile(optimizer=self.optimizer, loss=["binary_crossentropy", self.kl_loss])

        self.encoder = Model(outputs=self.z_mean, inputs=input_layer)


    def name(self):
        n = "bowvae_%s" % self.PoolMode if self.comp_topk == None else "bowkate_%s" % self.PoolMode
        n =  n + "_kl" if self.enableKL else n
        return n
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon


    def kl_loss(self, x, x_):
        if self.comp_topk != None:
            kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean_k) - K.exp(self.z_log_var), axis=-1)
        else:
            kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return self.kl_weight * kl_loss



class Seq2Seq(object):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), kl_rate=0.01, keep_rate_word_dropout=0.5, mode=1, enableKL=True, enableS2S=True):
        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.keep_rate_word_dropout = keep_rate_word_dropout
        self.kl_rate = kl_rate
        self.num_negatives = 1
        self.enableKL = enableKL
        self.enableS2S = enableS2S
        self.mode = mode
        self.build()

    def build(self):
        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        encoder_inputs = Input(shape=(self.max_len,))
        kl_inputs = Input(shape=(1,))

        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        encoder_lstm = GRU(hidden_dim, return_state=True)

        x = encoder_embedding(encoder_inputs)
        _, state = encoder_lstm(x)

        
        mean = Dense(latent_dim, name="kl_mean")
        var = Dense(latent_dim)

        state_mean = mean(state)
        state_var = var(state)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                      stddev=1)
            return z_mean + K.exp(z_log_var / 2) * epsilon 

        state_z = Lambda(sampling, name="kl")([state_mean, state_var]) if self.enableKL else state_mean
        
        decoder_inputs = Input(shape=(self.max_len,))
        
        
        latent2hidden = Dense(hidden_dim)
        decoder_lstm = GRU(hidden_dim, return_sequences=True)
        decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")
        
        
        x = encoder_embedding(decoder_inputs)
        decoder_outputs = decoder_lstm(x, initial_state=latent2hidden(state_z))
        rec_outputs = decoder_dense(decoder_outputs)

        
        if self.mode > 1:

            pos_inputs = Input(shape=(self.max_len,))
            neg_inputs = Input(shape=(self.max_len,))
            pos_decoder_inputs = Input(shape=(self.max_len,))
            neg_decoder_inputs = Input(shape=(self.max_len,))

            _, p_state = encoder_lstm(encoder_embedding(pos_inputs))
            _, n_state = encoder_lstm(encoder_embedding(neg_inputs))


            if self.mode == 8:
                p_state = merge([state, p_state], mode="mul")
                n_state = merge([state, n_state], mode="mul")


            p_state_mean = mean(p_state)
            p_state_var = var(p_state)

            n_state_mean = mean(n_state)
            n_state_var = var(n_state)

            p_state_z = Lambda(sampling, name="pos_kl")([p_state_mean, p_state_var])
            n_state_z = Lambda(sampling, name="neg_kl")([n_state_mean, n_state_var])


            if self.mode in [2,4]:
                query_sem = state_mean
                pos_doc_sem = p_state_mean
                neg_doc_sem = n_state_mean
            elif self.mode in [3,5]:
                query_sem = state_z
                pos_doc_sem = p_state_z
                neg_doc_sem = n_state_z
            elif self.mode in [6,7]:
                query_sem = state
                pos_doc_sem = p_state
                neg_doc_sem = n_state

            if self.mode < 8:

                R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
                R_Q_D_ns = dot([query_sem, neg_doc_sem], axes = 1, normalize = True) # See equation (4).
                concat_Rs = concatenate([R_Q_D_p, R_Q_D_ns])
                concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)
                weight = np.array([1]).reshape(1, 1, 1)
                with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
                with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)
                pairwise_pred = Activation("softmax", name="pair")(with_gamma) # See equation (5).

                pos_rec_outputs = decoder_dense(decoder_lstm(encoder_embedding(pos_decoder_inputs), initial_state=latent2hidden(p_state_z)))
                neg_rec_outputs = decoder_dense(decoder_lstm(encoder_embedding(neg_decoder_inputs), initial_state=latent2hidden(n_state_z)))
            else:
                pos_rec_outputs = decoder_dense(decoder_lstm(encoder_embedding(decoder_inputs), initial_state=latent2hidden(p_state_z)))
                neg_rec_outputs = decoder_dense(decoder_lstm(encoder_embedding(decoder_inputs), initial_state=latent2hidden(n_state_z)))



            if self.mode < 8:

                def kl_loss(x, x_):

                    kl_loss = - 0.5 * K.sum(1 + state_var - K.square(state_mean) - K.exp(state_var), axis=-1)
                    p_kl_loss = - 0.5 * K.sum(1 + p_state_var - K.square(p_state_mean) - K.exp(p_state_var), axis=-1)
                    n_kl_loss = - 0.5 * K.sum(1 + n_state_var - K.square(n_state_mean) - K.exp(n_state_var), axis=-1)

                    return kl_inputs * (kl_loss + p_kl_loss + n_kl_loss)

                self.model = Model([encoder_inputs, pos_inputs, neg_inputs, decoder_inputs, pos_decoder_inputs, neg_decoder_inputs, kl_inputs], [rec_outputs, pos_rec_outputs, neg_rec_outputs, state_z, pairwise_pred])
                if self.mode in [2,3]:
                    self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', kl_loss, 'categorical_crossentropy'], loss_weights=[0.05,0.05,0.05,0.05,2])
                elif self.mode in [4,5]:
                    self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', kl_loss, 'categorical_crossentropy'], loss_weights=[1,1,-1,1,1])
                elif self.mode in [7]:
                    self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', kl_loss, 'categorical_crossentropy'], loss_weights=[0.05,0.05,-0.05,0.05,2])
                elif self.mode in [6]:
                    self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', kl_loss, 'categorical_crossentropy'], loss_weights=[0.05,0.05,0.05,0.05,2])

                self.encoder = Model(encoder_inputs, state_mean)
            else:

                def kl_loss(x, x_):

                    p_kl_loss = - 0.5 * K.sum(1 + p_state_var - K.square(p_state_mean) - K.exp(p_state_var), axis=-1)
                    n_kl_loss = - 0.5 * K.sum(1 + n_state_var - K.square(n_state_mean) - K.exp(n_state_var), axis=-1)

                    return kl_inputs * (p_kl_loss + n_kl_loss)


                self.model = Model([encoder_inputs, pos_inputs, neg_inputs, decoder_inputs, kl_inputs], [pos_rec_outputs, neg_rec_outputs, p_state_z])
                self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', kl_loss], loss_weights=[1,-1,0.05])
                self.encoder = Model(encoder_inputs, state)
        else:

            

            if not self.enableS2S:

                if self.enableKL:

                    def kl_annealing_loss(x, x_):
                        kl_loss = - 0.5 * K.sum(1 + state_var - K.square(state_mean) - K.exp(state_var), axis=-1)
                        return kl_inputs * kl_loss

                    self.model = Model([encoder_inputs, decoder_inputs, kl_inputs], [rec_outputs, state_z])
                    self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy', kl_annealing_loss])
                
                else:
                    def kl__loss(x, x_):
                        kl_loss = - 0.5 * K.sum(1 + state_var - K.square(state_mean) - K.exp(state_var), axis=-1)
                        return kl_loss

                    self.model = Model([encoder_inputs, decoder_inputs], [rec_outputs, state_z])
                    self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy', kl__loss])        
                

            elif not self.enableKL and self.enableS2S:
                self.model = Model([encoder_inputs, decoder_inputs], [rec_outputs])
                self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy'])

            self.encoder = Model(encoder_inputs, state_mean)

    def name(self):
        if not self.enableKL and self.enableS2S:
            return "seq2seq_ori"
        elif not self.enableKL and not self.enableS2S:
            return "vae"
        elif self.enableKL and not self.enableS2S:
            return "vae_kl"

        if self.mode == 1:
            return "seq2seq_kl%.2f_wd%.2f" % (self.kl_rate, self.keep_rate_word_dropout)
        else:
            return "s2s_dssm_m%d_kl%.2f_wd%.2f" % (self.mode, self.kl_rate, self.keep_rate_word_dropout)
    
    def word_dropout(self, x, unk_token):

        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

        return x_



class SSVAE(object):
    

    def __init__(self, nb_words, max_len, embedding_matrix, dim, comp_topk=None, ctype=None, epsilon_std=1.0, optimizer=Adadelta(lr=2.), kl_weight=0, FocusMode="fair", mode=1):
        self.dim = dim
        self.embedding_matrix = embedding_matrix
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.max_len = max_len
        self.nb_words = nb_words
        self.optimizer = optimizer
        self.kl_weight = kl_weight
        self.num_negatives = 1
        self.FocusMode = FocusMode
        self.mode = mode

        if self.FocusMode == "fair":
            # self.loss_weights = [1/self.max_len,1/self.max_len,1/self.max_len,1]
            self.loss_weights = [1] * 4
        elif self.FocusMode == "pair":
            self.loss_weights = [0.05] * 3 + [2] 
        elif self.FocusMode == "rec":
            self.loss_weights = [0.9] * 3 + [0.1] 


        self.build()

    def build(self):
        
        query = Input(shape = (self.max_len,))
        pos_doc = Input(shape = (self.max_len,))
        neg_doc = Input(shape = (self.max_len,))




        emb_layer = Embedding(self.nb_words,
                            self.embedding_matrix.shape[-1],
                            weights=[self.embedding_matrix],
                            input_length=self.max_len,
                            trainable=True)

        enc_lstm = GRU(self.dim[0], return_state=True)

            
        # query_sem, query_state_h, query_state_c = enc_lstm(emb_layer(query))
        # pos_doc_sem, pos_doc_state_h, pos_doc_state_c = enc_lstm(emb_layer(pos_doc))
        # neg_doc_sem, neg_doc_state_h, neg_doc_state_c = enc_lstm(emb_layer(neg_doc))

        _, query_sem = enc_lstm(emb_layer(query))
        _, pos_doc_sem = enc_lstm(emb_layer(pos_doc))
        _, neg_doc_sem = enc_lstm(emb_layer(neg_doc))

        

        
        # mean_layer = Dense(self.dim[1])
        # var_layer = Dense(self.dim[1])

        # self.q_mean_state_h = mean_layer(query_state_h)
        # self.q_log_var_state_h = var_layer(query_state_h)
        # self.q_mean_state_c = mean_layer(query_state_c)
        # self.q_log_var_state_c = var_layer(query_state_c)


        # self.pd_mean_state_h = mean_layer(pos_doc_state_h)
        # self.pd_log_var_state_h = var_layer(pos_doc_state_h)
        # self.pd_mean_state_c = mean_layer(pos_doc_state_c)
        # self.pd_log_var_state_c = var_layer(pos_doc_state_c)


        # # only support 1 negative samples
        # self.nd_mean_state_h = mean_layer(neg_doc_state_h)
        # self.nd_log_var_state_h = var_layer(neg_doc_state_h)
        # self.nd_mean_state_c = mean_layer(neg_doc_state_c)
        # self.nd_log_var_state_c = var_layer(neg_doc_state_c)

        # if self.comp_topk != None:
        #     # TODO
        #     kcomp_layer = KCompetitive(self.comp_topk, self.ctype)
        #     self.q_mean_k = kcomp_layer(self.q_mean)
        #     self.pd_mean_k = kcomp_layer(self.pd_mean)
        #     self.nd_mean_k = kcomp_layer(self.nd_mean)

        #     encoded_q = Lambda(self.sampling, output_shape=(self.dim[1],))([self.q_mean_k, self.q_log_var])
        #     encoded_pd = Lambda(self.sampling, output_shape=(self.dim[1],))([self.pd_mean_k, self.pd_log_var])
        #     encoded_nd = Lambda(self.sampling, output_shape=(self.dim[1],))([self.nd_mean_k, self.nd_log_var])
        # else:
        #     encoded_q_h = Lambda(self.sampling, output_shape=(self.dim[1],))([self.q_mean_state_h, self.q_log_var_state_h])
        #     encoded_q_c = Lambda(self.sampling, output_shape=(self.dim[1],))([self.q_mean_state_c, self.q_log_var_state_c])

        #     encoded_pd_h = Lambda(self.sampling, output_shape=(self.dim[1],))([self.pd_mean_state_h, self.pd_log_var_state_h])
        #     encoded_pd_c = Lambda(self.sampling, output_shape=(self.dim[1],))([self.pd_mean_state_c, self.pd_log_var_state_c])

        #     encoded_nd_h = Lambda(self.sampling, output_shape=(self.dim[1],))([self.nd_mean_state_h, self.nd_log_var_state_h])
        #     encoded_nd_c = Lambda(self.sampling, output_shape=(self.dim[1],))([self.nd_mean_state_c, self.nd_log_var_state_c])


        # query_encoder_states = [encoded_q_h, encoded_q_c]
        # pos_doc_encoder_states = [encoded_pd_h, encoded_pd_c]
        # neg_doc_encoder_states = [encoded_nd_h, encoded_nd_c]

        query_encoder_states = query_sem
        pos_doc_encoder_states = pos_doc_sem
        neg_doc_encoder_states = neg_doc_sem


        #       DSSM
        if self.mode == 1:
            R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
            R_Q_D_n = dot([query_sem, neg_doc_sem], axes = 1, normalize = True)
        elif self.mode == 2: 
            R_Q_D_p = dot([encoded_q_h, encoded_pd_h], axes = 1, normalize = True) # See equation (4).
            R_Q_D_n = dot([encoded_q_h, encoded_nd_h], axes = 1, normalize = True)
        elif self.mode == 3:
            R_Q_D_p = dot([encoded_q_c, encoded_pd_c], axes = 1, normalize = True) # See equation (4).
            R_Q_D_n = dot([encoded_q_c, encoded_nd_c], axes = 1, normalize = True)
        elif self.mode == 4:
            R_Q_D_p = dot([self.q_mean_state_h, self.pd_mean_state_h], axes = 1, normalize = True) # See equation (4).
            R_Q_D_n = dot([self.q_mean_state_h, self.nd_mean_state_h], axes = 1, normalize = True)
        elif self.mode == 5:
            R_Q_D_p = dot([self.q_mean_state_c, self.pd_mean_state_c], axes = 1, normalize = True) # See equation (4).
            R_Q_D_n = dot([self.q_mean_state_c, self.nd_mean_state_c], axes = 1, normalize = True)



        concat_Rs = concatenate([R_Q_D_p] + [R_Q_D_n])
        concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)
        weight = np.array([1]).reshape(1, 1, 1)
        with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
        with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)
        pairwise_pred = Activation("softmax", name="pair")(with_gamma) # See equation (5).




        # Set up the decoder, using `encoder_states` as initial state.

        # dec_query = Input(shape = (self.max_len,))
        dec_pos_doc = Input(shape = (self.max_len,))
        dec_neg_doc = Input(shape = (self.max_len,))

        # decoder_embedding = Embedding(self.nb_words,
        #                                 self.embedding_matrix.shape[-1],
        #                                 weights=[self.embedding_matrix],
        #                                 input_length=self.max_len,
        #                                 mask_zero=True,
        #                                 trainable=True)

        decoder_lstm = GRU(self.dim[0], return_sequences=True)
        softmax_layer = Dense(self.nb_words, activation='softmax')



        # dec_query_embeding = decoder_embedding(dec_query)
        # dec_pos_doc_embeding = decoder_embedding(dec_pos_doc)
        # dec_neg_doc_embeding = decoder_embedding(dec_neg_doc)

        # dec_query_embeding = emb_layer(dec_query)
        dec_pos_doc_embeding = emb_layer(dec_pos_doc)
        dec_neg_doc_embeding = emb_layer(dec_neg_doc)


        # dec_query_outputs = decoder_lstm(dec_query_embeding, initial_state=query_encoder_states)
        # dec_pos_doc_outputs = decoder_lstm(dec_pos_doc_embeding, initial_state=pos_doc_encoder_states)
        # dec_neg_doc_outputs = decoder_lstm(dec_neg_doc_embeding, initial_state=neg_doc_encoder_states)

        dec_pos_doc_outputs = decoder_lstm(dec_pos_doc_embeding, initial_state=query_encoder_states)
        dec_neg_doc_outputs = decoder_lstm(dec_neg_doc_embeding, initial_state=query_encoder_states)

        # elif self.mode == 2:
            # decoder_outputs, _, _ = decoder_lstm(RepeatVector(self.max_len)(encoder_states))
        

        # dec_q = softmax_layer(dec_query_outputs)
        dec_pd = softmax_layer(dec_pos_doc_outputs)
        dec_nd = softmax_layer(dec_neg_doc_outputs)


        def pos_rec_loss(y_true, y_pred):
            return K.sparse_categorical_crossentropy(y_true, y_pred)

        def neg_rec_loss(y_true, y_pred):
            return -1 * K.sparse_categorical_crossentropy(y_true, y_pred)


        self.model = Model([query, pos_doc, neg_doc, dec_pos_doc, dec_neg_doc] , [dec_pd, dec_nd, pairwise_pred])
        # self.model.compile(optimizer=self.optimizer, loss= 3 * ["sparse_categorical_crossentropy"] + ["categorical_crossentropy"], loss_weights=self.loss_weights)
        self.model.compile(optimizer=self.optimizer, loss =  [ pos_rec_loss, neg_rec_loss, "categorical_crossentropy"])

        if self.mode == 1:
            self.encoder = Model(outputs=query_sem, inputs=query)
        elif self.mode == 2:
            self.encoder = Model(outputs=encoded_q_h, inputs=query)
        elif self.mode == 3:
            self.encoder = Model(outputs=encoded_q_c, inputs=query)
        elif self.mode == 4:
            self.encoder = Model(outputs=self.q_mean_state_h, inputs=query)
        elif self.mode == 5:
            self.encoder = Model(outputs=self.q_mean_state_c, inputs=query)

    


    def kl_loss(self, x, x_):

        q_kl_loss = - 0.5 * K.sum(1 + self.q_log_var - K.square(self.q_mean) - K.exp(self.q_log_var), axis=-1) 
        pd_kl_loss = - 0.5 * K.sum(1 + self.pd_log_var - K.square(self.pd_mean) - K.exp(self.pd_log_var), axis=-1) 
        nd_kl_loss = - 0.5 * K.sum(1 + self.nd_log_var - K.square(self.nd_mean) - K.exp(self.nd_log_var), axis=-1) 

        return K.mean(self.kl_weight * (q_kl_loss + pd_kl_loss + nd_kl_loss))

    def name(self):
        return "ssvae%s_%d" % (self.FocusMode, self.mode) if self.comp_topk == None else "sskate_%s_%d" % (self.FocusMode, self.mode)

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon


class SSVAE2(object):
    

    def __init__(self, nb_words, max_len, embedding_matrix, dim, comp_topk=None, ctype=None, epsilon_std=1.0, optimizer=Adadelta(lr=2.), kl_weight=0, FocusMode="fair", mode=1):
        self.dim = dim
        self.embedding_matrix = embedding_matrix
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.max_len = max_len
        self.nb_words = nb_words
        self.optimizer = optimizer
        self.kl_weight = kl_weight
        self.num_negatives = 1
        self.FocusMode = FocusMode
        self.mode = mode

        if self.FocusMode == "fair":
            self.loss_weights = [1] * 4
        elif self.FocusMode == "pair":
            self.loss_weights = [0.05] * 3 + [2] 
        elif self.FocusMode == "rec":
            self.loss_weights = [0.9] * 3 + [0.1] 


        self.build()

    def build(self):
        
        query = Input(shape = (self.max_len,))
        pos_doc = Input(shape = (self.max_len,))
        neg_doc = Input(shape = (self.max_len,))


        emb_layer = Embedding(self.nb_words,
                            self.embedding_matrix.shape[-1],
                            weights=[self.embedding_matrix],
                            input_length=self.max_len,
                            mask_zero=True,
                            trainable=True)

        enc_lstm = GRU(self.dim[0], return_state=False)

        mean = Dense(self.dim[1])
        var = Dense(self.dim[1])

        query_sem = enc_lstm(emb_layer(query))
        pos_doc_sem = enc_lstm(emb_layer(pos_doc))
        neg_doc_sem = enc_lstm(emb_layer(neg_doc))

        z_mean = mean(query_sem)
        z_log_var = var(query_sem)

        

        z = Lambda(self.sampling, output_shape=(self.dim[1],))([z_mean, z_log_var])


        R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
        R_Q_D_n = dot([query_sem, neg_doc_sem], axes = 1, normalize = True)


        concat_Rs = concatenate([R_Q_D_p] + [R_Q_D_n])
        concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)
        weight = np.array([1]).reshape(1, 1, 1)
        with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
        with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)
        pairwise_pred = Activation("softmax", name="pair")(with_gamma) # See equation (5).


        decoder_lstm = GRU(self.dim[0], return_sequences=True)
        softmax_layer = Dense(self.nb_words, activation='softmax', name="rec")

        dec_doc_outputs = decoder_lstm(RepeatVector(self.max_len)(z))

        dec_doc_outputs = softmax_layer(dec_doc_outputs)


        def pos_rec_loss(y_true, y_pred):
            return K.sparse_categorical_crossentropy(y_true, y_pred)

        def neg_rec_loss(y_true, y_pred):
            return -1 * K.sparse_categorical_crossentropy(y_true, y_pred)

        def kl_loss(y_true, y_pred):
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) 
            return kl_loss


        self.model = Model([query, pos_doc, neg_doc] , [dec_doc_outputs, dec_doc_outputs, dec_doc_outputs, pairwise_pred])
        self.model.compile(optimizer=self.optimizer, loss =  [ pos_rec_loss, neg_rec_loss, kl_loss, "categorical_crossentropy"])
        self.encoder = Model(outputs=query_sem, inputs=query)

    
    def name(self):
        return "ssvae2%s_%d" % (self.FocusMode, self.mode) if self.comp_topk == None else "sskate2_%s_%d" % (self.FocusMode, self.mode)

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon
