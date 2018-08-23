from Models import *



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




class Seq2Seq():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), kl_rate=0.01, keep_rate_word_dropout=0.5, mode=1, enableKL=True, enableS2S=True, comp_topk=None, ctype=None, epsilon_std=1.0, separateEmbedding=False, enableNeg=False):
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
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.separateEmbedding = separateEmbedding
        self.enableNeg = enableNeg

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
                                        name="enc_embedding",
                                        mask_zero=True,
                                        trainable=True)

        encoder_lstm = GRU(hidden_dim, return_state=True, name="enc_gru")

        x = encoder_embedding(encoder_inputs)
        _, state = encoder_lstm(x)

        
        mean = Dense(latent_dim)
        var = Dense(latent_dim)

        state_mean = mean(state)
        state_var = var(state)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                      stddev=1)
            return z_mean + K.exp(z_log_var / 2) * epsilon 


        # model with no kl loss
        if self.enableS2S and not self.enableKL:
            state_z = state_mean
        else:
            if self.comp_topk != None:
                state_mean_k = KCompetitive(self.comp_topk, self.ctype)(state_mean)
                state_z = Lambda(sampling, name="kl")([state_mean_k, state_var])
            else:
                state_z = Lambda(sampling, name="kl")([state_mean, state_var])

        
        decoder_inputs = Input(shape=(self.max_len,))
        decoder_inputs2 = Input(shape=(self.max_len,))

        
        
        latent2hidden = Dense(hidden_dim)
        decoder_lstm = GRU(hidden_dim, return_sequences=True, name="dec_gru")
        decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")
        decoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="dec_embedding",
                                        mask_zero=True,
                                        trainable=True)
        
        
        x = encoder_embedding(decoder_inputs) if not self.separateEmbedding else decoder_embedding(decoder_inputs)
        x2 = encoder_embedding(decoder_inputs2) if not self.separateEmbedding else decoder_embedding(decoder_inputs2)
        
        decoder_outputs = decoder_lstm(x, initial_state=latent2hidden(state_z))
        decoder_outputs2 = decoder_lstm(x2, initial_state=latent2hidden(state_z))

        rec_outputs = decoder_dense(decoder_outputs)
        rec_outputs2 = decoder_dense(decoder_outputs2)


        
        if self.mode > 1:

            pos_inputs = Input(shape=(self.max_len,))
            neg_inputs = Input(shape=(self.max_len,))
            pos_decoder_inputs = Input(shape=(self.max_len,))
            neg_decoder_inputs = Input(shape=(self.max_len,))

            _, p_state = encoder_lstm(encoder_embedding(pos_inputs))
            _, n_state = encoder_lstm(encoder_embedding(neg_inputs))


            # if self.mode == 8:
                # p_state = merge([state, p_state], mode="mul")
                # n_state = merge([state, n_state], mode="mul")


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
                    self.model = Model([encoder_inputs, pos_inputs, neg_inputs, decoder_inputs, pos_decoder_inputs, neg_decoder_inputs, kl_inputs], [rec_outputs, pos_rec_outputs, neg_rec_outputs, state_z, pairwise_pred])
                    self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', kl_loss, 'categorical_crossentropy'], loss_weights=[1e-3,1e-3,-1e-3,1e-3,1])
                elif self.mode in [4,5]:
                    self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', kl_loss, 'categorical_crossentropy'], loss_weights=[1e-3,1e-3,-1e-3,1e-3,1])
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

                    if self.enableNeg:

                        def pos_rec_loss(y_true, y_pred):
                            return K.sparse_categorical_crossentropy(y_true, y_pred)

                        def neg_rec_loss(y_true, y_pred):
                            return -1 * K.sparse_categorical_crossentropy(y_true, y_pred)


                        self.model = Model([encoder_inputs, decoder_inputs, decoder_inputs2, kl_inputs], [rec_outputs, rec_outputs2, state_z])
                        self.model.compile(optimizer=self.optimizer, loss=[pos_rec_loss, neg_rec_loss, kl_annealing_loss])
                
                    else:
                        self.model = Model([encoder_inputs, decoder_inputs, kl_inputs], [rec_outputs, state_z])
                        self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy', kl_annealing_loss])
                
                else:
                    def kl_loss(x, x_):
                        kl_loss = - 0.5 * K.sum(1 + state_var - K.square(state_mean) - K.exp(state_var), axis=-1)
                        return kl_loss

                    if self.enableNeg:
                        self.model = Model([encoder_inputs, decoder_inputs, decoder_inputs2], [rec_outputs, rec_outputs2, state_z])
                        self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', kl_loss], loss_weights=[1,-1,1])
                    else:

                        self.model = Model([encoder_inputs, decoder_inputs], [rec_outputs, state_z])
                        self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy', kl_loss])        
                

            elif not self.enableKL and self.enableS2S:
                self.model = Model([encoder_inputs, decoder_inputs], [rec_outputs])
                self.model.compile(optimizer=self.optimizer, loss=['sparse_categorical_crossentropy'])

            self.encoder = Model(encoder_inputs, state_mean)

    def name(self):

        if self.enableNeg:
            return "vae_neg" if not self.enableKL else "vae_neg_kl"

        if not self.enableKL and self.enableS2S:
            if self.mode == 2:
                return "seq2seq_ori_dssm"
            return "seq2seq_ori" if not self.separateEmbedding else "seq2seq_ori2"
        elif not self.enableKL and not self.enableS2S:
            return "vae" if self.comp_topk == None else "kate"
        elif self.enableKL and not self.enableS2S:
            return "vae_kl" if self.comp_topk == None else "kate_kl"

        if self.mode == 1:
            return "seq2seq_kl%.2f_wd%.2f" % (self.kl_rate, self.keep_rate_word_dropout)
        else:
            return "s2s_dssm_m%d_kl%.2f_wd%.2f" % (self.mode, self.kl_rate, self.keep_rate_word_dropout)
    
    def word_dropout(self, x, unk_token):
        np.random.seed(0)
        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

        return x_



class SSVAE():
    

    def __init__(self, nb_words, max_len, embedding_matrix, dim, comp_topk=None, ctype=None, epsilon_std=1.0, optimizer=Adam(lr=2.), kl_weight=0, FocusMode="fair", mode=1):
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


class AE():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), keep_rate_word_dropout=0.5):
        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.keep_rate_word_dropout = keep_rate_word_dropout

        self.build()

    def build(self):
        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        encoder_inputs = Input(shape=(self.max_len,))

        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="enc_embedding",
                                        mask_zero=True,
                                        trainable=True)

        fro_encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="fro_enc_embedding",
                                        mask_zero=False,
                                        trainable=False)

        tmp = GlobalAveragePooling1D()(fro_encoder_embedding(encoder_inputs))

        encoder_lstm = GRU(hidden_dim, return_state=True, name="enc_gru")

        x = encoder_embedding(encoder_inputs)
        _, state = encoder_lstm(x)

        mean = Dense(latent_dim, activation="tanh")

        state_mean = mean(state)

        state_z = state_mean

        dot_output = merge([tmp, state], mode="cos")
        
        # decoder_inputs = Input(shape=(self.max_len,))
        
        latent2hidden = Dense(hidden_dim, activation="tanh")
        decoder_lstm = GRU(hidden_dim, return_sequences=True, name="dec_gru")
        decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")
        decoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="dec_embedding",
                                        mask_zero=True,
                                        trainable=True)
        
        
        # x = encoder_embedding(decoder_inputs)
        # decoder_outputs = decoder_lstm(x, initial_state=latent2hidden(state_z))
        decoder_outputs = decoder_lstm(RepeatVector(self.max_len)(state_z))

        rec_outputs = decoder_dense(decoder_outputs)

        

        # self.model = Model([encoder_inputs, decoder_inputs], rec_outputs)
        self.model = Model([encoder_inputs], [rec_outputs, dot_output])
        self.model.compile(optimizer=self.optimizer, loss=["sparse_categorical_crossentropy", "cosine_proximity"])
        self.encoder = Model(encoder_inputs, state_mean)

    def name(self):
        return "ae"
    
    def word_dropout(self, x, unk_token):

        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

        return x_
