from Models import *

class VDSH_Loss_Layer(Layer):
    def __init__(self, max_len, batch_size, kl_weight, **kwargs):
        self.is_placeholder = True
        super(VDSH_Loss_Layer, self).__init__(**kwargs)
        self.target_weights = tf.constant(np.ones((batch_size, max_len)), tf.float32)
        self.kl_weight = kl_weight

    
    def cal_rec_loss(self, input_layer, output_layer):
        labels = tf.cast(input_layer, tf.int32)

        loss = K.sum(tf.contrib.seq2seq.sequence_loss(output_layer, labels, 
                                                     weights=self.target_weights,
                                                     average_across_timesteps=False,
                                                     average_across_batch=False), axis=-1)
        return loss
        
    def cal_kl_loss(self, z_mean, z_log_var):
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return kl_loss

    def call(self, inputs):
        q, q_decoded, q_mean, q_var, d, d_decoded, d_mean, d_var = inputs
        
        q_rec_loss = self.cal_rec_loss(q, q_decoded)
        d_rec_loss = self.cal_rec_loss(d, d_decoded)
        
        q_kl_loss = self.cal_kl_loss(q_mean, q_var)
        d_kl_loss = self.cal_kl_loss(d_mean, d_var)
        
        loss = K.mean(q_rec_loss + (self.kl_weight * q_kl_loss) + d_rec_loss + (self.kl_weight * d_kl_loss))
        self.add_loss(loss, inputs=inputs)
        # we don't use this output, but it has to have the correct shape:
        return K.ones_like(q)

class VDSH(object):
    

    def __init__(self, input_size, max_len, embedding_matrix, dim, batch_size, comp_topk=None, ctype=None, epsilon_std=1.0, optimizer=Adadelta(lr=2.), kl_weight=0, cos_weight=1):
        self.input_size = input_size
        self.dim = dim
        self.embedding_matrix = embedding_matrix
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.max_len = max_len
        self.batch_size = batch_size
        self.nb_words = input_size
        self.optimizer = optimizer
        self.kl_weight = kl_weight
        self.cos_weight = cos_weight
        self.target_weights = tf.constant(np.ones((batch_size, max_len)), tf.float32)

        self.build()

    def name(self):
        if self.ctype != None:
            return "vdsh_kate_k%d" % self.comp_topk
        else:
            return "vdsh"

    def build(self):
        act = 'tanh'
        input_q = Input(shape=(self.max_len,))
        input_d = Input(shape=(self.max_len,))

        hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        hidden_layer1_cos = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)

        embedding_layer = Embedding(nb_words,
                            embedding_matrix.shape[-1],
                            weights=[self.embedding_matrix],
                            input_length=max_len,
                            trainable=True)

        q_embedding_layer = Embedding(nb_words,
                            embedding_matrix.shape[-1],
                            weights=[self.embedding_matrix],
                            input_length=max_len,
                            trainable=True)
        
        embed_q = GlobalMaxPooling1D()(embedding_layer(input_q))
        embed_d = GlobalMaxPooling1D()(embedding_layer(input_d))

        h1_q = hidden_layer1(embed_q)
        h1_d = hidden_layer1(embed_d)


        
        
        embed_q_cos = GlobalMaxPooling1D()(q_embedding_layer(input_q))
        embed_d_cos = GlobalMaxPooling1D()(q_embedding_layer(input_d))

        h1_q_cos = hidden_layer1_cos(embed_q_cos)
        h1_d_cos = hidden_layer1_cos(embed_d_cos)
        
#       merge here
#         h1_q = merge([h1_q, h1_q_cos], mode="mul")
#         h1_d = merge([h1_d, h1_d_cos], mode="mul")


        mean_q = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1_q)
        var_q = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1_q)
        
        mean_d = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1_d)
        var_d = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1_d)

        if self.comp_topk != None:
            mean_k_q = KCompetitive(self.comp_topk, self.ctype)(mean_q)
            mean_k_d = KCompetitive(self.comp_topk, self.ctype)(mean_d)

            encoded_q = Lambda(self.sampling, output_shape=(self.dim[1],))([mean_k_q, var_q])
            encoded_d = Lambda(self.sampling, output_shape=(self.dim[1],))([mean_k_d, var_d])

        else:
            encoded_q = Lambda(self.sampling, output_shape=(self.dim[1],))([mean_q, var_q])
            encoded_d = Lambda(self.sampling, output_shape=(self.dim[1],))([mean_d, var_d])

        cos_output = Flatten(name="cos")(merge([h1_q, h1_d], mode="cos"))

        
        repeated_context = RepeatVector(self.max_len)
        decoder_h = Dense(self.dim[1], kernel_initializer='glorot_normal', activation=act)
        decoder_mean = TimeDistributed(Dense(self.nb_words, activation='tanh'))#softmax is applied in the seq2seqloss by tf
       
    
        h_decoded_q = decoder_h(repeated_context(encoded_q))
        decoded_q = decoder_mean(h_decoded_q)
        
        h_decoded_d = decoder_h(repeated_context(encoded_d))
        decoded_d = decoder_mean(h_decoded_d)

        
        
        self.loss_layer = VDSH_Loss_Layer(self.max_len, self.batch_size, self.kl_weight)
        
        vae_output = self.loss_layer([input_q, decoded_q, mean_q, var_q, input_d, decoded_d, mean_d, var_d])

        
        self.model = Model([input_q, input_d], [vae_output, cos_output])
        self.model.compile(optimizer=self.optimizer, loss=[zero_loss, "cosine_proximity"], loss_weights=[1, self.cos_weight])
        # build a model to project inputs on the latent space
        self.encoder = Model(outputs=mean_q, inputs=input_q)
        
        
        
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon

class VAE_LSTM(object):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, comp_topk=None, ctype=None, epsilon_std=1.0, optimizer=Adadelta(lr=2.), enableGAN=False, kl_weight=0):
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std

        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.enableGAN = enableGAN
        self.optimizer = optimizer
        self.kl_weight = kl_weight
        
        act = 'tanh'
        input_layer = Input(shape=(self.max_len,))
        e_mask = Masking(mask_value=0)


        embedding_layer = Embedding(nb_words,
                            embedding_matrix.shape[-1],
                            weights=[self.embedding_matrix],
                            input_length=max_len,
                            trainable=True)

#         bilstm = Bidirectional(LSTM(self.dim[0], name='lstm_1'))
        bilstm = GRU(self.dim[0], return_state=True)


        hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        
        h1 = e_mask(embedding_layer(input_layer))
        _, h1 = bilstm(h1)
        # h1 = hidden_layer1(h1)

        self.z_mean = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)
        self.z_log_var = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)

        if self.comp_topk != None:
            self.z_mean_k = KCompetitive(self.comp_topk, self.ctype)(self.z_mean)
            encoded = Lambda(self.sampling, output_shape=(self.dim[1],))([self.z_mean_k, self.z_log_var])
        else:
            encoded = Lambda(self.sampling, output_shape=(self.dim[1],))([self.z_mean, self.z_log_var])


        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        # decoder_mean = Dense_tied(self.nb_words, activation='softmax', tied_to=hidden_layer1)
        decoder_mean = Dense(self.nb_words, activation='softmax')

        h_decoded = decoder_h(encoded)
        h_decoded = RepeatVector(self.max_len)(h_decoded)
        h_decoded = GRU(self.dim[0], return_sequences=True)(h_decoded)
        x_decoded_mean = TimeDistributed(decoder_mean, name='decoded_mean')(h_decoded)

        if self.enableGAN:
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

            validity = self.discriminator(x_decoded_mean)

        if self.enableGAN:
            self.model = Model(outputs=[x_decoded_mean, validity], inputs=input_layer)
        else:
            self.model = Model(outputs=[x_decoded_mean, x_decoded_mean], inputs=input_layer)

        self.encoder = Model(outputs=self.z_mean, inputs=input_layer)

        # build a digit generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.dim[1],))
        _h_decoded = decoder_h(decoder_input)
        _h_decoded = RepeatVector(self.max_len)(_h_decoded)
        _h_decoded = LSTM(self.dim[0], return_sequences=True, unroll=True)(_h_decoded)
        _x_decoded_mean = TimeDistributed(decoder_mean, name='decoded_mean')(_h_decoded)

        self.decoder = Model(outputs=_x_decoded_mean, inputs=decoder_input)



        if enableGAN:
            self.model.compile(optimizer=self.optimizer, loss=[self.vae_loss, "binary_crossentropy"])
        else:    
            self.model.compile(optimizer=self.optimizer, loss=["sparse_categorical_crossentropy", self.kl_loss])

    def name(self):
        return "kate_lstm_k%d" % self.comp_topk if self.ctype != None else "vae_lstm"
    def kl_loss(self, x, x_):

        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)

        return self.kl_weight * kl_loss



    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon


class SeqVAE(object):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=RMSprop(), word_dropout_prob=0.5, kl_weight=0):
        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.kl_weight = kl_weight
        self.word_dropout_prob = word_dropout_prob
        
        self.build()
        
    def build(self):
                
#       Encoder
        
        e_input = Input(shape=(self.max_len,))
        e_mask = Masking(mask_value=0)

        embedding_layer = Embedding(nb_words,
                            embedding_matrix.shape[-1],
                            weights=[self.embedding_matrix],
                            input_length=max_len,
                            trainable=True)


        e_lstm = GRU(self.dim[0], return_state=True)
        
#         h, state_h, state_c = e_lstm(e_emb(e_mask(e_input)))#         
        _, h = e_lstm(e_mask(embedding_layer(e_input)))

        
        mean = Dense(self.dim[1])
        var = Dense(self.dim[1])
        
#         self.state_h_mean = mean(state_h)
#         self.state_h_log_var = var(state_h)
        
#         self.state_c_mean = mean(state_c)
#         self.state_c_log_var = var(state_c)
        
        self.h_mean = mean(h)
        self.h_log_var = var(h)
        
#         state_h_z = Lambda(self.sampling)([self.state_h_mean, self.state_h_log_var])     
#         state_c_z = Lambda(self.sampling)([self.state_c_mean, self.state_c_log_var])
        z = Lambda(self.sampling)([self.h_mean, self.h_log_var])

#         self.encoder = Model(e_input, self.state_h_mean)
        self.encoder = Model(e_input, self.h_mean)
        self.encoder2 = Model(e_input, z)

        
#       Decoder

        d_input = Input(shape=(self.max_len,))
        d_latent2hidden = Dense(self.dim[0])
        d_lstm = GRU(self.dim[0], return_sequences=True)
        d_output2vocab = TimeDistributed(Dense(self.nb_words, activation="softmax"))


#         state_h_z = d_latent2hidden(state_h_z)
#         state_c_z = d_latent2hidden(state_c_z)
        h_z = d_latent2hidden(z)
        
        d_embed_input = embedding_layer(Dropout(self.word_dropout_prob)(d_input))
#         d_embed_input = e_emb(d_input)

#         outputs = d_lstm(d_embed_input, initial_state=[state_h_z, state_c_z])        
        outputs = d_lstm(d_embed_input, initial_state=[h_z])

        pred = d_output2vocab(outputs)

        
        
#       VAE model
        self.model = Model(inputs=[e_input, d_input], outputs=[pred, pred])
        self.model.compile(optimizer=self.optimizer, loss=["sparse_categorical_crossentropy", self.kl_loss])




    def name(self):
        
        return "seqvae"
    
    def kl_loss(self, x, x_):
#         x = K.flatten(x)
#         x_ = K.flatten(x_)
#         x = tf.cast(x, tf.int32)

#         rec_loss = objectives.sparse_categorical_crossentropy(x,x_)

#         state_h_kl_loss = - 0.5 * K.sum(1 + self.state_h_log_var - K.square(self.state_h_mean) - K.exp(self.state_h_log_var), axis=-1)
#         state_c_kl_loss = - 0.5 * K.sum(1 + self.state_c_log_var - K.square(self.state_c_mean) - K.exp(self.state_c_log_var), axis=-1)
        
#         return (self.kl_weight * state_h_kl_loss) + (self.kl_weight * state_c_kl_loss)
        kl_loss = - 0.5 * K.sum(1 + self.h_log_var - K.square(self.h_mean) - K.exp(self.h_log_var), axis=-1)
        return (self.kl_weight * kl_loss) 

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=1)
    
        return z_mean + K.exp(z_log_var / 2) * epsilon 
    
    def nosampling(self, args):
        z_mean, z_log_var = args
        return z_mean + K.exp(z_log_var / 2)
