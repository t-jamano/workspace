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
    

    def __init__(self, input_size, max_len, emb, dim, batch_size, comp_topk=None, ctype=None, epsilon_std=1.0, optimizer=Adadelta(lr=2.), kl_weight=0, cos_weight=1):
        self.input_size = input_size
        self.dim = dim
        self.emb = emb
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

    def build(self):
        act = 'tanh'
        input_q = Input(shape=(self.max_len,))
        input_d = Input(shape=(self.max_len,))

        hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        hidden_layer1_cos = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        
        embed_q = GlobalMaxPooling1D()(self.emb[0](input_q))
        embed_d = GlobalMaxPooling1D()(self.emb[0](input_d))

        h1_q = hidden_layer1(embed_q)
        h1_d = hidden_layer1(embed_d)
        
        
        embed_q_cos = GlobalMaxPooling1D()(self.emb[1](input_q))
        embed_d_cos = GlobalMaxPooling1D()(self.emb[1](input_d))

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

#         self.cos_loss = Flatten()(merge([h1_q_cos, h1_d_cos], mode="cos"))
        cos_output = Flatten()(merge([h1_q, h1_d], mode="cos"))

        
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
        self.model.compile(optimizer=self.optimizer, loss=[zero_loss, "binary_crossentropy"], loss_weights=[1, self.cos_weight])
        # build a model to project inputs on the latent space
        self.encoder = Model(outputs=mean_q, inputs=input_q)
        
        
        
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon