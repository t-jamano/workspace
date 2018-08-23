from Models import *

class BOW_VAE(object):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), kl_rate=0.01, mode=1, enableKL=False, comp_topk=None, ctype=None, epsilon_std=1.0, PoolMode="max"):
        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.kl_rate = kl_rate
        self.enableKL = enableKL
        self.mode = mode
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.PoolMode = PoolMode
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
                                        trainable=True)


        Pooling = GlobalMaxPooling1D() if self.PoolMode == "max" else GlobalAveragePooling1D()


        x = encoder_embedding(encoder_inputs)
        state = Pooling(x)

        
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
        if self.comp_topk != None:
            state_mean_k = KCompetitive(self.comp_topk, self.ctype)(state_mean)
            state_z = Lambda(sampling, name="kl")([state_mean_k, state_var])
        else:
            state_z = Lambda(sampling, name="kl")([state_mean, state_var])

        
        latent2hidden = Dense(hidden_dim)
        hidden2vocab = Dense(self.nb_words, activation='sigmoid', name="rec")
        
        rec_outputs = hidden2vocab(latent2hidden(state_z))

        if self.enableKL:

            def kl_annealing_loss(x, x_):
                kl_loss = - 0.5 * K.sum(1 + state_var - K.square(state_mean) - K.exp(state_var), axis=-1)
                return kl_inputs * kl_loss

            self.model = Model([encoder_inputs, kl_inputs], [rec_outputs, state_z])
            self.model.compile(optimizer=self.optimizer, loss=['binary_crossentropy', kl_annealing_loss])
        
        else:
            def kl_loss(x, x_):
                kl_loss = - 0.5 * K.sum(1 + state_var - K.square(state_mean) - K.exp(state_var), axis=-1)
                return kl_loss

            self.model = Model([encoder_inputs], [rec_outputs, state_z])
            self.model.compile(optimizer=self.optimizer, loss=['binary_crossentropy', kl_loss])        
                


        self.encoder = Model(encoder_inputs, state_mean)

    def name(self):
        model = "bow_kate" if self.comp_topk != None else "bow_vae"
        isKL = "_kl" if self.enableKL else ""
        pool = "_" + self.PoolMode

        return model + isKL + pool
    
