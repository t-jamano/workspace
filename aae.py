from Models import *
from vae import *

class AdversarialAutoEncoderModel():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), keep_rate_word_dropout=0.5, enableWasserstein=False, mode=1, enableS2S=False, enableSeparate=False):

        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.keep_rate_word_dropout = keep_rate_word_dropout
        self.hidden_dim = self.dim[0]
        self.latent_dim = self.dim[1]
        self.mode = mode
        self.enableS2S = enableS2S
        self.enableWasserstein = enableWasserstein
        self.enableSeparate = enableSeparate
        self.adversarial_optimizer = AdversarialOptimizerAlternating()
        self.dis_loss = "binary_crossentropy" if not self.enableWasserstein else self.wasserstein_loss
        self.build()

    def build(self):

        query_inputs = Input(shape=(self.max_len,))

        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="q_embedding",
                                        trainable=True)

        dense = Dense(self.latent_dim, activation="tanh", name="q_dense")

        encoder_lstm = Bidirectional(GRU(self.hidden_dim, return_sequences=True), name='q_gru')


        state = dense(GlobalMaxPooling1D()(encoder_lstm(encoder_embedding(query_inputs))))



        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)

        self.state_mean = self.mean(state)
        self.state_var = self.var(state)

        state_z = Lambda(self.sampling)([self.state_mean, self.state_var])

        # Adversarial
        self.discriminator = self.build_discriminator()
        gs_latents = normal_latent_sampling((self.latent_dim,))(query_inputs)

        query_real = self.discriminator(gs_latents)
        query_fake = self.discriminator(state_z)


        decoder_inputs = Input(shape=(self.max_len,))


        self.latent2hidden = Dense(self.hidden_dim, activation="relu")
        decoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru")
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="rec", use_bias=False, weights=[self.embedding_matrix.T])

        rec_outputs = self.decoder_dense(self.decoder_lstm(decoder_embedding(decoder_inputs), initial_state=self.latent2hidden(state_z)))

        self.ae = Model([query_inputs, decoder_inputs], [rec_outputs])

        inputs = self.ae.inputs
        outputs = fix_names([rec_outputs, query_fake, query_real], ["qpred","qfake","qreal"])
        self.aae = Model(inputs, outputs)

        generative_params = self.ae.trainable_weights
        discriminative_params = self.discriminator.trainable_weights
        self.model = AdversarialModel(base_model=self.aae, player_params=[generative_params, discriminative_params], player_names=["generator", "discriminator"])        
        rec_loss = "sparse_categorical_crossentropy"
        
        self.model.adversarial_compile(adversarial_optimizer=self.adversarial_optimizer, player_optimizers=[self.optimizer, self.optimizer], loss={"qfake": self.dis_loss, "qreal": self.dis_loss, "qpred": rec_loss}, player_compile_kwargs=[{"loss_weights": {"qfake": 1e-2, "qreal": 1e-2, "qpred": 1}}] * 2)
        self.encoder = Model(query_inputs, state)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def name(self):
        if self.enableS2S:
            return "aae_s2s" if not self.enableWasserstein else "wae_s2s"
        if self.enableSeparate:
            return "aae2" if not self.enableWasserstein else "wae2"
            
        return "aae" if not self.enableWasserstein else "wae"
    
    def word_dropout(self, x, unk_token):
        # np.random.seed(0)
        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

            return x_

    def build_discriminator(self):
        
        inputs = Input((self.latent_dim,), name="gs_dis_input")
        
        dense1 = Dense(self.hidden_dim, activation="relu")
        dense2 = Dense(self.latent_dim, activation="relu")
        dense3 = Dense(1, activation="sigmoid" if not self.enableWasserstein else "linear")

        outputs = dense3(dense2(dense1(inputs)))
        # outputs = dense3(inputs)

        
        return Model(inputs, outputs)

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                  stddev=1)
        return z_mean + K.exp(z_log_var / 2) * epsilon


