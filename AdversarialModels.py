from Models import *



class PRA():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), keep_rate_word_dropout=0.5, mode=1,enableWasserstein=False, enableGRU=False, enableS2S=False):
        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.enableWasserstein = enableWasserstein
        self.enableS2S = enableS2S
        self.num_negatives = 1
        self.mode = mode
        self.enableGRU=enableGRU
        self.keep_rate_word_dropout = keep_rate_word_dropout

        
        self.hidden_dim = self.dim[0]
        self.latent_dim = self.dim[1]

        self.dis_loss = "binary_crossentropy" if not self.enableWasserstein else self.wasserstein_loss
        self.gan_loss = "binary_crossentropy" if not self.enableGRU else "sparse_categorical_crossentropy"
        
        if self.mode == 1:
            self.adversarial_optimizer = AdversarialOptimizerSimultaneous()
        elif self.mode == 2:
            self.adversarial_optimizer = AdversarialOptimizerAlternating()
        elif self.mode == 3:
            self.adversarial_optimizer = AdversarialOptimizerScheduled([1,1,1,1,1,0])

        self.build()

    def build(self):
        
        self.encoder = self.build_main_encoder()
        self.doc_encoder = self.build_main_encoder()
        
        ae_encoders = self.build_ae_encoder()
        ae_decoders = self.build_ae_decoder()
        dssm = self.build_pairwise_model()
        
        self.pr_discriminator = self.build_pr_discriminator()
        self.gs_discriminator = self.build_gs_discriminator()
        
        q_inputs = Input((self.max_len,))
        pd_inputs = Input((self.max_len,))
        nd_inputs = Input((self.max_len,))

        dec_inputs = Input((self.max_len,))

        q_latents = self.encoder(q_inputs)
        pd_latents = self.doc_encoder(pd_inputs)
        nd_latents = self.doc_encoder(nd_inputs)


        ae_enc_latents = ae_encoders(q_latents)
        gs_latents = normal_latent_sampling((self.latent_dim,))(q_inputs)

        
        
        if self.enableGRU:
            # rec_outputs = [ae_decoders([ae_encoders(i), j]) for i, j in zip(main_latents, dec_inputs)]
            rec_outputs = ae_decoders([ae_enc_latents, dec_inputs])

        else:
            # rec_outputs = [ae_decoders(ae_encoders(i)) for i in main_latents]
            rec_outputs = ae_decoders(ae_enc_latents)


        pair_output = dssm([q_latents, pd_latents, nd_latents])
        
        # gs_fake_outputs = [self.gs_discriminator(i) for i in ae_enc_latents]
        # gs_real_outputs = [self.gs_discriminator(i) for i in gs_latents]
        gs_fake_outputs = self.gs_discriminator(ae_enc_latents)
        gs_real_outputs = self.gs_discriminator(gs_latents)
        
        # pair_pr_output = dssm([self.pr_discriminator(i) for i in ae_enc_latents])
        pair_pr_output = dssm([self.pr_discriminator(i) for i in [q_latents, pd_latents, nd_latents]])

        if self.enableGRU:
            # recon_pr_outputs = [ae_decoders([self.pr_discriminator(i), j]) for i, j in zip(ae_enc_latents, dec_inputs)]
            recon_pr_outputs = ae_decoders([self.pr_discriminator(ae_enc_latents), dec_inputs])

        else:
            # recon_pr_outputs = [ae_decoders(self.pr_discriminator(i)) for i in ae_enc_latents]
            recon_pr_outputs = ae_decoders(self.pr_discriminator(ae_enc_latents))


        
        outputs = [rec_outputs, pair_output, gs_fake_outputs, gs_real_outputs, recon_pr_outputs, pair_pr_output]
        

        outputs_name = ["rec", "pair", "yfake", "yreal", "pr_yfake", "pr_yreal"]

        # outputs_name = ["q_pred", "pd_pred", "nd_pred", 
        #                 "pair", "q_yfake", "pd_yfake", 
        #                 "nd_yfake", "q_yreal", "pd_yreal", 
        #                 "nd_yreal", "pr_q_fake", "pr_pd_fake", 
        #                 "pr_nd_fake", "pr_y_real"]

        if self.enableGRU:
            self.vae = Model([q_inputs, pd_inputs, nd_inputs, dec_inputs], fix_names(outputs, outputs_name))
        else:
            self.vae = Model(inputs, fix_names(outputs, outputs_name))

        # build adversarial model
        generative_params = self.encoder.trainable_weights + ae_encoders.trainable_weights + ae_decoders.trainable_weights + dssm.trainable_weights + self.doc_encoder.trainable_weights

        gs_discriminative_params = self.gs_discriminator.trainable_weights
        pr_discriminative_params = self.pr_discriminator.trainable_weights

        self.model = AdversarialModel(base_model=self.vae,
                                 player_params=[generative_params, gs_discriminative_params + pr_discriminative_params],
                                 player_names=["generator", "discriminator"])


        # gen_weights = [{"loss_weights": {"pr_q_fake": 1e-3, "pr_pd_fake": 1e-3, "pr_nd_fake": 1e-3, "pr_y_real": -1e-3, "q_yfake": 1e-3, "pd_yfake": 1e-3, "nd_yfake": 1e-3, "q_yreal": -1e-3, "pd_yreal": -1e-3, "nd_yreal": -1e-3, "q_pred": 1e-1, "pd_pred": 1e-1, "nd_pred": 1e-1, "pair": 1}}]
        # dis_weights = [{"loss_weights": {"pr_q_fake": -1e-3, "pr_pd_fake": -1e-3, "pr_nd_fake": -1e-3, "pr_y_real": 1e-3, "q_yfake": -1e-3, "pd_yfake": -1e-3, "nd_yfake": -1e-3, "q_yreal": 1e-3, "pd_yreal": 1e-3, "nd_yreal": 1e-3, "q_pred": 1e-1, "pd_pred": 1e-1, "nd_pred": 1e-1, "pair": 1}}]
        
        gen_weights = [{"loss_weights": {"pr_yfake": 1e-4, "pr_yreal": -1e-4, "yfake": 1e-4, "yreal": 1e-3, "rec": 1e-3, "pair": 1}}]
        dis_weights = [{"loss_weights": {"pr_yfake": -1e-4, "pr_yreal": 1e-4, "yfake": 1e-4, "yreal": 1e-3, "rec": 1e-3, "pair": 1}}]
      
        # losses = {"q_yfake": self.dis_loss, "q_yreal": self.dis_loss,
        #                             "pd_yfake": self.dis_loss, "pd_yreal": self.dis_loss,
        #                             "nd_yfake": self.dis_loss, "nd_yreal": self.dis_loss,
        #                             "q_pred": self.gan_loss,
        #                             "pd_pred": self.gan_loss,
        #                             "nd_pred": self.gan_loss,
        #                             "pr_q_fake": self.gan_loss,
        #                             "pr_pd_fake": self.gan_loss,
        #                             "pr_nd_fake": self.gan_loss,
        #                             "pair": "categorical_crossentropy",
        #                             "pr_y_real": "categorical_crossentropy"}

        losses = {"yfake": self.dis_loss, "yreal": self.dis_loss,
                            "rec": self.gan_loss,
                            "pair": "categorical_crossentropy",
                            "pr_yfake": self.gan_loss,
                            "pr_yreal": "categorical_crossentropy"}
        
        
        self.model.adversarial_compile(adversarial_optimizer=self.adversarial_optimizer,
                              player_optimizers=[self.optimizer, self.optimizer],
                              loss=losses,
                              player_compile_kwargs=gen_weights+dis_weights)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    def build_main_encoder(self):

        inputs = Input(shape=(self.max_len,))
        
        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="enc_embedding",
                                        mask_zero=False if not self.enableGRU else True,
                                        trainable=True)
        hidden2latent = Dense(self.latent_dim)

        if self.enableGRU:
            encoder_gru = GRU(self.hidden_dim, name="enc_gru")
            outputs = hidden2latent(encoder_gru(encoder_embedding(inputs)))
        else:
            outputs = hidden2latent(GlobalAveragePooling1D()(encoder_embedding(inputs)))
        
        return Model(inputs, outputs)
    

    
    def build_ae_encoder(self):
        inputs = Input((self.latent_dim,), name="ae_encoder_input")
        
        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)
        
        state_z = Lambda(self.sampling, name="kl")([self.mean(inputs), self.var(inputs)])
        
        return Model(inputs, state_z)

    def build_ae_decoder(self):
        
        latent_inputs = Input((self.latent_dim,), name="dec_latent_input")
    
        if self.enableGRU:
            inputs = Input((self.max_len,), name="dec_input")
            latent2hidden = Dense(self.hidden_dim)
            decoder_embedding = Embedding(self.nb_words,
                                            self.embedding_matrix.shape[-1],
                                            weights=[self.embedding_matrix],
                                            input_length=self.max_len,
                                            name="dec_embedding",
                                            mask_zero=True,
                                            trainable=True)
            decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru")
            decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")

            embed_x = decoder_embedding(inputs)
            latents = RepeatVector(self.max_len)(latent_inputs)
            concat = merge([embed_x, latents], mode="concat")
            # concat = concatenate([embed_x, latents], axis=-1)

            outputs = decoder_dense(decoder_lstm(concat))
            return Model([latent_inputs, inputs], outputs)

        else:
            dec_dense = Dense(self.hidden_dim)
            softmax = Dense(self.nb_words, activation="sigmoid")
            outputs = softmax(dec_dense(latent_inputs))
        
            return Model(latent_inputs, outputs)

    
    def build_pairwise_model(self):
        
        query_sem = Input(shape=(self.latent_dim,), name="q_pair_input")
        pos_doc_sem = Input(shape=(self.latent_dim,), name="pos_pair_input")
        neg_doc_sem = Input(shape=(self.latent_dim,), name="neg_pair_input")
        
        weight = np.array([1]).reshape(1, 1, 1)
        conv = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])
        R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
        R_Q_D_ns = dot([query_sem, neg_doc_sem], axes = 1, normalize = True) # See equation (4).
        concat_Rs = concatenate([R_Q_D_p, R_Q_D_ns])
        concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)
        with_gamma = conv(concat_Rs) # See equation (5).
        with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)
        pairwise_pred = Activation("softmax", name="pair")(with_gamma) # See equation (5).
        
        return Model([query_sem, pos_doc_sem, neg_doc_sem], pairwise_pred)
    
    def build_pr_discriminator(self):
        
        inputs = Input((self.latent_dim,), name="pr_dis_input")
        
        dense1 = Dense(self.hidden_dim)
        dense2 = Dense(self.latent_dim)
        
        outputs = dense2(dense1(inputs))
        
        return Model(inputs, outputs)
    
    def build_gs_discriminator(self):
        
        inputs = Input((self.latent_dim,), name="gs_dis_input")
        
        dense1 = Dense(self.hidden_dim)
        dense2 = Dense(self.latent_dim)
        dense3 = Dense(1, activation="sigmoid" if not self.enableWasserstein else "linear")

        outputs = dense3(dense2(dense1(inputs)))
        
        return Model(inputs, outputs)
    
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                  stddev=1)
        return z_mean + K.exp(z_log_var / 2) * epsilon 

    def name(self):
        if self.enableGRU:
            return "pr_aae_m%d" % self.mode if not self.enableWasserstein else "pr_wae_m%d" % self.mode
        else:
            return "bow_pr_avg_aae_m%d" % self.mode if not self.enableWasserstein else "bow_pr_avg_wae_m%d" % self.mode


    def word_dropout(self, x, unk_token):

        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

        return x_

class PRA2(PRA):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), keep_rate_word_dropout=0.5, mode=1, enableWasserstein=False, enableGRU=False, enableS2S=False):
        PRA.__init__(self, nb_words, max_len, embedding_matrix, dim, optimizer, keep_rate_word_dropout, mode, enableWasserstein, enableGRU, enableS2S)
        

    def build(self):
        
        pair_encoder = self.build_main_encoder()
        rec_encoder = self.build_main_encoder()
        
        self.vae, ae_encoders = self.build_ae()
        dssm = self.build_pairwise_model()
        
        self.gs_discriminator = self.build_gs_discriminator()
        self.pr_discriminator = self.build_gs_discriminator()
        
        inputs = [Input((self.max_len,)) for i in range(3)]
        dec_inputs = [Input((self.max_len,)) for i in range(3)]
        
        pair_latents = [pair_encoder(i) for i in inputs]
        rec_latents = [rec_encoder(i) for i in inputs]
        
        main_latents = [merge([i, j], mode="mul") for i, j in zip(pair_latents, rec_latents)]
        

        ae_enc_latents = [ae_encoders(i) for i in main_latents]
        gs_latents = [normal_latent_sampling((self.latent_dim,))(i) for i in inputs]
        
        rec_outputs = [self.vae([i, j]) for i, j in zip(main_latents, dec_inputs)]
        pair_output = dssm(main_latents)
        
        gs_fake_outputs = [self.gs_discriminator(i) for i in ae_enc_latents]
        gs_real_outputs = [self.gs_discriminator(i) for i in gs_latents]

        pr_fake_outputs = [self.pr_discriminator(i) for i in rec_latents]
        pr_real_outputs = [self.pr_discriminator(i) for i in pair_latents]
        
        outputs = rec_outputs + [pair_output] + gs_fake_outputs + gs_real_outputs + pr_fake_outputs + pr_real_outputs
        
        losses = ['sparse_categorical_crossentropy'] * 3 + ['categorical_crossentropy'] + ['binary_crossentropy'] * 12

        outputs_name = ["q_pred", "pd_pred", "nd_pred", 
                        "pair", "q_yfake", "pd_yfake", 
                        "nd_yfake", "q_yreal", "pd_yreal", 
                        "nd_yreal", "pr_q_fake", "pr_pd_fake", 
                        "pr_nd_fake", "pr_q_real", "pr_pd_real", "pr_nd_real"]

        self.encoder = Model(inputs[0], main_latents[0])

        combine_models = Model(inputs+dec_inputs, fix_names(outputs, outputs_name))

        # build adversarial model
        generative_params = pair_encoder.trainable_weights + rec_encoder.trainable_weights + self.vae.trainable_weights + dssm.trainable_weights

        gs_discriminative_params = self.gs_discriminator.trainable_weights
        pr_discriminative_params = self.pr_discriminator.trainable_weights

        self.model = AdversarialModel(base_model=combine_models,
                                 player_params=[generative_params, gs_discriminative_params + pr_discriminative_params],
                                 player_names=["generator", "discriminator"])

        

        gen_weights = [{"loss_weights": {"pr_q_fake": 1e-3, "pr_pd_fake": 1e-3, "pr_nd_fake": 1e-3, "pr_nd_real": -1e-3, "pr_pd_real": -1e-3, "pr_q_real": -1e-3, "q_yfake": 1e-3, "pd_yfake": 1e-3, "nd_yfake": 1e-3, "q_yreal": -1e-3, "pd_yreal": -1e-3, "nd_yreal": -1e-3, "q_pred": 1e-1, "pd_pred": 1e-1, "nd_pred": 1e-1, "pair": 1}}]
        dis_weights = [{"loss_weights": {"pr_q_fake": -1e-3, "pr_pd_fake": -1e-3, "pr_nd_fake": -1e-3, "pr_nd_real": 1e-3, "pr_pd_real": 1e-3, "pr_q_real": 1e-3, "q_yfake": -1e-3, "pd_yfake": -1e-3, "nd_yfake": -1e-3, "q_yreal": 1e-3, "pd_yreal": 1e-3, "nd_yreal": 1e-3, "q_pred": 1e-1, "pd_pred": 1e-1, "nd_pred": 1e-1, "pair": 1}}]
        
        losses = {"q_yfake": self.dis_loss, "q_yreal": self.dis_loss,
                                    "pd_yfake": self.dis_loss, "pd_yreal": self.dis_loss,
                                    "nd_yfake": self.dis_loss, "nd_yreal": self.dis_loss,
                                    "q_pred": "sparse_categorical_crossentropy",
                                    "pd_pred": "sparse_categorical_crossentropy",
                                    "nd_pred": "sparse_categorical_crossentropy",
                                    "pr_q_fake": self.dis_loss,
                                    "pr_pd_fake": self.dis_loss,
                                    "pr_nd_fake": self.dis_loss,
                                    "pr_q_real": self.dis_loss,
                                    "pr_pd_real": self.dis_loss,
                                    "pr_nd_real": self.dis_loss,
                                    "pair": "categorical_crossentropy"}
        
        
        self.model.adversarial_compile(adversarial_optimizer=self.adversarial_optimizer,
                              player_optimizers=[self.optimizer, self.optimizer],
                              loss=losses,
                              player_compile_kwargs=gen_weights+dis_weights)

    def build_main_encoder(self):

        inputs = Input(shape=(self.max_len,))
        
        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)
        
        encoder_lstm = GRU(self.hidden_dim, return_state=True)

        outputs = Dense(self.latent_dim)(encoder_lstm(encoder_embedding(inputs))[-1])
        
        return Model(inputs, outputs)
    
    
    def build_ae(self):
        
        inputs = Input((self.latent_dim,))
        dec_inputs = Input(shape=(self.max_len,))

        
        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)
        
        state_z = Lambda(self.sampling, name="kl")([self.mean(inputs), self.var(inputs)])

        self.latent2hidden = Dense(self.hidden_dim)
        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True)
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")
        self.decoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="dec_embedding",
                                        mask_zero=True,
                                        trainable=True)

        latents = RepeatVector(self.max_len)(self.latent2hidden(state_z))
        dec_latents = self.decoder_embedding(dec_inputs)
        concat = merge([dec_latents, latents], mode="concat")
    
        rec_outputs = self.decoder_dense(self.decoder_lstm(concat))
        
        return Model([inputs, dec_inputs], rec_outputs), Model(inputs, state_z)

    
    def name(self):
        return "pra2_aae_m%d" % self.mode if not self.enableWasserstein else "pra2_wae_m%d" % self.mode