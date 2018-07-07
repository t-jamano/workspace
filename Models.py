import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras import backend as K
K.set_session(session)


from Utils import *
from random import shuffle
from keras.layers import Bidirectional, Dense, Embedding, Concatenate, Flatten, Reshape, Input, Lambda, LSTM, merge, GlobalAveragePooling1D, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras import objectives
from keras import backend as K
from keras.models import Model, load_model, Sequential
from keras.engine import Layer
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.layers.advanced_activations import LeakyReLU
# from keras_tqdm import TQDMNotebookCallback
# from keras_tqdm import TQDMCallback
from keras_adversarial.legacy import l1l2
from keras_adversarial import AdversarialModel, fix_names, n_choice
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras.layers import LeakyReLU, Activation
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras import initializers
from gensim.models import KeyedVectors
from gensim.models import KeyedVectors
import sentencepiece as spm
from copy import copy



class KCompetitive(Layer):
    '''Applies K-Competitive layer.

    # Arguments
    '''
    def __init__(self, topk, ctype, **kwargs):
        self.topk = topk
        self.ctype = ctype
        self.uses_learning_phase = True
        self.supports_masking = True
        super(KCompetitive, self).__init__(**kwargs)

    def call(self, x):
        if self.ctype == 'ksparse':
            return K.in_train_phase(self.kSparse(x, self.topk), x)
        elif self.ctype == 'kcomp':
            return K.in_train_phase(self.k_comp_tanh(x, self.topk), x)
        else:
            warnings.warn("Unknown ctype, using no competition.")
            return x

    def get_config(self):
        config = {'topk': self.topk, 'ctype': self.ctype}
        base_config = super(KCompetitive, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def k_comp_tanh(self, x, topk, factor=6.26):
        dim = int(x.get_shape()[1])
        # batch_size = tf.to_float(tf.shape(x)[0])
        if topk > dim:
            warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
            topk = dim

        P = (x + tf.abs(x)) / 2
        N = (x - tf.abs(x)) / 2

        values, indices = tf.nn.top_k(P, int(topk / 2)) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]
        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, int(topk / 2)])  # will be [[0, 0], [1, 1]]
        full_indices = tf.stack([my_range_repeated, indices], axis=2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)


        values2, indices2 = tf.nn.top_k(-N, topk - int(topk / 2))
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices2)[0]), 1)
        my_range_repeated = tf.tile(my_range, [1, topk - int(topk / 2)])
        full_indices2 = tf.stack([my_range_repeated, indices2], axis=2)
        full_indices2 = tf.reshape(full_indices2, [-1, 2])
        N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(values2, [-1]), default_value=0., validate_indices=False)


        P_tmp = factor * tf.reduce_sum(P - P_reset, 1, keep_dims=True) # 6.26
        N_tmp = factor * tf.reduce_sum(-N - N_reset, 1, keep_dims=True)
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, P_tmp), [-1]), default_value=0., validate_indices=False)
        N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, N_tmp), [-1]), default_value=0., validate_indices=False)

        res = P_reset - N_reset

        return res

    def kSparse(self, x, topk):
        dim = int(x.get_shape()[1])
        if topk > dim:
            warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
            topk = dim

        k = dim - topk
        values, indices = tf.nn.top_k(-x, k) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]

        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]

        full_indices = tf.stack([my_range_repeated, indices], axis=2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])

        to_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)

        res = tf.add(x, to_reset)

        return res

class Dense_tied(Dense):
    """
    A fully connected layer with tied weights.
    """
    def __init__(self, units,
                 activation=None, use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 tied_to=None, **kwargs):
        self.tied_to = tied_to

        super(Dense_tied, self).__init__(units=units,
                 activation=activation, use_bias=use_bias,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                 activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                 **kwargs)

    def build(self, input_shape):
        super(Dense_tied, self).build(input_shape)  # be sure you call this somewhere!
        if self.kernel in self.trainable_weights:
            self.trainable_weights.remove(self.kernel)


    def call(self, x, mask=None):
        # Use tied weights
        self.kernel = K.transpose(self.tied_to.kernel)
        output = K.dot(x, self.kernel)
        if self.use_bias:
            output += self.bias
        return self.activation(output)

class AAE(object):
    def __init__(self, nb_words, max_len, emb, dim, comp_topk=None, ctype=None, epsilon_std=1.0, save_model='best_model'):
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.save_model = save_model

        self.nb_words = nb_words
        self.max_len = max_len
        self.emb = emb

        self.decoder = self.create_decoder()
        self.encoder = self.create_encoder()
        autoencoder = Model(self.encoder.inputs, self.decoder(self.encoder(self.encoder.inputs)))
        discriminator = self.create_discriminator()

        x = self.encoder.inputs[0]
        z = self.encoder(x)
        xpred = self.decoder(z)
        zreal = normal_latent_sampling((self.dim[1],))(x)
        yreal = discriminator(zreal)
        yfake = discriminator(z)
        aae = Model(x, fix_names([xpred, yfake, yreal], ["xpred", "yfake", "yreal"]))
        
        generative_params = self.decoder.trainable_weights + self.encoder.trainable_weights
        
        self.model = AdversarialModel(base_model=aae,
                                 player_params=[generative_params, discriminator.trainable_weights],
                                 player_names=["decoder", "discriminator"])
        self.model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                                  player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                                  loss={"yfake": "binary_crossentropy", "yreal": "binary_crossentropy",
                                    "xpred": "mean_squared_error"},
                              player_compile_kwargs=[{"loss_weights": {"yfake": 1e-2, "yreal": 1e-2, "xpred": 1}}] * 2)

    def create_discriminator(self):

        z = Input((self.dim[1],))
        h = z
        h = Dense(self.dim[0], name="discriminator_h1")(h)
        h = LeakyReLU(0.2)(h)
        h = Dense(self.dim[1], name="discriminator_h2")(h)
        h = LeakyReLU(0.2)(h)
        y = Dense(1, name="discriminator_y", activation="sigmoid")(h)
        return Model(z, y)
    
    def create_decoder(self):
        
        z = Input(shape=(self.dim[1],))

        act = 'tanh'
        decoder_h = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        decoder_mean = Dense(self.nb_words, activation='softmax')

        h_decoded = decoder_h(z)
        h_decoded = RepeatVector(self.max_len)(h_decoded)
        h_decoded = Bidirectional(LSTM(self.dim[0], return_sequences=True, name='dec_lstm_1'))(h_decoded)
        x_decoded_mean = TimeDistributed(decoder_mean, name='decoded_mean')(h_decoded)


        return Model(z, x_decoded_mean, name="decoder")
    
    def create_encoder(self):
        
        act = 'tanh'
        x = Input(shape=(self.max_len,))
        embed_layer = self.emb
        bilstm = Bidirectional(LSTM(self.dim[0], name='lstm_1'))
        hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        
        h1 = embed_layer(x)
        h1 = bilstm(h1)
        h1 = hidden_layer1(h1)

        self.z_mean = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)
        self.z_log_var = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)

        if self.comp_topk != None:
            self.z_mean = KCompetitive(self.comp_topk, self.ctype)(self.z_mean)

        z = Lambda(self.sampling, output_shape=(self.dim[1],))([self.z_mean, self.z_log_var])
        
        return Model(x, z, name="encoder")



    def vae_loss(self, x, x_decoded_mean):
        # xent_loss =  self.max_len * K.sum(K.binary_crossentropy(x_decoded_mean, x), axis=-1)
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.max_len * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)

        return xent_loss + kl_loss



    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon

    def initModel(self, sp, bpe_dict):
        self.sp = sp
        self.bpe_dict = bpe_dict

    def batch_generator(self, reader, train_data, batch_size):
        while True:
            for df in reader:
                
                x = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                x_one_hot = to_categorical(x, self.nb_words)
                x_one_hot = x_one_hot.reshape(batch_size, self.max_len, self.nb_words)
                valid = np.ones(batch_size)
                fake = np.zeros(batch_size)
                yield x, [x_one_hot, valid, fake, x_one_hot, fake, valid]


class VarAutoEncoderQD3(object):
    """VarAutoEncoder for topic modeling.

        Parameters
        ----------
        dim : dimensionality of encoding space.

        nb_epoch :

        """
    def __init__(self, nb_words, max_len, emb, dim, comp_topk=None, ctype=None, epsilon_std=1.0, save_model='best_model', enableCross=False, enableMemory=False, enableGAN=False, alpha=0.5, optimizer=Adadelta(lr=2.)):
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.save_model = save_model

        self.nb_words = nb_words
        self.max_len = max_len
        self.enableCross = enableCross
        self.enableMemory = enableMemory
        self.enableGAN = enableGAN
        self.alpha = alpha

        act = 'tanh'
        
        q_input_layer = Input(shape=(self.max_len,))
        d_input_layer = Input(shape=(self.max_len,))

        
        embed_layer = emb
        bilstm = Bidirectional(LSTM(self.dim[0], name='lstm_1'))
        hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)

        cos_embed_layer = emb
        cos_bilstm = Bidirectional(LSTM(self.dim[0]))
        cos_hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        

        q = embed_layer(q_input_layer)
        q = bilstm(q)
        q = hidden_layer1(q)
        
        d = embed_layer(d_input_layer)
        d = bilstm(d)
        d = hidden_layer1(d)

        cos_q = cos_embed_layer(q_input_layer)
        cos_q = cos_bilstm(cos_q)
        cos_q = cos_hidden_layer1(cos_q)
        
        cos_d = cos_embed_layer(d_input_layer)
        cos_d = cos_bilstm(cos_d)
        cos_d = cos_hidden_layer1(cos_d)






        self.hidden_q = q
        self.hidden_d = d
        
        dense_mean = Dense(self.dim[1], kernel_initializer='glorot_normal')
        dense_var = Dense(self.dim[1], kernel_initializer='glorot_normal')

        cos_dense_mean = Dense(self.dim[1], kernel_initializer='glorot_normal')
        cos_dense_var = Dense(self.dim[1], kernel_initializer='glorot_normal')

        self.q_mean = dense_mean(q)
        self.q_log_var = dense_var(q)
        self.d_mean = dense_mean(d)
        self.d_log_var = dense_var(d)

        # cosine part 
        self.cos_q_mean = cos_dense_mean(cos_q)
        self.cos_q_log_var = cos_dense_var(cos_q)
        self.cos_d_mean = cos_dense_mean(cos_d)
        self.cos_d_log_var = cos_dense_var(cos_d)

        # add private
        # self.q_mean = merge([self.q_mean, self.cos_q_mean], mode="sum")
        # self.q_log_var = merge([self.q_log_var, self.cos_q_log_var], mode="sum")

        # self.d_mean = merge([self.d_mean, self.cos_d_mean], mode="sum")
        # self.d_log_var = merge([self.d_log_var, self.cos_d_log_var], mode="sum")

        if self.comp_topk != None:
            kc_layer = KCompetitive(self.comp_topk, self.ctype)
            self.q_mean = kc_layer(self.q_mean)
            self.d_mean = kc_layer(self.d_mean)

            kc_layer = KCompetitive(self.comp_topk, self.ctype)
            self.cos_q_mean = kc_layer(self.cos_q_mean)
            self.cos_d_mean = kc_layer(self.cos_d_mean)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        encoded_q = Lambda(self.sampling, output_shape=(self.dim[1],))([self.q_mean, self.q_log_var])
        encoded_d = Lambda(self.sampling, output_shape=(self.dim[1],))([self.d_mean, self.d_log_var])


        cos_encoded_q = Lambda(self.sampling, output_shape=(self.dim[1],))([self.cos_q_mean, self.cos_q_log_var])
        cos_encoded_d = Lambda(self.sampling, output_shape=(self.dim[1],))([self.cos_d_mean, self.cos_d_log_var])
        # add private

        encoded_q = merge([encoded_q, cos_encoded_q], mode="sum")
        encoded_d = merge([encoded_d, cos_encoded_d], mode="sum")



        # old
        # cos_qd = Flatten(name="cos_qd")(merge([encoded_q, encoded_d], mode="cos",))
        # 
        cos_qd = Flatten()(merge([cos_encoded_q, cos_encoded_d], mode="cos",))
        cos_qd = Dense(1, activation="sigmoid", name="cos_qd")(cos_qd)



        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        # decoder_mean = Dense_tied(self.nb_words, activation='softmax', tied_to=hidden_layer1)
        decoder_mean = Dense(self.nb_words, activation='softmax')
        
        decoder_bilstm = Bidirectional(LSTM(self.dim[0], return_sequences=True, name='dec_lstm_1'))

        self.hidden_decoded_q = decoder_h(encoded_q)
        self.hidden_decoded_d = decoder_h(encoded_d)


        q_decoded = decoder_h(encoded_q)
        q_decoded = RepeatVector(self.max_len)(q_decoded)
        q_decoded = decoder_bilstm(q_decoded)
        
        d_decoded = decoder_h(encoded_d)
        d_decoded = RepeatVector(self.max_len)(d_decoded)
        d_decoded = decoder_bilstm(d_decoded)

        
        
        q_decoded_mean = TimeDistributed(decoder_mean, name='recon_q')(q_decoded)
        d_decoded_mean = TimeDistributed(decoder_mean, name='recon_d')(d_decoded)


        cos_m_q = Flatten()(merge([self.hidden_q, self.hidden_decoded_q], mode="cos", name="cos_q"))
        cos_m_d = Flatten()(merge([self.hidden_d, self.hidden_decoded_d], mode="cos", name="cos_d"))


        if self.enableCross:
            self.model = Model(outputs=[q_decoded_mean, d_decoded_mean, q_decoded_mean, d_decoded_mean, cos_qd], inputs=[q_input_layer, d_input_layer])
        
        elif self.enableMemory:
            self.model = Model(outputs=[q_decoded_mean, d_decoded_mean, cos_m_q, cos_m_d, cos_qd], inputs=[q_input_layer, d_input_layer])
        
        elif self.enableGAN:
            self.discriminator = self.build_discriminator()
            validity_q = self.discriminator(encoded_q)
            validity_d = self.discriminator(encoded_d)
            self.model = Model(outputs=[q_decoded_mean, d_decoded_mean, validity_q, validity_d, cos_qd], inputs=[q_input_layer, d_input_layer])
    
        else:
            self.model = Model(outputs=[q_decoded_mean, d_decoded_mean, cos_qd], inputs=[q_input_layer, d_input_layer])


        self.encoder = Model(outputs=self.q_mean, inputs=q_input_layer)


        # optimizer = Adam()
        # optimizer = Adadelta(lr=2.)
        if self.enableCross:
            self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, self.vae_loss_cross, self.vae_loss_cross, 'binary_crossentropy'], loss_weights=[1, 1, 1, 1, self.pred_weight])
        
        elif self.enableMemory:
            self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'])
        
        elif self.enableGAN:
            # , loss_weights=[0.999, 0.999, 0.001, 0.001, 0.999]
            self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
            self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'])

        else:
            self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, 'binary_crossentropy'])

            # self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, 'binary_crossentropy'], loss_weights=[1,1, self.pred_weight])

        

    def vae_loss_q(self, x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.max_len * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.q_log_var - K.square(self.q_mean) - K.exp(self.q_log_var), axis=-1)
        return xent_loss + kl_loss
    
    def vae_loss_d(self, x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.max_len * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.d_log_var - K.square(self.d_mean) - K.exp(self.d_log_var), axis=-1)
        return xent_loss + kl_loss

    def vae_loss_cross(self, x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.max_len * objectives.binary_crossentropy(x, x_decoded_mean)
        return xent_loss

    def build_discriminator(self):

        dis = Sequential()

        dis.add(Dense(self.dim[0], input_dim=self.dim[1]))
        dis.add(LeakyReLU(alpha=0.2))
        dis.add(Dense(self.dim[1]))
        dis.add(LeakyReLU(alpha=0.2))
        dis.add(Dense(1, activation="sigmoid"))

        encoded_repr = Input(shape=(self.dim[1], ))
        validity = dis(encoded_repr)

        return Model(encoded_repr, validity)


    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon

    def initModel(self, sp, bpe_dict):
        self.sp = sp
        self.bpe_dict = bpe_dict

    def batch_generator(self, reader, train_data, batch_size):
        while True:
            for df in reader:

                q = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                d = parse_texts_bpe(df.d.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                
                q_one_hot = to_categorical(q, self.nb_words)
                q_one_hot = q_one_hot.reshape(batch_size, self.max_len, self.nb_words)
                
                d_one_hot = to_categorical(d, self.nb_words)
                d_one_hot = d_one_hot.reshape(batch_size, self.max_len, self.nb_words)

                if self.enableCross:
                    yield [q,d], [q_one_hot, d_one_hot, d_one_hot, q_one_hot, df.label.values]
                elif self.enableMemory or self.enableGAN:
                    one = np.ones(batch_size)
                    yield [q,d], [q_one_hot, d_one_hot, one, one, df.label.values]
                else:
                    yield [q,d], [q_one_hot, d_one_hot, df.label.values]


# BPE version
class VarAutoEncoderQD2(object):
    """VarAutoEncoder for topic modeling.

        Parameters
        ----------
        dim : dimensionality of encoding space.

        nb_epoch :

        """
    def __init__(self, nb_words, max_len, emb, dim, comp_topk=None, ctype=None, epsilon_std=1.0, save_model='best_model', enableCross=False, enableMemory=False, enableGAN=False, alpha=0.5, optimizer=Adadelta(lr=2.)):
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.save_model = save_model

        self.nb_words = nb_words
        self.max_len = max_len
        self.enableCross = enableCross
        self.enableMemory = enableMemory
        self.enableGAN = enableGAN
        self.alpha = alpha

        act = 'tanh'
        
        q_input_layer = Input(shape=(self.max_len,))
        d_input_layer = Input(shape=(self.max_len,))

        
        embed_layer = emb
        bilstm = Bidirectional(LSTM(self.dim[0], name='lstm_1'))
        hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)

        d_bilstm = Bidirectional(LSTM(self.dim[0], name='lstm_1'))
        d_hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        

        q = embed_layer(q_input_layer)
        q = bilstm(q)
        q = hidden_layer1(q)
        
        d = embed_layer(d_input_layer)
        d = d_bilstm(d)
        d = d_hidden_layer1(d)

        self.hidden_q = q
        self.hidden_d = d
        
        dense_mean = Dense(self.dim[1], kernel_initializer='glorot_normal')
        dense_var = Dense(self.dim[1], kernel_initializer='glorot_normal')

        self.q_mean = dense_mean(q)
        self.q_log_var = dense_var(q)
        
        self.d_mean = dense_mean(d)
        self.d_log_var = dense_var(d)

        if self.comp_topk != None:
            kc_layer = KCompetitive(self.comp_topk, self.ctype)
            self.q_mean = kc_layer(self.q_mean)
            self.d_mean = kc_layer(self.d_mean)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        encoded_q = Lambda(self.sampling, output_shape=(self.dim[1],))([self.q_mean, self.q_log_var])
        encoded_d = Lambda(self.sampling, output_shape=(self.dim[1],))([self.d_mean, self.d_log_var])
        
        # old
        # cos_qd = Flatten(name="cos_qd")(merge([encoded_q, encoded_d], mode="cos",))
        # 
        cos_qd = Flatten()(merge([self.q_mean, self.d_mean], mode="cos",))
        cos_qd = Dense(1, activation="sigmoid", name="cos_qd")(cos_qd)



        decoder_h = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        decoder_mean = Dense(self.nb_words, activation='softmax')
        decoder_bilstm = Bidirectional(LSTM(self.dim[0], return_sequences=True, name='dec_lstm_1'))

        d_decoder_h = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        d_decoder_mean = Dense(self.nb_words, activation='softmax')
        d_decoder_bilstm = Bidirectional(LSTM(self.dim[0], return_sequences=True, name='dec_lstm_1'))

        self.hidden_decoded_q = decoder_h(encoded_q)
        self.hidden_decoded_d = d_decoder_h(encoded_d)


        q_decoded = decoder_h(encoded_q)
        q_decoded = RepeatVector(self.max_len)(q_decoded)
        q_decoded = decoder_bilstm(q_decoded)
        
        d_decoded = d_decoder_h(encoded_d)
        d_decoded = RepeatVector(self.max_len)(d_decoded)
        d_decoded = d_decoder_bilstm(d_decoded)

        
        
        q_decoded_mean = TimeDistributed(decoder_mean, name='recon_q')(q_decoded)
        d_decoded_mean = TimeDistributed(decoder_mean, name='recon_d')(d_decoded)


        cos_m_q = Flatten()(merge([self.hidden_q, self.hidden_decoded_q], mode="cos", name="cos_q"))
        cos_m_d = Flatten()(merge([self.hidden_d, self.hidden_decoded_d], mode="cos", name="cos_d"))


        if self.enableCross:
            self.model = Model(outputs=[q_decoded_mean, d_decoded_mean, q_decoded_mean, d_decoded_mean, cos_qd], inputs=[q_input_layer, d_input_layer])
        
        elif self.enableMemory:
            self.model = Model(outputs=[q_decoded_mean, d_decoded_mean, cos_m_q, cos_m_d, cos_qd], inputs=[q_input_layer, d_input_layer])
        
        elif self.enableGAN:
            self.discriminator = self.build_discriminator()
            validity_q = self.discriminator(encoded_q)
            validity_d = self.discriminator(encoded_d)
            self.model = Model(outputs=[q_decoded_mean, d_decoded_mean, validity_q, validity_d, cos_qd], inputs=[q_input_layer, d_input_layer])
    
        else:
            self.model = Model(outputs=[q_decoded_mean, d_decoded_mean, cos_qd], inputs=[q_input_layer, d_input_layer])


        self.encoder = Model(outputs=self.q_mean, inputs=q_input_layer)






        # optimizer = Adam()
        

        if self.enableCross:
            self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, self.vae_loss_cross, self.vae_loss_cross, 'binary_crossentropy'])
        
        elif self.enableMemory:
            self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'])
        
        elif self.enableGAN:
            # , loss_weights=[0.999, 0.999, 0.001, 0.001, 0.999]
            self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
            self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'])

        else:
            self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, 'binary_crossentropy'])

        

    def vae_loss_q(self, x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.max_len * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.q_log_var - K.square(self.q_mean) - K.exp(self.q_log_var), axis=-1)
        return xent_loss + kl_loss
    
    def vae_loss_d(self, x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.max_len * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.d_log_var - K.square(self.d_mean) - K.exp(self.d_log_var), axis=-1)
        return xent_loss + kl_loss

    def vae_loss_cross(self, x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.max_len * objectives.binary_crossentropy(x, x_decoded_mean)
        return xent_loss

    def build_discriminator(self):

        dis = Sequential()

        dis.add(Dense(self.dim[0], input_dim=self.dim[1]))
        dis.add(LeakyReLU(alpha=0.2))
        dis.add(Dense(self.dim[1]))
        dis.add(LeakyReLU(alpha=0.2))
        dis.add(Dense(1, activation="sigmoid"))

        encoded_repr = Input(shape=(self.dim[1], ))
        validity = dis(encoded_repr)

        return Model(encoded_repr, validity)


    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon

    def initModel(self, sp, bpe_dict):
        self.sp = sp
        self.bpe_dict = bpe_dict

    def batch_generator(self, reader, train_data, batch_size):
        while True:
            for df in reader:

                q = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                d = parse_texts_bpe(df.d.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                
                q_one_hot = to_categorical(q, self.nb_words)
                q_one_hot = q_one_hot.reshape(batch_size, self.max_len, self.nb_words)
                
                d_one_hot = to_categorical(d, self.nb_words)
                d_one_hot = d_one_hot.reshape(batch_size, self.max_len, self.nb_words)

                if self.enableCross:
                    yield [q,d], [q_one_hot, d_one_hot, d_one_hot, q_one_hot, df.label.values]
                elif self.enableMemory or self.enableGAN:
                    one = np.ones(batch_size)
                    yield [q,d], [q_one_hot, d_one_hot, one, one, df.label.values]
                else:
                    yield [q,d], [q_one_hot, d_one_hot, df.label.values]

    def batch_GAN_generator(self, reader, train_data, batch_size, graph):
        while True:
            for df in reader:

                q = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                d = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.max_len, True)

                with graph.as_default():
                    # shuffle(x)
                    q_fake = self.encoder.predict(q)
                    # q_fake = np.random.normal(size=(batch_size, self.dim[1]))
                    q_real = np.random.normal(size=(batch_size, self.dim[1]))
                    d_fake = self.encoder.predict(d)
                    # d_fake = np.random.normal(size=(batch_size, self.dim[1]))
                    d_real = np.random.normal(size=(batch_size, self.dim[1]))
                    
                    ones = np.ones(batch_size)
                    zeros = np.zeros(batch_size)


                    x_ = np.concatenate([q_real, q_fake, d_real, d_fake])
                    y_ = np.concatenate([ones, zeros, ones, zeros])

                    idx = np.random.randint(batch_size * 4, size=batch_size * 4)

                yield x_[idx], y_[idx]



# BPE version
class VarAutoEncoderQD(object):
    """VarAutoEncoder for topic modeling.

        Parameters
        ----------
        dim : dimensionality of encoding space.

        nb_epoch :

        """
    def __init__(self, nb_words, max_len, emb, dim, comp_topk=None, ctype=None, epsilon_std=1.0, save_model='best_model', enableCross=False, enableMemory=False, enableGAN=False, alpha=0.5, optimizer=Adadelta(lr=2.)):
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.save_model = save_model

        self.nb_words = nb_words
        self.max_len = max_len
        self.enableCross = enableCross
        self.enableMemory = enableMemory
        self.enableGAN = enableGAN
        self.alpha = alpha

        self.kl_weight = 0.0
        self.pred_weight = 0.0

        act = 'tanh'
        
        q_input_layer = Input(shape=(self.max_len,))
        d_input_layer = Input(shape=(self.max_len,))

        
        embed_layer = emb
        bilstm = Bidirectional(LSTM(self.dim[0], name='lstm_1'))


        hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        

        q = embed_layer(q_input_layer)
        q = bilstm(q)
        q = hidden_layer1(q)
        
        d = embed_layer(d_input_layer)
        d = bilstm(d)
        d = hidden_layer1(d)

        self.hidden_q = q
        self.hidden_d = d
        
        dense_mean = Dense(self.dim[1], kernel_initializer='glorot_normal')
        dense_var = Dense(self.dim[1], kernel_initializer='glorot_normal')

        self.q_mean = dense_mean(q)
        self.q_log_var = dense_var(q)
        
        self.d_mean = dense_mean(d)
        self.d_log_var = dense_var(d)

        if self.comp_topk != None:
            kc_layer = KCompetitive(self.comp_topk, self.ctype)
            self.q_mean = kc_layer(self.q_mean)
            self.d_mean = kc_layer(self.d_mean)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        encoded_q = Lambda(self.sampling, output_shape=(self.dim[1],))([self.q_mean, self.q_log_var])
        encoded_d = Lambda(self.sampling, output_shape=(self.dim[1],))([self.d_mean, self.d_log_var])
        
        # old
        # cos_qd = Flatten(name="cos_qd")(merge([encoded_q, encoded_d], mode="cos",))
        # 
        cos_qd = Flatten()(merge([self.q_mean, self.d_mean], mode="cos",))
        cos_qd = Dense(1, activation="sigmoid", name="cos_qd")(cos_qd)



        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        # decoder_mean = Dense_tied(self.nb_words, activation='softmax', tied_to=hidden_layer1)
        decoder_mean = Dense(self.nb_words, activation='softmax')
        
        decoder_bilstm = Bidirectional(LSTM(self.dim[0], return_sequences=True, name='dec_lstm_1'))

        self.hidden_decoded_q = decoder_h(encoded_q)
        self.hidden_decoded_d = decoder_h(encoded_d)


        q_decoded = decoder_h(encoded_q)
        q_decoded = RepeatVector(self.max_len)(q_decoded)
        q_decoded = decoder_bilstm(q_decoded)
        
        d_decoded = decoder_h(encoded_d)
        d_decoded = RepeatVector(self.max_len)(d_decoded)
        d_decoded = decoder_bilstm(d_decoded)

        
        
        q_decoded_mean = TimeDistributed(decoder_mean, name='recon_q')(q_decoded)
        d_decoded_mean = TimeDistributed(decoder_mean, name='recon_d')(d_decoded)


        cos_m_q = Flatten()(merge([self.hidden_q, self.hidden_decoded_q], mode="cos", name="cos_q"))
        cos_m_d = Flatten()(merge([self.hidden_d, self.hidden_decoded_d], mode="cos", name="cos_d"))


        if self.enableCross:
            self.model = Model(outputs=[q_decoded_mean, d_decoded_mean, q_decoded_mean, d_decoded_mean, cos_qd], inputs=[q_input_layer, d_input_layer])
        
        elif self.enableMemory:
            self.model = Model(outputs=[q_decoded_mean, d_decoded_mean, cos_m_q, cos_m_d, cos_qd], inputs=[q_input_layer, d_input_layer])
        
        elif self.enableGAN:
            self.discriminator = self.build_discriminator()
            validity_q = self.discriminator(encoded_q)
            validity_d = self.discriminator(encoded_d)
            self.model = Model(outputs=[q_decoded_mean, d_decoded_mean, validity_q, validity_d, cos_qd], inputs=[q_input_layer, d_input_layer])
    
        else:
            self.model = Model(outputs=[q_decoded_mean, d_decoded_mean, cos_qd], inputs=[q_input_layer, d_input_layer])


        self.encoder = Model(outputs=self.q_mean, inputs=q_input_layer)


        # optimizer = Adam()
        # optimizer = Adadelta(lr=2.)
        if self.enableCross:
            self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, self.vae_loss_cross, self.vae_loss_cross, 'binary_crossentropy'], loss_weights=[1, 1, 1, 1, self.pred_weight])
        
        elif self.enableMemory:
            self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'])
        
        elif self.enableGAN:
            # , loss_weights=[0.999, 0.999, 0.001, 0.001, 0.999]
            self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
            self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'])

        else:
            self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, 'binary_crossentropy'], loss_weights=[self.alpha, self.alpha, 1])

            # self.model.compile(optimizer=optimizer, loss=[self.vae_loss_q, self.vae_loss_d, 'binary_crossentropy'], loss_weights=[1,1, self.pred_weight])

        

    def vae_loss_q(self, x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.max_len * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.q_log_var - K.square(self.q_mean) - K.exp(self.q_log_var), axis=-1)
        # return xent_loss + (self.kl_weight * kl_loss)
        return xent_loss + kl_loss

    def vae_loss_d(self, x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.max_len * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.d_log_var - K.square(self.d_mean) - K.exp(self.d_log_var), axis=-1)
        return xent_loss + kl_loss

    def vae_loss_cross(self, x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.max_len * objectives.binary_crossentropy(x, x_decoded_mean)
        return xent_loss

    def build_discriminator(self):

        dis = Sequential()

        dis.add(Dense(self.dim[0], input_dim=self.dim[1]))
        dis.add(LeakyReLU(alpha=0.2))
        dis.add(Dense(self.dim[1]))
        dis.add(LeakyReLU(alpha=0.2))
        dis.add(Dense(1, activation="sigmoid"))

        encoded_repr = Input(shape=(self.dim[1], ))
        validity = dis(encoded_repr)

        return Model(encoded_repr, validity)


    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon

    def initModel(self, sp, bpe_dict):
        self.sp = sp
        self.bpe_dict = bpe_dict

    def batch_generator(self, reader, train_data, batch_size):
        while True:
            for df in reader:

                q = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                d = parse_texts_bpe(df.d.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                
                q_one_hot = to_categorical(q, self.nb_words)
                q_one_hot = q_one_hot.reshape(batch_size, self.max_len, self.nb_words)
                
                d_one_hot = to_categorical(d, self.nb_words)
                d_one_hot = d_one_hot.reshape(batch_size, self.max_len, self.nb_words)

                if self.enableCross:
                    yield [q,d], [q_one_hot, d_one_hot, d_one_hot, q_one_hot, df.label.values]
                elif self.enableMemory or self.enableGAN:
                    one = np.ones(batch_size)
                    yield [q,d], [q_one_hot, d_one_hot, one, one, df.label.values]
                else:
                    yield [q,d], [q_one_hot, d_one_hot, df.label.values]

    def batch_GAN_generator(self, reader, train_data, batch_size, graph):
        while True:
            for df in reader:

                q = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                d = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.max_len, True)

                with graph.as_default():
                    # shuffle(x)
                    q_fake = self.encoder.predict(q)
                    # q_fake = np.random.normal(size=(batch_size, self.dim[1]))
                    q_real = np.random.normal(size=(batch_size, self.dim[1]))
                    d_fake = self.encoder.predict(d)
                    # d_fake = np.random.normal(size=(batch_size, self.dim[1]))
                    d_real = np.random.normal(size=(batch_size, self.dim[1]))
                    
                    ones = np.ones(batch_size)
                    zeros = np.zeros(batch_size)


                    x_ = np.concatenate([q_real, q_fake, d_real, d_fake])
                    y_ = np.concatenate([ones, zeros, ones, zeros])

                    idx = np.random.randint(batch_size * 4, size=batch_size * 4)

                yield x_[idx], y_[idx]

# BPE version
class VarAutoEncoder2(object):
    """VarAutoEncoder for topic modeling.

        Parameters
        ----------
        dim : dimensionality of encoding space.

        nb_epoch :

        """
    def __init__(self, nb_words, max_len, emb, dim, comp_topk=None, ctype=None, epsilon_std=1.0, save_model='best_model', optimizer=Adadelta(lr=2.)):
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.save_model = save_model

        self.nb_words = nb_words
        self.max_len = max_len

        act = 'tanh'
        input_layer = Input(shape=(self.max_len,))
        embed_layer = emb
        bilstm = Bidirectional(LSTM(self.dim[0], name='lstm_1'))


        hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        
        h1 = embed_layer(input_layer)
        h1 = bilstm(h1)
        h1 = hidden_layer1(h1)

        self.z_mean = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)
        self.z_log_var = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)

        if self.comp_topk != None:
            self.z_mean = KCompetitive(self.comp_topk, self.ctype)(self.z_mean)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        encoded = Lambda(self.sampling, output_shape=(self.dim[1],))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        # decoder_mean = Dense_tied(self.nb_words, activation='softmax', tied_to=hidden_layer1)
        decoder_mean = Dense(self.nb_words, activation='softmax')

        h_decoded = decoder_h(encoded)
        h_decoded = RepeatVector(self.max_len)(h_decoded)
        h_decoded = Bidirectional(LSTM(self.dim[0], return_sequences=True, name='dec_lstm_1'))(h_decoded)
        x_decoded_mean = TimeDistributed(decoder_mean, name='decoded_mean')(h_decoded)

        
        self.model = Model(outputs=x_decoded_mean, inputs=input_layer)
        # build a model to project inputs on the latent space
        self.encoder = Model(outputs=self.z_mean, inputs=input_layer)

        # build a digit generator that can sample from the learned distribution
        # decoder_input = Input(shape=(self.dim[1],))
        # _h_decoded = decoder_h(decoder_input)
        # _x_decoded_mean = decoder_mean(_h_decoded)
        # self.decoder = Model(outputs=_x_decoded_mean, inputs=decoder_input)
        self.model.compile(optimizer=optimizer, loss=self.vae_loss)


    def vae_loss(self, x, x_decoded_mean):
        # xent_loss =  self.max_len * K.sum(K.binary_crossentropy(x_decoded_mean, x), axis=-1)
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.max_len * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)

        return xent_loss + kl_loss



    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon

    def initModel(self, sp, bpe_dict):
        self.sp = sp
        self.bpe_dict = bpe_dict

    def batch_generator(self, reader, train_data, batch_size):
        while True:
            for df in reader:
                
                x = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                x_one_hot = to_categorical(x, self.nb_words)
                x_one_hot = x_one_hot.reshape(batch_size, self.max_len, self.nb_words)
                yield x, x_one_hot


class VarAutoEncoder(object):
    """VarAutoEncoder for topic modeling.

        Parameters
        ----------
        dim : dimensionality of encoding space.

        nb_epoch :

        """

    def __init__(self, input_size, dim, comp_topk=None, ctype=None, epsilon_std=1.0, save_model='best_model'):
        self.input_size = input_size
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.save_model = save_model

        self.nb_words = input_size

        act = 'tanh'
        input_layer = Input(shape=(self.input_size,))
        hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        h1 = hidden_layer1(input_layer)

        self.z_mean = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)
        self.z_log_var = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)

        if self.comp_topk != None:
            self.z_mean = KCompetitive(self.comp_topk, self.ctype)(self.z_mean)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        encoded = Lambda(self.sampling, output_shape=(self.dim[1],))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        h_decoded = decoder_h(encoded)
        decoder_mean = Dense_tied(self.input_size, activation='sigmoid', tied_to=hidden_layer1)
        x_decoded_mean = decoder_mean(h_decoded)

        self.model = Model(outputs=x_decoded_mean, inputs=input_layer)
        # build a model to project inputs on the latent space
        self.encoder = Model(outputs=self.z_mean, inputs=input_layer)

        # build a digit generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.dim[1],))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self.decoder = Model(outputs=_x_decoded_mean, inputs=decoder_input)
        optimizer = Adadelta(lr=2.)
        self.model.compile(optimizer=optimizer, loss=self.vae_loss)


    def vae_loss(self, x, x_decoded_mean):
        xent_loss =  K.sum(K.binary_crossentropy(x_decoded_mean, x), axis=-1)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)

        return xent_loss + kl_loss

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon

    def initModel(self, sp, bpe_dict):
        self.sp = sp
        self.bpe_dict = bpe_dict

    def batch_generator(self, reader, train_data, batch_size):
        while True:
            for df in reader:
                
                x = []
                for text in df.q.tolist():
                    x.append([self.bpe_dict[t] if t in self.bpe_dict else self.bpe_dict['<unk>'] for t in self.sp.EncodeAsPieces(text)])
                
                x = np.array(x)
                # No need for paddind, we do bow vector 
                x_one_hot = to_2D_one_hot(x, self.nb_words)
                                
                yield x_one_hot, x_one_hot





class CosineSim():
    def __init__(self, feature_num):
        q_input = Input(shape=(feature_num,))
        d_input = Input(shape=(feature_num,))

        pred = merge([q_input, d_input], mode="cos")
        self.model = Model([q_input, d_input], pred)


class LSTM_Model():
    def __init__(self, hidden_dim=300, latent_dim=128, max_len=10, emb_dim=200, nb_words=50000, emb=None):

        self.max_len = max_len
        self.nb_words = nb_words
        q_input = Input(shape=(max_len,))
        d_input = Input(shape=(max_len,))
        
        emb = Embedding(nb_words, emb_dim, mask_zero=True) if emb == None else emb

        bilstm = Bidirectional(LSTM(hidden_dim, name='lstm_1'))
        dense1 = Dense(hidden_dim, activation = "tanh")

        dense2 = Dense(latent_dim, activation = "tanh")


        self.q_embed = dense2(dense1(bilstm(emb(q_input))))
        self.d_embed = dense2(dense1(bilstm(emb(d_input))))

        concat = Concatenate()([self.q_embed, self.d_embed])

        pred = Dense(1, activation='sigmoid')(concat)

        self.encoder = Model(inputs=q_input, outputs=self.q_embed)

        self.model = Model(inputs=[q_input, d_input], outputs=pred)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def initModel(self, sp, bpe_dict):
        self.sp = sp
        self.bpe_dict = bpe_dict

    def batch_generator(self, reader, train_data, batch_size):
        while True:
            for df in reader:

                q = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                d = parse_texts_bpe(df.d.tolist(), self.sp, self.bpe_dict, self.max_len, True)

                yield [q,d], df.label.values

class BiLSTM():
    def __init__(self, hidden_dim=300, latent_dim=128, q_max_len=10, d_max_len=15, emb_dim=200, nb_words=50000, emb=None, mode="concat"):

        self.q_max_len = q_max_len
        self.d_max_len = d_max_len
        self.nb_words = nb_words
        q_input = Input(shape=(self.q_max_len,))
        d_input = Input(shape=(self.d_max_len,))
        
        q_emb = Embedding(nb_words, emb_dim, mask_zero=True) if emb == None else emb
        q_bilstm = Bidirectional(LSTM(hidden_dim, name='lstm_1'))
        q_dense1 = Dense(hidden_dim, activation = "tanh")
        q_dense2 = Dense(latent_dim, activation = "tanh")

        d_emb = Embedding(nb_words, emb_dim, mask_zero=True) if emb == None else emb
        d_bilstm = Bidirectional(LSTM(hidden_dim, name='lstm_1'))
        d_dense1 = Dense(hidden_dim, activation = "tanh")
        d_dense2 = Dense(latent_dim, activation = "tanh")


        self.q_embed = q_dense2(q_dense1(q_bilstm(q_emb(q_input))))
        self.d_embed = d_dense2(d_dense1(d_bilstm(d_emb(d_input))))

        combine = merge([self.q_embed, self.d_embed], mode=mode)

        pred = Dense(1, activation='sigmoid')(combine)

        self.encoder = Model(inputs=q_input, outputs=self.q_embed)



        self.model = Model(inputs=[q_input, d_input], outputs=pred)
        # self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model.compile(loss='binary_crossentropy', optimizer=Adadelta(lr=2.), metrics=['accuracy'])


    def initModel(self, sp, bpe_dict):
        self.sp = sp
        self.bpe_dict = bpe_dict

    def batch_generator(self, reader, train_data, batch_size):
        while True:
            for df in reader:

                q = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.q_max_len, True)
                d = parse_texts_bpe(df.d.tolist(), self.sp, self.bpe_dict, self.d_max_len, True)

                yield [q,d], df.label.values





class MLP():
    def __init__(self, input_dim):

        que_input = Input(shape=(input_dim,))
        doc_input = Input(shape=(input_dim,))
        
        

        concat = merge([que_input, doc_input], mode="concat")


        pred = Dense(1, activation='sigmoid')(concat)

        self.model = Model(input=[que_input, doc_input], output=pred)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class W2V_MLP():
    def __init__(self, max_len, input_dim):

        que_input = Input(shape=(max_len, input_dim,))
        doc_input = Input(shape=(max_len, input_dim,))
        
        x = GlobalAveragePooling1D()(que_input)
        y = GlobalAveragePooling1D()(doc_input)

        concat = merge([x, y], mode="concat")


        pred = Dense(1, activation='sigmoid')(concat)

        self.model = Model(input=[que_input, doc_input], output=pred)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class AVGPolling():
    def __init__(self, max_len, emb_dim):
        x_input = Input(shape=(max_len, emb_dim,))
        avg = GlobalAveragePooling1D()(x_input)
        self.model = Model(input=x_input, output=avg)



# http://alexadam.ca/ml/2017/05/05/keras-vae.html
class EMB_LSTM_VAE():
    def __init__(self, vocab_size=50000, max_length=300, latent_rep_size=50):
        self.encoder = None
        self.decoder = None
        self.sentiment_predictor = None
        self.autoencoder = None
        
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.latent_rep_size = latent_rep_size

        x = Input(shape=(max_length,))
        self.x_embed = Embedding(vocab_size, 100, input_length=max_length, mask_zero=True)(x)

        vae_loss, encoded = self._build_encoder(self.x_embed, latent_rep_size=latent_rep_size, max_length=max_length)
        self.encoder = Model(x, encoded)

        encoded_input = Input(shape=(latent_rep_size,))

        decoded = self._build_decoder(encoded_input, vocab_size, max_length)
        self.decoder = Model(encoded_input, decoded)

        self.model = Model(x, self._build_decoder(encoded, vocab_size, max_length))

        self.model.compile(optimizer='Adam',
                                 loss=vae_loss)
        
    def _build_encoder(self, x, latent_rep_size=200, max_length=300, epsilon_std=0.01):
        h = LSTM(200, name='lstm_1')(x)


        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            latent_dim = K.sahep(z_mean_)[1]
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))
    
    def _build_decoder(self, encoded, vocab_size, max_length):
        
        repeated_context = RepeatVector(max_length)(encoded)
        h = LSTM(200, return_sequences=True, name='dec_lstm_1')(repeated_context)
        decoded = TimeDistributed(Dense(vocab_size, activation='softmax'), name='decoded_mean')(h)
        
        return decoded


class VAE_BPE():

    def __init__(self, hidden_dim=300, latent_dim=128, nb_words=50005, max_len=10, emb=None, emb_dim=200, activation="relu"):

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.nb_words = nb_words
        self.activation = activation
        self.emb_dim = emb_dim
        self.max_len = max_len

        x = Input(shape=(self.max_len,))
        
        embed_layer = Embedding(self.nb_words, self.emb_dim, input_length=self.max_len) if emb == None else emb


        emb_x = embed_layer(x)

        vae_loss, encoded = self.build_encoder(emb_x)
        self.encoder = Model(x, encoded)

        encoded_input = Input(shape=(self.latent_dim,))

        decoded = self.build_decoder(encoded_input)
        self.decoder = Model(encoded_input, decoded)

        self.model = Model(x, self.build_decoder(encoded))

        self.model.compile(optimizer='Adam',
                                 loss=vae_loss)
        
    def build_encoder(self, z):

        z = Bidirectional(LSTM(self.hidden_dim, name='lstm_1'))(z)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            latent_dim = K.shape(z_mean_)[1]
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(self.latent_dim, name='z_mean', activation='linear')(z)
        z_log_var = Dense(self.latent_dim, name='z_log_var', activation='linear')(z)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = self.max_len * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(self.latent_dim,), name='lambda')([z_mean, z_log_var]))
    
    def build_decoder(self, encoded):
        repeated_context = RepeatVector(self.max_len)(encoded)
        h = Bidirectional(LSTM(self.hidden_dim, return_sequences=True, name='dec_lstm_1'))(repeated_context)
        decoded = TimeDistributed(Dense(self.nb_words, activation='softmax'), name='decoded_mean')(h)

        return decoded
    
    def get_name(self):
        return "vae_bpe_h%d_l%d_w%d_%s_ml%d" % (self.hidden_dim, self.latent_dim, self.nb_words, self.activation, self.max_len)

    def initModel(self, sp, bpe_dict):
        self.sp = sp
        self.bpe_dict = bpe_dict

    def batch_generator(self, reader, train_data, batch_size):
        while True:
            for df in reader:

                x = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                x_one_hot = to_categorical(x, self.nb_words)
                x_one_hot = x_one_hot.reshape(batch_size, self.max_len, self.nb_words)
                                
                yield x, x_one_hot
                


class VAE_DSSM():

    def __init__(self, hidden_dim=300, latent_dim=128, nb_words=50005, activation="relu"):

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.nb_words = nb_words
        self.activation = activation


        x = Input(shape=(self.nb_words,))
        enc_dense = Dense(self.hidden_dim, activation=self.activation)

        emb_x = enc_dense(x)

        vae_loss, encoded = self.build_encoder(emb_x)
        self.encoder = Model(x, encoded)

        encoded_input = Input(shape=(self.latent_dim,))

        decoded = self.build_decoder(encoded_input)
        self.decoder = Model(encoded_input, decoded)

        self.model = Model(x, self.build_decoder(encoded))

        self.model.compile(optimizer='Adam',
                                 loss=vae_loss)
        
    def build_encoder(self, z):


        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            latent_dim = K.shape(z_mean_)[1]
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(self.latent_dim, name='z_mean', activation='linear')(z)
        z_log_var = Dense(self.latent_dim, name='z_log_var', activation='linear')(z)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(self.latent_dim,), name='lambda')([z_mean, z_log_var]))
    
    def build_decoder(self, encoded):

        h = Dense(self.hidden_dim, activation=self.activation)(encoded)
#       output is 2D one-hot vector, sigmoid is appropirate
        decoded = Dense(self.nb_words, activation='sigmoid', name='decoded_mean')(h)
        
        return decoded

    def get_name(self):
        return "vae_dssm_h%d_l%d_w%d_%s" % (self.hidden_dim, self.latent_dim, self.nb_words, self.activation)


    def batch_generator(self, reader, train_data, tokeniser, batch_size, max_len, nb_words):
        while True:
            for df in reader:
                q = df.q.tolist()
#                 if train_data == "1M_EN_QQ_log":
#                     d = [i.split("<sep>")[0] for i in df.d.tolist()]
#                 else:
#                     d = df.d.tolist()
                
                q = pad_sequences(tokeniser.texts_to_sequences(q), maxlen=max_len)
                
                q_one_hot = np.zeros((batch_size, nb_words))
                for i in range(len(q)):
                    q_one_hot[i][q[i]] = 1
                    
                
                yield q_one_hot, q_one_hot



class CLSM():

    # K - hidden layer dimension
    # L - latent dimension
    # J - number of negative samples
    # self.nb_words - feature number / nb_words
    
    def __init__(self, hidden_dim=300, latent_dim=128, FILTER_LENGTH=1, num_negatives=1,  nb_words=50005, activation="relu"):

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.nb_words = nb_words
        self.activation = activation


        self.num_negatives = num_negatives

        # Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
        # The first dimension is None because the queries and documents can vary in length.
        query = Input(shape = (None, self.nb_words))
        pos_doc = Input(shape = (None, self.nb_words))
        neg_docs = [Input(shape = (None, self.nb_words)) for j in range(self.num_negatives)]

        query_conv = Convolution1D(self.hidden_dim, FILTER_LENGTH, padding = "same", input_shape = (None, self.nb_words), activation = "tanh")(query) # See equation (2).
        query_max = Lambda(lambda x: K.max(x, axis = 1), output_shape = (self.hidden_dim, ))(query_conv) # See section 3.4.

        query_sem = Dense(self.latent_dim, activation = "tanh", input_dim = self.hidden_dim)(query_max) # See section 3.5.

        # The document equivalent of the above query model.
        doc_conv = Convolution1D(self.hidden_dim, FILTER_LENGTH, padding = "same", input_shape = (None, self.nb_words), activation = "tanh")
        doc_max = Lambda(lambda x: K.max(x, axis = 1), output_shape = (self.hidden_dim, ))
        doc_sem = Dense(self.latent_dim, activation = "tanh", input_dim = self.hidden_dim)

        pos_doc_conv = doc_conv(pos_doc)
        neg_doc_convs = [doc_conv(neg_doc) for neg_doc in neg_docs]

        pos_doc_max = doc_max(pos_doc_conv)
        neg_doc_maxes = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs]

        pos_doc_sem = doc_sem(pos_doc_max)
        neg_doc_sems = [doc_sem(neg_doc_max) for neg_doc_max in neg_doc_maxes]

        # This layer calculates the cosine similarity between the semantic representations of
        # a query and a document.
        R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
        R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True) for neg_doc_sem in neg_doc_sems] # See equation (4).

        concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
        concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)

        # In this step, we multiply each R(Q, D) value by gamma. In the paper, gamma is
        # described as a smoothing factor for the softmax function, and it's set empirically
        # on a held-out data set. We're going to learn gamma's value by pretending it's
        # a single 1 x 1 kernel.
        weight = np.array([1]).reshape(1, 1, 1)
        with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
        with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)

        # Finally, we use the softmax function to calculate P(D+|Q).
        prob = Activation("softmax")(with_gamma) # See equation (5).

        # We now have everything we need to define our model.
        self.model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
        self.model.compile(optimizer = "adadelta", loss = "categorical_crossentropy")

        self.encoder = Model(inputs=query, outputs=query_sem)

    def batch_generator(self, reader, train_data, tokeniser, batch_size, max_len, nb_words):
        while True:
            for df in reader:
                q = df.q.tolist()
                if train_data == "1M_EN_QQ_log":
                    d = [i.split("<sep>")[0] for i in df.d.tolist()]
                else:
                    d = df.d.tolist()
                
                q = pad_sequences(tokeniser.texts_to_sequences(q), maxlen=max_len)
                d = pad_sequences(tokeniser.texts_to_sequences(d), maxlen=max_len)
                
                
                q_one_hot = to_categorical(q, nb_words)   
                q_one_hot = q_one_hot.reshape(batch_size, max_len, nb_words)
                
                d_one_hot = to_categorical(d, nb_words)   
                d_one_hot = q_one_hot.reshape(batch_size, max_len, nb_words)
                    
                    
                # negative sampling from positive pool
                neg_d_one_hot = [[] for j in range(self.num_negatives)]
                for i in range(batch_size):
                    possibilities = list(range(batch_size))
                    possibilities.remove(i)
                    negatives = np.random.choice(possibilities, self.num_negatives, replace = False)
                    for j in range(self.num_negatives):
                        negative = negatives[j]
                        neg_d_one_hot[j].append(d_one_hot[negative].tolist())
                
                y = np.zeros((batch_size, self.num_negatives + 1))
                y[:, 0] = 1
                
                for j in range(self.num_negatives):
                    neg_d_one_hot[j] = np.array(neg_d_one_hot[j])
                
                
                yield [q_one_hot, d_one_hot] + [neg_d_one_hot[j] for j in range(self.num_negatives)], y
       
    
    
    
class DSSM():
    
    def __init__(self, hidden_dim=300, latent_dim=128, num_negatives=1, nb_words=50005, max_len=10, emb=None):

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_negatives = num_negatives
        self.nb_words = nb_words
        self.max_len = max_len
        # Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
        # The first dimension is None because the queries and documents can vary in length.
        query = Input(shape = (self.max_len,))
        pos_doc = Input(shape = (self.max_len,))
        neg_docs = [Input(shape = (self.max_len,)) for j in range(self.num_negatives)]

        embed_layer = emb
        bilstm = Bidirectional(LSTM(hidden_dim, name='lstm_1'))
        dense1 = Dense(self.hidden_dim, activation = "tanh")
        dense2 = Dense(self.latent_dim, activation = "tanh")



        query_sem = dense2(dense1(bilstm(embed_layer(query))))



        pos_doc_sem = dense2(dense1(bilstm(embed_layer(pos_doc))))
        neg_doc_sems = [dense2(dense1(bilstm(embed_layer(neg_doc)))) for neg_doc in neg_docs]

        # This layer calculates the cosine similarity between the semantic representations of
        # a query and a document.
        R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
        R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True) for neg_doc_sem in neg_doc_sems] # See equation (4).

        concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
        concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)

        # In this step, we multiply each R(Q, D) value by gamma. In the paper, gamma is
        # described as a smoothing factor for the softmax function, and it's set empirically
        # on a held-out data set. We're going to learn gamma's value by pretending it's
        # a single 1 x 1 kernel.
        weight = np.array([1]).reshape(1, 1, 1)
        with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
        with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)

        # Finally, we use the softmax function to calculate P(D+|Q).
        prob = Activation("softmax")(with_gamma) # See equation (5).

        # We now have everything we need to define our model.
        self.model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
        self.model.compile(optimizer = "adadelta", loss = "categorical_crossentropy")

        self.encoder = Model(inputs=query, outputs=query_sem)

    def initModel(self, sp, bpe_dict):
        self.sp = sp
        self.bpe_dict = bpe_dict

    def batch_generator(self, reader, train_data, batch_size):
        while True:
            for df in reader:

                q = parse_texts_bpe(df.q.tolist(), self.sp, self.bpe_dict, self.max_len, True)
                d = parse_texts_bpe(df.d.tolist(), self.sp, self.bpe_dict, self.max_len, True)

                # negative sampling from positive pool
                neg = [[] for j in range(self.num_negatives)]
                for i in range(batch_size):
                    possibilities = list(range(batch_size))
                    possibilities.remove(i)
                    negatives = np.random.choice(possibilities, self.num_negatives, replace = False)
                    for j in range(self.num_negatives):
                        negative = negatives[j]
                        neg[j].append(d[negative].tolist())
                
                y = np.zeros((batch_size, self.num_negatives + 1))
                y[:, 0] = 1
                
                for j in range(self.num_negatives):
                    neg[j] = np.array(neg[j])
                
                
                yield [q, d] + [neg[j] for j in range(self.num_negatives)], y
