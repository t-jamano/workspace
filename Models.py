import numpy as np
np.random.seed(0)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras import backend as K
K.set_session(session)


from Utils import *
from random import shuffle
from keras.layers import Bidirectional, Dense, Embedding, BatchNormalization, GRU, GlobalMaxPooling1D, Concatenate, Flatten, Reshape, Input, Lambda, LSTM, merge, GlobalAveragePooling1D, RepeatVector, TimeDistributed, Layer, Activation, Dropout, Masking
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras import objectives
from keras import backend as K
from keras.models import Model, load_model, Sequential
from keras.engine import Layer
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import LambdaCallback
from keras_tqdm import TQDMNotebookCallback
from keras_tqdm import TQDMCallback
from keras.models import load_model
from keras.callbacks import CSVLogger
# from keras_adversarial.legacy import l1l2
from keras_adversarial import AdversarialModel, fix_names
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling, AdversarialOptimizerAlternating, AdversarialOptimizerScheduled 
from keras.layers import LeakyReLU, Activation, Concatenate, Dot, Add, Subtract, Multiply
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import initializers
from gensim.models import KeyedVectors
from gensim.models import KeyedVectors
import sentencepiece as spm
from copy import copy
from nltk.translate.bleu_score import sentence_bleu
import random

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

class OnehotEmbedding(Layer):

    def __init__(self, Nembeddings, d=2, **kwargs):
        self.Nembeddings = Nembeddings
        self.d = d
        super(OnehotEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[self.d], self.Nembeddings),
                                      initializer='uniform',
                                      trainable=True)
        super(OnehotEmbedding, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.Nembeddings)


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
        self.kernel = K.transpose(self.tied_to.embeddings)
        output = K.dot(x, self.kernel)
        if self.use_bias:
            output += self.bias
        return self.activation(output)




class VAE():
    

    def __init__(self, input_size, max_len, embedding_matrix, dim, batch_size, comp_topk=None, ctype=None, epsilon_std=1.0, optimizer=Adadelta(lr=2.), kl_weight=0, PoolMode="max"):
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
        
        # h1 = hidden_layer1(h1)

        self.z_mean = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)
        self.z_log_var = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)

        if self.comp_topk != None:
            self.z_mean_k = KCompetitive(self.comp_topk, self.ctype)(self.z_mean)
            encoded = Lambda(self.sampling, output_shape=(self.dim[1],))([self.z_mean_k, self.z_log_var])
        else:
            encoded = Lambda(self.sampling, output_shape=(self.dim[1],))([self.z_mean, self.z_log_var])


        repeated_context = RepeatVector(self.max_len)
        decoder_h = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)

        softmax_layer = Dense(self.nb_words, activation='softmax')
        decoder_mean = TimeDistributed(softmax_layer, name="rec")
        decoder_kl = TimeDistributed(softmax_layer, name="kl")

        h_decoded = decoder_h(repeated_context(encoded))
        x_decoded_mean = decoder_mean(h_decoded)
        x_decoded_kl = decoder_kl(h_decoded)
     
        
        
        self.model = Model(input_layer, [x_decoded_mean, x_decoded_kl])
        self.model.compile(optimizer=self.optimizer, loss=["sparse_categorical_crossentropy", self.kl_loss])

        self.encoder = Model(outputs=self.z_mean, inputs=input_layer)



    def name(self):
        n = "vae_%s" % self.PoolMode if self.comp_topk == None else "kate_%s" % self.PoolMode
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




class CosineSim():
    def __init__(self, feature_num):
        q_input = Input(shape=(feature_num,))
        d_input = Input(shape=(feature_num,))

        pred = merge([q_input, d_input], mode="cos")
        self.model = Model([q_input, d_input], pred)


class S2S_AAE(object):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), keep_rate_word_dropout=0.5, mode=1, enableWasserstein=False, enableS2S=False, separateEmbedding=False, enablePairLoss=False, enablePR=False):
        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.keep_rate_word_dropout = keep_rate_word_dropout
        self.mode = mode
        self.enableWasserstein = enableWasserstein
        self.enableS2S = enableS2S
        self.separateEmbedding = separateEmbedding
        self.enablePairLoss = enablePairLoss
        self.enablePR = enablePR
        self.num_negatives = 1
        self.build()

    def build(self):

        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        self.vae, self.enc, self.encoder = self.build_encoder()

        self.dis = self.build_gs_discriminator()

        self.pr_dis = self.build_pair_recon_discriminator()


        # assemple AAE
        enc_inputs = self.enc.inputs[0]
        z = self.enc(enc_inputs)

        zreal = normal_latent_sampling((self.dim[1],))(enc_inputs)
        yreal = self.dis(zreal)
        yfake = self.dis(z)

        if self.enablePairLoss:
            q_pred, pd_pred, nd_pred, pair = self.vae(self.vae.inputs)
            aae = Model(self.vae.inputs, fix_names([q_pred, pd_pred, nd_pred, pair, yfake, yreal], ["q_pred", "pd_pred", "nd_pred", "pair", "yfake", "yreal"]))
        else:
            xpred = self.vae(self.vae.inputs)
            aae = Model(self.vae.inputs, fix_names([xpred, yfake, yreal], ["xpred", "yfake", "yreal"]))

        # build adversarial model
        generative_params = self.vae.trainable_weights

        self.model = AdversarialModel(base_model=aae,
                                 player_params=[generative_params, self.dis.trainable_weights],
                                 player_names=["generator", "discriminator"])

        if self.mode == 1:
            adversarial_optimizer = AdversarialOptimizerSimultaneous()
        elif self.mode == 2:
            adversarial_optimizer = AdversarialOptimizerAlternating()
        elif self.mode == 3:
            adversarial_optimizer = AdversarialOptimizerScheduled([0]+ ([1] * 10))


        dis_loss = "binary_crossentropy" if not self.enableWasserstein else self.wasserstein_loss
        gen_loss = "sparse_categorical_crossentropy"

        if self.enablePairLoss:
            nd_weight = 1e-2 if not self.enableS2S else -1e-2
            self.model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                                  player_optimizers=[self.optimizer, self.optimizer],
                                  loss={"yfake": dis_loss, "yreal": dis_loss,
                                        "q_pred": gen_loss,
                                        "pd_pred": gen_loss,
                                        "nd_pred": gen_loss,
                                        "pair": "categorical_crossentropy"},
                                  player_compile_kwargs=[{"loss_weights": {"yfake": 1e-3, "yreal": 1e-3, "q_pred": 1e-2, "pd_pred": 1e-2, "nd_pred": nd_weight, "pair": 2}}] * 2)

        else:
            self.model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                                      player_optimizers=[self.optimizer, self.optimizer],
                                      loss={"yfake": dis_loss, "yreal": dis_loss,
                                            "xpred": gen_loss},
                                      player_compile_kwargs=[{"loss_weights": {"yfake": 1e-2, "yreal": 1e-2, "xpred": 1}}] * 2)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)



    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                  stddev=1)
        return z_mean + K.exp(z_log_var / 2) * epsilon 

    def build_encoder(self):

        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        encoder_inputs = Input(shape=(self.max_len,))
        self.encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="enc_embedding",
                                        mask_zero=True,
                                        trainable=True)

        self.encoder_lstm = GRU(hidden_dim, return_state=True, name="enc_gru")

        x = self.encoder_embedding(encoder_inputs)
        _, self.state = self.encoder_lstm(x)

        self.mean = Dense(latent_dim)
        self.var = Dense(latent_dim)

        state_mean = self.mean(self.state)
        state_var = self.var(self.state)

        

        state_z = Lambda(self.sampling, name="kl")([state_mean, state_var])


        decoder_inputs = Input(shape=(self.max_len,), name="dec_input")
        # state_inputs = Input(shape=(latent_dim,), name="dec_state_input")

        self.latent2hidden = Dense(hidden_dim)
        self.decoder_lstm = GRU(hidden_dim, return_sequences=True)
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")
        self.decoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="dec_embedding",
                                        mask_zero=True,
                                        trainable=True)
        
        x = self.encoder_embedding(decoder_inputs) if not self.separateEmbedding else self.decoder_embedding(decoder_inputs)
        decoder_outputs = self.decoder_lstm(x, initial_state=self.latent2hidden(state_z))
        rec_outputs = self.decoder_dense(decoder_outputs)

        if self.enablePairLoss:
            pos_inputs, neg_inputs, pos_decoder_inputs, neg_decoder_inputs, pos_rec_outputs, neg_rec_outputs, pairwise_pred = self.build_pairwise()
            return Model([encoder_inputs, pos_inputs, neg_inputs, decoder_inputs, pos_decoder_inputs, neg_decoder_inputs], [rec_outputs, pos_rec_outputs, neg_rec_outputs, pairwise_pred]), Model(encoder_inputs, state_z), Model(encoder_inputs, state_mean)
        else:
            return Model([encoder_inputs, decoder_inputs], rec_outputs), Model(encoder_inputs, state_z), Model(encoder_inputs, state_mean)

    def build_pair_recon_discriminator(self):

        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        latent_inputs = Input(shape=(latent_dim,), name="latent_dis_latent_input")

        h = Dense(hidden_dim)(latent_inputs)
        h = LeakyReLU(0.2)(h)
        h = Dense(latent_dim)(h)
        h = LeakyReLU(0.2)(h)

        return Model(latent_inputs, h)

    def build_gs_discriminator(self):

        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        latent_inputs = Input(shape=(latent_dim,), name="dis_latent_input")

        h = Dense(hidden_dim)(latent_inputs)
        h = LeakyReLU(0.2)(h)
        h = Dense(hidden_dim)(h)
        h = LeakyReLU(0.2)(h)

        pred = Dense(1, activation='sigmoid' if not self.enableWasserstein else 'linear', name="dis")(h)

        return Model(latent_inputs, pred)

    def build_pairwise(self):

        pos_inputs = Input(shape=(self.max_len,))
        neg_inputs = Input(shape=(self.max_len,))
        pos_decoder_inputs = Input(shape=(self.max_len,))
        neg_decoder_inputs = Input(shape=(self.max_len,))

        _, p_state = self.encoder_lstm(self.encoder_embedding(pos_inputs))
        _, n_state = self.encoder_lstm(self.encoder_embedding(neg_inputs))

        p_state_mean = self.mean(p_state)
        p_state_var = self.var(p_state)

        n_state_mean = self.mean(n_state)
        n_state_var = self.var(n_state)

        p_state_z = Lambda(self.sampling, name="pos_kl")([p_state_mean, p_state_var])
        n_state_z = Lambda(self.sampling, name="neg_kl")([n_state_mean, n_state_var])


        query_sem = self.state
        pos_doc_sem = p_state
        neg_doc_sem = n_state

        R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
        R_Q_D_ns = dot([query_sem, neg_doc_sem], axes = 1, normalize = True) # See equation (4).
        concat_Rs = concatenate([R_Q_D_p, R_Q_D_ns])
        concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)
        weight = np.array([1]).reshape(1, 1, 1)
        with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
        with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)
        pairwise_pred = Activation("softmax", name="pair")(with_gamma) # See equation (5).

        pos_rec_outputs = self.decoder_dense(self.decoder_lstm(self.encoder_embedding(pos_decoder_inputs), initial_state=self.latent2hidden(p_state_z)))
        neg_rec_outputs = self.decoder_dense(self.decoder_lstm(self.encoder_embedding(neg_decoder_inputs), initial_state=self.latent2hidden(n_state_z)))

        return pos_inputs, neg_inputs, pos_decoder_inputs, neg_decoder_inputs, pos_rec_outputs, neg_rec_outputs, pairwise_pred

    def build_pairwise_model(self):

        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        query_sem = Input(shape=(latent_dim,))
        pos_doc_sem = Input(shape=(latent_dim,))
        neg_doc_sem = Input(shape=(latent_dim,))

        R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
        R_Q_D_ns = dot([query_sem, neg_doc_sem], axes = 1, normalize = True) # See equation (4).
        concat_Rs = concatenate([R_Q_D_p, R_Q_D_ns])
        concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)
        weight = np.array([1]).reshape(1, 1, 1)
        with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
        with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)
        pairwise_pred = Activation("softmax", name="pair")(with_gamma) # See equation (5).

        return Model([query_sem, pos_doc_sem, neg_doc_sem], pairwise_pred)

    def name(self):

        model_name = "s2s_" if self.enableS2S else ""
        pair_name = "dssm_" if self.enablePairLoss else ""
        loss_name = "aae" if not self.enableWasserstein else "wae"
        loss_name = loss_name + "_" if not self.separateEmbedding else loss_name + "2_"

        return "%s%s%sm%d_wd%.2f" % (pair_name, loss_name, model_name, self.mode, self.keep_rate_word_dropout)
    
    def word_dropout(self, x, unk_token):

        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

        return x_

class BOE():
    def __init__(self, nb_words=50005, max_len=10, embedding_matrix=None):

        q_embed_layer = Embedding(nb_words,
                    embedding_matrix.shape[-1],
                    weights=[embedding_matrix],
                    input_length=max_len,
                    name="q_embedding",
                    trainable=True)

        self.encoder = Sequential([q_embed_layer, GlobalMaxPooling1D()])
    def name(self):
        return "boe"
class BinaryClassifier():

    def __init__(self, hidden_dim=300, latent_dim=128, nb_words=50005, max_len=10, embedding_matrix=None, optimizer=None, enableLSTM=False, enableSeparate=False, PoolMode="max", mode=1):

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.enableSeparate = enableSeparate
        self.enableLSTM = enableLSTM
        self.PoolMode = PoolMode


        query = Input(shape = (self.max_len,))
        doc = Input(shape = (self.max_len,))

        q_embed_layer = Embedding(nb_words,
                    embedding_matrix.shape[-1],
                    weights=[embedding_matrix],
                    input_length=max_len,
                    # mask_zero=True,
                    name="q_embedding",
                    trainable=True)

        d_embed_layer = q_embed_layer
        
        q_bilstm = Bidirectional(GRU(hidden_dim, return_sequences=True), name='q_gru')
        d_bilstm = q_bilstm

        Pooling = GlobalMaxPooling1D() if self.PoolMode == "max" else GlobalAveragePooling1D()
        hidden2latent = Dense(latent_dim, activation="tanh", name="q_dense")

        query_sem = hidden2latent(GlobalMaxPooling1D()(q_bilstm(q_embed_layer(query))))
        doc_sem = hidden2latent(GlobalMaxPooling1D()(d_bilstm(d_embed_layer(doc))))
        # query_sem = hidden2latent(q_bilstm(q_embed_layer(query)))
        # doc_sem = hidden2latent(d_bilstm(d_embed_layer(doc)))
        # query_sem = GlobalMaxPooling1D()(q_bilstm(q_embed_layer(query)))
        # doc_sem = GlobalMaxPooling1D()(d_bilstm(d_embed_layer(doc)))

# #       concat > mul > cos
#         # old
        cos = Flatten()(merge([query_sem, doc_sem], mode="cos"))
        # cos = BatchNormalization()(cos)
#         cos = Dropout(0.2)(cos)
#         # concat = Concatenate()([query_sem, doc_sem])
#         # sub = Subtract()([query_sem, doc_sem])
#         # mul = Multiply()([query_sem, doc_sem])
#         # merge_all = merge([concat, sub, mul], mode="concat")

        # latent2hidden = Dense(hidden_dim, activation="relu")
        # cos = Activation("sigmoid")(cos)
        # cos = latent2hidden(cos)
        # pred = Dense(1, activation="sigmoid")(cos)

        self.model = Model([query, doc] , cos)
        self.model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics=["accuracy"])
        # self.model.compile(optimizer = optimizer, loss = "mean_squared_error", metrics=["accuracy"])
        # cosine_proximity
        self.encoder = Model(inputs=query, outputs=query_sem)

    def name(self):
        return "binary" if self.enableLSTM else "binary_%s" % self.PoolMode


class AdversarialPairwiseModel(object):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), keep_rate_word_dropout=0.5, mode=1, enableWasserstein=False, enableS2S=False):
        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.keep_rate_word_dropout = keep_rate_word_dropout
        self.mode = mode
        self.enableWasserstein = enableWasserstein
        self.enableS2S = enableS2S
        self.num_negatives = 1
        self.build()

    def build(self):

        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        self.ae, self.encoder, self.ae_encoder, self.ae_decoder, self.dssm = self.build_autoencoder()

        self.gs_dis = self.build_gs_discriminator()
        self.pr_dis = self.build_pr_discriminator()

        # inputs = [Input((self.max_len,), name="pr_input%d" % i) for i in range(3)]
        inputs = [self.ae.inputs[i] for i in range(3)]

        decoder_inputs = [self.ae.inputs[i+3] for i in range(3)]


        latents = [self.encoder(i) for i in inputs]

        zreal = [normal_latent_sampling((latent_dim,))(i) for i in inputs]
        yreal = [self.gs_dis(i) for i in zreal]
        yfake = [self.gs_dis(self.ae_encoder(i)) for i in latents]


        dis_latents = [self.pr_dis(i) for i in latents]

        pr_y_real = self.dssm(dis_latents)
        pr_y_fake = [self.ae_decoder([i, j]) for i, j in zip(decoder_inputs, dis_latents)]



        # xpred = [self.ae([i, j]) for i, j in zip(latents, decoder_inputs)]
        xpred = self.ae(self.ae.inputs)



        combine_outputs = xpred + yfake + yreal  + pr_y_fake + [pr_y_real]
        combine_inputs = self.ae.inputs

        outputs_name = ["q_pred", "pd_pred", "nd_pred", "pair", "q_yfake", "pd_yfake", "nd_yfake", "q_yreal", "pd_yreal", "nd_yreal", "pr_q_fake", "pr_pd_fake", "pr_nd_fake", "pr_y_real"]

        combine_models = Model(combine_inputs, fix_names(combine_outputs, outputs_name))

        # build adversarial model
        generative_params = self.ae.trainable_weights + self.encoder.trainable_weights + self.dssm.trainable_weights

        gs_discriminative_params = self.gs_dis.trainable_weights
        pr_discriminative_params = self.pr_dis.trainable_weights

        self.model = AdversarialModel(base_model=combine_models,
                                 player_params=[generative_params, gs_discriminative_params + pr_discriminative_params],
                                 player_names=["generator", "discriminator"])




        if self.mode == 1:
            adversarial_optimizer = AdversarialOptimizerSimultaneous()
        elif self.mode == 2:
            adversarial_optimizer = AdversarialOptimizerAlternating()
        elif self.mode == 3:
            adversarial_optimizer = AdversarialOptimizerScheduled([1,1,1,1,1,0])


        dis_loss = "binary_crossentropy" if not self.enableWasserstein else self.wasserstein_loss

        gen_weights = [{"loss_weights": {"pr_q_fake": 1e-3, "pr_pd_fake": 1e-3, "pr_nd_fake": 1e-3, "pr_y_real": -1e-3, "q_yfake": 1e-3, "pd_yfake": 1e-3, "nd_yfake": 1e-3, "q_yreal": -1e-3, "pd_yreal": -1e-3, "nd_yreal": -1e-3, "q_pred": 1e-1, "pd_pred": 1e-1, "nd_pred": 1e-1, "pair": 1}}]
        dis_weights = [{"loss_weights": {"pr_q_fake": -1e-3, "pr_pd_fake": -1e-3, "pr_nd_fake": -1e-3, "pr_y_real": 1e-3, "q_yfake": -1e-3, "pd_yfake": -1e-3, "nd_yfake": -1e-3, "q_yreal": 1e-3, "pd_yreal": 1e-3, "nd_yreal": 1e-3, "q_pred": 1e-1, "pd_pred": 1e-1, "nd_pred": 1e-1, "pair": 1}}]

        self.model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[self.optimizer, self.optimizer],
                              loss={"q_yfake": dis_loss, "q_yreal": dis_loss,
                                    "pd_yfake": dis_loss, "pd_yreal": dis_loss,
                                    "nd_yfake": dis_loss, "nd_yreal": dis_loss,
                                    "q_pred": "sparse_categorical_crossentropy",
                                    "pd_pred": "sparse_categorical_crossentropy",
                                    "nd_pred": "sparse_categorical_crossentropy",
                                    "pr_q_fake": "sparse_categorical_crossentropy",
                                    "pr_pd_fake": "sparse_categorical_crossentropy",
                                    "pr_nd_fake": "sparse_categorical_crossentropy",
                                    "pair": "categorical_crossentropy",
                                    "pr_y_real": "categorical_crossentropy"},
                              player_compile_kwargs=gen_weights+dis_weights)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                  stddev=1)
        return z_mean + K.exp(z_log_var / 2) * epsilon 

    def build_autoencoder(self):
        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        q_inputs = Input(shape=(self.max_len,), name="main_encoder_input1")
        pd_inputs = Input(shape=(self.max_len,), name="main_encoder_input2")
        nd_inputs = Input(shape=(self.max_len,), name="main_encoder_input3")



        self.encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="enc_embedding",
                                        mask_zero=True,
                                        trainable=True)

        self.encoder_lstm = GRU(hidden_dim, return_state=True, name="enc_gru")
        self.hidden2latent = Dense(latent_dim)


        self.q_state = self.hidden2latent(self.encoder_lstm(self.encoder_embedding(q_inputs))[-1])
        self.pd_state = self.hidden2latent(self.encoder_lstm(self.encoder_embedding(pd_inputs))[-1])
        self.nd_state = self.hidden2latent(self.encoder_lstm(self.encoder_embedding(nd_inputs))[-1])


        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]


        self.mean = Dense(latent_dim)
        self.var = Dense(latent_dim)

        q_state_z = Lambda(self.sampling, name="q_kl")([self.mean(self.q_state), self.var(self.q_state)])
        pd_state_z = Lambda(self.sampling, name="pd_kl")([self.mean(self.pd_state), self.var(self.pd_state)])
        nd_state_z = Lambda(self.sampling, name="nd_kl")([self.mean(self.nd_state), self.var(self.nd_state)])



        q_decoder_inputs = Input(shape=(self.max_len,), name="q_dec_input")
        pd_decoder_inputs = Input(shape=(self.max_len,), name="pd_dec_input")
        nd_decoder_inputs = Input(shape=(self.max_len,), name="nd_dec_input")

        # q_decoder_inputs = Input(shape=(self.max_len,), name="q_dec_input")
        # pd_decoder_inputs = Input(shape=(self.max_len,), name="pd_dec_input")
        # nd_decoder_inputs = Input(shape=(self.max_len,), name="nd_dec_input")


        self.latent2hidden = Dense(hidden_dim)
        self.decoder_lstm = GRU(hidden_dim, return_sequences=True)
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")
        self.decoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="dec_embedding",
                                        mask_zero=True,
                                        trainable=True)
        
        q_rec_outputs = self.decoder_dense(self.decoder_lstm(self.decoder_embedding(q_decoder_inputs), initial_state=self.latent2hidden(q_state_z)))
        pd_rec_outputs = self.decoder_dense(self.decoder_lstm(self.decoder_embedding(pd_decoder_inputs), initial_state=self.latent2hidden(pd_state_z)))
        nd_rec_outputs = self.decoder_dense(self.decoder_lstm(self.decoder_embedding(nd_decoder_inputs), initial_state=self.latent2hidden(nd_state_z)))



        query_sem = self.q_state
        pos_doc_sem = self.pd_state
        neg_doc_sem = self.nd_state


        weight = np.array([1]).reshape(1, 1, 1)
        conv = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])
        pairwise_pred = self.pairwise_function(query_sem, pos_doc_sem, neg_doc_sem, conv)

        
        ae_encoder_input = Input((latent_dim, ), name="ae_encoder_input")
        ae_encoder_output = Lambda(self.sampling, name="kl")([self.mean(ae_encoder_input), self.var(ae_encoder_input)])
        
        ae_encoder = Model(ae_encoder_input, ae_encoder_output)


        ae_decoder_latent_input = Input((latent_dim, ), name="ae_decoder_latent_input")
        ae_decoder_input = Input((self.max_len, ), name="ae_decoder_input")
        infer_rec_outputs = self.decoder_dense(self.decoder_lstm(self.decoder_embedding(ae_decoder_input), initial_state=self.latent2hidden(ae_decoder_latent_input)))

        ae_decoder = Model([ae_decoder_input, ae_decoder_latent_input], infer_rec_outputs)


        query_dssm_input = Input(shape=(latent_dim,), name="q_pair_input")
        pos_doc_dssm_input = Input(shape=(latent_dim,), name="pos_pair_input")
        neg_doc_dssm_input = Input(shape=(latent_dim,), name="neg_pair_input")
        infer_pairwise_pred = self.pairwise_function(query_dssm_input, pos_doc_dssm_input, neg_doc_dssm_input, conv)

        dssm = Model([query_dssm_input, pos_doc_dssm_input, neg_doc_dssm_input], infer_pairwise_pred)

        vae_dssm = Model([q_inputs, pd_inputs, nd_inputs, q_decoder_inputs, pd_decoder_inputs, nd_decoder_inputs], [q_rec_outputs, pd_rec_outputs, nd_rec_outputs, pairwise_pred])
        encoder = Model(q_inputs, self.q_state)

        return vae_dssm, encoder, ae_encoder, ae_decoder, dssm


    def build_pr_discriminator(self):

        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        latent_inputs = Input(shape=(latent_dim,), name="pr_dis_input")

        h = Dense(hidden_dim)(latent_inputs)
        h = LeakyReLU(0.2)(h)
        h = Dense(latent_dim)(h)
        h = LeakyReLU(0.2)(h)

        return Model(latent_inputs, h)

    def build_gs_discriminator(self):

        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        latent_inputs = Input(shape=(latent_dim,), name="gs_dis_input")

        h = Dense(hidden_dim)(latent_inputs)
        h = LeakyReLU(0.2)(h)
        h = Dense(hidden_dim)(h)
        h = LeakyReLU(0.2)(h)

        pred = Dense(1, activation='sigmoid' if not self.enableWasserstein else 'linear', name="dis")(h)

        return Model(latent_inputs, pred)


    def pairwise_function(self, query_sem, pos_doc_sem, neg_doc_sem, conv):

        R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
        R_Q_D_ns = dot([query_sem, neg_doc_sem], axes = 1, normalize = True) # See equation (4).
        concat_Rs = concatenate([R_Q_D_p, R_Q_D_ns])
        concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)
        with_gamma = conv(concat_Rs) # See equation (5).
        with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)
        pairwise_pred = Activation("softmax", name="pair")(with_gamma) # See equation (5).

        return pairwise_pred

    def name(self):

        model_name = "s2s_" if self.enableS2S else ""
        loss_name = "aae" if not self.enableWasserstein else "wae"
        loss_name = "pr_"+loss_name

        return "%s%sm%d_wd%.2f" % (model_name, loss_name, self.mode, self.keep_rate_word_dropout)
    
    def word_dropout(self, x, unk_token):

        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

        return x_
    
class DSSM():
    
    def __init__(self, hidden_dim=300, latent_dim=128, num_negatives=1, nb_words=50005, max_len=10, embedding_matrix=None, optimizer=None, enableLSTM=False, enableSeparate=False, PoolMode="max", enableHybrid=False, limit=None):

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_negatives = num_negatives
        self.nb_words = nb_words
        self.max_len = max_len
        self.enableSeparate = enableSeparate
        self.enableLSTM = enableLSTM
        self.PoolMode = PoolMode
        self.enableHybrid = enableHybrid
        self.limit = limit

        # Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
        # The first dimension is None because the queries and documents can vary in length.
        query = Input(shape = (self.max_len,))
        pos_doc = Input(shape = (self.max_len,))
        neg_docs = [Input(shape = (self.max_len,)) for j in range(self.num_negatives)]

        embed_layer = Embedding(nb_words,
                    embedding_matrix.shape[-1],
                    weights=[embedding_matrix],
                    input_length=max_len,
                    name="q_embedding",
                    mask_zero=False if self.enableLSTM else False,
                    trainable=True)


        
        # bilstm = GRU(hidden_dim, name='q_gru', return_sequences=False, trainable=True)
        bilstm = Bidirectional(GRU(hidden_dim, return_sequences=True, trainable=True), name='q_gru', trainable=True)

        if enableSeparate:
            d_embed_layer = Embedding(nb_words,
                    embedding_matrix.shape[-1],
                    weights=[embedding_matrix],
                    input_length=max_len,
                    name="d_embedding",
                    mask_zero=False if self.enableLSTM else False,
                    trainable=True)
            # d_bilstm = GRU(hidden_dim, name="d_gru")
            d_bilstm = Bidirectional(GRU(hidden_dim, return_sequences=True, trainable=True), name='d_gru')


        if self.enableHybrid:
            bpe_embed_layer = Embedding(nb_words,
                    embedding_matrix.shape[-1],
                    weights=[embedding_matrix],
                    input_length=max_len,
                    mask_zero=True if self.enableLSTM else False,
                    trainable=True)
            bpe_bilstm = GRU(hidden_dim)

        Pooling = GlobalMaxPooling1D() if self.PoolMode == "max" else GlobalAveragePooling1D()

        dense = Dense(latent_dim, activation="tanh", name="q_dense")

        if self.enableLSTM:
            if enableHybrid:
                # query_sem = merge([bilstm(embed_layer(query)), bpe_bilstm(bpe_embed_layer(query))], mode="concat")
                # query_sem = Dense(hidden_dim)(query_sem)
                query_sem = bilstm(merge([embed_layer(query), bpe_embed_layer(query)]))
            else:
                query_sem = bilstm(embed_layer(query))
                query_sem = dense(GlobalMaxPooling1D()(query_sem))

            pos_doc_sem = bilstm(embed_layer(pos_doc)) if not enableSeparate else d_bilstm(d_embed_layer(pos_doc))
            pos_doc_sem = dense(GlobalMaxPooling1D()(pos_doc_sem))
            neg_doc_sems = [bilstm(embed_layer(neg_doc)) for neg_doc in neg_docs] if not enableSeparate else [d_bilstm(d_embed_layer(neg_doc)) for neg_doc in neg_docs]
            neg_doc_sems = [dense(GlobalMaxPooling1D()(i)) for i in neg_doc_sems]
        else:
            query_sem = Pooling(embed_layer(query))
            pos_doc_sem = Pooling(embed_layer(pos_doc)) if not enableSeparate else Pooling(d_embed_layer(pos_doc))
            neg_doc_sems = [Pooling(embed_layer(neg_doc)) for neg_doc in neg_docs] if not enableSeparate else [Pooling(d_embed_layer(neg_doc)) for neg_doc in neg_docs]


        # def Manhattan_distance(A,B):
            # return K.sum( K.abs( A-B),axis=1,keepdims=True)




        # This layer calculates the cosine similarity between the semantic representations of
        # a query and a document.
        R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
        R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True) for neg_doc_sem in neg_doc_sems] # See equation (4).
        # R_Q_D_p = Merge(mode=lambda x:Manhattan_distance(query_sem, pos_doc_sem)) # See equation (4).
        # R_Q_D_ns = [Merge(mode=lambda x:Manhattan_distance(query_sem, neg_doc_sem)) for neg_doc_sem in neg_doc_sems] # See equation (4).



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
        self.model.compile(optimizer = optimizer, loss = "categorical_crossentropy")

        self.encoder = Model(inputs=query, outputs=query_sem)
        # self.model.summary()

    def name(self):
        if self.enableLSTM and self.limit != None:
            return "dssm_s_%d" % self.limit
        elif self.enableLSTM:
            return "dssm_gru2" if self.enableSeparate else "dssm_gru"
        else:
            return "dssm2_%s" % self.PoolMode if self.enableSeparate else "dssm_%s" % self.PoolMode

class DSSMClassifier():
    
    def __init__(self, hidden_dim=300, latent_dim=128, num_negatives=1, nb_words=50005, max_len=10, embedding_matrix=None, optimizer=None, mode="bpe", trainable=False):


        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_negatives = num_negatives
        self.nb_words = nb_words
        self.max_len = max_len
        self.mode = mode
        self.trainable = trainable

        query = Input(shape = (self.max_len,))
        pos_doc = Input(shape = (self.max_len,))
        neg_docs = [Input(shape = (self.max_len,)) for j in range(self.num_negatives)]

        ae_embed_layer = Embedding(nb_words,
            embedding_matrix.shape[-1],
            weights=[embedding_matrix],
            input_length=max_len,
            name="q_embedding",
            trainable=self.trainable) if "ae" in mode else None

        bpe_embed_layer = Embedding(nb_words,
            embedding_matrix.shape[-1],
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=self.trainable) if "bpe" in mode else None

        bilstm = Bidirectional(GRU(hidden_dim, return_sequences=True, trainable=self.trainable), name='q_gru', trainable=self.trainable) if "ae" in mode else None
        bpe_bilstm = Bidirectional(GRU(hidden_dim, return_sequences=True, trainable=self.trainable), trainable=self.trainable)
        
        dense = Dense(latent_dim, activation="tanh", name="q_dense", trainable=self.trainable) if "ae" in mode else None
        bpe_dense = Dense(latent_dim, activation="tanh", trainable=self.trainable)



        if self.mode == "bpe_ae" and not self.trainable:

            query_sem = GlobalAveragePooling1D()(bpe_embed_layer(query))
            pos_doc_sem = GlobalAveragePooling1D()(bpe_embed_layer(pos_doc))
            neg_doc_sems = [GlobalAveragePooling1D()(bpe_embed_layer(neg_doc)) for neg_doc in neg_docs]

            ae_query_sem = dense(GlobalMaxPooling1D()(bilstm(ae_embed_layer(query))))
            ae_pos_doc_sem = dense(GlobalMaxPooling1D()(bilstm(ae_embed_layer(pos_doc))))
            ae_neg_doc_sems = [dense(GlobalMaxPooling1D()(bilstm(ae_embed_layer(neg_doc)))) for neg_doc in neg_docs]

            query_sem = merge([query_sem, ae_query_sem], mode='concat')
            pos_doc_sem = merge([pos_doc_sem, ae_pos_doc_sem], mode='concat')
            neg_doc_sems = [merge([i, j], mode="concat") for i, j in zip(neg_doc_sems, ae_neg_doc_sems)]

        elif self.mode == "bpe_ae" and self.trainable:

            query_sem = bpe_dense(GlobalMaxPooling1D()(bpe_bilstm(bpe_embed_layer(query))))
            pos_doc_sem = bpe_dense(GlobalMaxPooling1D()(bpe_bilstm(bpe_embed_layer(pos_doc))))
            neg_doc_sems = [bpe_dense(GlobalMaxPooling1D()(bpe_bilstm(bpe_embed_layer(neg_doc)))) for neg_doc in neg_docs]

            ae_query_sem = dense(GlobalMaxPooling1D()(bilstm(ae_embed_layer(query))))
            ae_pos_doc_sem = dense(GlobalMaxPooling1D()(bilstm(ae_embed_layer(pos_doc))))
            ae_neg_doc_sems = [dense(GlobalMaxPooling1D()(bilstm(ae_embed_layer(neg_doc)))) for neg_doc in neg_docs]

            query_sem = merge([query_sem, ae_query_sem], mode='concat')
            pos_doc_sem = merge([pos_doc_sem, ae_pos_doc_sem], mode='concat')
            neg_doc_sems = [merge([i, j], mode="concat") for i, j in zip(neg_doc_sems, ae_neg_doc_sems)]


        elif "bpe" in self.mode and not self.trainable:

            query_sem = GlobalAveragePooling1D()(bpe_embed_layer(query))
            pos_doc_sem = GlobalAveragePooling1D()(bpe_embed_layer(pos_doc))
            neg_doc_sems = [GlobalAveragePooling1D()(bpe_embed_layer(neg_doc)) for neg_doc in neg_docs] 

        elif "bpe" in self.mode and self.trainable:

            query_sem = bpe_dense(GlobalMaxPooling1D()(bpe_bilstm(bpe_embed_layer(query))))
            pos_doc_sem = bpe_dense(GlobalMaxPooling1D()(bpe_bilstm(bpe_embed_layer(pos_doc))))
            neg_doc_sems = [bpe_dense(GlobalMaxPooling1D()(bpe_bilstm(bpe_embed_layer(neg_doc)))) for neg_doc in neg_docs] 

        elif "ae" in self.mode:

            query_sem = dense(GlobalMaxPooling1D()(bilstm(ae_embed_layer(query))))
            pos_doc_sem = dense(GlobalMaxPooling1D()(bilstm(ae_embed_layer(pos_doc))))
            neg_doc_sems = [dense(GlobalMaxPooling1D()(bilstm(ae_embed_layer(neg_doc)))) for neg_doc in neg_docs] 


        if not self.trainable:
            dense1 = Dense(hidden_dim, activation="tanh", name="q_dense1")
            dense2 = Dense(latent_dim, activation="tanh", name="q_dense2")

            query_sem = dense2(dense1(query_sem))
            pos_doc_sem = dense2(dense1(pos_doc_sem))
            neg_doc_sems = [dense2(dense1(neg_doc)) for neg_doc in neg_doc_sems] 

        R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
        R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True) for neg_doc_sem in neg_doc_sems] # See equation (4).

        concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
        concat_Rs = Reshape((self.num_negatives + 1, 1))(concat_Rs)

        weight = np.array([1]).reshape(1, 1, 1)
        with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (self.num_negatives + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
        with_gamma = Reshape((self.num_negatives + 1, ))(with_gamma)

        # Finally, we use the softmax function to calculate P(D+|Q).
        prob = Activation("softmax")(with_gamma) # See equation (5).

        # We now have everything we need to define our model.
        self.model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
        self.model.compile(optimizer = optimizer, loss = "categorical_crossentropy")

        self.encoder = Model(inputs=query, outputs=query_sem)
        # self.model.summary()

    def name(self):
        # return "clf_%s" % self.mode 
        if not self.trainable:
            return "clf_%s" % self.mode
        else:
            return "clf_pre_kate_bow"



class AAE():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), mode=1, keep_rate_word_dropout=0.5, enableWasserstein=False, enableBOW=False, enableS2S=False, enablePairLoss=False):
        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.enableWasserstein = enableWasserstein
        self.enableS2S = enableS2S
        self.enableBOW = enableBOW
        self.mode = mode
        self.enablePairLoss = enablePairLoss
        self.keep_rate_word_dropout = keep_rate_word_dropout
        self.hidden_dim = self.dim[0]
        self.latent_dim = self.dim[1]
        self.num_negatives = 1

        self.dis_loss = "binary_crossentropy" if not self.enableWasserstein else self.wasserstein_loss

        self.build()


    def build(self):


        self.ae, self.gs_encoder, self.encoder = self.build_ae()
        self.discriminator = self.build_gs_discriminator()
        self.discriminator.compile(optimizer=Adam(), loss=self.dis_loss, metrics=['accuracy'])
        self.discriminator.trainable = False

        inputs = self.ae.inputs
        rec_pred = self.ae(inputs)
        aae_penalty = self.discriminator(self.gs_encoder(inputs[0]))

        if self.enablePairLoss:
            self.doc_ae, self.doc_gs_encoder, self.doc_encoder = self.build_ae2()
            self.doc_discriminator = self.build_gs_discriminator()
            self.doc_discriminator.compile(optimizer=Adam(), loss=self.dis_loss, metrics=['accuracy'])
            self.doc_discriminator.trainable = False

            self.dssm = self.build_pairwise_model()

            pd_inputs, nd_inputs, dec_pd_inputs, dec_nd_inputs = self.doc_ae.inputs

            pd_rec_pred, nd_rec_pred = self.doc_ae([pd_inputs, nd_inputs, dec_pd_inputs, dec_nd_inputs])

            if self.mode == 1:
                triplet_latents = [self.encoder(inputs[0]), self.doc_encoder(pd_inputs), self.doc_encoder(nd_inputs)]
            elif self.mode == 2:
                triplet_latents = [self.gs_encoder(inputs[0]), self.doc_gs_encoder(pd_inputs), self.doc_gs_encoder(nd_inputs)]

            pair_pred = self.dssm(triplet_latents)


            pd_aae_penalty = self.discriminator(self.doc_gs_encoder(pd_inputs))
            nd_aae_penalty = self.discriminator(self.doc_gs_encoder(nd_inputs))


            combine_inputs = [inputs[0], pd_inputs, nd_inputs, inputs[1], dec_pd_inputs, dec_nd_inputs]

            self.model = Model(combine_inputs, [rec_pred, pd_rec_pred, nd_rec_pred, pair_pred, aae_penalty, pd_aae_penalty, nd_aae_penalty])
            # self.model.compile(optimizer=Adam(), loss=["sparse_categorical_crossentropy"] * 3 + ["categorical_crossentropy"] + ["binary_crossentropy"] * 3 , loss_weights=[1e-3, 1e-3, 1e-3, 1, 1e-4, 1e-4, 1e-4])
            self.model.compile(optimizer=Adam(), loss=["sparse_categorical_crossentropy"] * 3 + ["categorical_crossentropy"] + [self.dis_loss] * 3 , loss_weights=[1e-4, 1e-4, 1e-4, 1, 1e-4, 1e-4, 1e-4])

        else:
            
            self.model = Model(inputs, [rec_pred, aae_penalty])
            self.model.compile(optimizer=Adam(), loss=["sparse_categorical_crossentropy", self.dis_loss], loss_weights=[0.9999, 0.0001])
        
        
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    def build_ae(self):

        encoder_inputs = Input(shape=(self.max_len,))
        self.encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="q_embedding_layer",
                                        mask_zero=True,
                                        trainable=True)

        self.encoder_lstm = GRU(self.hidden_dim, name="q_gru")



        x = self.encoder_embedding(encoder_inputs)
        self.state = self.encoder_lstm(x)

        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)

        state_mean = self.mean(self.state)
        state_var = self.var(self.state)

        state_z = Lambda(self.sampling, name="kl")([state_mean, state_var])


        decoder_inputs = Input(shape=(self.max_len,), name="dec_input")

        self.latent2hidden = Dense(self.hidden_dim)
        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True)
        self.decoder_dense = Dense(self.nb_words, activation='softmax' if not self.enableWasserstein else "linear", name="rec")
        self.decoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        name="dec_embedding",
                                        mask_zero=True,
                                        trainable=True)

        x = self.decoder_embedding(decoder_inputs)
        decoder_outputs = self.decoder_lstm(x, initial_state=self.latent2hidden(state_z))
        rec_outputs = self.decoder_dense(decoder_outputs)

        return Model([encoder_inputs, decoder_inputs], rec_outputs), Model(encoder_inputs, state_z), Model(encoder_inputs, state_mean)

    def build_ae2(self):

        encoder_inputs = Input(shape=(self.max_len,))
        encoder_inputs2 = Input(shape=(self.max_len,))

        self.encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        self.encoder_lstm = GRU(self.hidden_dim)



        x = self.encoder_embedding(encoder_inputs)
        x2 = self.encoder_embedding(encoder_inputs)

        self.state = self.encoder_lstm(x)
        self.state2 = self.encoder_lstm(x2)


        self.mean = Dense(self.latent_dim)
        self.var = Dense(self.latent_dim)

        state_mean = self.mean(self.state)
        state_var = self.var(self.state)

        state_mean2 = self.mean(self.state2)
        state_var2 = self.var(self.state2)

        state_z = Lambda(self.sampling)([state_mean, state_var])
        state_z2 = Lambda(self.sampling)([state_mean2, state_var2])

        decoder_inputs = Input(shape=(self.max_len,))
        decoder_inputs2 = Input(shape=(self.max_len,))


        self.latent2hidden = Dense(self.hidden_dim)
        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True)
        self.decoder_dense = Dense(self.nb_words, activation='softmax' if not self.enableWasserstein else "linear")
        self.decoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        x = self.decoder_embedding(decoder_inputs)
        x2 = self.decoder_embedding(decoder_inputs2)

        decoder_outputs = self.decoder_lstm(x, initial_state=self.latent2hidden(state_z))
        decoder_outputs2 = self.decoder_lstm(x2, initial_state=self.latent2hidden(state_z2))
        
        rec_outputs = self.decoder_dense(decoder_outputs)
        rec_outputs2 = self.decoder_dense(decoder_outputs2)

        return Model([encoder_inputs, encoder_inputs2, decoder_inputs, decoder_inputs2], [rec_outputs, rec_outputs2]), Model(encoder_inputs, state_z), Model(encoder_inputs, state_mean)
    
    def build_gs_discriminator(self):
        
        inputs = Input((self.latent_dim,))
        
        dense1 = Dense(self.hidden_dim)
        dense2 = Dense(self.latent_dim)
        dense3 = Dense(1, activation="sigmoid" if not self.enableWasserstein else "linear")

        outputs = dense3(dense2(dense1(inputs)))
        
        return Model(inputs, outputs)

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
    
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,\
                                  stddev=1)
        return z_mean + K.exp(z_log_var / 2) * epsilon 

    def name(self):
        if self.enablePairLoss:
            return "aae_pair_m%d" % self.mode if not self.enableWasserstein else "wae_pair_m%d" % self.mode
        else:
            return "aae" if not self.enableWasserstein else "wae"

    def word_dropout(self, x, unk_token):

        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

        return x_