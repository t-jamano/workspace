import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras import backend as K
K.set_session(session)


from Utils import *
from random import shuffle
from keras.layers import Bidirectional, Dense, Merge, Embedding, BatchNormalization, GRU, GlobalMaxPooling1D, Concatenate, Flatten, Reshape, Input, Lambda, LSTM, merge, GlobalAveragePooling1D, RepeatVector, TimeDistributed, Layer, Activation, Dropout, Masking
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
# from keras_adversarial.legacy import l1l2
from keras_adversarial import AdversarialModel, fix_names
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling, AdversarialOptimizerAlternating
from keras.layers import LeakyReLU, Activation, Concatenate, Dot, Add, Subtract, Multiply
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
        self.kernel = K.transpose(self.tied_to.kernel)
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
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), keep_rate_word_dropout=0.5, mode=1, enableWasserstein=False, enableS2S=True):
        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.keep_rate_word_dropout = keep_rate_word_dropout
        self.mode = mode
        self.enableWasserstein = enableWasserstein
        self.enableS2S = enableS2S

        self.build()

    def build(self):

        vae, enc, self.encoder = self.build_encoder()

        dis = self.build_discriminator()

        # assemple AAE
        enc_inputs = enc.inputs[0]
        z = enc(enc_inputs)

        xpred = vae(vae.inputs)
        zreal = normal_latent_sampling((self.dim[1],))(enc_inputs)
        yreal = dis(zreal)
        yfake = dis(z)
        
        self.model = Model(vae.inputs, fix_names([xpred, yfake, yreal], ["xpred", "yfake", "yreal"]))

        # build adversarial model
        generative_params = vae.trainable_weights

        self.model = AdversarialModel(base_model=self.model,
                                 player_params=[generative_params, dis.trainable_weights],
                                 player_names=["generator", "discriminator"])

        if self.mode == 1:
            adversarial_optimizer = AdversarialOptimizerSimultaneous()
        elif self.mode == 2:
            adversarial_optimizer = AdversarialOptimizerAlternating()


        rec_loss = "sparse_categorical_crossentropy" if not self.enableWasserstein else self.wasserstein_loss


        self.model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                                  player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                                  loss={"yfake": "binary_crossentropy", "yreal": "binary_crossentropy",
                                        "xpred": rec_loss},
                                  player_compile_kwargs=[{"loss_weights": {"yfake": 1e-2, "yreal": 1e-2, "xpred": 1}}] * 2)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_encoder(self):

        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        encoder_inputs = Input(shape=(self.max_len,))
        self.encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        encoder_lstm = GRU(hidden_dim, return_state=True)

        x = self.encoder_embedding(encoder_inputs)
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

        # state_z = Lambda(sampling, name="kl")([state_mean, state_var])
        state_z = merge([state_mean, state_var], mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
              output_shape=lambda p: p[0])

        # return Model(encoder_inputs, state_z), Model(encoder_inputs, state_mean)

    # def build_decoder(self):

        # hidden_dim = self.dim[0]
        # latent_dim = self.dim[1]

        decoder_inputs = Input(shape=(self.max_len,), name="dec_input")
        # state_inputs = Input(shape=(latent_dim,), name="dec_state_input")

        latent2hidden = Dense(hidden_dim)
        decoder_lstm = GRU(hidden_dim, return_sequences=True)
        decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")
        
        x = self.encoder_embedding(decoder_inputs)
        decoder_outputs = decoder_lstm(x, initial_state=latent2hidden(state_z))
        rec_outputs = decoder_dense(decoder_outputs)

        return Model([encoder_inputs, decoder_inputs], rec_outputs), Model(encoder_inputs, state_z), Model(encoder_inputs, state_mean)

    def build_discriminator(self):

        hidden_dim = self.dim[0]
        latent_dim = self.dim[1]

        latent_inputs = Input(shape=(latent_dim,), name="dis_latent_input")

        latent2hidden = Dense(hidden_dim)
        pred = Dense(1, activation='sigmoid', name="dis")(latent2hidden(latent_inputs))

        return Model(latent_inputs, pred)

    def name(self):

        model_name = "s2s_" if self.enableS2S else ""
        loss_name = "aae_" if not self.enableWasserstein else "wae_"

        return "%s%sm%d_wd%.2f" % (model_name, loss_name, self.mode, self.keep_rate_word_dropout)
    
    def word_dropout(self, x, unk_token):

        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

        return x_

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
                    mask_zero=True if self.enableLSTM else False,
                    trainable=True)

        d_embed_layer = Embedding(nb_words,
                    embedding_matrix.shape[-1],
                    weights=[embedding_matrix],
                    input_length=max_len,
                    mask_zero=True if self.enableLSTM else False,
                    trainable=True)
        
        q_bilstm = LSTM(hidden_dim, return_sequences=False)
        d_bilstm = LSTM(hidden_dim, return_sequences=False)

        Pooling = GlobalMaxPooling1D() if self.PoolMode == "max" else GlobalAveragePooling1D()

        if self.enableLSTM:
            query_sem = q_bilstm(q_embed_layer(query))
            doc_sem = d_bilstm(d_embed_layer(doc)) 
        else:
            query_sem = Pooling(q_embed_layer(query))
            doc_sem = Pooling(d_embed_layer(doc)) 

        def exponent_neg_manhattan_distance(left, right):
            ''' Helper function for the similarity estimate of the LSTMs outputs'''
            return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


        malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([query_sem, doc_sem])

# #       concat > mul > cos
#         # old
#         cos = merge([query_sem, doc_sem], mode="concat")
#         cos = BatchNormalization()(cos)
#         cos = Dropout(0.2)(cos)
#         # concat = Concatenate()([query_sem, doc_sem])
#         # sub = Subtract()([query_sem, doc_sem])
#         # mul = Multiply()([query_sem, doc_sem])
#         # merge_all = merge([concat, sub, mul], mode="concat")


#         pred = Dense(1, activation="sigmoid")(cos)

        self.model = Model(inputs = [query, doc] , outputs = malstm_distance)
        # self.model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics=["accuracy"])
        self.model.compile(optimizer = optimizer, loss = "mean_squared_error", metrics=["accuracy"])
        

        self.encoder = Model(inputs=query, outputs=query_sem)

    def name(self):
        return "binary_lstm" if self.enableLSTM else "binary_%s" % self.PoolMode
    
class DSSM():
    
    def __init__(self, hidden_dim=300, latent_dim=128, num_negatives=1, nb_words=50005, max_len=10, embedding_matrix=None, optimizer=None, enableLSTM=False, enableSeparate=False, PoolMode="max"):

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_negatives = num_negatives
        self.nb_words = nb_words
        self.max_len = max_len
        self.enableSeparate = enableSeparate
        self.enableLSTM = enableLSTM
        self.PoolMode = PoolMode

        # Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
        # The first dimension is None because the queries and documents can vary in length.
        query = Input(shape = (self.max_len,))
        pos_doc = Input(shape = (self.max_len,))
        neg_docs = [Input(shape = (self.max_len,)) for j in range(self.num_negatives)]

        embed_layer = Embedding(nb_words,
                    embedding_matrix.shape[-1],
                    weights=[embedding_matrix],
                    input_length=max_len,
                    name="q_embeding_layer",
                    mask_zero=True if self.enableLSTM else False,
                    trainable=True)
        
        bilstm = GRU(hidden_dim, name='q_gru', return_sequences=False)

        if enableSeparate:
            d_embed_layer = Embedding(nb_words,
                    embedding_matrix.shape[-1],
                    weights=[embedding_matrix],
                    input_length=max_len,
                    name="d_embed_layer",
                    mask_zero=True if self.enableLSTM else False,
                    trainable=True)
            d_bilstm = GRU(hidden_dim, name="d_gru")

        Pooling = GlobalMaxPooling1D() if self.PoolMode == "max" else GlobalAveragePooling1D()

        if self.enableLSTM:
            query_sem = bilstm(embed_layer(query))
            pos_doc_sem = bilstm(embed_layer(pos_doc)) if not enableSeparate else d_bilstm(d_embed_layer(pos_doc))
            neg_doc_sems = [bilstm(embed_layer(neg_doc)) for neg_doc in neg_docs] if not enableSeparate else [d_bilstm(d_embed_layer(neg_doc)) for neg_doc in neg_docs]
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

    def name(self):
        if self.enableLSTM:
            return "dssm_gru2" if self.enableSeparate else "dssm_gru"
        else:
            return "dssm2_%s" % self.PoolMode if self.enableSeparate else "dssm_%s" % self.PoolMode

class VRAE(object):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), kl_weight=1, enableKL=False):
        act = ELU()
        self.kl_weight = kl_weight
        self.enableKL = enableKL


        x = Input(batch_shape=(None, max_len))

        embedding_layer = Embedding(nb_words, embedding_matrix[0].shape[-1], weights=[embedding_matrix],
                                    input_length=max_len, trainable=True, mask_zero=True)
        bilstm = LSTM(dim[0], return_sequences=False, recurrent_dropout=0.2)
        hidden_layer = Dense(dim[0], activation='linear')
        mean_layer = Dense(dim[1])
        var_layer = Dense(dim[1])


        h = embedding_layer(x)
        h = bilstm(h)
        # h = Dropout(0.2)(h)
        h = hidden_layer(h)
        h = act(h)
        # h = Dropout(0.2)(h)
        self.z_mean = mean_layer(h)
        self.z_log_var = var_layer(h)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[-1]), mean=0.,
                                      stddev=1)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling, output_shape=(dim[1],))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        repeated_context = RepeatVector(max_len)
        decoder_h = LSTM(dim[0], return_sequences=True, recurrent_dropout=0.2)
        softmax_layer = Dense(nb_words, activation='softmax')
        decoder_mean = TimeDistributed(softmax_layer, name="rec")
        decoder_kl = TimeDistributed(softmax_layer, name="kl")

        h_decoded = decoder_h(repeated_context(z))
        x_decoded_mean = decoder_mean(h_decoded)        
        x_decoded_kl = decoder_kl(h_decoded)

        
        self.encoder = Model(x, self.z_mean) 
        self.model = Model(x, [x_decoded_mean, x_decoded_kl])
        # if mode in [3,4]:
            # self.model.compile(optimizer=optimizer, loss=["categorical_crossentropy", self.kl_loss])
        # else:
        self.model.compile(optimizer=optimizer, loss=["sparse_categorical_crossentropy", self.kl_loss])

    def name(self):
        return "vrae_kl" if self.enableKL else "vrae"

    def kl_loss(self, x, x_):

        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)

        return self.kl_weight * kl_loss