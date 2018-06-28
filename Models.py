import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.layers import Bidirectional, Dense, Embedding, Concatenate, Flatten, Reshape, Input, Lambda, LSTM, merge, GlobalAveragePooling1D, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop
from keras import objectives
from keras import backend as K
from keras.models import Model, load_model
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras_tqdm import TQDMNotebookCallback
from keras_tqdm import TQDMCallback
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import numpy as np
from gensim.models import KeyedVectors





class CosineSim():
    def __init__(self, feature_num):
        q_input = Input(shape=(feature_num,))
        d_input = Input(shape=(feature_num,))

        pred = merge([q_input, d_input], mode="cos")
        self.model = Model([q_input, d_input], pred)


class LSTM_Model():
    def __init__(self, max_len=10, emb_dim=100, nb_words=50000):

        q_input = Input(shape=(max_len,))
        d_input = Input(shape=(max_len,))
        
        emb = Embedding(nb_words, emb_dim, mask_zero=True)

        lstm = LSTM(256)

        self.q_embed = lstm(emb(q_input))
        self.d_embed = lstm(emb(d_input))

        concat = Concatenate()([self.q_embed, self.d_embed])

        pred = Dense(1, activation='sigmoid')(concat)

        self.model = Model(inputs=[q_input, d_input], outputs=pred)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



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
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
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
        decoded = Dense(self.nb_words, activation='sigmoid', name='decoded_mean')(h)
        
        return decoded

    def get_name(self):
        return "vae_dssm_h%d_l%d_w%d_%s" % (self.hidden_dim, self.latent_dim, self.nb_words, self.activation)


    def batch_generator(self, reader, train_data, tokeniser, batch_size, max_len, nb_words):
        while True:
            for df in reader:
                q = df.q.tolist()
                if train_data == "1M_EN_QQ_log":
                    d = [i.split("<sep>")[0] for i in df.d.tolist()]
                else:
                    d = df.d.tolist()
                
                q = pad_sequences(tokeniser.texts_to_sequences(q), maxlen=max_len)
                
                q_one_hot = np.zeros((batch_size, nb_words))
                for i in range(len(q)):
                    q_one_hot[i][q[i]] = 1
                    
                
                yield q_one_hot, q_one_hot



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
        decoded = Dense(self.nb_words, activation='sigmoid', name='decoded_mean')(h)
        
        return decoded

    def get_name(self):
        return "vae_dssm_h%d_l%d_w%d_%s" % (self.hidden_dim, self.latent_dim, self.nb_words, self.activation)


    def batch_generator(self, reader, train_data, tokeniser, batch_size, max_len, nb_words):
        while True:
            for df in reader:
                q = df.q.tolist()
                if train_data == "1M_EN_QQ_log":
                    d = [i.split("<sep>")[0] for i in df.d.tolist()]
                else:
                    d = df.d.tolist()
                
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
    
    def __init__(self, hidden_dim=300, latent_dim=128, num_negatives=1, nb_words=50005):

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_negatives = num_negatives
        self.nb_words = nb_words
        # Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
        # The first dimension is None because the queries and documents can vary in length.
        query = Input(shape = (self.nb_words,))
        pos_doc = Input(shape = (self.nb_words,))
        neg_docs = [Input(shape = (self.nb_words,)) for j in range(self.num_negatives)]

        dense = Dense(self.latent_dim, activation = "tanh")
        query_sem = dense(query)
        # query_sem = Dense(L, activation = "tanh")(query) # See section 3.5.
        # doc_sem = Dense(L, activation = "tanh")
        # shared dense
        doc_sem = dense

        pos_doc_sem = doc_sem(pos_doc)
        neg_doc_sems = [doc_sem(neg_doc) for neg_doc in neg_docs]

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
                if train_data in ["1M_EN_QQ_log", "200_log"]:
                    d = [i.split("<sep>")[0] for i in df.d.tolist()]
                else:
                    d = df.d.tolist()
                
                q = pad_sequences(tokeniser.texts_to_sequences(q), maxlen=max_len)
                d = pad_sequences(tokeniser.texts_to_sequences(d), maxlen=max_len)
                
                q_one_hot = np.zeros((batch_size, nb_words))
                for i in range(len(q)):
                    q_one_hot[i][q[i]] = 1
                    
                d_one_hot = np.zeros((batch_size, nb_words))
                for i in range(len(d)):
                    d_one_hot[i][d[i]] = 1
                    
                    
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