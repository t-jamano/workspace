
# coding: utf-8

# In[2]:

from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import ELU
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from scipy import spatial
import tensorflow as tf
import pandas as pd
import numpy as np
import codecs
import csv
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[60]:

batch_size = 1000
max_len = 5
emb_dim = 50
NB_WORDS = 100000
latent_dim = 32
intermediate_dim = 96
epsilon_std = 1.0
num_sampled=500
act = ELU()


# In[6]:

get_ipython().run_cell_magic('time', '', 'import pandas as pd\n\nTRAIN_DATA_FILE = \'/data/chzho/deepqts/train_data/unifiedclick/join_oneyearsample_2B_training_all_top10\'\nnum_read_row = 1000000\ndf = pd.read_csv(TRAIN_DATA_FILE, sep="\\t", usecols=[0,1,3], names=[\'label\', \'q\', \'d\'], header=None , error_bad_lines=False, nrows=num_read_row)\ndf = df.dropna()\n\nTEST_DATA_FILE = \'/data/chzho/deepqts/test_data/uhrs/unified/uhrs_do_10\'\ndf_qd = pd.read_csv(TEST_DATA_FILE, sep="\\t", usecols=[0,1,3,5], names=[\'label\', \'q\', \'d\', \'market\'], header=None , error_bad_lines=False)\ndf_qd = df_qd.dropna()\ndf_qd = df_qd[df_qd.market == "en-US"]\n\nTEST_DATA_FILE = \'/data/chzho/deepqts/test_data/julyflower/julyflower_original.tsv\'\ndf_qq = pd.read_csv(TEST_DATA_FILE, sep="\\t", names=[\'q\', \'d\', \'label\'], header=None , error_bad_lines=False)\ndf_qq = df_qq.dropna()')


# In[9]:

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

texts = df_qd.q.tolist() + df_qd.d.tolist() + df_qq.q.tolist() + df_qq.d.tolist() + df.q.tolist() + df.d.tolist()

tokenizer = Tokenizer(NB_WORDS)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index #the dict values start from 1 so this is fine with zeropadding
index2word = {v: k for k, v in word_index.items()}
print('Found %s unique tokens' % len(word_index))
NB_WORDS = (min(tokenizer.num_words, len(word_index)) + 1 ) #+1 for zero padding
print('Number of Vocab: %d' % NB_WORDS)



# In[66]:

uns_q = pad_sequences(tokenizer.texts_to_sequences(df.q.tolist()), maxlen=max_len)
uns_d = pad_sequences(tokenizer.texts_to_sequences(df.d.tolist()), maxlen=max_len)


# In[67]:



#y = Input(batch_shape=(None, max_len, NB_WORDS))
x = Input(batch_shape=(None, max_len))
x_embed = Embedding(NB_WORDS, emb_dim, input_length=max_len)(x)
h = Bidirectional(LSTM(intermediate_dim, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(x_embed)
#h = Bidirectional(LSTM(intermediate_dim, return_sequences=False), merge_mode='concat')(h)
h = Dropout(0.2)(h)
h = Dense(intermediate_dim, activation='linear')(h)
h = act(h)
h = Dropout(0.2)(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
#     batch = K.shape(z_mean)[0]
#     dim = K.int_shape(z_mean)[1]
    
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
repeated_context = RepeatVector(max_len)
decoder_h = LSTM(intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
decoder_mean = TimeDistributed(Dense(NB_WORDS, activation='linear'))#softmax is applied in the seq2seqloss by tf
h_decoded = decoder_h(repeated_context(z))
x_decoded_mean = decoder_mean(h_decoded)


# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

# Custom VAE loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.target_weights = tf.constant(np.ones((batch_size, max_len)), tf.float32)

    def vae_loss(self, x, x_decoded_mean):
        #xent_loss = K.sum(metrics.categorical_crossentropy(x, x_decoded_mean), axis=-1)
        labels = tf.cast(x, tf.int32)
        xent_loss = K.sum(tf.contrib.seq2seq.sequence_loss(x_decoded_mean, labels, 
                                                     weights=self.target_weights,
                                                     average_across_timesteps=False,
                                                     average_across_batch=False), axis=-1)
                                                     #softmax_loss_function=softmax_loss_f), axis=-1)#, uncomment for sampled doftmax
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        print(x.shape, x_decoded_mean.shape)
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # we don't use this output, but it has to have the correct shape:
        return K.ones_like(x)

loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, [loss_layer])
opt = Adam(lr=0.01) #SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
vae.compile(optimizer='adam', loss=[zero_loss])
# vae.summary()


# In[ ]:

vae.fit(uns_q, uns_q,  verbose=2, batch_size=batch_size, epochs=10)


# In[ ]:

# vae.fit(uns_d, uns_d,  verbose=2, batch_size=batch_size, epochs=10)

