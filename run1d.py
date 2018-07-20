# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from keras_tqdm import TQDMNotebookCallback
from keras_tqdm import TQDMCallback

from Utils import *
from Models import *

q_train, d_train, y_train, q_test, d_test, y_test, qrel, df, df_test, max_len, nb_words, tokeniser = get_data(sup_train_data="1M_qq_log", test_data="MayFlower", tokenize= "trigram")


q_train_one_hot = np.zeros((q_train.shape[0], nb_words))
for i, j in zip(q_train_one_hot, q_train):
    i[j] = 1


class VRAE_1D():
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
        h = LSTM(200, return_sequences=False, name='dec_lstm_1')(repeated_context)
#         decoded = TimeDistributed(Dense(vocab_size, activation='softmax'), name='decoded_mean')(h)
        decoded = Dense(vocab_size, activation='softmax')(h)

        return decoded

print("Start training the model")
latent_dim = 100
vae = VRAE_1D(nb_words, max_len, latent_dim)  
cosine = CosineSim(latent_dim)
for i in range(2):

    vae.model.fit(q_train, q_train_one_hot, batch_size=128, verbose=2, epochs=1, callbacks=[TQDMCallback()])
    # running python script
    #     vae.model.fit(q_train, q_train_one_hot, batch_size=128, verbose=2, epochs=1, callbacks=[TQDMCallback()])

    pred = cosine.model.predict([vae.encoder.predict(q_test), vae.encoder.predict(d_test)])
    pred = convert_2_trec(df_test.q.tolist(), df_test.d.tolist(), pred, False)
    evaluate(qrel, pred)
