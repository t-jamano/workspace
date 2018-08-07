
# coding: utf-8

# # Generating Sentences from a Continuous Space
# ### Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz, Samy Bengio

# In[1]:

from Models import *
from Utils import *
from FastModels import *
import warnings
# warnings.filterwarnings('ignore')


# In[2]:

from gensim.models import KeyedVectors
import sentencepiece as spm

max_len = 10
enablePadding = True

sp = spm.SentencePieceProcessor()
sp.Load('/work/data/bpe/en.wiki.bpe.op50000.model')
bpe = KeyedVectors.load_word2vec_format("/work/data/bpe/en.wiki.bpe.op50000.d200.w2v.bin", binary=True)
bpe.index2word = [''] + bpe.index2word # add empty string
nb_words = len(bpe.index2word)
# word2index
bpe_dict = {bpe.index2word[i]: i for i in range(len(bpe.index2word))}
# construct embedding_matrix
embedding_matrix = np.concatenate([np.zeros((1, bpe.vector_size)), bpe.vectors]) # add zero vector for empty string (i.e. used for padding)

embedding_layer = Embedding(nb_words,
                    embedding_matrix.shape[-1],
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=True)


# In[3]:

df_may, qrel_may = get_test_data("MayFlower", "/work/")
df_june, qrel_june = get_test_data("JuneFlower", "/work/")
df_july, qrel_july = get_test_data("JulyFlower", "/work/")

q_may = parse_texts_bpe(df_may.q.tolist(), sp, bpe_dict, max_len, enablePadding)
d_may = parse_texts_bpe(df_may.d.tolist(), sp, bpe_dict, max_len, enablePadding)

q_june = parse_texts_bpe(df_june.q.tolist(), sp, bpe_dict, max_len, enablePadding)
d_june = parse_texts_bpe(df_june.d.tolist(), sp, bpe_dict, max_len, enablePadding)

q_july = parse_texts_bpe(df_july.q.tolist(), sp, bpe_dict, max_len, enablePadding, "pre")
d_july = parse_texts_bpe(df_july.d.tolist(), sp, bpe_dict, max_len, enablePadding, "pre")

test_set = [[q_may, d_may, qrel_may, df_may, "MayFlower"], [q_june, d_june, qrel_june, df_june, "JuneFlower"], [q_july, d_july, qrel_july, df_july, "JulyFlower"]]


# In[4]:

# df = pd.read_csv("/work/data/train_data/QueryQueryLog", names=["q", "d", "label"], sep="\t", header=None, error_bad_lines=False)
# df = pd.read_csv("/work/data/train_data/QueryLog", nrows=100000, names=["q"], sep="\t", header=None, error_bad_lines=False)


# In[5]:

# q_df = parse_texts_bpe(df.q.tolist(), sp, bpe_dict, max_len, enablePadding)
# d_df = parse_texts_bpe(df.d.tolist(), sp, bpe_dict, max_len, enablePadding)


# In[6]:

# np.save("/work/data/train_data/QueryQueryLog.q.npy", q_df)


# In[7]:

q_df = np.load("/work/data/train_data/QueryQueryLog.q.npy")
d_df = np.load("/work/data/train_data/QueryQueryLog.d.npy")


# In[8]:

def eval(epoch, logs):
#     print(logs)
#     loss = logs.get("loss")
#     val_loss = logs.get("val_loss")
    
#     rec_loss = logs.get("time_distributed_1_loss_1")
#     kl_loss = logs.get("time_distributed_1_loss_2")

#     print(epoch, loss, val_loss)
#     aucs = []
#     for encoder in [run.encoder, run.encoder2]:
#         q = encoder.predict(q_july)
#         d = encoder.predict(d_july)

#         pred = cosine.model.predict([q, d])
        
#         aucs.append(auc(qrel_july, pred.flatten()))
    
    cosine = CosineSim(200)
    print(evaluate(run, cosine, test_set))
#     print(loss, rec_loss, kl_loss, val_loss, evaluate(run, cosine, test_set))


# In[9]:

def generate_output(model, bpe, x, nrows=None, idx=None):

    gen_x = np.argmax(model.predict(x), axis=-1) if idx == None else np.argmax(model.predict([x,x])[idx], axis=-1)

    bleu = []
    results = ""
    for i, k in zip(gen_x, x):
        gen_x = " ".join([bpe.index2word[t] for t in i])
        real_x = " ".join([bpe.index2word[t] for t in k])

        bleu.append(sentence_bleu(real_x, gen_x))
        
        real_x = real_x.replace("▁the", "")
        real_x = real_x.replace("▁","")
        gen_x = gen_x.replace("▁the", "")
        gen_x = gen_x.replace("▁","")
        if nrows != None:
            if nrows == 0:
                break
            else:
                nrows = nrows - 1
        
        results = results + "%s : %s\n\n" % (real_x, gen_x)
    print("BLEU: %.4f" % np.mean(bleu))
    print(results)

# generate_output(run.model, bpe, q_july,idx=0)


# In[10]:

class SeqVAE(object):
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=RMSprop(), word_dropout_prob=0.5, kl_weight=0):
        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.kl_weight = K.variable(kl_weight)
        self.word_dropout_prob = word_dropout_prob
        
        self.build()
        
    def build(self):
                
#       Encoder
        
        e_input = Input(shape=(self.max_len,))
        e_mask = Masking(mask_value=0)

        embedding_layer = Embedding(nb_words,
                            embedding_matrix.shape[-1],
                            weights=[self.embedding_matrix],
                            input_length=max_len,
                            trainable=True)


        e_lstm = GRU(self.dim[0], return_state=True)
        
#         h, state_h, state_c = e_lstm(e_emb(e_mask(e_input)))#         
        _, h = e_lstm(e_mask(embedding_layer(e_input)))

        
        mean = Dense(self.dim[1])
        var = Dense(self.dim[1])
        
#         self.state_h_mean = mean(state_h)
#         self.state_h_log_var = var(state_h)
        
#         self.state_c_mean = mean(state_c)
#         self.state_c_log_var = var(state_c)
        
        self.h_mean = mean(h)
        self.h_log_var = var(h)
        
#         state_h_z = Lambda(self.sampling)([self.state_h_mean, self.state_h_log_var])     
#         state_c_z = Lambda(self.sampling)([self.state_c_mean, self.state_c_log_var])
        z = Lambda(self.sampling)([self.h_mean, self.h_log_var])

#         self.encoder = Model(e_input, self.state_h_mean)
        self.encoder = Model(e_input, self.h_mean)
        self.encoder2 = Model(e_input, z)

        
#       Decoder

        d_input = Input(shape=(self.max_len,))
        d_latent2hidden = Dense(self.dim[0])
        d_lstm = GRU(self.dim[0], return_sequences=True)
        d_output2vocab = Dense(self.nb_words, activation="softmax", name="dec")


#         state_h_z = d_latent2hidden(state_h_z)
#         state_c_z = d_latent2hidden(state_c_z)
        h_z = d_latent2hidden(z)
        
        d_embed_input = embedding_layer(Dropout(self.word_dropout_prob)(d_input))
#         d_embed_input = e_emb(d_input)

#         outputs = d_lstm(d_embed_input, initial_state=[state_h_z, state_c_z])        
        outputs = d_lstm(d_embed_input, initial_state=[h_z])

        pred = d_output2vocab(outputs)

        
        
#       VAE model
        self.model = Model(inputs=[e_input, d_input], outputs=[pred, pred])
        self.model.compile(optimizer=self.optimizer, loss=["sparse_categorical_crossentropy", self.kl_loss])




    def name(self):
        
        return "seqvae"
    
    def kl_loss(self, x, x_):
#         x = K.flatten(x)
#         x_ = K.flatten(x_)
#         x = tf.cast(x, tf.int32)

#         rec_loss = objectives.sparse_categorical_crossentropy(x,x_)

#         state_h_kl_loss = - 0.5 * K.sum(1 + self.state_h_log_var - K.square(self.state_h_mean) - K.exp(self.state_h_log_var), axis=-1)
#         state_c_kl_loss = - 0.5 * K.sum(1 + self.state_c_log_var - K.square(self.state_c_mean) - K.exp(self.state_c_log_var), axis=-1)
        
#         return (self.kl_weight * state_h_kl_loss) + (self.kl_weight * state_c_kl_loss)
        kl_loss = - 0.5 * K.sum(1 + self.h_log_var - K.square(self.h_mean) - K.exp(self.h_log_var), axis=-1)
        return (self.kl_weight * kl_loss) 

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,                                  stddev=1)
    
        return z_mean + K.exp(z_log_var / 2) * epsilon 
    
    def nosampling(self, args):
        z_mean, z_log_var = args
        return z_mean + K.exp(z_log_var / 2)


# In[11]:

# del run
run = SeqVAE(nb_words, max_len, embedding_matrix, [200,200], optimizer=Adam(), word_dropout_prob=0.5, kl_weight=0)


# In[12]:

#     def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=RMSprop(), word_dropout_prob=0.5, kl_weight=0):


# In[ ]:

q_df_small = q_df[:10000]
d_df_small = d_df[:10000]


# In[ ]:

from livelossplot import PlotLossesKeras


x_train = np.concatenate([q_df_small, d_df_small])
y_train = np.expand_dims(x_train, axis=-1)
callbacks = []

for i in range(50):
#     idx = np.arange(len(q_df_small))
#     shuffle(idx)
#     x_train = [q_df_small, d_df_small, d_df_small[idx]]
#     y_train = np.zeros((len(q_df_small), 2))
#     y_train[:, 0] = 1

    
    # 
    print(i)
    hist = run.model.fit([x_train, x_train], [y_train, y_train], validation_split=0.2, verbose=2, batch_size=256, epochs=1, callbacks=callbacks)
    
    run.kl_weight += 0.1
    # run.model.compile(optimizer=run.optimizer, loss=["sparse_categorical_crossentropy", run.kl_loss])

    
    # print(hist.history['loss'], hist.history['val_loss'])

#     if i % 5 == 0:
#         cosine = CosineSim(200)
#         q = run.encoder.predict(q_july)
#         d = run.encoder.predict(d_july)
#         pred = cosine.model.predict([q, d])
#         print(auc(qrel_july, pred.flatten()))
    


# In[ ]:

cosine = CosineSim(200)
q = run.encoder.predict(q_july)
d = run.encoder.predict(d_july)
pred = cosine.model.predict([q, d])
print( auc(qrel_july, pred.flatten()))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



