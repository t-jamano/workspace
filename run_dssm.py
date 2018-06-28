from Models import *
from Utils import *



LETTER_GRAM_SIZE = 3 # See section 3.2.
WINDOW_SIZE = 3 # See section 3.2.
TOTAL_LETTER_GRAMS = 50005
WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS # See equation (1).
K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
L = 128 # Dimensionality of latent semantic space. See section 3.5.
J = 1 # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 1 # We only consider one time step for convolutions.


nb_words = 50005
max_len = 10
batch_size = 128
steps_per_epoch = 8000
# steps_per_epoch = 10000
# steps_per_epoch = 1

tokenise_name = "2M_50k_trigram"
# train_data = "30M_EN_pos_qd_log"
train_data = "1M_EN_QQ_log2"


model_name = "dssm_h%d_l%d_n%d_i%d_ml%d_w%d_b%d_%s_%s" % (K, L, J, steps_per_epoch, max_len, nb_words, batch_size, tokenise_name, train_data)

# sys.stdout = open('/work/data/out/%s' % model_name, 'w')

dssm = DSSM(J=J)

# load pre-trained trigram
tokeniser = L3wTransformer()
tokeniser = tokeniser.load("/work/data/trigram/%s" % tokenise_name)


df_may, qrel_may = get_test_data("MayFlower")
df_june, qrel_june = get_test_data("JuneFlower")
df_july, qrel_july = get_test_data("JulyFlower")

q_may = to_2D_one_hot(parse_texts(df_may.q.tolist(), tokeniser, max_len), nb_words)
d_may = to_2D_one_hot(parse_texts(df_may.d.tolist(), tokeniser, max_len), nb_words)

q_june = to_2D_one_hot(parse_texts(df_june.q.tolist(), tokeniser, max_len), nb_words)
d_june = to_2D_one_hot(parse_texts(df_june.d.tolist(), tokeniser, max_len), nb_words)

q_july = to_2D_one_hot(parse_texts(df_july.q.tolist(), tokeniser, max_len), nb_words)
d_july = to_2D_one_hot(parse_texts(df_july.d.tolist(), tokeniser, max_len), nb_words)

# file_dir = '/data/t-mipha/data/agi_encoder/v4/universal/CLICKED_QQ_EN_universal_train_1M.txt'
# reader = pd.read_csv(file_dir, chunksize=batch_size, iterator=True, usecols=[0,1], names=["q", "d"], sep="\t", header=None, error_bad_lines=False)

train_data_dir = '/work/data/train_data/%s' % train_data
if train_data == "30M_EN_pos_qd_log":
	reader = pd.read_csv(train_data_dir, chunksize=batch_size, iterator=True, usecols=[0,1,2], names=["label","q", "d"], sep="\t", header=0, error_bad_lines=False)
elif train_data == "1M_EN_QQ_log2":
	reader = pd.read_csv(train_data_dir, chunksize=batch_size, iterator=True, usecols=[0,1], names=["q", "d"], sep="\t", header=None, error_bad_lines=False)



def qq_batch_generator(reader, tokeniser, batch_size, max_len, nb_words):
	while True:
		for df in reader:
		    q = df.q.tolist()
		    if train_data == "1M_EN_QQ_log2":
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
		    neg_d_one_hot = [[] for j in range(J)]
		    for i in range(batch_size):
		        possibilities = list(range(batch_size))
		        possibilities.remove(i)
		        negatives = np.random.choice(possibilities, J, replace = False)
		        for j in range(J):
		            negative = negatives[j]
		            neg_d_one_hot[j].append(d_one_hot[negative].tolist())
		    
		    y = np.zeros((batch_size, J + 1))
		    y[:, 0] = 1
		    
		    for j in range(J):
		        neg_d_one_hot[j] = np.array(neg_d_one_hot[j])
		    
		#         print(q_one_hot.shape, d_one_hot.shape, len(neg_d_one_hot))
		#         print(neg_d_one_hot[0])

		    # negative sampling from randomness
		    # for j in range(J):
		    #     neg_d_one_hot[j] = np.random.randint(2, size=(batch_size, 10, WORD_DEPTH))
		    

		#         q_one_hot = to_categorical(q, nb_words)   
		#         q_one_hot = q_one_hot.reshape(batch_size, max_len, nb_words)
		    
		    
		    yield [q_one_hot, d_one_hot] + [neg_d_one_hot[j] for j in range(J)], y

        
dssm.model.fit_generator(qq_batch_generator(reader, tokeniser, batch_size, max_len, nb_words), steps_per_epoch=steps_per_epoch, epochs=1, verbose=1)       
cosine = CosineSim(L)

# Save the weights
dssm.model.save_weights('/work/data/models/%s.h5' % model_name)

# Save the model architecture
with open('/work/data/models/%s.json' % model_name, 'w') as f:
    f.write(dssm.model.to_json())



for q, d, qrel, df, test_data in [[q_may, d_may, qrel_may, df_may, "MayFlower"], [q_june, d_june, qrel_june, df_june, "JuneFlower"], [q_july, d_july, qrel_july, df_july, "JulyFlower"]]:
    
    pred = cosine.model.predict([dssm.encoder.predict(q), dssm.encoder.predict(d)])
    print(test_data)
    if test_data in ["MayFlower", "JuneFlower"]:
        pred = convert_2_trec(df.q.tolist(), df.d.tolist(), pred, False)
        ndcg_score, map_score = evaluate(qrel, pred)
        print("NDCG: %f" % ndcg_score)
        print("MAP: %f" % map_score)
        save_pkl(pred, '/work/data/res/%s_%s_%f_%f.pkl' % (model_name, test_data, ndcg_score, map_score))  

    elif test_data in ["JulyFlower"]:
        auc_score = auc(qrel, pred.flatten())
        print("AUC: %f" % auc_score)
        save_pkl(pred.flatten(), '/work/data/res/%s_%s.pkl' % (model_name, test_data, auc_score))