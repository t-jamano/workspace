from Models import *
from Utils import *

clsm = CLSM()

LETTER_GRAM_SIZE = 3 # See section 3.2.
WINDOW_SIZE = 3 # See section 3.2.
TOTAL_LETTER_GRAMS = 5005
WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS # See equation (1).
K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
L = 128 # Dimensionality of latent semantic space. See section 3.5.
J = 1 # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 1 # We only consider one time step for convolutions.

nb_words = 50005
max_len = 10
batch_size = 128


# load pre-trained trigram
tokeniser = L3wTransformer(TOTAL_LETTER_GRAMS)
tokeniser = tokeniser.load("/work/data/trigram/2M_50k_trigram")

df_may, qrel_may = get_test_data("MayFlower")
df_june, qrel_june = get_test_data("JuneFlower")

q_may = to_categorical(parse_texts(df_may.q.tolist(), tokeniser, max_len), nb_words)
d_may = to_categorical(parse_texts(df_may.d.tolist(), tokeniser, max_len), nb_words)

q_may = q_may.reshape(len(df_may), max_len, nb_words)
d_may = d_may.reshape(len(df_may), max_len, nb_words)

q_june = to_categorical(parse_texts(df_june.q.tolist(), tokeniser, max_len), nb_words)
d_june = to_categorical(parse_texts(df_june.d.tolist(), tokeniser, max_len), nb_words)

q_june = q_june.reshape(len(df_june), max_len, nb_words)
d_june = d_june.reshape(len(df_june), max_len, nb_words)


file_dir = '/data/t-mipha/data/agi_encoder/v4/universal/CLICKED_QQ_EN_universal_train_1M.txt'

reader = pd.read_csv(file_dir, chunksize=batch_size, iterator=True, usecols=[0,1], names=["q", "d"], sep="\t", header=None, error_bad_lines=False)

def qq_batch_generator(reader, tokeniser, batch_size, max_len, nb_words):
	while True:
		for df in reader:
		    q = df.q.tolist()
		    d = [i.split("<sep>")[0] for i in df.d.tolist()]
		    
		    q = pad_sequences(tokeniser.texts_to_sequences(q), maxlen=max_len)
		    d = pad_sequences(tokeniser.texts_to_sequences(d), maxlen=max_len)
		    
		    
		    q_one_hot = to_categorical(q, nb_words)   
		    q_one_hot = q_one_hot.reshape(batch_size, max_len, nb_words)
		    
		    d_one_hot = to_categorical(d, nb_words)   
		    d_one_hot = q_one_hot.reshape(batch_size, max_len, nb_words)
		        
		        
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
		    
		    
		    yield [q_one_hot, d_one_hot] + [neg_d_one_hot[j] for j in range(J)], y

        
clsm.model.fit_generator(qq_batch_generator(reader, tokeniser, batch_size, max_len, nb_words), steps_per_epoch=1000, epochs=1, verbose=2, callbacks=[TQDMNotebookCallback()])       
cosine = CosineSim(L)


for q, d, qrel, df in [[q_may, d_may, qrel_may, df_may], [q_june, d_june, qrel_june, df_june]]:
    pred = cosine.model.predict([clsm.encoder.predict(q), clsm.encoder.predict(d)])
    pred = convert_2_trec(df.q.tolist(), df.d.tolist(), pred, False)
    evaluate(qrel, pred)       