from Utils import *
from Models import *
from BatchGenerator import *

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Query Similarity Experiments")

    parser.add_argument('--dataset', type=str, help='Choose a dataset. (30M_EN_pos_qd_log, 1M_EN_QQ_log2)')

    parser.add_argument('--model', type=str,
                        help='Model Name: dssm, vae_dssm')

    parser.add_argument('--h', type=int, default=300,
                        help='Hidden Layer Dimension')

    parser.add_argument('--l', type=int, default=128,
                        help='Latent/Encoded Dimension')

    parser.add_argument('--neg', type=int, default=1,
                        help='Number of negative instances to pair with a positive instance.')


    # May need them later
    parser.add_argument('--e', type=int, default=1,
                        help='Number of epochs.')
    parser.add_argument('--b', type=int, default=128,
                        help='Number of batch size.')
    # parser.add_argument('--lr', type=float, default=0.001,
    #                     help='Learning rate.')
    # parser.add_argument('--learner', nargs='?', default='adam',
    #                     help='Specify an optimizer: adagrad, adam, rmsprop, sgd')

    return parser.parse_args()



if __name__ == '__main__':

	args = parse_args()
	print(args)


	model = args.model
	# dataset = args.dataset
    # 30M_EN_pos_qd_log, 1M_EN_QQ_log2
	train_data = args.dataset
	hidden_dim = args.h
	latent_dim = args.l
	num_negatives = args.neg
	epochs = args.e

	out_dir = "/work/data/out/"

# 950000
	train_data_size = {"1M_EN_QQ_log": 950000, "30M_EN_pos_qd_log": 30000000, "100M_query": 30000000}
	eval_every_step = 500


	# LETTER_GRAM_SIZE = 3 # See section 3.2.
	# WINDOW_SIZE = 3 # See section 3.2.
	# TOTAL_LETTER_GRAMS = 50005
	# WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS # See equation (1).
	# FILTER_LENGTH = 1 # We only consider one time step for convolutions.


	nb_words = 50005
	max_len = 10
	batch_size = args.b
	# steps_per_epoch = args.i

	# tokenise_name = "2M_50k_trigram"
	tokenise_name = "50K_BPE"

	# sys.stdout = open('/work/data/out/%s' % model_name, 'w')

	

	# load pre-trained tokeniser
	if "BPE" in tokenise_name:

		sp = spm.SentencePieceProcessor()
		sp.Load('/work/data/bpe/en.wiki.bpe.op50000.model')
		bpe = KeyedVectors.load_word2vec_format("/work/data/bpe/en.wiki.bpe.op50000.d200.w2v.bin", binary=True)
		nb_words = len(bpe.index2word)

		bpe_dict = {bpe.index2word[i]: i for i in range(len(bpe.index2word))}


	elif "trigram" in tokenise_name:

		tokeniser = L3wTransformer()
		tokeniser = tokeniser.load("/work/data/trigram/%s" % tokenise_name)

	# =================================== Initiate Model ==============================================

	if model == "dssm":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words)
	elif model == "vae_dssm":
		run = VAE_DSSM(hidden_dim, latent_dim, nb_words)	
	elif model == "vae_bpe":
		#TODO Frozen or Trainable embedding option
		run = VAE_BPE(hidden_dim, latent_dim, nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True))
		run.initModel(sp, bpe_dict)
	elif model == "kate1":
		run = VarAutoEncoder(nb_words, [hidden_dim, latent_dim])
		run.initModel(sp, bpe_dict)
	elif model == "kate2":
		run = VarAutoEncoder(nb_words, [hidden_dim, latent_dim], 2, "kcomp")
		run.initModel(sp, bpe_dict)

	elif model == "kate1_bpe":
		run = VarAutoEncoder2(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True), [hidden_dim, latent_dim])
		run.initModel(sp, bpe_dict)
	elif model == "kate2_bpe":
		run = VarAutoEncoder2(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True), [hidden_dim, latent_dim], 2, "kcomp")
		run.initModel(sp, bpe_dict)
	elif model == "kate3_bpe":
		run = VarAutoEncoder2(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=False), [hidden_dim, latent_dim], 2, "kcomp")
		run.initModel(sp, bpe_dict)


	model_name = "%s_h%d_l%d_n%d_ml%d_w%d_b%d_%s_%s" % (model, hidden_dim, latent_dim, num_negatives, max_len, nb_words, batch_size, tokenise_name, train_data)

	

	# =================================== Get testing data ==============================================


	df_may, qrel_may = get_test_data("MayFlower")
	df_june, qrel_june = get_test_data("JuneFlower")
	df_july, qrel_july = get_test_data("JulyFlower")

	if model in ["dssm", "vae_dssm", "vae_bpe", "kate1", "kate2", "kate1_bpe", "kate2_bpe", "kate3_bpe"]:
		# Requres 2D inputs
		#  these two condition can be minimised
		if "BPE" in tokenise_name:

			enablePadding = False if model in ["kate1", "kate2"] else True


			q_may = parse_texts_bpe(df_may.q.tolist(), sp, bpe_dict, max_len, enablePadding)
			d_may = parse_texts_bpe(df_may.d.tolist(), sp, bpe_dict, max_len, enablePadding)

			q_june = parse_texts_bpe(df_june.q.tolist(), sp, bpe_dict, max_len, enablePadding)
			d_june = parse_texts_bpe(df_june.d.tolist(), sp, bpe_dict, max_len, enablePadding)

			q_july = parse_texts_bpe(df_july.q.tolist(), sp, bpe_dict, max_len, enablePadding)
			d_july = parse_texts_bpe(df_july.d.tolist(), sp, bpe_dict, max_len, enablePadding)


			# do to one-hot vector 
			if model in ["kate1", "kate2"]:

				# q_may = to_2D_one_hot(q_may, nb_words)
				# d_may = to_2D_one_hot(d_may, nb_words)

				# q_june = to_2D_one_hot(q_june, nb_words)
				# d_june = to_2D_one_hot(d_june, nb_words)

				q_july = to_2D_one_hot(q_july, nb_words)
				d_july = to_2D_one_hot(d_july, nb_words)

			# elif model in ["kate1_bpe", "kate2_bpe"]:
				# q_july = to_categorical(q_july, nb_words)
				# q_july = q_july.reshape(int(q_july.shape[0]/max_len), max_len, nb_words)
				# d_july = to_categorical(d_july, nb_words)
				# d_july = d_july.reshape(int(d_july.shape[0]/max_len), max_len, nb_words)

				# print(q_july.shape, d_july.shape)

		else:

			q_may = to_2D_one_hot(parse_texts(df_may.q.tolist(), tokeniser, max_len, enablePadding), nb_words)
			d_may = to_2D_one_hot(parse_texts(df_may.d.tolist(), tokeniser, max_len, enablePadding), nb_words)

			q_june = to_2D_one_hot(parse_texts(df_june.q.tolist(), tokeniser, max_len, enablePadding), nb_words)
			d_june = to_2D_one_hot(parse_texts(df_june.d.tolist(), tokeniser, max_len, enablePadding), nb_words)

			q_july = to_2D_one_hot(parse_texts(df_july.q.tolist(), tokeniser, max_len, enablePadding), nb_words)
			d_july = to_2D_one_hot(parse_texts(df_july.d.tolist(), tokeniser, max_len, enablePadding), nb_words)


	test_set = [[q_may, d_may, qrel_may, df_may, "MayFlower"], [q_june, d_june, qrel_june, df_june, "JuneFlower"], [q_july, d_july, qrel_july, df_july, "JulyFlower"]]

	# test_set = [[q_july, d_july, qrel_july, df_july, "JulyFlower"]]


	print("============Start Training================")
	cosine = CosineSim(latent_dim)

	best_auc_score = 0

	iterations = int(train_data_size[train_data] / batch_size)
	for epoch in range(epochs):		
		print("------------------Epoch %d---------------------" % epoch)
		# restart the reader thread
		reader = get_reader(train_data, batch_size)

		
		for iteration in range(int(iterations / eval_every_step)):

			run.model.fit_generator(run.batch_generator(reader, train_data, batch_size), steps_per_epoch=eval_every_step, epochs=1, verbose=1)       
			print("----------------%s--Epoch: %d Iteration: %d ---------------------" % (model, epoch, iteration*eval_every_step))
			best_auc_score = evaluate(run, cosine, test_set, best_auc_score, model_name)

	print("Finall Evalutation")
	best_auc_score = evaluate(run, cosine, test_set, best_auc_score, model_name)


		        	