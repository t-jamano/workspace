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

    parser.add_argument('--i', type=int, default=7000,
                        help='iteration')

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


	# LETTER_GRAM_SIZE = 3 # See section 3.2.
	# WINDOW_SIZE = 3 # See section 3.2.
	# TOTAL_LETTER_GRAMS = 50005
	# WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS # See equation (1).
	# FILTER_LENGTH = 1 # We only consider one time step for convolutions.


	nb_words = 50005
	max_len = 10
	batch_size = args.b
	steps_per_epoch = args.i

	# tokenise_name = "2M_50k_trigram"
	tokenise_name = "50K_BPE"


	if model == "dssm":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words)
	elif model == "vae_dssm":
		run = VAE_DSSM(hidden_dim, latent_dim, nb_words)	


	model_name = "%s_h%d_l%d_n%d_i%d_ml%d_w%d_b%d_%s_%s" % (model, hidden_dim, latent_dim, num_negatives, steps_per_epoch, max_len, nb_words, batch_size, tokenise_name, train_data)

	# sys.stdout = open('/work/data/out/%s' % model_name, 'w')

	

	# load pre-trained tokeniser
	if "BPE" in tokenise_name:

		sp = spm.SentencePieceProcessor()
		sp.Load('/work/data/bpe/en.wiki.bpe.op50000.model')
		bpe = KeyedVectors.load_word2vec_format("/work/data/bpe/en.wiki.bpe.op50000.d200.w2v.bin", binary=True)

	elif "trigram" in tokenise_name:

		tokeniser = L3wTransformer()
		tokeniser = tokeniser.load("/work/data/trigram/%s" % tokenise_name)


	df_may, qrel_may = get_test_data("MayFlower")
	df_june, qrel_june = get_test_data("JuneFlower")
	df_july, qrel_july = get_test_data("JulyFlower")

	if model in ["dssm", "vae_dssm"]:
		# Requres 2D inputs

		if "BPE" in tokenise_name:

			q_may = to_2D_one_hot(parse_texts_bpe(df_may.q.tolist(), sp, bpe, max_len), nb_words)
			d_may = to_2D_one_hot(parse_texts_bpe(df_may.d.tolist(), sp, bpe, max_len), nb_words)

			q_june = to_2D_one_hot(parse_texts_bpe(df_june.q.tolist(), sp, bpe, max_len), nb_words)
			d_june = to_2D_one_hot(parse_texts_bpe(df_june.d.tolist(), sp, bpe, max_len), nb_words)

			q_july = to_2D_one_hot(parse_texts_bpe(df_july.q.tolist(), sp, bpe, max_len), nb_words)
			d_july = to_2D_one_hot(parse_texts_bpe(df_july.d.tolist(), sp, bpe, max_len), nb_words)

		else:

			q_may = to_2D_one_hot(parse_texts(df_may.q.tolist(), tokeniser, max_len), nb_words)
			d_may = to_2D_one_hot(parse_texts(df_may.d.tolist(), tokeniser, max_len), nb_words)

			q_june = to_2D_one_hot(parse_texts(df_june.q.tolist(), tokeniser, max_len), nb_words)
			d_june = to_2D_one_hot(parse_texts(df_june.d.tolist(), tokeniser, max_len), nb_words)

			q_july = to_2D_one_hot(parse_texts(df_july.q.tolist(), tokeniser, max_len), nb_words)
			d_july = to_2D_one_hot(parse_texts(df_july.d.tolist(), tokeniser, max_len), nb_words)




	

	print("============Start Training================")
	cosine = CosineSim(latent_dim)

	best_auc_score = 0


	for i in range(epochs):		
		print("------------------Epoch %d---------------------" %i)
		reader = get_reader(train_data, batch_size)

		
		run.model.fit_generator(run.batch_generator(reader, train_data, tokeniser, batch_size, max_len, nb_words), steps_per_epoch=steps_per_epoch, epochs=1, verbose=1)       


		for q, d, qrel, df, test_data in [[q_may, d_may, qrel_may, df_may, "MayFlower"], [q_june, d_june, qrel_june, df_june, "JuneFlower"], [q_july, d_july, qrel_july, df_july, "JulyFlower"]]:
		    
		    pred = cosine.model.predict([run.encoder.predict(q), run.encoder.predict(d)])
		    print(test_data)
		    if test_data in ["MayFlower", "JuneFlower"]:
		        pred = convert_2_trec(df.q.tolist(), df.d.tolist(), pred, False)
		        ndcg_score, map_score = evaluate(qrel, pred)
		        print("NDCG: %f" % ndcg_score)
		        print("MAP: %f" % map_score)
		        save_pkl(pred, '/work/data/res/%s_%s_%f_%f.pkl' % (model_name, test_data, ndcg_score, map_score))  
		        with open("%s%s\n" % (out_dir,model_name), "a") as myfile:
		        	myfile.write("NDCG: %f" % ndcg_score)
		        	myfile.write("MAP: %f" % map_score)

		    elif test_data in ["JulyFlower"]:
		        auc_score = auc(qrel, pred.flatten())
		        print("AUC: %f" % auc_score)
		        save_pkl(pred.flatten(), '/work/data/res/%s_%s_%f.pkl' % (model_name, test_data, auc_score))
		        with open("%s%s\n" % (out_dir,model_name), "a") as myfile:
		        	myfile.write("AUC: %f" % auc_score)


		        if auc_score > best_auc_score:
		        	best_auc_score = auc_score
		        	run.model.save('/work/data/models/%s.h5' % model_name)
		        	run.encoder.save('/work/data/models/%s.encoder.h5' % model_name)

		        	