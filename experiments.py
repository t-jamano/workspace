from Utils import *
from Models import *
from BatchGenerator import *
import datetime
from time import time

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Query Similarity Experiments")

    parser.add_argument('--dataset', type=str, help='Choose a dataset. (30M_EN_pos_qd_log, 1M_EN_QQ_log)')

    parser.add_argument('--model', type=str,
                        help='Model Name: dssm, vae_dssm')

    parser.add_argument('--h', type=int, default=300,
                        help='Hidden Layer Dimension')

    parser.add_argument('--l', type=int, default=128,
                        help='Latent/Encoded Dimension')

    parser.add_argument('--neg', type=int, default=1,
                        help='Number of negative instances to pair with a positive instance.')

    parser.add_argument('--path', type=str, default="/work/",
                        help='Path to dir')

    # May need them later
    parser.add_argument('--e', type=int, default=1,
                        help='Number of epochs.')
    parser.add_argument('--b', type=int, default=128,
                        help='Number of batch size.')
    parser.add_argument('--a', type=float, default=0.5,
                        help='Alpha param')

    parser.add_argument('--o', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')



    return parser.parse_args()



if __name__ == '__main__':




	args = parse_args()
	print(args)

	date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

	path = args.path
	model = args.model
	# dataset = args.dataset
    # 30M_EN_pos_qd_log, 1M_EN_QQ_log2
	train_data = args.dataset
	hidden_dim = args.h
	latent_dim = args.l
	num_negatives = args.neg
	epochs = args.e
	alpha = args.a

	out_dir = "%sdata/out/" % path

# 950000
	# train_data_size = {"1M_EN_QQ_log": 950000, "30M_EN_pos_qd_log": 20000000, "100M_query": 10000000, "30M_QD.txt": 20000000}

	train_data_size = {"1M_EN_QQ_log": 950000, "30M_EN_pos_qd_log": 25000000, "100M_query": 10000000, "30M_QD_lower2.txt": 25000000}
	eval_every_step = 1
	# eval_every_step = 10


	# LETTER_GRAM_SIZE = 3 # See section 3.2.
	# WINDOW_SIZE = 3 # See section 3.2.
	# TOTAL_LETTER_GRAMS = 50005
	# WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS # See equation (1).
	# FILTER_LENGTH = 1 # We only consider one time step for convolutions.


	nb_words = 50005
	max_len = 10
	max_len2 = 15
	batch_size = args.b
	# steps_per_epoch = args.i

	# tokenise_name = "2M_50k_trigram"
	tokenise_name = "50K_BPE"

	# sys.stdout = open('/work/data/out/%s' % model_name, 'w')

	optimizer=args.o

	if optimizer == "adadelta":
		optimizer = Adadelta(lr=2.)
	

	# load pre-trained tokeniser
	if "BPE" in tokenise_name:

		sp = spm.SentencePieceProcessor()
		sp.Load('%sdata/bpe/en.wiki.bpe.op50000.model' % path)
		bpe = KeyedVectors.load_word2vec_format("%sdata/bpe/en.wiki.bpe.op50000.d200.w2v.bin" % path, binary=True)
		nb_words = len(bpe.index2word)

		bpe_dict = {bpe.index2word[i]: i for i in range(len(bpe.index2word))}


	elif "trigram" in tokenise_name:

		tokeniser = L3wTransformer()
		tokeniser = tokeniser.load("%sdata/trigram/%s" % (path,tokenise_name))

	# =================================== Initiate Model ==============================================

	if model == "dssm":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, bpe.get_keras_embedding(True), optimizer=optimizer)
		run.initModel(sp, bpe_dict)
	elif model == "bilstm":
		run = LSTM_Model(hidden_dim, latent_dim, nb_words=nb_words, max_len=max_len, emb=bpe.get_keras_embedding(True))
		run.initModel(sp, bpe_dict)
	elif model == "bilstm2":
		run = BiLSTM(hidden_dim, latent_dim, nb_words=nb_words, q_max_len=max_len, d_max_len=max_len2, emb=bpe.get_keras_embedding(True))
		run.initModel(sp, bpe_dict)
	elif model == "bilstm2_cos":
		run = BiLSTM(hidden_dim, latent_dim, nb_words=nb_words, q_max_len=max_len, d_max_len=max_len2, emb=bpe.get_keras_embedding(True), mode="cos")
		run.initModel(sp, bpe_dict)
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
	elif model == "kate2_bpeg":
		run = VarAutoEncoder2(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True), [hidden_dim, latent_dim], 2, "kcomp", enableGAN=True)
		run.initModel(sp, bpe_dict)
	elif model == "kate2_bpe_adam":
		run = VarAutoEncoder2(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True), [hidden_dim, latent_dim], 2, "kcomp", optimizer=optimizer)
		run.initModel(sp, bpe_dict)

	elif model == "aae":
		run = AAE(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True), [hidden_dim, latent_dim], 2, "kcomp")
		run.initModel(sp, bpe_dict)

	elif model == "kate1_qd":
		run = VarAutoEncoderQD(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True), [hidden_dim, latent_dim], 2)
		run.initModel(sp, bpe_dict)
	elif model == "kate2_qd":
		run = VarAutoEncoderQD(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True), [hidden_dim, latent_dim], 2, "kcomp", alpha=alpha, optimizer=optimizer)
		run.initModel(sp, bpe_dict)
	elif model == "kate2_qd2":
		run = VarAutoEncoderQD2(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True), [hidden_dim, latent_dim], 2, "kcomp", alpha=alpha, optimizer=optimizer)
		run.initModel(sp, bpe_dict)
	elif model == "kate2_qd3":
		run = VarAutoEncoderQD3(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True), [hidden_dim, latent_dim], 2, "kcomp", alpha=alpha, optimizer=optimizer)
		run.initModel(sp, bpe_dict)
	elif model == "kate2_qdc":
		run = VarAutoEncoderQD(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True), [hidden_dim, latent_dim], 2, "kcomp", enableCross=True)
		run.initModel(sp, bpe_dict)
	elif model == "kate2_qdm":
		run = VarAutoEncoderQD(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True), [hidden_dim, latent_dim], 2, "kcomp", enableMemory=True)
		run.initModel(sp, bpe_dict)
	elif model in ["kate2_qdg1", "kate2_qdg2"]:
		run = VarAutoEncoderQD(nb_words, max_len, bpe.get_keras_embedding(train_embeddings=True), [hidden_dim, latent_dim], 2, "kcomp", enableGAN=True)
		run.initModel(sp, bpe_dict)
		if model == "kate2_qdg2":
			run.discriminator.trainable = False
		run.encoder._make_predict_function()
		graph = tf.get_default_graph()
			



	model_name = "%s_h%d_l%d_n%d_ml%d_w%d_b%d_a%.1f_%s_%s_%s_%s" % (model, hidden_dim, latent_dim, num_negatives, max_len, nb_words, batch_size, alpha, optimizer, tokenise_name, train_data, date_time)

	

	# =================================== Get testing data ==============================================


	df_may, qrel_may = get_test_data("MayFlower", path)
	df_june, qrel_june = get_test_data("JuneFlower", path)
	df_july, qrel_july = get_test_data("JulyFlower", path)

	enablePadding = True

	q_may = parse_texts_bpe(df_may.q.tolist(), sp, bpe_dict, max_len, enablePadding)
	d_may = parse_texts_bpe(df_may.d.tolist(), sp, bpe_dict, max_len, enablePadding)

	q_june = parse_texts_bpe(df_june.q.tolist(), sp, bpe_dict, max_len, enablePadding)
	d_june = parse_texts_bpe(df_june.d.tolist(), sp, bpe_dict, max_len, enablePadding)

	q_july = parse_texts_bpe(df_july.q.tolist(), sp, bpe_dict, max_len, enablePadding)
	d_july = parse_texts_bpe(df_july.d.tolist(), sp, bpe_dict, max_len, enablePadding)


	test_set = [[q_may, d_may, qrel_may, df_may, "MayFlower"], [q_june, d_june, qrel_june, df_june, "JuneFlower"], [q_july, d_july, qrel_july, df_july, "JulyFlower"]]

	# test_set = [[q_july, d_july, qrel_july, df_july, "JulyFlower"]]


	print("============Start Training================")
	cosine = CosineSim(latent_dim)

	best_auc_score = 0

	iterations = int(train_data_size[train_data] / batch_size)
	for epoch in range(epochs):		
		# restart the reader thread
		reader = get_reader(train_data, batch_size, path)
		reader2 = get_reader(train_data, batch_size, path)

		
		for iteration in range(int(iterations / eval_every_step)):

			
			try:
				if model in ["kate2_qdg1", "kate2_qdg2", "kate2bpeg"]:
					t1 = time()
					hist = run.model.fit_generator(run.batch_generator(reader, train_data, batch_size), steps_per_epoch=eval_every_step, epochs=1, verbose=0)       
					hist_dis = run.discriminator.fit_generator(run.batch_GAN_generator(reader2, train_data, batch_size, graph), steps_per_epoch=eval_every_step, epochs=1, verbose=0)       
					t2 = time()
					losses = ', '.join([ "%s = %.4f" % (k, hist.history[k][-1]) for k in hist.history] + [ "%s = %.4f" % (k, hist_dis.history[k][-1]) for k in hist_dis.history])
				
				else:
					t1 = time()

					# if "kate2_qd2" in model:
					# 	pass
					# elif "kate2_qd" in model:
					# 	max_pred_weight = 150
					# 	kl_inc = 1.0 / 5000
					# 	pred_inc = 0.1
					# 	run.kl_weight = min(run.kl_weight + kl_inc, 1.0)
					# 	run.pred_weight = min(run.pred_weight + pred_inc, max_pred_weight)

					hist = run.model.fit_generator(run.batch_generator(reader, train_data, batch_size), steps_per_epoch=eval_every_step, epochs=1, verbose=0)       
					t2 = time()
					losses = ', '.join([ "%s = %.4f" % (k, hist.history[k][-1]) for k in hist.history])

				may_ndcg, june_ndcg, july_auc = evaluate(run, cosine, test_set, model_name)

				

				print_output = '%s_a%.1f, Epoch %d, Iteration %d, [%.1f s], May = %.4f, June = %.4f, July = %.4f, Loss = %.4f, [%.1f s] \n' % (model, alpha, epoch, (iteration+1)*eval_every_step, t2-t1, may_ndcg, june_ndcg, july_auc, hist.history['loss'][-1], time()-t2)
				file_output = '%s_a%.1f, Epoch %d, Iteration %d, [%.1f s], May = %.4f, June = %.4f, July = %.4f, %s, [%.1f s] \n' % (model, alpha, epoch, (iteration+1)*eval_every_step, t2-t1, may_ndcg, june_ndcg, july_auc, losses, time()-t2)

				print(print_output)
				with open("%sdata/out/%s" % (path,model_name), "a") as myfile:
					myfile.write(file_output)



				if july_auc > best_auc_score:
					best_auc_score = july_auc
					run.model.save('%sdata/models/%s.h5' % (path,model_name), overwrite=True)
					run.encoder.save('%sdata/models/%s.encoder.h5' % (path,model_name), overwrite=True)


			except Exception as e:
				print(e)
				pass

	print("Finall Evalutation")
	may_ndcg, june_ndcg, july_auc = evaluate(run, cosine, test_set, model_name)
	str_output = 'May = %.4f, June = %.4f, July = %.4f \n' % (may_ndcg, june_ndcg, july_auc)
	print(str_output)
	with open("%sdata/out/%s" % (path,model_name), "a") as myfile:
		myfile.write(str_output)

		        	