from Utils import *
from Models import *
from FastModels import *
from AdversarialModels import *
from BatchGenerator import *
from BOW_Models import *
from SemiSupervisedModels import *
import datetime
import math


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Query Similarity Experiments")

    parser.add_argument('--dataset', type=str, help='Choose a dataset. (30M_EN_pos_qd_log, 1M_EN_QQ_log)')

    parser.add_argument('--model', type=str,
                        help='Model Name: dssm, vae_dssm')

    parser.add_argument('--h', type=int, default=200,
                        help='Hidden Layer Dimension')

    parser.add_argument('--l', type=int, default=100,
                        help='Latent/Encoded Dimension')

    parser.add_argument('--neg', type=int, default=1,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--ml', type=int, default=10,
                        help='Max Length')
    parser.add_argument('--path', type=str, default="/work/",
                        help='Path to dir')

    # May need them later
    parser.add_argument('--e', type=int, default=1,
                        help='Number of epochs.')
    parser.add_argument('--k', type=int, default=2, help='Number of K.')
    parser.add_argument('--b', type=int, default=128,
                        help='Number of batch size.')
    parser.add_argument('--a', type=float, default=0.5,
                        help='Alpha param')

    parser.add_argument('--o', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')

    parser.add_argument('--pm', type=str, default='max', help='Pooling Mode: max or avg')
    parser.add_argument('--fm', type=str, default='fair', help='Focus Mode: fair, pair or rec')

    parser.add_argument('--m', type=int, default=1, help='Mode')

    parser.add_argument('--lstm', type=int, default=0, help='LSTM model')

    parser.add_argument('--af', type=str, default='logistic', help='anneal_function: logistic or linear')
    parser.add_argument('--klrate', type=float, default=0.01, help='KL anneal rate')
    parser.add_argument('--wd', type=float, default=0.75, help='keep_rate_word_dropout')


    return parser.parse_args()



if __name__ == '__main__':

	args = parse_args()

	date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
	out_dir = "%sdata/out/" % args.path
	enablePadding = True
	tokenise_name = "50K_BPE"


	
	path = args.path
	model = args.model
	train_data = args.dataset
	hidden_dim = args.h
	latent_dim = args.l
	num_negatives = args.neg
	epochs = args.e
	alpha = args.a
	k = args.k
	pm = args.pm
	fm = args.fm
	optimizer=args.o
	batch_size = args.b
	max_len = args.ml
	mode = args.m

	enableLSTM = True if args.lstm == 1 else False
	enableS2S = True if "s2s" in model else False

	anneal_function = args.af
	kl_rate = args.klrate
	keep_rate_word_dropout = args.wd

	eval_every_one_epoch = False
	

	print(args)

	if optimizer == "adadelta":
		optimizer = Adadelta()
	elif optimizer == "rmsprop":
		optimizer = RMSprop()
	elif optimizer == "adam":
		# optimizer = Adam(clipvalue=5)
		# optimizer = Adam(0.0002, 0.5)
		optimizer = Adam()



	# load pre-trained tokeniser
	if "BPE" in tokenise_name:

		sp = spm.SentencePieceProcessor()
		sp.Load('%sdata/bpe/en.wiki.bpe.op50000.model' % path)
		bpe = KeyedVectors.load_word2vec_format("%sdata/bpe/en.wiki.bpe.op50000.d200.w2v.bin" % path, binary=True)
		bpe.index2word = [''] + bpe.index2word + ['<sos>'] + ['<eos>']  # add empty string 
		nb_words = len(bpe.index2word)
		# word2index
		bpe_dict = {bpe.index2word[i]: i for i in range(len(bpe.index2word))}
		# construct embedding_matrix
		embedding_matrix = np.concatenate([np.zeros((1, bpe.vector_size)), bpe.vectors, np.zeros((2, bpe.vector_size))]) # add zero vector for empty string (i.e. used for padding)
		
	elif "trigram" in tokenise_name:

		tokeniser = L3wTransformer()
		tokeniser = tokeniser.load("%sdata/trigram/%s" % (path,tokenise_name))

	# =================================== Initiate Model ==============================================

	if model == "dssm":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, PoolMode=pm)
	elif model == "dssm2":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableSeparate=True, PoolMode=pm)
	elif model == "dssm_gru":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableLSTM=True, PoolMode=pm)
	elif model == "dssm_gru2":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableLSTM=True, enableSeparate=True, PoolMode=pm)
	elif model == "small_dssm":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableLSTM=True, enableSeparate=True, PoolMode=pm)
	elif model == "binary":
		run = BinaryClassifier(hidden_dim, latent_dim, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableLSTM=True, PoolMode=pm)

	elif model == "vae_dssm2":
		run = VAE_DSSM2(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer,  PoolMode=pm, FocusMode=fm, mode=mode)
	elif model == "kate_dssm2":
		run = VAE_DSSM2(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], k, "kcomp", optimizer=optimizer,  PoolMode=pm, FocusMode=fm)
	
	elif model == "ssvae":
		run = SSVAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer,  FocusMode=fm, mode=mode)
	
	elif model == "ssvae2":
		run = SSVAE2(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer,  FocusMode=fm, mode=mode)


	elif model == "s2s_aae":
		run = S2S_AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, keep_rate_word_dropout=keep_rate_word_dropout, mode=mode)
	
	elif model == "s2s_wae":
		run = S2S_AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, keep_rate_word_dropout=keep_rate_word_dropout, mode=mode, enableWasserstein=True)


	elif model == "aae":
		# run = S2S_AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, keep_rate_word_dropout=keep_rate_word_dropout, mode=mode, enableS2S=False)
	    run = AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, keep_rate_word_dropout=keep_rate_word_dropout, enableWasserstein=False, enableBOW=False, enableS2S=False)

	# elif model == "wae":
		# run = S2S_AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, keep_rate_word_dropout=keep_rate_word_dropout, mode=mode, enableWasserstein=True, enableS2S=False)
	elif model == "wae":
	    run = AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, keep_rate_word_dropout=keep_rate_word_dropout, enableWasserstein=True, enableBOW=False, enableS2S=False)

	elif model == "aae_pair":
	    run = AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, mode=mode, keep_rate_word_dropout=keep_rate_word_dropout, enableWasserstein=False, enableBOW=False, enableS2S=False, enablePairLoss=True)
	elif model == "wae_pair":
	    run = AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, mode=mode, keep_rate_word_dropout=keep_rate_word_dropout, enableWasserstein=True, enableBOW=False, enableS2S=False, enablePairLoss=True)



	elif model == "dssm_aae":
		run = S2S_AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, keep_rate_word_dropout=keep_rate_word_dropout, mode=mode, enableS2S=False, enablePairLoss=True)
	elif model == "dssm_aae_s2s":
		run = S2S_AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, keep_rate_word_dropout=keep_rate_word_dropout, mode=mode, enableS2S=True, enablePairLoss=True)
	


	elif model == "bow_pr_aae":
		run = PRA(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, mode=mode, enableS2S=False)
	elif model == "bow_pr_wae":
		run = PRA(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, mode=mode, enableS2S=False, enableWasserstein=True)
	elif model == "pr_aae":
		run = PRA(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, mode=mode, enableS2S=False, enableGRU=True)
	elif model == "pr_wae":
		run = PRA(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, mode=mode, enableS2S=False, enableGRU=True, enableWasserstein=True)
	
	elif model == "pra2_aae":
		run = PRA2(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, mode=mode, enableS2S=False)
	elif model == "pra2_wae":
		run = PRA2(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, mode=mode, enableS2S=False, enableWasserstein=True)

	# elif model == "vae":
	# 	run = VAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], batch_size, optimizer=optimizer, PoolMode=pm, kl_weight=kl_weight)
	# elif model == "kate":
	# 	run = VAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], batch_size, k, "kcomp", optimizer=optimizer, PoolMode=pm, kl_weight=kl_weight)
	# elif model == "kate2":
	# 	run = VAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], batch_size, k, "kcomp", optimizer=optimizer, enableKL=False, PoolMode=pm, kl_weight=kl_weight)
	# elif model == "bowvae":
	# 	run = BOW_VAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer,  PoolMode=pm)


	elif model == "vae":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=1, enableKL=False, enableS2S=False)
	elif model == "vae_kl":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=1, enableKL=True, enableS2S=False)
	
	elif model == "vae_neg":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=1, enableKL=False, enableS2S=False, enableNeg=True)
	elif model == "vae_neg_kl":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=1, enableKL=True, enableS2S=False, enableNeg=True)
	

	elif model == "kate":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=1, enableKL=False, enableS2S=False, comp_topk=k, ctype="kcomp")
	elif model == "kate_kl":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=1, enableKL=True, enableS2S=False, comp_topk=k, ctype="kcomp")
	

	elif model == "bow_vae":
		run = BOW_VAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, mode=1, enableKL=False, comp_topk=None, ctype=None, epsilon_std=1.0, PoolMode=pm)
	elif model == "bow_vae_kl":
		run = BOW_VAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, mode=1, enableKL=True, comp_topk=None, ctype=None, epsilon_std=1.0, PoolMode=pm)
	elif model == "bow_kate":
		run = BOW_VAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, mode=1, enableKL=False, comp_topk=k, ctype="kcomp", epsilon_std=1.0, PoolMode=pm)
	elif model == "bow_kate_kl":
		run = BOW_VAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, mode=1, enableKL=True, comp_topk=k, ctype="kcomp", epsilon_std=1.0, PoolMode=pm)
	


	elif model == "ae":
		run = AE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, keep_rate_word_dropout=keep_rate_word_dropout)



	elif model == "seq2seq_ori":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=1, enableKL=False, enableS2S=True)
	elif model == "seq2seq_ori2":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=1, enableKL=False, enableS2S=True, separateEmbedding=True)
	
	elif model == "seq2seq_ori_dssm":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=2, enableKL=False, enableS2S=True)


	elif model == "seq2seq":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=1)
	
	elif model == "s2s_dssm":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=mode)


	elif model == "ss_pr_aae":
		run = SS_PR(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, mode=mode, enableS2S=False, enableGRU=True, enableWasserstein=False)



	isPairModel = ["dssm", "dssm2", "dssm_gru", "dssm_gru2", "vae_dssm2", "kate_dssm2"]
	isBinaryModel = ["binary"]
	isHybridModel = [ "ssvae", "ssvae2", "s2s_dssm", "dssm_aae", "dssm_aae_s2s", "bow_pr_aae", "bow_pr_wae", "pr_aae", "pr_wae", "pra2_aae", "pra2_wae", "seq2seq_ori_dssm", "aae_pair", "wae_pair"]
	isBOWModel = ["bow_vae_kl", "bow_vae", "bow_kate", "bow_kate_kl"]

	isSemiModel = ["ss_pr_aae", "ss_pr_wae", "small_dssm"]

	run.encoder._make_predict_function()
	graph = tf.get_default_graph()

	model_name = "%s_%s_%s_h%d_l%d_k%d_n%d_ml%d_w%d_b%d_e%d_a%.1f_%s_%s" % (run.name(), train_data, date_time, hidden_dim, latent_dim, k, num_negatives, max_len, nb_words, batch_size, epochs, alpha, optimizer, tokenise_name)

	
	# =================================== Get testing data ==============================================
	test_set = []
	for i in ["MayFlower", "JuneFlower", "JulyFlower", "sts", "quora", "para"]:
		df, qrel = get_test_data(i, path)
		q_ = parse_texts_bpe(df.q.tolist(), sp, bpe_dict, max_len, enablePadding)
		d_ = parse_texts_bpe(df.d.tolist(), sp, bpe_dict, max_len, enablePadding)
		test_set.append([q_, d_, qrel, df, i])

		
	# Read pair data
	if model in isPairModel:
		train_data_dir = '%sdata/train_data/%s.q.npy' % (path,train_data)

		if os.path.exists(train_data_dir):
			q_train = np.load('%sdata/train_data/%s.q.npy' % (path,train_data))
			d_train = np.load('%sdata/train_data/%s.d.npy' % (path,train_data))

			train_data_dir = '%sdata/train_data/%s.label.npy' % (path,train_data)
			if os.path.exists(train_data_dir):
				y_train = np.load('%sdata/train_data/%s.label.npy' % (path,train_data))

		else:
			reader = get_reader(train_data, path)
			q_train = parse_texts_bpe(reader.q.tolist(), sp, bpe_dict, max_len, enablePadding=True)
			d_train = parse_texts_bpe(reader.d.tolist(), sp, bpe_dict, max_len, enablePadding=True)

	elif model in isBinaryModel:

		q_train = np.load('%sdata/train_data/%s.q.npy' % (path,train_data))
		d_train = np.load('%sdata/train_data/%s.d.npy' % (path,train_data))
		_y_train = np.load('%sdata/train_data/%s.label.npy' % (path,train_data))


	elif model in isHybridModel:

		q_enc_inputs = np.load('%sdata/train_data/%s.q.npy' % (path,train_data))
		q_dec_outputs = np.load('%sdata/train_data/%s.q.do.npy' % (path,train_data))
		q_dec_inputs = np.load('%sdata/train_data/%s.q.di.npy' % (path,train_data))

		d_enc_inputs = np.load('%sdata/train_data/%s.d.npy' % (path,train_data))
		d_dec_outputs = np.load('%sdata/train_data/%s.d.do.npy' % (path,train_data))
		d_dec_inputs = np.load('%sdata/train_data/%s.d.di.npy' % (path,train_data))

		q_train = q_enc_inputs

	# Read unsuperivsed data
	else:

		train_data_dir = '%sdata/train_data/%s.q.npy' % (path,train_data)
		if os.path.exists(train_data_dir):

			q_enc_inputs = np.load('%sdata/train_data/%s.q.npy' % (path,train_data))

			# Q > D models
			if model in ["seq2seq", "seq2seq_ori", "seq2seq_ori2", "s2s_aae", "s2s_wae"]:
				q_dec_outputs = np.load('%sdata/train_data/%s.d.do.npy' % (path,train_data))
				q_dec_inputs = np.load('%sdata/train_data/%s.d.di.npy' % (path,train_data))

			# Q > Q models
			elif model in ["aae", "wae", "vae", "vae_kl", "kate", "kate_kl", "ae", "vae_neg", "vae_neg_kl"]:
				q_dec_outputs = np.load('%sdata/train_data/%s.q.do.npy' % (path,train_data))
				q_dec_inputs = np.load('%sdata/train_data/%s.q.di.npy' % (path,train_data))


			q_train = q_enc_inputs



	print("============Start Training================")
	best_auc_score = 0
	# feay
	step = 400 * batch_size # consider to increase 100 to 1000 due to slow inference time.
	# step = 1 * batch_size

	all_steps = int(len(q_train) / step) + 1
	may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = 0, 0, 0, 0, 0, 0

	step_count = 0


	if model in ["ignore"]:

		x_train = [q_train, d_train]
		y_train = _y_train
		hist = run.model.fit(x_train, y_train,
								        shuffle=True,
								        epochs=100,
								        verbose=2,
								        batch_size=batch_size,
								        validation_split=0.2,
								        callbacks=[EarlyStopping()]
								        )
		may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = evaluate(run.encoder, test_set)
		print(may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc)


	elif model in isSemiModel:

		# q_enc_inputs = np.load('%sdata/train_data/%s.q.npy' % (path,train_data))[:1000]
		# q_dec_outputs = np.load('%sdata/train_data/%s.q.do.npy' % (path,train_data))[:1000]
		# q_dec_inputs = np.load('%sdata/train_data/%s.q.di.npy' % (path,train_data))[:1000]

		# d_enc_inputs = np.load('%sdata/train_data/%s.d.npy' % (path,train_data))[:1000]
		# d_dec_outputs = np.load('%sdata/train_data/%s.d.do.npy' % (path,train_data))[:1000]
		# d_dec_inputs = np.load('%sdata/train_data/%s.d.di.npy' % (path,train_data))[:1000]

		q_enc_inputs = np.load('%sdata/train_data/%s.q.npy' % (path,train_data))[:10000000]
		q_dec_outputs = np.load('%sdata/train_data/%s.q.do.npy' % (path,train_data))[:10000000]
		q_dec_inputs = np.load('%sdata/train_data/%s.q.di.npy' % (path,train_data))[:10000000]

		d_enc_inputs = np.load('%sdata/train_data/%s.d.npy' % (path,train_data))[:10000000]
		d_dec_outputs = np.load('%sdata/train_data/%s.d.do.npy' % (path,train_data))[:10000000]
		d_dec_inputs = np.load('%sdata/train_data/%s.d.di.npy' % (path,train_data))[:10000000]

		q_train = q_enc_inputs

		small_q_enc_inputs = q_enc_inputs[:1000000]
		small_d_enc_inputs = d_enc_inputs[:1000000]

		valid = np.ones((len(q_enc_inputs), 1))
		fake = np.zeros((len(q_enc_inputs), 1)) if "wae" not in model else -np.ones((len(q_enc_inputs), 1))

		idx = np.arange(len(small_q_enc_inputs))
		shuffle(idx)
		x_pair_train = [small_q_enc_inputs, small_d_enc_inputs, small_d_enc_inputs[idx]]
		y_pair_train = np.zeros((len(small_q_enc_inputs), 2))
		y_pair_train[:, 0] = 1

		val_loss = 9999999


		if model == "small_dssm":
			for epoch in range(50):
				idx = np.arange(len(small_q_enc_inputs))
				shuffle(idx)
				x_pair_train = [small_q_enc_inputs, small_d_enc_inputs, small_d_enc_inputs[idx]]
				t1 = time()
				hist = run.model.fit(x_pair_train, y_pair_train,
									        shuffle=True,
									        epochs=1,
									        verbose=0,
									        batch_size=batch_size,
									        validation_split=0.2,
								        	callbacks=[EarlyStopping()]
									        )
				t2 = time()
				losses = ', '.join([ "%s = %f" % (k, hist.history[k][-1]) for k in hist.history])
				may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = evaluate(run.encoder, test_set)
				print_output = '%s, Epoch %d, Iteration %d,  [%.1f s - %.1f s], May = %.4f, June = %.4f, July = %.4f, Quora = %.4f, Para = %.4f, STS = %.4f, Loss = %.4f, V_Loss = %.4f \n' % (run.name(), epoch, 0, t2-t1, 0, may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc, hist.history['loss'][-1], hist.history['val_loss'][-1])
				file_output = '%s, Epoch %d, Iteration %d, [%.1f s - %.1f s], May = %.4f, June = %.4f, July = %.4f, Quora = %.4f, Para = %.4f, STS = %.4f, %s \n' % (run.name(), epoch, 0, t2-t1, 0, may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc, losses)
				best_auc_score = write_to_files(run, print_output, file_output, path, model_name,  hist.history['val_loss'][-1], best_auc_score, model)


			# may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = evaluate(run.encoder, test_set)
			# print("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (model, may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc))

		else:

			
			for epoch in range(100):

				x_train = [q_enc_inputs, run.word_dropout(q_dec_inputs, bpe_dict['<unk>'])]
				y_train = [np.expand_dims(q_dec_outputs, axis=-1), np.ones(len(q_dec_outputs))]

				hist = run.q_ae.fit(x_train, y_train,
										        shuffle=True,
										        epochs=1,
										        verbose=1,
										        batch_size=batch_size
										        )

				latent_fake = run.q_gs_encoder.predict(q_enc_inputs)
				latent_real = np.random.normal(size=(len(q_enc_inputs), latent_dim))
				d_loss_real = run.discriminator.fit(latent_real, valid, epochs=1, batch_size=batch_size, verbose=1)
				d_loss_fake = run.discriminator.fit(latent_fake, fake, epochs=1, batch_size=batch_size, verbose=1)
				d_loss = 0.5 * np.add(d_loss_real.history['loss'], d_loss_fake.history['loss'])
				d_acc = 0.5 * np.add(d_loss_real.history['acc'], d_loss_fake.history['acc'])

				hist = run.dssm.fit(x_pair_train, y_pair_train,
										        shuffle=True,
										        epochs=1,
										        verbose=1,
										        batch_size=batch_size,
								        		validation_split=0.2
										        )

				may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = evaluate(run.encoder, test_set)
				print(epoch, hist.history["val_loss"][0], may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc)
				if val_loss < hist.history["val_loss"][0]:
					print("early stoping")
					break
				else:
					val_loss = hist.history["val_loss"][0]
	else:
	
		
		for epoch in range(epochs):
			for i in range(0, len(q_train), step):

				if model in isPairModel:
					# TODO separate train and validation set here, important
					_q_train = q_train[i: i + step]
					_d_train = d_train[i: i + step]

					idx = np.arange(len(_q_train))
					shuffle(idx)

					_y_train = np.zeros((len(_q_train), 2))
					_y_train[:, 0] = 1

					x_train = [_q_train, _d_train, _d_train[idx]]
					if model in ["dssm", "dssm2", "dssm_gru", "dssm_gru2"]:
						y_train = _y_train
					else:
						y_train = [np.expand_dims(_q_train, axis=-1), np.expand_dims(_d_train, axis=-1), np.expand_dims(_d_train[idx], axis=-1), np.expand_dims(_d_train, axis=-1)] + [_y_train]
				
				elif model in isHybridModel:

					q_enc_inputs_ = q_enc_inputs[i: i + step]
					q_dec_inputs_ = q_dec_inputs[i: i + step]
					q_dec_outputs_ = q_dec_outputs[i: i + step]

					d_enc_inputs_ = d_enc_inputs[i: i + step]
					d_dec_inputs_ = d_dec_inputs[i: i + step]
					d_dec_outputs_ = d_dec_outputs[i: i + step]

					idx = np.arange(len(q_enc_inputs_))
					shuffle(idx)

					_y_train = np.zeros((len(q_enc_inputs_), 2))
					_y_train[:, 0] = 1


					if model == "ssvae":
						x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx], d_dec_inputs_, d_dec_inputs_[idx]]
						y_train = [np.expand_dims(d_dec_outputs_, axis=-1), np.expand_dims(d_dec_outputs_[idx], axis=-1), _y_train]
					elif model == "ssvae2":
						x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx]]
						y_train = [np.expand_dims(q_dec_outputs_, axis=-1), np.expand_dims(d_dec_outputs_, axis=-1), np.expand_dims(d_dec_outputs_[idx], axis=-1), _y_train]
					elif model == "seq2seq_ori_dssm":
						x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx], run.word_dropout(d_dec_inputs_, bpe_dict['<unk>']), run.word_dropout(q_dec_inputs_, bpe_dict['<unk>']), run.word_dropout(q_dec_inputs_, bpe_dict['<unk>'])]
						y_train = [np.expand_dims(d_dec_outputs_, axis=-1), np.expand_dims(q_dec_outputs_, axis=-1), np.expand_dims(q_dec_outputs_, axis=-1), _y_train]
					elif model == "s2s_dssm":
						if mode in [4,5,7]:
							x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx], run.word_dropout(d_dec_inputs_, bpe_dict['<unk>']), run.word_dropout(q_dec_inputs_, bpe_dict['<unk>']), run.word_dropout(q_dec_inputs_, bpe_dict['<unk>'])]
							y_train = [np.expand_dims(d_dec_outputs_, axis=-1), np.expand_dims(q_dec_outputs_, axis=-1), np.expand_dims(q_dec_outputs_, axis=-1), np.ones((len(d_dec_outputs_))), _y_train]
						if mode in [8]:
							x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx], run.word_dropout(q_dec_inputs_, bpe_dict['<unk>'])]
							y_train = [np.expand_dims(q_dec_outputs_, axis=-1), np.expand_dims(q_dec_outputs_, axis=-1), np.ones((len(d_dec_outputs_)))]
						else:
							x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx], run.word_dropout(q_dec_inputs_, bpe_dict['<unk>']), run.word_dropout(d_dec_inputs_, bpe_dict['<unk>']), run.word_dropout(d_dec_inputs_[idx], bpe_dict['<unk>'])]
							y_train = [np.expand_dims(q_dec_outputs_, axis=-1), np.expand_dims(d_dec_outputs_, axis=-1), np.expand_dims(d_dec_outputs_[idx], axis=-1), np.ones((len(d_dec_outputs_))), _y_train]

					elif model == "dssm_aae":
						x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx], run.word_dropout(q_dec_inputs_, bpe_dict['<unk>']), run.word_dropout(d_dec_inputs_, bpe_dict['<unk>']), run.word_dropout(d_dec_inputs_[idx], bpe_dict['<unk>'])]
						y_train = [np.expand_dims(q_dec_outputs_, axis=-1), np.expand_dims(d_dec_outputs_, axis=-1), np.expand_dims(d_dec_outputs_[idx], axis=-1), _y_train]
						y_train = y_train + [np.ones((len(q_dec_outputs_), 1)), np.zeros((len(q_dec_outputs_), 1))] + y_train + [np.zeros((len(q_dec_outputs_), 1)), np.ones((len(q_dec_outputs_), 1))]

					elif model == "dssm_aae_s2s":
						x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx], run.word_dropout(d_dec_inputs_, bpe_dict['<unk>']), run.word_dropout(q_dec_inputs_, bpe_dict['<unk>']), run.word_dropout(q_dec_inputs_, bpe_dict['<unk>'])]
						y_train = [np.expand_dims(d_dec_outputs_, axis=-1), np.expand_dims(q_dec_outputs_, axis=-1), np.expand_dims(q_dec_outputs_, axis=-1), _y_train]
						y_train = y_train + [np.ones((len(q_dec_outputs_), 1)), np.zeros((len(q_dec_outputs_), 1))] + y_train + [np.zeros((len(q_dec_outputs_), 1)), np.ones((len(q_dec_outputs_), 1))]



					elif model in ["pr_aae", "pr_wae"]:

						x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx]] if "bow" in model else [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx], run.word_dropout(q_dec_inputs_, bpe_dict['<unk>'])]
						reals = [np.ones((len(q_dec_outputs_), 1))] 
						fakes = [np.zeros((len(q_dec_outputs_), 1))] if "aae" in model else [-1 * np.ones((len(q_dec_outputs_), 1))]

						if "bow" in model:
							y_train = [toBOW(q_enc_inputs_, nb_words), _y_train]
						else:
							y_train = [np.expand_dims(q_enc_inputs_, axis=-1), _y_train]


						y_train = y_train + reals + fakes + y_train + y_train + fakes + reals + y_train

					elif model in ["bow_pr_wae", "bow_pr_aae", "pra2_wae", "pra2_aae", "aae_pair", "wae_pair"]:
						x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx]] if "bow" in model else [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx], run.word_dropout(q_dec_inputs_, bpe_dict['<unk>']), run.word_dropout(d_dec_inputs_, bpe_dict['<unk>']), run.word_dropout(d_dec_inputs_[idx], bpe_dict['<unk>'])]
						reals = [np.ones((len(q_dec_outputs_), 1))] * 3
						fakes = [np.zeros((len(q_dec_outputs_), 1))] * 3 if "aae" in model else [-1 * np.ones((len(q_dec_outputs_), 1))] * 3

						if "bow" in model:
							y_train = [toBOW(q_enc_inputs_, nb_words), toBOW(d_enc_inputs_, nb_words), toBOW(d_enc_inputs_[idx], nb_words), _y_train]
						else:
							y_train = [np.expand_dims(q_enc_inputs_, axis=-1), np.expand_dims(d_enc_inputs_, axis=-1), np.expand_dims(d_enc_inputs_[idx], axis=-1), _y_train]

						if model in ["aae_pair", "wae_pair"]:
							y_train = y_train + reals
						else:
							y_train = y_train + reals + fakes + reals + fakes + y_train + fakes + reals + fakes + reals if "pra2" in model else y_train + reals + fakes + y_train + y_train + fakes + reals + y_train


				elif model in isBinaryModel:
					_q_train = q_train[i: i + step]
					_d_train = d_train[i: i + step]
					x_train = [_q_train, _d_train]
					y_train = _y_train[i: i + step]

				elif model in isBOWModel:
					tmp = q_enc_inputs[i: i + step]
					x_train = [tmp]
					y_train = [toBOW(tmp, nb_words), np.ones(len(tmp))]

				elif model in ["vae_neg", "vae_neg_kl"]:

					x_train = [q_enc_inputs[i: i + step], q_dec_inputs[i: i + step]]
					idx = np.arange(len(x_train[0]))
					shuffle(idx)
					x_train = x_train + [x_train[1][idx]]
					dropout_x_train = run.word_dropout(x_train[1], bpe_dict['<unk>'])
					x_train[1] = dropout_x_train
					dropout_x_train = run.word_dropout(x_train[2], bpe_dict['<unk>'])
					x_train[2] = dropout_x_train

					tmp = np.expand_dims(q_dec_outputs[i: i + step], axis=-1)
					tmp2 = np.expand_dims(q_dec_outputs[i: i + step][idx], axis=-1)

					y_train = [tmp, tmp2, np.ones((len(tmp)))]


				else:


					x_train = [q_enc_inputs[i: i + step], q_dec_inputs[i: i + step]]

					dropout_x_train = run.word_dropout(x_train[1], bpe_dict['<unk>'])
					x_train[1] = dropout_x_train

					if model == "ae":
						x_train = [q_enc_inputs[i: i + step]]

					tmp = np.expand_dims(q_dec_outputs[i: i + step], axis=-1)

					if model in ["s2s_aae", "aae", "wae"]:
						# y_train = [tmp, np.ones((len(tmp), 1)), np.zeros((len(tmp), 1)), tmp, np.zeros((len(tmp), 1)), np.ones((len(tmp), 1))]
						y_train = [tmp, np.ones((len(tmp), 1))]

					# elif model in ["s2s_wae", "wae"]:
						# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
						# tmp = to_categorical(q_dec_outputs[i: i + step], nb_words).reshape(len(tmp), max_len, nb_words)
						# tmp[tmp==0] = -1
						# y_train = [tmp, np.ones((len(tmp), 1)), -1 * np.ones((len(tmp), 1)), tmp, -1 * np.ones((len(tmp), 1)), np.ones((len(tmp), 1))]
						# y_train = [tmp, np.ones((len(tmp), 1)), -1 * np.ones((len(tmp), 1)), tmp, -1 * np.ones((len(tmp), 1)), np.ones((len(tmp), 1))]

					elif model in ["vae", "kate", "vae_kl", "kate_kl", "ae"]:
						tmp = np.expand_dims(q_dec_outputs[i: i + step], axis=-1)

						y_train = [tmp, np.ones((len(tmp)))]

					elif model in ["seq2seq_ori", "seq2seq_ori2"]:
						y_train = [tmp]

				if hasattr(run, 'enableKL'):
					if run.enableKL:
						kl_weight = kl_anneal_function(anneal_function, step_count, kl_rate, int(all_steps/2)+1)
						x_train = x_train + [np.array([kl_weight] * len(x_train[0]))]


				t1 = time()

				if model in ["aae", "wae"]:

					valid = np.ones((len(x_train[0]), 1))
					fake = np.zeros((len(x_train[0]), 1)) if "wae" not in model else -np.ones((len(x_train[0]), 1))

					latent_fake = run.gs_encoder.predict(x_train[0])
					latent_real = np.random.normal(size=(len(x_train[0]), latent_dim))
					d_loss_real = run.discriminator.fit(latent_real, valid, epochs=1, batch_size=batch_size, verbose=0)
					d_loss_fake = run.discriminator.fit(latent_fake, fake, epochs=1, batch_size=batch_size, verbose=0)
					d_loss = 0.5 * np.add(d_loss_real.history['loss'], d_loss_fake.history['loss'])
					d_acc = 0.5 * np.add(d_loss_real.history['acc'], d_loss_fake.history['acc'])


					hist = run.model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0, validation_split=0.2)


					print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_acc[0], hist.history['loss'][0]))

				elif model  in ["aae_pair", "wae_pair"]:
					valid = np.ones((len(x_train[0]), 1))
					fake = np.zeros((len(x_train[0]), 1)) if "wae" not in model else -np.ones((len(x_train[0]), 1))

					d_loss, d_acc = [], []
					encs = [run.gs_encoder, run.doc_gs_encoder, run.doc_gs_encoder]
					encs_inputs = x_train[:3]
					for enc, inp in zip(encs, encs_inputs):

						latent_fake = enc.predict(inp)
						latent_real = np.random.normal(size=(len(inp), latent_dim))
						d_loss_real = run.discriminator.fit(latent_real, valid, epochs=1, batch_size=batch_size, verbose=0)
						d_loss_fake = run.discriminator.fit(latent_fake, fake, epochs=1, batch_size=batch_size, verbose=0)
						d_loss.append(0.5 * np.add(d_loss_real.history['loss'], d_loss_fake.history['loss']))
						d_acc.append(0.5 * np.add(d_loss_real.history['acc'], d_loss_fake.history['acc']))


					hist = run.model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0, validation_split=0.2)
				else:

					hist = run.model.fit(x_train, y_train,
								        shuffle=True,
								        epochs=1,
								        verbose=0,
								        batch_size=batch_size,
								        validation_split=0.2, # change to valication_data=(x_val, y_val)
								        )


				losses = ', '.join([ "%s = %f" % (k, hist.history[k][-1]) for k in hist.history])

				t2 = time()
				
				if not eval_every_one_epoch:
					t3 = time()
					may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = evaluate(run.encoder, test_set)
					t4 = time()


				print_output = '%s, Epoch %d, Iteration %d,  [%.1f s - %.1f s], May = %.4f, June = %.4f, July = %.4f, Quora = %.4f, Para = %.4f, STS = %.4f, Loss = %.4f, V_Loss = %.4f \n' % (run.name(), epoch, i + step, t2-t1 , t4-t3, may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc, hist.history['loss'][-1], hist.history['val_loss'][-1])
				file_output = '%s, Epoch %d, Iteration %d, [%.1f s - %.1f s], May = %.4f, June = %.4f, July = %.4f, Quora = %.4f, Para = %.4f, STS = %.4f, %s \n' % (run.name(), epoch, i + step, t2-t1 , t4-t3, may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc, losses)


				best_auc_score = write_to_files(run, print_output, file_output, path, model_name, july_auc, best_auc_score, model)

			step_count = step_count + 1

			if eval_every_one_epoch:
				t3 = time()
				may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = evaluate(run.encoder, test_set)
				t4 = time()
				print_output = '%s, Epoch %d, Iteration %d,  [%.1f s - %.1f s], May = %.4f, June = %.4f, July = %.4f, Quora = %.4f, Para = %.4f, STS = %.4f, Loss = %.4f, V_Loss = %.4f \n' % (run.name(), epoch, i + step, t2-t1 , t4-t3, may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc, hist.history['loss'][-1], hist.history['val_loss'][-1])
				file_output = '%s, Epoch %d, Iteration %d, [%.1f s - %.1f s], May = %.4f, June = %.4f, July = %.4f, Quora = %.4f, Para = %.4f, STS = %.4f, %s \n' % (run.name(), epoch, i + step, t2-t1 , t4-t3, may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc, losses)
				best_auc_score = write_to_files(run, print_output, file_output, path, model_name, july_auc, best_auc_score, model)
				
