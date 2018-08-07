from Utils import *
from Models import *
from FastModels import *
from BatchGenerator import *
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

		embedding_layer = Embedding(nb_words,
		                    embedding_matrix.shape[-1],
		                    weights=[embedding_matrix],
		                    input_length=max_len,
		                    trainable=True)
		
	elif "trigram" in tokenise_name:

		tokeniser = L3wTransformer()
		tokeniser = tokeniser.load("%sdata/trigram/%s" % (path,tokenise_name))

	# =================================== Initiate Model ==============================================

	if model == "dssm":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, PoolMode=pm)
	elif model == "dssm2":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableSeparate=True, PoolMode=pm)
	elif model == "dssm_lstm":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableLSTM=True, PoolMode=pm)
	elif model == "dssm_lstm2":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableLSTM=True, enableSeparate=True, PoolMode=pm)

	elif model == "binary":
		run = BinaryClassifier(hidden_dim, latent_dim, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableLSTM=enableLSTM, PoolMode=pm)

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
		run = S2S_AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, keep_rate_word_dropout=keep_rate_word_dropout, mode=mode, enableS2S=False)
	
	elif model == "wae":
		run = S2S_AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, keep_rate_word_dropout=keep_rate_word_dropout, mode=mode, enableWasserstein=True, enableS2S=False)
	


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

	elif model == "seq2seq_ori":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=1, enableKL=False, enableS2S=True)
	
	elif model == "seq2seq":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=1)
	
	elif model == "s2s_dssm":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, kl_rate=kl_rate, keep_rate_word_dropout=keep_rate_word_dropout, mode=mode)



	isPairModel = ["dssm", "dssm2", "dssm_lstm", "dssm_lstm2", "vae_dssm2", "kate_dssm2",]
	isBinaryModel = ["binary"]
	isHybridModel = [ "ssvae", "ssvae2", "s2s_dssm"]

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
			# for feaytest
			# q_train = q_train[:1000]
			# d_train = d_train[:1000]

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
			if model in ["seq2seq", "seq2seq_ori", "s2s_aae", "s2s_wae"]:
				q_dec_outputs = np.load('%sdata/train_data/%s.d.do.npy' % (path,train_data))
				q_dec_inputs = np.load('%sdata/train_data/%s.d.di.npy' % (path,train_data))

			# Q > Q models
			elif model in ["aae", "wae", "vae", "vae_kl"]:
				q_dec_outputs = np.load('%sdata/train_data/%s.q.do.npy' % (path,train_data))
				q_dec_inputs = np.load('%sdata/train_data/%s.q.di.npy' % (path,train_data))

			# q_train = q_enc_inputs[:1000]
			# q_enc_inputs = q_enc_inputs[:1000]
			# q_dec_outputs = q_dec_outputs[:1000]
			# q_dec_inputs = q_dec_inputs[:1000]

			q_train = q_enc_inputs



	print("============Start Training================")

	
	best_auc_score = 0
	step = 100 * batch_size # consider to increase 100 due to slow inference time.

	all_steps = int(len(q_train) / step) + 1
	may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = 0, 0, 0, 0, 0, 0

	for epoch in range(epochs):
		step_count = 0
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
				if model in ["dssm", "dssm2", "dssm_lstm", "dssm_lstm2"]:
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


				# x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx], q_dec_inputs_, d_dec_inputs_, d_dec_inputs_[idx]]
				# y_train = [np.expand_dims(q_dec_outputs_, axis=-1), np.expand_dims(d_dec_outputs_, axis=-1), np.expand_dims(d_dec_outputs_[idx], axis=-1), _y_train]
				if model == "ssvae":
					x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx], d_dec_inputs_, d_dec_inputs_[idx]]
					y_train = [np.expand_dims(d_dec_outputs_, axis=-1), np.expand_dims(d_dec_outputs_[idx], axis=-1), _y_train]
				elif model == "ssvae2":
					x_train = [q_enc_inputs_, d_enc_inputs_, d_enc_inputs_[idx]]
					y_train = [np.expand_dims(q_dec_outputs_, axis=-1), np.expand_dims(d_dec_outputs_, axis=-1), np.expand_dims(d_dec_outputs_[idx], axis=-1), _y_train]
				
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



			elif model in isBinaryModel:
				_q_train = q_train[i: i + step]
				_d_train = d_train[i: i + step]
				x_train = [_q_train, _d_train]
				y_train = _y_train[i: i + step]

			else:

				# if model == "bowvae":
				# 	tmp = toBOW(q_enc_inputs[i: i + step], nb_words)
				# 	y_train =[tmp, tmp]

				x_train = [q_enc_inputs[i: i + step], q_dec_inputs[i: i + step]]

				dropout_x_train = run.word_dropout(x_train[1], bpe_dict['<unk>'])
				x_train[1] = dropout_x_train


				if model in ["s2s_aae", "aae"]:
					y_train = [tmp, np.ones((len(tmp), 1)), np.zeros((len(tmp), 1)), tmp, np.zeros((len(tmp), 1)), np.ones((len(tmp), 1))]

				elif model in ["s2s_wae", "wae"]:
					tmp = to_categorical(q_dec_outputs[i: i + step], nb_words).reshape(len(q_enc_inputs[i: i + step]), max_len, nb_words)
					y_train = [tmp, np.ones((len(tmp), 1)), np.zeros((len(tmp), 1)), tmp, np.zeros((len(tmp), 1)), np.ones((len(tmp), 1))]

				elif model in ["vae", "kate", "vae_kl", "kate_kl"]:
					tmp = np.expand_dims(q_dec_outputs[i: i + step], axis=-1)
					y_train = [tmp, np.ones((len(tmp)))]


			if hasattr(run, 'enableKL'):
				if run.enableKL:
					kl_weight = kl_anneal_function(anneal_function, step_count, kl_rate, int(all_steps/2)+1)
					x_train = x_train + [np.array([kl_weight] * len(x_train[0]))]


			t1 = time()

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
			
