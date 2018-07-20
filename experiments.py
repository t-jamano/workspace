from Utils import *
from Models import *
from FastModels import *
from BatchGenerator import *
import datetime

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
    parser.add_argument('--k', type=int, default=2, help='Number of K.')
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
	k = args.k

	out_dir = "%sdata/out/" % path

# 950000
	# train_data_size = {"1M_EN_QQ_log": 950000, "30M_EN_pos_qd_log": 25000000, "100M_query": 10000000, "30M_QD.txt": 20000000}

	train_data_size = {"1M_EN_QQ_log": 950000, "30M_EN_pos_qd_log": 10000, "100M_query": 10000000, "30M_QD.txt": 150000, "30M_QD_lower2.txt":150000, "QueryLog": 4000000, "QueryQueryLog": 3600000}
	eval_every_step = 1000
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
	elif optimizer == "rmsprop":
		optimizer = RMSprop(lr=0.01)

	# load pre-trained tokeniser
	if "BPE" in tokenise_name:

		sp = spm.SentencePieceProcessor()
		sp.Load('%sdata/bpe/en.wiki.bpe.op50000.model' % path)
		bpe = KeyedVectors.load_word2vec_format("%sdata/bpe/en.wiki.bpe.op50000.d200.w2v.bin" % path, binary=True)
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

	elif "trigram" in tokenise_name:

		tokeniser = L3wTransformer()
		tokeniser = tokeniser.load("%sdata/trigram/%s" % (path,tokenise_name))

	# =================================== Initiate Model ==============================================

	if model == "dssm":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_layer, optimizer=optimizer)
		run.initModel(sp, bpe_dict)
	if model == "dssm2":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_layer, optimizer=optimizer, enableSeparate=True)
		run.initModel(sp, bpe_dict)
	if model == "dssm_lstm":
		run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_layer, optimizer=optimizer, enableLSTM=True)
		run.initModel(sp, bpe_dict)



	elif model == "bilstm":
		run = LSTM_Model(hidden_dim, latent_dim, nb_words=nb_words, max_len=max_len, emb=embedding_layer)
		run.initModel(sp, bpe_dict)
	elif model == "bilstm2":
		run = BiLSTM(hidden_dim, latent_dim, nb_words=nb_words, q_max_len=max_len, d_max_len=max_len2, emb=embedding_layer)
		run.initModel(sp, bpe_dict)
	elif model == "bilstm2_cos":
		run = BiLSTM(hidden_dim, latent_dim, nb_words=nb_words, q_max_len=max_len, d_max_len=max_len2, emb=embedding_layer, mode="cos")
		run.initModel(sp, bpe_dict)
	elif model == "vae_dssm":
		run = VAE_DSSM(hidden_dim, latent_dim, nb_words)	
	elif model == "vae_bpe":
		#TODO Frozen or Trainable embedding option
		run = VAE_BPE(hidden_dim, latent_dim, nb_words, max_len, embedding_layer)
		run.initModel(sp, bpe_dict)





	elif model == "vae":
		run = VAE(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], batch_size, optimizer=optimizer)
	elif model == "kate":
		run = VAE(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], batch_size, k, "kcomp", optimizer=optimizer)
	elif model == "kate2":
		run = VAE(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], batch_size, k, "kcomp", optimizer=optimizer, enableKL=False)
	




	elif model == "vdsh":
		run = VDSH(nb_words, max_len, [embedding_layer, bpe2.get_keras_embedding(True)], [hidden_dim, latent_dim], batch_size, optimizer=optimizer)
		run.encoder._make_predict_function()
		graph = tf.get_default_graph()
	elif model == "vdsh_kate":
		run = VDSH(nb_words, max_len, [embedding_layer, bpe2.get_keras_embedding(True)], [hidden_dim, latent_dim], batch_size, k, "kcomp", optimizer=optimizer)
		
	elif model == "seqvae":
		run = SeqVAE(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], optimizer=optimizer, word_dropout_prob=0.5, kl_weight=0.0001)




	elif model == "vae_lstm":
		run = VAE_LSTM(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], optimizer=optimizer)
	elif model == "kate_lstm":
		run = VAE_LSTM(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], optimizer=optimizer)



	elif model == "kate2_bpe":
		run = KATE3D(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], k, "kcomp")
		run.initModel(sp, bpe_dict, bpe.index2word)
	elif model == "kate2_bpeg":
		run = KATE3D(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], k, "kcomp", enableGAN=True)
		run.initModel(sp, bpe_dict)
		run.encoder._make_predict_function()
		graph = tf.get_default_graph()
	elif model == "kate2_bpe_adam":
		run = KATE3D(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], k, "kcomp", optimizer=optimizer)
		run.initModel(sp, bpe_dict)

	elif model == "aae":
		run = AAE(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], k, "kcomp")
		run.initModel(sp, bpe_dict)

	elif model == "kate1_qd":
		run = VarAutoEncoderQD(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], k)
		run.initModel(sp, bpe_dict)
	elif model == "kate2_qd":
		run = VarAutoEncoderQD(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], k, "kcomp", alpha=alpha, optimizer=optimizer)
		run.initModel(sp, bpe_dict)
	elif model == "kate2_qd2":
		run = VarAutoEncoderQD2(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], k, "kcomp", alpha=alpha, optimizer=optimizer)
		run.initModel(sp, bpe_dict)
	elif model == "kate2_qd3_dssm":
		run = VarAutoEncoderQD3(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], k, "kcomp", alpha=alpha, optimizer=optimizer)
		run.initModel(sp, bpe_dict)
	elif model == "kate2_qdc":
		run = VarAutoEncoderQD(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], k, "kcomp", enableCross=True)
		run.initModel(sp, bpe_dict)
	elif model == "kate2_qdm":
		run = VarAutoEncoderQD(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], k, "kcomp", enableMemory=True)
		run.initModel(sp, bpe_dict)
	elif model in ["kate2_qdg1", "kate2_qdg2"]:
		run = VarAutoEncoderQD(nb_words, max_len, embedding_layer, [hidden_dim, latent_dim], k, "kcomp", enableGAN=True)
		run.initModel(sp, bpe_dict)
		if model == "kate2_qdg2":
			run.discriminator.trainable = False
		run.encoder._make_predict_function()
		graph = tf.get_default_graph()
			

	run.encoder._make_predict_function()
	graph = tf.get_default_graph()

	model_name = "%s_h%d_l%d_k%d_n%d_ml%d_w%d_b%d_e%d_a%.1f_%s_%s_%s_%s" % (model, hidden_dim, latent_dim, k, num_negatives, max_len, nb_words, batch_size, epochs, alpha, optimizer, tokenise_name, train_data, date_time)

	

	# =================================== Get testing data ==============================================


	df_may, qrel_may = get_test_data("MayFlower", path)
	df_june, qrel_june = get_test_data("JuneFlower", path)
	df_july, qrel_july = get_test_data("JulyFlower", path)


	# if model in ["kate"]:
	# 	enablePadding = True

	# 	q_may = to_2D_one_hot(parse_texts_bpe(df_may.q.tolist(), sp, bpe_dict, max_len, enablePadding=enablePadding), nb_words)
	# 	d_may = to_2D_one_hot(parse_texts_bpe(df_may.d.tolist(), sp, bpe_dict, max_len, enablePadding=enablePadding), nb_words)

	# 	q_june = to_2D_one_hot(parse_texts_bpe(df_june.q.tolist(), sp, bpe_dict, max_len, enablePadding=enablePadding), nb_words)
	# 	d_june = to_2D_one_hot(parse_texts_bpe(df_june.d.tolist(), sp, bpe_dict, max_len, enablePadding=enablePadding), nb_words)

	# 	q_july = to_2D_one_hot(parse_texts_bpe(df_july.q.tolist(), sp, bpe_dict, max_len, enablePadding=enablePadding), nb_words)
	# 	d_july = to_2D_one_hot(parse_texts_bpe(df_july.d.tolist(), sp, bpe_dict, max_len, enablePadding=enablePadding), nb_words)

	# else:

	enablePadding = True

	q_may = parse_texts_bpe(df_may.q.tolist(), sp, bpe_dict, max_len, enablePadding)
	d_may = parse_texts_bpe(df_may.d.tolist(), sp, bpe_dict, max_len, enablePadding)

	q_june = parse_texts_bpe(df_june.q.tolist(), sp, bpe_dict, max_len, enablePadding)
	d_june = parse_texts_bpe(df_june.d.tolist(), sp, bpe_dict, max_len, enablePadding)

	q_july = parse_texts_bpe(df_july.q.tolist(), sp, bpe_dict, max_len, enablePadding)
	d_july = parse_texts_bpe(df_july.d.tolist(), sp, bpe_dict, max_len, enablePadding)


	test_set = [[q_may, d_may, qrel_may, df_may, "MayFlower"], [q_june, d_june, qrel_june, df_june, "JuneFlower"], [q_july, d_july, qrel_july, df_july, "JulyFlower"]]

	# test_set = [[q_july, d_july, qrel_july, df_july, "JulyFlower"]]

	if model in ["vae", "kate", "kate2"]:
		reader = get_reader(train_data, batch_size, path)
		idx = int(len(reader) - ( len(reader) % batch_size))

		batches = int(idx / (batch_size * 1.0))
		val_idx = int(0.1 * batches) * batch_size
		# print(idx, val_idx, batches)

		x_train = parse_texts_bpe(reader.q.tolist()[:idx - val_idx], sp, bpe_dict, max_len, enablePadding=True)
		x_val = parse_texts_bpe(reader.q.tolist()[idx - val_idx: idx], sp, bpe_dict, max_len, enablePadding=True)
		
		y_train = x_train
		y_val = x_val


		print("Train %d : Val %d" % (len(x_train), len(x_val)))

	elif model in ["vae_lstm", "kate_lstm", "seqvae", "seqvae2"]:

		train_data_dir = '%sdata/train_data/%s.npy' % (path,train_data)
		if os.path.exists(train_data_dir):
			x_train = np.load(train_data_dir)
		else:
			reader = get_reader(train_data, path)
			x_train = parse_texts_bpe(reader.q.tolist(), sp, bpe_dict, max_len, enablePadding=True)
		
		y_train =[np.expand_dims(x_train, axis=-1), np.expand_dims(x_train, axis=-1)]
		if model in ["seqvae", "seqvae2"]:
			x_train = [x_train, x_train]

	elif "vdsh" in model or "dssm" in model:

		reader = get_reader(train_data, batch_size, path)
		idx = int(len(reader) - ( len(reader) % batch_size))
		batches = int(idx / (batch_size * 1.0))
		val_idx = int(0.1 * batches) * batch_size

		q_train = parse_texts_bpe(reader.q.tolist()[:idx - val_idx], sp, bpe_dict, max_len, enablePadding=True)
		d_train = parse_texts_bpe(reader.d.tolist()[:idx - val_idx], sp, bpe_dict, max_len, enablePadding=True)

		q_val = parse_texts_bpe(reader.q.tolist()[idx - val_idx: idx], sp, bpe_dict, max_len, enablePadding=True)
		d_val = parse_texts_bpe(reader.d.tolist()[idx - val_idx: idx], sp, bpe_dict, max_len, enablePadding=True)


		idx_train = np.arange(len(q_train))
		shuffle(idx_train)

		idx_val = np.arange(len(q_val))
		shuffle(idx_val)


		if "dssm" in model:

			x_train = [q_train, d_train, d_train[idx_train]]
			y_train = np.zeros((len(q_train), 2))
			y_train[:, 0] = 1

			x_val = [q_val, d_val, d_val[idx_val]]
			y_val = np.zeros((len(q_val), 2))
			y_val[:, 0] = 1
		else:

			q_train = np.concatenate([q_train, q_train])
			d_train = np.concatenate([d_train, d_train[idx_train]])

			idx = np.arange(len(q_val))
			shuffle(idx)

			q_val = np.concatenate([q_val, q_val])
			d_val = np.concatenate([d_val, d_val[idx_val]])

			x_train = [q_train, d_train]
			y_train =[q_train, np.concatenate([np.ones(int(len(q_train)/2.0)), np.zeros(int(len(q_train)/2.0))])]

			x_val = [q_val, d_val]
			y_val = [q_val, np.concatenate([np.ones(int(len(q_val)/2.0)), np.zeros(int(len(q_val)/2.0))])]

		# print("Train %d : Val %d" % (len(x_train), len(x_val)))


	print("============Start Training================")
	cosine = CosineSim(latent_dim)


	if model in ["vae", "kate", "vdsh", "vdsh_kate", "kate2", "dssm", "dssm2", "dssm_lstm"]:
		try:
			hist = run.model.fit(x_train, y_train,
						        shuffle=True,
						        epochs=epochs,
						        verbose=0,
						        batch_size=batch_size,
						        validation_data=(x_val, y_val),
						        callbacks=[ ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
						                    # TQDMCallback(),
						                    EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
						                    CustomModelCheckpoint(run.encoder, '%sdata/models/fastModels/%s.h5' % (path,model_name), monitor='val_loss', save_best_only=True, mode='auto'),
						                    CustomModelCheckpoint(run.model, '%sdata/models/fastModels/%s.encoder.h5' % (path,model_name), monitor='val_loss', save_best_only=True, mode='auto'),
						                    EvaluationCheckpoint(run, cosine, test_set, model_name, path, graph),
						                    KL_Annealing(run)
						                  ]
						        )
		except Exception as e:
				print(e)
				pass

	elif model in ["vae_lstm", "kate_lstm", "seqvae", "seqvae2"]:
		try:
			hist = run.model.fit(x_train, y_train,
						        shuffle=True,
						        epochs=epochs,
						        verbose=0,
						        batch_size=batch_size,
						        validation_split=0.2,
						        callbacks=[ ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
						                    # TQDMCallback(),
						                    EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
						                    CustomModelCheckpoint(run.encoder, '%sdata/models/fastModels/%s.h5' % (path,model_name), monitor='val_loss', save_best_only=True, mode='auto'),
						                    CustomModelCheckpoint(run.model, '%sdata/models/fastModels/%s.encoder.h5' % (path,model_name), monitor='val_loss', save_best_only=True, mode='auto'),
						                    EvaluationCheckpoint(run, cosine, test_set, model_name, path, graph),
						                    KL_Annealing(run)
						                  ]
						        )
		except Exception as e:
				print(e)
				pass
	else:

		best_auc_score = 0

		iterations = int(train_data_size[train_data] / batch_size)
		for epoch in range(epochs):		


	# 		Run models that cannot fit all training dataset into memory
			# restart the reader thread
			reader = get_reader(train_data, batch_size, path)
			reader2 = get_reader(train_data, batch_size, path)

			
			for iteration in range(int(iterations / eval_every_step)):

				
				try:
					if model in ["kate2_qdg1", "kate2_qdg2", "kate2_bpeg"]:
						t1 = time()
						hist = run.model.fit_generator(run.batch_generator(reader, train_data, batch_size), steps_per_epoch=eval_every_step, epochs=1, verbose=0)       
						hist_dis = run.discriminator.fit_generator(run.batch_GAN_generator(reader2, train_data, batch_size, graph), steps_per_epoch=eval_every_step, epochs=1, verbose=0)       
						t2 = time()
						losses = ', '.join([ "%s = %.4f" % (k, hist.history[k][-1]) for k in hist.history] + [ "%s = %.4f" % (k, hist_dis.history[k][-1]) for k in hist_dis.history])
						print(losses)

					else:
						t1 = time()

						hist = run.model.fit_generator(run.batch_generator(reader, train_data, batch_size), steps_per_epoch=eval_every_step, epochs=1, verbose=0)       
						t2 = time()
						losses = ', '.join([ "%s = %.4f" % (k, hist.history[k][-1]) for k in hist.history])

					may_ndcg, june_ndcg, july_auc = evaluate(run, cosine, test_set)

					# generate output
					# run.generate_output(q_july, d_july)

					

					print_output = '%s_a%.1f_k%d, Epoch %d, Iteration %d, [%.1f s], May = %.4f, June = %.4f, July = %.4f, Loss = %.4f, [%.1f s] \n' % (model, alpha, k, epoch, (iteration+1)*eval_every_step, t2-t1, may_ndcg, june_ndcg, july_auc, hist.history['loss'][-1], time()-t2)
					file_output = '%s_a%.1f_k%d, Epoch %d, Iteration %d, [%.1f s], May = %.4f, June = %.4f, July = %.4f, %s, [%.1f s] \n' % (model, alpha, k, epoch, (iteration+1)*eval_every_step, t2-t1, may_ndcg, june_ndcg, july_auc, losses, time()-t2)

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
