from Utils import *
from Models import *
from FastModels import *
from AdversarialModels import *
from BatchGenerator import *
from BOW_Models import *
from SemiSupervisedModels import *
from vae import *
from aae import *
from s2s import *
from dssm_vae import *
from dssm_aae import *

from dssm_vae2 import *

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
    parser.add_argument('--pre', type=int, default=0, help='Initialise model with pre-trained vae')
    parser.add_argument('--save', type=int, default=0, help='Save model')
    parser.add_argument('--limit', type=int, default=1, help='Size of Labelled data')



    parser.add_argument('--af', type=str, default='logistic', help='anneal_function: logistic or linear')
    parser.add_argument('--klrate', type=float, default=0.01, help='KL anneal rate')
    parser.add_argument('--wd', type=float, default=0.75, help='keep_rate_word_dropout')

    parser.add_argument('--qd', type=str, default='q', help='train (q)uery or (d)ocument encoder')



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
	qd = args.qd
	optimizer=args.o
	batch_size = args.b
	max_len = args.ml
	mode = args.m
	limit = args.limit
	pre = True if args.pre == 1 else False
	save = True if args.save == 1 else False
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
		# optimizer = Adadelta(lr=2.)


	# load pre-trained tokeniser
	if "BPE" in tokenise_name:

		sp = spm.SentencePieceProcessor()
		vsize = 50000 if "v200k" not in train_data else 200000
		sp.Load('%sdata/bpe/en.wiki.bpe.op%d.model' % (path, vsize))
		bpe = KeyedVectors.load_word2vec_format("%sdata/bpe/en.wiki.bpe.op%d.d200.w2v.bin" % (path, vsize), binary=True)
		if pre:
			bpe.index2word = [''] + bpe.index2word + ['<sos>'] + ['<eos>']   # add empty string 
		else:
			bpe.index2word = [''] + bpe.index2word + ['<sos>'] + ['<eos>'] + ['<drop>']   # add empty string 

		nb_words = len(bpe.index2word)
		# word2index
		bpe_dict = {bpe.index2word[i]: i for i in range(len(bpe.index2word))}
		# construct embedding_matrix
		if pre:
			embedding_matrix = np.concatenate([np.zeros((1, bpe.vector_size)), bpe.vectors, np.zeros((2, bpe.vector_size))]) # add zero vector for empty string (i.e. used for padding)
		else:
			embedding_matrix = np.concatenate([np.zeros((1, bpe.vector_size)), bpe.vectors, np.zeros((3, bpe.vector_size))]) # add zero vector for empty string (i.e. used for padding)

	elif "trigram" in tokenise_name:

		tokeniser = L3wTransformer()
		tokeniser = tokeniser.load("%sdata/trigram/%s" % (path,tokenise_name))

	# =================================== Initiate Model ==============================================

	if model in ["dssm", "dssm_hybrid", "dssm_max", "dssm_s"]:
		if "QQ" in train_data:
			enableSeparate = False
		elif "QD" in train_data:
			enableSeparate = True
		if model in ["dssm", "dssm_s"]:
			run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableLSTM=True, enableSeparate=enableSeparate)
		elif model == "dssm_hybrid":
			run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableLSTM=True, enableSeparate=enableSeparate, enableHybrid=True)
		elif model == "dssm_max":
			run = DSSM(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableLSTM=False, enableSeparate=enableSeparate)

	elif model == "dssm_bpe":
		run = DSSMClassifier(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, mode="bpe")
	elif model == "dssm_pre_bpe":
		run = DSSMClassifier(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, mode="bpe", trainable=True)

	elif model in ["dssm_ae", "dssm_pre_ae"]:
		if "pre" in model:
			run = DSSMClassifier(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, mode="ae", trainable=True)
		else:
			run = DSSMClassifier(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, mode="ae")
		# pre_ae = load_model("/work/data/logs/new/aae_m1_200M_QQ_ml15_limit1_2018_09_01_02:08:34.encoder.h5")
		# pre_ae = load_model("/work/data/logs/new/all/kate_bow_m1_200M_QQ_ml15_limit1_2018_09_02_03:10:46.encoder.h5", custom_objects={"KCompetitive": KCompetitive})
		# pre_ae = load_model("/work/data/logs/new/all/vae_bow_m1_200M_QQ_ml15_limit1_2018_09_02_02:00:57.encoder.h5")
		# pre_ae = load_model("/work/data/logs/new/wae_m1_200M_QQ_ml15_limit1_2018_09_01_02:22:09.encoder.h5")
		# pre_ae = load_model("/work/data/logs/new/vae_kl_m1_200M_QQ_ml15_limit1_2018_09_01_02:31:47.encoder.h5")
		# pre_ae = load_model("/work/data/logs/new/all/aae_m1_200M_QQ_ml15_limit1_2018_09_02_21:12:21.last.encoder.h5")
		# pre_ae =load_model("/work/data/logs/new/all/wae_m1_200M_QQ_ml15_limit1_2018_09_02_21:11:59.last.encoder.h5")
		pre_ae = load_model("/work/data/logs/new/vae_m1_200M_QQ_ml15_limit1_2018_09_01_02:22:16.encoder.h5")
		for layer in ["q_embedding", "q_gru", "q_dense"]:
				run.model.get_layer(layer).set_weights(pre_ae.get_layer(layer).get_weights())

	elif model in  ["dssm_bpe_ae", "dssm_pre_bpe_ae"]:
		if "pre" in model:
			run = DSSMClassifier(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, mode="bpe_ae", trainable=True)
		else:
			run = DSSMClassifier(hidden_dim, latent_dim, num_negatives, nb_words, max_len, embedding_matrix, optimizer=optimizer, mode="bpe_ae")
		# pre_ae = load_model("/work/data/logs/new/vae_kl_m1_200M_QQ_ml15_limit1_2018_09_01_02:31:47.encoder.h5")
		pre_ae = load_model("/work/data/logs/new/aae_m1_200M_QQ_ml15_limit1_2018_09_01_02:08:34.encoder.h5")
		# pre_ae = load_model("/work/data/logs/new/all/kate_bow_m1_200M_QQ_ml15_limit1_2018_09_02_03:10:46.encoder.h5")
		for layer in ["q_embedding", "q_gru", "q_dense"]:
				run.model.get_layer(layer).set_weights(pre_ae.get_layer(layer).get_weights())




	elif model == "binary":
		run = BinaryClassifier(hidden_dim, latent_dim, nb_words, max_len, embedding_matrix, optimizer=optimizer, enableLSTM=True)

	elif model == "aae":
	    run = AdversarialAutoEncoderModel(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, keep_rate_word_dropout=keep_rate_word_dropout, enableWasserstein=False, mode=mode)

	elif model == "dssm_aae_s":
		run = DSSM_AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, keep_rate_word_dropout=keep_rate_word_dropout, enableSemi=True, limit=limit)

	elif model == "wae":
	    run = AdversarialAutoEncoderModel(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, keep_rate_word_dropout=keep_rate_word_dropout, enableWasserstein=True, mode=mode)

	elif model == "vae":
		run = VariationalAutoEncoder(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, enableKL=False, enableCond=False)
	elif model == "vae_bow":
		run = VariationalAutoEncoderBOW(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, enableKL=False, enableCond=False)
	elif model == "vae_kl":
		run = VariationalAutoEncoder(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, enableKL=True)

	elif model == "vae_bi":
		run = VariationalAutoEncoder(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, enableKL=False, enableCond=False, enableBi=True, mode=mode)
	elif model == "vae_max":
		run = VariationalAutoEncoder(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, enableKL=False, enableCond=False, enableGRU=False)

	elif model == "s2s":
		run = Seq2Seq(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, keep_rate_word_dropout=keep_rate_word_dropout)



	elif model == "vaes":
		run = VariationalAutoEncoder(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, enableKL=False, enableCond=False)		
	elif model == "cvae":
		run = VariationalAutoEncoder(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, enableKL=False, enableCond=True)


	elif model == "dssm_vae":
		run = DSSM_VAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, mode=mode)
	elif model == "dssm_vae_s":
		run = DSSM_VAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, mode=mode, enableSemi=True)
	elif model == "dssm_vae_max_s":
		run = DSSM_VAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, mode=mode, enableSemi=True, enableGRU=False)
	elif model == "dssm_vae_max":
		run = DSSM_VAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, mode=mode, enableGRU=False)
	elif model == "dssm_aae":
		run = DSSM_AAE(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, mode=mode)
	elif model == "dssm_aae_ss":
		run = DSSM_AAE_SS(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, mode=mode)
	elif model == "dssm_vae_pr":
		run = DSSM_VAE_PR(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer, mode=mode)
	elif model == "dssm_vae2":
		run = DSSM_VAE2(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer)	

	elif model == "ss_pr_aae":
		run = SS_PR(nb_words, max_len, embedding_matrix, [hidden_dim, latent_dim], optimizer=optimizer, mode=mode, enableS2S=False, enableGRU=True, enableWasserstein=False)


	run.encoder._make_predict_function()
	graph = tf.get_default_graph()

	limit = limit if model in ["dssm_s", "dssm_aae_s", "dssm_aae_ss"] else 1
	model_name = "%s_m%d_%s_limit%d_%s" % (run.name(), mode, train_data, limit, date_time)

	isSemiExperiment = True if model in ["dssm_aae_s", "dssm_s", "dssm_aae_ss"] else False

	# =================================== Get testing data ==============================================
	test_set = []
	for i in ["MayFlower", "JuneFlower", "JulyFlower", "sts", "quora", "para"]:
		df, qrel = get_test_data(i, path)
		q_ = parse_texts_bpe(df.q.tolist(), sp, bpe_dict, max_len, enablePadding)
		d_ = parse_texts_bpe(df.d.tolist(), sp, bpe_dict, max_len, enablePadding)
		test_set.append([q_, d_, qrel, df, i])

	# # =================================== Get Validation data ==============================================
	# if qd == "q":
	# 	q_enc_inputs = np.load('%sdata/train_data/%s.q.npy' % (path,train_data))
	# 	q_dec_inputs = np.load('%sdata/train_data/%s.q.di.npy' % (path,train_data))
	# 	q_dec_outputs = np.load('%sdata/train_data/%s.q.do.npy' % (path,train_data))
	# elif qd == "d" and model != "dssm":
	# 	q_enc_inputs = np.load('%sdata/train_data/%s.d.npy' % (path,train_data))
	# 	q_dec_inputs = np.load('%sdata/train_data/%s.d.di.npy' % (path,train_data))
	# 	q_dec_outputs = np.load('%sdata/train_data/%s.d.do.npy' % (path,train_data))



	# d_enc_inputs = np.load('%sdata/train_data/%s.d.npy' % (path,train_data))
	# d_dec_inputs = np.load('%sdata/train_data/%s.d.di.npy' % (path,train_data))
	# d_dec_outputs = np.load('%sdata/train_data/%s.d.do.npy' % (path,train_data))

	# labels = np.load('%sdata/train_data/%s.label.npy' % (path,train_data))

	

	if limit > 1:

		q_s_enc_inputs = np.load('%sdata/train_data/%d_QQ_ml15.q.npy' % (path, limit))
		d_s_enc_inputs = np.load('%sdata/train_data/%d_QQ_ml15.d.npy' % (path, limit))

		q_v_enc_inputs = np.load('%sdata/train_data/50K_QQ_ml15.q.v.npy' % (path))[:2560]
		d_v_enc_inputs = np.load('%sdata/train_data/50K_QQ_ml15.d.v.npy' % (path))[:2560]

		semi_num = limit

		v_idx = np.arange(len(q_v_enc_inputs))
		shuffle(v_idx)

		x_val = [q_v_enc_inputs, d_v_enc_inputs, d_v_enc_inputs[v_idx]]
		y_val = np.zeros((len(q_v_enc_inputs), 2))
		y_val[:, 0] = 1

		semi_num = len(d_s_enc_inputs)
		semi_idx = np.arange(semi_num)

		# print(q_s_enc_inputs.shape, d_s_enc_inputs.shape)

	# labels = np.load('%sdata/train_data/%s.label.npy' % (path,train_data))[:100]


	# real = np.ones((len(q_enc_inputs), 1))
	# fake = np.zeros((len(q_enc_inputs), 1)) if "wae" not in model else -valid

	print("============Start Training================")
	min_val_loss = 9999999

	may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = 0, 0, 0, 0, 0, 0

	batch_size = 256

	step = 0
	# if not isSemiExperiment:

	# 	if model == "dssm_s":

	# 		for epoch in range(100):
	# 			t1 = time()
	# 			for i in range(0, len(q_s_enc_inputs), batch_size):
	# 				q = q_s_enc_inputs[i: i + batch_size]
	# 				d = d_s_enc_inputs[i: i + batch_size]
	# 				train_num = len(q)

	# 				mat = np.matmul(run.encoder.predict(q), run.encoder.predict(d).T)
	# 				idx = []
	# 				mul = np.argsort(mat)
	# 				for j in range(mat.shape[0]):
	# 					idx.append(mul[j][-1] if mul[j][-1] != j else mul[j][-2])

	# 				x_train = [q, d, d[idx]]
	# 				y_train = np.zeros((train_num, 2))
	# 				y_train[:, 0] = 1

	# 				loss = run.model.train_on_batch(x_train, y_train)
				
	# 			may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = evaluate(run.encoder, test_set)
	# 			val_loss = run.model.test_on_batch(x_val, y_val)
	# 			t2 = time()
	# 			outputs = "%s,%.1fs,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (model, t2-t1, loss, val_loss, may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc)
	# 			print(outputs)
	# 			if min_val_loss > val_loss:
	# 				min_val_loss = val_loss
	# 			else:
	# 				break
	# else:


	step = 0
	kl_step = 0
	max_july = 0
	semi_step = 0
	t1 = time()
	for df in pd.read_csv("/data/t-mipha/agi_encoder_recipe/datasets/query_logs/CLICKED_QQ_MUL_2017-01-01_2017-06-10_r_train_ASCIIonly.txt", iterator=True, chunksize=batch_size, sep="\t", header=None, names=['q', 'd', 'label', 'feature', 'null']):
		
		df = df.dropna()
		df.d = [i.split("<sep>")[0] for i in df.d.tolist()]
		train_num = len(df)

		if train_num < batch_size:
			continue
	
		enablePadding = False
		q_df = parse_texts_bpe(df.q.tolist(), sp, bpe_dict, max_len, enablePadding, "post")
		d_df = parse_texts_bpe(df.d.tolist(), sp, bpe_dict, max_len, enablePadding, "post")

		q_dec_inputs, q_dec_outputs = addTags(q_df, bpe_dict, max_len)
		d_dec_inputs, d_dec_outputs = addTags(d_df, bpe_dict, max_len)

		enablePadding = True
		q_enc_inputs = parse_texts_bpe(df.q.tolist(), sp, bpe_dict, max_len, enablePadding, "post")
		d_enc_inputs = parse_texts_bpe(df.d.tolist(), sp, bpe_dict, max_len, enablePadding, "post")

		if "dssm" in model or "binary" in model:

			if isSemiExperiment:
				# shuffle(semi_idx)
				# semi_q_inputs = q_s_enc_inputs[semi_idx][:train_num]
				# semi_d_inputs = d_s_enc_inputs[semi_idx][:train_num]
				semi_q_inputs = q_s_enc_inputs[semi_step: semi_step + train_num]
				semi_d_inputs = d_s_enc_inputs[semi_step: semi_step + train_num]

				mat = np.matmul(run.encoder.predict(semi_q_inputs), run.encoder.predict(d_enc_inputs).T)
				mul = np.argsort(mat)
				idx = mul[:, -1]
			else:
				mat = np.matmul(run.encoder.predict(q_enc_inputs), run.encoder.predict(d_enc_inputs).T)
				idx = []
				mul = np.argsort(mat)
				for j in range(mat.shape[0]):
					idx.append(mul[j][-1] if mul[j][-1] != j else mul[j][-2])


		if model in ["dssm_s", "dssm_aae_s", "dssm_aae_ss"]:
			x_train = [semi_q_inputs, semi_d_inputs, d_enc_inputs[idx]]
			y_train = np.zeros((train_num, 2))
			y_train[:, 0] = 1

			if model == "dssm_aae_s":
				ae_x_train = [d_enc_inputs[idx], run.word_dropout(d_dec_inputs[idx], bpe_dict['<drop>'])]
				y_ = np.expand_dims(d_dec_outputs[idx], axis=-1)
				real = np.ones((len(y_), 1))
				fake = np.zeros((len(y_), 1)) if "wae" not in model else -real
				ae_y_train = [y_, real, fake, y_, fake, real]

				loss = run.semi_model.train_on_batch(ae_x_train, ae_y_train)

			elif model == "dssm_aae_ss":
				x_train = x_train + [run.word_dropout(d_dec_inputs[idx], bpe_dict['<drop>'])]
				y_ = np.expand_dims(d_dec_outputs[idx], axis=-1)
				real = np.ones((len(y_), 1))
				fake = np.zeros((len(y_), 1)) if "wae" not in model else -real
				y_train = [y_, y_train, real, fake, y_, y_train, fake, real]

		elif model in ["vae", "vae_kl"]:
			x_train = [q_enc_inputs, run.word_dropout(q_dec_inputs, bpe_dict['<drop>'])]
			y_train = np.expand_dims(q_dec_outputs, axis=-1)

			if "kl" in model:
				kl_weight = kl_anneal_function(anneal_function, kl_step, 0.005, int(3900/2))
				x_train = x_train + [np.array([kl_weight] * len(x_train[0]))]
				# x_train = x_train + [np.array([0] * len(x_train[0]))]

		elif model in ["vae_bow"]:
			x_train = q_enc_inputs
			y_train = toBOW(q_enc_inputs, nb_words)

		elif model in ["aae", "wae"]:
			x_train = [q_enc_inputs, run.word_dropout(q_dec_inputs, bpe_dict['<drop>'])]
			y_train = np.expand_dims(q_dec_outputs, axis=-1)
			real = np.ones((len(q_enc_inputs), 1))
			fake = np.zeros((len(q_enc_inputs), 1)) if "wae" not in model else -real
			y_train = [y_train, real, fake, y_train, fake, real]

		elif model in ["s2s"]:
			x_train = [q_enc_inputs, run.word_dropout(d_dec_inputs, bpe_dict['<drop>'])]
			y_train = np.expand_dims(d_dec_outputs, axis=-1)

		elif model in ["dssm", "dssm_aae", "dssm_bpe", "dssm_bpe_ae", "dssm_ae", "dssm_pre_bpe", "dssm_pre_ae", "dssm_pre_bpe_ae"]:

			x_train = [q_enc_inputs, d_enc_inputs, d_enc_inputs[idx]]
			y_train = np.zeros((train_num, 2))
			y_train[:, 0] = 1

			if model in ["dssm_aae"]:

				x_train = x_train + [run.word_dropout(d_dec_inputs, bpe_dict['<drop>'])]
				y_ = np.expand_dims(d_dec_outputs, axis=-1)
				real = np.ones((len(y_), 1))
				fake = np.zeros((len(y_), 1)) if "wae" not in model else -real
				y_train = [y_, y_train, real, fake, y_, y_train, fake, real]


		try:
			csv_logger = CSVLogger('/work/data/logs/new/all/%s.model.csv' % model_name, append=True, separator=';')
			hist = run.model.fit(x_train, y_train, batch_size=train_num, verbose=0, shuffle=False, nb_epoch=1, callbacks=[csv_logger])
				
			condition = step % limit == 0 if model in ["dssm_s", "dssm_aae_s", "dssm_aae_ss"] else step % (batch_size * 100) == 0
			
			if condition and step != 0 :

				may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = evaluate(run.encoder, test_set)
				# val_loss = run.model.test_on_batch(x_val, y_val)
				loss = hist.history['loss'][-1]
				if model in ["dssm_s", "dssm_aae_s", "dssm_aae_ss"]:
					if model == "dssm_aae_ss":
						val_loss = run.dssm.evaluate(x_val, y_val, batch_size=train_num, verbose=0)
					else:
						val_loss = run.model.evaluate(x_val, y_val, batch_size=train_num, verbose=0)
					t2 = time()
					outputs = "%s,%.1fs,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (run.name(), t2-t1, step, loss, val_loss, may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc)
				else:
					t2 = time()
					outputs = "%s,%.1fs,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (run.name(), t2-t1, step, loss, may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc)
					

				# print(hist.history)
				print(outputs)

				output_to_file(model_name, outputs, file_format=".res")

				
				# print(may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc)
				if max_july < july_auc:
					max_july = july_auc
					run.encoder.save('/work/data/logs/new/all/%s.encoder.h5' % (model_name), overwrite=True)

				# save last model
				run.encoder.save('/work/data/logs/new/all/%s.last.encoder.h5' % (model_name), overwrite=True)


				t1 = time()

				if model in ["dssm_s", "dssm_aae_s", "dssm_aae_ss"]:
					if min_val_loss > val_loss:
						min_val_loss = val_loss
					else:
						break

				if model in ["dssm_s", "dssm_aae_s", "dssm_aae_ss"] and (step / limit) >= 100:
					break

		except Exception as e:
			print(e)
			pass

		

			

		step = step + batch_size
		semi_step = semi_step + batch_size if semi_step < limit - batch_size else 0
		kl_step = kl_step + 1






