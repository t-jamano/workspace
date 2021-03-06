import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras import backend as K

K.set_session(session)

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from gensim.models import KeyedVectors
from l3wtransformer import L3wTransformer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import pytrec_eval
import numpy as np
import pickle
import scipy.stats
import math
from nltk.translate.bleu_score import sentence_bleu
from keras_tqdm import TQDMNotebookCallback
from keras_tqdm import TQDMCallback
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, merge
from keras.callbacks import Callback
from scipy import stats
import sys, re, os, os.path
import argparse
from time import time



# Pandas scripts
# df[~df.isnull().any(axis=1)] -> remove null rows
# df[(df.market.str.contains("en-")) & (df.label == 1)] -> where AND

class CosineSim():
    def __init__(self, feature_num):
        q_input = Input(shape=(feature_num,))
        d_input = Input(shape=(feature_num,))

        pred = merge([q_input, d_input], mode="cos")
        self.model = Model([q_input, d_input], pred)


def load_keras_model():
    json_file = open('/work/data/models/dssm_h300_l128_n1_i1_ml10_w50005_b128_2M_50k_trigram_30M_EN_pos_qd_log.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/work/data/models/dssm_h300_l128_n1_i1_ml10_w50005_b128_2M_50k_trigram_30M_EN_pos_qd_log.h5")
    print("Loaded model from disk")

def save_pkl(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def auc(y_test, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def get_cosine_sim(x, y):
    tmp = []
    for i,j in zip(x,y):
        tmp.append(cosine_similarity(i.reshape(1, -1),j.reshape(1, -1)))
    return np.array(tmp).flatten()


def convert_2_trec(query, document, label, isQrel):
    trec = {}
    for i, j, k in zip(query, document, label):
        if i not in trec:
            trec[i] = {}
        if j not in trec[i]:
            trec[i][j] = {}
        trec[i][j] = int(k) if isQrel else float(k)
    return trec

def ranking_measure(qrel, pred):

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {'map', 'ndcg'})

    results = evaluator.evaluate(pred)
    ndcg_score = np.mean([i['ndcg'] for i in results.values()])
    map_score = np.mean([i['map'] for i in results.values()])

    return ndcg_score, map_score

def evaluate(run, test_set, model_name=None):
    may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc = 0, 0, 0, 0, 0, 0
    for q, d, qrel, df, test_data in test_set:
    
        q_ = run.predict(q, batch_size=1024)
        d_ = run.predict(d, batch_size=1024)
        cosine = CosineSim(q_.shape[-1])

        pred = cosine.model.predict([q_, d_]).flatten()
        # pred = run.predict([q, d], batch_size=1024).flatten()
        
        if test_data in ["MayFlower", "JuneFlower"]:
            pred = convert_2_trec(df.q.tolist(), df.d.tolist(), pred, False)
            ndcg_score, map_score = ranking_measure(qrel, pred)

            if test_data == "MayFlower":
                may_ndcg = ndcg_score
            elif test_data == "JuneFlower":
                june_ndcg = ndcg_score

        elif test_data == "JulyFlower":
            july_auc = auc(qrel, pred)
        elif test_data ==  "para":
            para_auc = auc(qrel, pred)
        elif test_data == "quora":
            quora_auc = auc(qrel, pred)
        elif test_data in ["sts"]:
            sts_pcc = scipy.stats.pearsonr(pred, qrel)[0]




    return may_ndcg, june_ndcg, july_auc, quora_auc, para_auc, sts_pcc


def get_reader(train_data, path, iterator=False, batch_size=256):
    train_data_dir = '%sdata/train_data/%s' % (path,train_data)
    if train_data in ["30M_EN_pos_qd_log", "30M_QD.txt", "30M_QD_lower2.txt"]:
        if iterator:
            reader = pd.read_csv(train_data_dir, chunksize=batch_size, iterator=True, usecols=[0,1,2], names=["label","q", "d"], sep="\t", header=None, error_bad_lines=False)
        else:
            reader = pd.read_csv(train_data_dir, usecols=[0,1,2], names=["label","q", "d"], sep="\t", header=None, error_bad_lines=False)
    elif train_data == "1M_EN_QQ_log":
        reader = pd.read_csv(train_data_dir, chunksize=batch_size, iterator=True, usecols=[0,1], names=["q", "d"], sep="\t", header=None, error_bad_lines=False)
    elif train_data == "100M_query":
        reader = pd.read_csv(train_data_dir, chunksize=batch_size, iterator=True, usecols=[0], names=["q"], sep="\t", header=None, error_bad_lines=False)
    elif train_data == "QueryLog":
        if iterator:
            reader = pd.read_csv(train_data_dir, chunksize=batch_size, iterator=True, names=["q"], sep="\t", header=None, error_bad_lines=False)
        else:
            reader = pd.read_csv(train_data_dir, names=["q"], sep="\t", header=None, error_bad_lines=False)
        # reader = pd.read_csv(train_data_dir, nrows=1000, names=["q"], sep="\t", header=None, error_bad_lines=False)
    elif train_data == "QueryQueryLog":
        reader = pd.read_csv(train_data_dir, names=["q", "d", "label"], sep="\t", header=None, error_bad_lines=False)
        # reader = pd.read_csv(train_data_dir, nrows=5000, names=["q", "d", "label"], sep="\t", header=None, error_bad_lines=False)
    return reader

def toBOW(x, nb_words):
    x_ = np.zeros((len(x), nb_words))
    for idx, i in enumerate(x):
        x_[idx][i] = 1
    return x_

def addTags(x, bpe_dict, max_len):
    dec_input = np.copy(x)
    dec_output = np.copy(x)
    for i in range(len(x)):
        dec_input[i] = [bpe_dict['<sos>']] + dec_input[i] + [bpe_dict['<eos>']]
        dec_output[i] = dec_output[i] + [bpe_dict['<eos>']]

    return pad_sequences(dec_input, maxlen=max_len, padding="post"), pad_sequences(dec_output, maxlen=max_len, padding="post")

def sent_generator(reader, tokeniser, batch_size, max_len, feature_num):
    for df in reader:
        q = pad_sequences(tokeniser.texts_to_sequences(df.q.tolist()), maxlen=max_len)

        q_one_hot = to_categorical(q, feature_num)   
        q_one_hot = q_one_hot.reshape(batch_size, max_len, feature_num)
        
        
        yield q, q_one_hot


def generate_output(model, bpe, x, idx=None):

    gen_x = np.argmax(model.predict(x), axis=-1) if idx == None else np.argmax(model.predict(x)[idx], axis=-1)

    results = ""
    for i, k in zip(gen_x, x[0]):
        gen_x = " ".join([bpe.index2word[t] for t in i])
        real_x = " ".join([bpe.index2word[t] for t in k])

        real_x = real_x.replace("▁the", "")
        real_x = real_x.replace("▁","")
        gen_x = gen_x.replace("▁the", "")
        gen_x = gen_x.replace("▁","")

        results = results + "%s : %s" % (real_x, gen_x)
    return results


def get_test_data(filename, path):

    if filename in ["MayFlower", "JuneFlower"]:
        # if "/work/" in path:
        # file_dir = '/data/t-mipha/data/query_similarity_ndcg/%sIdeal.txt' % filename
        file_dir = '%sdata/test_data/%sIdeal.txt' % (path,filename)
        df = pd.read_csv(file_dir, names=["market", "qid", "q", "label", "d", "date"], sep="\t", header=0, error_bad_lines=False)
        df = df.dropna()
        y = np.array([0 if i == "Bad" else 1 if i == "Fare" else 2 for i in df.label.tolist()])

        qrel = convert_2_trec(df.q.tolist(), df.d.tolist(), y, True)

    elif filename in ["JulyFlower"]:
        file_dir = file_dir = '%sdata/test_data/%sIdeal.txt' % (path,filename)
        # file_dir = '/data/chzho/deepqts/test_data/julyflower/julyflower_original.tsv'
        df = pd.read_csv(file_dir, names=["q", "d", "label"], sep="\t", header=0, error_bad_lines=False)
        df = df.dropna()
        qrel = df.label.values

    elif filename == "sts":

        file_dir = '%sdata/train_data/sts-b/test.tsv' % (path)
        df = pd.read_csv(file_dir, sep="\t",encoding='utf-8',quoting=3,  header=None, error_bad_lines=False, usecols=[5,6,4], names=["label","q","d"])
        df = df.dropna()
        qrel = df.label.values

    elif filename == "quora":
        file_dir = "%sdata/train_data/quora/dev.tsv" % path
        df = pd.read_csv(file_dir, sep="\t",encoding='utf-8',quoting=3,  header=None, error_bad_lines=False, usecols=[3,4,5], names=["q","d","label"])
        df = df.dropna()
        qrel = df.label.values

    elif filename == "para":
        file_dir = "%sdata/train_data/para/ParaphraseIdeal.txt" % path
        df = pd.read_csv(file_dir, sep="\t",encoding='utf-8',quoting=3,  header=None, error_bad_lines=False, usecols=[0,1,2], names=["q","d","label"])
        df = df.dropna()
        qrel = df.label.values

    df.q = df.q.astype(str).str.lower()
    df.d = df.d.astype(str).str.lower()

    return df, qrel

def parse_texts(texts, tokeniser, max_len):   

    x = tokeniser.texts_to_sequences(texts)
    x = pad_sequences(x, maxlen=max_len)

    return x

def parse_texts_bpe(texts, sp, bpe_dict, max_len=0, enablePadding=True, padding='pre'):

    x = []
    for text in texts:
        tmp = []

        # with 30M_QD_lower2.txt we dont need these lines
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        try:
            for t in sp.EncodeAsPieces(text):
                if not isinstance(t, str):
                    t = str(t, "utf-8")
                
                if t in bpe_dict:
                    tmp.append(bpe_dict[t])
                else:
                    tmp.append(bpe_dict['<unk>'])
        except Exception as e:
                print(e)
                pass
        x.append(tmp)
    
    return np.array(x) if not enablePadding else pad_sequences(x, maxlen=max_len, padding=padding)


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0/2)

def to_2D_one_hot(x, nb_words):

    x_one_hot = np.zeros((x.shape[0], nb_words))
    for i in range(len(x)):
        x_one_hot[i][x[i]] = 1

    return x_one_hot

def get_data(sup_train_data, test_data, tokenize=None, uns_train_data=None, length_mode="avg", testing=False, nb_words=50000, max_len=15):

    
    if sup_train_data == "1M_qq_log":
        file_dir = '/data/t-mipha/data/agi_encoder/v4/universal/CLICKED_QQ_EN_universal_train_1M.txt'
        if testing:
            df = pd.read_csv(file_dir, nrows=5000, usecols=[0,1], names=["q", "d"], sep="\t", header=None, error_bad_lines=False)
        else:
            df = pd.read_csv(file_dir, usecols=[0,1], names=["q", "d"], sep="\t", header=None, error_bad_lines=False)
        df = df.dropna()

        q_train = df.q.tolist()
        # use only first similar query
        d_train = [i.split("<sep>")[0] for i in df.d.tolist()]
        y_train = np.ones(len(df))
        
    elif sup_train_data == "1M_qd_log":
        print("TODO")
        
    
    if test_data == "MayFlower" or test_data == "JuneFlower":
        file_dir = '/data/t-mipha/data/query_similarity_ndcg/%sIdeal.txt' % test_data
        df_test = pd.read_csv(file_dir, names=["market", "qid", "q", "label", "d", "date"], sep="\t", header=0, error_bad_lines=False)
        df_test = df_test.dropna()
        
        q_test = df_test.q.tolist()
        d_test = df_test.d.tolist()
        y_test = np.array([0 if i == "Bad" else 1 if i == "Fare" else 2 for i in df_test.label.tolist()])
        
        qrel = convert_2_trec(df_test.q.tolist(), df_test.d.tolist(), y_test, True)
        
    elif test_data == "JulyFlower":
        
        file_dir = '/data/t-mipha/data/query_similarity_ndcg/MayFlowerIdeal.txt'
        df_test = pd.read_csv(file_dir, names=["q", "d", "label"], sep="\t", header=None, error_bad_lines=False)
        df_test = df_test.dropna()
        y_test = df_test.label.values
        
    if tokenize != None:

        tokeniser = L3wTransformer(nb_words) if tokenize == "trigram" else Tokenizer(nb_words)
        print("Training Tokeniser")
        max_len = 10 if tokenize == "word" else 20

        # TODO : check if there is pre-trained one
#       only train tokeniser on training set
        texts = q_train + d_train
        tokeniser.fit_on_texts(texts)
        
        q_train = tokeniser.texts_to_sequences(q_train)
        d_train = tokeniser.texts_to_sequences(d_train)
        q_test = tokeniser.texts_to_sequences(q_test)
        d_test = tokeniser.texts_to_sequences(d_test)

    # max_len = np.mean(list(map(len, texts)), dtype='int32') if length_mode == "avg" else np.max(list(map(len, q_train + d_train)))
    
    # print("Average query's length: {}".format(max_len))

    nb_words = tokeniser.max_ngrams + 5 if tokenize == "trigram" else tokeniser.num_words + 1
    
    q_train = pad_sequences(q_train, maxlen=max_len)
    d_train = pad_sequences(d_train, maxlen=max_len)
    q_test = pad_sequences(q_test, maxlen=max_len)
    d_test = pad_sequences(d_test, maxlen=max_len)

        
    return q_train, d_train, y_train, q_test, d_test, y_test, qrel, df, df_test, max_len, nb_words, tokeniser

def ttest(res1, res2, metric="ndcg"):
    res1 = [i[metric] for i in res1.values()]
    res2 = [i[metric] for i in res2.values()]
    print(scipy.stats.ttest_rel(res1, res2))
#   Do tokenize on texts
#     if tokenize = "bpe":
#         model = KeyedVectors.load_word2vec_format("/work/data/bpe/en.wiki.bpe.op200000.d100.w2v.bin", binary=True)
        
#         if test_data != None:
            
#             tmp = []
#             for i in q_test:
#                 tokens = text_to_word_sequence(i)
                
#             [model[] for i in texts]

def generate_reconstruct_query(model, bpe, x, idx=None):

    gen_x = np.argmax(model.predict(x), axis=-1) if idx == None else np.argmax(model.predict(x)[0], axis=-1)
    bleu = []
    results = []
    count = 0
    for i, k in zip(gen_x, x[0]):
        gen_x = " ".join([bpe.index2word[t] for t in i])
        real_x = " ".join([bpe.index2word[t] for t in k])

        bleu.append(sentence_bleu(real_x, gen_x))
        
        real_x = real_x.replace("▁the", "")
        real_x = real_x.replace("▁","")
        gen_x = gen_x.replace("▁the", "")
        gen_x = gen_x.replace("▁","")
        gen_x = gen_x.replace("\t","")
        gen_x = gen_x.replace("<eos>","")

        print("%s\t%s" % (real_x, gen_x))
        results.append("%s\t%s" % (real_x, gen_x))

def output_to_file(model_name, file_output, file_format=".out"):
    with open("/work/data/logs/new/all/%s%s" % (model_name, file_format), "a") as myfile:
        myfile.write(file_output+"\n")

def write_to_files(run, print_output, file_output, path, model_name, model, save=False):
    print(print_output)
    if save:
        with open("%sdata/out/%s" % (path,model_name), "a") as myfile:
            myfile.write(file_output)

        if model in ["bow_pr_aae", "bow_pr_wae", "pr_aae", "pr_wae"]:
            
            run.encoder.save('%sdata/models/%s.encoder.h5' % (path,model_name), overwrite=True)
            run.vae.save('%sdata/models/%s.vae.h5' % (path,model_name), overwrite=True)
        else:

            run.model.save('%sdata/models/%s.h5' % (path,model_name), overwrite=True)
            run.encoder.save('%sdata/models/%s.encoder.h5' % (path,model_name), overwrite=True)
            

class CustomModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, custom_model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CustomModelCheckpoint, self).__init__()
        self.custom_model = custom_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('CustomModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        model = self.custom_model
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            model.save_weights(filepath, overwrite=True)
                        else:
                            model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    model.save_weights(filepath, overwrite=True)
                else:
                    model.save(filepath, overwrite=True)


class EvaluationCheckpoint(Callback):
    
    def __init__(self, run, test_set, model_name, path, graph):
        super(EvaluationCheckpoint, self).__init__()
        
        self.run = run
        self.test_set = test_set
        self.model_name = model_name
        self.path = path
        self.graph = graph


    def on_epoch_begin(self, batch, logs={}):

        self.epoch_time_start = time()

    def on_epoch_end(self, epoch, logs=None):
        
        # print(logs)
        with self.graph.as_default():

            loss = logs.get("loss")
            val_loss = logs.get("val_loss")

            may_ndcg, june_ndcg, july_auc = evaluate(self.run, self.test_set)
            print_output = '%s, Epoch %d, [%.1f s], May = %.4f, June = %.4f, July = %.4f, Loss = %.4f, V_Loss = %.4f \n' % (self.run.name(), epoch, time() - self.epoch_time_start, may_ndcg, june_ndcg, july_auc, loss, val_loss)

            print(print_output)
            with open("%sdata/out/%s" % (self.path, self.model_name), "a") as myfile:
                myfile.write(print_output)


class KL_Annealing(Callback):
    
    def __init__(self, run):
        super(KL_Annealing, self).__init__()
        
        self.run = run
        self.kl_inc_rate = 1 / 5000. # set the annealing rate for KL loss
        self.cos_inc_rate = 1
        self.max_cos_weight = 150.
        self.max_kl_weight = 1.

    def on_epoch_end(self, epoch, logs=None):
        if hasattr(self.run, 'loss_layer'):
            self.run.loss_layer.kl_weight = min(self.run.loss_layer.kl_weight + self.kl_inc_rate, self.max_kl_weight)
        elif hasattr(self.run, "kl_weight"):
            self.run.kl_weight = min(self.run.kl_weight + self.kl_inc_rate, self.max_kl_weight)
            self.run.model.compile(optimizer=self.run.optimizer, loss=["sparse_categorical_crossentropy", self.run.kl_loss])

