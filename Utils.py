from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
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
from keras_tqdm import TQDMNotebookCallback
from keras_tqdm import TQDMCallback
from keras.models import model_from_json
import sys
import argparse


# Pandas scripts
# df[~df.isnull().any(axis=1)] -> remove null rows
# df[(df.market.str.contains("en-")) & (df.label == 1)] -> where AND


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

def evaluate(run, cosine, test_set, best_auc_score, model_name):

    for q, d, qrel, df, test_data in test_set:
    
        pred = cosine.model.predict([run.encoder.predict(q), run.encoder.predict(d)])
        print(test_data)
        if test_data in ["MayFlower", "JuneFlower"]:
            pred = convert_2_trec(df.q.tolist(), df.d.tolist(), pred, False)
            ndcg_score, map_score = ranking_measure(qrel, pred)
            print("NDCG: %f" % ndcg_score)
            print("MAP: %f" % map_score)
            save_pkl(pred, '/work/data/res/%s_%s_%f_%f.pkl' % (model_name, test_data, ndcg_score, map_score))  
            with open("/work/data/out/%s" % (model_name), "a") as myfile:
                myfile.write("NDCG: %f\n" % ndcg_score)
                myfile.write("MAP: %f\n" % map_score)

        elif test_data in ["JulyFlower"]:
            auc_score = auc(qrel, pred.flatten())
            print("AUC: %f" % auc_score)
            save_pkl(pred.flatten(), '/work/data/res/%s_%s_%f.pkl' % (model_name, test_data, auc_score))
            with open("/work/data/out/%s" % (model_name), "a") as myfile:
                myfile.write("AUC: %f\n" % auc_score)


            if auc_score > best_auc_score:
                best_auc_score = auc_score
                run.model.save('/work/data/models/%s.h5' % model_name)
                run.encoder.save('/work/data/models/%s.encoder.h5' % model_name)

    return best_auc_score


def get_reader(train_data, batch_size):
    train_data_dir = '/work/data/train_data/%s' % train_data
    if train_data in ["30M_EN_pos_qd_log", "30M_QD.txt"]:
        reader = pd.read_csv(train_data_dir, chunksize=batch_size, iterator=True, usecols=[0,1,2], names=["label","q", "d"], sep="\t", header=0, error_bad_lines=False)
    elif train_data == "1M_EN_QQ_log":
        reader = pd.read_csv(train_data_dir, chunksize=batch_size, iterator=True, usecols=[0,1], names=["q", "d"], sep="\t", header=None, error_bad_lines=False)
    elif train_data == "100M_query":
        reader = pd.read_csv(train_data_dir, chunksize=batch_size, iterator=True, usecols=[0], names=["q"], sep="\t", header=None, error_bad_lines=False)
    return reader



def sent_generator(reader, tokeniser, batch_size, max_len, feature_num):
    for df in reader:
        q = pad_sequences(tokeniser.texts_to_sequences(df.q.tolist()), maxlen=max_len)

        q_one_hot = to_categorical(q, feature_num)   
        q_one_hot = q_one_hot.reshape(batch_size, max_len, feature_num)
        
        
        yield q, q_one_hot

def get_test_data(filename):

    if filename in ["MayFlower", "JuneFlower"]:
        file_dir = '/data/t-mipha/data/query_similarity_ndcg/%sIdeal.txt' % filename
        df = pd.read_csv(file_dir, names=["market", "qid", "q", "label", "d", "date"], sep="\t", header=None, error_bad_lines=False)
        df = df.dropna()
        y = np.array([0 if i == "Bad" else 1 if i == "Fare" else 2 for i in df.label.tolist()])

        qrel = convert_2_trec(df.q.tolist(), df.d.tolist(), y, True)

    elif filename in ["JulyFlower"]:

        file_dir = '/data/chzho/deepqts/test_data/julyflower/julyflower_original.tsv'
        df = pd.read_csv(file_dir, names=["q", "d", "label"], sep="\t", header=None, error_bad_lines=False)
        df = df.dropna()
        qrel = df.label.values

    return df, qrel

def parse_texts(texts, tokeniser, max_len):   

    x = tokeniser.texts_to_sequences(texts)
    x = pad_sequences(x, maxlen=max_len)

    return x

def parse_texts_bpe(texts, sp, bpe_dict, max_len, enablePadding=True):

    x = []
    for text in texts:
        tmp = []
        for t in sp.EncodeAsPieces(text):
            if not isinstance(t, str):
                t = str(t, "utf-8")
            if t in bpe_dict:
                tmp.append(bpe_dict[t])
            else:
                tmp.append(bpe_dict['<unk>'])
        x.append(tmp)
    
    return np.array(x) if not enablePadding else pad_sequences(x, maxlen=max_len)


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


