import os, logging
import numpy as np
import jieba
jieba.setLogLevel(logging.INFO)
# coding:utf-8
import gensim
from gensim.models.doc2vec import Doc2Vec
import math, os
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import codecs
import regex



def sen2vec (sen_squence):

    token2id = {}
    id2token = {}
    with open('sequence-labeling-BiLSTM-CRF-master\\data\\example_datasets\\vocabs\\token2id', 'r', encoding='utf-8') as infile:  # 读token
        for row in infile:
            row = row.rstrip()
            token = row.split('\t')[0]
            token_id = int(row.split('\t')[1])
            token2id[token] = token_id
            id2token[token_id] = token

    model_dm = Doc2Vec.load("sequence-labeling-BiLSTM-CRF-master\\sen_vec_model")

    # sen_squence_lis = list(sen_squence)
    sentence = []
    single_sen = []
    for i in range(len(sen_squence)):
        for j in range(len(sen_squence[i])):
            single_sen.append(id2token[sen_squence[i][j]])
        inferred_vector_dm = model_dm.infer_vector(single_sen)
        single_sen = []
        sentence.append(inferred_vector_dm)

    # a = tf.fill([300, 1], 1.)
    a = np.ones([300, 1])
    for x in range(len(sentence)):
        sentence[x] = np.multiply(a, sentence[x])


    sentence = np.vstack(sentence)
    # sentence = tf.concat(sentence, 0)
    # sentence = np.array(sentence)

    return sentence

