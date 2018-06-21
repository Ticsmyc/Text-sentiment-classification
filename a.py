import json
import pandas as pd
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec
import jieba
import pickle

#导入数据
data=pd.read_csv("data_single.csv")

jieba.setLogLevel('WARN')

#分词
def cut_texts(texts=None, need_cut=True, word_len=1, savepath=None):
    '''
    Use jieba to cut texts
    :param texts:list of texts
    :param need_cut:whether need cut text
    :param word_len:min length of words to keep,in order to delete stop-words
    :param savepath:path to save word list in json file
    :return:
    '''
    if need_cut:
        if word_len > 1:
            texts_cut = [[word for word in jieba.lcut(text) if len(word) >= word_len] for text in texts]
        else:
            texts_cut = [jieba.lcut(one_text) for one_text in texts]
    else:
        if word_len > 1:
            texts_cut = [[word for word in text if len(word) >= word_len] for text in texts]
        else:
            texts_cut = texts

    if savepath is not None:
        with open(savepath, 'w') as f:
            json.dump(texts_cut, f)
    return texts_cut

def text2seq(texts_cut=None, tokenizer=None, tokenizer_savapah=None,
             num_words=2000, maxlen=30, batchsize=10000):
    '''
    文本转序列，用于神经网络的ebedding层输入。训练集过大全部转换会内存溢出，每次放10000个样本
    :param texts_cut: 分词后的文本列表
    :param tokenizer:转换字典，keras的一个方法
    :param tokenizer_savapah:字典保存路径
    :param num_words: the maximum number of words to keep
    :param maxlen: the number of words to keep in sentence
    :param batchsize: Size of batch put in tokenizer
    :return:sequence list
    eg. ata_transform.text2seq(texts_cut=train_fact_cut,num_words=2000, maxlen=500)
    '''
    texts_cut_len = len(texts_cut)

    if tokenizer is None:
        tokenizer = Tokenizer(num_words=num_words)
        n = 0
        # 分批训练
        while n < texts_cut_len:
            tokenizer.fit_on_texts(texts=texts_cut[n:n + batchsize])
            n += batchsize
            if n < texts_cut_len:
                print('tokenizer finish fit %d samples' % n)
            else:
                print('tokenizer finish fit %d samples' % texts_cut_len)


    if tokenizer_savapah:
        with open(tokenizer_savapah, mode='wb') as f:
            pickle.dump(tokenizer, f)

    # 全部转为数字序列
    fact_seq = tokenizer.texts_to_sequences(texts=texts_cut)
    print('finish texts to sequences')

    # 内存不够，删除
    del texts_cut

    n = 0
    fact_pad_seq = []
    # 分批执行pad_sequences
    while n < texts_cut_len:
        fact_pad_seq += list(pad_sequences(fact_seq[n:n + 10000], maxlen=maxlen,
                                           padding='post', value=0, dtype='int'))
        n += 10000
        if n < texts_cut_len:
            print('finish pad sequences %d/%d' % (n, texts_cut_len))
        else:
            print('finish pad sequences %d/%d' % (texts_cut_len, texts_cut_len))
    return fact_pad_seq

def creat_label_set( labels):
    '''
    获取标签集合，用于one-hot
    :param labels: 原始标签集
    :return:
    '''
    label_set = []
    for i in labels:
        label_set += i
    return np.array(list(set(label_set)))

def creat_label(label, label_set):
    '''
    构建标签one-hot
    :param label: 原始标签
    :param label_set: 标签集合
    :return: 标签one-hot形式的array
    eg. creat_label(label=data_valid_accusations[12], label_set=accusations_set)
    '''
    label_zero = np.zeros(len(label_set))
    label_zero[np.in1d(label_set, label)] = 1
    return label_zero

def creat_labels( labels=None, label_set=None):
    '''
    调用creat_label遍历标签列表生成one-hot二维数组
    :param label: 原始标签集
    :param label_set: 标签集合
    :return:
    '''
    labels_one_hot = list(map(lambda x: self.creat_label(label=x, label_set=label_set), labels))
    return labels_one_hot

def fit( x=None, y=None, model=None,
        method='CNN', epochs=10, batchsize=256,
        x_need_preprocess=False, y_need_preprocess=False,
        tokenizer=None, num_words=2000, maxlen=None,
        vec_size=128, output_shape=None, output_type='multiple',
        **sklearn_param):
    # cut texts
    x_cut = cut_texts(texts=x, need_cut=True, word_len=2, savepath=None)
    # use average length
    if maxlen is None:
        maxlen = int(np.array([len(x) for i in x_cut]).mean())
    # texts to sequence
    x_seq = text2seq(texts_cut=x_cut, tokenizer=tokenizer, tokenizer_savapah=None,
                             num_words=num_words, maxlen=maxlen, batchsize=10000)
    # list to array
    x_seq = np.array(x_seq)
    x = x_seq

    self.tokenizer = process.tokenizer

        if y_need_preprocess:
            process = DataPreprocess()
            label_set = process.creat_label_set(y)
            labels = process.creat_labels(labels=y, label_set=label_set)
            labels = np.array(labels)
            output_shape = labels.shape[1]
            y = labels
            self.output_shape = output_shape
            self.label_set = label_set

        if model is None:
            if method == 'CNN':
                model = CNN(input_dim=num_words, input_length=maxlen,
                            vec_size=vec_size, output_shape=output_shape,
                            output_type=output_type)
            elif method == 'RNN':
                model = RNN(input_dim=num_words, input_length=maxlen,
                            vec_size=vec_size, output_shape=output_shape,
                            output_type=output_type)
        model.fit(x=x, y=y, epochs=epochs, batch_size=batchsize)

    elif method in ['SVM', 'Logistic']:
        if output_type != 'single':
            raise ValueError('sklearn output_type should be single')
        else:
            if x_need_preprocess:
                process = DataPreprocess()
                # cut texts
                x_cut = process.cut_texts(texts=x, need_cut=True, word_len=2, savepath=None)
                x_seq_vec = process.text2vec(texts_cut=x, sg=1, size=128, window=5, min_count=1)
                x_vec = np.array([sum(i) for i in x_seq_vec])
                x = x_vec
                self.model_word2vec = process.model_word2vec

            model = SklearnClf(method=method, **sklearn_param)
            model.fit(X=x, y=y)

    self.model = model
