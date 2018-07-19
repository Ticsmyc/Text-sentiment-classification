import json
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, Dropout
from gensim.models import word2vec
import jieba
import pickle
from pandas.core.frame import DataFrame


#分词
def cut_texts(texts=None, need_cut=True, word_len=1, savepath=None):
    '''
    使用jieba分词
    :param texts:list of texts
    :param word_len:min length of words to keep,in order to delete stop-words
    :param savepath:path to save word list in json file
    :return:
    '''
    if word_len > 1:
        texts_cut = [[word for word in jieba.lcut(text) if len(word) >= word_len] for text in texts]
    else:
        texts_cut = [jieba.lcut(one_text) for one_text in texts]

    if savepath is not None:
        with open(savepath, 'w') as f:
            json.dump(texts_cut, f)
    return texts_cut

def text2seq(texts_cut=None,num_words=2000, maxlen=30, batchsize=10000):
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
    #分词器，限制为待处理数据集中最常见的num_words个单词
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
    # 全部转为数字序列
    fact_seq = tokenizer.texts_to_sequences(texts=texts_cut)
    print('finish texts to sequences')
#    del texts_cut
    n = 0
    fact_pad_seq = []
    # 分批执行pad_sequences,将每一条评论都填充（pad）到一个矩阵中。
    while n < texts_cut_len:
        fact_pad_seq += list(pad_sequences(fact_seq[n:n + batchsize], maxlen=maxlen,
                                           padding='post', value=0, dtype='int'))
        n += batchsize
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
    labels_one_hot = list(map(lambda x: creat_label(label=x, label_set=label_set), labels))
    return labels_one_hot



def label2toptag( predictions, labelset):
    labels = []
    for prediction in predictions:
        label = labelset[prediction == prediction.max()]
        labels.append(label.tolist())
    return labels

def label2half(predictions, labelset):
    labels = []
    for prediction in predictions:
        label = labelset[prediction > 0.5]
        labels.append(label.tolist())
    return labels

def label2tag( predictions, labelset):
    labels1 = label2toptag(predictions, labelset)
    labels2 = label2half(predictions, labelset)
    labels = []
    for i in range(len(predictions)):
        if len(labels2[i]) == 0:
            labels.append(labels1[i])
        else:
            labels.append(labels2[i])
    return labels

#导入数据并分割。
data=pd.read_csv("data_single.csv")
x = data['evaluation']
y = [[i] for i in data['label']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
jieba.setLogLevel('WARN')

# 分词并转成序列再转成向量
#训练集
X_train_cut = cut_texts(texts=X_train, need_cut=True, word_len=2, savepath=None)
X_train_seq = text2seq(texts_cut=X_train_cut,num_words=500, maxlen=20, batchsize=10000)
X_train_seq = np.array(X_train_seq)
#测试集
X_test_cut = cut_texts(texts=X_test, need_cut=True, word_len=2, savepath=None)
X_test_seq = text2seq(texts_cut=X_test_cut,num_words=500, maxlen=20, batchsize=10000)
X_test_seq = np.array(X_test_seq)
#标签转成独热码
label_set = creat_label_set(y_train)
train_labels = creat_labels(labels=y_train, label_set=label_set)
train_labels = np.array(train_labels)
test_labels = creat_labels(labels=y_test, label_set=label_set)
test_labels = np.array(test_labels)


num_words=2000
maxlen=20
vec_size=128
output_shape=train_labels.shape[1]
#构建模型
data_input = Input(shape=[maxlen])
word_vec = Embedding(input_dim=num_words+1,
                     input_length=maxlen,
                     output_dim=vec_size,
                     mask_zero=0,
                     name='Embedding')(data_input)
x = Conv1D(filters=128, kernel_size=[3], strides=1, padding='same', activation='relu')(word_vec)
x = GlobalMaxPool1D()(x)
x = Dropout(0.2)(x)
x = Dense(500, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(output_shape, activation='softmax')(x)
model = Model(inputs=data_input, outputs=x)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#训练
model.fit(x=X_train_seq, y=train_labels, epochs=20, batch_size=128,verbose=1,validation_split=0.1)
#预测
y_predict = model.predict(X_test_seq)

y_predict_label = label2tag(predictions=y_predict, labelset=label_set)

print(sum([y_predict_label[i] == y_test[i] for i in range(len(y_predict))]) / len(y_predict))

#Series转成dateframe
out_x=X_test.to_frame(name=None)
out_y=DataFrame(y_predict_label)
out_x.to_csv('x.csv')
out_y.to_csv('y.csv')