import json
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, Dropout
from gensim.models import word2vec
import jieba
from pandas.core.frame import DataFrame



def cut_texts(texts=None, need_cut=True, word_len=1, savepath=None):
    #分词
    texts_cut=[]
    if word_len > 1:
        for text in texts:
            text_cut=[]
            words=jieba.lcut(text)
            for word in words:
                if len(word)>=word_len:
                    text_cut.append(word)
            texts_cut.append(text_cut)
    else:
        for text in texts:
            words=jieba.lcut(text)
            texts_cut.append(words)
    if savepath is not None:
#python3的json在做dumps时，会把中文转成unicode编码，以16进制存储。读取时，再将unicode编码转回中文。
#        with open(savepath, 'w') as f:
#             json.dump(texts_cut, f)
        out_put=DataFrame(texts_cut)
        out_put.to_csv('fenci.csv')
    return texts_cut


def text2seq(texts_cut=None, maxlen=30,tokenizer=None):
    #文本转序列
    fact_seq = tokenizer.texts_to_sequences(texts=texts_cut)
    print('finish texts to sequences')
    fact_pad_seq = []
    # pad_sequences,将每一条评论都填充（pad）到一个矩阵中。 最大长度30，结尾补0
    fact_pad_seq += list(pad_sequences(fact_seq, maxlen=maxlen,
        padding='post', value=0, dtype='int'))
    return fact_pad_seq




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
X_cut= cut_texts(texts=x, need_cut=True, word_len=2,savepath='fenci.csv')
#y = [[i] for i in data['label']]
y=data['label']

# 对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示。
tokenizer = Tokenizer(num_words=500)
tokenizer.fit_on_texts(texts=X_cut)
#index=tokenizer.word_index
#counts=tokenizer.word_counts
maxlen=30
X_seq = text2seq(texts_cut=x,maxlen=maxlen, tokenizer=tokenizer)
X_seq = np.array(X_seq)
#标签转成独热码
y_one_hot = pd.get_dummies(y)
y_one_hot_labels = np.asarray(y_one_hot)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_one_hot_labels, test_size=0.2)


num_words=2000

vec_size=128
output_shape=2

#构建模型
data_input = Input(shape=[maxlen])
word_vec = Embedding(input_dim=num_words+1,
                     input_length=maxlen,
                     output_dim=vec_size,
                     mask_zero=0,
                     name='Embedding')(data_input)
x = Conv1D(filters=128, kernel_size=[3], strides=1, padding='same', activation='relu')(word_vec)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(500, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(output_shape, activation='softmax')(x)
model = Model(inputs=data_input, outputs=x)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#训练
model.fit(x=X_train, y=y_train, epochs=10, batch_size=128,verbose=2,validation_split=0.1)
#预测
y_predict = model.predict(X_test)

y_predict_label = label2tag(predictions=y_predict, labelset=label_set)

print(sum([y_predict_label[i] == y_test[i] for i in range(len(y_predict))]) / len(y_predict))

#导入另一个测试集
test_data=pd.read_csv("xiaomi.csv")
x = test_data['comment']
X_cut = cut_texts(texts=x, need_cut=True, word_len=2, savepath=None)
X_seq = text2seq(texts_cut=X_cut,num_words=500, maxlen=20, batchsize=10000,tokenizer=tokenizer)
X_seq = np.array(X_seq)
y_predict = model.predict(X_seq)
y_predict_label = label2tag(predictions=y_predict, labelset=label_set)
#Series转成dateframe
out_x=x.to_frame(name=None)
out_y=DataFrame(y_predict_label)
out_x.to_csv('x.csv')
out_y.to_csv('y.csv')
