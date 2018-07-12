import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, Dropout
import jieba
from pandas.core.frame import DataFrame



def cut_texts(texts=None, word_len=1, savename=None):
    #分词
    texts_cut=[]
    text_one=[]
    if word_len > 1:
        for text in texts:
            text_cut=[]
            words=jieba.lcut(text)
            for word in words:
                if len(word)>=word_len:
                    text_cut.append(word)
                    text_one.append(word)
            texts_cut.append(text_cut)
    else:
        for text in texts:
            words=jieba.lcut(text)
            for word in words:
                text_one.append(word)
            texts_cut.append(words)
    if savename is not None:
        file=open(savename,'w',encoding='utf-8')
        file.write(' '.join(text_one))
        file.close()
    return texts_cut


def text2seq(texts_cut=None, maxlen=30,tokenizer=None):
    #文本转序列
    fact_seq = tokenizer.texts_to_sequences(texts=texts_cut)
    print('finish texts to sequences')
    fact_pad_seq = []
    # pad_sequences,将每一条评论都填充（pad）到一个矩阵中。 最大长度30，超出长度从前面截断。结尾补0。
    fact_pad_seq += list(pad_sequences(fact_seq, maxlen=maxlen,
        padding='post', truncating='pre', value=0, dtype='int'))
    return fact_pad_seq





def label2tag( predictions,y):
    label_set = []
    for i in y:
        label_set.append(i)
    label_set=np.array(list(set(label_set)))

    labels = []
    for prediction in predictions:
        label = label_set[prediction == prediction.max()]
        labels.append(label.tolist())

    return labels

#导入数据
data=pd.read_csv("data_single.csv")
x = data['evaluation']
y=data['label']
#分词
X_cut= cut_texts(texts=x, word_len=2,savename='ciyun.txt')

# 对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示。
tokenizer = Tokenizer(num_words=500)
tokenizer.fit_on_texts(texts=X_cut)
#index=tokenizer.word_index
#counts=tokenizer.word_counts

#文本转矩阵
maxlen=20 #矩阵维度
X_seq = text2seq(texts_cut=X_cut,maxlen=maxlen, tokenizer=tokenizer)
X_seq = np.array(X_seq)

#标签转成独热码
y_one_hot = pd.get_dummies(y)
y_one_hot_labels = np.asarray(y_one_hot)

#分割训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_one_hot_labels, test_size=0.2)


num_words = 2000
vec_size = 128
output_shape = 2

#构建模型
data_input = Input(shape=[maxlen])
word_vec = Embedding(input_dim=num_words+1,
                     input_length=maxlen,
                     output_dim=vec_size,
                     mask_zero=0)(data_input)
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
model.summary()

#训练
model.fit(x=X_train, y=y_train, epochs=3, batch_size=128,verbose=2,validation_split=0.1)
#预测
y_predict = model.predict(X_test)
#转换预测结果
y_predict_label = label2tag(predictions=y_predict, y=y)
#统计正确率
Y_test=label2tag(predictions=y_test,y=y)
print(sum([y_predict_label[i] == Y_test[i] for i in range(len(y_predict))]) / len(y_predict))

#导入另一个测试集进行预测，并导出结果
filename='xiaomi5a.csv'
test_data=pd.read_csv(filename)
x = test_data['comment']
X_cut = cut_texts(texts=x, need_cut=True, word_len=2, savepath=None)
X_seq = text2seq(texts_cut=X_cut,maxlen=maxlen,tokenizer=tokenizer)
X_seq = np.array(X_seq)
y_predict = model.predict(X_seq)
y_predict_label = label2tag(predictions=y_predict, y=y)
#Series转成dateframe
out_x=x.to_frame(name=None)
out_y=DataFrame(y_predict_label)
out_x.to_csv('x.csv')
out_y.to_csv('y.csv')
