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
from keras.layers import Conv1D, GlobalMaxPool1D, Dropout, MaxPooling1D
from gensim.models import word2vec
import jieba
from pandas.core.frame import DataFrame
from keras.models import Sequential
from keras.layers import Merge


def cut_texts(texts=None, word_len=1, savename=None):
    #分词
    texts_cut=[]
    text_one=[]
    stopwords_file = "stop_words2.txt"
    stop_f = open(stopwords_file,"r",encoding='utf-8')
    stop_words = list()
    for line in stop_f.readlines():
        line = line.strip()
        if not len(line):
            continue
        stop_words.append(line)
    stop_f.close
    print('The length of stop words are '+str(len(stop_words)))
    if word_len > 1:
        for text in texts:
            text_cut=[]
#            words=jieba.lcut(text)
            words=jieba.cut(text,cut_all=False, HMM=True)
            for word in words:
                if word not in stop_words and len(word)>=word_len:
                    text_cut.append(word)
                    text_one.append(word)
            texts_cut.append(text_cut)
    else:
        for text in texts:
            text_cut=[]
            words=jieba.lcut(text)
            for word in words:
                if word not in stop_words:
                    text_cut.append(word)
                    text_one.append(word)
            texts_cut.append(text_cut)
    if savename is not None:
        file=open(savename,'w',encoding='utf-8')
        file.write(' '.join(text_one))
        file.close()
    return texts_cut

def text2vec(texts_cut=None, model_word2vec=None,
              word2vec_savepath=None, word2vec_loadpath=None,
              sg=1, size=128, window=5, min_count=1, merge=True):
     '''
     文本的词语序列转为词向量序列，可以用于机器学习或者深度学习
     :param texts_cut: Word list of texts
     :param model_word2vec: word2vec model of gensim
     :param word2vec_savepath: word2vec savrpath
     :param word2vec_loadpath: word2vec loadpath
     :param sg: 0 CBOW,1 skip-gram
     :param size: The dimensionality of the feature vectors
     :param window: The maximum distance between the current and predicted word within a sentence
     :param min_count: Ignore all words with total frequency lower than this
     :param merge: If Ture, calculate sentence average vector
     :return:
     '''
     if word2vec_savepath:
         model_word2vec.save(word2vec_savepath)
     text_vec = [[model_word2vec[word] for word in text_cut if word in model_word2vec] for text_cut in texts_cut]
     if merge:
         return np.array([sum(i) / len(i) for i in text_vec])
     else:
         return text_vec


#导入数据
data=pd.read_csv("data_single.csv")
x = data['evaluation']
y = data['label']
#分词
X_cut= cut_texts(texts=x, word_len=1)


# texts to word vector、
model_word2vec = word2vec.Word2Vec(X_cut, sg=1, size=128, window=5, min_count=1)
x_word_vec = text2vec(texts_cut=X_cut,model_word2vec=model_word2vec, sg=1, size=20, window=5, min_count=1)
# texts vector
x_vec = np.array([sum(i) / len(i) for i in x_word_vec])


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=1)




model = TextClassification()
# use process
model.fit(x=X_train,
          y=y_train,
          x_need_preprocess=True,
          y_need_preprocess=False,
          method='SVM', output_type='single')

y_predict = model.predict(x=X_test, x_need_preprocess=True)
# score 0.8331
print(sum(y_predict == np.array(y_test)) / len(y_predict))
