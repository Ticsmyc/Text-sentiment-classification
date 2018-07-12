import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import jieba
from sklearn.svm import SVC
import gensim

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


#导入数据
data=pd.read_csv("data_single.csv")
x = data['evaluation']
y = data['label']
#分词并解决分词后空白list的问题
X_cut_n= cut_texts(texts=x, word_len=2)
X_cut=[]
label=[]
for i in range(0,len(X_cut_n)):
    if (len(X_cut_n[i])!=0) :
        X_cut.append(X_cut_n[i])
        label.append(y[i])
del X_cut_n
del y

# 转为词向量，去掉空list、

word2vec_loadpath='sgns.weibo.word'
needSave='False'
model = gensim.models.KeyedVectors.load_word2vec_format('sgns.weibo.word',binary=False)
if needSave:
    model.save('word2vec_model')
text_vec = [[model[word] for word in text_cut if word in model] for text_cut in X_cut]
text=[]
labels=[]
for index,i in enumerate(text_vec):
   if len(i)!=0:
        text.append(sum(i)/len(i))
        labels.append(label[index])
text=np.array(text)
# texts vector
#x_vec = np.array([sum(i) / len(i) for i in x_word_vec])
X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.2,random_state=1)

#构造SVC模型
model = SVC(C=10,kernel='linear',)
model.fit(X=X_train,y=y_train,)
y_predict  = model.predict(X_test)
# score 0.819672（C=1）  0.83957(C=2)  0.850117(C=3)  0.862997(C=5)
print(sum(y_predict == np.array(y_test)) / len(y_predict))





