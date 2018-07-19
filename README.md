

# 基于卷积神经网络的文本情感分类

> 注：上述题目是网上申请时提交的题目。后来加以引申，最终项目准确的名称应该是 “实现基于卷积神经网络和支持向量机的文本情感分类，并讨论数据量对这两种方式准确率的影响。”

[TOC]

## 内容

1、通过网络爬虫从淘宝、京东爬取商品评论，对其进行人工分类、标注，得到数据集。

2、对数据集进行预处理，进行中文分词、去除停用词，标点符号。

3、分别使用 “卷积神经网络” 和 “支持向量机” 进行训练和预测，计算准确率。

4、改变用于训练的数据量大小，比较这两种方式下的准确率变化情况。

## 依赖库

```
python=3.6
pandas=0.22.0
numpy=1.14.0
jieba=0.39
gensim=3.4.0
scikit-learn=0.19.1
keras=2.1.5
wordcloud=1.4.1
```



## 具体实现

### 构建数据集

1、 原始数据爬取

​	创建商品评论页面的循环链接，使用Python的requests库循环抓取数据，使用正则表达式匹配查询，得到相关文本，将结果保存为.csv文档。

```python
for url in urls:
    content = requests.get(url).text
# 借助正则表达式使用findall进行匹配查询
    nickname.extend(re.findall('"displayUserNick":"(.*?)"',content)) #用户名
    ratecontent.extend(re.findall(re.compile('"rateContent":"(.*?)","rateDate"'),content)) #评论内容
    ratedate.extend(re.findall(re.compile('"rateDate":"(.*?)","reply"'),content)) #评论时间
```

2、标注情感

正面：

![正面评论](C:\Users\liu01\Desktop\Text sentiment classification\截图\数据集标注_正面.jpg)

负面：

![负面评论](C:\Users\liu01\Desktop\Text sentiment classification\截图\数据集标注_负面.jpg)

### 分词	

​	在英文的行文中，单词之间是以空格作为自然分界符的，而中文只是字、句和段能通过明显的分界符来简单划界，但是词却没有一个形式上的分界符。要获得句子的语义特征，必须对句子进行分词。

​	使用"jieba分词"实现:     ```words=jieba.lcut(text)```

#### 分词结果

​	使用基于wordcloud生成的词云表示分词结果：

​	![词云](C:\Users\liu01\Desktop\Text sentiment classification\截图\词云效果2.png)

#### 关于最小词长的讨论

最小词长为1时的词频：

![最小词长为1时的词频](C:\Users\liu01\Desktop\Text sentiment classification\截图\分词为1时的词频排序.jpg)

最小词长为2时的词频：

![最小词长为2时的词频](C:\Users\liu01\Desktop\Text sentiment classification\截图\分词为2时的词频排序.jpg)

​	由图可见，当最小词长选为1时，高频词出现了大量的标点符号、语气词等无关词语。虽然可以通过通用词将其剔除，但是这么做比较复杂，并且需要构建一个庞大的停用词表。可以看到选最小词长为2时，明显可以得到很多表示情感的词，比如像”不错“、”满意“等等，已经满足了需求。所以最终选择的最小词长为2。 

### 卷积神经网络

代码见  CNN.py

流程： 导入数据 -> 数据预处理（分词，统计词频、编号，标签转成独热码） -> 构建模型 -> 训练和预测 -> 计算准确率

#### 数据预处理部分

1、分词（同上）

2、统计词频、编码

​	使用Keras的Tokenizer类 对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示。

````python
tokenizer = Tokenizer(num_words=2000) #初始化Tokenizer
tokenizer.fit_on_texts(texts=X_cut) #训练
counts=tokenizer.word_counts #词频
index=tokenizer.word_index #编号
````

> 注：在Keras官方文档中，Tokenizer被翻译成“分词器”，这个翻译其实不太恰当。它实现的功能并不是分词，而是统计词频，并且可以根据词频从高到低对词语依次编码为1,2,3……。

3、标签转成独热码

​	使用Pandas的get_dummies方法实现。

```python
y_one_hot = pd.get_dummies(y)
y_one_hot_labels = np.asarray(y_one_hot)
```



#### 模型构建部分

```python
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
```



#### 训练和预测部分

```python
#训练
model.fit(x=X_train, y=y_train, epochs=3, batch_size=64,verbose=2)
#预测
y_predict = model.predict(X_test)
#转换预测结果
y_predict_label = label2tag(predictions=y_predict, y=y)
#统计正确率
Y_test=label2tag(predictions=y_test,y=y)
print(sum([y_predict_label[i] == Y_test[i] for i in range(len(y_predict))]) / len(y_predict))

```



### 支持向量机

代码见 SVM.py

流程： 导入数据 -> 数据预处理（分词，词语转为词向量） -> 构建模型 -> 训练和预测 -> 计算准确率

#### 数据预处理部分

1、分词，同上

2、词向量生成

​	使用中文词向量语料库（Shen Li, Zhe Zhao, Renfen Hu, Wensi Li, Tao Liu, Xiaoyong Du, [*Analogical Reasoning on Chinese Morphological and Semantic Relations*](http://aclweb.org/anthology/P18-2023), ACL 2018.）中的中文预训练词向量（sgns.weibo.word）对分词结果进行转换。

```python
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_loadpath,binary=False)
text_vec = [[model[word] for word in text_cut if word in model] for text_cut in X_cut]
```



#### 模型构建部分

```python
model_svc = SVC(C=10,kernel='linear',)
#C似乎越大准确率越高： score 0.819672（C=1）  0.83957(C=2)  0.850117(C=3)  0.862997(C=5)
#尝试了所有的kernel，最后linear的准确率最高。
```



#### 训练和预测部分

```python
model_svc.fit(X=X_train,y=y_train,)
y_predict  = model_svc.predict(X_test)
print(sum(y_predict == np.array(y_test)) / len(y_predict))
```



### 准确率比较

| 数据量 | 卷积神经网络 | 支持向量机 |
| :----: | :----------: | :--------: |
|   50   |    0.500     |   0.900    |
|  100   |    0.550     |   0.900    |
|  200   |    0.650     |   0.900    |
|  300   |    0.733     |   0.883    |
|  400   |    0.825     |   0.763    |
|  500   |    0.850     |   0.820    |
|  1000  |    0.875     |   0.844    |
|  2000  |    0.900     |   0.869    |
|  3000  |    0.901     |   0.875    |
|  4000  |    0.9175    |   0.886    |

![折线图](C:\Users\liu01\Desktop\Text sentiment classification\截图\结果.jpg)

​	可以看出，卷积神经网络随着数据量的增加，正确率也是不断地提高，最后收敛在0.90左右。而支持向量机的正确率却是经历了先减小后增大的过程。 说明对于卷积神经网络，用于训练的数据量越大，效果越好。而对于支持向量机，更重要的可能是参数的设置。



