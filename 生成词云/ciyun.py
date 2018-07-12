from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import jieba
#添加用户词库作为主词典
#jieba.load_userdict("txt.txt")
from wordcloud import WordCloud,ImageColorGenerator

#获取当前文件路径
# __file__ 为当前文件, 在ide中运行报错，改为
d=path.dirname('.')
#d=path.dirname(__file__)

stopwords={}
isCN=0 # 是否需要分词
back_coloring_path = "img.jpg" #设置背景图片路径
text_path="ciyun_single.txt" #设置要分析的文本路径
font_path= 'STXINWEI.TTF' #中文字体路径
imgname1 = 'WordCloudDefautColors.png' #保存图片1 ，只按照背景图片形状
imgname2= 'WordCloudColorByImg.png' #保存图片名字2 ，颜色按照背景图片颜色布局生成

back_coloring=imread(path.join(d,back_coloring_path)) #设置背景图片


#设置词云属性
wc= WordCloud(font_path=font_path,
    background_color='white',
    max_words=2000,
    mask=back_coloring,
    max_font_size=100,
    random_state=30,
    width=600,height=400,margin=5,
    relative_scaling=0.5
    )

text=open(path.join(d,text_path),encoding='utf-8').read()

def jiebaclearText(text):
    myword_list=[]
    seg_list = jieba.cut(text, cut_all=False)
    liststr='/'.join(seg_list)
    for myword in liststr.split('/'):
        if len(myword.strip())>1 :
            myword_list.append(myword)
    return ' '.join(myword_list)

if isCN:
    text=jiebaclearText(text)

#生成词云
wc.generate(text)

#从背景图片生成颜色值
image_colors=ImageColorGenerator(back_coloring)

plt.figure()
plt.imshow(wc)
plt.axis('off')
plt.show()

wc.to_file(path.join(d,imgname1))

img_colors=ImageColorGenerator(back_coloring)
plt.imshow(wc.recolor(color_func= image_colors))
plt.axis('off')
plt.figure()
plt.imshow(back_coloring,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
wc.to_file(path.join(d,imgname2))
