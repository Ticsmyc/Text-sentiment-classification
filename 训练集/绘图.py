#coding:utf-8
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#We give the coordinate date directly to give an example.
x=[50,100,200,300,400,500,1000,2000,3000,4000]
y1 = [0.5,0.55,0.65,0.73,0.825,0.850,0.875,0.900,0.901,0.9175]
y2 = [0.9,0.9,0.9,0.883,0.763,0.820,0.844,0.869,0.875,0.886]


#添加图例
plt.plot(x, y1, color="black", linewidth=2.5, linestyle="-", label="卷积神经网络")
plt.plot(x, y2, color="red",  linewidth=2.5, linestyle="-", label="支持向量机")
plt.legend(loc='lower right')

plt.xlabel('数据量')
plt.ylabel("准确率")


plt.savefig('结果.jpg',dpi=500)

