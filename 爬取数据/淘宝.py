# 导入所需的开发模块
import requests
import re
# 创建循环链接
urls = []
for i in list(range(1,100)):
    urls.append('https://rate.tmall.com/list_detail_rate.htm?itemId=560135062971&spuId=889297741&sellerId=1714128138&order=3&currentPage=%s' %i)

# 构建字段容器
nickname = []
color = []
ratecontent = []
i=0
# 循环抓取数据
for url in urls:
    content = requests.get(url).text

# 借助正则表达式使用findall进行匹配查询
    nickname.extend(re.findall('"displayUserNick":"(.*?)"',content))
    color.extend(re.findall(re.compile('颜色分类:(.*?);'),content))
    ratecontent.extend(re.findall(re.compile('"rateContent":"(.*?)","rateDate"'),content))
 #   ratedate.extend(re.findall(re.compile('"rateDate":"(.*?)","reply"'),content))
    print(i)
    i=i+1
# 写入数据

file = open('小米手机5A 评价.csv','w')
for i in list(range(0,len(nickname))):
    file.write(','.join((nickname[i],color[i],ratecontent[i]))+'\n')
file.close()
