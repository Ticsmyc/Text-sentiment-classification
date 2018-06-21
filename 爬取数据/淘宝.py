# 导入所需的开发模块
import requests
import re
# 创建循环链接
urls = []
for i in list(range(1,7)):
    urls.append('https://rate.tmall.com/list_detail_rate.htm?itemId=44142685694&spuId=331499563&sellerId=2169969258&order=3&currentPage=%s' %i)

# 构建字段容器
nickname = []
ratedate = []
ratecontent = []

# 循环抓取数据
for url in urls:
    content = requests.get(url).text

# 借助正则表达式使用findall进行匹配查询
    nickname.extend(re.findall('"displayUserNick":"(.*?)"',content))

    ratecontent.extend(re.findall(re.compile('"rateContent":"(.*?)","rateDate"'),content))
    ratedate.extend(re.findall(re.compile('"rateDate":"(.*?)","reply"'),content))
    print(nickname)
# 写入数据

file = open('数学之美评价.csv','w')
for i in list(range(0,len(nickname))):
    file.write(','.join((nickname[i],ratedate[i],ratecontent[i]))+'\n')
file.close()
