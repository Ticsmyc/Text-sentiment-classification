filename = "stop_word.txt"

f = open(filename,"r")
result = list()
for line in f.readlines():
    line = line.strip()
    if not len(line):
        continue

    result.append(line)
f.close
with open("stop_words2.txt","w",encoding='utf-8') as fw:
    for sentence in result:
        sentence.encode('utf-8')
        data=sentence.strip()
        if len(data)!=0:
            fw.write(data)
            fw.write("\n")
print ("end")
