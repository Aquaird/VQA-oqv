import json
import sys
import random

dic=json.load(open('answer_one_dic_raw.json','r'))
groundtruth=json.load(open('../answer_one.json','r'))
# keylist=dic.keys()
sort=sorted(dic, key=lambda word:dic[word][1], reverse=True)

output=[]
for key in groundtruth:
    can=[key]
    i=0
    while(i<3):
        randword=sort[random.randint(0,50)]
        if not (randword in can):
            can.append(randword)
            i+=1

    output.append(can)
json.dump(output, open('../candidate.json','w'))

