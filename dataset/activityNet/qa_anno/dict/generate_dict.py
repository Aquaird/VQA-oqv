import nltk
import json
import sys

qatype=raw_input('input_qatype:')
num = 1
raw_dict = {}

def add_word(data, num):
    for key in data:
        value = str(key)
        tokens = nltk.word_tokenize(value)
        #print tokens
        tags = nltk.pos_tag(tokens)
        #print tags
        for tag in tags:
            if (tag[1] != '.'):
                word = tag[0].lower()
                if not (word in raw_dict):
                    raw_dict[word] = [num,int(1)]
                    num = num + 1
                    sys.stdout.write('\r%d words are added.'%(num-1))
                    sys.stdout.flush()
                else:
                    raw_dict[word][1] = raw_dict[word][1]+1
    return num

data=json.load(open('../%s.json'%qatype,'r'))

num=add_word(data,num)

raw_sort=sorted(raw_dict, key=lambda word:raw_dict[word][1], reverse=True)

i=1
raw_sort_1000={}
res_sort_1000={}
for key in raw_sort:
    if (i<1001):
        raw_sort_1000[raw_sort[i]] = [i,raw_dict[raw_sort[i]][1]]
        i=i+1

for key in raw_sort_1000:
    res_sort_1000[raw_sort_1000[key][0]]=[key,raw_sort_1000[key][1]]


json.dump(raw_sort_1000,open('%s_dic_raw.json'%qatype,'w'))
json.dump(res_sort_1000,open('%s_dic.json'%qatype,'w'))
