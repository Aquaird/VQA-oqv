import h5py
import json
import nltk
import sys
import numpy as np

def word_embed(dic, word):
    if word in dic.keys():
        embed=int(dic[word][0])
    else:
        embed=int(0)
    return embed

def embed(data, qatype, dic):
    output=[]
    out_count=0
    for line in data:
        count=0
        embedded=[]
        # value=str(line)
        for value in line:
            tokens=nltk.word_tokenize(value)
            tags=nltk.pos_tag(tokens)
            # print tags
            # if len(tags)>1:
            #     print tags
            for tag in tags:
                if (tag[1]!='.' and tag[1]!='\'\'' and tag[1]!=',' and tag[1]!='``' and tag[1]!='\'\''):
                    word=tag[0].lower()
                    embedded.append(word_embed(dic,word))
                    count += 1

            if qatype=='question':
                while(count<30):
                    embedded.append(int(0))
                    count += 1
                if (embedded[-1]!=0):
                    print ('too long!\n')
                    embedded=embedded[:30]

        # print embedded
        if len(embedded)!=4:
            print tags
        output.append(embedded)
        out_count+=1
        sys.stdout.write('\r%d words are added.'%(out_count))
        sys.stdout.flush()
    output_array=np.array(output)
    # print np.shape(output_array)
    # print output_array
    return output_array

# qatype=raw_input('input qatype:')
qatype='candidate'
dic=json.load(open('../dict/%s_dic_raw.json'%qatype,'r'))
data=json.load(open('../%s.json'%qatype,'r'))
# data=data[110950:]
out=embed(data, qatype, dic)

f=h5py.File('%s.h5'%qatype,'w')
f['data']=out
# f.create_dataset('data',)
f.close()

