import json
import os
from nltk.tokenize import WordPunctTokenizer

qa_json = json.load(open('../qa_1018.json'))
answer_json = json.load(open('../answer_one.json'))
candidate_json = json.load(open('../candidate.json'))
train_list = json.load(open('../list/train_list.json'))
valid_list = json.load(open('../list/valid_list.json'))
test_list = json.load(open('../list/test_list.json'))
answer_encode = json.load(open('../dict/answer_one_dic_raw.json'))
question_encode = json.load(open('../dict/question_dic_raw.json'))

result = {}

def look_code(st, dictionary, length):
    if type(st) is str:
        word_list = WordPunctTokenizer().tokenize(st)
    else:
        word_list = st

    out = [0 for i in range(length)]
    for idx, i in enumerate(word_list):
        if i.lower() in dictionary:
            out[idx] = dictionary[i.lower()][0]
        else:
            out[idx] = 0
    return out

train_video_list = os.listdir('/home/mm/dataset/activityNet/frames/training')
valid_video_list = os.listdir('/home/mm/dataset/activityNet/frames/validation')


for idx, i in enumerate(qa_json):
    temp = {}
    qa_id = i[0]
    temp['video_name'] = i[1]

    if len(temp['video_name']) == 13:
        video_name = temp['video_name'][2:]
    else:
        video_name = temp['video_name']
    if video_name in train_video_list:
        frame_path = os.path.join('/home/mm/dataset/activityNet/frames/training', video_name)
    elif video_name in valid_video_list:
        frame_path = os.path.join('/home/mm/dataset/activityNet/frames/validation', video_name)
    else:
        print(video_name + ' : not founded!')
        continue

    temp['frame_path']=frame_path
    temp['duration'] = i[2]
    temp['video_code'] = i[0]
    start_end = i[3][1:-1].strip(' ').split(',')
    temp['start'] = int(25*float(start_end[0]))
    temp['end'] = int(25*float(start_end[1]))
    last_frame = len(os.listdir(frame_path))
    if temp['end'] > last_frame:
        temp['end'] = last_frame
    if temp['end']-temp['start'] < 40:
        continue
    temp['question'] = i[4]
    temp['answer'] = i[5]
    temp['answer_one'] = answer_json[idx]
    temp['candidate'] = candidate_json[idx]

    temp['answer_code'] = look_code([temp['answer_one']], answer_encode, 1)
    temp['candidate_code'] = look_code(temp['candidate'], answer_encode, 4)
    temp['question_code'] = look_code(temp['question'], question_encode, 30)

    if(temp['video_name'] in train_list):
        temp['split'] = 'training'
    elif temp['video_name'] in valid_list:
        temp['split'] = 'validation'
    elif temp['video_name'] in test_list:
        temp['split'] = 'testing'
    else:
        print(temp['video_name']+':  not found on the list!')

    result[str(idx)] = temp

fp = open('../no_missing_activityNet_qa_40.json', 'w')
json.dump(result, fp)
fp.close()
