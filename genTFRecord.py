import tensorflow as tf
import json
import h5py
import numpy as np

def _floats_feature(value):
    value = np.reshape(value, (-1))
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64s_feature(value):
    value = np.reshape(value, (-1))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value)) 

def _detect_qtype(question):
    if question.split(' ')[0].lower() == 'what':
        return [0]
    if question.split(' ')[0].lower() == 'who':
        return [1]
    if question.split(' ')[0].lower() == 'how':
        return [2]
    if question.split(' ')[0].lower() == 'where':
        return [3]
    return [4]


def _one_hot(code_list, size):
    result = np.zeros((len(code_list), size), dtype=np.int64)
    for i, label in enumerate(code_list):
        result[i][label] = 1

    return result

anno = json.load(open('./dataset/activityNet/qa_anno/anno_all.json', 'r'))
h5feature = h5py.File('./dataset/activityNet/features_h5/all_20.h5', 'r')

split = 'train'
dset = h5feature['training']
j = anno[split]

with tf.python_io.TFRecordWriter('/data/mm/dataset/activityNet/train.tfrecords') as writer:
    count = 0
    qtype_count = [0,0,0,0,0]
    for i in j:
        # [20,7,4096] float
        features = dset[count]
        # [30, 1001] int
        question_code = j[i]['question_code']
        question = _one_hot(question_code, 1001)
        # [1, 1001] int
        answer_code = j[i]['answer_code']
        answer = _one_hot(answer_code, 1001)
        # [4, 1001] int
        candidate_code = j[i]['candidate_code']
        candidate = _one_hot(candidate_code, 1001)
        # [1] int
        question_kind = _detect_qtype(j[i]['question'])
        qtype_count[question_kind[0]] += 1
        # [1] int
        qa_id = [int(i)]

        example = tf.train.Example(
            features=tf.train.Features(feature={
                'qa_id':_int64s_feature(qa_id),
                'qtype':_int64s_feature(question_kind),
                'video_feature': _floats_feature(features),
                'question':_int64s_feature(question),
                'answer':_int64s_feature(answer),
                'candidate':_int64s_feature(candidate)
            })
        )
        writer.write(example.SerializeToString())
        count += 1
        print(count)
print(qtype_count)
