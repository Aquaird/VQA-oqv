import tensorflow as tf
import os.path as osp
import numpy as np
from model.config import cfg

class tf_reader(object):
    '''dataset reader'''

    def __init__(self, tf_record_path, config):
        self.config = config
        print(config)
        dataset = tf.data.TFRecordDataset([tf_record_path], buffer_size=config.BUFFER_SIZE)
        dataset = dataset.repeat(cfg.TRAIN.EPOCH)
        if self.config.SHUFFLE:
            dataset = dataset.shuffle(self.config.BATCH_SIZE * 3)
        dataset = dataset.map(self.parse_function, num_parallel_calls=12)
        dataset = dataset.prefetch(self.config.BUFFER_SIZE)
        self.dataset = dataset.batch(self.config.BATCH_SIZE)
        self.batch_iterator = self.dataset.make_initializable_iterator()

    def parse_function(self, example_proto):
        features = {
            'qa_id': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            'qtype': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            'video_feature': tf.FixedLenFeature(shape=[20,7,4100], dtype=tf.float32),
            'question': tf.FixedLenFeature(shape=[30,1001], dtype=tf.int64),
            'answer': tf.FixedLenFeature(shape=[1001], dtype=tf.int64),
            'candidate': tf.FixedLenFeature(shape=[4,1001], dtype=tf.int64),
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        return parsed_features
