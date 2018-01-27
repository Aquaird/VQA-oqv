# --------------------------------------------------------
# Tensorflow GCA-Net
# Licensed under The MIT License [see LICENSE for details]
# Written by Yimeng Li
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np
from utils.visualization import draw_bounding_boxes

from model.config import cfg

class Network(object):
    def __init__(self):
        self._predictions = {}
        self._losses = {}
        self._layers = {}
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}


    def _build_t_atten(self):
        with tf.variable_scope('T_ATTEN'):
            #[batch_size, 20, 4096]
            video_feature = tf.add((cfg.GAMMA * self._vgg_feature), (1-cfg.GAMMA) * self._c3d_feature)
            # q embedding
            ques_embedding_input = tf.reshape(self._question, [-1, 1001])
            w_ques_embedding = tf.get_variable('ques_embedding', [1001, cfg.T_ATTEN_DIM], initializer=self.initializer)
            ques_embeded = tf.nn.tanh(tf.matmul(ques_embedding_input, w_ques_embedding))
            ques_embeded = tf.reshape(ques_embeded, [-1, 30, cfg.T_ATTEN_DIM])
            # [batch_size, T_ATTEN_DIM]
            ques_atten = tf.reduce_sum(ques_embeded, axis=1)
            # [batch_size, 20, T_ATTEN_DIM]
            ques_atten = tf.tile(tf.reshape(ques_atten, [-1,1,cfg.T_ATTEN_DIM]), [1, 20, 1])

            # v encoding
            video_fc_input = tf.reshape(video_feature, [-1, 4096])
            w_video_fc = tf.get_variable('video_fc', [4096, cfg.T_ATTEN_DIM], initializer=self.initializer)
            video_fced = tf.nn.tanh(tf.matmul(video_fc_input, w_video_fc))
            # [batch_size, 20, T_ATTEN_DIM]
            video_fced = tf.reshape(video_fced, [-1, 20, cfg.T_ATTEN_DIM])

            # attention
            t_atten_bias = tf.get_variable('bias', [cfg.T_ATTEN_DIM], initializer=self.initializer)
            t_atten_h = tf.tanh(tf.nn.bias_add(tf.add(video_fced, ques_atten), t_atten_bias))
            t_atten_h = tf.reshape(t_atten_h, [-1, cfg.T_ATTEN_DIM])
            w_softmax = tf.get_variable('softmax', [cfg.T_ATTEN_DIM, 1], initializer=self.initializer)
            t_atten_h = tf.reshape(tf.matmul(t_atten_h, w_softmax), [-1,20])
            t_atten = tf.nn.softmax(t_atten_h, dim=-1)
            self._t_atten = t_atten
            # [batch_size, 1, 20]
            t_atten = tf.reshape(t_atten, (-1,1,20))
            feature_t_atten = tf.matmul(t_atten, video_feature)
            # [batch_size, 4096]
            feature_t_atten = tf.reshape(feature_t_atten, [-1, 4096])
            w_t_atten_feat = tf.get_variable('w_t_atten_feat', [4096, cfg.T_ATTEN_FEATURE_DIM])
            feature_t_atten = tf.nn.sigmoid(tf.matmul(feature_t_atten, w_t_atten_feat))
            self._feature_t_atten = feature_t_atten

    def _build_mlp(self):
        with tf.variable_scope('MLP'):
            obj_feature = tf.reshape(self._obj_feature, [-1, 5, 4096])
            w_mlp_top = tf.get_variable('w_mlp_top', (cfg.MLP_SIZE, 5), initializer=self.initializer)
            w_mlp_bottom = tf.get_variable('w_mlp_bottom', (5, cfg.MLP_SIZE), initializer=self.initializer)

            w_mlp_top_broadcast = tf.tile(tf.reshape(w_mlp_top, [1,3,5]), [obj_feature.get_shape().as_list()[0], 1, 1])
            w_mlp_bottom_broadcast = tf.tile(tf.reshape(w_mlp_bottom, [1,3,5]), [obj_feature.get_shape().as_list()[0], 1, 1])

            # [batch_size*20, 3, 4096]
            h_mlp = tf.sigmoid(tf.matmul(w_mlp_top_broadcast, obj_feature))
            # [batch_size*20, 5, 4096]
            out_mlp = tf.sigmoid(tf.matmul(w_mlp_top_broadcast, h_mlp))
            out_mlp = tf.reshape(out_mlp, [-1,4096])
            w_pca = tf.get_variable('pca', [4096, cfg.LSTM_SIZE], initializer=self.initializer)
            out_mlp = tf.reshape(tf.nn.sigmoid(tf.matmul(out_mlp, w_pca)), (-1,20,5,cfg.LSTM_SIZE))
            self.drop_prob = tf.placeholder_with_default(1.0, shape=())
            self.out_mlp = tf.nn.dropout(out_mlp, keep_prob=self.drop_prob)

    def _build_g_atten(self):
        with tf.variable_scope('G_ATTEN'):
            embedding_question = tf.reshape(self._question, [-1,1001])
            w_embedding = tf.get_variable('embedding', [1001, cfg.LSTM_SIZE], initializer=self.initializer)
            embeded_question = tf.reshape(tf.nn.sigmoid(tf.matmul(embedding_question, w_embedding)), (-1,30,cfg.LSTM_SIZE))

            # q att step.1
            ques_atten_embed = tf.reshape(embeded_question, [-1, cfg.LSTM_SIZE])
            w_ques_atten_embed = tf.get_variable('w_ques_atten_embed', [cfg.LSTM_SIZE, cfg.G_ATTEN_DIM], initializer=self.initializer)
            ques_embeded = tf.nn.tanh(tf.matmul(ques_atten_embed, w_ques_atten_embed))
            w_ques_g1 = tf.get_variable('w_ques_g1', [cfg.G_ATTEN_DIM, 1], initializer=self.initializer)
            ques_h1 = tf.reshape(tf.matmul(ques_embeded, w_ques_g1), [-1, 30])
            # [batch_size, 30]
            ques_p1 = tf.nn.softmax(ques_h1, dim=-1)
            self._quesAtt_1 = ques_p1
            ques_p1 = tf.reshape(ques_p1, [-1, 1, 30])
            # [batch_size, lstm_size]
            feature_quesAtt1 = tf.reshape(tf.matmul(ques_p1, embeded_question), (-1, cfg.LSTM_SIZE))

            # obj att step
            obj_fc_input = tf.reshape(self.out_mlp, [-1, cfg.LSTM_SIZE])
            w_obj_fc = tf.get_variable('obj_fc', [cfg.LSTM_SIZE, cfg.G_ATTEN_DIM], initializer=self.initializer)
            obj_fced = tf.nn.tanh(tf.matmul(obj_fc_input, w_obj_fc))
            # [batch_size*20, 5, G_ATTEN_DIM]
            obj_fced = tf.reshape(obj_fced, [-1, 5, cfg.G_ATTEN_DIM])
            w_quesAtt1_embed = tf.get_variable('w_quesAtt1_embed', [cfg.LSTM_SIZE, cfg.G_ATTEN_DIM], initializer=self.initializer)
            quesAtt1_embeded = tf.matmul(ques_atten_embed, w_quesAtt1_embed)
            # [batch_size*20, 5, G_ATTEN_DIM]
            quesAtt1_broadcast = tf.tile(tf.reshape(quesAtt1_embeded, [-1,1,1,cfg.G_ATTEN_DIM]), [1,20,5,1])
            quesAtt1_broadcast = tf.reshape(quesAtt1_broadcast, [-1,5,cfg.G_ATTEN_DIM])
            obj_h = tf.nn.tanh(tf.add(obj_fced, quesAtt1_broadcast))
            obj_h = tf.reshape(obj_h, [-1, cfg.G_ATTEN_DIM])
            w_obj_g = tf.get_variable('w_obj_g', [cfg.G_ATTEN_DIM, 1], initializer=self.initializer)
            obj_p = tf.reshape(tf.matmul(obj_h, w_obj_g), [-1,20,5])
            obj_p = tf.nn.softmax(obj_p, dim=-1)
            self._objAtt = obj_p
            obj_p = tf.reshape(obj_p, [-1, 20, 1, 5])
            obj_p = tf.reshape(obj_p, [-1, 1, 5])
            # [batch_size, lstm_size]
            feature_objAtt = tf.reshape(tf.matmul(obj_p, tf.reshape(self.out_mlp, (-1,5,cfg.LSTM_SIZE))), (-1, cfg.LSTM_SIZE))
            self._feature_objAtt = tf.reshape(feature_objAtt, [-1,20,cfg.LSTM_SIZE])

            # question att step2
            ques_embed_2 = tf.reshape(self._question, [-1, cfg.LSTM_SIZE])
            w_ques_embed2 = tf.get_variable('ques_embed2', [cfg.LSTM_SIZE, cfg.G_ATTEN_DIM], initializer=self.initializer)
            ques_embeded_2 = tf.nn.tanh(tf.matmul(ques_embed_2, w_ques_embed2))
            # [batch_size, 30, G_ATTEN_DIM]
            ques_embeded_2 = tf.reshape(ques_embeded_2, [-1, 30, cfg.G_ATTEN_DIM])
            # [batch_size, 20, 30, G_ATTEN_DIM]
            ques_embeded_2_broadcast = tf.tile(tf.reshape(ques_embeded_2,[-1,1,30,cfg.G_ATTEN_DIM]), [1,20,1,1])
            w_objAtt_fc = tf.get_variable('w_objAtt_fc', [cfg.LSTM_SIZE, cfg.G_ATTEN_DIM], initializer=self.initializer)
            objAtt_fced = tf.matmul(feature_objAtt, w_objAtt_fc)
            # [batch_size, 20, 30, G_ATTEN_DIM]
            obj_fced = tf.reshape(objAtt_fced, [-1,20,cfg.G_ATTEN_DIM])
            objAtt_broadcast = tf.tile(tf.reshape(objAtt_fced, [-1,20,1,cfg.G_ATTEN_DIM]), [1,1,30,1])
            ques_h_2 = tf.nn.tanh(tf.add(ques_embeded_2_broadcast, objAtt_broadcast))
            ques_h_2 = tf.reshape(ques_h_2, [-1, cfg.G_ATTEN_DIM])
            w_ques_g_2 = tf.get_variable('w_ques_g_2', [cfg.G_ATTEN_DIM, 1], initializer=self.initializer)
            # [batch_size, 20, 30]
            ques_p_2 = tf.reshape(tf.matmul(ques_h_2, w_ques_g_2), [-1, 20, 30])
            ques_p_2 = tf.nn.softmax(ques_p_2, dim=-1)
            # [batch_size, 20, 30]
            self._quesAtt2 = ques_p_2
            ques_p_2 = tf.reshape(ques_p_2, [-1,20, 1,30])
            ques_p_2 = tf.reshape(ques_p_2, [-1, 1, 30])
            # [batch_size, lstm_size]
            embeded_ques_broadcast = tf.tile(tf.reshape(embeded_question,[-1,1,30,cfg.LSTM_SIZE]), [1,20,1,1])
            embeded_ques_broadcast = tf.reshape(embeded_ques_broadcast, [-1, 30, cfg.LSTM_SIZE])
            feature_quesAtt2 = tf.reshape(tf.matmul(ques_p_2, embeded_ques_broadcast), (-1, 20, cfg.LSTM_SIZE))
            self._feature_quesAtt2 = feature_quesAtt2

    def _build_lstm(self):
        lstm_input = tf.concat([self._feature_quesAtt2, self._feature_objAtt], axis=)









    def _build_network(self, is_training=True):
        # select initializers
        if cfg.TRAIN.TRUNCATED:
            self.initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        else:
            self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        self._build_t_atten()
        self._build_mlp()





    def create_architecture(self, mode, tag=None):
        self._vgg_feature = tf.placeholder(tf.float32, shape=[None, 20, 4096])
        self._c3d_feature = tf.placeholder(tf.float32, shape=[None, 20, 4096])
        self._obj_feature = tf.placeholder(tf.float32, shape=[None, 20, 5, 4096])
        self._question = tf.placeholder(tf.int64, shape=[None, 30, 1001])
        self._question = tf.placeholder(tf.int64, shape=[None, 30, 1001])
        self._answer = tf.placeholder(tf.int64, shape=[None, 1, 1001])

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        #regularizers
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        self._build_network(training)
