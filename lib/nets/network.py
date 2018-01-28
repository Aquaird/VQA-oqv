# --------------------------------------------------------
# Tensorflow GCA-Net
# Licensed under The MIT License [see LICENSE for details]
# Written by Yimeng Li
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nltk.corpus import wordnet as wn
import json
import numpy as np
from model.config import cfg

answer_dict = json.load(open('/home/mm/workspace/VQA-oqv/dataset/activityNet/qa_anno/dict/answer_one_dic.json'))

class Network(object):
    def __init__(self, handle_types, handle_shapes):
        self._predictions = {}
        self._losses = {}
        self._layers = {}
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self.handle_types = handle_types
        self.handle_shapes = handle_shapes

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)


    def _build_t_atten(self):
        with tf.variable_scope('T_ATTEN'):
            #[batch_size, 20, 4096]
            video_feature = tf.add((cfg.GAMMA * self._vgg_feature), (1-cfg.GAMMA) * self._c3d_feature)
            # [batch_size, 4096]
            self._mean_video_feature = tf.reduce_mean(video_feature, axis=1)
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
            # [batch_size, 20, T_ATTEN_DIM]
            t_atten_h = tf.tanh(tf.nn.bias_add(tf.add(video_fced, ques_atten), t_atten_bias))
            t_atten_h = tf.reshape(t_atten_h, [-1, cfg.T_ATTEN_DIM])
            w_softmax = tf.get_variable('softmax', [cfg.T_ATTEN_DIM, 1], initializer=self.initializer)
            # [batch_size, 20]
            t_atten_h = tf.reshape(tf.matmul(t_atten_h, w_softmax), [-1,20])
            t_atten = tf.nn.softmax(t_atten_h, dim=-1)
            self._t_atten = t_atten
            # [batch_size, 1, 20]
            t_atten = tf.reshape(t_atten, (-1,1,20))
            # [batch_size, 20, 4096]
            feature_t_atten = tf.matmul(t_atten, video_feature)
            # [batch_size, 4096]
            feature_t_atten = tf.reshape(feature_t_atten, [-1, 4096])
            w_t_atten_feat = tf.get_variable('w_t_atten_feat', [4096, cfg.T_ATTEN_FEATURE_DIM])
            feature_t_atten = tf.matmul(feature_t_atten, w_t_atten_feat)
            # [batch_size, cfg.T_ATTEN_FEATURE_DIM]
            self._feature_t_atten = feature_t_atten

    def _build_mlp(self):
        with tf.variable_scope('MLP'):
            # [batch_size*20, 5, 4096]
            obj_feature = tf.reshape(self._obj_feature, [-1, 5, 4096])
            w_mlp_top = tf.get_variable('w_mlp_top', (cfg.MLP_SIZE, 5), initializer=self.initializer)
            w_mlp_bottom = tf.get_variable('w_mlp_bottom', (5, cfg.MLP_SIZE), initializer=self.initializer)

            #[batch_size*20, mlp_size, 5]
            to_tile = tf.shape(obj_feature)[0]
            w_mlp_top_broadcast = tf.tile(tf.reshape(w_mlp_top, [1,cfg.MLP_SIZE,5]), tf.stack([to_tile, 1, 1]))
            #[batch_size*20, 5, mlp_size]
            w_mlp_bottom_broadcast = tf.tile(tf.reshape(w_mlp_bottom, [1,5,cfg.MLP_SIZE]), [to_tile, 1, 1])

            # [batch_size*20, mlp_size, 4096]
            h_mlp = tf.nn.tanh(tf.matmul(w_mlp_top_broadcast, obj_feature))
            # [batch_size*20, 5, 4096]
            out_mlp = tf.nn.tanh(tf.matmul(w_mlp_bottom_broadcast, h_mlp))
            out_mlp = tf.reshape(out_mlp, [-1,4096])
            w_pca = tf.get_variable('pca', [4096, cfg.LSTM_SIZE], initializer=self.initializer)
            out_mlp = tf.reshape(tf.nn.tanh(tf.matmul(out_mlp, w_pca)), (-1,20,5,cfg.LSTM_SIZE))
            self.drop_prob = tf.placeholder_with_default(1.0, shape=())
            # [batch_size, 20, 5, lstm_size]
            self.out_mlp = tf.nn.dropout(out_mlp, keep_prob=self.drop_prob)

    def _build_g_atten(self):
        with tf.variable_scope('G_ATTEN'):
            embedding_question = tf.reshape(self._question, [-1,1001])
            w_embedding = tf.get_variable('embedding', [1001, cfg.LSTM_SIZE], initializer=self.initializer)
            # [batch_size, 30, lstm_size]
            embeded_question = tf.reshape(tf.matmul(embedding_question, w_embedding), (-1,30,cfg.LSTM_SIZE))

            # q att step.1
            ques_atten_embed = tf.reshape(embeded_question, [-1, cfg.LSTM_SIZE])
            w_ques_atten_embed = tf.get_variable('w_ques_atten_embed', [cfg.LSTM_SIZE, cfg.G_ATTEN_DIM], initializer=self.initializer)
            #[batch_size*30, G_ATTEN_DIM]
            ques_embeded = tf.nn.tanh(tf.matmul(ques_atten_embed, w_ques_atten_embed))
            w_ques_g1 = tf.get_variable('w_ques_g1', [cfg.G_ATTEN_DIM, 1], initializer=self.initializer)
            ques_h1 = tf.reshape(tf.matmul(ques_embeded, w_ques_g1), [-1, 30])
            # [batch_size, 30]
            ques_p1 = tf.nn.softmax(ques_h1, dim=-1)
            # [batch_size, 30]
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
            # [batch_size, G_ATTEN_DIM]
            quesAtt1_embeded = tf.matmul(feature_quesAtt1, w_quesAtt1_embed)
            # [batch_size, 20, 5, G_ATTEN_DIM]
            quesAtt1_broadcast = tf.tile(tf.reshape(quesAtt1_embeded, [-1,1,1,cfg.G_ATTEN_DIM]), [1,20,5,1])
            # [batch_size*20, 5, G_ATTEN_DIM]
            quesAtt1_broadcast = tf.reshape(quesAtt1_broadcast, [-1,5,cfg.G_ATTEN_DIM])
            obj_h = tf.nn.tanh(tf.add(obj_fced, quesAtt1_broadcast))
            obj_h = tf.reshape(obj_h, [-1, cfg.G_ATTEN_DIM])
            w_obj_g = tf.get_variable('w_obj_g', [cfg.G_ATTEN_DIM, 1], initializer=self.initializer)
            # [batch_size,20,5]
            obj_p = tf.reshape(tf.matmul(obj_h, w_obj_g), [-1,20,5])
            obj_p = tf.nn.softmax(obj_p, dim=-1)
            self._objAtt = obj_p
            obj_p = tf.reshape(obj_p, [-1, 20, 1, 5])
            # [batch_size*20, 1, 5]
            obj_p = tf.reshape(obj_p, [-1, 1, 5])
            # [batch_size*20, lstm_size]
            feature_objAtt = tf.reshape(tf.matmul(obj_p, tf.reshape(self.out_mlp, (-1,5,cfg.LSTM_SIZE))), (-1, cfg.LSTM_SIZE))
            # [batch_size, 20, lstm_size]
            self._feature_objAtt = tf.reshape(feature_objAtt, [-1,20,cfg.LSTM_SIZE])

            # question att step2
            ques_embed_2 = tf.reshape(embeded_question, [-1, cfg.LSTM_SIZE])
            w_ques_embed2 = tf.get_variable('ques_embed2', [cfg.LSTM_SIZE, cfg.G_ATTEN_DIM], initializer=self.initializer)
            ques_embeded_2 = tf.nn.tanh(tf.matmul(ques_embed_2, w_ques_embed2))
            # [batch_size, 30, G_ATTEN_DIM]
            ques_embeded_2 = tf.reshape(ques_embeded_2, [-1, 30, cfg.G_ATTEN_DIM])
            # [batch_size, 20, 30, G_ATTEN_DIM]
            ques_embeded_2_broadcast = tf.tile(tf.reshape(ques_embeded_2,[-1,1,30,cfg.G_ATTEN_DIM]), [1,20,1,1])

            w_objAtt_fc = tf.get_variable('w_objAtt_fc', [cfg.LSTM_SIZE, cfg.G_ATTEN_DIM], initializer=self.initializer)
            # [batch*20, G_ATTEN_DIM]
            objAtt_fced = tf.nn.tanh(tf.matmul(feature_objAtt, w_objAtt_fc))
            obj_fced = tf.reshape(objAtt_fced, [-1,20,cfg.G_ATTEN_DIM])
            # [batch_size, 20, 30, G_ATTEN_DIM]
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
            # [batch_size*20, 1, 30]
            ques_p_2 = tf.reshape(ques_p_2, [-1, 1, 30])
            # [batch_size, lstm_size]
            embeded_ques_broadcast = tf.tile(tf.reshape(embeded_question,[-1,1,30,cfg.LSTM_SIZE]), [1,20,1,1])
            # [batch_size*20, 30, lstm_size]
            embeded_ques_broadcast = tf.reshape(embeded_ques_broadcast, [-1, 30, cfg.LSTM_SIZE])
            # [batch_size, 20, lstm_size]
            feature_quesAtt2 = tf.reshape(tf.matmul(ques_p_2, embeded_ques_broadcast), (-1, 20, cfg.LSTM_SIZE))
            self._feature_quesAtt2 = feature_quesAtt2

    def _build_lstm(self):
        with tf.variable_scope('lstm'):
            #[batch_size, 20, 2*lstm_size]
            lstm_input = tf.concat([self._feature_quesAtt2, self._feature_objAtt], axis=-1)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(2*cfg.LSTM_SIZE)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.drop_prob)
            w_mean_video_fc = tf.get_variable('w_mean', [4096, 2*cfg.LSTM_SIZE], initializer=self.initializer)
            # [batch_size, 2*lstm_size]
            init_state = tf.nn.tanh(tf.matmul(self._mean_video_feature, w_mean_video_fc))
            hidden_state = init_state
            current_state = init_state
            state = tf.nn.rnn_cell.LSTMStateTuple(hidden_state, current_state)
            lstm_out, state = tf.nn.dynamic_rnn(lstm_cell, lstm_input, initial_state=state)
            # [batch_size, 20, lstm_size]
            self._feature_lstm = lstm_out


    def _build_network(self, is_training=True):
        # select initializers
        self.initializer = tf.random_uniform_initializer(-0.01, 0.01)

        self._build_t_atten()
        self._build_mlp()
        self._build_g_atten()
        self._build_lstm()

        with tf.variable_scope('final'):
            # [batch_size, lstm_size]
            att_lstm = tf.reshape(tf.matmul(tf.reshape(self._t_atten, [-1, 1, 20]), self._feature_lstm), [-1, 2*cfg.LSTM_SIZE])
            # [batch_size, t_att_size+lstm_size]
            softmax_input = tf.concat([self._feature_t_atten, att_lstm], axis=-1)
            w_class = tf.get_variable('w_class', [cfg.T_ATTEN_FEATURE_DIM+2*cfg.LSTM_SIZE, 1001], initializer=self.initializer)
            softmax_input = tf.matmul(softmax_input, w_class)
            softmax_output = tf.nn.softmax(softmax_input, dim=-1)
            word_pred = tf.reshape(tf.argmax(softmax_output, axis=1, name='word_pred'), (-1,1))

            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self._answer, logits=softmax_input)
            self._loss = tf.reduce_mean(cross_entropy)

        self._losses['class_loss'] = self._loss
        self._event_summaries.update(self._losses)
        answer_word = tf.reshape(tf.argmax(self._answer, axis=1, name='word'), (-1,1))
        self._answer_code = answer_word
        word_accuracy = tf.to_float(tf.equal(word_pred, answer_word))

        # Set learning rate and momentum
        self.lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        self.optimizer = tf.train.MomentumOptimizer(self.lr, cfg.TRAIN.MOMENTUM)
        #self.optimizer = tf.train.GradientDescentOptimizer(self.lr)

        # Compute the gradients with regard to the loss
        gvs = self.optimizer.compute_gradients(self._loss)
        self.train_op = self.optimizer.apply_gradients(gvs)


        #[ batch_size, 1001]
        self._predictions['word_score'] = softmax_output
        #[ batch_size, 20]
        self._predictions['t_attention'] = self._t_atten
        #[ batch_size, 20, 5]
        self._predictions['o_attention'] = self._objAtt
        #[ batch_size, 20, 30]
        self._predictions['q_attention'] = self._quesAtt2
        self._predictions['word_pred'] = word_pred
        self._predictions['word_accuracy'] = word_accuracy

        self._score_summaries.update(self._predictions)


        return word_pred


    def create_architecture(self, mode, tag=None):
        self.handle = tf.placeholder(tf.string, [])
        data_iterator = tf.data.Iterator.from_string_handle(self.handle, self.handle_types, self.handle_shapes)
        features = data_iterator.get_next()
        self._vgg_feature = tf.reshape((features['video_feature'][:,:,0,:4096]), [-1,20,4096])
        self._c3d_feature = tf.reshape((features['video_feature'][:,:,1,:4096]), [-1,20,4096])
        self._obj_feature = features['video_feature'][:,:,2:,:4096]
        self._question = tf.cast(features['question'], tf.float32)
        self._answer = tf.cast(features['answer'], tf.float32)
        self._candidate = tf.cast(features['candidate'], tf.float32)
        self.qa_id = tf.cast(features['qa_id'], tf.int32)
        self.qtype = tf.cast(features['qtype'], tf.int32)

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        #regularizers
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        word_pred = self._build_network(training)
        layers_to_output = {'word_pred':word_pred}
        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        layers_to_output.update(self._losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            for key, var in self._event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            for key, var in self._score_summaries.items():
                self._add_score_summary(key, var)
            for var in self._train_summaries:
                self._add_train_summary(var)

        self._summary_op = tf.summary.merge_all()
        self._summary_op_val = tf.summary.merge(val_summaries)

        layers_to_output.update(self._predictions)

        return layers_to_output

    def get_summary(self, sess, handle):
        feed_dict = {self.handle: handle}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary


    def test_step(self, sess, handle):
        feed_dict = {self.handle:handle}
        qtype, word_pred_code, answer_code, word_pred_accuracy, loss = sess.run([
            self.qtype,
            self._predictions['word_pred'],
            self._answer_code,
            self._predictions['word_accuracy'],
            self._loss,
        ], feed_dict=feed_dict)

        qtype= qtype[0][0]
        word_pred_code= word_pred_code[0][0]
        word_pred_accuracy= word_pred_accuracy[0][0]
        answer_code = answer_code[0][0]

        if answer_code == 0:
            return [qtype, word_pred_accuracy, -1, loss]
        else:
            answer = wn.synsets(answer_dict[str(answer_code)][0])
            if len(answer) == 0:
                return [qtype, word_pred_accuracy, -1, loss]
            answer = answer[0]
            if word_pred_code == 0:
                return [qtype, word_pred_accuracy, 0, loss]
            else:
                predic = wn.synsets(answer_dict[str(word_pred_code)][0])
                if len(predic) != 0:
                    predic = predic[0]
                    wup_value = answer.wup_similarity(predic)
                    if wup_value:
                        return [qtype, word_pred_accuracy, wup_value, loss]
                    else:
                        return [qtype, word_pred_accuracy, 0, loss]
                else:
                    return [qtype, word_pred_accuracy, 0, loss]

    def train_step(self, sess, handle):
        feed_dict = {self.handle:handle}
        qtype, word_pred_code, answer_code, word_pred_accuracy ,ta,qa,oa, loss, _ = sess.run([
            self.qtype,
            self._predictions['word_pred'],
            self._answer_code,
            self._predictions['word_accuracy'],
            self._predictions['t_attention'],
            self._predictions['q_attention'],
            self._predictions['o_attention'],
            self._loss,
            self.train_op
        ], feed_dict=feed_dict)

        qtype= np.reshape(qtype, [-1])
        word_pred_code= np.reshape(word_pred_code, [-1])
        word_pred_accuracy= np.reshape(word_pred_accuracy, [-1])
        answer_code = np.reshape(answer_code, [-1])

        accuracy = np.zeros((5,4), dtype='float')
        for i in range(len(qtype)):
            accuracy[qtype[i]][0] += 1
            accuracy[qtype[i]][1] += word_pred_accuracy[i]
            if answer_code[i] == 0:
                continue
            else:
                answer = wn.synsets(answer_dict[str(answer_code[i])][0])
                if len(answer) == 0:
                    continue
                answer = answer[0]
                accuracy[qtype[i]][2] += 1
                if word_pred_code[i] != 0:
                    predic = wn.synsets(answer_dict[str(word_pred_code[i])][0])
                    if len(predic) != 0:
                        predic = predic[0]
                        wup_value = answer.wup_similarity(predic)
                        if wup_value:
                            accuracy[qtype[i]][3] += answer.wup_similarity(predic)

        return accuracy, loss

    def train_step_with_summary(self, sess, handle):
        feed_dict = {self.handle:handle}
        qtype, word_pred_code, answer_code, word_pred_accuracy ,loss, summary, _ = sess.run([
            self.qtype,
            self._predictions['word_pred'],
            self._answer_code,
            self._predictions['word_accuracy'],
            self._loss,
            self._summary_op,
            self.train_op
        ], feed_dict=feed_dict)

        qtype= np.reshape(qtype, [-1])
        word_pred_code= np.reshape(word_pred_code, [-1])
        word_pred_accuracy= np.reshape(word_pred_accuracy, [-1])
        answer_code = np.reshape(answer_code, [-1])


        accuracy = np.zeros((5,4))
        for i in range(len(qtype)):
            accuracy[qtype[i]][0] += 1
            accuracy[qtype[i]][1] += word_pred_accuracy[i]
            if answer_code[i] == 0:
                continue
            else:
                answer = wn.synsets(answer_dict[str(answer_code[i])][0])
                if len(answer) == 0:
                    continue
                answer = answer[0]
                accuracy[qtype[i]][2] += 1
                if word_pred_code[i] != 0:
                    predic = wn.synsets(answer_dict[str(word_pred_code[i])][0])
                    if len(predic) != 0:
                        predic = predic[0]
                        wup_value = answer.wup_similarity(predic)
                        if wup_value:
                            accuracy[qtype[i]][3] += answer.wup_similarity(predic)

        return accuracy, loss, summary


