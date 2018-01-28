# --------------------------------------------------------
# GCA-Net
# Licensed under The MIT License [see LICENSE for details]
# Written by Yimeng Li
# --------------------------------------------------------
from model.config import cfg
from utils.timer import Timer
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import os
import sys
import glob
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

class TestWrapper(object):
    '''
    wrapper class for the training process
    '''
    def __init__(self, sess, network, test_handle, output_dir):
        self.net = network
        self.test_handle = test_handle
        self.output_dir = output_dir

    def from_snapshot(self, sess, sfile, nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(sess, sfile)
        print('Restored.')
        with open(nfile, 'rb') as fid:
            st0 = pickle.load(fid)
            last_snapshot_iter = pickle.load(fid)

            np.random.set_state(st0)

        return last_snapshot_iter

    def construct_graph(self, sess):
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)
            # Build the main computation graph
            layers = self.net.create_architecture('TRAIN', tag='default')
            self.saver = tf.train.Saver()

    def find_previous(self):
        sfiles = os.path.join(self.output_dir, 'iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in TensorFlow
        redfiles = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            redfiles.append(os.path.join(self.output_dir, 'iter_{:d}.ckpt.meta'.format(stepsize+1)))
        sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

        nfiles = os.path.join(self.output_dir, 'iter_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn not in redfiles]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        return lsf, nfiles, sfiles

    def restore(self, sess, sfile, nfile):
        # Get the most recent snapshot and restore
        np_paths = [nfile]
        ss_paths = [sfile]
        # Restore model from snapshots
        last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
        rate = cfg.TRAIN.LEARNING_RATE

        return rate, last_snapshot_iter
    def test_model(self, sess):
        test_result = open(self.output_dir+'/out.txt', 'w')

        self.construct_graph(sess)
        # Find previous snapshots if there is any to restore from
        lsf, nfiles, sfiles = self.find_previous()

        # Initialize the variables or restore them from the last snapshot
        rate, last_snapshot_iter = self.restore(sess, str(sfiles[-1]), str(nfiles[-1]))
        timer = Timer()
        timer.tic()
        now = time.time()
        print('run testing...')
        test_word_accuracy = [0.0 for p in range(5)]
        test_wup_count = [0 for p in range(5)]
        test_wup_value = [0.0 for p in range(5)]
        all_loss = 0.0
        for i in range(cfg.TEST.NUMBER):
            print(i)
            result = self.net.test_step(sess, self.test_handle)
            all_loss += result[3]
            test_word_accuracy[result[0]] += result[1]
            if result[2] != -1:
                test_wup_count[result[0]] += 1
                test_wup_value[result[0]] += result[2]

        test_word_accuracy = [i/cfg.TEST.NUMBER for i in test_word_accuracy]
        test_wup_accuracy = list(map(lambda x,y: x/y, test_wup_value, test_wup_count))
        all_loss /= cfg.TEST.NUMBER
        print(test_word_accuracy)
        print(test_wup_accuracy)
        print(last_snapshot_iter, file=test_result)
        print(test_word_accuracy, file=test_result)
        print(test_wup_accuracy, file =test_result)
        print(all_loss, file =test_result)





def test_net(network, test_reader, output_dir, tb_dir):
    """Train a GCA-net"""

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:

        iter_test_handle = test_reader.batch_iterator.string_handle()
        sess.run(test_reader.batch_iterator.initializer)
        handle_test = sess.run(iter_test_handle)

        sw = TestWrapper(sess, network, handle_test, output_dir)
        print('Solving...')
        sw.test_model(sess)
        print('done solving')
