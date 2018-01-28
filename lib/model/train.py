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

class SolverWrapper(object):
    '''
    wrapper class for the training process
    '''
    def __init__(self, sess, network, train_handle, valid_handle, test_handle, output_dir, tbdir):
        self.net = network
        self.train_handle = train_handle
        self.valid_handle = valid_handle
        self.test_handle = test_handle
        self.output_dir = output_dir
        self.tbdir = tbdir
        self.tbvaldir = tbdir + '_val'
        self.tbtestdir = tbdir + '_test'

        if not os.path.exists(self.tbvaldir):
            os.makedirs(self.tbvaldir)
        if not os.path.exists(self.tbtestdir):
            os.makedirs(self.tbtestdir)


    def snapshot(self, sess, iter):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = 'iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = 'iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

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
            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=70000)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
            self.valwriter = tf.summary.FileWriter(self.tbvaldir)
            self.testwriter = tf.summary.FileWriter(self.tbtestdir)

        return self.net.lr

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

    def initialize(self, sess):
        # Initial file lists are empty
        np_paths = []
        ss_paths = []
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        last_snapshot_iter = 0
        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = list(cfg.TRAIN.STEPSIZE)

        return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

    def restore(self, sess, sfile, nfile):
        # Get the most recent snapshot and restore
        np_paths = [nfile]
        ss_paths = [sfile]
        # Restore model from snapshots
        last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
        # Set the learning rate
        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            if last_snapshot_iter > stepsize:
                rate *= cfg.TRAIN.LEARNING_RATE_DECAY
            else:
                stepsizes.append(stepsize)

        return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

    def remove_snapshot(self, np_paths, ss_paths):
        to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)

        to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            # To make the code compatible to earlier versions of Tensorflow,
            # where the naming tradition for checkpoints are different
            if os.path.exists(str(sfile)):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile + '.data-00000-of-00001'))
                os.remove(str(sfile + '.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)


    def train_model(self, sess, max_iters):
        test_result = open(self.tbtestdir+'/out.txt', 'w')

        # Construct the computation graph
        lr = self.construct_graph(sess)

        # Find previous snapshots if there is any to restore from
        lsf, nfiles, sfiles = self.find_previous()

        # Initialize the variables or restore them from the last snapshot
        if lsf == 0:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(sess)
        else:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess, str(sfiles[-1]), str(nfiles[-1]))
        timer = Timer()
        iter = last_snapshot_iter + 1
        last_summary_time = time.time()
        # Make sure the lists are not empty
        lr_decay = cfg.TRAIN.LEARNING_RATE_DECAY ** max(iter/cfg.TRAIN.SNAPSHOT_ITERS-cfg.TRAIN.DECAY_EPOCH, 0.0)
        print(lr_decay)
        sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * lr_decay))
        while iter < max_iters + 1:
            timer.tic()
            now = time.time()
            if iter == 1 or now - last_summary_time > 180:
                # Compute the graph with summary
                accuracy, loss, summary = self.net.train_step_with_summary(sess, self.train_handle)
                self.writer.add_summary(summary, float(iter))
            else:
                # Compute the graph without summary
                accuracy, loss = self.net.train_step(sess, self.train_handle)
            timer.toc()

            # Display training information
            if iter % (cfg.TRAIN.DISPLAY) == 0:
                word_accuracy = [0.0 for p in range(5)]
                wup_accuracy = [0.0 for p in range(5)]

                for b in range(len(accuracy)):
                    if accuracy[b][0] == 0:
                        word_accuracy[b] = -1
                    else:
                        word_accuracy[b] = accuracy[b][1]/accuracy[b][0]
                    if accuracy[b][2] == 0:
                        wup_accuracy[b] = -1
                    else:
                        wup_accuracy[b] = accuracy[b][3]/accuracy[b][2]

                print('iter: %d / %d, loss: %.6f\n >>> lr: %f' % \
                      (iter, max_iters, loss, lr.eval()))
                print('speed: {:.3f}s / iter'.format(timer.average_time))
                print(word_accuracy)
                print(wup_accuracy)

            # Snapshotting
            if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                self.snapshot(sess, iter)
                # Learning rate
                lr_decay = cfg.TRAIN.LEARNING_RATE_DECAY ** max((iter/cfg.TRAIN.SNAPSHOT_ITERS-cfg.TRAIN.DECAY_EPOCH, 0.0))
                print(lr_decay)
                sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * lr_decay))

                print('run testing...')
                test_word_accuracy = [0.0 for p in range(5)]
                test_wup_count = [0 for p in range(5)]
                test_wup_value = [0.0 for p in range(5)]
                all_loss = 0.0
                for i in range(cfg.TEST.NUMBER):
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
                print(iter, file=test_result)
                print(test_word_accuracy, file=test_result)
                print(test_wup_accuracy, file =test_result)
                print(all_loss, file =test_result)
                print(loss, file =test_result)

                last_summary_time = now

                last_snapshot_iter = iter
                ss_path, np_path = self.snapshot(sess, iter)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                # Remove the old snapshots if there are too many
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    self.remove_snapshot(np_paths, ss_paths)

            iter += 1

        if last_snapshot_iter != iter - 1:
            self.snapshot(sess, iter - 1)

        self.writer.close()
        self.valwriter.close()
        self.testwriter.close()




def train_net(network, train_reader, valid_reader, test_reader, output_dir, tb_dir, max_iters=40000):
    """Train a GCA-net"""

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:

        iter_train_handle = train_reader.batch_iterator.string_handle()
        iter_valid_handle = valid_reader.batch_iterator.string_handle()
        iter_test_handle = test_reader.batch_iterator.string_handle()

        sess.run(train_reader.batch_iterator.initializer)
        sess.run(valid_reader.batch_iterator.initializer)
        sess.run(test_reader.batch_iterator.initializer)

        handle_train = sess.run(iter_train_handle)
        handle_valid = sess.run(iter_valid_handle)
        handle_test = sess.run(iter_test_handle)

        sw = SolverWrapper(sess, network, handle_train, handle_valid, handle_test, output_dir, tb_dir)
        print('Solving...')
        sw.train_model(sess, max_iters)
        print('done solving')
