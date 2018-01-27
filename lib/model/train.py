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
    def __init__(self, sess, network, data_batch, output_dir, tbdir):
        self.net = network
        self.train_batch = data_batch[0]
        self.valid_batch = data_batch[1]
        self.test_batch = data_batch[2]
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
        filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
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

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map 
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                "with SNAPPY.")

    def construct_graph(self, sess):
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)
            # Build the main computation graph
            layers = self.net.create_architecture('TRAIN', tag='default')
            # Define the loss
            loss = layers['total_loss']
            # Set learning rate and momentum
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

            # Compute the gradients with regard to the loss
            gvs = self.optimizer.compute_gradients(loss)
            # Double the gradient of the bias if set
            if cfg.TRAIN.DOUBLE_BIAS:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult') as scope:
                    for grad, var in gvs:
                        scale = 1.
                        if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = self.optimizer.apply_gradients(final_gvs)
            else:
                train_op = self.optimizer.apply_gradients(gvs)

            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=70000)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
            self.valwriter = tf.summary.FileWriter(self.tbvaldir)
            self.testwriter = tf.summary.FileWriter(self.tbtestdir)

        return lr, train_op

    def find_previous(self):
        sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in TensorFlow
        redfiles = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            redfiles.append(os.path.join(self.output_dir, 
                                         cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize+1)))
        sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

        nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
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
        # Fresh train directly from ImageNet weights
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
                rate *= cfg.TRAIN.GAMMA
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

        # Construct the computation graph
        lr, train_op = self.construct_graph(sess)

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
        stepsizes.append(max_iters)
        stepsizes.reverse()
        next_stepsize = stepsizes.pop()
        while iter < max_iters + 1:
            # Learning rate
            if iter == next_stepsize + 1:
                # Add snapshot here before reducing the learning rate
                self.snapshot(sess, iter)
                rate *= cfg.TRAIN.GAMMA
                sess.run(tf.assign(lr, rate))
                next_stepsize = stepsizes.pop()

            timer.tic()

            now = time.time()
            if iter == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
                # Compute the graph with summary
                total_loss, q_attention, o_attentions, t_attentions, summary = self.net.train_step_with_summary(sess, self.train_batch, train_op)
                self.writer.add_summary(summary, float(iter))
                # Also check the summary on the validation set
                summary_val = self.net.get_summary(sess, self.valid_batch)
                summary_test = self.net.get_summary(sess, self.test_batch)
                self.testwriter.add_summary(summary_test, float(iter))
                last_summary_time = now
            else:
                # Compute the graph without summary
                total_loss, q_attention, o_attentions, t_attentions, summary = self.net.train_step_with_summary(sess, self.train_batch, train_op)
            timer.toc()

            # Display training information
            if iter % (cfg.TRAIN.DISPLAY) == 0:
                print('iter: %d / %d, total loss: %.6f\n >>> lr: %f' % \
                      (iter, max_iters, total_loss, lr.eval()))
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            # Snapshotting
            if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
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




def train_net(network, data_iterators, output_dir, tb_dir, max_iters=40000):
    """Train a GCA-net"""

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        data_batch = []
        for i in data_iterators:
            data_batch.append(i.get_next())
            sess.run(i.initializer)

        sw = SolverWrapper(sess, network, data_batch, output_dir, tb_dir)
        print('Solving...')
        sw.train_model(sess, max_iters)
        print('done solving')
