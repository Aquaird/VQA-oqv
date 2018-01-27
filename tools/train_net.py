# --------------------------------------------------------
# Video Question Answering via Grounded Co-attention Networking Learning
# Licensed under The MIT License [see LICENSE for details]
# Written by Yimeng Li
# --------------------------------------------------------

import _init_paths
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from model.train import train_net
from dataset.tfrecord import tf_reader
from nets.network import Network
import argparse
import numpy as np
import sys
import h5py
import json

import tensorflow as tf

def parse_args():
    '''
    Parse input argument
    '''

    parser = argparse.ArgumentParser(description='Train a GCA-Net')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='activityNet', type=str)
    parser.add_argument('--weight', dest='weight',
                        help='initialize with pretrained model')
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=700*cfg.TRAIN.EPOCH, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default=None, type=str)


    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args

def get_features(h5_file, json_file, split, with_candidate=False):
    question_vectors = []
    answer_vectors = []
    candidate_vectors = []

    features = h5_file[split]
    anno = json_file[split]

    if with_candidate:
        return features, (question_vectors, answer_vectors, candidate_vectors)
    else:
        return features, (question_vectors, answer_vectors)

if __name__ == '__main__':
    args = parse_args()

    print('Called with args: ')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    #if args.set_cfgs is not None:
    #    cfg_from_list(args.set_cfgs)

    print("Using config:")
    print(cfg)

    output_dir = get_output_dir(args.dataset_name, args.tag)
    print('Output will be saved to `{:s}`'.format(output_dir))
    tb_dir =  get_output_tb_dir(args.dataset_name, args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    train_reader = tf_reader('/data/mm/dataset/activityNet/train.tfrecords', cfg['TRAIN'])
    valid_reader = tf_reader('/data/mm/dataset/activityNet/valid.tfrecords', cfg['TEST'])
    test_reader = tf_reader('/data/mm/dataset/activityNet/test.tfrecords', cfg['TEST'])

    net = Network(train_reader.dataset.output_types, train_reader.dataset.output_shapes)
    train_net(net, train_reader, valid_reader, test_reader, output_dir, tb_dir,args.max_iters)
