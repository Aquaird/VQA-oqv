"""Preprocess the data of ActivityNet"""
import os
import sys
import json
import numpy as np

import h5py
import tensorflow as tf

from util.preprocess import VideoVGGExtractor
from util.preprocess import VideoC3DExtractor
from util.bar import ShowProcess
import time

def extract_video_feature(feature_path, json_path, gpu, chunks_size=32):
    """Extract video features(vgg, c3d) and store in hdf5 file."""

    anno_jsons = json.load(open(json_path))
    all_chunks = int(len(anno_jsons)/chunks_size)
    f = h5py.File(feature_path, 'w')
    

    dset = f.create_dataset('vgg', shape=(chunks_size, 20, 4096), maxshape=(None, 20, 4096), chunks=(chunks_size,20,4096), dtype='float32')
    # Session config.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = gpu

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        extractor = VideoVGGExtractor(20, sess)
        tmp_list = []
        video_code = '0'
        feature_before = []
        count = chunks_size
        process_bar = ShowProcess(all_chunks)
        for i in anno_jsons:
            if anno_jsons[i]['video_code'] == video_code:
                feature = feature_before
            else:
                feature = extractor.extract(anno_jsons[i])
                video_code = anno_jsons[i]['video_code']
                feature_before = feature
            tmp_list.append(feature)
            if len(tmp_list) == chunks_size:
                dset[-chunks_size:] = tmp_list
                count += chunks_size
                dset.resize(count, axis=0)
                tmp_list = []
                process_bar.show_process()

        dset[count-chunks_size:count-chunks_size+len(tmp_list),:,:] = np.array(tmp_list)
        dset.resize(count-chunks_size+len(tmp_list), axis=0)

    f.close()


def main():
    extract_video_feature('dataset/activityNet/vgg_feature_20.h5', 'dataset/activityNet/qa_anno/activityNet_qa_40.json', '0,1')

if __name__ == '__main__':
    main()
