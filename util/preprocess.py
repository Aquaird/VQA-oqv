"""Preprocess the data for model."""
import os
import inspect
import csv

import numpy as np
from PIL import Image
import scipy
import tensorflow as tf
import pandas as pd

from .vgg16 import Vgg16
from .c3d import c3d


class VideoVGGExtractor(object):
    """Select uniformly distributed frames and extract its VGG feature."""

    def __init__(self, frame_num, sess):
        """Load VGG model.

        Args:
            frame_num: number of frames per video.
            sess: tf.Session()
        """
        self.frame_num = frame_num
        self.inputs = tf.placeholder(tf.float32, [self.frame_num, 224, 224, 3])
        self.vgg16 = Vgg16()
        self.vgg16.build(self.inputs)
        self.sess = sess

    def _select_frames(self, path, start, end):
        """Select representative frames for video.

        Ignore some frames both at begin and end of video.

        Args:
            path: Path of video.
            start: Start frame of this qa
            end: End frame of this qa
        Returns:
            frames: list of frames.
        """
        frames = list()
        total_frames = end - start
        # Ignore some frame at begin and end.
        for i in np.linspace(start, end, self.frame_num + 2)[1:self.frame_num + 1]:
            img = Image.open(os.path.join(path, 'image_'+str(int(i)+1).zfill(5)+'.jpg'))
            img = img.resize((224, 224), Image.BILINEAR)
            frame_data = np.array(img)
            frames.append(frame_data)
        return frames

    def extract(self, anno):
        """Get VGG fc7 activations as representation for video.

        Args:
            frame_path: frames directory for qa data.
        Returns:
            feature: [self.frame_number, 4096]
        """
        path = anno['frame_path']
        start = anno['start']
        end = anno['end']
        frames = self._select_frames(path, start, end)
        # We usually take features after the non-linearity, by convention.
        feature = self.sess.run(
            self.vgg16.relu7, feed_dict={self.inputs: frames})
        return feature

    def extract_from_h5(self, h5array):
        """Get VGG fc7 activations as representation for video.

        Args:
            h5array: frames list for image data.
        Returns:
            feature: [self.frame_number, 4096]
        """
        # Ignore some frame at begin and end.
        end = h5array.shape[0]
        frames = list()
        for i in np.linspace(0, end, self.frame_num + 2)[1:self.frame_num + 1]:
            img = h5array[int(i)]
            frames.append(np.array(img))
        # We usually take features after the non-linearity, by convention.
        feature = self.sess.run(
            self.vgg16.relu7, feed_dict={self.inputs: frames})

        return feature


class VideoC3DExtractor(object):
    """Select uniformly distributed clips and extract its C3D feature."""

    def __init__(self, clip_num, sess):
        """Load C3D model."""
        self.clip_num = clip_num
        self.inputs = tf.placeholder(
            tf.float32, [self.clip_num, 16, 112, 112, 3])
        _, self.c3d_features = c3d(self.inputs, 1, clip_num)
        saver = tf.train.Saver()
        path = inspect.getfile(VideoC3DExtractor)
        path = os.path.dirname(path)
        saver.restore(sess, os.path.join(
            path, 'sports1m_finetuning_ucf101.model'))
        self.mean = np.load(os.path.join(path, 'crop_mean.npy'))
        self.sess = sess

    def _select_clips(self, path, start, end):
        """Select self.batch_size clips for video. Each clip has 16 frames.

        Args:
            path: Path of frames of qa data.
            start: start frame
            end: end frame
        Returns:
            clips: list of clips.
        """
        clips = list()
        for i in np.linspace(start, end, self.clip_num + 2)[1:self.clip_num + 1]:
            # Select center frame first, then include surrounding frames
            clip_start = int(i) - 8
            clip_end = int(i) + 8
            if clip_start < start:
                clip_end = clip_end - clip_start
                clip_start = start
            if clip_end > end:
                clip_start = clip_start - (clip_end - end)
                clip_end = end
            new_clip = []
            for j in range(16):
                img = Image.open(os.path.join(path, 'image_'+str(int(j)+1+clip_start).zfill(5)+'.jpg'))
                img = img.resize((112, 112), Image.BILINEAR)
                frame_data = np.array(img) * 1.0
                frame_data -= self.mean[j]
                new_clip.append(frame_data)
            clips.append(new_clip)
        return clips

    def extract(self, anno):
        """Get 4096-dim activation as feature for video.

        Args:
            path: Path of frames of qa data.
            start: start frame of qa data
        Returns:
            feature: [self.num_clips, 4096]
        """
        path = anno['frame_path']
        start = anno['start']
        end = anno['end']

        clips = self._select_clips(path, start, end)
        feature = self.sess.run(
            self.c3d_features, feed_dict={self.inputs: clips})
        return feature

    def extract_from_h5(self, h5array):
        """Get C3D fc7 activations as representation for video.

        Args:
            h5array: frames list for image data.
        Returns:
            feature: [self.frame_number, 4096]
        """
        # Ignore some frame at begin and end.
        end = h5array.shape[0]
        clips = list()

        for i in np.linspace(0, end, self.clip_num + 2)[1:self.clip_num + 1]:
            # Select center frame first, then include surrounding frames
            clip_start = int(i) - 8
            clip_end = int(i) + 8
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > end:
                clip_start = clip_start - (clip_end - end)
                clip_end = end
            new_clip = []
            for j in range(16):
                img = h5array[j]
                frame_data = np.array(img) * 1.0
                frame_data -= self.mean[j]
                new_clip.append(frame_data)
            clips.append(new_clip)

        feature = self.sess.run(
            self.c3d_features, feed_dict={self.inputs: clips})

        return feature


