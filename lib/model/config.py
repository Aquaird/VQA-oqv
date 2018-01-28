import os
import os.path as osp
import numpy as np

from easydict import EasyDict as edict

__C = edict()
cfg = __C
# Training options
__C.TRAIN = edict()
# Shuffle
__C.TRAIN.SHUFFLE = False
# Prefetch Buffer size
__C.TRAIN.BUFFER_SIZE = 1000
# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001
# learning decay
__C.TRAIN.LEARNING_RATE_DECAY = 0.98
# Epoch
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.EPOCH = 70
__C.TRAIN.DECAY_EPOCH = 1
# Momentum
__C.TRAIN.MOMENTUM = 0.9
# Weight decay
__C.TRAIN.WEIGHT_DECAY = 0.00001
# Learning rate decay
# Step size for reducing the learning rate
__C.TRAIN.STEPSIZE = [5000]
# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 10
# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = True
# Whether to initialize the weights with truncated normal distribution 
__C.TRAIN.TRUNCATED = False
# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False
# The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.SNAPSHOT_KEPT = 10
# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 5000
# Minibatch size
# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = int(90000/__C.TRAIN.BATCH_SIZE)
# Whether or not has candidate
__C.TRAIN.CANDIDATE = True

#
# Testing options
#
__C.TEST = edict()
__C.TEST.SHUFFLE=False
__C.TEST.BATCH_SIZE=1
__C.TEST.BUFFER_SIZE=1000
__C.TEST.NUMBER= 22472
# For reproducibility
__C.RNG_SEED = 3
# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'dataset'))
# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'
# Place outputs under an experiments directory
__C.EXP_DIR = 'init_ex'

__C.T_ATTEN_DIM = 50
__C.G_ATTEN_DIM = 50
__C.LSTM_SIZE = 512
__C.T_ATTEN_FEATURE_DIM = 1024
__C.MLP_SIZE = 10
__C.GAMMA = 0.5

def get_output_dir(dataset, tag):
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, dataset))
  if tag is None:
    tag = 'default'
  outdir = osp.join(outdir, tag)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir

def get_output_tb_dir(dataset, tag):
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, dataset))
  if tag is None:
    tag = 'default'
  outdir = osp.join(outdir, tag)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir

def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))
    print(yaml_cfg)

  _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value
