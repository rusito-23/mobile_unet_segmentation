"""
Train Configuration Node
"""
from yacs.config import CfgNode as CN


""" CONFIG """

_C = CN()

# train config

_C.LEARNING_RATE = 1e-3
_C.BATCH_SIZE = 96
_C.EPOCHS = 100

# model config

_C.WEIGHTS_URL = ('https://github.com/fchollet/'
                  'deep-learning-models/releases/download/v0.6/'
                  'mobilenet_1_0_224_tf_no_top.h5')

# data config

_C.IN_SIZE = 224
_C.OUT_SIZE = 128
_C.N_CLASSES = 1
_C.AUG = False

# dataset paths config

_C.TRAIN_PATH = 'dataset/supervisely/train'
_C.VAL_PATH = 'dataset/supervisely/val'
_C.TEST_PATH = 'dataset/supervisely/test'
_C.IM_PATH = 'images'
_C.SEG_PATH = 'segs'
