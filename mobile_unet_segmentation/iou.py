"""
Intersection Over Union (Jaccard) for Keras and Numpy.
"""
from tensorflow.keras import backend as K
import numpy as np


def iou(y_true, y_pred, thres=0.5, label=1):
    y_pred = K.cast(K.greater(y_pred, thres), dtype='float32')
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection / union


def iou_np(y_true, y_pred, thres=0.5, label=1):
    y_pred = (y_pred > thres).astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / union
