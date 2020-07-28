"""
Handles InferenceSession Prediction.
Applies pre & post-processing.
"""
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import cv2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def relu6(x):
    return K.relu(x, max_value=6)


class Predictor:

    def __init__(self, in_size, out_size, model_file):

        # inference session
        with keras.utils.CustomObjectScope({'relu6': relu6}):
            self.sess = keras.models.load_model(model_file)

        # tensor variables
        self.in_shape = (in_size, in_size)
        self.out_shape = (out_size, out_size)
        self.threshold = 0.8

    def preprocess(self, image):
        _input = cv2.resize(image, self.in_shape)
        _input = _input / 255.0
        _input -= 0.5
        _input *= 2.
        return _input

    def postprocess(self, _output):
        _output = _output.reshape(self.out_shape)
        mask = (_output > self.threshold).astype(np.uint8) * 255
        mask = np.tile(mask.reshape(*self.out_shape, 1), 3)
        return mask

    def get_mask(self, image):
        # create input with shape
        _input = self.preprocess(image)
        _input = np.array([_input])

        # pass input through the net

        _output = self.sess.predict(_input)[0]

        # transform output into mask
        mask = self.postprocess(_output)

        return mask
