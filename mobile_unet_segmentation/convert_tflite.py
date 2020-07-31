"""
Convert model to tflite.
"""

import os
import argparse
import tensorflow as tf
from model import relu6


def convert_tflite(model_dir):
    # create required path
    keras_model_path = os.path.join(model_dir, 'model.h5')
    tflite_model_path = os.path.join(model_dir, 'model.tflite')

    # load
    with tf.keras.utils.CustomObjectScope({'relu6': relu6}):
        keras_model = tf.keras.models.load_model(keras_model_path)

    # convert
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open(tflite_model_path, "wb").write(tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()
    convert_tflite(args.model_dir)
