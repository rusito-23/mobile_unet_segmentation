"""
MobileUNet Segmentation Model.
Uses the `fchollet` pretrained weights on ImageNet.
Refactored from: https://github.com/divamgupta/image-segmentation-keras
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    ZeroPadding2D,
    Conv2D,
    Input,
    BatchNormalization,
    Activation,
    DepthwiseConv2D,
    UpSampling2D,
    Reshape,
    concatenate
)
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_file


DATA_FORMAT = 'channels_last'
MERGE_AXIS = -1


def relu6(x):
    return K.relu(x, max_value=6)


def _conv_block(inputs,
                filters,
                alpha,
                kernel=(3, 3),
                strides=(1, 1)):
    """ Convolutional Block """
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad',
                      data_format=DATA_FORMAT)(inputs)
    x = Conv2D(filters, kernel, data_format=DATA_FORMAT,
               padding='valid',
               use_bias=False,
               strides=strides,
               name='conv1')(x)
    x = BatchNormalization(axis=MERGE_AXIS, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_block(inputs,
                     pointwise_conv_filters,
                     alpha,
                     depth_multiplier=1,
                     strides=(1, 1),
                     block=1):
    """ Deptwise Convolutional Block """
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = ZeroPadding2D((1, 1), data_format=DATA_FORMAT,
                      name='conv_pad_%d' % block)(inputs)
    x = DepthwiseConv2D((3, 3), data_format=DATA_FORMAT,
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block)(x)
    x = BatchNormalization(
        axis=MERGE_AXIS, name='conv_dw_%d_bn' % block)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1), data_format=DATA_FORMAT,
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block)(x)
    x = BatchNormalization(axis=MERGE_AXIS,
                           name='conv_pw_%d_bn' % block)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block)(x)


def MobileNet(in_size,
              weights_url,
              alpha=1.0,
              depth_mul=1):
    """ MobileNet Encoder """

    # check data format
    assert (K.image_data_format() == 'channels_last'),\
        'Currently only `channels_last` mode is supported'

    # create encoder
    im_input = Input(shape=(in_size, in_size, 3))
    x = _conv_block(im_input, 32, alpha, strides=(2, 2))
    x = _depthwise_block(x, 64, alpha, depth_mul, block=1)
    f1 = x
    x = _depthwise_block(x, 128, alpha, depth_mul, strides=(2, 2), block=2)
    x = _depthwise_block(x, 128, alpha, depth_mul, block=3)
    f2 = x
    x = _depthwise_block(x, 256, alpha, depth_mul, strides=(2, 2), block=4)
    x = _depthwise_block(x, 256, alpha, depth_mul, block=5)
    f3 = x
    x = _depthwise_block(x, 512, alpha, depth_mul, strides=(2, 2), block=6)
    x = _depthwise_block(x, 512, alpha, depth_mul, block=7)
    x = _depthwise_block(x, 512, alpha, depth_mul, block=8)
    x = _depthwise_block(x, 512, alpha, depth_mul, block=9)
    x = _depthwise_block(x, 512, alpha, depth_mul, block=10)
    x = _depthwise_block(x, 512, alpha, depth_mul, block=11)
    f4 = x
    x = _depthwise_block(x, 1024, alpha, depth_mul, strides=(2, 2), block=12)
    x = _depthwise_block(x, 1024, alpha, depth_mul, block=13)
    f5 = x

    # load pretrained weights
    weights_path = get_file('mobilenet_imagenet_weights', weights_url)
    Model(im_input, x).load_weights(weights_path)

    return im_input, [f1, f2, f3, f4, f5]


def SegmentationModel(i, o):
    """ Segmentation Model with Sigmoid Activation """
    o_shape = Model(i, o).output_shape
    i_shape = Model(i, o).input_shape
    out_height, out_width = o_shape[1], o_shape[2]
    in_height, in_width = i_shape[1], i_shape[2]
    n_classes = o_shape[3]
    o = (Reshape((out_height*out_width, -1)))(o)
    o = (Activation('sigmoid'))(o)
    model = Model(i, o)
    model.n_classes = n_classes
    model.input_height = in_height
    model.input_width = in_width
    model.output_width = out_width
    model.output_height = out_height
    return model


def MobileUNet(n_classes,
               in_size,
               out_size,
               weights_url):
    """
    Segmentation Model
        - MobileNet Backbone
        - UNet Head
        - Sigmoid Activation
    """

    i, levels = MobileNet(in_size, weights_url)
    [f1, f2, f3, f4, f5] = levels

    o = f4
    o = (ZeroPadding2D((1, 1), data_format=DATA_FORMAT))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=DATA_FORMAT))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format=DATA_FORMAT))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=DATA_FORMAT))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=DATA_FORMAT))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format=DATA_FORMAT))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=DATA_FORMAT))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format=DATA_FORMAT))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format=DATA_FORMAT))(o)
    o = (concatenate([o, f1], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=DATA_FORMAT))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=DATA_FORMAT))(o)
    o = (BatchNormalization())(o)
    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=DATA_FORMAT)(o)

    model = SegmentationModel(i, o)
    model.model_name = "mobilenet_unet"
    return model
