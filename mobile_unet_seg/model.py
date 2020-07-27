"""
Semantic Segmentation Model.
MobileNet Backbone + UNet Head.
"""

from tensorflow.keras import (
    Model,
    Sequential
)
from tensorflow.keras.layers import (
    Input,
    ZeroPadding2D,
    Conv2D,
    BatchNormalization,
    Activation,
    DepthwiseConv2D,
    Reshape,
    UpSampling2D,
    concatenate
)
from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K

DF = 'channels_last'
CHANNEL_AXIS = -1
ALPHA = 1
DEPTH_MULT = 1


def relu6(x):
    return K.relu(x, max_value=6)


class ConvBlock(Sequential):
    """
    Convolutional Block with:
        - Zero Padding
        - Conv2D
        - Batch Normalization
        - ReLu6 Activation
    Used exclusively for the MobileNet backbone.
    """
    def __init__(self,
                 filters,
                 kernel=(3, 3),
                 strides=(1, 1)):
        super(ConvBlock, self).__init__([
            ZeroPadding2D(padding=(1, 1), data_format=DF, name='conv1_pad'),
            Conv2D(int(filters * ALPHA),
                   kernel,
                   data_format=DF,
                   padding='valid',
                   use_bias=False,
                   strides=strides,
                   name='conv1'),
            BatchNormalization(axis=CHANNEL_AXIS, name='conv1_bn'),
            Activation(relu6, name='conv1_relu'),
        ])


class DepthwiseBlock(Sequential):
    """
    Depthwise Block:
        - Zero Padding
        - DepthwiseConv2D
        - Batch Normalization
        - ReLu6 Activation
        - Extra Conv2D Layer + BN + ReLu6
    Used exclusively for the MobileNet backbone.
    """
    def __init__(self,
                 filters,
                 strides=(1, 1),
                 block=1):
        super(DepthwiseBlock, self).__init__([
            ZeroPadding2D((1, 1), data_format=DF, name=f'conv_pad_{block}'),
            DepthwiseConv2D((3, 3),
                            data_format=DF,
                            padding='valid',
                            depth_multiplier=DEPTH_MULT,
                            strides=strides,
                            use_bias=False,
                            name=f'conv_dw_{block}'),
            BatchNormalization(axis=CHANNEL_AXIS, name=f'conv_dw_{block}_bn'),
            Activation(relu6, name=f'conv_dw_{block}_relu'),
            Conv2D(int(filters * ALPHA),
                   (1, 1),
                   data_format=DF,
                   padding='same',
                   use_bias=False,
                   strides=(1, 1),
                   name=f'conv_pw_{block}'),
            BatchNormalization(axis=CHANNEL_AXIS, name=f'conv_pw_{block}_bn'),
            Activation(relu6, name=f'conv_pw_{block}_relu')
        ])


class EncoderBlock(Sequential):
    """
    MobileNet Encoder Block.
    Performs Depthwise Blocks to add channels to the input.
    The initial Depthwise Block uses (2, 2) stride, the rest of them
    use (1, 1) (default).
    Parameters:
        - filters: channels
        - n_blocks: number of blocks to create
        - block_id: initial block number
    """
    def __init__(self,
                 filters,
                 n_blocks,
                 block_id):
        super(EncoderBlock, self).__init__()
        self.add(DepthwiseBlock(filters, strides=(2, 2), block=block_id))
        for i in range(1, n_blocks):
            self.add(DepthwiseBlock(filters, block=(block_id + i)))


class MobileNet(Model):
    """
    MobileNet Encoder
    Saves partial output to perform skip connections for each Encoder Block.
    """

    def __init__(self, cfg):
        super(MobileNet, self).__init__()
        # setup blocks
        self.conv_32 = ConvBlock(32, strides=(2, 2))
        self.encoder_block_64 = EncoderBlock(64, n_blocks=1, block_id=1)
        self.encoder_block_128 = EncoderBlock(128, n_blocks=2, block_id=2)
        self.encoder_block_256 = EncoderBlock(256, n_blocks=2, block_id=4)
        self.encoder_block_512 = EncoderBlock(512, n_blocks=6, block_id=6)
        self.encoder_block_1024 = EncoderBlock(1024, n_blocks=2, block_id=12)

    def call(self, x):
        x = self.conv_32(x)
        x = self.encoder_block_64(x)
        f1 = x
        x = self.encoder_block_128(x)
        f2 = x
        x = self.encoder_block_256(x)
        f3 = x
        x = self.encoder_block_512(x)
        f4 = x
        x = self.encoder_block_1024(x)
        f5 = x
        return (f1, f2, f3, f4, f5)


class DecoderBlock(Sequential):
    """
    Performs operations to decode the input data.
    Uses operations:
        - ZeroPadding - Conv2D - BN - (optional) UpSampling
    Parameters:
        - filters: target channels
        - with_upsampling: bool to perform final upsampling
    """
    def __init__(self, filters, with_upsampling=True):
        super(DecoderBlock, self).__init__([
            ZeroPadding2D((1, 1), data_format=DF),
            Conv2D(filters, (3, 3),
                   padding='valid',
                   activation='relu',
                   data_format=DF),
            BatchNormalization(),
        ])
        if with_upsampling:
            self.add(UpSampling2D((2, 2), data_format=DF))


class UNet(Model):
    """
    UNet Decoder.
    Runs decoder blocks over the input data, performing skip connections
    (in fact, the `call` method takes the skip connection tuple)
    and reaches the `n_classes`.
    """
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.decoder_block_512 = DecoderBlock(512)
        self.decoder_block_256 = DecoderBlock(256)
        self.decoder_block_128 = DecoderBlock(128)
        self.decoder_block_64 = DecoderBlock(64, with_upsampling=False)
        self.conv_out = Conv2D(cfg.N_CLASSES,
                               (3, 3),
                               padding='same',
                               data_format=DF)

    def call(self, levels):
        (f1, f2, f3, f4, f5) = levels
        o = f4
        o = self.decoder_block_512(o)
        o = concatenate([o, f3], axis=CHANNEL_AXIS)
        o = self.decoder_block_256(o)
        o = concatenate([o, f2], axis=CHANNEL_AXIS)
        o = self.decoder_block_128(o)
        o = concatenate([o, f1], axis=CHANNEL_AXIS)
        o = self.decoder_block_64(o)
        o = self.conv_out(o)
        return o


class MobileUNetSegmentation(Model):
    """
    MobileUNetSegmentation:
    Uses MobileNet Encoder and UNet Decoder to perform Semantic Segmentation.
    Intended for binary classification, uses Sigmoid activation.
    """

    def __init__(self, cfg):
        super(MobileUNetSegmentation, self).__init__()

        # set up backbone - head
        self.encoder = MobileNet(cfg)
        self.decoder = UNet(cfg)

        # set up config
        self.n_classes = cfg.N_CLASSES
        self.input_height = self.input_width = cfg.IN_SIZE
        self.output_height = self.out_width = cfg.OUT_SIZE

        # set up segmentation layers
        self.reshape = Reshape((self.output_height*self.out_width, -1))
        self.activation = Activation('sigmoid')

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        x = self.reshape(x)
        x = self.activation(x)
        return x
