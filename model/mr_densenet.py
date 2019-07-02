"""
DenseNet models
"""

from keras.models import Model
from keras.layers import Lambda, Input, Add, Concatenate, Flatten, Dense
from keras.layers.convolutional import AveragePooling3D
from model.utils import conv_bn_relu, dense_bn_relu



def densenet_baseline_1(num_channel):
    
    """
    The baseline (regular) DenseNet with two DenseNet blocks
    Each block has 4 small conv_bn_relu blocks
    num_channel: number of channels
    """
    
    # input layer
    inp = Input((16, 16, 16, num_channel))
    # first transition layer
    x = conv_bn_relu(inp, 64, 3, "same")
    # first block
    x1 = conv_bn_relu(x, 256, 1, "same")
    x1 = conv_bn_relu(x1, 64, 1, "same")
    x1 = conv_bn_relu(x1, 64, 3, "same")
    x2 = Concatenate()([x, x1])
    x2 = conv_bn_relu(x2, 256, 1, "same")
    x2 = conv_bn_relu(x2, 64, 1, "same")
    x2 = conv_bn_relu(x2, 64, 3, "same")
    x3 = Concatenate()([x, x1, x2])
    x3 = conv_bn_relu(x3, 256, 1, "same")
    x3 = conv_bn_relu(x3, 64, 1, "same")
    x3 = conv_bn_relu(x3, 64, 3, "same")
    x4 = Concatenate()([x, x1, x2, x3])
    x4 = conv_bn_relu(x4, 256, 1, "same")
    x4 = conv_bn_relu(x4, 64, 1, "same")
    x4 = conv_bn_relu(x4, 64, 3, "same")
    x = AveragePooling3D(2)(x4)
    
    # second transition layer
    x = conv_bn_relu(x, 64, 1, "same")
    # second block
    x1 = conv_bn_relu(x, 256, 1, "same")
    x1 = conv_bn_relu(x1, 64, 1, "same")
    x1 = conv_bn_relu(x1, 64, 3, "same")
    x2 = Concatenate()([x, x1])
    x2 = conv_bn_relu(x2, 256, 1, "same")
    x2 = conv_bn_relu(x2, 64, 1, "same")
    x2 = conv_bn_relu(x2, 64, 3, "same")
    x3 = Concatenate()([x, x1, x2])
    x3 = conv_bn_relu(x3, 256, 1, "same")
    x3 = conv_bn_relu(x3, 64, 1, "same")
    x3 = conv_bn_relu(x3, 64, 3, "same")
    x4 = Concatenate()([x, x1, x2, x3])
    x4 = conv_bn_relu(x4, 256, 1, "same")
    x4 = conv_bn_relu(x4, 64, 1, "same")
    x4 = conv_bn_relu(x4, 64, 3, "same")
    x = AveragePooling3D(2)(x4)
    
    # final block
    x = Flatten()(x)
    x = dense_bn_relu(x, 256, 0.1)
    x = dense_bn_relu(x, 128, 0.1)
    y = Dense(1)(x)
    return Model(inp, y)



def densenet_baseline_2(num_channel):
    
    """
    The baseline (regular) DenseNet with two DenseNet blocks
    Each block has 4 small conv_bn_relu blocks
    At the end of each block, there are two 1x1x1 convolution layers to have similar number of parameters as MR-3D-DenseNet
    num_channel: number of channels
    """
    
    # input layer
    inp = Input((16, 16, 16, num_channel))
    # first transition layer
    x = conv_bn_relu(inp, 64, 3, "same")
    # first block
    x1 = conv_bn_relu(x, 256, 1, "same")
    x1 = conv_bn_relu(x1, 64, 1, "same")
    x1 = conv_bn_relu(x1, 64, 3, "same")
    x2 = Concatenate()([x, x1])
    x2 = conv_bn_relu(x2, 256, 1, "same")
    x2 = conv_bn_relu(x2, 64, 1, "same")
    x2 = conv_bn_relu(x2, 64, 3, "same")
    x3 = Concatenate()([x, x1, x2])
    x3 = conv_bn_relu(x3, 256, 1, "same")
    x3 = conv_bn_relu(x3, 64, 1, "same")
    x3 = conv_bn_relu(x3, 64, 3, "same")
    x4 = Concatenate()([x, x1, x2, x3])
    x4 = conv_bn_relu(x4, 256, 1, "same")
    x4 = conv_bn_relu(x4, 64, 1, "same")
    x4 = conv_bn_relu(x4, 64, 3, "same")
    x = AveragePooling3D(2)(x4)
    x = conv_bn_relu(x, 256, 1, "same")
    x = conv_bn_relu(x, 64, 1, "same")
    
    # second transition layer
    x = conv_bn_relu(x, 64, 1, "same")
    # second block
    x1 = conv_bn_relu(x, 256, 1, "same")
    x1 = conv_bn_relu(x1, 64, 1, "same")
    x1 = conv_bn_relu(x1, 64, 3, "same")
    x2 = Concatenate()([x, x1])
    x2 = conv_bn_relu(x2, 256, 1, "same")
    x2 = conv_bn_relu(x2, 64, 1, "same")
    x2 = conv_bn_relu(x2, 64, 3, "same")
    x3 = Concatenate()([x, x1, x2])
    x3 = conv_bn_relu(x3, 256, 1, "same")
    x3 = conv_bn_relu(x3, 64, 1, "same")
    x3 = conv_bn_relu(x3, 64, 3, "same")
    x4 = Concatenate()([x, x1, x2, x3])
    x4 = conv_bn_relu(x4, 256, 1, "same")
    x4 = conv_bn_relu(x4, 64, 1, "same")
    x4 = conv_bn_relu(x4, 64, 3, "same")
    x = AveragePooling3D(2)(x4)
    x = conv_bn_relu(x, 256, 1, "same")
    x = conv_bn_relu(x, 64, 1, "same")
    
    # final block
    x = Flatten()(x)
    x = dense_bn_relu(x, 256, 0.1)
    x = dense_bn_relu(x, 128, 0.1)
    y = Dense(1)(x)
    return Model(inp, y)



def mr_densenet(num_channel):
    
    """
    The MR-DenseNet with two DenseNet blocks
    Each block has 4 small conv_bn_relu blocks
    At the end of each block, we concatenate the center segment of the pooling layer
    num_channel: number of channels
    """
    
    # input layer
    inp = Input((16,16,16,num_channel))
    # first transition layer
    x = conv_bn_relu(inp, 64, 3, "same")
    # first block
    x1 = conv_bn_relu(x, 256, 1, "same")
    x1 = conv_bn_relu(x1, 64, 1, "same")
    x1 = conv_bn_relu(x1, 64, 3, "same")
    x2 = Concatenate()([x, x1])
    x2 = conv_bn_relu(x2, 256, 1, "same")
    x2 = conv_bn_relu(x2, 64, 1, "same")
    x2 = conv_bn_relu(x2, 64, 3, "same")
    x3 = Concatenate()([x, x1, x2])
    x3 = conv_bn_relu(x3, 256, 1, "same")
    x3 = conv_bn_relu(x3, 64, 1, "same")
    x3 = conv_bn_relu(x3, 64, 3, "same")
    x4 = Concatenate()([x, x1, x2, x3])
    x4 = conv_bn_relu(x4, 256, 1, "same")
    x4 = conv_bn_relu(x4, 64, 1, "same")
    x = conv_bn_relu(x4, 64, 3, "same")
    x1 = AveragePooling3D(2)(x)
    x2 = Lambda(lambda x: x[:,4:-4,4:-4,4:-4])(x)
    x = Concatenate()([x1, x2])
    x = conv_bn_relu(x, 256, 1, "same")
    x = conv_bn_relu(x, 64, 1, "same")
    
    # second transition layer
    x = conv_bn_relu(x, 64, 1, "same")
    # second block
    x1 = conv_bn_relu(x, 256, 1, "same")
    x1 = conv_bn_relu(x1, 64, 1, "same")
    x1 = conv_bn_relu(x1, 64, 3, "same")
    x2 = Concatenate()([x, x1])
    x2 = conv_bn_relu(x2, 256, 1, "same")
    x2 = conv_bn_relu(x2, 64, 1, "same")
    x2 = conv_bn_relu(x2, 64, 3, "same")
    x3 = Concatenate()([x, x1, x2])
    x3 = conv_bn_relu(x3, 256, 1, "same")
    x3 = conv_bn_relu(x3, 64, 1, "same")
    x3 = conv_bn_relu(x3, 64, 3, "same")
    x4 = Concatenate()([x, x1, x2, x3])
    x4 = conv_bn_relu(x4, 256, 1, "same")
    x4 = conv_bn_relu(x4, 64, 1, "same")
    x = conv_bn_relu(x4, 64, 3, "same")
    x1 = AveragePooling3D(2)(x)
    x2 = Lambda(lambda x: x[:,2:-2,2:-2,2:-2])(x)
    x = Concatenate()([x1, x2])
    x = conv_bn_relu(x, 256, 1, "same")
    x = conv_bn_relu(x, 64, 1, "same")
    
    # final block
    x = Flatten()(x)
    x = dense_bn_relu(x, 256, 0.1)
    x = dense_bn_relu(x, 128, 0.1)
    y = Dense(1)(x)
    return Model(inp, y)

