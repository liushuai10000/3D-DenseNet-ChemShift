"""
Regular 3D-CNN and 3D-ResNet models
"""


from keras.models import Model
from keras.layers.core import Flatten, Dense
from keras.layers import Lambda, Input, Add, Concatenate
from keras.layers.convolutional import AveragePooling3D
from model.utils import conv_bn_relu, dense_bn_relu



def cnn_3d(num_channel):
    
    """
    The cnn with same number of 3x3x3 convolutional layers as 3D densenet
    Each block has 4 small conv_bn_relu blocks
    num_channel: number of channels
    """
    
    # input layer
    inp = Input((16, 16, 16, num_channel))
    # first block
    x = conv_bn_relu(inp, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = AveragePooling3D(2)(x)
    
    # second block
    x = conv_bn_relu(x, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = AveragePooling3D(2)(x)
    
    # final block
    x = Flatten()(x)
    x = dense_bn_relu(x, 256, 0.1)
    x = dense_bn_relu(x, 128, 0.1)
    y = Dense(1)(x)
    return Model(inp, y)



def res_net_3d(num_channel):
    
    """
    The resnet with same number of 3x3x3 convolutional layers as 3D densenet
    Each block has 4 small conv_bn_relu blocks
    num_channel: number of channels
    """
    
    # input layer
    inp = Input((16, 16, 16, num_channel))
    # first block
    x = conv_bn_relu(inp, 64, 3, "same")
    x1 = x
    x = conv_bn_relu(x, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = Add()([x, x1])
    x = AveragePooling3D(2)(x)
    
    # second block
    x1 = x
    x = conv_bn_relu(x, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = conv_bn_relu(x, 64, 3, "same")
    x = Add()([x, x1])
    x = AveragePooling3D(2)(x)
    
    # final block
    x = Flatten()(x)
    x = dense_bn_relu(x, 256, 0.1)
    x = dense_bn_relu(x, 128, 0.1)
    y = Dense(1)(x)
    return Model(inp, y)