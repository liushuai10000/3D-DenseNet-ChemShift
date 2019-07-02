from keras import regularizers as reg
from keras.models import Sequential, load_model, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import BatchNormalization, Activation, ReLU, Lambda, Input, Add, Concatenate
from keras.layers.convolutional import Convolution3D, Convolution1D, MaxPooling3D, AveragePooling3D
from keras.regularizers import l2
import tensorflow as tf
from keras import backend as K
import numpy as np


def conv_bn_relu(x, n, s, pad, reg=3e-5, d=1):
    
    """
    A block in different CNNs with convolution layer, batch_norm and relu.
    x: input keras layer
    n: number of filters
    s: size of the kernel
    pad: padding method, "valid" or "same"
    reg: lambda in l2 regularization
    d: dilation rate
    """
    
    x = Convolution3D(filters=n, kernel_size=s, W_regularizer=l2(reg), padding=pad, dilation_rate=d)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x



def dense_bn_relu(x, n, d, reg=3e-5):
    
    """
    A block in different CNNs with convolution layer, batch_norm, relu and dropout layer.
    x: input keras layer
    n: number of nodes in fully connected layer
    d: dropout rate
    """
    
    x = Dense(n, W_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(d)(x)
    return x

def exp_decay(epoch):
    
    """
    the exponential decay function, starting from 1e-3, decay rate with 0.6
    epoch: epoch index
    """
    
    initial_lrate = 1e-3
    rate = 0.25
    return initial_lrate * np.exp(-rate*epoch)