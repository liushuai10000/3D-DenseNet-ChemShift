"""
Testing script: example for hydrogen
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import concurrent.futures
from keras import regularizers as reg
from keras.models import Sequential, load_model, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import BatchNormalization, Activation, ReLU, Lambda, Input, Add, Concatenate
from keras.layers.convolutional import Convolution3D, Convolution1D
from keras.optimizers import Nadam
from keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K
from keras.callbacks import LearningRateScheduler

data_dir = "preprocessing/"
test_y = np.load(data_dir + "test_H_y.npy")
m = np.load(data_dir + "train_H_y.npy").mean()
scale = 1
size = test_y.shape[0]
test_x = np.zeros((size*8, 16, 16, 16, 20), dtype=np.float16)
for i in range(8):
    s = str(i)
    test_x[size*i:size*(i+1)] = np.concatenate([np.load(data_dir + "test_x_H_2A_" + s + ".npy"), 
                                                np.load(data_dir + "test_x_H_3A_" + s + ".npy"), 
                                                np.load(data_dir + "test_x_H_4A_" + s + ".npy"),
                                                np.load(data_dir + "test_x_H_5A_" + s + ".npy"),
                                                np.load(data_dir + "test_x_H_7A_" + s + ".npy")], axis=-1)
model = load_model("model_H.h5")
pred = model.predict(test_x, batch_size=128)
np.save("predicted_value", pred)
pred = np.mean(pred.reshape((8, -1)), axis=0)
rms = np.sqrt(np.mean((pred*scale-test_y+m) ** 2))
print(rms)
# 0.36-0.37 for H
# 3.2-3.3 for C
# 9.4-10.6 for N
# 14.5-15.9 for O

