"""
The training script: an example for hydrogen
"""

import numpy as np
from model.mr_densenet import mr_densenet
from model.utils import exp_decay
import tensorflow as tf
from keras import backend as K
from keras.callbacks import LearningRateScheduler


data_dir = "preprocessing/"
train_y = np.load(data_dir + "train_H_y.npy")
size = train_y.shape[0]
m = np.mean(train_y) # substract the mean during training
scale = 1 # scale the std of different atom type to encourage convergence
train_y = np.concatenate([train_y for _ in range(8)]) # because it's augmented
train_x = np.zeros((size*8, 16, 16, 16, 20), dtype=np.float16) # generate the tensor first

# fill the tensor with data
for i in range(8):
    s = str(i)
    train_x[size*i:size*(i+1)] = np.concatenate([np.load(data_dir + "train_H_x_2A_" + s + ".npy"),
                                                 np.load(data_dir + "train_H_x_3A_" + s + ".npy"),
                                                 np.load(data_dir + "train_H_x_4A_" + s + ".npy"),
                                                 np.load(data_dir + "train_H_x_5A_" + s + ".npy"),
                                                 np.load(data_dir + "train_H_x_7A_" + s + ".npy")], axis=-1)

lrate = LearningRateScheduler(exp_decay)
model = mr_densenet(20)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
# mean is substracted. for C, N and O, we also need to scale it with a factor
model.fit(train_x, (train_y-m)/scale, batch_size=128, nb_epoch=12, callbacks=[lrate], shuffle=True) 
model.save("model_H.h5")

