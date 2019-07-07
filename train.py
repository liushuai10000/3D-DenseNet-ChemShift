"""
The training script: an example for oxygen
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
from model.mr_densenet import mr_densenet, densenet_baseline_1, densenet_baseline_2
from model.regular_cnns import cnn_3d, res_net_3d
from model.utils import exp_decay
import tensorflow as tf
from keras import backend as K
from keras.callbacks import LearningRateScheduler


data_dir = "preprocessing/"
train_y = np.load(data_dir + "train_O_y.npy")
size = train_y.shape[0]
m = np.mean(train_y) # substract the mean during training
num_aug_fold = 1 # min 1 max 8
scale = 40 # scale the std of different atom type to encourage convergence
train_y = np.concatenate([train_y for _ in range(num_aug_fold)]) # because it's augmented
train_x = np.zeros((size*num_aug_fold, 16, 16, 16, 20), dtype=np.float16) # generate the tensor first

# fill the tensor with data
for i in range(num_aug_fold):
    s = str(i)
    train_x[size*i:size*(i+1)] = np.concatenate([np.load(data_dir + "train_O_x_2A_" + s + ".npy"),
                                                 np.load(data_dir + "train_O_x_4A_" + s + ".npy"),
                                                 np.load(data_dir + "train_O_x_3A_" + s + ".npy"),
                                                 np.load(data_dir + "train_O_x_5A_" + s + ".npy"),
                                                 np.load(data_dir + "train_O_x_7A_" + s + ".npy")], axis=-1)

lrate = LearningRateScheduler(exp_decay)
model = mr_densenet(20)
#model = densenet_baseline_1(20)
#model = densenet_baseline_2(20)
#model = cnn_3d(20)
#model = res_net_3d(20)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
# mean is substracted. for C, N and O, we also need to scale it with a factor
model.fit(train_x, (train_y-m)/scale, batch_size=128, nb_epoch=24, callbacks=[lrate], shuffle=True) 
model.save("model_O.h5")

