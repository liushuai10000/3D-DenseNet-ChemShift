"""
Testing script: example for nitrogen
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
from keras.models import load_model

data_dir = "preprocessing/"
test_y = np.load(data_dir + "test_N_y.npy")
m = np.load(data_dir + "train_N_y.npy").mean()
scale = 30
size = test_y.shape[0]
test_x = np.zeros((size*8, 16, 16, 16, 20), dtype=np.float16)
for i in range(8):
    s = str(i)
    test_x[size*i:size*(i+1)] = np.concatenate([np.load(data_dir + "test_N_x_2A_" + s + ".npy"), 
                                                np.load(data_dir + "test_N_x_4A_" + s + ".npy"), 
                                                np.load(data_dir + "test_N_x_3A_" + s + ".npy"),
                                                np.load(data_dir + "test_N_x_5A_" + s + ".npy"),
                                                np.load(data_dir + "test_N_x_7A_" + s + ".npy")], axis=-1)
model = load_model("trained_models/nitrogen_model.h5")
pred = model.predict(test_x, batch_size=128)
np.save("predicted_value", pred)
pred = np.mean(pred.reshape((8, -1)), axis=0)
rms = np.sqrt(np.mean((pred*scale-test_y+m) ** 2))
print(rms)
# 0.36-0.37 for H
# 3.2-3.3 for C
# 9.4-10.6 for N
# 14.5-15.9 for O

