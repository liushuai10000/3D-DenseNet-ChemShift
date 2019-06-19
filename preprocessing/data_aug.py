"""
Augment the data by rotating xyz coordinates between -pi/2 and pi/2 along x, y and z directions
"""

import numpy as np
import random
import math


def eulerAnglesToRotationMatrix(theta):
    
    """
    Rotate the xyz values based on the theta. Directly copied from:
    Credit:"https://www.learnopencv.com/rotation-matrix-to-euler-angles/"
    theta: numpy array with angles along x, y and z directions
    """
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def rot_aug_data(xyz_np_file, new_file_name, aug_fold=8):
    """
    Augment the xyz data by rotate between -pi/2 to pi/2.
    xyz_np_file: the file name of the numpy array with xyz coordinates
    new_file_name: the generated augmented file name
    aug_fold: fold of data augmentation
    """
    xyz = np.load(xyz_np_file)
    np.save(new_file_name + "_0", xyz) # copy itself in the first fold
    for j in range(1, aug_fold):
        xyz_new = []
        for i in range(xyz.shape[0]):
            theta = np.random.uniform(-3.14/2, 3.14/2, size=3)
            rot_mat = eulerAnglesToRotationMatrix(theta)
            xyz_new.append(np.dot(xyz[i], rot_mat))
        np.save(new_file_name + "_" + str(j), xyz_new)

        
        
# For loop version not tested. Original version is each atom type one time.
if __name__ == '__main__':
    for atom_type in ["H", "C", "N", "O"]:
        rot_aug_data("train_" + atom_type + "_xyz.npy", "train_" + atom_type + "_aug_xyz")
        rot_aug_data("test_" + atom_type + "_xyz.npy", "test_" + atom_type + "_aug_xyz")
