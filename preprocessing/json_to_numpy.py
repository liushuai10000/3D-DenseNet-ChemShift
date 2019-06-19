"""
Turn json files into numpy files with xyz values, atom type and chemical shielding values into:
prefix_xyz.npy, prefix_points.npy and prefix_y.npy
"""
import json
import os
import numpy as np


def json_to_numpy(json_folder, prefix, atom_type='H', k=320):
    
    """
    Turn json files to numpy files
    input: json_folder, the json file folder
           prefix, the prefix string of the numpy files "$prefix$_xyz.npy"
           atom_type, the atom type of processing, 'H', 'C', 'O' or 'N'
           k, number of nearest atoms, default 320
    """
    xyz_n = []
    points_n = []
    y = []
    files = []
    num_folder = sum([len(d) for r, d, f in os.walk(json_folder)])
    for i in range(num_folder):
        folder_path = json_folder + "/" + str(i)
        num_files = sum([len(f) for r, d, f in os.walk(folder_path)])
        for j in range(num_files):
            files.append(folder_path + "/" + str(j) + ".json")
    for f in files:
        #print(f)
        with open(f, 'r') as ff:
            lines = json.load(ff)
        if lines[0][0] != atom_type:
            continue
        y.append(lines[0][-1])
        xyz = []
        points = []
        for l in range(1, k+1):
            xyz.append(lines[l][1:4])
            if lines[l][0] == "H":
                points.append([1,0,0,0])
            elif lines[l][0] == "C":
                points.append([0,1,0,0])
            elif lines[l][0] == "O":
                points.append([0,0,1,0])
            else:
                points.append([0,0,0,1])
        xyz_n.append(xyz)
        points_n.append(points)
    np.save(prefix + "_xyz", xyz_n)
    np.save(prefix + "_points", points_n)
    np.save(prefix + "_y", y)

    
    
# For loop version not tested. Original version is each atom type one time.
if __name__ == '__main__':
    for atom_type in ["H", "C", "N", "O"]:
        json_to_numpy("train_json", "train_" + atom_type + "_", atom_type)
        json_to_numpy("test_json", "test_" + atom_type + "_", atom_type)
