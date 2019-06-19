"""
Transform the .xyz files to .json files with k nearest atoms into destination folder
"""

from copy import deepcopy
import json
import os
import numpy as np
import multiprocessing
from functools import partial


def k_nearest_atom(folder, config_x, config_y, config_z, atoms, k=320, rep_num=3):
    
    """
    Query k nearest atoms from the xyz file and write into json files by extending the current unit cell.
    input: folder, the destination path of the json file
           config_x, config_y, config_z, the a, b, c values of the unit cell
           atoms, a list with each element as [atom_type, x, y, z, shift]
           k: number of nearest atoms. default is 320
           rep_num: repeat units along x, y and z direction by periodic boundary condition.
                    IMPORTANT: this is based on the assumption that the unit cell is at least 2.3A on each dimension.
                    This is true based on van der waals distance between atoms in this dataset. 
                    However, if this is applied to other dataset, this number possibly needs to be adjusted.
    """
    
    # The neighbor atoms are calculated by extending the unit cell over x, y and z directions
    new_atoms_x = deepcopy(atoms)
    for atom in atoms:
        for i in range(1, rep_num+1):
            new_atom = [atom[0], atom[1]-config_x[0]*i, atom[2]-config_x[1]*i, atom[3]-config_x[2]*i, atom[4]]
            new_atoms_x.append(new_atom)
            new_atom = [atom[0], atom[1]+config_x[0]*i, atom[2]+config_x[1]*i, atom[3]+config_x[2]*i, atom[4]]
            new_atoms_x.append(new_atom)
    new_atoms_y = deepcopy(new_atoms_x)
    for atom in new_atoms_x:
        for i in range(1, rep_num+1):
            new_atom = [atom[0], atom[1]-config_y[0]*i, atom[2]-config_y[1]*i, atom[3]-config_y[2]*i, atom[4]]
            new_atoms_y.append(new_atom)
            new_atom = [atom[0], atom[1]+config_y[0]*i, atom[2]+config_y[1]*i, atom[3]+config_y[2]*i, atom[4]]
            new_atoms_y.append(new_atom)
    new_atoms_z = deepcopy(new_atoms_y)
    for atom in new_atoms_y:
        for i in range(1, rep_num+1):
            new_atom = [atom[0], atom[1]-config_z[0]*i, atom[2]-config_z[1]*i, atom[3]-config_z[2]*i, atom[4]]
            new_atoms_z.append(new_atom)
            new_atom = [atom[0], atom[1]+config_z[0]*i, atom[2]+config_z[1]*i, atom[3]+config_z[2]*i, atom[4]]
            new_atoms_z.append(new_atom)
    for i in range(len(atoms)):
        k_nearest = []
        for j in range(len(new_atoms_z)):
            a = []
            a.append(new_atoms_z[j][0])
            a.append(new_atoms_z[j][1] - atoms[i][1])
            a.append(new_atoms_z[j][2] - atoms[i][2])
            a.append(new_atoms_z[j][3] - atoms[i][3])
            a.append(a[1]**2 + a[2]**2 + a[3]**2)
            if a[-1] != 0:
                k_nearest.append(a)
        k_nearest.sort(key=lambda x:x[-1])
        k_nearest = k_nearest[:k]
        k_nearest.insert(0, atoms[i])
        new_path = folder + "/" + str(i) + ".json"
        with open(new_path, 'w+') as f:
            json.dump(k_nearest, f)


            
def xyz_to_json_parallel(xyz_folder, json_folder, nums, k=320):
    
    """
    Turn xyz files to json files
    input: xyz_folder, the folder of xyz_folder
           json_folder, the json file folder
           nums, a list of file numbers
           k, number of nearest atoms, default 320
    """
    
    for i in nums:
        print(i)
        file_name = xyz_folder + "/" + str(i) + ".xyz"
        with open(file_name, 'r') as f:
            lines = f.readlines()
        configs = lines[1].split('"')[1]
        configs = configs.split(" ")
        config_x = [float(configs[0]), float(configs[1]), float(configs[2])]
        config_y = [float(configs[3]), float(configs[4]), float(configs[5])]
        config_z = [float(configs[6]), float(configs[7]), float(configs[8])]
        atoms = []
        for j in range(2, len(lines)):
            line = lines[j].split("\t")
            atom = []
            atom.append(line[0])
            atom.append(float(line[1]))
            atom.append(float(line[2]))
            atom.append(float(line[3]))
            atom.append(float(line[4]))
            atoms.append(atom)
        try:
            os.mkdir(json_folder + "/" + str(i))
        except:
            pass
        k_nearest_atom(json_folder + "/" + str(i), config_x, config_y, config_z, atoms, k)
    
    
    
def xyz_to_json(xyz_folder, json_folder, k=320, num_workers=32):
    
    """
    A parallel version to expedite this process with num_workers
    input: xyz_folder, the folder of xyz_folder
           json_folder, the json file folder
           nums, a list of file numbers
           k, number of nearest atoms, default 320
    """
    
    num_files = sum([len(f) for r, d, f in os.walk(xyz_folder)])
    print(num_files)
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)
    num_per_worker = num_files // num_workers + 1
    nums = [list(range(i * num_per_worker, min((i+1) * num_per_worker, num_files))) for i in range(num_workers)]
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(partial(xyz_to_json_parallel, xyz_folder, json_folder), nums)
        

if __name__ == "__main__":
    xyz_to_json("training", "train_json")
    xyz_to_json("testing", "test_json")