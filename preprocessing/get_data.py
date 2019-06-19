"""
Download .txt files from the data attached to original paper:
training dataset: 
https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-06972-x/MediaObjects/41467_2018_6972_MOESM3_ESM.txt
testing dataset:
https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-06972-x/MediaObjects/41467_2018_6972_MOESM4_ESM.txt

Credit:
Paruzzo F M, Hofstetter A, Musil F, et al. Chemical shifts in molecular solids by machine learning. Nature communications, 2018, 9(1): 4501.
"""

import requests
from copy import deepcopy
import os
import argparse

def txt_to_xyz(file_path, folder):
    
    """
    Turn .txt files (from original dataset) into .xyz file in folders
    input: file_path, the file name of the .txt file, e.g., "train_data.txt", or download from original database
           folder, the folder/destination of .xyz files, e.g. "train_xyz_files". will create if not exist
    """
    
    text_file = open(file_path, "r")
    lines = text_file.readlines()
    text_file.close()
    index = -1  # start with a dummy index
    xyz_file = [] # the xyz file
    if not os.path.exists(folder):
        os.makedirs(folder)
    for line in lines:
        try:
            # a messy version to parse the txt file to xyz files
            # try if the line can be casted as an int, a trick to parse the .txt file into .xyz files
            # if yes, it starts with a new xyz file. 
            # otherwise, just append the current line
            x = int(line)
            if index != -1:
                with open("%s/%d.xyz" % (folder, index), "w+") as f:
                    f.writelines(xyz_file)
            xyz_file = [line]
            index += 1
        except Exception as e:
            xyz_file.append(line)
    # write the last xyz file
    with open("%s/%d.xyz" % (folder, index), "w+") as f:
        f.writelines(xyz_file)

        
def download_data(url, file_name):
    
    """
    Download the dataset from the source. Please ignore this function if you already downloaded the .txt file
    """
    
    r = requests.get(url)
    with open(file_name, "wb") as f:
        f.write(r.content)
              

if __name__ == "__main__":
    training_url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-06972-x/MediaObjects/41467_2018_6972_MOESM3_ESM.txt"
    testing_url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-06972-x/MediaObjects/41467_2018_6972_MOESM4_ESM.txt"
    download_data(training_url, "training_data.txt")
    download_data(testing_url, "testing_data.txt")
    txt_to_xyz("training_data.txt", "training")
    txt_to_xyz("testing_data.txt", "testing")
 
    
    