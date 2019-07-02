# A Multi-Resolution 3D-DenseNet for Chemical Shift Prediction in NMR Crystallography

Shuai Liu, Jie Li, Kochise C. Bennett, Brad Ganoe, Tim Stauch, Martin Head-Gordon, Alexander Hexemer, Daniela Ushizima and Teresa Head-Gordon


Dependences: requests, pytorch, tensorflow, keras

Currently we only have scripts to train and test the model.

We will upload the trained model and easy-to-use interface to predict the chemical shift given xyz file (under construction).


## Data Preprocessing
Under preprocessing directory
`cd preprocessing`
Please run the current script sequencially. We do not support the argparse yet (under construction).

### Step 1: Download the data
Download .txt files from the data attached to original paper:

training dataset: 
https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-06972-x/MediaObjects/41467_2018_6972_MOESM3_ESM.txt

testing dataset:
https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-06972-x/MediaObjects/41467_2018_6972_MOESM4_ESM.txt

Credit:
Paruzzo F M, Hofstetter A, Musil F, et al. Chemical shifts in molecular solids by machine learning. Nature communications, 2018, 9(1): 4501.

Then the data are transformed to .xyz files into different folders

`python get_data.py`

### Step 2: Transform the xyz files into json files
Transform the .xyz files to .json files with k nearest atoms into destination folder

`python xyz_to_json.py`

### Step 3: Transform json files into numpy files 
The xyz values, atom type and chemical shielding values are transfomed into into:

prefix_xyz.npy, prefix_points.npy and prefix_y.npy

`python json_to_numpy.py`

### Step 4: Data augmentation
Augment xyz files with 8 fold

`python data_aug.py`

### Step 5: Density generation
Generate the density given xyz and one-hot atom type vector numpy files

`python density_gen.py`


## Models
We provide the following models in model directory:
1) MR-3D-DenseNet
2) Two baseline DenseNets
3) Regular CNN and ResNet with same number of 3x3x3 filters

## Data Analysis Tools
Under construction. We plan to provide the tools to:
1) Plot the density of input, intermediate layers and weights
2) PCA for the intermediate layers to interpret the model

## Training and Testing script
We also provide the trainin gand testing script examples. 

Under the default setting, please run following command sequentially:

`python train.py`

`python test.py`

Important: these two only provide examples for hydrogen. For other atom types, the file names and scale need to be changed.

