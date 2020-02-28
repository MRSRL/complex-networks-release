# MRI Reconstruction - Unrolled Architecture
Image Reconstruction using an Unrolled DL Architecture
* 2018 Joseph Y. Cheng, Stanford University (jycheng AT stanford DOT edu)
* 2018 Feiyu Chen, Stanford University
* 2018 Chris Sandino, Stanford University
* 2018 Elizabeth Cole, Stanford University

## Setup
Make sure the python requirements are installed

    pip install -r requirements.txt

The setup assumes that the latest Berkeley Advanced Reconstruction Toolbox is installed [1]. The scripts have all been tested with v0.4.01.

Make sure you install Keras 2.2.0, not any higher version.

## Data preparation
We will first download data, generate sampling masks, and generate TFRecords for training. The datasets downloaded are fully sampled volumetric knee scans from mridata [2]. The setup script uses the BART binary. In a new folder, run the follwing script:

    python3 setup_mri.py -v

## Training
The training can be ran using the following script in the same folder as the prepared data.

    python3 train_mri.py --train_dir RUN_NAME \
        --shape_z 256 --shape_y 320 \
        --mask_path masks \
        --dataset_dir data \
        --device 1


## References
1. https://github.com/mrirecon/bart
2. http://mridata.org