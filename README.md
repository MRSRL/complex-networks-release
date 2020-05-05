Implementation related to the paper "Complex-Valued Convolutional Neural Networks for MRI Reconstruction" by Elizabeth K. Cole et. al: 

# MRI Reconstruction - Unrolled Architecture
Image Reconstruction using an Unrolled DL Architecture including Complex-Valued Convolution and Activation Functions
* 2018 Elizabeth Cole, Stanford University (ekcole@stanford.edu)
* 2018 Joseph Y. Cheng, Stanford University
* 2018 Feiyu Chen, Stanford University
* 2018 Chris Sandino, Stanford University

## Complex-Valued Utilities
If you are only interested in the complex-valued utilities, they are located in complex_utils.py. This includes complex-valued convolution, complex-valued transposed convolution, and the CReLU, modReLU, zReLU, and cardioid activation functions.

## Setup
Make sure the python requirements are installed

    pip3 install -r requirements.txt

The setup assumes that the latest Berkeley Advanced Reconstruction Toolbox is installed [1]. The scripts have all been tested with v0.4.01.

## Data preparation
We will first download data, generate sampling masks, and generate TFRecords for training. The datasets downloaded are fully sampled volumetric knee scans from mridata [2]. The setup script uses the BART binary. In a new folder, run the follwing script:

    python3 setup_mri.py -v

## Training
The training can be ran using the following script in the same folder as the prepared data.

    TYPE=complex
    ITERATIONS=4
    FEAT=256
    ACTIVATION=cardioid
    LOG_DIR="f"$FEAT"_g"$ITERATIONS
    python3 train_loop.py \
        --train_dir $TYPE"_"$ACTIVATION \
        --mask_path masks \
        --dataset_dir data \
        --log_root $LOG_DIR \
        --shape_z 256 --shape_y 320 \
        --num_channels 8 \
        --batch_size 2 \
        --device 0 \
        --max_steps 50000 \
        --feat_map $FEAT \
        --num_grad_steps $ITERATIONS \
        --activation $ACTIVATION \
        --conv $TYPE

TYPE denotes the type of convolution; options include "real" or "complex".

ITERATIONS denotes the number of iterations in the unrolled architecture.

FEAT denotes the number of feature maps in each convolution layer.

ACTIVATION denotes the activation function used after each convolution layer; options include "relu", "crelu", "zrelu", "modrelu", and "cardioid" [3].
If running real convolution, the activation must be relu.

LOG_DIR indicates the directory checkpoints and training logs are saved to. You can view a tensorboard summary for this training run by running:

    tensorboard --logdir=./

in this directory.

Various complex-valued utility functions are in complex_utils.py. This includes complex-valued convolution, complex-valued transposed convolution, and various complex-valeud activation functions such as CReLU, zReLU, modReLU, and cardioid.

## References
1. https://github.com/mrirecon/bart
2. http://mridata.org
3. https://arxiv.org/pdf/1705.09792.pdf 
