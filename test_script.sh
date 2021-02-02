WORK_DIR=~/Workspace/complex-networks-release   #repo directory
DATASET_DIR=/home_local/ekcole/knee_data        #where the knee dataset was downloaded to
MASKS_PATH=/home_local/ekcole/knee_masks        #where the masks were generated by BART

TYPE=complex        #real or complex convolution
ITERATIONS=4        #number of unrolled iterations
FEAT=128            #number of feature maps
ACTIVATION=relu  #activation function

#To test images, batch size has to be 1

LOG_DIR="f"$FEAT"_g"$ITERATIONS
# training
python3 $WORK_DIR/test_images.py \
    --train_dir $TYPE"_"$ACTIVATION \
    --mask_path $MASKS_PATH \
    --dataset_dir $DATASET_DIR \
    --log_root $LOG_DIR \
    --shape_z 256 --shape_y 320 \
    --num_channels 8 \
    --batch_size 1 \
    --device 0 \
    --max_steps 10000 \
    --feat_map $FEAT \
    --num_grad_steps $ITERATIONS \
    --activation $ACTIVATION \
    --conv $TYPE