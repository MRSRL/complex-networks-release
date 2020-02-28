WORK_DIR=~/Workspace/complex-networks-release
DATASET_DIR=/home_local/ekcole/knee_data
MASKS_PATH=/home_local/ekcole/knee_masks

TYPE=complex
STEPS=4
FEAT=256

LOG_DIR="/home_local/ekcole/f"$FEAT"_g4"
python3 $WORK_DIR/train_loop.py \
    --train_dir $TYPE \
    --mask_path $MASKS_PATH \
    --dataset_dir $DATASET_DIR \
    --log_root $LOG_DIR \
    --shape_z 256 --shape_y 320 \
    --num_channels 8 \
    --batch_size 2 \
    --device 0 \
    --max_steps 50000 \
    --feat_map $FEAT \
    --num_grad_steps 4 \
    --activation cardioid \
    --conv $TYPE

#testing
# python3 $WORK_DIR/test_loop.py --train_dir $TYPE \
#     --shape_z 256 --shape_y 320 \
#     --batch_size 1 \
#     --feat_map $FEAT \
#     --num_grad_steps 4 \
#     --mask_path $MASKS_PATH \
#     --dataset_dir $DATASET_DIR \
#     --device 0 \

#     --mode train_validate \
#     --max_steps 50000 \
#     --log_root $LOG_DIR \
#     --num_channels 8 \
#     --activation relu \
#     --gpu single \
#     --conv $TYPE
