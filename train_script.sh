WORK_DIR=~/Workspace/complex-networks-release
DATASET_DIR=/home_local/ekcole/knee_data
MASKS_PATH=/home_local/ekcole/knee_masks

TYPE=real
STEPS=1
FEAT=10
ACTIVATION=relu

LOG_DIR="/home_local/ekcole/f"$FEAT"_g4"
# LOG_DIR="f"$FEAT"_g"$STEPS
python3 $WORK_DIR/train_loop.py \
    --train_dir $TYPE"_"$ACTIVATION \
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
    --activation $ACTIVATION \
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
