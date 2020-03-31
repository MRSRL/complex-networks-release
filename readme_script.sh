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