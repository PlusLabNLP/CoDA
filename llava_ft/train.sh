#!/bin/bash

HOME_PATH={PATH_TO_PIPELINE_DIR}

DATA_PATH=$1
NAME=$2
EPOCH=$3
BATCH_SIZE=$4
PORT=$5
MASTER_PORT=$6
IMAGE_FOLDER=$7
BASE_MODEL=$8
# BASE_MODEL=liuhaotian/llava-v1.6-vicuna-13b

echo "DATA_PATH: $DATA_PATH"
echo "NAME: $NAME"
echo "EPOCH: $EPOCH"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "PORT: $PORT"
echo "MASTER_PORT: $MASTER_PORT"
echo ""

export PYTHONPATH=$PYTHONPATH:$HOME_PATH/llava
deepspeed --master_port $MASTER_PORT --include localhost:$PORT llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed $HOME_PATH/llava/scripts/zero3.json \
    --model_name_or_path $BASE_MODEL \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $NAME \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True