#!/bin/bash

# ---- paths: edit or export before submitting --------------------------------
# LLaVA-1.5 training data (LLaVA-Pretrain/, llava_v1_5_mix665k.json and its image folders)
DATA_DIR=${DATA_DIR:-/path/to/llava/playground/data}
# Where pretrain (mm_projector.bin) and LoRA checkpoints are written
CKPT_DIR=${CKPT_DIR:-/path/to/checkpoints}

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path $DATA_DIR/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder $DATA_DIR/LLaVA-Pretrain/images \
    --vision_tower facebook/dinov2-base \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $CKPT_DIR/llava-v1.5-7b-pretrain-dinov2b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb
