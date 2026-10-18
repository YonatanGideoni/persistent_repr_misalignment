#!/bin/bash

# ---- paths: edit or export before submitting --------------------------------
# LLaVA-1.5 training data (LLaVA-Pretrain/, llava_v1_5_mix665k.json and its image folders)
DATA_DIR=${DATA_DIR:-/path/to/llava/playground/data}
# Where pretrain (mm_projector.bin) and LoRA checkpoints are written
CKPT_DIR=${CKPT_DIR:-/path/to/checkpoints}
# Held-out concept test set used for the in-training concept-detection eval
CONCEPT_TESTSET_DIR=${CONCEPT_TESTSET_DIR:-../concept_testset/img_concepts_outs}

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path $DATA_DIR/llava_v1_5_mix665k.json \
    --image_folder $DATA_DIR \
    --vision_tower google/vit-large-patch16-384 \
    --pretrain_mm_mlp_adapter $CKPT_DIR/llava-v1.5-7b-pretrain-nocsf-vit/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --conceptless \
    --sim_filter \
    --output_dir $CKPT_DIR/llava-v1.5-7b-lora-nocsf-vit \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_data_path $CONCEPT_TESTSET_DIR \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
