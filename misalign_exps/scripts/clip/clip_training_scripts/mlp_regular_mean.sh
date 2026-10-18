#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name="mlp_regular_mean"
#SBATCH --output=mlp_regular_mean.out
#SBATCH --error=mlp_regular_mean.err

set -a; [ -f ../.env ] && source ../.env; set +a  # credentials: copy ../.env.example to ../.env

# Submit from the misalign_exps/ directory: sbatch scripts/clip/clip_training_scripts/mlp_regular_mean.sh
# WANDB_API_KEY (or `wandb login`) and optionally HF_HOME are read from your environment.

# Edit these placeholders for your setup
OUT_DIR=/path/to/outputs
export COCO_DIR=/path/to/coco2014  # HF datasets cache dir for COCO 2014
export IMAGENET_DIR=/path/to/imagenet  # used for the zero-shot ImageNet eval

eval "$(conda shell.bash hook)"
conda activate misalign_exps

srun python -m small_scale_failures.clip_failure.train_clip \
    --out_dir $OUT_DIR/clip \
    \
    --adapter mlp \
    --early_stopping_patience 9999 \
    --eval_epoch_frac 0.1 \
    --vis_pooling mean \
    --lang_pooling mean \
    --lr 1e-4 \
    --wd 1e-2
