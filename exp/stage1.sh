#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors"
export HF_LEROBOT_HOME="/home/nyk7535/data"

python scripts/train_rl_token.py \
    --train.vla-config-name pi05_droid_finetune \
    --train.vla-checkpoint-dir "$CHECKPOINT_DIR" \
    --train.vla-finetune-alpha 1.0 \
    --train.batch-size 32 \
    --train.num-train-steps 3000 \
    --repo-id pick_pen_image \
    --data-transforms-fn rlt_openpi.policies.franka.config.three_camera_droid
