#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/keyPath.sh"

export WANDB_MODE=disabled

python scripts/train_rl_token.py \
    --train.vla-config-name pi05_droid_finetune \
    --train.vla-checkpoint-dir "$VLA_CHECKPOINT" \
    --train.vla-finetune-alpha 1.0 \
    --train.batch-size 32 \
    --train.num-train-steps 3000 \
    --repo-id pick_pen_image \
    --data-transforms-fn rlt_openpi.policies.franka.config.three_camera_droid
