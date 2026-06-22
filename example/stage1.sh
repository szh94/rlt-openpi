#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./keyPara.sh
source "$SCRIPT_DIR/keyPara.sh"

export WANDB_MODE=disabled

echo "Training RL token model with VLA checkpoint: $VLA_CHECKPOINT"
python scripts/train_rl_token.py \
    --train.vla-config-name pi05_droid_finetune \
    --train.vla-checkpoint-dir "$VLA_CHECKPOINT" \
    --train.vla-finetune-alpha 0.0 \
    --train.batch-size 8 \
    --train.num-train-steps 10000 \
    --repo-id "$HF_LEROBOT_HOME" \
    --data-transforms-fn "$DATA_TRANSFORMS_FN"
