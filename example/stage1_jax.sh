#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LOG_DIR="$SCRIPT_DIR/../log_action"
# mkdir -p "$LOG_DIR"
# LOG_FILE="$LOG_DIR/stage2_aloha_jax_$(date +%Y%m%d_%H%M%S).log"
# exec > >(tee -a "$LOG_FILE") 2>&1
# echo "日志文件: $LOG_FILE"

# source "$SCRIPT_DIR/../.venv/bin/activate"
# shellcheck source=../keyPara.sh
source "$SCRIPT_DIR/../keyPara.sh"

export WANDB_MODE=offline
# export WANDB_MODE=disabled
NUM_WORKERS=${NUM_WORKERS:-2}

echo "========================================"
echo " Stage 1 RL Token Training (JAX VLA)"
echo "   VLA checkpoint  = $VLA_CHECKPOINT_JAX"
echo "   HF Dataset      = $S1_HF_LEROBOT_HOME"
echo "   Save dir        = $STAGE1_RLT_CHECKPOINT_DIR"
echo "   Num workers     = $NUM_WORKERS"
echo "========================================"
echo ""

python scripts/train_rl_token_jax.py \
    --train.vla-config-name pi05_jax_full \
    --train.vla-checkpoint-dir "$VLA_CHECKPOINT_JAX" \
    --train.save-dir "$STAGE1_RLT_CHECKPOINT_DIR" \
    --train.vla-finetune-alpha 0.0 \
    --train.batch-size 8 \
    --train.num-train-steps 20000 \
    --train.save-every 10000 \
    --repo-id "$S1_HF_LEROBOT_HOME" \
    --num-workers "$NUM_WORKERS"

