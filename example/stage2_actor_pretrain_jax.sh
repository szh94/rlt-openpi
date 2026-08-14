#!/usr/bin/env bash
set -euo pipefail

# Standalone Stage 2 actor BC pre-training with the native JAX VLA.
# Saves <save_dir>/<run_name>/actor_pretrain.pt and exits.
#
# Usage:
#   bash example/stage2_actor_pretrain_jax.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../keyPara.sh
source "$SCRIPT_DIR/../keyPara.sh"

VLA_CONFIG_NAME=${VLA_CONFIG_NAME:-pi05_jax_full}
ACTION_DIM=${ACTION_DIM:-14}
CHUNK_LENGTH=${CHUNK_LENGTH:-10}
ACTOR_PRETRAIN_STEPS=${ACTOR_PRETRAIN_STEPS:-10000}
ACTOR_PRETRAIN_BATCH_SIZE=${ACTOR_PRETRAIN_BATCH_SIZE:-16}
ACTOR_PRETRAIN_SAVE_EVERY=${ACTOR_PRETRAIN_SAVE_EVERY:-1000}
NUM_WORKERS=${NUM_WORKERS:-0}
RUN_NAME=${RUN_NAME:-"actor_pretrain_$(date +%Y%m%d_%H%M%S)"}

export WANDB_MODE=${WANDB_MODE:-offline}

echo "========================================"
echo " Stage 2 Actor Pretrain [JAX VLA]"
echo "   VLA checkpoint  = $VLA_CHECKPOINT_JAX"
echo "   RLToken ckpt    = $STAGE1_RLT_CHECKPOINT"
echo "   Dataset         = $HF_LEROBOT_HOME"
echo "   Steps           = $ACTOR_PRETRAIN_STEPS"
echo "   Batch size      = $ACTOR_PRETRAIN_BATCH_SIZE"
echo "   Save every      = $ACTOR_PRETRAIN_SAVE_EVERY"
echo "   Save dir        = $STAGE2_AC_CHECKPOINT_DIR"
echo "   Run name        = $RUN_NAME"
echo "========================================"

python scripts/train_actor_pretrain_jax.py \
    --vla-config-name "$VLA_CONFIG_NAME" \
    --vla-checkpoint-dir "$VLA_CHECKPOINT_JAX" \
    --rl-token-checkpoint "$STAGE1_RLT_CHECKPOINT" \
    --repo-id "$HF_LEROBOT_HOME" \
    --save-dir "$STAGE2_AC_CHECKPOINT_DIR" \
    --run-name "$RUN_NAME" \
    --action-dim "$ACTION_DIM" \
    --chunk-length "$CHUNK_LENGTH" \
    --steps "$ACTOR_PRETRAIN_STEPS" \
    --batch-size "$ACTOR_PRETRAIN_BATCH_SIZE" \
    --save-every "$ACTOR_PRETRAIN_SAVE_EVERY" \
    --num-workers "$NUM_WORKERS"
