#!/usr/bin/env bash
set -euo pipefail

# Inference — load trained model and run rollout on real robot.
#
# Usage:
#   bash example/infer/infer.sh vla aloha     # VLA-only on ALOHA
#   bash example/infer/infer.sh full alicd    # Full pipeline on Alicia-D
#   bash example/infer/infer.sh full          # defaults: full aloha
#
# Modes:
#   vla   — VLA-only (Stage 1 fine-tuned VLA, no RL token / actor)
#   full  — Full pipeline (VLA + RL token + Stage 2 actor)

MODE="${1:-full}"
PLATFORM="${2:-aloha}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../keyPara.sh
source "$SCRIPT_DIR/../keyPara.sh"

TASK_PROMPT="stack the three blocks on the tray"

case "$PLATFORM" in
aloha)
    ENV_FACTORY="rlt_openpi.envs.aloha.env_factory.make_aloha_env"
    ACTION_DIM=14
    ;;
alicd)
    source "$SCRIPT_DIR/../../.venv/bin/activate"
    ENV_FACTORY="rlt_openpi.envs.alicd.env_factory.make_alicd_env"
    ACTION_DIM=10
    ;;
*)
    echo "ERROR: Unknown platform '$PLATFORM'."
    echo "Supported platforms: aloha | alicd"
    exit 1
    ;;
esac

echo "========================================"
echo " Inference ($MODE mode, $PLATFORM)"
echo "   VLA checkpoint  = $VLA_CHECKPOINT"
if [[ "$MODE" == "full" ]]; then
    echo "   RLToken ckpt    = $STAGE1_RLT_CHECKPOINT"
    echo "   Stage2 ckpt     = $STAGE2_AC_CHECKPOINT"
fi
echo "   Task prompt     = $TASK_PROMPT"
echo "========================================"

MODE_ARGS=()
if [[ "$MODE" == "full" ]]; then
    MODE_ARGS+=(--checkpoint "$STAGE2_AC_CHECKPOINT")
    MODE_ARGS+=(--rl-token-checkpoint "$STAGE1_RLT_CHECKPOINT")
    NUM_EPISODES=50
else
    MODE_ARGS+=(--stage1-checkpoint "$STAGE1_RLT_CHECKPOINT")
    NUM_EPISODES=10
fi

python scripts/inference.py \
    --env-factory "$ENV_FACTORY" \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir "$VLA_CHECKPOINT" \
    --task-prompt "$TASK_PROMPT" \
    --action-dim "$ACTION_DIM" \
    --chunk-length 10 \
    --num-episodes "$NUM_EPISODES" \
    "${MODE_ARGS[@]}" \
    --save-dir "results/${MODE}_${PLATFORM}"
