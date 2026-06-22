#!/bin/bash
# Evaluate the Stage 1 fine-tuned VLA (no RL token head, no actor).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./keyPara.sh
source "$SCRIPT_DIR/keyPara.sh"

python scripts/evaluate.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir "$VLA_CHECKPOINT" \
    --stage1-checkpoint "$STAGE1_RLT_CHECKPOINT" \
    --task-prompt "stack the three blocks on the tray" \
    --num-episodes 10 \
    --save-dir results/vla_only
