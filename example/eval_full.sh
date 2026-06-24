#!/bin/bash
# Evaluate the full trained model (Stage 1 RL token + Stage 2 actor).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"
# shellcheck source=./keyPara.sh
source "$SCRIPT_DIR/keyPara.sh"

python scripts/evaluate.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir "$VLA_CHECKPOINT" \
    --rl-token-checkpoint "$STAGE1_RLT_CHECKPOINT" \
    --checkpoint "$STAGE2_AC_CHECKPOINT" \
    --task-prompt "stack the three blocks on the tray" \
    --num-episodes 50
