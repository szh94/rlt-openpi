#!/bin/bash
# Evaluate the Stage 1 fine-tuned VLA (no RL token head, no actor).

python scripts/evaluate.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir checkpoints/pi05_droid_pytorch/model.safetensors \
    --stage1-checkpoint checkpoints/stage1_rlt_encoder/rl_token_step3000.pt \
    --task-prompt "stack the three blocks on the tray" \
    --num-episodes 10 \
    --save-dir results/vla_only
