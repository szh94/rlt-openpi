#!/bin/bash
# Evaluate the VLA-only baseline (no RL token, no actor).
# Use this to measure the pretrained/fine-tuned VLA's performance
# before Stage 2 RL training, as a baseline comparison.

uv run python scripts/rollout_vla.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir checkpoints/pi05_droid_pytorch/model.safetensors \
    --task-prompt "stack the three blocks on the tray" \
    --num-episodes 50 \
    --save-dir results/vla_baseline
