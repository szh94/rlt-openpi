python scripts/rollout_vla.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir /home/alin/workspace/yknah/rlt-openpi/checkpoints/pi05_droid_pytorch/model.safetensors \
    --task-prompt "stack the three blocks on the tray" \
    --num-episodes 10