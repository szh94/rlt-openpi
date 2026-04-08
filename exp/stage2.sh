python scripts/train_online_rl.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --intervention-factory rlt_openpi.envs.franka.intervention.make_vr_intervention \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir /home/alin/workspace/yknah/rlt-openpi/checkpoints/pi05_droid_pytorch/model.safetensors \
    --rl-token-checkpoint /home/alin/workspace/yknah/rlt-openpi/checkpoints/rl_token/rl_token_step5000.pt \
    --task-prompt "stack the three blocks on the tray" \
    --warmup-steps 20 \
    --save-dir checkpoints/online_rl