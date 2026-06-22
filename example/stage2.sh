#!/usr/bin/env bash
set -euo pipefail

# Stage 2 online RL training — real robot + VR intervention.
#
# Usage:
#   bash example/stage2.sh

VLA_CHECKPOINT="/home/zhike/model/openpi_pytorch/30000/model.safetensors"
RLT_CHECKPOINT="checkpoints/stage1_rlt_encoder/run_20260615_160105/rl_token_step10000.pt"

echo "========================================"
echo " Stage 2 Online RL (Real Robot)"
echo "========================================"

python scripts/train_online_rl.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --intervention-factory rlt_openpi.envs.franka.intervention.make_vr_intervention \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir checkpoints/pi05_droid_pytorch/model.safetensors \
    --rl-token-checkpoint checkpoints/stage1_rlt_encoder/rl_token_step3000.pt \
    --task-prompt "stack the three blocks on the tray" \
    --warmup-steps 250 \
    --chunk-length 5 \
    --max-episode-chunks 150 \
    --save-dir checkpoints/stage2_ac_online \
    $(
    # === 以下参数使用默认值，必要时取消注释修改 ===
    # --max-env-steps 100000              # 总环境交互步数上限
    # --save-every 50                     # 每 N 个 episode 保存一次 checkpoint
    # --utd-ratio 5                       # 每 episode 梯度更新次数 (G)
    # --batch-size 256                    # 训练 minibatch 大小
    # --buffer-capacity 100000            # ReplayBuffer 最大容量
    # --embedding-dim 2048                # RL token 嵌入维度 (需与 Stage1 一致)
    # --action-dim 8                      # 单步动作维度
    # --mlp-hidden-dim 256                # Actor/Critic MLP 隐藏层宽度
    # --mlp-num-hidden-layers 2           # Actor/Critic MLP 层数
    # --gamma 0.99                        # 折扣因子
    # --tau 0.005                         # 目标网络 Polyak 软更新系数
    # --actor-lr 3e-4                     # Actor 学习率
    # --critic-lr 3e-4                    # Critic 学习率
    # --bc-regularizer-beta 0.5           # BC 正则化强度
    # --actor-noise-sigma 0.1             # Actor 探索噪声标准差
    # --ref-action-dropout 0.5            # VLA 参考动作 dropout 概率
    # --target-noise-sigma 0.2            # TD3 目标平滑噪声标准差
    # --target-noise-clip 0.5             # TD3 目标噪声裁剪范围
    # --critic-updates-per-actor 2        # Actor 延迟更新间隔
    # --resume-checkpoint ""              # 中断恢复: Stage2 checkpoint 路径
    # --warmup-buffer ""                  # 跳过 warmup: 预填充 buffer 路径
    )
