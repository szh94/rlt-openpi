#!/usr/bin/env bash
set -euo pipefail

# Stage 2 offline training — same as stage2.sh but with MockEnv.
# No robot, no VR intervention, fake observations → real VLA.
#
# Usage:
#   bash example/stage2_mock.sh

VLA_CKPT="${VLA_CKPT:-checkpoints/pi05_droid_pytorch/model.safetensors}"
RL_TOKEN_CKPT="${RL_TOKEN_CKPT:-checkpoints/rl_token/rl_token_step3000.pt}"

echo "========================================"
echo " Stage 2 Offline (MockEnv)"
echo "   VLA checkpoint  = $VLA_CKPT"
echo "   RLToken ckpt    = $RL_TOKEN_CKPT"
echo "========================================"

python scripts/train_online_rl_offline.py \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir "$VLA_CKPT" \
    --rl-token-checkpoint "$RL_TOKEN_CKPT" \
    --save-dir checkpoints/online_rl \
    --task-prompt "stack the three blocks on the tray" \
    --max-env-steps 31500 \
    --warmup-steps 150 \
    --chunk-length 10 \
    --max-episode-chunks 150 \
    --save-every 10 \
    $(
    # === 以下参数使用默认值，必要时取消注释修改 ===
    # --max-env-steps 100000              # 总环境交互步数上限，包含warmup步数
    # --save-every 50                     # 每 N 个 episode 保存一次 checkpoint, 不计数warmup阶段
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
