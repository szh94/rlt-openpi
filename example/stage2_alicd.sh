#!/usr/bin/env bash
set -euo pipefail

# Stage 2 online RL training for Alicia-D arm.
# Sources keyPara.sh for shared checkpoint paths.
#
# Usage:
#   bash example/stage2_alicd.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"
# shellcheck source=./keyPara.sh
source "$SCRIPT_DIR/keyPara.sh"

# --- Alicia-D hardware parameters ---
# Update these for your machine.
ALICD_PORT=""                        # e.g. "/dev/ttyACM0" (Linux) or "COM3" (Windows)
ALICD_CAM_IDS='{"exterior_image_1_left": 0, "wrist_image_left": 2}'
ALICD_SPEED_DEG_S=30.0
ALICD_CONTROL_HZ=15
DRY_RUN=false                       # true=打印action不驱动机器人, false=真实驱动
# 关节安全过滤: {} 表示不过滤, 指定关节索引和角度值(°)来锁定特定关节
# 示例: 锁定全部6个关节 (角度制, 自动转弧度)
#   JOINT_OVERRIDE='{"0": 0, "1": 110, "2": -15, "3": 0, "4": -20, "5": 0}'
# 示例: 仅锁定关节0和2
#   JOINT_OVERRIDE='{"0": 0, "2": -25}'
JOINT_OVERRIDE='{"0": 0, "1": 110, "2": -15, "3": 0, "4": -20, "5": 0}'
PRINT_ACTIONS=true                  # true=打印action数值, false=不打印 (独立于dry-run)
ALICD_IMAGE_SIZE=224
LIVE_IMAGE_DIR="/home/shenzh/Robot/rlt-openpi/live_image"

TASK_PROMPT="pick up the cup"

export WANDB_MODE=offline
# export WANDB_MODE=disabled

echo "========================================"
echo " Stage 2 Online RL (Alicia-D)"
echo "   VLA checkpoint  = $VLA_CHECKPOINT"
echo "   RLToken ckpt    = $STAGE1_RLT_CHECKPOINT"
echo "   Port            = ${ALICD_PORT:-auto}"
echo "   Cameras         = $ALICD_CAM_IDS"
echo "   Dry run         = $DRY_RUN"
echo "   Print actions   = $PRINT_ACTIONS"
echo "   Joint override  = $JOINT_OVERRIDE"
echo "========================================"

python scripts/train_online_rl.py \
    --env-factory rlt_openpi.envs.alicd.env_factory.make_alicd_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir "$VLA_CHECKPOINT" \
    --rl-token-checkpoint "$STAGE1_RLT_CHECKPOINT" \
    --save-dir "$STAGE2_AC_CHECKPOINT_DIR" \
    --task-prompt "$TASK_PROMPT" \
    --action-dim 10 \
    --chunk-length 10 \
    --warmup-steps 150 \
    --max-episode-chunks 150 \
    --env-kwargs "{\"port\": \"${ALICD_PORT}\", \"camera_ids\": ${ALICD_CAM_IDS}, \"control_hz\": ${ALICD_CONTROL_HZ}, \"speed_deg_s\": ${ALICD_SPEED_DEG_S}, \"image_size\": [${ALICD_IMAGE_SIZE}, ${ALICD_IMAGE_SIZE}], \"live_image_dir\": \"${LIVE_IMAGE_DIR}\", \"print_actions\": ${PRINT_ACTIONS}, \"joint_override\": ${JOINT_OVERRIDE}}" \
    --dry-run $DRY_RUN \
    $(
    # === 默认参数，必要时取消注释修改 ===
    # --intervention-factory rlt_openpi.envs.alicd.intervention.make_alicd_intervention \
    # --max-env-steps 100000              # 总环境交互步数上限，包含warmup步数
    # --save-every 50                     # 每 N 个 episode 保存一次 checkpoint, 不计数warmup阶段
    # --utd-ratio 5                       # 每 episode 梯度更新次数 (G)
    # --batch-size 256                    # 训练 minibatch 大小
    # --buffer-capacity 100000            # ReplayBuffer 最大容量
    # --embedding-dim 2048                # RL token 嵌入维度 (需与 Stage1 一致)
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
    # --dry-run                           # 空跑模式: 打印action数值, 不驱动机械臂
    # --resume-checkpoint ""              # 中断恢复: Stage2 checkpoint 路径
    # --warmup-buffer ""                  # 跳过 warmup: 预填充 buffer 路径
    )
