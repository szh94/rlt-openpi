#!/usr/bin/env bash
set -euo pipefail

# Stage 2 online RL training — switch robot platform via first argument.
#
# Usage:
#   bash example/train/stage2.sh aloha
#   bash example/train/stage2.sh alicd
#
# Supported platforms:
#   aloha   — ALOHA dual-arm
#   alicd   — Alicia-D single arm

# =========================================================================
# 0. Platform selection
# =========================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../keyPara.sh
source "$SCRIPT_DIR/../keyPara.sh"

PLATFORM="${1:-aloha}"
# =========================================================================
# 1. Per-platform parameter tables
# =========================================================================

TASK_PROMPT="stack the three blocks on the tray"
export WANDB_MODE="offline"
LIVE_IMAGE_DIR="$SCRIPT_DIR/../../live_image"

case "$PLATFORM" in
# -------------------------------------------------------------------
# ALOHA — 14-DoF dual-arm, multiple cameras
# -------------------------------------------------------------------
aloha)
    ENV_FACTORY="rlt_openpi.envs.aloha.env_factory.make_aloha_env"
    DATA_TRANSFORMS_FN="rlt_openpi.policies.aloha.config.aloha_data_transforms"
    ACTION_DIM=14

    ENV_KWARGS_JSON="{\"control_hz\": 15, \
\"image_size\": [224, 224], \
\"camera_names\": [\"cam_high\", \"cam_left_wrist\", \"cam_right_wrist\"], \
\"print_actions\": true, \
\"joint_override\": {}, \
\"adapt_to_pi\": true}"
    ;;

# -------------------------------------------------------------------
# Alicia-D — 10-DoF single arm, serial port control
# -------------------------------------------------------------------
alicd)
    source "$SCRIPT_DIR/../../.venv/bin/activate"
    ENV_FACTORY="rlt_openpi.envs.alicd.env_factory.make_alicd_env"
    DATA_TRANSFORMS_FN="rlt_openpi.policies.alicd.config.alicd_data_transforms"
    ACTION_DIM=10

    ENV_KWARGS_JSON="{\"port\": \"\", \
\"camera_ids\": {\"exterior_image_1_left\": 0, \"wrist_image_left\": 2}, \
\"control_hz\": 15, \
\"speed_deg_s\": 30.0, \
\"image_size\": [224, 224], \
\"print_actions\": true, \
\"joint_override\": {\"0\": 0, \"1\": 110, \"2\": -15, \"3\": 0, \"4\": -20, \"5\": 0}}"
    ;;

*)
    echo "ERROR: Unknown platform '$PLATFORM'."
    echo "Supported platforms: aloha | alicd"
    exit 1
    ;;
esac

# =========================================================================
# 2. Log-to-file
# =========================================================================

LOG_DIR="$SCRIPT_DIR/../../log_action"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/stage2_${PLATFORM}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

# =========================================================================
# 3. Print summary
# =========================================================================

echo "========================================"
echo " Stage 2 Online RL"
echo "   Platform        = $PLATFORM"
echo "   VLA config      = pi05_droid_finetune"
echo "   VLA checkpoint  = $VLA_CHECKPOINT"
echo "   RLToken ckpt    = $STAGE1_RLT_CHECKPOINT"
echo "   Action dim      = $ACTION_DIM"
echo "   Data transforms = $DATA_TRANSFORMS_FN"
echo "   Task prompt     = $TASK_PROMPT"
echo "   Env factory     = $ENV_FACTORY"
echo "========================================"

# =========================================================================
# 4. Build and execute command
# =========================================================================

python scripts/train_online_rl.py \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir "$VLA_CHECKPOINT" \
    --rl-token-checkpoint "$STAGE1_RLT_CHECKPOINT" \
    --save-dir "$STAGE2_AC_CHECKPOINT_DIR" \
    --task-prompt "$TASK_PROMPT" \
    --action-dim "$ACTION_DIM" \
    --chunk-length 10 \
    --warmup-steps 250 \
    --max-episode-chunks 150 \
    --data-transforms-fn "$DATA_TRANSFORMS_FN" \
    --env-factory "$ENV_FACTORY" \
    --live-image-dir "$LIVE_IMAGE_DIR" \
    --env-kwargs "$ENV_KWARGS_JSON" \
    $(
    # === 以下参数使用默认值，必要时取消注释修改 ===
    # --save-every 50                     # 每 N 个 episode 保存一次 checkpoint
    # --max-env-steps 100000              # 总环境交互步数上限，包含warmup步数
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
    # --resume-checkpoint ""              # 中断恢复: Stage2 checkpoint 路径
    # --warmup-buffer ""                  # 跳过 warmup: 预填充 buffer 路径
    )
