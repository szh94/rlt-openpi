#!/usr/bin/env bash
set -euo pipefail

# Stage 2 online RL training for ALOHA dual-arm robot with JAX VLA model.
# Uses JaxVLAWrapper to load the native JAX Pi0 model from an Orbax checkpoint.
#
# Key difference from stage2_aloha.sh: VLA weight restoration from the
# Stage 1 checkpoint is skipped because JAX NNX models do not use
# load_state_dict.
#
# Usage:
#   bash example/stage2_aloha_jax.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# === 自动保存终端输出到带时间戳的日志文件 ===
# LOG_DIR="$SCRIPT_DIR/../log_action"
# mkdir -p "$LOG_DIR"
# LOG_FILE="$LOG_DIR/stage2_aloha_jax_$(date +%Y%m%d_%H%M%S).log"
# exec > >(tee -a "$LOG_FILE") 2>&1
# echo "日志文件: $LOG_FILE"

# source "$SCRIPT_DIR/../.venv/bin/activate"
# shellcheck source=../keyPara.sh
source "$SCRIPT_DIR/../keyPara.sh"

# --- ALOHA hardware parameters ---
# Update these for your machine.
ALOHA_CONTROL_HZ=15
ALOHA_IMAGE_SIZE=224
ALOHA_CHUNK_LENGTH=10
ALOHA_MAX_EPISODE_CHUNKS=150
DRY_RUN=true                        # true=打印action不驱动机器人, false=真实驱动

# 相机列表: 3个相机 (不再使用 cam_low)
# 默认: cam_high, cam_left_wrist, cam_right_wrist
ALOHA_CAMERAS='["cam_high", "cam_left_wrist", "cam_right_wrist"]'

# 关节安全过滤: {} 表示不过滤, 指定关节索引和角度值(°)来锁定特定关节
# ALOHA 14-DoF: 索引 0-5=左臂关节, 6=左夹爪, 7-12=右臂关节, 13=右夹爪
# 示例: 锁住左臂关节0和右臂关节7
#   JOINT_OVERRIDE='{"0": 0, "7": 90}'
JOINT_OVERRIDE='{}'
PRINT_ACTIONS=true                  # true=打印action数值, false=不打印 (独立于dry-run)
LIVE_IMAGE_DIR="$SCRIPT_DIR/../live_image"

# Run example/stage2_actor_pretrain_jax.sh first, then point this variable at
# its <save_dir>/<run_name>/actor_pretrain.pt output.
ACTOR_PRETRAIN_CHECKPOINT=${ACTOR_PRETRAIN_CHECKPOINT:-""}
if [[ -z "$ACTOR_PRETRAIN_CHECKPOINT" ]]; then
    echo "ERROR: ACTOR_PRETRAIN_CHECKPOINT is required"
    exit 1
fi
if [[ ! -f "$ACTOR_PRETRAIN_CHECKPOINT" ]]; then
    echo "ERROR: Actor pretrain checkpoint not found: $ACTOR_PRETRAIN_CHECKPOINT"
    exit 1
fi

# 黑盒 obs 来源: robot=真实机械臂硬件(默认) | mock=随机假obs(测试) | dataset=从数据集加载
OBS_SOURCE=${OBS_SOURCE:-mock}
# OBS_SOURCE=${OBS_SOURCE:-robot}
# OBS_SOURCE=${OBS_SOURCE:-dataset}

TASK_PROMPT="place phone"

export WANDB_MODE=offline
# export WANDB_MODE=disabled

echo "========================================"
echo " Stage 2 Online RL (ALOHA Dual-Arm)"
echo " [JAX VLA]"
echo "   VLA checkpoint  = $VLA_CHECKPOINT_JAX"
echo "   RLToken ckpt    = $STAGE1_RLT_CHECKPOINT"
echo "   Control Hz      = $ALOHA_CONTROL_HZ"
echo "   Chunk length    = $ALOHA_CHUNK_LENGTH"
echo "   Max ep chunks   = $ALOHA_MAX_EPISODE_CHUNKS"
echo "   Cameras         = $ALOHA_CAMERAS"
echo "   Dry run         = $DRY_RUN"
echo "   Print actions   = $PRINT_ACTIONS"
echo "   Joint override  = $JOINT_OVERRIDE"
echo "   Obs source      = $OBS_SOURCE"
echo "   Actor ckpt      = $ACTOR_PRETRAIN_CHECKPOINT"
echo "========================================"

# Build env-kwargs JSON
ENV_KWARGS="{\"control_hz\": ${ALOHA_CONTROL_HZ}, \"image_size\": [${ALOHA_IMAGE_SIZE}, ${ALOHA_IMAGE_SIZE}], \"camera_names\": ${ALOHA_CAMERAS}, \"print_actions\": ${PRINT_ACTIONS}, \"live_image_dir\": \"${LIVE_IMAGE_DIR}\", \"joint_override\": ${JOINT_OVERRIDE}"

# Add reset_position if set
if [[ -n "${ALOHA_RESET_POSITION_LEFT:-}" ]]; then
    ENV_KWARGS="${ENV_KWARGS}, \"reset_position_left\": ${ALOHA_RESET_POSITION_LEFT}"
    ENV_KWARGS="${ENV_KWARGS}, \"reset_position_right\": ${ALOHA_RESET_POSITION_RIGHT}"
fi

ENV_KWARGS="${ENV_KWARGS}}"

python scripts/train_online_rl_jax.py \
    --env-factory rlt_openpi.envs.aloha.env_factory.make_aloha_env \
    --vla-config-name pi05_jax_full \
    --vla-checkpoint-dir "$VLA_CHECKPOINT_JAX" \
    --rl-token-checkpoint "$STAGE1_RLT_CHECKPOINT" \
    --save-dir "$STAGE2_AC_CHECKPOINT_DIR" \
    --task-prompt "$TASK_PROMPT" \
    --action-dim 14 \
    --chunk-length "$ALOHA_CHUNK_LENGTH" \
    --warmup-steps 50 \
    --max-episode-chunks "$ALOHA_MAX_EPISODE_CHUNKS" \
    --env-kwargs "$ENV_KWARGS" \
    --dry-run $DRY_RUN \
    --obs-source "$OBS_SOURCE" \
    --repo-id "$HF_LEROBOT_HOME" \
    --actor-pretrain-checkpoint "$ACTOR_PRETRAIN_CHECKPOINT" \
    --save-every 40
