#!/usr/bin/env bash
set -euo pipefail

# ALOHA Stage 2 online RL with a frozen JAX VLA and RL token.
# Usage: bash example/jax_s2_onlinerl.sh

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

# 黑盒 obs 来源: robot=真实机械臂硬件 | mock=随机假obs(默认) | dataset=从数据集加载
OBS_SOURCE=${OBS_SOURCE:-mock}

# --- 所有 OBS_SOURCE 共用 ---
ALOHA_CONTROL_HZ=15
ALOHA_IMAGE_SIZE=224
ALOHA_CHUNK_LENGTH=10
ALOHA_MAX_EPISODE_CHUNKS=150
ALOHA_CAMERAS='["cam_high", "cam_left_wrist", "cam_right_wrist"]'
TASK_PROMPT="place phone"
LOG_EVERY=${LOG_EVERY:-100}

# --- OBS_SOURCE=robot 专用 ---
PRINT_ACTIONS=true                  # true=打印 action, false=不打印
DRY_RUN=true                        # true=不驱动机器人, false=真实驱动
LIVE_IMAGE_DIR="$SCRIPT_DIR/../live_image"
# {} 表示不过滤；指定关节索引和角度值(°)可锁定关节。
# ALOHA 14-DoF: 索引 0-5=左臂关节, 6=左夹爪, 7-12=右臂关节, 13=右夹爪
# 示例: JOINT_OVERRIDE='{"0": 0, "7": 90}'
JOINT_OVERRIDE='{}'

# --- Actor 初始化（所有 OBS_SOURCE 共用）---
# Run example/stage2_actor_pretrain_jax.sh first, then point this variable at

export WANDB_MODE=offline
# export WANDB_MODE=disabled
export RLT_METRICS_LIVE="${RLT_METRICS_LIVE:-0}"

# 根据 OBS_SOURCE 只传递当前模式需要的环境参数。
OBS_SOURCE_ARGS=(--obs-source "$OBS_SOURCE")
case "$OBS_SOURCE" in
    robot)
        ENV_KWARGS="{\"control_hz\": ${ALOHA_CONTROL_HZ}, \"image_size\": [${ALOHA_IMAGE_SIZE}, ${ALOHA_IMAGE_SIZE}], \"camera_names\": ${ALOHA_CAMERAS}, \"print_actions\": ${PRINT_ACTIONS}, \"live_image_dir\": \"${LIVE_IMAGE_DIR}\", \"joint_override\": ${JOINT_OVERRIDE}"
        if [[ -n "${ALOHA_RESET_POSITION_LEFT:-}" ]]; then
            ENV_KWARGS="${ENV_KWARGS}, \"reset_position_left\": ${ALOHA_RESET_POSITION_LEFT}"
            ENV_KWARGS="${ENV_KWARGS}, \"reset_position_right\": ${ALOHA_RESET_POSITION_RIGHT}"
        fi
        ENV_KWARGS="${ENV_KWARGS}}"
        OBS_SOURCE_ARGS+=(--dry-run "$DRY_RUN")
        ;;
    mock)
        ENV_KWARGS="{\"control_hz\": ${ALOHA_CONTROL_HZ}, \"image_size\": [${ALOHA_IMAGE_SIZE}, ${ALOHA_IMAGE_SIZE}], \"camera_names\": ${ALOHA_CAMERAS}}"
        ;;
    dataset)
        ENV_KWARGS="{\"control_hz\": ${ALOHA_CONTROL_HZ}, \"image_size\": [${ALOHA_IMAGE_SIZE}, ${ALOHA_IMAGE_SIZE}], \"camera_names\": ${ALOHA_CAMERAS}}"
        OBS_SOURCE_ARGS+=(--repo-id "$HF_LEROBOT_HOME")
        ;;
    *)
        echo "ERROR: unsupported OBS_SOURCE=$OBS_SOURCE (expected robot, mock, or dataset)"
        exit 1
        ;;
esac
OBS_SOURCE_ARGS+=(--env-kwargs "$ENV_KWARGS")

# 参数汇总紧邻启动命令，且只打印当前 OBS_SOURCE 相关配置。
echo "========================================"
echo " Stage 2 Online RL"
echo "   Obs source      = $OBS_SOURCE"
echo "   VLA checkpoint  = $VLA_CHECKPOINT_JAX"
echo "   RLToken ckpt    = $STAGE1_RLT_CP"
echo "   Actor ckpt      = $STAGE2_A_PRET_CP"
echo "   Task prompt     = $TASK_PROMPT"
echo "   Control Hz      = $ALOHA_CONTROL_HZ"
echo "   Image size      = ${ALOHA_IMAGE_SIZE}x${ALOHA_IMAGE_SIZE}"
echo "   Chunk length    = $ALOHA_CHUNK_LENGTH"
echo "   Max ep chunks   = $ALOHA_MAX_EPISODE_CHUNKS"
echo "   Cameras         = $ALOHA_CAMERAS"
if [[ "$OBS_SOURCE" == "robot" ]]; then
    echo "   Dry run         = $DRY_RUN"
    echo "   Print actions   = $PRINT_ACTIONS"
    echo "   Joint override  = $JOINT_OVERRIDE"
    echo "   Live image dir  = $LIVE_IMAGE_DIR"
    echo "   Reset left      = ${ALOHA_RESET_POSITION_LEFT:-not set}"
    echo "   Reset right     = ${ALOHA_RESET_POSITION_RIGHT:-not set}"
elif [[ "$OBS_SOURCE" == "dataset" ]]; then
    echo "   Dataset repo    = $HF_LEROBOT_HOME"
fi
echo "   Save every      = 40 episodes"
echo "   Log every       = $LOG_EVERY"
echo "   Live metrics    = $RLT_METRICS_LIVE (1=enabled)"
echo "========================================"
python scripts/train_jax_s2_onlinerl.py \
    --env-factory rlt_openpi.envs.aloha.env_factory.make_aloha_env \
    --vla-config-name pi05_jax_full \
    --vla-checkpoint-dir "$VLA_CHECKPOINT_JAX" \
    --rl-token-checkpoint "$STAGE1_RLT_CP" \
    --save-dir "$STAGE2_AC_CPD" \
    --task-prompt "$TASK_PROMPT" \
    --action-dim 14 \
    --chunk-length "$ALOHA_CHUNK_LENGTH" \
    --warmup-steps 5 \
    --max-episode-chunks "$ALOHA_MAX_EPISODE_CHUNKS" \
    "${OBS_SOURCE_ARGS[@]}" \
    --actor-pretrain-checkpoint "$STAGE2_A_PRET_CP" \
    --save-every 40 \
    --log-every "$LOG_EVERY"
