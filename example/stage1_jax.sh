#!/usr/bin/env bash
set -euo pipefail

# Stage 1 RL Token training with JAX VLA model.
# Uses JaxVLAWrapper instead of VLAWrapper to load the native JAX Pi0
# model from an Orbax checkpoint.
#
# Joint training (--train.vla-finetune-alpha > 0) is NOT supported —
# a JAX NNX model cannot be optimised by a PyTorch optimizer.
#
# Usage:
#   bash example/stage1_jax.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"
# shellcheck source=./keyPara.sh
source "$SCRIPT_DIR/keyPara.sh"

export WANDB_MODE=offline
# export WANDB_MODE=disabled

echo "========================================"
echo " Stage 1 RL Token Training (JAX VLA)"
echo "   VLA checkpoint  = $VLA_CHECKPOINT"
echo "   Save dir        = $STAGE1_RLT_CHECKPOINT_DIR"
echo "   HF dataset      = $HF_LEROBOT_HOME"
echo "========================================"
echo ""

python scripts/train_rl_token_jax.py \
    --train.vla-config-name pi05_droid_finetune \
    --train.vla-checkpoint-dir "$VLA_CHECKPOINT" \
    --train.save-dir "$STAGE1_RLT_CHECKPOINT_DIR" \
    --train.vla-finetune-alpha 0.0 \
    --train.batch-size 8 \
    --train.num-train-steps 10000 \
    --train.save-every 10000 \
    --repo-id "$HF_LEROBOT_HOME" \
    --data-transforms-fn "$DATA_TRANSFORMS_FN" \
    $(
    # === 以下参数使用默认值，必要时取消注释修改 ===
    # --train.embedding-dim 2048              # RL token 嵌入维度
    # --train.encoder-layers 2                # Transformer encoder 层数
    # --train.encoder-heads 8                 # encoder 注意力头数
    # --train.decoder-layers 2                # Transformer decoder 层数
    # --train.decoder-heads 8                 # decoder 注意力头数
    # --train.learning-rate 1e-4              # 学习率
    # --train.weight-decay 1e-5               # AdamW weight decay
    # --train.warmup-steps 500                # LR 线性 warmup 步数
    # --train.max-grad-norm 1.0              # 全局梯度裁剪阈值
    # --train.resume-checkpoint ""            # 中断恢复: Stage1 checkpoint 路径
    # --train.run-name ""                     # 子目录名 (空则自动生成时间戳)
    # --train.log-every 1000                  # wandb 日志间隔 (steps)
    # --no-train.wandb-enabled                # 禁用 wandb 日志 (默认启用)
    # --train.wandb-project "rlt-openpi"      # wandb 项目名
    # --num-workers 4                         # DataLoader 进程数
    )

# 注意: JAX VLA 不支持联合训练 (vla_finetune_alpha > 0)。
# 如需联合训练，请使用 example/stage1.sh。
