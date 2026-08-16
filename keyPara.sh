#!/usr/bin/env bash
# Shared parameters for the training and evaluation scripts under example/.
# 运行 example 脚本前，把下面的路径改成你自己的。

# --- VLA checkpoints ---

# torch
# VLA_CHECKPOINT_DIR="/home/zhike/model/openpi_pytorch/30000"
# jax data 1414 groups
# VLA_CHECKPOINT_DIR_JAX="/home/zhike/model/openpi-jax/full/1414-60000"
# export HF_LEROBOT_HOME="/home/zhike/data/openpi/train/1414"
# jax data 160 groups
VLA_CHECKPOINT_DIR_JAX="/home/zhike/model/openpi-jax/full/160-3w/models (1)/models/pretrained_model"

export S1_HF_LEROBOT_HOME="/home/zhike/data/openpi/150/data_allforce_all"

export HF_LEROBOT_HOME="/home/zhike/data/openpi/150/data_allforce_all"

STAGE1_RLT_CHECKPOINT_DIR="checkpoints/stage1_rlt_encoder"
# Jax 1414
# STAGE1_RLT_CHECKPOINT="${STAGE1_RLT_CHECKPOINT_DIR}/run_20260720_190830/rl_token_step10000.pt"
# Jax 160
STAGE1_RLT_CHECKPOINT="${STAGE1_RLT_CHECKPOINT_DIR}/run_20260729_180948/rl_token_step20000.pt"

STAGE2_AC_CHECKPOINT_DIR="checkpoints/stage2_ac_online"
STAGE2_AC_CHECKPOINT="${STAGE2_AC_CHECKPOINT_DIR}/run_xxxx_xxxx/online_rl_epxxx.pt"

# --- Derived: these usually don't need to be changed ---
# torch
# VLA_CHECKPOINT="${VLA_CHECKPOINT_DIR}/model.safetensors"
# jax
VLA_CHECKPOINT_JAX="${VLA_CHECKPOINT_DIR_JAX}"

# 重置位姿 (可选)
ALOHA_RESET_POSITION_LEFT="[164.70703, -30.7578125, 55.32031, 36.982422, 92.15625, 72.361328]"
ALOHA_RESET_POSITION_RIGHT="[17.402344, 6.064453, -68.64258, -27.597656, -83.32031, -19.59961]"
