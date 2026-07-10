#!/usr/bin/env bash
# Example template for keyPara.sh — copy this file to keyPara.sh and fill in your values.
#
# Expected variables (all must be set to non-empty before running any example script):
#
#   STAGE1_RLT_CHECKPOINT_DIR — base directory for Stage 1 RL token checkpoints
#   STAGE1_RLT_CHECKPOINT     — path to the Stage 1 RL token .pt checkpoint file
#   STAGE2_AC_CHECKPOINT_DIR  — base directory for Stage 2 actor-critic checkpoints
#   STAGE2_AC_CHECKPOINT      — path to the Stage 2 actor-critic .pt checkpoint file
#   VLA_CHECKPOINT_DIR       — directory containing the VLA .safetensors model file
#   HF_LEROBOT_HOME           — HuggingFace LeRobot dataset root (exported)
#
# Derived (auto-computed, no need to edit):
#
#   VLA_CHECKPOINT   = ${VLA_CHECKPOINT_DIR}/model.safetensors
#   DATA_TRANSFORMS_FN = rlt_openpi.policies.aloha.config.aloha_data_transforms

# --- Expected variables: user must fill in the paths below ---

STAGE1_RLT_CHECKPOINT_DIR="checkpoints/stage1_rlt_encoder"
STAGE1_RLT_CHECKPOINT="${STAGE1_RLT_CHECKPOINT_DIR}/run_xxxx_xxxx/rl_token_stepxxx.pt"
STAGE2_AC_CHECKPOINT_DIR="checkpoints/stage2_ac_online"
STAGE2_AC_CHECKPOINT="${STAGE2_AC_CHECKPOINT_DIR}/run_xxxx_xxxx/online_rl_epxxx.pt"

VLA_CHECKPOINT_DIR=""
export HF_LEROBOT_HOME=""

# --- Derived: these usually don't need to be changed ---
VLA_CHECKPOINT="${VLA_CHECKPOINT_DIR}/model.safetensors"
DATA_TRANSFORMS_FN="rlt_openpi.policies.aloha.config.aloha_data_transforms"
