#!/usr/bin/env bash
# Example template for keyPara.sh — copy this file to keyPara.sh and fill in your values.
#
# After the JAX migration, all checkpoints use Orbax directory format
# (each checkpoint is a directory containing a ``params/`` subdirectory).
#
# Expected variables (all must be set to non-empty before running any example script):
#
#   VLA_CHECKPOINT            — path to the Orbax checkpoint directory (with params/ inside)
#   STAGE1_RLT_CHECKPOINT     — path to the Stage 1 RL token Orbax checkpoint directory
#   STAGE2_AC_CHECKPOINT      — path to the Stage 2 actor-critic Orbax checkpoint directory
#   HF_LEROBOT_HOME           — HuggingFace LeRobot dataset root (exported)
#   DATA_TRANSFORMS_FN        — Python dotted path to the data transforms function
#
# Orbax checkpoint directory structure example:
#   VLA_CHECKPOINT_DIR/
#     params/          ← Orbax checkpointer saves/loads from here
#                       (containing the VLA model weights as JAX arrays)
#
# Stage1 checkpoint directory structure:
#   STAGE1_RLT_CHECKPOINT_DIR/run_20250101_120000/rl_token_step10000/
#     params/          ← model + optimizer state
#
# Stage2 checkpoint directory structure:
#   STAGE2_AC_CHECKPOINT_DIR/run_20250101_120000/online_rl_ep100/
#     params/          ← actor + critic + optimizer state

# --- user must fill in the paths below ---

STAGE1_RLT_CHECKPOINT_DIR="checkpoints/stage1_rlt_encoder"
STAGE1_RLT_CHECKPOINT="${STAGE1_RLT_CHECKPOINT_DIR}/run_xxxx_xxxx/rl_token_stepxxx"

STAGE2_AC_CHECKPOINT_DIR="checkpoints/stage2_ac_online"
STAGE2_AC_CHECKPOINT="${STAGE2_AC_CHECKPOINT_DIR}/run_xxxx_xxxx/online_rl_epxxx"

# VLA checkpoint — path to the Orbax checkpoint directory (containing params/).
# For openpi pretrained models, this is the directory that contains params/.
# Example: "/path/to/pi05_droid/checkpoints/pi05_droid_finetune/stepXXXX/"
VLA_CHECKPOINT_DIR=""

export HF_LEROBOT_HOME=""

# --- Derived: these usually don't need to be changed ---
VLA_CHECKPOINT="${VLA_CHECKPOINT_DIR}"
DATA_TRANSFORMS_FN="rlt_openpi.policies.aloha.config.aloha_data_transforms"
