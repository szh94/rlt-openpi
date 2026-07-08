#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"

# --- Default parameters ---
J2_STEP_DEG=3
CONTROL_HZ=15
IMAGE_DIR="$SCRIPT_DIR/../live_image"
SAVE_LOG=false                     # set to true to save log file

echo "========================================"
echo " Alicia-D Joint Test"
echo "   J2 step     = ${J2_STEP_DEG} deg/step"
echo "   Control Hz  = $CONTROL_HZ"
echo "   Image dir   = $IMAGE_DIR"
echo "   Save log    = $SAVE_LOG"
echo "========================================"
echo ""

LOG_DIR_ARG=()
if [ "$SAVE_LOG" = true ]; then
    LOG_DIR_ARG=(--log-dir "$SCRIPT_DIR/../log_action")
fi

python "$SCRIPT_DIR/../scripts/test_alicd_joint.py" \
    --j2-step-deg "$J2_STEP_DEG" \
    --control-hz "$CONTROL_HZ" \
    --image-dir "$IMAGE_DIR" \
    --camera exterior_image_1_left 0 \
    --camera wrist_image_left 2 \
    "${LOG_DIR_ARG[@]}"
