#!/usr/bin/env bash
# Sets up a conda environment for rlt-openpi.
#
# Usage:
#   bash setup_env.sh            # creates env named 'rlt'
#   bash setup_env.sh myenvname  # creates env with custom name
#
# By default only the core dependencies (Stage 1 training, evaluation) are
# installed. Pass --robot to also install the DROID teleop stack, Oculus
# reader, ZED bindings, and related fixups needed for Stage 2 on a real
# Franka rig. Set DROID_DIR to point to your local DROID clone:
#
#   DROID_DIR=/path/to/droid bash setup_env.sh --robot
#   DROID_DIR=/path/to/droid bash setup_env.sh myenvname --robot
#
# Set OPENPI_DIR to point to a local openpi clone to skip the GitHub fetch:
#
#   OPENPI_DIR=/path/to/openpi bash setup_env.sh
#
# After setup:
#   conda activate <env_name>
#   bash exp/stage1.sh   # or stage2.sh, eval_*.sh, etc.
set -euo pipefail

# ── Parse args ──────────────────────────────────────────────────────────
INSTALL_ROBOT=false
ENV_NAME="rlt"
for arg in "$@"; do
    case "$arg" in
        --robot) INSTALL_ROBOT=true ;;
        *)       ENV_NAME="$arg" ;;
    esac
done

OPENPI_REV="fdc03f5"
OPENPI_DIR="${OPENPI_DIR:-}"        # set to local path to skip git clone
DROID_DIR="${DROID_DIR:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "$OPENPI_DIR" ]; then
    if [ ! -d "$OPENPI_DIR" ]; then
        echo "ERROR: OPENPI_DIR=${OPENPI_DIR} does not exist."
        exit 1
    fi
    OPENPI_ABS_DIR="$(cd "$OPENPI_DIR" && pwd)"
    OPENPI_URL="openpi @ ${OPENPI_ABS_DIR}"
    echo "==> Using local openpi at ${OPENPI_ABS_DIR}"
else
    OPENPI_URL="openpi @ git+https://github.com/Physical-Intelligence/openpi@${OPENPI_REV}"
fi

# ── Core environment ───────────────────────────────────────────────────
echo "==> Creating conda env '${ENV_NAME}' with Python 3.11..."
conda create -n "${ENV_NAME}" python=3.11 -y

echo "==> Installing uv (fast resolver, required for openpi's deep dep graph)..."
conda run -n "${ENV_NAME}" pip install uv

if [ -n "${OPENPI_DIR}" ]; then
    echo "==> Installing openpi from local path..."
else
    echo "==> Installing openpi from GitHub (rev ${OPENPI_REV})..."
fi
conda run -n "${ENV_NAME}" uv pip install "${OPENPI_URL}"

echo "==> Installing rlt-openpi (with dev dependencies)..."
conda run -n "${ENV_NAME}" uv pip install -e "${SCRIPT_DIR}[dev]" \
    --overrides /dev/stdin <<EOF
${OPENPI_URL}
EOF

echo "==> Patching transformers with openpi's transformers_replace files..."
OPENPI_PKG_DIR=$(conda run -n "${ENV_NAME}" python -c \
    "import openpi, pathlib; print(pathlib.Path(openpi.__file__).parent)")
TRANSFORMERS_DIR=$(conda run -n "${ENV_NAME}" python -c \
    "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)")
cp -r "${OPENPI_PKG_DIR}/models_pytorch/transformers_replace/"* "${TRANSFORMERS_DIR}/"

# ── Robot-specific dependencies (optional) ─────────────────────────────
if [ "$INSTALL_ROBOT" = true ]; then
    if [ -z "$DROID_DIR" ]; then
        echo "ERROR: --robot requires DROID_DIR to be set."
        echo "  DROID_DIR=/path/to/droid bash setup_env.sh --robot"
        exit 1
    fi
    if [ ! -d "$DROID_DIR" ]; then
        echo "ERROR: DROID_DIR=${DROID_DIR} does not exist."
        exit 1
    fi

    echo ""
    echo "==> Installing robot dependencies (DROID, Oculus, ZED, opencv fixups)..."

    echo "==> Installing droid from ${DROID_DIR}..."
    conda run -n "${ENV_NAME}" uv pip install -e "${DROID_DIR}"

    echo "==> Installing oculus_reader..."
    conda run -n "${ENV_NAME}" uv pip install -e "${DROID_DIR}/droid/oculus_reader"

    POLYMETIS_DIR="${DROID_DIR}/droid/fairo/polymetis/polymetis/python"
    if [ -d "$POLYMETIS_DIR" ]; then
        echo "==> Installing polymetis..."
        conda run -n "${ENV_NAME}" uv pip install -e "${POLYMETIS_DIR}"
    fi

    # opencv-python and opencv-contrib-python both install cv2 and conflict.
    # Ensure only the contrib variant (superset) is present so cv2.aruco works.
    echo "==> Fixing opencv: ensuring only contrib variant is installed..."
    conda run -n "${ENV_NAME}" uv pip uninstall opencv-python || true
    conda run -n "${ENV_NAME}" uv pip install "opencv-contrib-python==4.6.0.66"

    echo "==> Fixing numpy (pin <2.0 for compiled extension compatibility)..."
    conda run -n "${ENV_NAME}" uv pip install "numpy>=1.22.4,<2.0"

    echo "==> Fixing protobuf/wandb conflict..."
    conda run -n "${ENV_NAME}" uv pip install "protobuf>=4.21" --upgrade
    conda run -n "${ENV_NAME}" uv pip install wandb --reinstall

    echo "==> Installing pyzed (ZED camera Python bindings)..."
    if [ -f "/usr/local/zed/get_python_api.py" ]; then
        CONDA_PREFIX=$(conda run -n "${ENV_NAME}" python -c "import sys; print(sys.prefix)")
        "${CONDA_PREFIX}/bin/python" /usr/local/zed/get_python_api.py
    else
        echo "    ZED SDK not found at /usr/local/zed — skipping pyzed install."
        echo "    Install the ZED SDK from https://www.stereolabs.com/developers/release first."
    fi
fi

echo ""
echo "==> Done! Activate with:  conda activate ${ENV_NAME}"
