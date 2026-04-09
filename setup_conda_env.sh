#!/usr/bin/env bash
# Sets up a conda environment for rlt-openpi + droid (Franka robot).
#
# Usage:
#   bash setup_conda_env.sh            # creates env named 'rlt'
#   bash setup_conda_env.sh myenvname  # creates env with custom name
#
# After setup, activate and run stage2 with:
#   conda activate <env_name>
#   bash exp/stage2.sh
set -euo pipefail

ENV_NAME="${1:-rlt}"
OPENPI_REV="fdc03f5"
DROID_DIR="${HOME}/franka_teleop"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Creating conda env '${ENV_NAME}' with Python 3.11..."
conda create -n "${ENV_NAME}" python=3.11 -y

echo "==> Installing uv (fast resolver, required for openpi's deep dep graph)..."
conda run -n "${ENV_NAME}" pip install uv

OPENPI_URL="openpi @ git+https://github.com/Physical-Intelligence/openpi@${OPENPI_REV}"

echo "==> Installing real openpi from GitHub (rev ${OPENPI_REV})..."
conda run -n "${ENV_NAME}" uv pip install "${OPENPI_URL}"

echo "==> Installing rlt-openpi..."
conda run -n "${ENV_NAME}" uv pip install -e "${SCRIPT_DIR}" \
    --overrides /dev/stdin <<EOF
${OPENPI_URL}
EOF

echo "==> Installing droid from ${DROID_DIR}..."
conda run -n "${ENV_NAME}" uv pip install -e "${DROID_DIR}"

echo "==> Installing oculus_reader..."
conda run -n "${ENV_NAME}" uv pip install -e "${DROID_DIR}/droid/oculus_reader"

# opencv-python and opencv-contrib-python both install cv2 and conflict with each other.
# Ensure only the contrib variant (superset) is present so cv2.aruco is available.
echo "==> Fixing opencv: ensuring only contrib variant is installed..."
conda run -n "${ENV_NAME}" uv pip uninstall opencv-python || true
conda run -n "${ENV_NAME}" uv pip install "opencv-contrib-python==4.6.0.66"

echo "==> Fixing numpy (pin <2.0 for compiled extension compatibility)..."
conda run -n "${ENV_NAME}" uv pip install "numpy>=1.22.4,<2.0"

echo "==> Fixing protobuf/wandb conflict (droid pins protobuf==3.20.1 but wandb needs newer)..."
conda run -n "${ENV_NAME}" uv pip install "protobuf>=4.21" --upgrade
conda run -n "${ENV_NAME}" uv pip install wandb --reinstall

echo "==> Installing pyzed (ZED camera Python bindings)..."
if [ -f "/usr/local/zed/get_python_api.py" ]; then
    /home/alin/miniconda3/envs/"${ENV_NAME}"/bin/python /usr/local/zed/get_python_api.py
else
    echo "    ZED SDK not found at /usr/local/zed — skipping pyzed install."
    echo "    Install the ZED SDK from https://www.stereolabs.com/developers/release first."
fi

echo "==> Patching transformers with openpi's transformers_replace files..."
OPENPI_DIR=$(conda run -n "${ENV_NAME}" python -c \
    "import openpi, pathlib; print(pathlib.Path(openpi.__file__).parent)")
TRANSFORMERS_DIR=$(conda run -n "${ENV_NAME}" python -c \
    "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)")

cp -r "${OPENPI_DIR}/models_pytorch/transformers_replace/"* "${TRANSFORMERS_DIR}/"

echo ""
echo "==> Done! Activate with:  conda activate ${ENV_NAME}"
echo "    Then run:              bash exp/stage2.sh"
