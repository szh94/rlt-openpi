#!/usr/bin/env bash
# Sets up a conda environment for rlt-openpi.
#
# Usage:
#   bash setup_env.sh            # creates env named 'rlt'
#   bash setup_env.sh myenvname  # creates env with custom name
#
# Set OPENPI_DIR to point to a local openpi clone to skip the GitHub fetch:
#
#   OPENPI_DIR=/path/to/openpi bash setup_env.sh
#
# After setup:
#   conda activate <env_name>
#   bash example/torch_s1.sh   # or torch_s2.sh, jax_s1.sh, jax_s2_onlinerl.sh, eval_*.sh
set -euo pipefail

# ── Parse args ──────────────────────────────────────────────────────────
ENV_NAME="rl_token"
for arg in "$@"; do
    ENV_NAME="$arg"
done

OPENPI_REV="fdc03f5"
# === [必须修改] 设置你的本地 openpi 路径 ===
# 将 /home/path/to/openpi 替换为你的真实 openpi 克隆路径；留空 "" 则从 GitHub 克隆（需联网）。
OPENPI_DIR="${OPENPI_DIR:-/home/path/to/openpi}"   # ← 修改此路径！
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

# ── Check if env already exists ────────────────────────────────────────
ENV_EXISTS=false
if conda info --envs | grep -q "^${ENV_NAME}[[:space:]]"; then
    ENV_EXISTS=true
fi

if [ "$ENV_EXISTS" = true ]; then
    echo "==> Conda env '${ENV_NAME}' already exists. Skipping env creation and openpi install."
    echo "==> Only installing rlt-openpi and patching transformers..."
else
    # ── Core environment (new) ──────────────────────────────────────────
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
fi

# ── Install rlt-openpi ────────────────────────────────────────────────
# 新环境用 --overrides 覆盖 pyproject.toml 中的 openpi 来源；已有环境直接重装 rlt-openpi
if [ "$ENV_EXISTS" = true ]; then
    echo "==> Installing rlt-openpi (without touching existing openpi)..."
    conda run -n "${ENV_NAME}" uv pip install -e "${SCRIPT_DIR}[dev]" \
        --no-deps 2>/dev/null \
        || conda run -n "${ENV_NAME}" uv pip install -e "${SCRIPT_DIR}[dev]"
else
    echo "==> Installing rlt-openpi (with dev dependencies)..."
    conda run -n "${ENV_NAME}" uv pip install -e "${SCRIPT_DIR}[dev]" \
        --overrides /dev/stdin <<EOF
${OPENPI_URL}
EOF
fi

# ── Patch transformers ────────────────────────────────────────────────
echo "==> Patching transformers with openpi's transformers_replace files..."
OPENPI_PKG_DIR=$(conda run -n "${ENV_NAME}" python -c \
    "import openpi, pathlib; print(pathlib.Path(openpi.__file__).parent)")
TRANSFORMERS_DIR=$(conda run -n "${ENV_NAME}" python -c \
    "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)")
cp -r "${OPENPI_PKG_DIR}/models_pytorch/transformers_replace/"* "${TRANSFORMERS_DIR}/"

echo ""
echo "==> Done! Activate with:  conda activate ${ENV_NAME}"
