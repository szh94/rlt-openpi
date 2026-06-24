#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"

echo "========================================"
echo " Alicia-D Power-Off Safe Pose"
echo "   Home → J1 → J2 → J4 → torque off"
echo "   After this, the arm is safe to power off directly."
echo "========================================"
echo ""

read -rp "Proceed? This will disable torque and the arm will go limp. [y/N] " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

python "$SCRIPT_DIR/../scripts/test_alicd_poweroff.py" "$@"
