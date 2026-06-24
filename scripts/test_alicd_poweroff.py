#!/usr/bin/env python3
"""Alicia-D: move to safe pose then torque-off for power-down.

Home → J1 → J2 → J4 → torque off.  After this the arm is limp and
safe to cut power.

Usage::

    python scripts/test_alicd_poweroff.py [--port /dev/ttyACM0] [--no-torque-off]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import alicia_d_sdk

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from rlt_openpi.envs.alicd.safe_pose import move_to_safe_pose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger("alicd_poweroff")


def main() -> None:
    parser = argparse.ArgumentParser(description="Move robot to safe pose for power-off")
    parser.add_argument("--port", default="", help="Serial port (auto-detect if empty)")
    parser.add_argument("--speed", type=float, default=30.0, help="Joint speed in deg/s")
    parser.add_argument("--no-torque-off", action="store_true", help="Keep torque on")
    args = parser.parse_args()

    logger.info("Connecting to Alicia-D (port=%s)...", args.port or "<auto>")
    robot = alicia_d_sdk.create_robot(port=args.port or "")
    state = robot.get_robot_state("version")
    logger.info("Connected. SN=%s", state.get("serial_number", "?"))

    try:
        move_to_safe_pose(robot, speed=args.speed, torque_off=not args.no_torque_off)
    finally:
        robot.disconnect()
        logger.info("Disconnected.")

    logger.info("Exiting.")
    os._exit(0)


if __name__ == "__main__":
    main()
