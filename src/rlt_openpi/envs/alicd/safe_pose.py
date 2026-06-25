#!/usr/bin/env python3
"""Alicia-D safe-pose module.

Provides:
- ``SAFE_JOINTS_RAD`` : verified safe joint angles (2026-06-24)
- ``SAFE_GRIPPER``    : gripper value for safe open
- ``move_to_safe_pose(robot, speed, torque_off)`` : execute the power-off sequence

CLI usage::

    python scripts/poweroff_safe_pose.py [--port /dev/ttyACM0] [--no-torque-off]

Import usage::

    from rlt_openpi.envs.alicd.safe_pose import move_to_safe_pose, SAFE_JOINTS_RAD, SAFE_GRIPPER
    import alicia_d_sdk

    robot = alicia_d_sdk.create_robot()
    move_to_safe_pose(robot, speed=30.0, torque_off=True)
    robot.disconnect()
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# ── Verified safe pose (2026-06-24, read_joint_state.py) ──────────────────
# Arm folded, low CoG, gripper up — safe to go limp without collision.
SAFE_JOINTS_RAD: list[float] = [
    0.0,        # J0    0°     基座旋转
    +1.9548,   # J1  +112°    肩俯仰
    -0.4363,   # J2   -25°    肘
    0.0,        # J3    0°     腕滚
    -1.0821,   # J4   -62°    腕俯仰
    0.0,        # J5    0°     末端旋转
]
SAFE_GRIPPER = 1000.0  # fully open


# ── Public API ────────────────────────────────────────────────────────────

def move_to_safe_pose(
    robot,
    speed: float = 30.0,
    torque_off: bool = True,
) -> None:
    """Execute the power-off sequence on a connected *robot*.

    Sequence:
        1. Enable torque
        2. Home (all zeros)
        3. J1 only  → shoulder pitch
        4. J2 only  → elbow
        5. J4 only  → wrist pitch
        6. Torque off (unless *torque_off* is False)

    Args:
        robot: A connected ``alicia_d_sdk.SynriaRobotAPI`` instance.
        speed: Joint movement speed in deg/s.
        torque_off: If True, disable torque at the end.
    """
    # ── Step 1: Enable torque ────────────────────────────────────────
    logger.info("Enabling torque...")
    robot.servo_driver.acquire_info("torque_on", wait=True, timeout=2.0)

    # ── Step 2: Home (all zeros) ─────────────────────────────────────
    logger.info("Waypoint: moving to home (all joints = 0 rad)...")
    robot.set_home(speed_deg_s=speed)

    # ── Step 3: J1 only — shoulder pitch ─────────────────────────────
    j1_only = [0.0] * 6
    j1_only[1] = SAFE_JOINTS_RAD[1]
    logger.info("Seq 1/3: J1 → %+.4f rad (%+.1f°), others at 0",
                SAFE_JOINTS_RAD[1], SAFE_JOINTS_RAD[1] * math.degrees(1))
    robot.set_robot_state(
        target_joints=j1_only,
        gripper_value=int(SAFE_GRIPPER),
        speed_deg_s=speed,
        wait_for_completion=True,
    )

    # ── Step 4: J2 only — elbow ──────────────────────────────────────
    j1_j2 = [0.0] * 6
    j1_j2[1] = SAFE_JOINTS_RAD[1]
    j1_j2[2] = SAFE_JOINTS_RAD[2]
    logger.info("Seq 2/3: J2 → %+.4f rad (%+.1f°), J1 held",
                SAFE_JOINTS_RAD[2], SAFE_JOINTS_RAD[2] * math.degrees(1))
    robot.set_robot_state(
        target_joints=j1_j2,
        gripper_value=int(SAFE_GRIPPER),
        speed_deg_s=speed,
        wait_for_completion=True,
    )

    # ── Step 5: J4 only — wrist pitch ────────────────────────────────
    logger.info("Seq 3/3: J4 → %+.4f rad (%+.1f°), J1,J2 held",
                SAFE_JOINTS_RAD[4], SAFE_JOINTS_RAD[4] * math.degrees(1))
    robot.set_robot_state(
        target_joints=SAFE_JOINTS_RAD,
        gripper_value=int(SAFE_GRIPPER),
        speed_deg_s=speed,
        wait_for_completion=True,
    )

    # ── Step 6: Torque off ───────────────────────────────────────────
    if torque_off:
        logger.info("Disabling torque — arm will go limp.  Safe to power off now.")
        robot.servo_driver.acquire_info("torque_off", wait=True, timeout=2.0)
    else:
        logger.info("Torque kept ON.  Arm is actively holding position.")

    logger.info("Safe pose reached.  可以直接断电。")
