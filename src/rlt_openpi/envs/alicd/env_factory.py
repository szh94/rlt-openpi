"""Environment factory for Alicia-D robotic arm.

Wraps the Alicia-D-SDK in the ``RobotEnv`` interface for online RL.
The Alicia-D is a 6-DoF collaborative arm with a gripper end-effector,
communicating via USB serial (UART).  Actions are **joint position targets**
(not velocities).

Cameras are accessed through OpenCV on ``/dev/video*`` devices, configured
via the ``camera_ids`` parameter.

Usage::

    # Training
    python scripts/train_online_rl.py \\
        --env-factory rlt_openpi.envs.alicd.env_factory.make_alicd_env \\
        --task-prompt "pick up the cup" \\
        --action-dim 7 --chunk-length 10 \\
        --vla-config-name pi05_droid \\
        --vla-checkpoint-dir /path/to/vla.safetensors \\
        --rl-token-checkpoint /path/to/rl_token.pt

    # Evaluation
    python scripts/evaluate.py \\
        --env-factory rlt_openpi.envs.alicd.env_factory.make_alicd_env \\
        --task-prompt "pick up the cup" \\
        --checkpoint /path/to/online_rl.pt \\
        --vla-config-name pi05_droid \\
        --vla-checkpoint-dir /path/to/vla.safetensors \\
        --rl-token-checkpoint /path/to/rl_token.pt
"""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def make_alicd_env(
    port: str = "",
    action_dim: int = 7,
    chunk_length: int = 10,
    task_prompt: str = "",
    control_hz: int = 15,
    max_episode_chunks: int = 50,
    speed_deg_s: float = 30.0,
    camera_ids: dict[str, int] | None = None,
    image_size: tuple[int, int] = (224, 224),
    live_image_dir: str = "",
    **kwargs: Any,
):
    """Create an Alicia-D environment for online RL.

    Args:
        port: Serial port for the robot (e.g. ``/dev/ttyACM0`` or ``COM3``).
            Auto-detected if empty.
        action_dim: Dimension of a single-step action.  Default 7
            (6 joint angles + 1 gripper value).
        chunk_length: C, number of single-step actions per chunk.
        task_prompt: Task description passed through to observations for
            VLA embedding extraction.
        control_hz: Control-loop frequency in Hz.
        max_episode_chunks: Maximum chunks per episode before forced
            termination.
        speed_deg_s: Joint movement speed in degrees/second (range ~4-439).
        camera_ids: Optional mapping of camera name → OpenCV device index.
            Example: ``{"exterior_image_1_left": 0, "wrist_image_left": 2}``.
            If ``None``, no camera images are included in observations.
        image_size: ``(height, width)`` to which camera frames are resized.
        **kwargs: Forwarded to ``RobotEnv``.

    Returns:
        ``RobotEnv`` with ``.alicd_robot``, ``.alicd_get_obs_fn``, and
        ``.alicd_cameras`` attributes attached.
    """
    import alicia_d_sdk

    from rlt_openpi.envs.robot_env_base.robot_env import RobotEnv

    # ------------------------------------------------------------------
    # Connect to robot
    # ------------------------------------------------------------------
    logger.info("Connecting to Alicia-D (port=%s)...", port or "<auto>")
    robot = alicia_d_sdk.create_robot(port=port or "")
    logger.info("Connected. SN=%s", robot.get_robot_state("version").get("serial_number", "?"))

    # ------------------------------------------------------------------
    # Camera setup
    # ------------------------------------------------------------------
    cameras: dict[str, Any] = {}
    if camera_ids:
        import cv2

        for cam_name, dev_id in camera_ids.items():
            cap = cv2.VideoCapture(dev_id)
            if not cap.isOpened():
                logger.warning(
                    "Camera '%s' (device %d) could not be opened — will use zero images",
                    cam_name, dev_id,
                )
                cap.release()
                cameras[cam_name] = None
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[1])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[0])
                # Discard first few frames (camera auto-exposure settling)
                for _ in range(5):
                    cap.read()
                cameras[cam_name] = cap
                logger.info("Camera '%s' (device %d) ready (%dx%d)", cam_name, dev_id, *image_size)

    # ------------------------------------------------------------------
    # step_fn — execute a single action on the robot
    # ------------------------------------------------------------------
    def step_fn(action: np.ndarray) -> None:
        """Send joint position + gripper target to the robot.

        Args:
            action: ``[action_dim]`` — first 6 values (dims 0-5) are joint
                angles in radians; dim 7 is gripper position in [0,1]
                (scaled to 0-1000 below).  DROID's 7th joint (dim 6) is
                discarded because Alicia-D has only 6 joints.
        """
        # Joints: take first 6 DROID joints (dims 0-5), discard dim 6
        joint_targets = np.clip(action[:6].astype(np.float64), -math.pi, math.pi).tolist()
        # Gripper: DROID dim 7 is in [0,1], scale to [0, 1000]
        gripper_target = float(np.clip(action[7] * 1000.0, 0.0, 1000.0))

        robot.set_robot_state(
            target_joints=joint_targets,
            gripper_value=gripper_target,
            speed_deg_s=speed_deg_s,
            wait_for_completion=False,
        )

    # ------------------------------------------------------------------
    # reset_fn — send robot to home position
    # ------------------------------------------------------------------
    def reset_fn() -> None:
        """Move the robot to its home (all-zero) pose."""
        robot.set_home(speed_deg_s=speed_deg_s)
        # set_home blocks until home is reached, so we're done here

    # ------------------------------------------------------------------
    # Live image output directory (optional)
    # ------------------------------------------------------------------
    _live_dir = ""
    _last_save = [0.0]  # mutable for closure
    _save_interval = 1.0  # seconds between saves
    if live_image_dir:
        _live_dir = os.path.join(
            live_image_dir, datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(_live_dir, exist_ok=True)
        logger.info("Live images will be saved to %s (every %.1fs)", _live_dir, _save_interval)

    # ------------------------------------------------------------------
    # get_obs_fn — read robot state and camera images
    # ------------------------------------------------------------------
    _zero_frame = np.zeros((*image_size, 3), dtype=np.uint8)

    def get_obs_fn() -> dict[str, Any]:
        """Return observation dict in DROID-schema keys."""
        import cv2

        state = robot.get_robot_state("joint_gripper")
        if state is None:
            raise RuntimeError("Failed to read joint_gripper state from Alicia-D robot")

        obs: dict[str, Any] = {
            "observation/joint_position": np.array(state.angles, dtype=np.float32),
            "observation/gripper_position": np.array([state.gripper], dtype=np.float32),
            "prompt": task_prompt,
        }

        for cam_name, cap in cameras.items():
            if cap is None:
                obs[f"observation/{cam_name}"] = _zero_frame.copy()
                continue
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame.shape[:2] != image_size:
                    frame = cv2.resize(frame, image_size[::-1])
            else:
                frame = _zero_frame.copy()
            obs[f"observation/{cam_name}"] = frame

        # Save live images (throttled: every _save_interval seconds)
        if _live_dir:
            now = time.time()
            if now - _last_save[0] >= _save_interval:
                _last_save[0] = now
                for cam_name in cameras:
                    frame = obs.get(f"observation/{cam_name}")
                    if frame is not None:
                        safe_name = cam_name.replace("/", "_")
                        path = os.path.join(_live_dir, f"{safe_name}.png")
                        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        return obs

    # ------------------------------------------------------------------
    # Build RobotEnv
    # ------------------------------------------------------------------
    env = RobotEnv(
        step_fn=step_fn,
        reset_fn=reset_fn,
        get_obs_fn=get_obs_fn,
        action_dim=action_dim,
        chunk_length=chunk_length,
        control_hz=control_hz,
        max_episode_chunks=max_episode_chunks,
        **kwargs,
    )

    # Attach hardware handle and internals for external access
    # (e.g. intervention manager, cleanup)
    env.alicd_robot = robot  # type: ignore[attr-defined]
    env.alicd_get_obs_fn = get_obs_fn  # type: ignore[attr-defined]
    env.alicd_cameras = cameras  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # close — release robot and camera resources
    # ------------------------------------------------------------------
    def _close() -> None:
        for cap in cameras.values():
            if cap is not None:
                cap.release()
        robot.disconnect()
        logger.info("Alicia-D env closed (robot disconnected, cameras released)")

    env.close = _close  # type: ignore[attr-defined]

    logger.info(
        "Alicia-D env ready: action_dim=%d, chunk_length=%d, control_hz=%d, cameras=%s",
        action_dim, chunk_length, control_hz, list(cameras.keys()) if cameras else "none",
    )
    return env
