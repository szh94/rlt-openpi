"""Environment factory for the ALOHA dual-arm robot.

Wraps OpenPI's ``RealEnv`` (Interbotix ViperX 300s via ROS) in the
``RobotEnv`` interface for online RL.

ALOHA is a bimanual setup:
- 2 arms (puppet_left, puppet_right), each 6 joints + 1 gripper
- 14-dim action: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
- Gripper is normalized [0, 1] (0=close, 1=open)
- 4 cameras: cam_high, cam_low, cam_left_wrist, cam_right_wrist
- Action type: **joint position** (absolute, not velocity)

Observations follow the ALOHA schema:
``{"state": [14] float32, "images": {cam_name: (3,H,W) uint8}, "prompt": str}``

Usage::

    # Training
    python scripts/train_online_rl.py \\
        --env-factory rlt_openpi.envs.aloha.env_factory.make_aloha_env \\
        --task-prompt "pick up the cup" \\
        --action-dim 14 --chunk-length 10 \\
        --vla-config-name pi05_aloha \\
        --vla-checkpoint-dir /path/to/vla.safetensors \\
        --rl-token-checkpoint /path/to/rl_token.pt

    # Evaluation
    python scripts/evaluate.py \\
        --env-factory rlt_openpi.envs.aloha.env_factory.make_aloha_env \\
        --task-prompt "pick up the cup" \\
        --checkpoint /path/to/online_rl.pt \\
        --vla-config-name pi05_aloha \\
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


def make_aloha_env(
    action_dim: int = 14,
    chunk_length: int = 10,
    task_prompt: str = "",
    control_hz: int = 15,
    max_episode_chunks: int = 50,
    reset_position: list[float] | None = None,
    image_size: tuple[int, int] = (224, 224),
    camera_names: list[str] | None = None,
    dry_run: bool = False,
    print_actions: bool = False,
    live_image_dir: str = "",
    joint_override: dict[str, float] | None = None,
    speed_deg_s: float = 30.0,
    adapt_to_pi: bool = False,
    **kwargs: Any,
):
    """Create an ALOHA dual-arm environment for online RL.

    Args:
        action_dim: Dimension of a single-step action.  Default 14
            (6 joints + 1 gripper per arm).
        chunk_length: C, number of single-step actions per chunk.
        task_prompt: Task description passed through to observations for
            VLA embedding extraction.
        control_hz: Control-loop frequency in Hz.
        max_episode_chunks: Maximum chunks per episode before forced
            termination.
        reset_position: Optional 6-joint reset pose for both arms.
            Defaults to ``[0, -0.96, 1.16, 0, -0.3, 0]`` per arm.
        image_size: ``(height, width)`` to which camera frames are resized.
        camera_names: Camera names to include in observations.  Defaults to
            ``["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]``.
        dry_run: If True, print actions but do NOT send them to the robot.
        print_actions: If True, log action values at each step (works
            independently of ``dry_run``).
        live_image_dir: Optional directory to save camera frames
            periodically (throttled, ~1 Hz).  Images are saved as PNG
            to ``<live_image_dir>/<timestamp>/<cam_name>.png``.
        joint_override: Optional mapping of joint index → target angle
            in **degrees** (not radians).  The target angle is converted
            to radians and clamped before sending to the robot.
            Applied per-arm: indices 0-5 = left arm, 6-11 = right arm.
            Example: ``{"0": 0, "6": 90}`` locks left joint0 at 0° and
            right joint0 at 90°.
        speed_deg_s: Joint movement speed in degrees/second (for
            blocking operations like reset).
        adapt_to_pi: If True, apply PI-style joint-flip and gripper-space
            conversion in ``AlohaInputs`` / ``AlohaOutputs``.  Set to
            True for real ALOHA hardware, False for mock/simulated envs.
        **kwargs: Forwarded to ``RobotEnv``.

    Returns:
        ``RobotEnv`` with ``.aloha_real_env``, ``.aloha_get_obs_fn``, and
        ``.aloha_cameras`` attributes attached.
    """
    import cv2
    import einops
    from openpi_client import image_tools

    from examples.aloha_real.real_env import make_real_env
    from rlt_openpi.envs.robot_env_base.robot_env import RobotEnv

    if camera_names is None:
        camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

    # ------------------------------------------------------------------
    # Create the underlying ALOHA RealEnv (starts ROS node)
    # ------------------------------------------------------------------
    logger.info("Creating ALOHA RealEnv (init_node=True)...")
    real_env = make_real_env(init_node=True, reset_position=reset_position)
    logger.info("ALOHA RealEnv ready")

    # ------------------------------------------------------------------
    # step_fn — send one action to both arms
    # ------------------------------------------------------------------
    def step_fn(action: np.ndarray) -> None:
        """Send joint position + gripper targets to both arms.

        Args:
            action: ``[14]`` —
                action[0:6]   = left arm joint positions (rad)
                action[6]     = left gripper (0=close, 1=open)
                action[7:13]  = right arm joint positions (rad)
                action[13]    = right gripper (0=close, 1=open)
        """
        if dry_run:
            return

        state_len = len(action) // 2
        left_action = action[:state_len].copy()
        right_action = action[state_len:].copy()

        # Apply joint override (safety filter).  Values are in DEGREES.
        if joint_override is not None:
            for idx_str, val in joint_override.items():
                i = int(idx_str)
                target_rad = math.radians(float(val))
                if 0 <= i < 6:
                    left_action[i] = target_rad
                elif 6 <= i < 12:
                    right_action[i - 6] = target_rad
                logger.debug(
                    "Joint %d overridden to %.1f° (%.4f rad)", i, float(val), target_rad,
                )

        # Clamp joints to [-pi, pi]
        left_joints = np.clip(left_action[:6].astype(np.float64), -math.pi, math.pi)
        right_joints = np.clip(right_action[:6].astype(np.float64), -math.pi, math.pi)

        # Send arm joint position targets (non-blocking)
        real_env.puppet_bot_left.arm.set_joint_positions(
            left_joints, blocking=False,
        )
        real_env.puppet_bot_right.arm.set_joint_positions(
            right_joints, blocking=False,
        )

        # Send gripper commands
        real_env.set_gripper_pose(
            float(np.clip(left_action[-1], 0.0, 1.0)),
            float(np.clip(right_action[-1], 0.0, 1.0)),
        )

    # ------------------------------------------------------------------
    # reset_fn — reset both arms to home, reboot grippers
    # ------------------------------------------------------------------
    def reset_fn() -> None:
        """Reset robot: reboot gripper motors, move arms to reset pose,
        close then open grippers."""
        if dry_run:
            logger.info("[dry-run] reset() called (no robot action)")
            return
        real_env.reset()

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
    # get_obs_fn — read joint state + camera images in ALOHA schema
    # ------------------------------------------------------------------
    _frame_stats = [0, 0]  # [ok_count, fail_count]

    def get_obs_fn() -> dict[str, Any]:
        """Return observation dict in ALOHA format.

        Returns:
            Dict with keys ``"state"`` (float32[14]), ``"images"``
            (dict of cam_name → (3, H, W) uint8), and ``"prompt"`` (str).
        """
        obs = real_env.get_observation()

        # Convert images from ROS/cv_bridge (HWC BGR) to model format (CHW RGB)
        images: dict[str, np.ndarray] = {}
        for cam_name in camera_names:
            if cam_name not in obs["images"]:
                logger.warning(
                    "Camera '%s' missing from observation — using zero image", cam_name,
                )
                images[cam_name] = np.zeros((3, *image_size), dtype=np.uint8)
                continue

            img = obs["images"][cam_name]  # HWC BGR uint8 from cv_bridge
            if img is None:
                images[cam_name] = np.zeros((3, *image_size), dtype=np.uint8)
                _frame_stats[1] += 1
                continue

            _frame_stats[0] += 1
            if _frame_stats[0] == 1:  # only on first successful read
                logger.info(
                    "Camera '%s' first frame: shape=%s, dtype=%s, "
                    "min=%.1f max=%.1f mean=%.1f",
                    cam_name, img.shape, img.dtype,
                    img.min(), img.max(), img.mean(),
                )

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, *image_size),
            )
            img = einops.rearrange(img, "h w c -> c h w")
            images[cam_name] = img

        result: dict[str, Any] = {
            "state": obs["qpos"].astype(np.float32),
            "images": images,
            "prompt": task_prompt,
        }

        # Save live images (throttled: every _save_interval seconds)
        if _live_dir:
            now = time.time()
            if now - _last_save[0] >= _save_interval:
                _last_save[0] = now
                for cam_name in camera_names:
                    frame = images.get(cam_name)
                    if frame is not None and frame.max() > 0:
                        safe_name = cam_name.replace("/", "_")
                        path = os.path.join(_live_dir, f"{safe_name}.png")
                        img_hwc = einops.rearrange(frame, "c h w -> h w c")
                        cv2.imwrite(path, cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR))

        return result

    # ------------------------------------------------------------------
    # print_actions_fn — log action values (wrapped by RobotEnv if print_actions=True)
    # ------------------------------------------------------------------
    def _log_action(action: np.ndarray) -> None:
        """Log a single action step."""
        if not print_actions:
            return
        left = ", ".join(f"{action[i]:+.4f}" for i in range(6))
        right = ", ".join(f"{action[i]:+.4f}" for i in range(7, 13))
        logger.info(
            "action | L:[%s] Lg:%.3f | R:[%s] Rg:%.3f%s",
            left, action[6], right, action[13],
            " [DRY-RUN]" if dry_run else "",
        )

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
        print_actions=print_actions,
        **kwargs,
    )

    # Expose internals for external access (e.g. cleanup, debugging)
    env.aloha_real_env = real_env  # type: ignore[attr-defined]
    env.aloha_get_obs_fn = get_obs_fn  # type: ignore[attr-defined]
    env.aloha_cameras = camera_names  # type: ignore[attr-defined]
    env.aloha_dry_run = dry_run  # type: ignore[attr-defined]

    # Attach action logger so RobotEnv can call it after each step
    if print_actions:
        env._log_action_fn = _log_action  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # close — release ROS and camera resources
    # ------------------------------------------------------------------
    def _close() -> None:
        logger.info(
            "Closing ALOHA env (cameras: ok=%d fail=%d)",
            _frame_stats[0], _frame_stats[1],
        )
        # ALOHA RealEnv cleanup is handled by ROS node shutdown
        logger.info("ALOHA env closed")

    env.close = _close  # type: ignore[attr-defined]

    logger.info(
        "ALOHA env ready: action_dim=%d, chunk_length=%d, control_hz=%d, "
        "cameras=%s, dry_run=%s, adapt_to_pi=%s",
        action_dim, chunk_length, control_hz,
        camera_names, dry_run, adapt_to_pi,
    )
    return env
