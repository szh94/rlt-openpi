"""Environment factory for the ALOHA dual-arm robot.

Wraps the ``hansrobot`` direct-control interface in the ``RobotEnv``
interface for online RL (no ROS dependency).

ALOHA is a bimanual setup:
- 2 arms (puppet_left, puppet_right), each 6 joints + 1 gripper
- 14-dim action: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
- Gripper is normalized [0, 1] (0=close, 1=open)
- 3 cameras: cam_high, cam_left_wrist, cam_right_wrist (no cam_low)
- Action type: **joint position** (absolute, not velocity)

**Unit conversions**: The RL pipeline works in radians + normalized [0,1]
gripper, while hansrobot expects degrees + raw gripper values.  This factory
handles both conversions transparently in ``step_fn`` and ``get_obs_fn``.

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

import math
import os
import time
from datetime import datetime
from typing import Any

import numpy as np

# hansrobot gripper range (raw units): 0 = closed, ~1100 = open
_HANSROBOT_GRIPPER_MAX = 1000.0


def make_aloha_env(
    action_dim: int = 14,
    chunk_length: int = 10,
    task_prompt: str = "",
    control_hz: int = 15,
    max_episode_chunks: int = 50,
    reset_position_left: list[float] | None = None,
    reset_position_right: list[float] | None = None,
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
        reset_position: Optional 6-joint reset pose for both arms
            (in **radians**).  Used by ``hansrobot.move()`` if set.
        image_size: ``(height, width)`` to which camera frames are resized.
        camera_names: Camera names to include in observations.  Defaults to
            ``["cam_high", "cam_left_wrist", "cam_right_wrist"]``.
        dry_run: If True, print actions but do NOT send them to the robot.
        print_actions: If True, log action values at each step (works
            independently of ``dry_run``).
        live_image_dir: Optional directory to save camera frames
            periodically (throttled, ~1 Hz).  Images are saved as PNG
            to ``<live_image_dir>/<timestamp>/<cam_name>.png``.
        joint_override: Optional mapping of joint index → target angle
            in **degrees** (not radians).  Applied directly in degree
            space before sending to hansrobot.
            Indices: 0-5 = left arm, 6-11 = right arm.
            Example: ``{"0": 0, "6": 90}`` locks left joint0 at 0° and
            right joint0 at 90°.
        speed_deg_s: Joint movement speed in degrees/second (for
            blocking operations like reset).
        adapt_to_pi: If True, apply PI-style joint-flip and gripper-space
            conversion in ``AlohaInputs`` / ``AlohaOutputs``.  Set to
            True for real ALOHA hardware, False for mock/simulated envs.
        **kwargs: Forwarded to ``RobotEnv``.

    Returns:
        ``RobotEnv`` with ``.aloha_robot``, ``.aloha_get_obs_fn``, and
        ``.aloha_cameras`` attributes attached.
    """
    import cv2
    import einops
    from example.hansrobot.hansrobot_realsense import hansrobot
    from openpi_client import image_tools

    from rlt_openpi.envs.robot_env_base.robot_env import RobotEnv

    if camera_names is None:
        camera_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    # ------------------------------------------------------------------
    # Create the hansrobot instance
    # ------------------------------------------------------------------
    print("Creating hansrobot connection...")
    robot = hansrobot(hansrobot.config_class())
    robot.connect()
    print("hansrobot connected")

    # ------------------------------------------------------------------
    # step_fn — send one action to both arms via hansrobot
    #
    # The RL pipeline produces actions in **radians** for joints and
    # **[0,1]** for grippers.  hansrobot expects **degrees** and raw
    # gripper values (0 ~ 1100).  We convert here.
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

        print("=========\nSend action", action)
        # robot.send_action_safe(action)

    # ------------------------------------------------------------------
    # reset_fn — reset robot via hansrobot.move() (if reset_position set)
    # ------------------------------------------------------------------
    def reset_fn() -> None:
        """Reset robot to starting pose.

        Uses ``hansrobot.move()`` if ``reset_position`` was provided,
        otherwise logs a message (hansrobot has no built-in reset).
        """
        if dry_run:
            print("[dry-run] reset() called (no robot action)")
            return

        if reset_position_left is not None:
            # reset_position_left is in degrees
            left_joints_deg = reset_position_left
            right_joints_deg = reset_position_right
            print(
                f"Moving to reset position: left={[f'{v:.1f}' for v in left_joints_deg]}, "
                f"right={[f'{v:.1f}' for v in right_joints_deg]}",
            )
            robot.move(left_joints_deg, right_joints_deg)
        else:
            print("No reset_position set — skipping hardware reset")

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
        print(f"Live images will be saved to {_live_dir} (every {_save_interval:.1f}s)")

    # ------------------------------------------------------------------
    # Camera name mapping: hansrobot keys → ALOHA standard names
    #
    # hansrobot.get_observation() returns images under
    # ``observation.images.{cam_key}`` where ``cam_key`` comes from the
    # camera config.  We map these to ALOHA standard names used by the
    # downstream pipeline (AlohaInputs).
    # ------------------------------------------------------------------
    _CAMERA_MAP = {
        "head_image": "cam_high",
        "left_wrist_image": "cam_left_wrist",
        "right_wrist_image": "cam_right_wrist",
    }

    # ------------------------------------------------------------------
    # get_obs_fn — read joint state + camera images in ALOHA schema
    #
    # hansrobot returns degrees + raw gripper; we convert to radians +
    # [0,1] for the RL pipeline.
    # ------------------------------------------------------------------
    _frame_stats = [0, 0]  # [ok_count, fail_count]

    def get_obs_fn() -> dict[str, Any]:
        """Return observation dict in ALOHA format.

        Adapts hansrobot observation format::

            {
                "observation.state": [14],       # degrees + raw gripper
                "effort": [12],                  # TCP forces L+R
                "observation.images.<cam_key>": ...,
            }

        to the ALOHA standard schema (radians + normalized gripper).

        Returns:
            Dict with keys ``"state"`` (float32[14]), ``"images"``
            (dict of cam_name → (3, H, W) uint8), and ``"prompt"`` (str).
        """
        obs = robot.get_observation()

        # ------------------------------------------------------------------
        # State: convert degrees → radians, raw gripper → [0, 1]
        # ------------------------------------------------------------------
        raw_state = obs.get("observation.state")
        if raw_state is None:
            state = np.zeros(14, dtype=np.float32)
        else:
            raw_state = np.asarray(raw_state, dtype=np.float64)
            # Joints: deg → rad  (indices 0-5 and 7-12)
            joints_rad = np.radians(
                np.concatenate([raw_state[:6], raw_state[7:13]]),
            )
            # Grippers: raw → [0, 1]  (indices 6 and 13)
            gripper_norm = np.clip(
                np.array([raw_state[6], raw_state[13]]) / _HANSROBOT_GRIPPER_MAX,
                0.0, 1.0,
            )
            # Reassemble: [left_j(6), left_g(1), right_j(6), right_g(1)]
            state = np.zeros(14, dtype=np.float32)
            state[:6] = joints_rad[:6]
            state[6] = gripper_norm[0]
            state[7:13] = joints_rad[6:]
            state[13] = gripper_norm[1]

        # ------------------------------------------------------------------
        # Images: map hansrobot camera keys to ALOHA standard names
        # ------------------------------------------------------------------
        images: dict[str, np.ndarray] = {}
        for cam_name in camera_names:
            # Find hansrobot key for this ALOHA cam name
            hr_key = None
            for hr_suffix, aloha_name in _CAMERA_MAP.items():
                if aloha_name == cam_name:
                    hr_key = f"observation.images.{hr_suffix}"
                    break

            if hr_key is None or hr_key not in obs:
                print(f"[WARNING] Camera '{cam_name}' missing from hansrobot observation — using zero image")
                images[cam_name] = np.zeros((3, *image_size), dtype=np.uint8)
                continue

            img = obs[hr_key]
            if img is None:
                images[cam_name] = np.zeros((3, *image_size), dtype=np.uint8)
                _frame_stats[1] += 1
                continue

            _frame_stats[0] += 1
            if _frame_stats[0] == 1:  # only on first successful read
                print(
                    f"Camera '{cam_name}' first frame: shape={img.shape}, dtype={img.dtype}, "
                    f"min={img.min():.1f} max={img.max():.1f} mean={img.mean():.1f}",
                )

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, *image_size),
            )
            img = einops.rearrange(img, "h w c -> c h w")
            images[cam_name] = img

        result: dict[str, Any] = {
            "state": state,
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
        print(
            f"action | L:[{left}] Lg:{action[6]:.3f} | R:[{right}] Rg:{action[13]:.3f}"
            f"{' [DRY-RUN]' if dry_run else ''}",
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
    env.aloha_robot = robot  # type: ignore[attr-defined]
    env.aloha_get_obs_fn = get_obs_fn  # type: ignore[attr-defined]
    env.aloha_cameras = camera_names  # type: ignore[attr-defined]
    env.aloha_dry_run = dry_run  # type: ignore[attr-defined]

    # Attach action logger so RobotEnv can call it after each step
    if print_actions:
        env._log_action_fn = _log_action  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # close — disconnect hansrobot
    # ------------------------------------------------------------------
    def _close() -> None:
        print(
            f"Closing ALOHA env (cameras: ok={_frame_stats[0]} fail={_frame_stats[1]})",
        )
        robot.disconnect()
        print("ALOHA env closed")

    env.close = _close  # type: ignore[attr-defined]

    print(
        f"ALOHA env ready: action_dim={action_dim}, chunk_length={chunk_length}, control_hz={control_hz}, "
        f"cameras={camera_names}, dry_run={dry_run}, adapt_to_pi={adapt_to_pi}",
    )
    return env
