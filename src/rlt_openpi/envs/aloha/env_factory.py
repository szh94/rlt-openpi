"""Environment factory for the ALOHA dual-arm robot.

Wraps the ``hansrobot`` direct-control interface in the ``RobotEnv``
interface for online RL (no ROS dependency).

ALOHA is a bimanual setup:
- 2 arms (puppet_left, puppet_right), each 6 joints + 1 gripper
- 14-dim action: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
- Gripper is normalized [0, 1] (0=close, 1=open)
- 3 cameras: cam_high, cam_left_wrist, cam_right_wrist (no cam_low)
- Action type: **joint position** (absolute, not velocity)

The environment passes the 14-dimensional action directly to
``hansrobot.send_action_safe`` and converts camera images from HWC to CHW.

Observations follow the ALOHA schema:
``{"state": [14] float32, "images": {cam_name: (3,H,W) uint8}, "prompt": str}``

Usage::

    # Training
    python scripts/train_jax_s2_onlinerl.py \\
        --env-factory rlt_openpi.envs.aloha.env_factory.make_aloha_env \\
        --task-prompt "pick up the cup" \\
        --action-dim 14 --chunk-length 10 \\
        --vla-config-name pi05_aloha \\
        --vla-checkpoint-dir /path/to/orbax_checkpoint \\
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
    dry_run: bool = True,
    print_actions: bool = False,
    live_image_dir: str = "",
    joint_override: dict[str, float] | None = None,
    speed_deg_s: float = 30.0,
    adapt_to_pi: bool = False,
    obs_source: "ObsSource | None" = None,
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
        reset_position_left: Optional left-arm reset pose containing six
            joint values in degrees. Must be provided with
            ``reset_position_right``.
        reset_position_right: Optional right-arm reset pose containing six
            joint values in degrees. Must be provided with
            ``reset_position_left``.
        image_size: ``(height, width)`` to which camera frames are resized.
        camera_names: Camera names to include in observations.  Defaults to
            ``["cam_high", "cam_left_wrist", "cam_right_wrist"]``.
        dry_run: If True, print actions but do NOT send them to the robot.
        print_actions: If True, log action values at each step (works
            independently of ``dry_run``).
        live_image_dir: Optional directory to save camera frames
            periodically (throttled, ~1 Hz).  Images are saved as PNG
            to ``<live_image_dir>/<timestamp>/<cam_name>.png``.
        joint_override: Optional mapping of action index → joint target in
            **degrees**. Valid indices are 0-5 for the left arm and 7-12
            for the right arm; 6 and 13 are grippers and cannot be overridden.
        speed_deg_s: Joint movement speed in degrees/second (for
            blocking operations like reset).
        adapt_to_pi: Whether to enable PI-style coordinate adaptation.
        obs_source: Optional :class:`~rlt_openpi.envs.obs_source.ObsSource`
            to inject as the obs black box.  When ``None`` (default), obs
            come from the real hansrobot hardware.  When provided (e.g.
            ``MockObsSource`` / ``DatasetObsSource``), robot initialization
            is skipped and ``get_obs`` is delegated entirely to the source.
        **kwargs: Forwarded to ``RobotEnv``.

    Returns:
        ``RobotEnv`` with ``.aloha_robot``, ``.aloha_get_obs_fn``, and
        ``.aloha_cameras`` attributes attached.
    """
    import cv2
    import einops
    from hansrobot.hansrobot_realsense import hansrobot
    from openpi_client import image_tools

    from rlt_openpi.envs.envbase.robot_env import RobotEnv
    from rlt_openpi.envs.obs_source import ObsSource
    from rlt_openpi.envs.real.robot_obs_source import RobotObsSource

    if camera_names is None:
        camera_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    # ------------------------------------------------------------------
    # 黑盒 obs 来源注入（mock / dataset / 自定义 ObsSource）：
    # 非 None 时跳过机器人初始化，obs 全部来自 obs_source.get_obs()。
    # ------------------------------------------------------------------
    if obs_source is not None:
        print(
            f"[aloha] using injected obs_source={type(obs_source).__name__} — "
            "robot init skipped",
        )
        env = RobotEnv(
            step_fn=lambda action: None,
            reset_fn=lambda: None,
            get_obs_fn=obs_source.get_obs,
            action_dim=action_dim,
            chunk_length=chunk_length,
            control_hz=control_hz,
            max_episode_chunks=max_episode_chunks,
            print_actions=print_actions,
            **kwargs,
        )
        env.obs_source = obs_source  # type: ignore[attr-defined]
        return env

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
    # The RL pipeline and hansrobot both use degrees for arm joint positions.
    # ------------------------------------------------------------------
    def step_fn(action: np.ndarray) -> None:
        """Send joint position + gripper targets to both arms.

        Args:
            action: ``[14]`` —
                action[0:6]   = left arm joint positions (degrees)
                action[6]     = left gripper (0=close, 1=open)
                action[7:13]  = right arm joint positions (degrees)
                action[13]    = right gripper (0=close, 1=open)
        """
        if print_actions:
            print("Send action", action)

        # if not dry_run:
            # robot.send_action_safe(action)

    # ------------------------------------------------------------------
    # reset_fn — reset robot via hansrobot.move() (if reset_position set)
    # ------------------------------------------------------------------
    def reset_fn() -> None:
        """Reset robot to starting pose.

        Uses ``hansrobot.move()`` if ``reset_position`` was provided,
        otherwise logs a message (hansrobot has no built-in reset).
        """
        if reset_position_left is not None:
            # reset_position_left is in degrees
            left_joints_deg = reset_position_left
            right_joints_deg = reset_position_right
            if not dry_run:
                print(
                    f"Moving to reset position: left={[f'{v:.1f}' for v in left_joints_deg]}, "
                    f"right={[f'{v:.1f}' for v in right_joints_deg]}",
                )
                # robot.move(left_joints_deg, right_joints_deg)
            else:
                print("[dry-run] robot.move() skipped")
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
    # ------------------------------------------------------------------
    # obs_source — black-box obs access (RobotObsSource wraps the robot)
    #
    # read_obs_fn 读取 hansrobot 原始数据（degrees + raw gripper）；
    # _build_aloha_obs 只整理字段和图像格式，state 中的关节角度原样保持 degrees。
    # obs_source 对外统一提供 get_obs()。
    # ------------------------------------------------------------------
    _frame_stats = [0, 0]  # [ok_count, fail_count]

    def read_obs_fn() -> dict[str, Any]:
        """Read raw hansrobot observation (degrees + raw gripper)."""
        return robot.get_observation()

    def convert_image_hwc_to_chw(img):
        # img: [H, W, 3] -> [3, H, W]
        return np.transpose(img, (2, 0, 1))

    def _build_aloha_obs(raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw hansrobot observation to ALOHA schema.

        Adapts hansrobot observation format::

            {
                "observation.state": [14],       # degrees + raw gripper
                "effort": [12],                  # TCP forces L+R
                "observation.images.<cam_key>": ...,
            }

        Arm joint values remain in degrees; this function performs no angular
        unit conversion.

        Args:
            raw_obs: Raw dict from ``hansrobot.get_observation()``.

        Returns:
            Dict with keys ``"state"`` (float32[14]), ``"images"``
            (dict of cam_name → (3, H, W) uint8), and ``"prompt"`` (str).
        """
        images: dict[str, np.ndarray] = {}
        images["cam_high"] = convert_image_hwc_to_chw(raw_obs["observation.images.head_image"])
        images["cam_left_wrist"] = convert_image_hwc_to_chw(raw_obs["observation.images.left_wrist_image"])
        images["cam_right_wrist"] = convert_image_hwc_to_chw(raw_obs["observation.images.right_wrist_image"])

        result = {
            "images": images,
            "state": raw_obs["observation.state"],
            "prompt": "place phone",
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

    obs_source = RobotObsSource(
        read_obs_fn=read_obs_fn,
        build_obs_fn=_build_aloha_obs,
    )

    def get_obs_fn() -> dict[str, Any]:
        """Return observation dict in ALOHA format (delegates to obs_source)."""
        return obs_source.get_obs()

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
        # ALOHA dry-run is enforced only at the two hardware motion calls in
        # step_fn/reset_fn, so the rest of the control path still executes.
        dry_run=False,
        print_actions=print_actions,
        **kwargs,
    )

    # Expose internals for external access (e.g. cleanup, debugging)
    env.aloha_robot = robot  # type: ignore[attr-defined]
    env.aloha_get_obs_fn = get_obs_fn  # type: ignore[attr-defined]
    env.aloha_cameras = camera_names  # type: ignore[attr-defined]
    env.aloha_dry_run = dry_run  # type: ignore[attr-defined]
    env.obs_source = obs_source  # type: ignore[attr-defined]

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
