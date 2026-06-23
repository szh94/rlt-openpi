"""Environment factory for the ALOHA dual-arm robot.

Wraps OpenPI's ``RealEnv`` (Interbotix ViperX 300s via ROS) in the
``RobotEnv`` interface for online RL.

ALOHA is a bimanual setup:
- 2 arms (puppet_left, puppet_right), each 6 joints + 1 gripper
- 14-dim action: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
- Gripper is normalized [0, 1] (0=close, 1=open)
- 4 cameras: cam_high, cam_low, cam_left_wrist, cam_right_wrist
- Action type: **joint position** (absolute, not velocity)

Observations follow the MockEnv / AlohaInputs schema:
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
        reset_position: Optional 6-joint reset pose.  Defaults to
            ``[0, -0.96, 1.16, 0, -0.3, 0]`` per arm.
        image_size: ``(height, width)`` to which camera frames are resized.
        **kwargs: Forwarded to ``RobotEnv``.

    Returns:
        ``RobotEnv`` with ``.aloha_real_env`` and ``.aloha_get_obs_fn``
        attributes attached.
    """
    import cv2
    import einops
    from openpi_client import image_tools

    from examples.aloha_real.real_env import make_real_env
    from rlt_openpi.envs.robot_env_base.robot_env import RobotEnv

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
        state_len = len(action) // 2
        left_action = action[:state_len]
        right_action = action[state_len:]

        # Send arm joint position targets (non-blocking)
        real_env.puppet_bot_left.arm.set_joint_positions(
            left_action[:6].astype(np.float64), blocking=False
        )
        real_env.puppet_bot_right.arm.set_joint_positions(
            right_action[:6].astype(np.float64), blocking=False
        )

        # Send gripper commands
        real_env.set_gripper_pose(
            float(left_action[-1]),
            float(right_action[-1]),
        )

    # ------------------------------------------------------------------
    # reset_fn — reset both arms to home, reboot grippers
    # ------------------------------------------------------------------
    def reset_fn() -> None:
        """Reset robot: reboot gripper motors, move arms to reset pose,
        close then open grippers."""
        real_env.reset()

    # ------------------------------------------------------------------
    # get_obs_fn — read joint state + camera images in ALOHA schema
    # ------------------------------------------------------------------
    def get_obs_fn() -> dict[str, Any]:
        """Return observation dict in MockEnv / AlohaInputs format.

        Returns:
            Dict with keys ``"state"`` (float32[14]), ``"images"``
            (dict of cam_name → (3, H, W) uint8), and ``"prompt"`` (str).
        """
        obs = real_env.get_observation()

        # Convert images from ROS/cv_bridge (HWC BGR) to model format (CHW RGB)
        images: dict[str, np.ndarray] = {}
        for cam_name in ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]:
            if cam_name not in obs["images"]:
                logger.warning("Camera '%s' missing from observation — using zero image", cam_name)
                images[cam_name] = np.zeros(
                    (3, *image_size), dtype=np.uint8
                )
                continue

            img = obs["images"][cam_name]  # HWC BGR uint8 from cv_bridge
            if img is None:
                images[cam_name] = np.zeros(
                    (3, *image_size), dtype=np.uint8
                )
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, *image_size)
            )
            img = einops.rearrange(img, "h w c -> c h w")
            images[cam_name] = img

        return {
            "state": obs["qpos"].astype(np.float32),
            "images": images,
            "prompt": task_prompt,
        }

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

    # Expose internals for external access (e.g. cleanup, debugging)
    env.aloha_real_env = real_env  # type: ignore[attr-defined]
    env.aloha_get_obs_fn = get_obs_fn  # type: ignore[attr-defined]

    logger.info(
        "ALOHA env ready: action_dim=%d, chunk_length=%d, control_hz=%d, "
        "cameras=[cam_high, cam_low, cam_left_wrist, cam_right_wrist]",
        action_dim, chunk_length, control_hz,
    )
    return env
