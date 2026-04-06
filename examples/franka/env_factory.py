"""Environment factory for Franka Panda with DROID.

Requires the ``droid`` package (from franka_teleop) to be installed.
Cameras (ZED) must be connected to the machine running this script.
The Franka robot is accessed via zerorpc (set ``nuc_ip`` in
``droid.misc.parameters`` to the CPU machine's IP).

Usage::

    # Training
    uv run python scripts/train_online_rl.py \\
        --env-factory examples.franka.env_factory.make_franka_env \\
        --task-prompt "pick up the cup" \\
        --action-dim 7 --chunk-length 10 \\
        --vla-config-name pi05_droid \\
        --vla-checkpoint-dir /path/to/vla.safetensors \\
        --rl-token-checkpoint /path/to/rl_token.pt

    # Evaluation
    uv run python scripts/evaluate.py \\
        --env-factory examples.franka.env_factory.make_franka_env \\
        --task-prompt "pick up the cup" \\
        --checkpoint /path/to/online_rl.pt \\
        --vla-config-name pi05_droid \\
        --vla-checkpoint-dir /path/to/vla.safetensors \\
        --rl-token-checkpoint /path/to/rl_token.pt
"""

from __future__ import annotations

import numpy as np

# Camera serial numbers — update these for your setup.
CAM_BASE = "39790647_left"
CAM_WRIST = "15850436_left"
CAM_RIGHT = "35840217_left"


def make_franka_env(
    action_dim: int = 7,
    chunk_length: int = 10,
    task_prompt: str = "",
    control_hz: int = 15,
    max_episode_chunks: int = 50,
    **kwargs,
):
    """Create a Franka Panda environment for online RL.

    Returns an ``rlt_openpi.rollout.robot_env.RobotEnv`` backed by
    DROID's ``RobotEnv`` for robot control and camera reading.
    """
    from droid.robot_env import RobotEnv as DroidEnv

    from rlt_openpi.rollout.robot_env import RobotEnv

    droid = DroidEnv(action_space="cartesian_velocity", control_hz=control_hz)

    def step_fn(action: np.ndarray):
        droid.step(action)

    def reset_fn():
        droid.reset(randomize=False)

    def get_obs_fn() -> dict:
        obs = droid.get_observation()
        state = np.concatenate([
            np.array(obs["robot_state"]["cartesian_position"], dtype=np.float32),
            np.array([obs["robot_state"]["gripper_position"]], dtype=np.float32),
        ])
        return {
            "state": state,
            "base_0_rgb": obs["image"][CAM_BASE],
            "left_wrist_0_rgb": obs["image"][CAM_WRIST],
            "right_wrist_0_rgb": obs["image"][CAM_RIGHT],
            "prompt": task_prompt,
        }

    return RobotEnv(
        step_fn=step_fn,
        reset_fn=reset_fn,
        get_obs_fn=get_obs_fn,
        action_dim=action_dim,
        chunk_length=chunk_length,
        control_hz=control_hz,
        max_episode_chunks=max_episode_chunks,
    )
