"""Environment factory for Franka Panda with DROID.

Requires the ``droid`` package to be installed (see ``setup_env.sh --robot``).
Cameras (ZED) must be connected to the machine running this script.
The Franka robot is accessed via zerorpc (set ``nuc_ip`` in
``droid.misc.parameters`` to the CPU machine's IP).

Usage::

    # Training
    python scripts/train_online_rl.py \\
        --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \\
        --task-prompt "pick up the cup" \\
        --action-dim 7 --chunk-length 10 \\
        --vla-config-name pi05_droid \\
        --vla-checkpoint-dir /path/to/vla.safetensors \\
        --rl-token-checkpoint /path/to/rl_token.pt

    # Evaluation
    python scripts/evaluate.py \\
        --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \\
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
    action_dim: int = 8,
    chunk_length: int = 10,
    task_prompt: str = "",
    control_hz: int = 15,
    max_episode_chunks: int = 50,
    action_space: str = "joint_velocity",
    **kwargs,
):
    """Create a Franka Panda environment for online RL.

    Returns an ``rlt_openpi.envs.robot_env_base.robot_env.RobotEnv`` backed by
    DROID's ``RobotEnv`` for robot control and camera reading.
    """
    from droid.robot_env import RobotEnv as DroidEnv

    from rlt_openpi.envs.robot_env_base.robot_env import RobotEnv

    droid = DroidEnv(action_space=action_space, control_hz=control_hz)

    def step_fn(action: np.ndarray):
        droid.step(np.clip(action[:droid.DoF], -1.0, 1.0))

    def reset_fn():
        droid.reset(randomize=False)

    def get_obs_fn() -> dict:
        """Return observation in DROID-schema keys.

        ``DroidInputs`` (or a custom override like
        ``ThreeCameraDroidInputs``) in the VLA transform chain will
        convert these into the model-format dict (state, images, masks).
        """
        obs = droid.get_observation()
        return {
            "observation/joint_position": np.array(
                obs["robot_state"]["joint_positions"], dtype=np.float32
            ),
            "observation/gripper_position": np.array(
                [obs["robot_state"]["gripper_position"]], dtype=np.float32
            ),
            "observation/exterior_image_1_left": obs["image"][CAM_BASE],
            "observation/wrist_image_left": obs["image"][CAM_WRIST],
            "observation/exterior_image_2_left": obs["image"][CAM_RIGHT],
            "prompt": task_prompt,
        }

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

    # Expose internals so the VR intervention manager can step the robot
    # directly and read observations in the same format as the env.
    env.droid_env = droid  # type: ignore[attr-defined]
    env.droid_get_obs_fn = get_obs_fn  # type: ignore[attr-defined]
    return env
