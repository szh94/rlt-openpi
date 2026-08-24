"""Environment for the ALOHA dual-arm robot.

Provides ``make_aloha_env``, a factory function that wraps the
``hansrobot`` direct-control interface in the
:class:`~rlt_openpi.envs.envbase.robot_env.RobotEnv` interface
for online RL training (no ROS dependency).

ALOHA hardware: two arms (left/right) controlled via ``hansrobot``
with 3 cameras (cam_high, cam_left_wrist, cam_right_wrist).
Actions are **joint position targets** (absolute, not velocities).

Usage::

    python scripts/train_jax_s2_onlinerl.py \\
        --env-factory rlt_openpi.envs.aloha.env_factory.make_aloha_env \\
        --task-prompt "stack the blocks" \\
        --action-dim 14 --chunk-length 10 \\
        --vla-config-name pi05_aloha \\
        ...
"""

from rlt_openpi.envs.aloha.env_factory import make_aloha_env

__all__ = ["make_aloha_env"]
