"""Environment for the ALOHA dual-arm robot.

Provides ``make_aloha_env``, a factory function that wraps OpenPI's
``RealEnv`` in the :class:`~rlt_openpi.envs.robot_env_base.robot_env.RobotEnv`
interface for online RL training.

ALOHA hardware: two Interbotix ViperX 300s arms (puppet_left, puppet_right)
with 4 ROS cameras (cam_high, cam_low, cam_left_wrist, cam_right_wrist).
Actions are **joint position targets** (absolute, not velocities).

Usage::

    python scripts/train_online_rl.py \\
        --env-factory rlt_openpi.envs.aloha.env_factory.make_aloha_env \\
        --task-prompt "stack the blocks" \\
        --action-dim 14 --chunk-length 10 \\
        --vla-config-name pi05_aloha \\
        ...
"""

from rlt_openpi.envs.aloha.env_factory import make_aloha_env

__all__ = ["make_aloha_env"]
