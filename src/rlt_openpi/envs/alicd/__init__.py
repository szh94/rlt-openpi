"""Environment for the Alicia-D robotic arm (Synria Robotics).

Provides ``make_alicd_env``, a factory function that wraps the
Alicia-D-SDK in the :class:`~rlt_openpi.envs.robot_env_base.robot_env.RobotEnv`
interface for online RL training.

Usage::

    python scripts/train_online_rl.py \\
        --env-factory rlt_openpi.envs.alicd.env_factory.make_alicd_env \\
        --task-prompt "pick up the cup" \\
        --action-dim 7 --chunk-length 10 \\
        ...
"""

from rlt_openpi.envs.alicd.env_factory import make_alicd_env

__all__ = ["make_alicd_env"]
