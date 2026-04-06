"""Dynamic environment factory for pluggable robot/sim environments.

Users provide a Python import path to a factory function that returns
an environment compatible with ``RolloutWorker`` (must have ``reset()``,
``step()``, ``action_dim``, ``chunk_length``).

Example usage::

    env = make_env(
        "examples.franka.env_factory.make_franka_env",
        action_dim=7,
        chunk_length=10,
        task_prompt="pick up the cup",
    )
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def make_env(env_factory: str, **kwargs: Any) -> Any:
    """Import and call a user-provided env factory function.

    Args:
        env_factory: Dotted Python import path to a callable, e.g.
            ``"my_package.envs.make_franka_env"``.
        **kwargs: Forwarded to the factory function (typically
            ``action_dim``, ``chunk_length``, ``task_prompt``).

    Returns:
        An env object with ``reset()``, ``step(action_chunk)``,
        ``action_dim``, and ``chunk_length``.
    """
    module_path, func_name = env_factory.rsplit(".", 1)
    module = importlib.import_module(module_path)
    factory_fn = getattr(module, func_name)
    logger.info("Creating env via %s", env_factory)
    return factory_fn(**kwargs)
