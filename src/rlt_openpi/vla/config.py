"""Thin wrapper around openpi's config system.

Isolates RLT code from openpi's internal config API so that changes
in openpi only require updates here.
"""

import pathlib

from openpi.models import model as _model
from openpi.training.config import TrainConfig, get_config


def load_vla_config(config_name: str) -> TrainConfig:
    """Load an openpi TrainConfig by name.

    Args:
        config_name: One of the registered openpi config names,
            e.g. "pi0_aloha_sim", "pi05_libero", "pi0_libero".

    Returns:
        The openpi TrainConfig dataclass for the requested configuration.

    Raises:
        ValueError: If config_name is not found (with suggestions).
    """
    return get_config(config_name)


def load_jax_params(checkpoint_path: str | pathlib.Path):
    """Load JAX model parameters from an Orbax checkpoint.

    Args:
        checkpoint_path: Path to the Orbax checkpoint directory
            (containing ``params/`` subdirectory) or directly to a
            ``params/`` directory.

    Returns:
        Pure dict of JAX arrays representing model parameters.
    """
    return _model.restore_params(pathlib.Path(checkpoint_path))
