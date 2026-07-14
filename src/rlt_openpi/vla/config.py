"""Thin wrapper around openpi's config system.

Isolates RLT code from openpi's internal config API so that changes
in openpi only require updates here.
"""

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
