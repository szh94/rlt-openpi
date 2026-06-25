"""Data-transform config factory for the Alicia-D setup.

Since Alicia-D produces DROID-schema observations, standard
``DroidInputs`` from OpenPI works directly.  This module provides a
thin wrapper for API consistency with other robot policy configs.

Usage::

    from rlt_openpi.policies.alicd.config import alicd_data_transforms
    transforms_group = alicd_data_transforms(model_config)
"""

from __future__ import annotations

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.droid_policy import DroidInputs, DroidOutputs


def alicd_data_transforms(model_config: _model.BaseModelConfig) -> transforms.Group:
    """Return a ``Group`` that uses standard DroidInputs.

    Alicia-D produces the standard DROID observation schema
    (``observation/joint_position``, ``observation/gripper_position``,
    1-2 camera images, ``prompt``).  Standard ``DroidInputs`` handles
    the 2-camera case by zero-filling the unused third image slot
    with ``mask=False``.

    Args:
        model_config: The VLA model config (needed for ``model_type``).

    Returns:
        A ``transforms.Group`` suitable for passing as
        ``data_transforms`` to ``VLAWrapper``.
    """
    return transforms.Group(
        inputs=[DroidInputs(model_type=model_config.model_type)],
        outputs=[DroidOutputs()],
    )
