"""Data-transform config factories for the Franka Panda setup.

Mirrors OpenPI's ``training/config.py`` convention: wires policy-level
transform classes into ``transforms.Group`` objects that the data loader
consumes.

Usage (CLI)::

    --data-transforms-fn rlt_openpi.policies.franka.config.three_camera_droid
"""

from __future__ import annotations

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.droid_policy import DroidOutputs

from rlt_openpi.policies.franka.policy import ThreeCameraDroidInputs


def three_camera_droid(model_config: _model.BaseModelConfig) -> transforms.Group:
    """Return a ``Group`` that uses all three DROID cameras.

    Pass this function's dotted path to ``--data-transforms-fn``::

        --data-transforms-fn rlt_openpi.policies.franka.config.three_camera_droid
    """
    return transforms.Group(
        inputs=[ThreeCameraDroidInputs(model_type=model_config.model_type)],
        outputs=[DroidOutputs()],
    )
