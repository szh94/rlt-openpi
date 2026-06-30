"""Data-transform config for the ALOHA dual-arm setup.

Returns ``transforms.Group`` objects suitable for passing to
:class:`VLAWrapper` as ``data_transforms``.

Usage::

    from rlt_openpi.policies.aloha.config import aloha_data_transforms
    from openpi import transforms

    transforms = aloha_data_transforms(adapt_to_pi=True)
"""

from __future__ import annotations

from openpi import transforms
from openpi.policies.aloha_policy import AlohaInputs, AlohaOutputs


def aloha_data_transforms(adapt_to_pi: bool = False) -> transforms.Group:
    """Return a ``Group`` that uses standard ALOHA transforms.

    Args:
        adapt_to_pi: If True, enable joint-flip and gripper-space
            conversion for real ALOHA hardware.  Set to False for
            mock/simulated environments that don't need coordinate
            adaptation.  Default False.

    Does NOT include ``DeltaActions`` — the chain only remaps keys and
    decodes images, which is sufficient for single-observation inference.
    """
    return transforms.Group(
        inputs=[AlohaInputs(adapt_to_pi=adapt_to_pi)],
        outputs=[AlohaOutputs(adapt_to_pi=adapt_to_pi)],
    )
