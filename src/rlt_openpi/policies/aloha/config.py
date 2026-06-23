"""Data-transform config for the ALOHA dual-arm setup.

Returns ``transforms.Group`` objects suitable for passing to
:class:`VLAWrapper` as ``data_transforms``.

Usage::

    from rlt_openpi.policies.aloha.config import aloha_data_transforms
    from openpi import transforms

    transforms = aloha_data_transforms()
"""

from __future__ import annotations

from openpi import transforms
from openpi.policies.aloha_policy import AlohaInputs, AlohaOutputs


def aloha_data_transforms() -> transforms.Group:
    """Return a ``Group`` that uses standard ALOHA transforms.

    Uses ``adapt_to_pi=False`` since MockEnv has no real ALOHA hardware
    and does not need joint-flip / gripper-space conversion.

    Does NOT include ``DeltaActions`` — the chain only remaps keys and
    decodes images, which is sufficient for single-observation inference.
    """
    return transforms.Group(
        inputs=[AlohaInputs(adapt_to_pi=False)],
        outputs=[AlohaOutputs(adapt_to_pi=False)],
    )
