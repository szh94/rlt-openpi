"""DROID policy transforms for 3-camera setups.

Mirrors OpenPI's ``policies/droid_policy.py`` convention: defines the
per-sample input/output transform classes for a specific robot.

``ThreeCameraDroidInputs`` is a drop-in replacement for OpenPI's
``DroidInputs`` that populates the third image slot with a real camera
(``exterior_image_2_left``) instead of zeroing it out.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Ensure image is uint8 HWC (same logic as ``droid_policy._parse_image``)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ThreeCameraDroidInputs(transforms.DataTransformFn):
    """Like OpenPI's ``DroidInputs`` but maps a 3rd camera to the unused image slot.

    Standard ``DroidInputs`` zeros out ``right_wrist_0_rgb`` with
    ``mask=False``.  This variant fills it with ``exterior_image_2_left``
    (if present) so that PI0.5 can attend to all three camera views
    during fine-tuning.

    Falls back to a zeroed image with ``mask=False`` when the 3rd camera
    key is absent, making it safe to use on 2-camera datasets as well.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # gripper_pos = np.asarray(data["observation/gripper_position"])
        # if gripper_pos.ndim == 0:
        #     gripper_pos = gripper_pos[np.newaxis]
        # state = np.concatenate([data["observation/joint_position"], gripper_pos])
        state = np.asarray(data["observation/joint_position"])

        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        if "observation/exterior_image_2_left" in data:
            third_image = _parse_image(data["observation/exterior_image_2_left"])
            third_mask = np.True_
        else:
            third_image = np.zeros_like(base_image)
            third_mask = np.False_

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs: dict = {
            "state": state,
            "image": dict(zip(names, (base_image, wrist_image, third_image), strict=True)),
            "image_mask": dict(zip(names, (np.True_, np.True_, third_mask), strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        # if "prompt" in data:
        #     if isinstance(data["prompt"], bytes):
        #         data["prompt"] = data["prompt"].decode("utf-8")
        #     inputs["prompt"] = data["prompt"]
        inputs["prompt"] = "place phone"

        return inputs
