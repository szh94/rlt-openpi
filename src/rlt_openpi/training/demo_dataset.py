"""LeRobot demo dataset loader for Stage 1 RLT training.

Loads a LeRobot dataset, applies the OpenPI DROID transform chain,
and yields (observation_dict, actions) pairs
suitable for ``RLTokenTrainer.train()``.

The transform chain matches ``pi05_droid_finetune``:
  RepackTransform → DroidInputs(PI05) → Normalize → ResizeImages →
  TokenizePrompt(discrete_state=True) → PadStatesAndActions(32)
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any

import einops
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.models.model import ModelType, Observation
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies.droid_policy import DroidInputs
from openpi.shared import normalize as _normalize
from openpi.training.config import get_config
from openpi.transforms import (
    Normalize,
    PadStatesAndActions,
    ResizeImages,
    TokenizePrompt,
)
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass
class DemoDatasetConfig:
    """Configuration for the demo dataset."""

    repo_id: str = "local/stack_the_blocks"
    """LeRobot dataset repo ID (local or HuggingFace)."""

    openpi_config_name: str = "pi05_droid_finetune"
    """OpenPI config name for norm stats and model config."""

    action_horizon: int = 16
    """Number of future actions to chunk per sample."""

    use_all_cameras: bool = True
    """If True, use all 3 cameras (exterior_1, exterior_2, wrist) filling
    all PI0.5 image slots. If False, use default DroidInputs (2 cameras,
    3rd slot zeroed with mask=False)."""

    norm_stats_dir: str | None = None
    """Override path to norm_stats.json directory. If None, loads from the
    OpenPI config's assets_dir."""


class DemoDataset(Dataset):
    """PyTorch Dataset wrapping a LeRobot demo dataset for Stage 1 training.

    Each sample yields a dict ready to be passed through
    ``Observation.from_dict()`` after batching, plus an action chunk tensor.

    Args:
        config: Dataset configuration.
    """

    def __init__(self, config: DemoDatasetConfig) -> None:
        self.config = config

        # Load LeRobot dataset
        logger.info("Loading LeRobot dataset: %s", config.repo_id)
        self.lerobot_ds = LeRobotDataset(repo_id=config.repo_id)
        logger.info("Dataset loaded: %d frames", len(self.lerobot_ds))

        # Load OpenPI config for norm stats and model info
        openpi_config = get_config(config.openpi_config_name)
        model_config: Pi0Config = openpi_config.model
        self.action_dim = model_config.action_dim  # 32 (padded)
        self.max_token_len = model_config.max_token_len  # 200 for PI0.5

        # Build transform chain (applied per-sample, unbatched)
        norm_stats = self._load_norm_stats(config, openpi_config)

        self.normalize = Normalize(norm_stats) if norm_stats is not None else None
        self.resize_images = ResizeImages(224, 224)
        self.tokenize_prompt = TokenizePrompt(
            PaligemmaTokenizer(self.max_token_len),
            discrete_state_input=model_config.discrete_state_input,
        )
        self.pad = PadStatesAndActions(self.action_dim)
        self.use_all_cameras = config.use_all_cameras

        # Precompute episode boundaries for action chunking
        self._episode_ends: dict[int, int] = {}
        for ep_idx in range(self.lerobot_ds.num_episodes):
            ep_start = self.lerobot_ds.episode_data_index["from"][ep_idx].item()
            ep_end = self.lerobot_ds.episode_data_index["to"][ep_idx].item()
            for t in range(ep_start, ep_end):
                self._episode_ends[t] = ep_end

    @staticmethod
    def _load_norm_stats(
        config: DemoDatasetConfig, openpi_config: Any
    ) -> dict[str, _normalize.NormStats] | None:
        """Load normalization statistics."""
        if config.norm_stats_dir is not None:
            stats_path = pathlib.Path(config.norm_stats_dir)
            if (stats_path / "norm_stats.json").exists():
                logger.info("Loading norm stats from %s", stats_path)
                return _normalize.load(stats_path)

        # Try loading from OpenPI config's assets
        try:
            data_config = openpi_config.data.create(
                openpi_config.assets_dirs, openpi_config.model
            )
            if data_config.norm_stats is not None:
                logger.info("Loaded norm stats from OpenPI config assets")
                return data_config.norm_stats
        except Exception as e:
            logger.warning("Could not load norm stats from config: %s", e)

        logger.warning("No norm stats found — training without normalization")
        return None

    def __len__(self) -> int:
        return len(self.lerobot_ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single transformed sample.

        Returns a dict with keys:
            image: {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb} uint8 [224,224,3]
            image_mask: {base_0_rgb, ...} bool scalar
            state: float32 [32]  (normalized, padded)
            tokenized_prompt: int32 [200]
            tokenized_prompt_mask: bool [200]
            actions: float32 [action_horizon, 32]  (normalized, padded)
        """
        sample = self.lerobot_ds[idx]

        # --- Build raw observation dict (DROID schema keys) ---
        raw = self._build_raw_dict(sample)

        # --- Action chunking ---
        raw["actions"] = self._chunk_actions(idx)

        # --- Transform chain ---
        # 1. DroidInputs (or custom 3-camera) → {state, image, image_mask, prompt, actions}
        transformed = self._apply_input_transform(raw)

        # 2. Normalize state + actions
        if self.normalize is not None:
            transformed = self.normalize(transformed)

        # 3. Resize images to 224x224
        transformed = self.resize_images(transformed)

        # 4. Tokenize prompt (with discrete state for PI0.5)
        transformed = self.tokenize_prompt(transformed)

        # 5. Pad state and actions to model action_dim (32)
        transformed = self.pad(transformed)

        return transformed

    def _build_raw_dict(self, sample: dict) -> dict:
        """Convert LeRobot sample to DROID-schema observation dict."""
        raw = {}

        # Images — LeRobot stores as float32 CHW, convert to uint8 HWC
        for lerobot_key, droid_key in [
            ("exterior_image_1_left", "observation/exterior_image_1_left"),
            ("exterior_image_2_left", "observation/exterior_image_2_left"),
            ("wrist_image_left", "observation/wrist_image_left"),
        ]:
            if lerobot_key in sample:
                img = sample[lerobot_key]
                if isinstance(img, torch.Tensor):
                    img = img.numpy()
                raw[droid_key] = _parse_image(img)

        # State
        joint_pos = sample["joint_position"]
        gripper_pos = sample["gripper_position"]
        if isinstance(joint_pos, torch.Tensor):
            joint_pos = joint_pos.numpy()
        if isinstance(gripper_pos, torch.Tensor):
            gripper_pos = gripper_pos.numpy()
        raw["observation/joint_position"] = np.asarray(joint_pos, dtype=np.float64)
        raw["observation/gripper_position"] = np.asarray(gripper_pos, dtype=np.float64)

        # Prompt — from LeRobot task
        task_idx = int(sample.get("task_index", 0))
        raw["prompt"] = self.lerobot_ds.meta.tasks.get(task_idx, "do something")

        return raw

    def _chunk_actions(self, idx: int) -> np.ndarray:
        """Build action chunk [action_horizon, action_dim] for timestep idx.

        Reads actions directly from the underlying HuggingFace dataset
        to avoid reloading images for every action step.
        """
        ep_end = self._episode_ends[idx]
        horizon = self.config.action_horizon
        indices = [min(t, ep_end - 1) for t in range(idx, idx + horizon)]
        rows = self.lerobot_ds.hf_dataset.select(indices)
        actions = rows["actions"]
        return np.array(actions, dtype=np.float64)

    def _apply_input_transform(self, raw: dict) -> dict:
        """Map raw DROID-schema dict to model input format.

        If use_all_cameras=True, maps all 3 cameras to the 3 PI0.5 image
        slots. Otherwise, uses the standard DroidInputs which zeros out
        the 3rd slot.
        """
        if not self.use_all_cameras:
            droid_inputs = DroidInputs(model_type=ModelType.PI05)
            return droid_inputs(raw)

        # Custom 3-camera mapping: use all 3 left cameras
        gripper_pos = np.asarray(raw["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([raw["observation/joint_position"], gripper_pos])

        base_image = _parse_image(raw["observation/exterior_image_1_left"])
        wrist_image = _parse_image(raw["observation/wrist_image_left"])

        # Use exterior_image_2 for the 3rd slot (right_wrist_0_rgb)
        if "observation/exterior_image_2_left" in raw:
            third_image = _parse_image(raw["observation/exterior_image_2_left"])
            third_mask = np.True_
        else:
            third_image = np.zeros_like(base_image)
            third_mask = np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": third_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": third_mask,
            },
        }

        if "actions" in raw:
            inputs["actions"] = np.asarray(raw["actions"])
        if "prompt" in raw:
            inputs["prompt"] = raw["prompt"]

        return inputs


def collate_observation_batch(
    batch: list[dict[str, Any]],
) -> tuple[Observation, torch.Tensor]:
    """Custom collate function for DataLoader.

    Takes a list of per-sample dicts (output of DemoDataset.__getitem__)
    and returns (Observation, actions_tensor).  All observation fields are
    returned as torch tensors so they work with PI0Pytorch.

    Args:
        batch: List of transformed sample dicts.

    Returns:
        observation: Observation dataclass with torch tensors.
        actions: [B, action_horizon, action_dim] float32 tensor.
    """
    # Stack images as uint8 numpy, then convert to float32 torch [-1, 1]
    images = {}
    image_masks = {}
    for key in batch[0]["image"]:
        stacked = np.stack([s["image"][key] for s in batch], axis=0)  # [B,H,W,C] uint8
        # Convert uint8 [0,255] → float32 [-1,1] and keep as HWC (PI0Pytorch handles both)
        images[key] = torch.from_numpy(stacked).to(torch.float32) / 255.0 * 2.0 - 1.0
        image_masks[key] = torch.tensor(
            [bool(s["image_mask"][key]) for s in batch], dtype=torch.bool
        )

    state = torch.from_numpy(
        np.stack([s["state"] for s in batch], axis=0).astype(np.float32)
    )
    tokenized_prompt = torch.from_numpy(
        np.stack([s["tokenized_prompt"] for s in batch], axis=0).astype(np.int32)
    ).to(torch.long)
    tokenized_prompt_mask = torch.from_numpy(
        np.stack([s["tokenized_prompt_mask"] for s in batch], axis=0)
    ).to(torch.bool)
    actions = torch.from_numpy(
        np.stack([s["actions"] for s in batch], axis=0).astype(np.float32)
    )

    observation = Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )

    return observation, actions
