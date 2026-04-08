"""Stage 1: Train the RL token encoder-decoder on demonstration data.

Supports two modes:
  - Frozen VLA (alpha=0): Only trains the RL token encoder-decoder.
  - Joint training (alpha>0): Simultaneously fine-tunes the VLA and
    trains the RL token encoder-decoder with stop-gradient separation.

The data pipeline delegates entirely to OpenPI's transform chain
(RepackTransform → DroidInputs → Normalize → ResizeImages →
TokenizePrompt → PadStatesAndActions) so that normalisation and camera
layout exactly match the pretrained model.

Usage:
    uv run python scripts/train_rl_token.py --help

    # Joint training (recommended):
    uv run python scripts/train_rl_token.py \
        --train.vla-config-name pi05_droid_finetune \
        --train.vla-checkpoint-dir /path/to/model.safetensors \
        --train.vla-finetune-alpha 1.0 \
        --train.num-train-steps 5000 \
        --repo-id local/stack_the_blocks

    # With custom data transforms (e.g. 3-camera DROID):
    uv run python scripts/train_rl_token.py \
        --train.vla-config-name pi05_droid_finetune \
        --train.vla-checkpoint-dir /path/to/model.safetensors \
        --train.vla-finetune-alpha 1.0 \
        --repo-id local/stack_the_blocks \
        --data-transforms-fn rlt_openpi.policies.franka.config.three_camera_droid
"""

from __future__ import annotations

import dataclasses
import importlib
import logging

import tyro

from rlt_openpi.training.config import RLTokenTrainConfig
from rlt_openpi.training.data_loader import build_data_loader
from rlt_openpi.training.rl_token_trainer import RLTokenTrainer
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainConfig:
    """Full config for Stage 1 training."""

    train: RLTokenTrainConfig = dataclasses.field(default_factory=RLTokenTrainConfig)
    """RL token trainer config."""

    repo_id: str = "local/stack_the_blocks"
    """LeRobot dataset repo ID (local or HuggingFace)."""

    data_transforms_fn: str | None = None
    """Dotted import path to a ``(ModelConfig) -> transforms.Group``
    factory that overrides the default data transforms from the OpenPI
    config.  For example:
    ``rlt_openpi.policies.franka.config.three_camera_droid``."""

    num_workers: int = 4
    """DataLoader worker processes."""


def _resolve_data_transforms(dotted_path: str | None, openpi_config_name: str):
    """Dynamically import and call a data-transforms factory if provided."""
    if dotted_path is None:
        return None

    from openpi.training.config import get_config

    module_path, func_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    factory_fn = getattr(module, func_name)

    config = get_config(openpi_config_name)
    return factory_fn(config.model)


def main(config: TrainConfig) -> None:
    """Train the RL token encoder-decoder (Stage 1)."""
    log.info("Stage 1 config: %s", config)

    rl_logger = Logger.from_train_config(config.train)

    log.info(
        "Loading VLA: config=%s, checkpoint=%s",
        config.train.vla_config_name,
        config.train.vla_checkpoint_dir,
    )
    data_transforms = _resolve_data_transforms(
        config.data_transforms_fn, config.train.vla_config_name
    )

    vla = VLAWrapper(
        checkpoint_path=config.train.vla_checkpoint_dir,
        config_name=config.train.vla_config_name,
        device="cuda",
        data_transforms=data_transforms,
    )

    trainer = RLTokenTrainer(config.train, device="cuda")

    log.info("Loading demo dataset: %s", config.repo_id)
    data_loader = build_data_loader(
        openpi_config_name=config.train.vla_config_name,
        repo_id=config.repo_id,
        batch_size=config.train.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        data_transforms=data_transforms,
    )

    data_iter = iter(data_loader)
    trainer.train(vla, data_iter, log_fn=rl_logger.log)
    rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
