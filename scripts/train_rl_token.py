"""Stage 1: Train the RL token encoder-decoder on demonstration data.

Supports two modes:
  - Frozen VLA (alpha=0): Only trains the RL token encoder-decoder.
  - Joint training (alpha>0): Simultaneously fine-tunes the VLA and
    trains the RL token encoder-decoder with stop-gradient separation.

Usage:
    uv run python scripts/train_rl_token.py --help

    # Joint training (recommended):
    uv run python scripts/train_rl_token.py \
        --train.vla-config-name pi05_droid_finetune \
        --train.vla-checkpoint-dir /path/to/model.safetensors \
        --train.vla-finetune-alpha 1.0 \
        --train.num-train-steps 5000 \
        --dataset.repo-id local/stack_the_blocks

    # Frozen VLA (alpha=0):
    uv run python scripts/train_rl_token.py \
        --train.vla-config-name pi05_droid_finetune \
        --train.vla-checkpoint-dir /path/to/model.safetensors \
        --dataset.repo-id local/stack_the_blocks
"""

from __future__ import annotations

import dataclasses
import logging

import tyro
from torch.utils.data import DataLoader

from rlt_openpi.training.demo_dataset import (
    DemoDataset,
    DemoDatasetConfig,
    collate_observation_batch,
)
from rlt_openpi.training.config import RLTokenTrainConfig
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

    dataset: DemoDatasetConfig = dataclasses.field(default_factory=DemoDatasetConfig)
    """Demo dataset config."""

    num_workers: int = 4
    """DataLoader worker processes."""


def main(config: TrainConfig) -> None:
    """Train the RL token encoder-decoder (Stage 1)."""
    log.info("Stage 1 config: %s", config)

    # Set up logger
    rl_logger = Logger.from_train_config(config.train)

    # Load VLA
    log.info(
        "Loading VLA: config=%s, checkpoint=%s",
        config.train.vla_config_name,
        config.train.vla_checkpoint_dir,
    )
    vla = VLAWrapper(
        checkpoint_path=config.train.vla_checkpoint_dir,
        config_name=config.train.vla_config_name,
        device="cuda",
    )

    # Create trainer
    trainer = RLTokenTrainer(config.train, device="cuda")

    # Build dataset and dataloader
    log.info("Loading demo dataset: %s", config.dataset.repo_id)
    dataset = DemoDataset(config.dataset)
    log.info("Dataset size: %d samples", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_observation_batch,
        pin_memory=True,
        drop_last=True,
    )

    # Train (mode auto-selected by alpha)
    data_iter = _infinite_iter(dataloader)
    trainer.train(vla, data_iter, log_fn=rl_logger.log)
    rl_logger.finish()


def _infinite_iter(dataloader: DataLoader):
    """Wrap a DataLoader as an infinite iterator that loops forever."""
    while True:
        yield from dataloader


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
