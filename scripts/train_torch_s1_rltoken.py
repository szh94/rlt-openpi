"""Stage 1: Train the RL token encoder-decoder on demonstration data.

Trains the information-bottleneck encoder-decoder that compresses
variable-length VLA prefix embeddings z_{1:M} into a single RL token
z_rl via masked-MSE reconstruction loss.

Only frozen-VLA mode is supported. VLA joint training has been disabled.

The data pipeline delegates entirely to OpenPI's transform chain so
that normalisation, camera layout, and action chunking exactly match
the pretrained model.

Usage::

    # Default (2-camera, frozen VLA):
    uv run python scripts/train_torch_s1_rltoken.py \\
        --train.vla-checkpoint-dir /path/to/model.safetensors \\
        --repo-id local/stack_the_blocks

    # With 3-camera override:
    uv run python scripts/train_torch_s1_rltoken.py \\
        --train.vla-checkpoint-dir /path/to/model.safetensors \\
        --repo-id local/stack_the_blocks \\
        --data-transforms-fn rlt_openpi.policies.aloha.config.aloha_data_transforms
"""

from __future__ import annotations

import dataclasses

import tyro

from rlt_openpi.training.config import RLTokenTrainConfig
from rlt_openpi.training.data_loader import build_torch_data_loader, resolve_data_transforms
from rlt_openpi.training.trainer_s1_rltoken import RLTokenTrainer
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.vla_wrapper import VLAWrapper

@dataclasses.dataclass
class TrainConfig:
    """Top-level config for Stage 1 training.

    Wraps :class:`RLTokenTrainConfig` (architecture + training hypers)
    and adds dataset / data-transform settings that live outside the
    trainer.
    """

    train: RLTokenTrainConfig = dataclasses.field(default_factory=RLTokenTrainConfig)
    """RL token trainer hyperparameters."""

    repo_id: str = "local/stack_the_blocks"
    """LeRobot dataset repo ID (local or HuggingFace)."""

    data_transforms_fn: str | None = None
    """Dotted import path to a ``(ModelConfig) -> transforms.Group``
    factory that overrides the OpenPI config's default data transforms.
    Example: ``rlt_openpi.policies.aloha.config.aloha_data_transforms``."""

    num_workers: int = 4
    """DataLoader worker processes."""


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main(config: TrainConfig) -> None:
    print("=" * 60)
    print("Stage 1: RL Token Encoder-Decoder Training")
    print("=" * 60)
    print(f"  VLA config:      {config.train.vla_config_name}")
    print(f"  VLA checkpoint:  {config.train.vla_checkpoint_dir}")
    print(f"  Dataset:         {config.repo_id}")
    print(f"  Batch size:      {config.train.batch_size}")
    print(f"  Train steps:     {config.train.num_train_steps}")
    print(f"  Save dir:        {config.train.save_dir}")
    # VLA joint training disabled — VLA is always frozen.
    # print(f"  VLA finetune:    alpha={config.train.vla_finetune_alpha}")
    print("-" * 60)

    data_transforms = resolve_data_transforms(
        config.data_transforms_fn, config.train.vla_config_name
    )

    print("[1/4] Loading VLA model...")
    vla = VLAWrapper(
        checkpoint_path=config.train.vla_checkpoint_dir,
        config_name=config.train.vla_config_name,
        device="cuda",
        data_transforms=data_transforms,
    )
    print("VLA model loaded successfully.\n")

    print("[2/4] Creating RL token trainer...")
    trainer = RLTokenTrainer(config.train, device="cuda")
    rl_logger = Logger.from_train_config(config.train)
    print("  Trainer created (RLTokenModel + optimizer).")

    print("[3/4] Loading demonstration dataset...")
    data_loader = build_torch_data_loader(
        openpi_config_name=config.train.vla_config_name,
        repo_id=config.repo_id,
        batch_size=config.train.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        data_transforms=data_transforms,
    )
    print("  Data loader ready.")

    print("[4/4] Starting training loop...")
    print("-" * 60)
    trainer.train(vla, iter(data_loader), log_fn=rl_logger.log)

    print("-" * 60)
    print("Training complete.")
    print("=" * 60)
    rl_logger.finish()

if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
