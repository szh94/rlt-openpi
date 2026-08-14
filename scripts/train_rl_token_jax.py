"""Stage 1 with JAX VLA: Train the RL token encoder-decoder on demonstration data.

Same as ``train_rl_token.py`` but loads the native JAX Pi0 model via
:class:`JaxVLAWrapper` instead of the PyTorch port.

**Joint training is not supported** — ``--train.vla-finetune-alpha``
must be 0 (or omitted).  A JAX NNX model cannot be optimised by a
PyTorch optimizer.

Usage::

    # Default (2-camera, frozen VLA):
    uv run python scripts/train_rl_token_jax.py \\
        --train.vla-checkpoint-dir /path/to/orbax_checkpoint \\
        --repo-id local/stack_the_blocks

"""

from __future__ import annotations

import dataclasses
import os

# ---------------------------------------------------------------------------
# Limit JAX GPU memory BEFORE any JAX import, so PyTorch has room.
# JAX pre-allocates ~90% of GPU by default — this knocks it down to ~35%.
# Set via CLI for finer control, e.g. XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
# ---------------------------------------------------------------------------
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
# Disable preallocation — JAX only allocates what it needs, when it needs it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import tyro
import warnings

from rlt_openpi.training.config import RLTokenTrainConfig
from rlt_openpi.training.data_loader import build_jax_data_loader
from rlt_openpi.training.trainer_s1_rltoken import RLTokenTrainer
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.jax_vla_wrapper import JaxVLAWrapper

# 屏蔽 lerobot 内部 torch.tensor(tensor) 触发的无害 UserWarning
# ("To copy construct from a tensor...")
warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask.*")

@dataclasses.dataclass
class TrainConfig:
    """Top-level config for Stage 1 training with JAX VLA."""

    train: RLTokenTrainConfig = dataclasses.field(default_factory=RLTokenTrainConfig)
    """RL token trainer hyperparameters."""

    repo_id: str = "local/stack_the_blocks"
    """LeRobot dataset repo ID (local or HuggingFace)."""

    num_workers: int = 4
    """DataLoader worker processes."""


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main(config: TrainConfig) -> None:
    # Guard: joint training is not supported with JAX VLA.
    if config.train.vla_finetune_alpha > 0:
        raise ValueError(
            f"vla_finetune_alpha={config.train.vla_finetune_alpha} is not supported "
            f"with JaxVLAWrapper. Joint training requires the PyTorch VLAWrapper. "
            f"Use scripts/train_rl_token.py for joint training."
        )

    print("=" * 60)
    print("Stage 1: RL Token Encoder-Decoder Training (JAX VLA)")
    print("=" * 60)
    print(f"  VLA config:      {config.train.vla_config_name}")
    print(f"  VLA checkpoint:  {config.train.vla_checkpoint_dir}")
    print(f"  Dataset:         {config.repo_id}")
    print(f"  Batch size:      {config.train.batch_size}")
    print(f"  Train steps:     {config.train.num_train_steps}")
    print(f"  Save dir:        {config.train.save_dir}")
    # VLA joint training disabled — VLA is always frozen.
    # print(f"  VLA finetune:    alpha={config.train.vla_finetune_alpha} (frozen)")
    print("-" * 60)

    print("[1/4] Loading JAX VLA model...")
    vla = JaxVLAWrapper(
        checkpoint_dir=config.train.vla_checkpoint_dir,
        config_name=config.train.vla_config_name,
        device="cuda",
    )
    print("  JAX VLA model loaded successfully.")

    print("[2/4] Creating RL token trainer...")
    trainer = RLTokenTrainer(config.train, device="cuda")
    rl_logger = Logger.from_train_config(config.train)
    print("  Trainer created (RLTokenModel + optimizer).")

    print("[3/4] Loading demonstration dataset...")

    data_loader = build_jax_data_loader(
        openpi_config_name=config.train.vla_config_name,
        repo_id=config.repo_id,
        batch_size=config.train.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        norm_stats=vla.norm_stats,
        action_target_space="normalized",
        dataset_label="RL token training dataset",
    )
    print("  Data loader ready.")

    print("[4/4] Starting training loop...")
    print("-" * 60)
    trainer.train(vla, data_loader, log_fn=rl_logger.log)

    print("-" * 60)
    print("Training complete.")
    print("=" * 60)
    rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
