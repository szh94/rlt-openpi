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

    # With 3-camera override:
    uv run python scripts/train_rl_token_jax.py \\
        --train.vla-checkpoint-dir /path/to/orbax_checkpoint \\
        --repo-id local/stack_the_blocks \\
        --data-transforms-fn rlt_openpi.policies.franka.config.three_camera_droid
"""

from __future__ import annotations

import dataclasses
import importlib
import os
# ---------------------------------------------------------------------------
# Limit JAX GPU memory BEFORE any JAX import, so PyTorch has room.
# JAX pre-allocates ~90% of GPU by default — this knocks it down to ~35%.
# Set via CLI for finer control, e.g. XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
# ---------------------------------------------------------------------------
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
# Disable preallocation — JAX only allocates what it needs, when it needs it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import multiprocessing
import typing

import jax
import jax.numpy as jnp
import numpy as np
import torch
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from openpi.models.model import Observation
from openpi.training.config import get_config
from openpi.training.data_loader import create_torch_dataset, transform_dataset
import openpi.transforms as _transforms

from rlt_openpi.training.config import RLTokenTrainConfig
from rlt_openpi.training.rl_token_trainer import RLTokenTrainer
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.jax_vla_wrapper import JaxVLAWrapper

@dataclasses.dataclass
class TrainConfig:
    """Top-level config for Stage 1 training with JAX VLA."""

    train: RLTokenTrainConfig = dataclasses.field(default_factory=RLTokenTrainConfig)
    """RL token trainer hyperparameters."""

    repo_id: str = "local/stack_the_blocks"
    """LeRobot dataset repo ID (local or HuggingFace)."""

    data_transforms_fn: str | None = None
    """Dotted import path to a ``(ModelConfig) -> transforms.Group`` factory."""

    num_workers: int = 4
    """DataLoader worker processes."""


# ------------------------------------------------------------------
# Inline helpers
# ------------------------------------------------------------------


def _numpy_collate(items):
    """Collate batch elements into numpy arrays (no torch)."""
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _patch_repack_action_key(data_config, action_key: str):
    """Rewrite the repack transform so ``"actions"`` reads from *action_key*."""
    new_inputs = []
    for t in data_config.repack_transforms.inputs:
        if isinstance(t, _transforms.RepackTransform) and "actions" in t.structure:
            patched = dict(t.structure)
            patched["actions"] = action_key
            t = _transforms.RepackTransform(patched)
        new_inputs.append(t)
    repack = _transforms.Group(inputs=new_inputs)
    return dataclasses.replace(data_config, repack_transforms=repack)


def _resolve_data_transforms(dotted_path: str | None, openpi_config_name: str):
    """Dynamically import and call a data-transforms factory."""
    if dotted_path is None:
        return None

    module_path, func_name = dotted_path.rsplit(".", 1)
    factory_fn = getattr(importlib.import_module(module_path), func_name)
    return factory_fn(get_config(openpi_config_name).model)


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

    data_transforms = _resolve_data_transforms(
        config.data_transforms_fn, config.train.vla_config_name
    )

    print("[1/4] Loading JAX VLA model...")
    vla = JaxVLAWrapper(
        checkpoint_dir=config.train.vla_checkpoint_dir,
        config_name=config.train.vla_config_name,
        device="cuda",
        data_transforms=data_transforms,
    )
    print("  JAX VLA model loaded successfully.")

    print("[2/4] Creating RL token trainer...")
    trainer = RLTokenTrainer(config.train, device="cuda")
    rl_logger = Logger.from_train_config(config.train)
    print("  Trainer created (RLTokenModel + optimizer).")

    print("[3/4] Loading demonstration dataset...")

    # Build data config from OpenPI (same as build_data_loader).
    openpi_config = get_config(config.train.vla_config_name)
    data_config = openpi_config.data.create(openpi_config.assets_dirs, openpi_config.model)
    data_config = dataclasses.replace(data_config, repo_id=config.repo_id)

    # Auto-detect action column name.
    meta = LeRobotDatasetMetadata(config.repo_id)
    if "action" in meta.features and "actions" not in meta.features:
        data_config = dataclasses.replace(data_config, action_sequence_keys=("action",))
        data_config = _patch_repack_action_key(data_config, "action")

    if data_transforms is not None:
        print("Overriding data_transforms with custom Group")
        data_config = dataclasses.replace(data_config, data_transforms=data_transforms)

    # Create dataset + transforms (OpenPI pipeline).
    dataset = create_torch_dataset(data_config, openpi_config.model.action_horizon, openpi_config.model)
    dataset = transform_dataset(dataset, data_config)

    # Torch DataLoader as batching engine only — _numpy_collate keeps data as numpy.
    mp_context = multiprocessing.get_context("spawn") if config.num_workers > 0 else None
    raw_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, dataset),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=config.num_workers > 0,
        collate_fn=_numpy_collate,
        drop_last=True,
    )

    # numpy→JAX, matching OpenPI's TorchDataLoader.__iter__:
    #   numpy batch → jnp.asarray → Observation.from_dict
    def _jax_iter(loader):
        while True:
            for batch in loader:
                batch = jax.tree.map(jnp.asarray, batch)
                yield Observation.from_dict(batch), batch["actions"]

    data_loader = _jax_iter(raw_loader)
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
