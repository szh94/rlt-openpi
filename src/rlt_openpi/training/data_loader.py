"""RLT data adapters built on top of OpenPI's dataset pipeline.

Reuses OpenPI's full transform chain so that normalization, action
chunking, and camera layout exactly match the pretrained VLA config.

This module owns the small amount of adaptation that is specific to RLT:

* selecting a dataset at runtime instead of from an OpenPI train config;
* accepting both LeRobot ``action`` and OpenPI ``actions`` columns;
* overriding data transforms and checkpoint-provided normalization stats;
* producing either PyTorch-native or JAX-native ``Observation`` batches.

Dataset reading and the transform chain themselves remain owned by OpenPI.
"""

from __future__ import annotations

import dataclasses
import importlib
import multiprocessing
import os
import typing
from collections.abc import Iterator
from typing import Any, Literal

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
from openpi.models.model import Observation
from openpi.training.config import get_config
from openpi.training.data_loader import (
    create_torch_dataset,
    transform_dataset,
)
import openpi.transforms as _transforms


def _numpy_collate(items):
    """Collate without converting arrays to Torch tensors."""
    return jax.tree.map(
        lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items
    )


def _jax_worker_init_fn(worker_id: int) -> None:
    """Prevent JAX DataLoader workers from preallocating GPU memory."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def patch_repack_action_key(data_config, action_key: str):
    """Rewrite the repack transform so `"actions"` reads from *action_key*."""
    new_inputs = []
    for t in data_config.repack_transforms.inputs:
        if isinstance(t, _transforms.RepackTransform) and "actions" in t.structure:
            patched = dict(t.structure)
            patched["actions"] = action_key
            t = _transforms.RepackTransform(patched)
        new_inputs.append(t)
    repack = _transforms.Group(inputs=new_inputs)
    return dataclasses.replace(data_config, repack_transforms=repack)


def resolve_data_transforms(
    dotted_path: str | None,
    openpi_config_name: str,
) -> _transforms.Group | None:
    """Resolve a ``(ModelConfig) -> transforms.Group`` factory."""
    if dotted_path is None:
        return None

    module_path, func_name = dotted_path.rsplit(".", 1)
    factory_fn = getattr(importlib.import_module(module_path), func_name)
    return factory_fn(get_config(openpi_config_name).model)


def build_data_config(
    openpi_config_name: str,
    repo_id: str,
    *,
    data_transforms: _transforms.Group | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
):
    """Create an OpenPI data config with the RLT dataset adaptations."""
    openpi_config = get_config(openpi_config_name)
    data_config = openpi_config.data.create(openpi_config.assets_dirs, openpi_config.model)
    data_config = dataclasses.replace(data_config, repo_id=repo_id)

    # JAX checkpoints are the authoritative source of normalization stats for
    # configs which deliberately do not embed them.
    if norm_stats is not None:
        data_config = dataclasses.replace(data_config, norm_stats=norm_stats)

    # Standard LeRobot datasets use "action" while some OpenPI conversions use
    # "actions". Keep this compatibility policy in one place.
    metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    if "action" in metadata.features and "actions" not in metadata.features:
        data_config = dataclasses.replace(data_config, action_sequence_keys=("action",))
        data_config = patch_repack_action_key(data_config, "action")

    if data_transforms is not None:
        print("Overriding data_transforms with custom Group")
        data_config = dataclasses.replace(data_config, data_transforms=data_transforms)

    return openpi_config, data_config


def _build_transformed_dataset(
    openpi_config_name: str,
    repo_id: str,
    *,
    data_transforms: _transforms.Group | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
    include_raw_state: bool = False,
    include_raw_observation: bool = False,
):
    if include_raw_state and include_raw_observation:
        raise ValueError(
            "include_raw_state and include_raw_observation are mutually exclusive"
        )
    openpi_config, data_config = build_data_config(
        openpi_config_name,
        repo_id,
        data_transforms=data_transforms,
        norm_stats=norm_stats,
    )
    dataset = create_torch_dataset(
        data_config,
        openpi_config.model.action_horizon,
        openpi_config.model,
    )
    if include_raw_observation:
        transformed_dataset = _TransformedDatasetWithRawObservation(
            dataset, data_config
        )
    elif include_raw_state:
        transformed_dataset = _TransformedDatasetWithRawState(dataset, data_config)
    else:
        transformed_dataset = transform_dataset(dataset, data_config)
    return openpi_config, data_config, transformed_dataset, len(dataset)


class _TransformedDatasetWithRawState:
    """Apply OpenPI transforms while retaining the state from the same sample."""

    def __init__(self, dataset, data_config) -> None:
        self._dataset = dataset
        norm_stats = {}
        if data_config.repo_id != "fake":
            if data_config.norm_stats is None:
                raise ValueError(
                    "Normalization stats not found. "
                    "Make sure to run scripts/compute_norm_stats.py."
                )
            norm_stats = data_config.norm_stats
        self._transform = _transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                _transforms.Normalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *data_config.model_transforms.inputs,
            ]
        )

    def __getitem__(self, index):
        raw = self._dataset[index]
        raw_state = np.asarray(raw["observation.state"], dtype=np.float32).copy()
        return self._transform(raw), raw_state

    def __len__(self) -> int:
        return len(self._dataset)


class _TransformedDatasetWithRawObservation(_TransformedDatasetWithRawState):
    """Apply transforms while retaining raw state and the first raw image."""

    def __getitem__(self, index):
        raw = self._dataset[index]
        first_image = next(
            (
                (key, np.asarray(value).copy())
                for key, value in raw.items()
                if isinstance(key, str)
                and key.startswith("observation.images.")
                and value is not None
            ),
            None,
        )
        raw_observation = {
            "state": np.asarray(raw["observation.state"], dtype=np.float32).copy(),
            "images": dict([first_image]) if first_image is not None else {},
        }
        return self._transform(raw), raw_observation


def build_torch_data_loader(
    openpi_config_name: str,
    repo_id: str,
    batch_size: int,
    *,
    num_workers: int = 2,
    shuffle: bool = True,
    data_transforms: _transforms.Group | None = None,
):
    """Build a PyTorch data loader using OpenPI's full pipeline.

    Works for any robot/model registered in OpenPI (DROID, ALOHA, Libero,
    etc.).  Returns an infinite iterator yielding
    ``(Observation, actions)`` tuples with correctly normalised,
    tokenised, and padded tensors.

    Args:
        openpi_config_name: Registered OpenPI config name
            (e.g. ``"pi05_aloha"``).
        repo_id: LeRobot dataset repo ID (e.g. ``"local/stack_the_blocks"``).
        batch_size: Global batch size.
        num_workers: DataLoader workers.
        shuffle: Whether to shuffle.
        data_transforms: Optional override for the config's default
            ``data_transforms``.  If ``None``, the transforms from the
            OpenPI config are used as-is.  Provide a custom
            :class:`~openpi.transforms.Group` to change observation /
            action processing (e.g. different camera layouts).

    Yields:
        ``(observation, actions)`` –
        ``observation`` is an :class:`Observation` with torch tensors;
        ``actions`` is ``[B, action_horizon, action_dim]`` float32.
    """
    _, _, dataset, _ = _build_transformed_dataset(
        openpi_config_name,
        repo_id,
        data_transforms=data_transforms,
    )

    mp_context = None
    if num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")

    torch_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_numpy_collate,
        drop_last=True,
    )

    return _InfiniteLoader(torch_loader)


def build_jax_data_loader(
    openpi_config_name: str,
    repo_id: str,
    batch_size: int,
    *,
    num_workers: int = 2,
    shuffle: bool = True,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
    action_target_space: Literal["normalized", "model"] = "normalized",
    output_action_dim: int | None = None,
    dataset_label: str | None = None,
    include_raw_state: bool = False,
    include_raw_observation: bool = False,
) -> Iterator[tuple[Observation, Any] | tuple[Observation, Any, Any]]:
    """Build an infinite iterator whose observations are native JAX arrays.

    OpenPI's input transforms run before batching. Batches are deliberately
    collated as NumPy and then converted with :func:`jax.numpy.asarray`; this
    makes ``Observation.from_dict`` take its JAX/NumPy path and preserves the
    NHWC image layout expected by native JAX Pi models.

    ``action_target_space="normalized"`` returns the transformed dataset
    actions. ``"model"`` undoes only normalization, producing targets in the
    same action space as ``JaxVLAWrapper`` inference, and optionally slices the
    final dimension with ``output_action_dim``.

    If ``include_raw_state`` is true, each yielded item additionally contains
    the untransformed ``observation.state`` loaded from the exact same dataset
    sample as its observation and action.

    If ``include_raw_observation`` is true, the additional item contains the
    untransformed state and first image field from that same sample. It is
    mutually exclusive with ``include_raw_state``.
    """
    if action_target_space not in {"normalized", "model"}:
        raise ValueError(f"Unsupported action_target_space: {action_target_space!r}")

    openpi_config, data_config, dataset, dataset_size = _build_transformed_dataset(
        openpi_config_name,
        repo_id,
        norm_stats=norm_stats,
        include_raw_state=include_raw_state,
        include_raw_observation=include_raw_observation,
    )
    if dataset_label:
        print(
            f"[Data] {dataset_label}: {dataset_size:,} observation/action-window samples "
            f"(action horizon={openpi_config.model.action_horizon})"
        )

    mp_context = None
    if num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")

    raw_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_numpy_collate,
        worker_init_fn=_jax_worker_init_fn,
        drop_last=True,
    )

    unnormalize = None
    if action_target_space == "model":
        if data_config.norm_stats is None:
            raise ValueError("action_target_space='model' requires normalization stats")
        unnormalize = _transforms.Unnormalize(
            data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )

    target_dim = (
        output_action_dim
        if output_action_dim is not None
        else openpi_config.model.action_dim
    )

    def _iterator():
        include_raw = include_raw_state or include_raw_observation
        while True:
            for loader_batch in raw_loader:
                loader_batch = jax.tree.map(jnp.asarray, loader_batch)
                if include_raw:
                    batch, raw_data = loader_batch
                else:
                    batch = loader_batch
                actions = batch["actions"]
                if unnormalize is not None:
                    # Unnormalize is strict for stats selecting both fields.
                    actions = unnormalize(
                        {"state": batch["state"], "actions": actions}
                    )["actions"]
                    actions = actions[..., :target_dim]
                if include_raw:
                    yield Observation.from_dict(batch), actions, raw_data
                else:
                    yield Observation.from_dict(batch), actions

    return _iterator()


class _InfiniteLoader:
    """Wraps a torch DataLoader as an infinite iterator of ``(Observation, actions)``."""

    def __init__(self, loader):
        self._loader = loader

    @staticmethod
    def _to_float32(x):
        t = torch.as_tensor(x)
        if t.is_floating_point():
            t = t.float()
        return t

    def __iter__(self):
        while True:
            for batch in self._loader:
                batch = jax.tree.map(self._to_float32, batch)
                yield Observation.from_dict(batch), batch["actions"]
