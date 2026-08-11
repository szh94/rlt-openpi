"""Stage 2 with JAX VLA: Online RL training with frozen VLA + RL token.

Same as ``train_online_rl.py`` but loads the native JAX Pi0 model via
:class:`JaxVLAWrapper` instead of the PyTorch port.

Key difference: Skips VLA weight restoration from the Stage 1 checkpoint
because JAX NNX models do not use ``load_state_dict``.

Usage::

    uv run python scripts/train_online_rl_jax.py --help
    uv run python scripts/train_online_rl_jax.py \\
        --vla-config-name pi0_aloha_sim \\
        --vla-checkpoint-dir /path/to/orbax_checkpoint \\
        --rl-token-checkpoint /path/to/rl_token.pt \\
        --env-factory rlt_openpi.envs.sim.sim_env.make_sim_env
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import multiprocessing
import typing
import warnings

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

from rlt_openpi.envs.factory import make_env, make_intervention
from rlt_openpi.envs.intervention import InterventionManager
from rlt_openpi.envs.obs_source import make_obs_source
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.online_rl_trainer import OnlineRLTrainer
from rlt_openpi.utils.checkpoint import load_rl_token_model
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.jax_vla_wrapper import JaxVLAWrapper

# 屏蔽 lerobot 内部 torch.tensor(tensor) 触发的无害 UserWarning
# ("To copy construct from a tensor...")
warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")


def _resolve_data_transforms(dotted_path: str | None, openpi_config_name: str):
    """Dynamically import and call a data-transforms factory (same as Stage 1)."""
    if dotted_path is None:
        return None

    module_path, func_name = dotted_path.rsplit(".", 1)
    factory_fn = getattr(importlib.import_module(module_path), func_name)
    return factory_fn(get_config(openpi_config_name).model)


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


def _build_jax_data_iter(
    vla_config_name: str,
    repo_id: str,
    batch_size: int,
    *,
    num_workers: int = 4,
    data_transforms: _transforms.Group | None = None,
    output_action_dim: int | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
):
    """Build a JAX-native actor-pretraining iterator.

    Uses ``jnp.asarray`` (not ``torch.as_tensor``) so that
    ``Observation.from_dict`` goes through the numpy path, which keeps
    images in NHWC format that JAX models expect.  Each yielded action
    target is the dataset's future action sequence, unnormalized back to
    the Pi model action space.  This is intentionally the same action space
    as ``JaxVLAWrapper`` with ``output_action_dim`` (Unnormalize + slice).

    OpenPI's ALOHA input transforms use ``jnp.array``.  They therefore must
    run in the main process: spawning PyTorch workers would create additional
    JAX CUDA contexts, each of which can reserve GPU memory and cause OOM.
    """
    openpi_config = get_config(vla_config_name)
    data_config = openpi_config.data.create(openpi_config.assets_dirs, openpi_config.model)
    data_config = dataclasses.replace(data_config, repo_id=repo_id)
    # Some local JAX configs deliberately omit norm stats and rely on the
    # checkpoint assets at inference time.  Reuse those exact checkpoint stats
    # for dataset transforms so pretraining sees the same input/action space.
    if norm_stats is not None:
        data_config = dataclasses.replace(data_config, norm_stats=norm_stats)

    # Auto-detect action column name.
    meta = LeRobotDatasetMetadata(repo_id)
    if "action" in meta.features and "actions" not in meta.features:
        data_config = dataclasses.replace(data_config, action_sequence_keys=("action",))
        data_config = _patch_repack_action_key(data_config, "action")

    if data_transforms is not None:
        print("Overriding data_transforms with custom Group")
        data_config = dataclasses.replace(data_config, data_transforms=data_transforms)

    dataset = create_torch_dataset(data_config, openpi_config.model.action_horizon, openpi_config.model)
    print(
        "[Data] Actor pretrain dataset: "
        f"{len(dataset):,} observation/action-window samples "
        f"(action horizon={openpi_config.model.action_horizon})"
    )
    dataset = transform_dataset(dataset, data_config)

    if num_workers > 0:
        print(
            "[Data] JAX actor pretraining forces num_workers=0 to prevent "
            "DataLoader workers from creating extra CUDA contexts."
        )
        num_workers = 0

    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    raw_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_numpy_collate,
        drop_last=True,
    )

    # Dataset samples have already passed OpenPI's input transforms, including
    # Normalize and PadStatesAndActions.  Undo just normalization to obtain
    # actions in the same model/robot space used by JaxVLAWrapper's reference
    # actions; do not apply data_transforms.outputs here.
    unnormalize_actions = _transforms.Unnormalize(
        data_config.norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )
    target_dim = output_action_dim or openpi_config.model.action_dim

    # numpy → JAX, matching OpenPI's TorchDataLoader.__iter__:
    #   numpy batch → jnp.asarray → Observation.from_dict
    def _jax_iter(loader):
        while True:
            for batch in loader:
                batch = jax.tree.map(jnp.asarray, batch)
                # Unnormalize is strict: this checkpoint's stats select both
                # state and actions, so provide both even though only actions
                # are retained as the demo target.
                targets = unnormalize_actions(
                    {"state": batch["state"], "actions": batch["actions"]}
                )["actions"]
                yield Observation.from_dict(batch), targets[..., :target_dim]

    return _jax_iter(raw_loader)


def main(config: OnlineRLTrainConfig) -> None:
    """Run online RL training with JAX VLA (Stage 2, Algorithm 1)."""
    # Set up logger
    rl_logger = Logger.from_train_config(config)

    # Resolve data transforms (same pattern as Stage 1 JAX)
    data_transforms = _resolve_data_transforms(
        config.data_transforms_fn, config.vla_config_name
    )

    # Load frozen JAX VLA
    vla = JaxVLAWrapper(
        checkpoint_dir=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device="cuda",
        output_action_dim=config.action_dim,
        data_transforms=data_transforms,
    )

    # Load frozen RL token model from Stage 1
    rl_token_model = load_rl_token_model(config.rl_token_checkpoint, device="cuda")

    # NOTE: VLA weight restoration from Stage 1 checkpoint is SKIPPED.
    # The JAX NNX model does not use load_state_dict, and Stage 1 with
    # JaxVLAWrapper cannot fine-tune the VLA (vla_finetune_alpha must be 0).
    # The base JAX VLA loaded above is used as-is.

    # Build actor pre-training data independently of --obs-source.  The
    # dataset provides current observations plus future action sequences;
    # it is only used for this pre-warmup supervised phase.
    pretrain_data_iter = None
    if config.repo_id and config.actor_pretrain_steps > 0:
        print(f"[Data] Building actor pretrain data loader: {config.repo_id}")
        pretrain_data_iter = _build_jax_data_iter(
            vla_config_name=config.vla_config_name,
            repo_id=config.repo_id,
            batch_size=config.actor_pretrain_batch_size,
            num_workers=config.num_workers,
            data_transforms=data_transforms,
            output_action_dim=config.action_dim,
            norm_stats=vla.norm_stats,
        )

    # Create trainer
    trainer = OnlineRLTrainer(
        config=config,
        vla=vla,
        rl_token_model=rl_token_model,
        device="cuda",
    )

    # Resume from checkpoint if provided
    if config.resume_checkpoint:
        print(f"Resuming from checkpoint: {config.resume_checkpoint}")
        trainer.load(config.resume_checkpoint)

    # Create environment via pluggable factory.
    if not config.env_factory:
        print(
            "[ERROR] --env-factory is required. Provide a Python import path to an env factory function."
        )
        raise SystemExit(1)

    env_extra_kwargs = json.loads(config.env_kwargs)

    # 黑盒 obs 来源：除 "robot" 外（mock/dataset），构建 ObsSource 并注入 env 工厂，
    # 工厂会跳过机器人初始化，obs 全部来自该黑盒。
    obs_source = None
    if config.obs_source and config.obs_source != "robot":
        obs_kwargs: dict = {"image_size": tuple(env_extra_kwargs.get("image_size", (224, 224)))}
        if "camera_names" in env_extra_kwargs:
            obs_kwargs["camera_names"] = env_extra_kwargs["camera_names"]
        if config.obs_source == "dataset":
            if not config.repo_id:
                print("[ERROR] --obs-source dataset requires --repo-id (LeRobot 数据集 ID)")
                raise SystemExit(1)
            obs_kwargs["repo_id"] = config.repo_id
            obs_kwargs["vla_config_name"] = config.vla_config_name
            obs_kwargs["task_prompt"] = config.task_prompt
        else:  # mock
            obs_kwargs["task_prompt"] = config.task_prompt
        obs_source = make_obs_source(config.obs_source, **obs_kwargs)
        print(
            f"[ObsSource] built {type(obs_source).__name__} from --obs-source={config.obs_source}",
        )

    env_kwargs = {
        "action_dim": config.action_dim,
        "chunk_length": config.chunk_length,
        "task_prompt": config.task_prompt,
        "max_episode_chunks": config.max_episode_chunks,
        "dry_run": config.dry_run,
        **env_extra_kwargs,
    }
    if obs_source is not None:
        env_kwargs["obs_source"] = obs_source
    env = make_env(config.env_factory, **env_kwargs)
    print(
        f"Environment created: action_dim={env.action_dim}, chunk_length={env.chunk_length}"
    )

    # Create intervention manager (VR teleoperation, etc.) if specified.
    intervention_mgr: InterventionManager | None = None
    if config.intervention_factory:
        intervention_mgr = make_intervention(config.intervention_factory, env=env)
        print(f"Intervention manager created via {config.intervention_factory}")

    trainer.train(
        env=env,
        intervention_mgr=intervention_mgr,
        log_fn=rl_logger.log,
        pretrain_data_iter=pretrain_data_iter,
    )

    rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(OnlineRLTrainConfig))
