"""Standalone actor BC pre-training with a frozen JAX VLA and RL token."""

from __future__ import annotations

import warnings
from pathlib import Path

import tyro

from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.data_loader import build_jax_data_loader, resolve_data_transforms
from rlt_openpi.training.online_rl_trainer import OnlineRLTrainer
from rlt_openpi.utils.checkpoint import load_rl_token_model
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.jax_vla_wrapper import JaxVLAWrapper


warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")


def main(config: OnlineRLTrainConfig) -> None:
    """Pre-train the actor, save ``actor_pretrain.pt``, and exit."""
    if not config.repo_id:
        raise ValueError("--repo-id is required for actor pre-training")
    if config.actor_pretrain_steps <= 0:
        raise ValueError("--actor-pretrain-steps must be greater than 0")
    if not config.vla_checkpoint_dir:
        raise ValueError("--vla-checkpoint-dir is required")
    if not config.rl_token_checkpoint:
        raise ValueError("--rl-token-checkpoint is required")

    rl_logger = Logger.from_train_config(config)
    try:
        data_transforms = resolve_data_transforms(
            config.data_transforms_fn,
            config.vla_config_name,
        )

        vla = JaxVLAWrapper(
            checkpoint_dir=config.vla_checkpoint_dir,
            config_name=config.vla_config_name,
            device="cuda",
            output_action_dim=config.action_dim,
            data_transforms=data_transforms,
        )
        rl_token_model = load_rl_token_model(config.rl_token_checkpoint, device="cuda")

        print(f"[Data] Building actor pretrain data loader: {config.repo_id}")
        pretrain_data_iter = build_jax_data_loader(
            openpi_config_name=config.vla_config_name,
            repo_id=config.repo_id,
            batch_size=config.actor_pretrain_batch_size,
            num_workers=config.num_workers,
            data_transforms=data_transforms,
            output_action_dim=config.action_dim,
            norm_stats=vla.norm_stats,
            action_target_space="model",
            dataset_label="Actor pretrain dataset",
        )

        # Actor-only training never uses replay data. Avoid allocating the
        # full online-RL replay buffer when constructing the shared trainer.
        config.buffer_capacity = 1
        trainer = OnlineRLTrainer(
            config=config,
            vla=vla,
            rl_token_model=rl_token_model,
            device="cuda",
        )
        trainer.pretrain_actor(pretrain_data_iter, log_fn=rl_logger.log)
        checkpoint_path = Path(config.save_dir) / config.run_name / "actor_pretrain.pt"
        print(f"[Actor Pretrain] Pipeline complete: {checkpoint_path}")
    finally:
        rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(OnlineRLTrainConfig))
