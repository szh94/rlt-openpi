"""Standalone actor BC pre-training with a frozen JAX VLA and RL token."""

from __future__ import annotations

import warnings

import tyro

from rlt_openpi.training.trainer_s2_actorPretrain import ActorPretrainTrainer
from rlt_openpi.training.config import ActorPretrainConfig
from rlt_openpi.training.data_loader import build_jax_data_loader
from rlt_openpi.utils.checkpoint import load_rl_token_model
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.jax_vla_wrapper import JaxVLAWrapper


warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")


def main(config: ActorPretrainConfig) -> None:
    """Pre-train the actor, save ``actor_pretrain.pt``, and exit."""
    if not config.repo_id:
        raise ValueError("--repo-id is required for actor pre-training")
    if config.steps <= 0:
        raise ValueError("--steps must be greater than 0")
    if config.save_every <= 0:
        raise ValueError("--save-every must be greater than 0")
    if not config.vla_checkpoint_dir:
        raise ValueError("--vla-checkpoint-dir is required")
    if not config.rl_token_checkpoint:
        raise ValueError("--rl-token-checkpoint is required")

    rl_logger = Logger.from_train_config(config)
    try:
        vla = JaxVLAWrapper(
            checkpoint_dir=config.vla_checkpoint_dir,
            config_name=config.vla_config_name,
            device="cuda",
            output_action_dim=config.action_dim,
        )
        rl_token_model = load_rl_token_model(config.rl_token_checkpoint, device="cuda")

        print(f"[Data] Building actor pretrain data loader: {config.repo_id}")
        pretrain_data_iter = build_jax_data_loader(
            openpi_config_name=config.vla_config_name,
            repo_id=config.repo_id,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            output_action_dim=config.action_dim,
            norm_stats=vla.norm_stats,
            action_target_space="model",
            dataset_label="Actor pretrain dataset",
            include_raw_state=True,
            include_normalized_actions=True,
        )

        trainer = ActorPretrainTrainer(
            config=config,
            vla=vla,
            rl_token_model=rl_token_model,
            device="cuda",
        )
        checkpoint_path = trainer.train(pretrain_data_iter, log_fn=rl_logger.log)
        print(f"[Actor Pretrain] Pipeline complete: {checkpoint_path}")
    finally:
        rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(ActorPretrainConfig))
