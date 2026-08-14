"""Standalone actor BC pre-training with a frozen JAX VLA and RL token."""

from __future__ import annotations

import warnings

import openpi.training.data_loader as openpi_data_loader
import openpi.transforms as openpi_transforms
import tyro

from rlt_openpi.training.trainer_s2_actorPretrain import ActorPretrainTrainer
from rlt_openpi.training.config import ActorPretrainConfig
from rlt_openpi.training.data_loader import build_data_config
from rlt_openpi.utils.checkpoint import load_rl_token_model
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.jax_vla_wrapper import JaxVLAWrapper


warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")


def _build_openpi_data_iter(config: ActorPretrainConfig, vla: JaxVLAWrapper):
    """Build actor-pretrain batches with OpenPI's native JAX loader."""
    openpi_config, data_config = build_data_config(
        config.vla_config_name,
        config.repo_id,
        norm_stats=vla.norm_stats,
    )
    if data_config.norm_stats is None:
        raise ValueError("Actor pretraining requires normalization stats")

    data_loader = openpi_data_loader.create_torch_data_loader(
        data_config=data_config,
        model_config=openpi_config.model,
        action_horizon=openpi_config.model.action_horizon,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        framework="jax",
    )
    unnormalize = openpi_transforms.Unnormalize(
        data_config.norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )

    def _iterator():
        for observation, actions in data_loader:
            actions = unnormalize(
                {"state": observation.state, "actions": actions}
            )["actions"]
            yield observation, actions[..., : config.action_dim]

    return _iterator()


def main(config: ActorPretrainConfig) -> None:
    """Pre-train the actor, save ``actor_pretrain.pt``, and exit."""
    if not config.repo_id:
        raise ValueError("--repo-id is required for actor pre-training")
    if config.steps <= 0:
        raise ValueError("--steps must be greater than 0")
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

        print(
            f"[Data] Building actor pretrain data loader with OpenPI: "
            f"repo={config.repo_id} batch_size={config.batch_size} "
            f"num_workers={config.num_workers}"
        )
        pretrain_data_iter = _build_openpi_data_iter(
            config,
            vla)

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
