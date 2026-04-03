"""Stage 2: Online RL training with frozen VLA + RL token (Algorithm 1).

Usage:
    uv run python scripts/train_online_rl.py --help
    uv run python scripts/train_online_rl.py --vla-config-name pi0_aloha_sim \
        --vla-checkpoint-dir /path/to/vla.safetensors \
        --rl-token-checkpoint /path/to/rl_token.pt
"""

from __future__ import annotations

import logging

import torch
import tyro

from rlt_openpi.models.rl_token import RLTokenModel
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.online_rl_trainer import OnlineRLTrainer
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


def load_rl_token_model(
    ckpt_path: str,
    config: OnlineRLTrainConfig,
    device: str = "cuda",
) -> RLTokenModel:
    """Load a trained RL token model from a Stage 1 checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_config = ckpt["config"]
    model = RLTokenModel(
        embedding_dim=saved_config.embedding_dim,
        encoder_layers=saved_config.encoder_layers,
        encoder_heads=saved_config.encoder_heads,
        decoder_layers=saved_config.decoder_layers,
        decoder_heads=saved_config.decoder_heads,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    log.info("Loaded RL token model from %s (step %d)", ckpt_path, ckpt["step"])
    return model


def main(config: OnlineRLTrainConfig) -> None:
    """Run online RL training (Stage 2, Algorithm 1)."""
    log.info("Stage 2 config: %s", config)

    # Set up logger
    rl_logger = Logger.from_train_config(config)

    # Load frozen VLA
    log.info("Loading VLA: config=%s, checkpoint=%s", config.vla_config_name, config.vla_checkpoint_dir)
    vla = VLAWrapper(
        checkpoint_path=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device="cuda",
    )

    # Load frozen RL token model from Stage 1
    log.info("Loading RL token model from %s", config.rl_token_checkpoint)
    rl_token_model = load_rl_token_model(config.rl_token_checkpoint, config, device="cuda")

    # Create trainer
    trainer = OnlineRLTrainer(  # noqa: F841
        config=config,
        vla=vla,
        rl_token_model=rl_token_model,
        device="cuda",
    )

    # Create environment — placeholder, must be wired per-task.
    # Example for LIBERO:
    #   from rlt_openpi.rollout.env_wrapper import RLTEnv
    #   import gymnasium as gym
    #   raw_env = gym.make("libero/...", ...)
    #   env = RLTEnv(raw_env, action_dim=config.action_dim, chunk_length=config.chunk_length)
    #   trainer.train(env=env, log_fn=rl_logger.log)

    log.info(
        "Trainer ready. Provide an RLTEnv environment instance to begin training. See script comments for examples."
    )

    # Placeholder: when a real env is wired, uncomment:
    # trainer.train(env=env, log_fn=rl_logger.log)

    rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(OnlineRLTrainConfig))
