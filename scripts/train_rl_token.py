"""Stage 1: Train the RL token encoder-decoder on demonstration data.

Usage:
    uv run python scripts/train_rl_token.py --help
    uv run python scripts/train_rl_token.py --vla-config-name pi0_aloha_sim --vla-checkpoint-dir /path/to/ckpt
"""

from __future__ import annotations

import logging

import tyro

from rlt_openpi.training.config import RLTokenTrainConfig
from rlt_openpi.training.rl_token_trainer import RLTokenTrainer
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


def main(config: RLTokenTrainConfig) -> None:
    """Train the RL token encoder-decoder (Stage 1)."""
    log.info("Stage 1 config: %s", config)

    # Set up logger
    rl_logger = Logger.from_train_config(config)

    # Load frozen VLA
    log.info("Loading VLA: config=%s, checkpoint=%s", config.vla_config_name, config.vla_checkpoint_dir)
    vla = VLAWrapper(  # noqa: F841
        checkpoint_path=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device="cuda",
    )

    # Create trainer
    trainer = RLTokenTrainer(config, device="cuda")  # noqa: F841

    # Build a dataloader that extracts VLA embeddings from a demo dataset.
    # For now this is a placeholder — a real implementation would iterate
    # over a dataset of observations (e.g., from LIBERO demos or ALOHA
    # teleoperation data) and extract embeddings on the fly.
    #
    # Example with pre-extracted embeddings:
    #   dataloader = iter(torch.utils.data.DataLoader(embedding_dataset, ...))
    #   trainer.train(dataloader, log_fn=rl_logger.log)
    #
    # Example with raw observations:
    #   for obs_batch in obs_dataloader:
    #       metrics = trainer.train_step_from_obs(vla, obs_batch)
    #       rl_logger.log(metrics)

    log.info(
        "Trainer ready. Provide a dataloader of (z, pad_mask) tuples or "
        "raw observations to begin training. See script comments for examples."
    )

    # Placeholder: when a real dataloader is wired, uncomment:
    # trainer.train(dataloader, log_fn=rl_logger.log)

    rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(RLTokenTrainConfig))
