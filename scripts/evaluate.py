"""Evaluate a trained RL actor on an environment.

Loads the actor from a Stage 2 checkpoint, runs episodes, and reports
success rate and average reward.

Usage:
    uv run python scripts/evaluate.py --help
    uv run python scripts/evaluate.py --checkpoint /path/to/online_rl.pt \
        --vla-config-name pi0_aloha_sim \
        --vla-checkpoint-dir /path/to/vla.safetensors \
        --rl-token-checkpoint /path/to/rl_token.pt \
        --num-episodes 50
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import tyro

from rlt_openpi.models.actor import Actor
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    checkpoint: str = ""
    vla_config_name: str = "pi0_aloha_sim"
    vla_checkpoint_dir: str = ""
    rl_token_checkpoint: str = ""
    num_episodes: int = 50
    device: str = "cuda"


def main(config: EvalConfig) -> None:
    """Run evaluation."""
    log.info("Eval config: %s", config)

    # Load Stage 2 checkpoint
    ckpt = torch.load(config.checkpoint, map_location=config.device, weights_only=False)
    train_config: OnlineRLTrainConfig = ckpt["config"]

    # Load frozen VLA
    vla = VLAWrapper(  # noqa: F841
        checkpoint_path=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device=config.device,
    )

    # Load frozen RL token model
    from scripts.train_online_rl import load_rl_token_model

    rl_token_model = load_rl_token_model(config.rl_token_checkpoint, train_config, config.device)
    rl_token_model.eval()

    # Load actor
    actor = Actor(
        state_dim=train_config.state_dim,
        action_chunk_dim=train_config.action_chunk_dim,
        hidden_dim=train_config.mlp_hidden_dim,
        num_hidden_layers=train_config.mlp_num_hidden_layers,
        sigma=train_config.actor_noise_sigma,
        ref_dropout=0.0,  # no dropout during eval
    ).to(config.device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    log.info(
        "Actor loaded from %s (episode %d, %d env steps)",
        config.checkpoint,
        ckpt["total_episodes"],
        ckpt["total_env_steps"],
    )

    # Create environment — placeholder, must be wired per-task.
    # Example:
    #   from rlt_openpi.rollout.env_wrapper import RLTEnv
    #   from rlt_openpi.rollout.rollout_worker import RolloutWorker
    #   from rlt_openpi.rollout.intervention import InterventionManager
    #   from rlt_openpi.training.replay_buffer import ReplayBuffer
    #
    #   env = RLTEnv(raw_env, action_dim=train_config.action_dim,
    #                chunk_length=train_config.chunk_length)
    #   buf = ReplayBuffer(1, train_config.state_dim,
    #                      train_config.action_chunk_dim, train_config.chunk_length)
    #   worker = RolloutWorker(env, vla, rl_token_model, actor, buf,
    #                          InterventionManager(), train_config.chunk_length,
    #                          train_config.action_dim, config.device)
    #
    #   rewards, successes = [], []
    #   for ep in range(config.num_episodes):
    #       stats = worker.collect_episode()
    #       rewards.append(stats.total_reward)
    #       successes.append(stats.extra.get("success", False))
    #       log.info("Episode %d: reward=%.3f", ep, stats.total_reward)
    #
    #   log.info("Success rate: %.1f%% (%d/%d)",
    #            100 * sum(successes) / len(successes), sum(successes), len(successes))
    #   log.info("Mean reward: %.3f", sum(rewards) / len(rewards))

    log.info(
        "Actor ready for evaluation. Provide an RLTEnv environment and "
        "uncomment the evaluation loop. See script comments."
    )


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
