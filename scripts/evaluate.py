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

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import tyro

from rlt_openpi.models.actor import Actor
from rlt_openpi.rollout.factory import make_env
from rlt_openpi.rollout.intervention import InterventionManager
from rlt_openpi.rollout.rollout_worker import RolloutWorker
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.replay_buffer import ReplayBuffer
from rlt_openpi.utils.checkpoint import load_rl_token_model
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
    env_factory: str = ""  # Python import path, e.g. "rlt_openpi.envs.franka.env_factory.make_franka_env"
    task_prompt: str = ""  # Task instruction for VLA
    num_episodes: int = 50
    save_dir: str = ""  # Directory to save results JSON (defaults to checkpoint's parent dir)
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
    rl_token_model = load_rl_token_model(config.rl_token_checkpoint, device=config.device)
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

    # Create environment via pluggable factory
    if not config.env_factory:
        log.error("--env-factory is required. Provide a Python import path to an env factory function.")
        raise SystemExit(1)

    env = make_env(
        config.env_factory,
        action_dim=train_config.action_dim,
        chunk_length=train_config.chunk_length,
        task_prompt=config.task_prompt,
    )
    log.info("Environment created: action_dim=%d, chunk_length=%d", env.action_dim, env.chunk_length)

    # Dummy replay buffer (collect_episode requires one, but we don't train)
    buf = ReplayBuffer(1, train_config.state_dim, train_config.action_chunk_dim, train_config.chunk_length)
    worker = RolloutWorker(
        env, vla, rl_token_model, actor, buf,
        InterventionManager(), train_config.chunk_length,
        train_config.action_dim, config.device,
    )

    episodes = []
    for ep in range(config.num_episodes):
        stats = worker.collect_episode(store_transitions=False)
        success = stats.extra.get("success", False)
        episodes.append({
            "episode": ep,
            "reward": stats.total_reward,
            "success": success,
            "num_chunks": stats.num_chunks,
            "num_steps": stats.num_steps,
        })
        log.info("Episode %d: reward=%.3f, success=%s", ep, stats.total_reward, success)

    num_success = sum(e["success"] for e in episodes)
    success_rate = num_success / len(episodes)
    mean_reward = sum(e["reward"] for e in episodes) / len(episodes)

    log.info("Success rate: %.1f%% (%d/%d)", 100 * success_rate, num_success, len(episodes))
    log.info("Mean reward: %.3f", mean_reward)

    # Save results
    results = {
        "checkpoint": config.checkpoint,
        "train_episodes": ckpt["total_episodes"],
        "train_env_steps": ckpt["total_env_steps"],
        "eval_timestamp": datetime.now().isoformat(),
        "num_episodes": len(episodes),
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "episodes": episodes,
    }
    save_dir = Path(config.save_dir) if config.save_dir else Path(config.checkpoint).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / f"eval_{len(episodes)}ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.write_text(json.dumps(results, indent=2))
    log.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
