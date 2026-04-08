"""Roll out a fine-tuned VLA on a real robot (no RL actor, no RL token).

Loads the PI-0.5 model, creates the environment, and runs episodes using
only the VLA's sampled action chunks.  Useful for evaluating a
fine-tuned VLA before or without Stage 2 RL training.

Usage:
    python scripts/rollout_vla.py --help
    python scripts/rollout_vla.py \
        --env-factory examples.franka.env_factory.make_franka_env \
        --vla-config-name pi05_droid_finetune \
        --vla-checkpoint-dir /path/to/model.safetensors \
        --task-prompt "stack the three blocks on the tray" \
        --num-episodes 10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import tyro

from rlt_openpi.rollout.env_factory import make_env
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class RolloutConfig:
    """VLA-only rollout configuration."""

    vla_config_name: str = "pi05_droid_finetune"
    vla_checkpoint_dir: str = ""
    env_factory: str = ""
    task_prompt: str = ""
    action_dim: int = 8
    chunk_length: int = 10
    num_episodes: int = 10
    device: str = "cuda"


def main(config: RolloutConfig) -> None:
    """Run VLA-only rollout episodes."""
    log.info("Rollout config: %s", config)

    if not config.vla_checkpoint_dir:
        log.error("--vla-checkpoint-dir is required.")
        raise SystemExit(1)
    if not config.env_factory:
        log.error("--env-factory is required.")
        raise SystemExit(1)

    # Load VLA
    log.info("Loading VLA: config=%s, checkpoint=%s", config.vla_config_name, config.vla_checkpoint_dir)
    vla = VLAWrapper(
        checkpoint_path=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device=config.device,
    )

    # Create environment
    env = make_env(
        config.env_factory,
        action_dim=config.action_dim,
        chunk_length=config.chunk_length,
        task_prompt=config.task_prompt,
    )
    log.info("Environment created: action_dim=%d, chunk_length=%d", env.action_dim, env.chunk_length)

    successes: list[bool] = []
    rewards: list[float] = []

    for ep in range(config.num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_chunks = 0

        while True:
            # Preprocess observation and sample VLA action chunk
            with torch.no_grad():
                vla_input = vla.preprocess_obs(obs)
                action_chunk = vla.get_rl_chunk_reference(vla_input, config.chunk_length)
                action_chunk = action_chunk.squeeze(0).cpu().numpy()  # [C, action_dim]

            next_obs, chunk_rewards, done, info = env.step(action_chunk)
            episode_reward += float(chunk_rewards.sum())
            episode_chunks += 1

            if done:
                success = info.get("success", False)
                successes.append(success)
                rewards.append(episode_reward)
                log.info(
                    "Episode %d/%d: chunks=%d, reward=%.3f, success=%s",
                    ep + 1, config.num_episodes, episode_chunks, episode_reward, success,
                )
                break

            obs = next_obs

    num_success = sum(successes)
    log.info(
        "Done. Success rate: %.1f%% (%d/%d), mean reward: %.3f",
        100 * num_success / len(successes) if successes else 0,
        num_success,
        len(successes),
        sum(rewards) / len(rewards) if rewards else 0,
    )


if __name__ == "__main__":
    main(tyro.cli(RolloutConfig))
