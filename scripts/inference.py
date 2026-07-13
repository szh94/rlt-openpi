"""Run inference with a trained model on an environment (JAX).

Supports two modes:
  - **stage1**: VLA-only inference (fine-tuned VLA from Stage 1, no actor).
  - **stage2**: Full pipeline (VLA + RL token + actor from Stage 2).

The mode is auto-detected: if --checkpoint (Stage 2) is provided, runs
stage2 inference; otherwise runs stage1 VLA-only inference.

Usage:
    # Stage 1: VLA-only inference
    uv run python scripts/inference.py \\
        --vla-checkpoint-dir checkpoints/pi05_droid/params \\
        --stage1-checkpoint checkpoints/stage1_rlt_encoder/rl_token_step5000 \\
        --env-factory rlt_openpi.envs.aloha.env_factory.make_aloha_env \\
        --task-prompt "pick up the cup" \\
        --num-episodes 50

    # Stage 2: full pipeline inference
    uv run python scripts/inference.py \\
        --checkpoint checkpoints/stage2_ac_online/run_latest/online_rl_ep100 \\
        --vla-checkpoint-dir checkpoints/pi05_droid/params \\
        --rl-token-checkpoint checkpoints/stage1_rlt_encoder/rl_token_step5000 \\
        --env-factory rlt_openpi.envs.aloha.env_factory.make_aloha_env \\
        --task-prompt "pick up the cup" \\
        --num-episodes 50
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from flax import nnx
import tyro

from rlt_openpi.envs.factory import make_env
from rlt_openpi.envs.intervention import InterventionManager
from rlt_openpi.rollout.rollout_worker import RolloutWorker
from rlt_openpi.training.replay_buffer import ReplayBuffer
from rlt_openpi.utils.checkpoint import load_rl_token_model
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Stage 2 checkpoint (if provided, runs full pipeline eval)
    checkpoint: str = ""
    # Stage 1 checkpoint with fine-tuned VLA weights
    stage1_checkpoint: str = ""

    vla_config_name: str = "pi05_droid_finetune"
    vla_checkpoint_dir: str = ""
    rl_token_checkpoint: str = ""
    env_factory: str = ""
    task_prompt: str = ""
    action_dim: int = 8
    chunk_length: int = 10
    num_episodes: int = 50
    save_dir: str = ""


def _run(config: EvalConfig):
    """Evaluate full pipeline: VLA + RL token + actor."""
    import orbax.checkpoint as ocp

    ckpt_path = Path(config.checkpoint)
    params_dir = ckpt_path / "params" if (ckpt_path / "params").exists() else ckpt_path

    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(str(params_dir))

    total_episodes = int(ckpt.get("total_episodes", 0))
    total_env_steps = int(ckpt.get("total_env_steps", 0))

    vla = VLAWrapper(
        checkpoint_path=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
    )

    rl_token_model = load_rl_token_model(config.rl_token_checkpoint)

    # Create actor and load from checkpoint
    from rlt_openpi.models.actor import Actor

    actor_state = ckpt["actor"]

    # Infer architecture from saved params
    mlp = actor_state["mlp"]
    fc_keys = sorted([k for k in mlp if k.startswith("fc")], key=lambda x: int(x[2:]))
    num_hidden = len(fc_keys) - 1
    input_dim = int(mlp["fc0"]["kernel"].shape[0])  # state_dim + action_chunk_dim
    action_chunk_dim = int(mlp[fc_keys[-1]]["kernel"].shape[1])  # output dim
    hidden_dim = int(mlp["fc0"]["kernel"].shape[1])
    state_dim = input_dim - action_chunk_dim

    actor = Actor(
        state_dim=state_dim,
        action_chunk_dim=action_chunk_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden,
        sigma=0.0,  # eval: no noise
        ref_dropout=0.0,  # eval: no dropout
        rngs=nnx.Rngs(0),
    )

    # Load actor weights
    actor_graphdef, _ = nnx.split(actor)
    nnx.update(actor, nnx.State.from_pure_dict(actor_graphdef, actor_state))

    log.info(
        "Actor loaded from %s (episode %d, %d env steps)",
        config.checkpoint,
        total_episodes,
        total_env_steps,
    )

    action_dim = action_chunk_dim // config.chunk_length

    env = make_env(
        config.env_factory,
        action_dim=action_dim,
        chunk_length=config.chunk_length,
        task_prompt=config.task_prompt,
    )
    log.info(
        "Environment created: action_dim=%d, chunk_length=%d",
        env.action_dim,
        env.chunk_length,
    )

    buf = ReplayBuffer(1, state_dim, action_chunk_dim, config.chunk_length)
    worker = RolloutWorker(
        env, vla, rl_token_model, actor, buf,
        InterventionManager(), config.chunk_length,
        action_dim,
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
        log.info(
            "Episode %d: reward=%.3f, success=%s",
            ep, stats.total_reward, success,
        )

    return episodes, {
        "mode": "stage2",
        "checkpoint": config.checkpoint,
        "train_episodes": total_episodes,
        "train_env_steps": total_env_steps,
    }


def _run_vla(config: EvalConfig):
    """Evaluate VLA-only (with optional Stage 1 fine-tuned weights)."""
    vla = VLAWrapper(
        checkpoint_path=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
    )

    # Load fine-tuned VLA weights from Stage 1 checkpoint if available
    if config.stage1_checkpoint:
        import orbax.checkpoint as ocp

        stage1_path = Path(config.stage1_checkpoint)
        stage1_params = stage1_path / "params" if (stage1_path / "params").exists() else stage1_path
        try:
            checkpointer = ocp.PyTreeCheckpointer()
            ckpt = checkpointer.restore(str(stage1_params))
            if "vla_model" in ckpt and vla._jax_model is not None:
                vla_graphdef, _ = nnx.split(vla._jax_model)
                nnx.update(
                    vla._jax_model,
                    nnx.State.from_pure_dict(vla_graphdef, ckpt["vla_model"]),
                )
                log.info("Loaded fine-tuned VLA weights from %s", config.stage1_checkpoint)
            else:
                log.warning("No vla_model key in %s; using base VLA weights", config.stage1_checkpoint)
        except Exception as e:
            log.warning("Could not load VLA weights from checkpoint: %s", e)

    env = make_env(
        config.env_factory,
        action_dim=config.action_dim,
        chunk_length=config.chunk_length,
        task_prompt=config.task_prompt,
    )
    log.info(
        "Environment created: action_dim=%d, chunk_length=%d",
        env.action_dim,
        env.chunk_length,
    )

    episodes = []
    for ep in range(config.num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_chunks = 0

        while True:
            vla_input = vla.preprocess_obs(obs)
            action_chunk = vla.get_rl_chunk_reference(vla_input, config.chunk_length)
            action_chunk = np.array(action_chunk[0])  # [C, action_dim]

            next_obs, chunk_rewards, done, info = env.step(action_chunk)
            episode_reward += float(chunk_rewards.sum())
            episode_chunks += 1

            if done:
                success = info.get("success", False)
                episodes.append({
                    "episode": ep,
                    "reward": episode_reward,
                    "success": success,
                    "num_chunks": episode_chunks,
                    "num_steps": info.get(
                        "steps_executed", episode_chunks * config.chunk_length
                    ),
                })
                log.info(
                    "Episode %d/%d: chunks=%d, reward=%.3f, success=%s",
                    ep + 1,
                    config.num_episodes,
                    episode_chunks,
                    episode_reward,
                    success,
                )
                break

            obs = next_obs

    return episodes, {
        "mode": "stage1",
        "stage1_checkpoint": config.stage1_checkpoint,
    }


def main(config: EvalConfig) -> None:
    """Run evaluation (auto-detects stage1 vs stage2)."""
    log.info("Eval config: %s", config)

    if not config.env_factory:
        log.error("--env-factory is required.")
        raise SystemExit(1)

    if config.checkpoint:
        log.info("Full eval: VLA + RL token + actor")
        episodes, meta = _run(config)
    else:
        log.info("VLA-only eval")
        episodes, meta = _run_vla(config)

    num_success = sum(e["success"] for e in episodes)
    success_rate = num_success / len(episodes) if episodes else 0.0
    mean_reward = sum(e["reward"] for e in episodes) / len(episodes) if episodes else 0.0

    log.info(
        "Success rate: %.1f%% (%d/%d)", 100 * success_rate, num_success, len(episodes)
    )
    log.info("Mean reward: %.3f", mean_reward)

    results = {
        **meta,
        "vla_checkpoint_dir": config.vla_checkpoint_dir,
        "vla_config_name": config.vla_config_name,
        "eval_timestamp": datetime.now().isoformat(),
        "num_episodes": len(episodes),
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "episodes": episodes,
    }

    if config.save_dir:
        save_dir = Path(config.save_dir)
    elif config.checkpoint:
        save_dir = Path(config.checkpoint).parent
    else:
        save_dir = Path("results") / meta["mode"]
    save_dir.mkdir(parents=True, exist_ok=True)

    results_path = (
        save_dir
        / f"eval_{meta['mode']}_{len(episodes)}ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    results_path.write_text(json.dumps(results, indent=2))
    log.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
