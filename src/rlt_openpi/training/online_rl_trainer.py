"""Stage 2 trainer: Online RL with frozen VLA + RL token (Algorithm 1).

Implements the full online RL loop from the paper:
1. Warmup: fill replay buffer with VLA-only rollouts
2. Main loop: collect episode → update critic/actor (UTD ratio G)
3. TD3-style updates: twin Q-critics, delayed actor, Polyak targets
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

from rlt_openpi.models.actor import Actor
from rlt_openpi.models.critic import TwinQCritic
from rlt_openpi.models.rl_token import RLTokenModel
from rlt_openpi.rollout.intervention import InterventionManager
from rlt_openpi.rollout.rollout_worker import RolloutWorker
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.replay_buffer import ReplayBuffer
from rlt_openpi.training.td3_utils import actor_loss, compute_td_target, critic_loss
from rlt_openpi.utils import display
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logger = logging.getLogger(__name__)


class OnlineRLTrainer:
    """Stage 2: Online RL training with Algorithm 1.

    Loads frozen VLA + frozen RL token model, creates Actor + TwinQCritic +
    optimizers + ReplayBuffer + RolloutWorker, and runs the online RL loop.

    Args:
        config: Stage 2 training hyperparameters.
        vla: Frozen VLA wrapper (pre-loaded).
        rl_token_model: Frozen RL token model (loaded from Stage 1 checkpoint).
        device: Torch device for training.
    """

    def __init__(
        self,
        config: OnlineRLTrainConfig,
        vla: VLAWrapper,
        rl_token_model: RLTokenModel,
        device: torch.device | str = "cuda",
    ) -> None:
        self.config = config
        self.device = torch.device(device)

        # Frozen components
        self.vla = vla
        self.rl_token_model = rl_token_model
        self.rl_token_model.eval()
        for param in self.rl_token_model.parameters():
            param.requires_grad_(False)

        # Trainable actor
        self.actor = Actor(
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            hidden_dim=config.mlp_hidden_dim,
            num_hidden_layers=config.mlp_num_hidden_layers,
            sigma=config.actor_noise_sigma,
            ref_dropout=config.ref_action_dropout,
        ).to(self.device)

        # Trainable twin Q-critic
        self.critic = TwinQCritic(
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            hidden_dim=config.mlp_hidden_dim,
            num_hidden_layers=config.mlp_num_hidden_layers,
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            chunk_length=config.chunk_length,
        )

        # Counters
        self._total_env_steps = 0
        self._total_updates = 0
        self._total_episodes = 0

    def _create_rollout_worker(
        self,
        env: Any,
        intervention_mgr: InterventionManager | None = None,
    ) -> RolloutWorker:
        """Create a rollout worker wired to this trainer's components."""
        return RolloutWorker(
            env=env,
            vla=self.vla,
            rl_token_model=self.rl_token_model,
            actor=self.actor,
            replay_buffer=self.replay_buffer,
            intervention_mgr=intervention_mgr or InterventionManager(),
            chunk_length=self.config.chunk_length,
            action_dim=self.config.action_dim,
            device=self.device,
            max_deviation=self.config.max_deviation,
            deviation_abort_threshold=self.config.deviation_abort_threshold,
        )

    def _update_step(self, update_idx: int) -> dict[str, float]:
        """Run one TD3 update step.

        Critic is updated every call.  Actor is updated every
        ``critic_updates_per_actor`` calls.

        Args:
            update_idx: 0-based index within the current UTD batch.

        Returns:
            Dict of logged metrics.
        """
        cfg = self.config
        batch = self.replay_buffer.sample(batch_size=cfg.batch_size, device=str(self.device))

        x = batch["x"]
        a = batch["a"]
        a_tilde = batch["a_tilde"]
        rewards = batch["rewards"]
        next_x = batch["next_x"]
        dones = batch["dones"]

        # --- Critic update (every step) ---
        # Compute TD target
        # For next_a_tilde, we use a_tilde from the batch as an approximation
        # (the true next reference would require a VLA call, which is expensive)
        td_target = compute_td_target(
            rewards=rewards,
            dones=dones,
            next_x=next_x,
            next_a_tilde=a_tilde,  # approximate next reference
            actor=self.actor,
            critic=self.critic,
            gamma=cfg.gamma,
            chunk_length=cfg.chunk_length,
            target_noise_sigma=cfg.target_noise_sigma,
            target_noise_clip=cfg.target_noise_clip,
        )

        q1, q2 = self.critic(x, a)
        c_loss = critic_loss(q1, q2, td_target)

        self.critic_optimizer.zero_grad()
        c_loss.backward()
        self.critic_optimizer.step()

        metrics: dict[str, float] = {
            "critic_loss": c_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

        # --- Actor update (delayed) ---
        if update_idx % cfg.critic_updates_per_actor == 0:
            self.actor.train()
            a_actor = self.actor(x, a_tilde)
            q_value = self.critic.q_min(x, a_actor)
            a_loss = actor_loss(q_value, a_actor, a_tilde, cfg.bc_regularizer_beta)

            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

            metrics["actor_loss"] = a_loss.item()

        # --- Polyak target update ---
        self.critic.update_targets(cfg.tau)

        self._total_updates += 1
        return metrics

    def train(
        self,
        env: Any,
        intervention_mgr: InterventionManager | None = None,
        log_fn: Any | None = None,
    ) -> None:
        """Run the full online RL training loop (Algorithm 1).

        Args:
            env: Chunk-level environment wrapper.
            intervention_mgr: Optional human intervention manager.
            log_fn: Optional callable ``log_fn(metrics_dict)`` for logging.
        """
        cfg = self.config
        worker = self._create_rollout_worker(env, intervention_mgr)
        train_display = display.TrainingDisplay(window_size=20)
        train_start = time.time()

        display.training_start({
            "Task": cfg.task_prompt or "(not set)",
            "Max env steps": f"{cfg.max_env_steps:,}",
            "UTD ratio": str(cfg.utd_ratio),
            "Chunk length": str(cfg.chunk_length),
            "Action dim": str(cfg.action_dim),
            "Run name": cfg.run_name,
        })

        # Phase 1: Warmup with VLA-only policy (skip if buffer already has data)
        if self.replay_buffer.size > 0:
            logger.info(
                "Skipping warmup — replay buffer already has %d transitions (resumed from checkpoint)",
                self.replay_buffer.size,
            )
        elif cfg.warmup_buffer:
            self._load_warmup_buffer(cfg.warmup_buffer)
        else:
            display.warmup_start(cfg.warmup_steps)
            stored = 0
            obs = env.reset()
            for i in range(cfg.warmup_steps):
                action_chunk = worker._get_warmup_action(obs)
                x, a_tilde_flat = worker._extract_rl_state(obs)
                a_flat = action_chunk.reshape(-1)
                next_obs, rewards, done, _info = env.step(action_chunk)
                next_x, _ = worker._extract_rl_state(next_obs)
                self.replay_buffer.add(
                    x=x, a=a_flat, a_tilde=a_tilde_flat,
                    rewards=rewards, next_x=next_x, done=float(done),
                )
                stored += 1
                self._total_env_steps += cfg.chunk_length
                display.warmup_progress(i + 1, cfg.warmup_steps)
                if done:
                    obs = env.reset()
                else:
                    obs = next_obs
            display.warmup_done(stored, self.replay_buffer.size)
            self._save_warmup_buffer()

        # Phase 2: Online RL loop
        # Inject episode counter into env so RobotEnv.reset() can display it
        if hasattr(env, '_display_episode_num'):
            env._display_episode_num = self._total_episodes + 1

        while self._total_env_steps < cfg.max_env_steps:
            if hasattr(env, '_display_episode_num'):
                env._display_episode_num = self._total_episodes + 1

            self.actor.eval()
            stats = worker.collect_episode()
            self._total_episodes += 1
            self._total_env_steps += stats.num_steps

            success = stats.extra.get("success", False)
            train_display.record_episode(success, stats.total_reward)

            display.episode_result(
                episode_num=self._total_episodes,
                total_reward=stats.total_reward,
                success=success,
                num_chunks=stats.num_chunks,
                num_steps=stats.num_steps,
                interventions=stats.interventions,
            )

            episode_metrics = {
                "episode_reward": stats.total_reward,
                "episode_success": int(success),
                "episode_chunks": stats.num_chunks,
                "episode_steps": stats.num_steps,
                "episode_interventions": stats.interventions,
                "total_env_steps": self._total_env_steps,
                "total_episodes": self._total_episodes,
                "buffer_size": self.replay_buffer.size,
            }

            # Update actor and critic (UTD ratio G)
            update_metrics: dict[str, float] = {}
            for g in range(cfg.utd_ratio):
                step_metrics = self._update_step(g)
                update_metrics = step_metrics

            all_metrics = {**episode_metrics, **update_metrics}

            train_display.print_summary(
                total_episodes=self._total_episodes,
                total_env_steps=self._total_env_steps,
                max_env_steps=cfg.max_env_steps,
                buffer_size=self.replay_buffer.size,
                critic_loss=update_metrics.get("critic_loss", 0.0),
                actor_loss=update_metrics.get("actor_loss"),
                q_mean=update_metrics.get("q1_mean", 0.0),
            )

            if log_fn is not None:
                log_fn(all_metrics)

            if self._total_episodes % cfg.save_every == 0:
                ckpt_path = self.save()
                display.checkpoint_saved(str(ckpt_path))

        # Final save
        ckpt_path = self.save()
        display.checkpoint_saved(str(ckpt_path))
        display.training_done(
            self._total_episodes,
            self._total_env_steps,
            self._total_updates,
            time.time() - train_start,
        )

    def _save_warmup_buffer(self) -> Path:
        """Save the replay buffer as a standalone file after warmup."""
        save_dir = Path(self.config.save_dir) / self.config.run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        buf_path = save_dir / "warmup_buffer.pt"
        torch.save(self.replay_buffer.state_dict(), buf_path)
        logger.info("Saved warmup buffer (%d transitions) to %s", self.replay_buffer.size, buf_path)
        return buf_path

    def _load_warmup_buffer(self, path: str) -> None:
        """Load a standalone warmup buffer file into the replay buffer."""
        buf_state = torch.load(path, map_location="cpu", weights_only=False)
        self.replay_buffer.load_state_dict(buf_state)
        logger.info("Loaded warmup buffer (%d transitions) from %s", self.replay_buffer.size, path)

    def save(self, path: str | None = None, save_buffer: bool = True) -> Path:
        """Save actor, critic, optimizer, and replay buffer states.

        Args:
            path: Override save path. Defaults to config.save_dir.
            save_buffer: Whether to include replay buffer in the checkpoint.
                Disable for eval-only checkpoints to save disk space.

        Returns:
            Path to the saved checkpoint.
        """
        save_dir = Path(path or self.config.save_dir) / self.config.run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"online_rl_ep{self._total_episodes}.pt"
        payload: dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_env_steps": self._total_env_steps,
            "total_updates": self._total_updates,
            "total_episodes": self._total_episodes,
            "config": self.config,
        }
        if save_buffer:
            payload["replay_buffer"] = self.replay_buffer.state_dict()
        torch.save(payload, ckpt_path)
        logger.info("Saved checkpoint to %s (buffer=%s)", ckpt_path, save_buffer)
        return ckpt_path

    def load(self, ckpt_path: str) -> None:
        """Load actor, critic, optimizer, and replay buffer from checkpoint.

        Args:
            ckpt_path: Path to a saved checkpoint file.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self._total_env_steps = ckpt["total_env_steps"]
        self._total_updates = ckpt["total_updates"]
        self._total_episodes = ckpt["total_episodes"]

        if "replay_buffer" in ckpt:
            self.replay_buffer.load_state_dict(ckpt["replay_buffer"])
            logger.info(
                "Restored replay buffer (%d transitions)", self.replay_buffer.size
            )

        logger.info(
            "Loaded checkpoint from %s (episode %d, step %d)",
            ckpt_path,
            self._total_episodes,
            self._total_env_steps,
        )
