"""Stage 2 trainer: Online RL with frozen VLA + RL token (Algorithm 1) (JAX/Flax nnx).

Implements the full online RL loop from the paper:
1. Warmup: fill replay buffer with VLA-only rollouts
2. Main loop: collect episode -> update critic/actor (UTD ratio G)
3. TD3-style updates: twin Q-critics, delayed actor, Polyak targets
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from rlt_openpi.models.actor import Actor
from rlt_openpi.models.critic import TwinQCritic
from rlt_openpi.models.rl_token import RLTokenModel
from rlt_openpi.envs.intervention import InterventionManager
from rlt_openpi.rollout.rollout_worker import RolloutWorker
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.replay_buffer import ReplayBuffer
from rlt_openpi.training.td3_utils import actor_loss, compute_td_target, critic_loss
from rlt_openpi.utils import display
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logger = logging.getLogger(__name__)


class OnlineRLTrainer:
    """Stage 2: Online RL training with Algorithm 1 (JAX).

    Loads frozen VLA + frozen RL token model, creates Actor + TwinQCritic +
    optimizers + ReplayBuffer + RolloutWorker, and runs the online RL loop.

    Args:
        config: Stage 2 training hyperparameters.
        vla: Frozen VLA wrapper (pre-loaded).
        rl_token_model: Frozen RL token model (loaded from Stage 1 checkpoint).
    """

    def __init__(
        self,
        config: OnlineRLTrainConfig,
        vla: VLAWrapper,
        rl_token_model: RLTokenModel,
    ) -> None:
        self.config = config

        # Frozen components
        self.vla = vla
        self.rl_token_model = rl_token_model

        # Trainable actor (zero-init last layer)
        self.actor = Actor(
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            hidden_dim=config.mlp_hidden_dim,
            num_hidden_layers=config.mlp_num_hidden_layers,
            sigma=config.actor_noise_sigma,
            ref_dropout=config.ref_action_dropout,
            rngs=nnx.Rngs(0),
        )

        # Trainable twin Q-critic
        self.critic = TwinQCritic(
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            hidden_dim=config.mlp_hidden_dim,
            num_hidden_layers=config.mlp_num_hidden_layers,
            rngs=nnx.Rngs(1),
        )

        # Optimizers (nnx wrappers around optax)
        self.actor_optimizer = nnx.Optimizer(self.actor, optax.adam(config.actor_lr))
        self.critic_optimizer = nnx.Optimizer(
            self.critic, optax.adam(config.critic_lr)
        )

        # Replay buffer (numpy-backed, returns JAX arrays)
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            chunk_length=config.chunk_length,
        )

        # RNG state
        self._rng = jax.random.PRNGKey(42)

        # Counters
        self._total_env_steps = 0
        self._total_updates = 0
        self._total_episodes = 0

    def _next_rng(self):
        self._rng, rng = jax.random.split(self._rng)
        return rng

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
            max_deviation=self.config.max_deviation,
            deviation_abort_threshold=self.config.deviation_abort_threshold,
        )

    def _update_step(self, update_idx: int) -> dict[str, float]:
        """Run one TD3 update step (JAX gradient computation).

        Critic is updated every call.  Actor is updated every
        ``critic_updates_per_actor`` calls.

        Args:
            update_idx: 0-based index within the current UTD batch.

        Returns:
            Dict of logged metrics.
        """
        cfg = self.config
        batch = self.replay_buffer.sample(batch_size=cfg.batch_size)

        x = batch["x"]
        a = batch["a"]
        a_tilde = batch["a_tilde"]
        rewards = batch["rewards"]
        next_x = batch["next_x"]
        dones = batch["dones"]

        # --- Compute TD target ---
        rng = self._next_rng()
        td_target = compute_td_target(
            rewards=rewards,
            dones=dones,
            next_x=next_x,
            next_a_tilde=a_tilde,  # approximate next reference
            actor=self.actor,
            critic=self.critic,
            gamma=cfg.gamma,
            chunk_length=cfg.chunk_length,
            rng=rng,
            target_noise_sigma=cfg.target_noise_sigma,
            target_noise_clip=cfg.target_noise_clip,
        )

        # --- Critic update ---
        def _critic_loss_fn(critic):
            q1, q2 = critic(x, a)
            c_loss = jnp.mean((q1 - td_target) ** 2 + (q2 - td_target) ** 2)
            return c_loss, (q1, q2)

        (c_loss, (q1, q2)), c_grads = nnx.value_and_grad(
            _critic_loss_fn, has_aux=True
        )(self.critic)
        self.critic_optimizer.update(c_grads)

        metrics: dict[str, float] = {
            "critic_loss": float(c_loss),
            "q1_mean": float(q1.mean()),
            "q2_mean": float(q2.mean()),
        }

        # --- Actor update (delayed) ---
        if update_idx % cfg.critic_updates_per_actor == 0:
            rng = self._next_rng()

            def _actor_loss_fn(actor):
                a_actor = actor(x, a_tilde, training=True, rng=rng)
                q_value = self.critic.q_min(x, a_actor)
                a_loss = -q_value.mean() + cfg.bc_regularizer_beta * jnp.mean(
                    (a_actor - a_tilde) ** 2
                )
                return a_loss

            a_loss, a_grads = nnx.value_and_grad(_actor_loss_fn)(self.actor)
            self.actor_optimizer.update(a_grads)
            metrics["actor_loss"] = float(a_loss)

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

        # Phase 1: Warmup with VLA-only policy
        if self.replay_buffer.size > 0:
            logger.info(
                "Skipping warmup - replay buffer already has %d transitions (resumed from checkpoint)",
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
        if hasattr(env, "_display_episode_num"):
            env._display_episode_num = self._total_episodes + 1

        while self._total_env_steps < cfg.max_env_steps:
            if hasattr(env, "_display_episode_num"):
                env._display_episode_num = self._total_episodes + 1

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
        buf_path = save_dir / "warmup_buffer.npz"
        state = self.replay_buffer.state_dict()
        np.savez_compressed(buf_path, **{k: v for k, v in state.items() if isinstance(v, np.ndarray)})
        np.savez_compressed(
            buf_path.with_suffix(".meta.npz"),
            ptr=np.array(state["ptr"]),
            size=np.array(state["size"]),
        )
        logger.info("Saved warmup buffer (%d transitions) to %s", self.replay_buffer.size, buf_path)
        return buf_path

    def _load_warmup_buffer(self, path: str) -> None:
        """Load a standalone warmup buffer file into the replay buffer."""
        import numpy as np

        path = Path(path)
        data = dict(np.load(path, allow_pickle=True))
        meta = dict(np.load(path.with_suffix(".meta.npz"), allow_pickle=True))
        state = {
            "ptr": int(meta["ptr"]),
            "size": int(meta["size"]),
            "x": data["x"],
            "a": data["a"],
            "a_tilde": data["a_tilde"],
            "rewards": data["rewards"],
            "next_x": data["next_x"],
            "dones": data["dones"],
        }
        self.replay_buffer.load_state_dict(state)
        logger.info("Loaded warmup buffer (%d transitions) from %s", self.replay_buffer.size, path)

    def save(self, path: str | None = None, save_buffer: bool = True) -> Path:
        """Save actor, critic, optimizer, and replay buffer states via orbax.

        Args:
            path: Override save path. Defaults to config.save_dir.
            save_buffer: Whether to include replay buffer in the checkpoint.

        Returns:
            Path to the saved checkpoint directory.
        """
        save_dir = Path(path or self.config.save_dir) / self.config.run_name
        ckpt_dir = save_dir / f"online_rl_ep{self._total_episodes}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        import orbax.checkpoint as ocp

        checkpointer = ocp.PyTreeCheckpointer()

        # Extract nnx states as pure dicts
        _, actor_state = nnx.split(self.actor)
        _, critic_state = nnx.split(self.critic)

        payload = {
            "actor": actor_state.to_pure_dict(),
            "critic": critic_state.to_pure_dict(),
            "actor_optimizer": nnx.state(self.actor_optimizer).to_pure_dict(),
            "critic_optimizer": nnx.state(self.critic_optimizer).to_pure_dict(),
            "total_env_steps": self._total_env_steps,
            "total_updates": self._total_updates,
            "total_episodes": self._total_episodes,
        }
        if save_buffer:
            buf_state = self.replay_buffer.state_dict()
            # orbax can't handle dicts with mixed types, so save buffer separately
            import numpy as np
            buf_path = ckpt_dir / "replay_buffer.npz"
            np.savez_compressed(buf_path, **{k: v for k, v in buf_state.items() if isinstance(v, np.ndarray)})
            np.savez_compressed(
                buf_path.with_suffix(".meta.npz"),
                ptr=np.array(buf_state["ptr"]),
                size=np.array(buf_state["size"]),
            )

        checkpointer.save(str(ckpt_dir / "params"), payload)
        logger.info("Saved checkpoint to %s (buffer=%s)", ckpt_dir, save_buffer)
        return ckpt_dir

    def load(self, ckpt_path: str) -> None:
        """Load actor, critic, optimizer, and replay buffer from checkpoint.

        Args:
            ckpt_path: Path to a saved checkpoint directory.
        """
        import orbax.checkpoint as ocp
        import numpy as np

        ckpt_path = Path(ckpt_path)
        checkpointer = ocp.PyTreeCheckpointer()

        # Find the params subdir
        params_dir = ckpt_path / "params"
        if not params_dir.exists():
            # Try loading directly if ckpt_path is the params dir
            params_dir = ckpt_path

        ckpt = checkpointer.restore(str(params_dir))

        # Restore actor
        actor_graphdef, _ = nnx.split(self.actor)
        nnx.update(
            self.actor,
            nnx.State.from_pure_dict(actor_graphdef, ckpt["actor"]),
        )

        # Restore critic (includes target networks)
        critic_graphdef, _ = nnx.split(self.critic)
        nnx.update(
            self.critic,
            nnx.State.from_pure_dict(critic_graphdef, ckpt["critic"]),
        )

        # Restore optimizer states
        ao_gd, _ = nnx.split(self.actor_optimizer)
        nnx.update(
            self.actor_optimizer,
            nnx.State.from_pure_dict(ao_gd, ckpt["actor_optimizer"]),
        )

        co_gd, _ = nnx.split(self.critic_optimizer)
        nnx.update(
            self.critic_optimizer,
            nnx.State.from_pure_dict(co_gd, ckpt["critic_optimizer"]),
        )

        # Restore counters
        self._total_env_steps = int(ckpt["total_env_steps"])
        self._total_updates = int(ckpt["total_updates"])
        self._total_episodes = int(ckpt["total_episodes"])

        # Restore replay buffer if present
        buf_npz = ckpt_path / "replay_buffer.npz"
        if buf_npz.exists():
            data = dict(np.load(buf_npz, allow_pickle=True))
            meta = dict(np.load(buf_npz.with_suffix(".meta.npz"), allow_pickle=True))
            state = {
                "ptr": int(meta["ptr"]),
                "size": int(meta["size"]),
                "x": data["x"],
                "a": data["a"],
                "a_tilde": data["a_tilde"],
                "rewards": data["rewards"],
                "next_x": data["next_x"],
                "dones": data["dones"],
            }
            self.replay_buffer.load_state_dict(state)
            logger.info("Restored replay buffer (%d transitions)", self.replay_buffer.size)

        logger.info(
            "Loaded checkpoint from %s (episode %d, step %d)",
            ckpt_path,
            self._total_episodes,
            self._total_env_steps,
        )
