"""Stage 2 trainer: Online RL with frozen VLA + RL token (Algorithm 1).

Implements the full online RL loop from the paper:
1. Warmup: fill replay buffer with VLA-only rollouts
2. Main loop: collect episode → update critic/actor (UTD ratio G)
3. TD3-style updates: twin Q-critics, delayed actor, Polyak targets
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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

        # The VLA wrapper returns robot-space actions.  The actor, critic, and
        # replay buffer instead operate in the VLA-normalized action space.
        # This prevents large-range gripper coordinates from dominating MSE and
        # matches TD3's [-1, 1] action clipping.
        action_stats = self.vla.norm_stats["actions"]
        if self.vla.use_quantile_norm:
            low = np.asarray(action_stats.q01[:config.action_dim], dtype=np.float32)
            high = np.asarray(action_stats.q99[:config.action_dim], dtype=np.float32)
            action_center = (low + high) / 2.0
            action_scale = (high - low) / 2.0
        else:
            action_center = np.asarray(action_stats.mean[:config.action_dim], dtype=np.float32)
            action_scale = np.asarray(action_stats.std[:config.action_dim], dtype=np.float32)

        self.action_center = torch.as_tensor(action_center, device=self.device)
        self.action_scale = torch.as_tensor(action_scale, device=self.device).clamp_min(1e-6)
        self._action_center_flat = self.action_center.repeat(config.chunk_length)
        self._action_scale_flat = self.action_scale.repeat(config.chunk_length)

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
            action_center=self.action_center.cpu().numpy(),
            action_scale=self.action_scale.cpu().numpy(),
            max_deviation=self.config.max_deviation,
            deviation_abort_threshold=self.config.deviation_abort_threshold,
            max_episode_chunks=self.config.max_episode_chunks,
        )

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map flattened robot-space chunks to the actor/critic action space."""
        return (action - self._action_center_flat) / self._action_scale_flat

    @staticmethod
    def _print_vla_demo_difference_stats(
        a_tilde_raw: torch.Tensor,
        a_demo_raw: torch.Tensor,
        action_dim: int,
    ) -> None:
        """Print raw-space VLA/demo errors for joints and grippers separately.

        The robot convention places a gripper at one-based dimensions 7, 14,
        ... (zero-based indices 6, 13, ...).  Keeping this diagnostic in raw
        robot units makes gripper errors directly interpretable.
        """
        diff = (a_tilde_raw - a_demo_raw).reshape(-1, action_dim)
        dim_indices = torch.arange(action_dim, device=diff.device)
        gripper_mask = (dim_indices + 1) % 7 == 0

        for name, mask in (
            ("joint", ~gripper_mask),
            ("gripper", gripper_mask),
        ):
            values = diff[:, mask].reshape(-1)
            if values.numel() == 0:
                continue
            print(
                f"  a_tilde - a_demo [{name}] (raw): "
                f"mean={values.mean().item():.4f} "
                f"std={values.std(unbiased=False).item():.4f} "
                f"mae={values.abs().mean().item():.4f} "
                f"rmse={values.square().mean().sqrt().item():.4f} "
                f"min={values.min().item():.4f} "
                f"max={values.max().item():.4f}"
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
        next_a_tilde = batch["next_a_tilde"]
        dones = batch["dones"]

        # --- Critic update (every step) ---
        td_target = compute_td_target(
            rewards=rewards,
            dones=dones,
            next_x=next_x,
            next_a_tilde=next_a_tilde,
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

    def _pretrain_actor(
        self,
        data_iter: Any,
        log_fn: Any | None = None,
    ) -> None:
        """Phase 0: pre-train the actor against dataset action sequences.

        Uses a pre-built iterator yielding ``(Observation, demo_actions)``.
        Observations follow the usual frozen-VLA → RL-token → state path;
        the Actor receives VLA reference actions as its second input and is
        supervised against the corresponding future actions from the dataset.
        Skipped when *data_iter* is ``None``.

        Args:
            data_iter: Infinite iterator yielding ``(Observation, actions)``
                tuples, or ``None`` to skip pre-training.
            log_fn: Optional callable used to report pre-training metrics to
                the configured logger (including W&B).
        """
        config = self.config
        if data_iter is None:
            return

        print(f"\n[Actor Pretrain] Starting BC pre-training: {config.actor_pretrain_steps} steps")
        print(f"  Dataset: {config.repo_id}")
        print(f"  Batch size: {config.actor_pretrain_batch_size}")

        # Keep gradients enabled but disable exploration noise/reference dropout:
        # this phase is deterministic supervised regression to demo actions.
        self.actor.eval()

        for step in range(config.actor_pretrain_steps):
            observation, demo_actions = next(data_iter)

            # Single VLA forward pass: embeddings + reference actions
            z, pad_mask, actions = self.vla.extract_both(observation)

            # Move to device (VLA outputs are on CPU)
            z = z.to(self.device)
            pad_mask = pad_mask.to(self.device)
            actions = actions.to(device=self.device, dtype=torch.float32)

            # RL token encode
            z_rl = self.rl_token_model.encode(z, pad_mask)  # [B, 2048]

            # Proprioceptive state
            # JAX path: observation.state is jnp, need np.array() before torch.as_tensor()
            # torch path: s_p = observation.state[:, :config.action_dim].to(self.device)
            s_p = torch.as_tensor(np.array(
                observation.state[:, :config.action_dim]),
                dtype=torch.float32, device=self.device,
            )

            # Build RL state
            x = torch.cat([z_rl, s_p], dim=-1)  # [B, 2062]

            # VLA and dataset actions are robot-space values.  Convert both to
            # the normalized actor space before supervised regression.
            a_tilde_raw = actions[:, :config.chunk_length, :].reshape(z_rl.shape[0], -1)
            a_demo_raw = torch.as_tensor(
                np.array(demo_actions[:, :config.chunk_length, :config.action_dim]),
                dtype=torch.float32,
                device=self.device,
            ).reshape(z_rl.shape[0], -1)
            a_tilde = self._normalize_action(a_tilde_raw)
            a_demo = self._normalize_action(a_demo_raw)

            # Actor produces the VLA-action correction; train its final action
            # against the dataset action sequence rather than against a_tilde.
            a_actor = self.actor(x, a_tilde)  # [B, C*d]
            loss = F.mse_loss(a_actor, a_demo)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            step_num = step + 1
            loss_value = loss.item()
            print(
                f"[Actor Pretrain] step {step_num}/{config.actor_pretrain_steps}  "
                f"loss={loss_value:.6f}"
            )

            if log_fn is not None and (
                step_num == 1
                or step_num % config.log_every == 0
                or step_num == config.actor_pretrain_steps
            ):
                log_fn({
                    "pretrain/actor_bc_loss": loss_value,
                    "pretrain/step": step_num,
                })

            # Diagnostic prints (first 3 steps + every 100 steps)
            if step < 3 or (step + 1) % 100 == 0:
                with torch.no_grad():
                    diff = a_actor - a_demo
                    self._print_vla_demo_difference_stats(
                        a_tilde_raw, a_demo_raw, config.action_dim
                    )
                    grad_norm = sum(
                        p.grad.norm().item() ** 2 for p in self.actor.parameters() if p.grad is not None
                    ) ** 0.5
                    weight_norm = sum(
                        p.norm().item() ** 2 for p in self.actor.parameters()
                    ) ** 0.5
                    print(
                        f"[Actor Pretrain] step {step + 1}/{config.actor_pretrain_steps}  "
                        f"loss={loss.item():.6f}  "
                        f"|z_rl|={z_rl.norm(dim=-1).mean().item():.2f}  "
                        f"|s_p|={s_p.norm(dim=-1).mean().item():.2f}  "
                        f"|a_vla|={a_tilde.norm(dim=-1).mean().item():.3f}  "
                        f"|a_raw|={a_demo.norm(dim=-1).mean().item():.3f}  "
                        f"|a_act|={a_actor.norm(dim=-1).mean().item():.3f}  "
                        f"|diff|={diff.norm(dim=-1).mean().item():.3f}  "
                        f"|grad|={grad_norm:.4f}  "
                        f"|w|={weight_norm:.2f}"
                    )
                    print(f"[DEBUG] a_raw[0]: {a_demo[0][:14]}")
                    print(f"[DEBUG] a_vla[0]: {a_tilde[0][:14]}")
                    print(f"[DEBUG] a_act[0]: {a_actor[0][:14]}")
                    # Extra detail on first step
                    if step == 0:
                        print(
                            f"  a_vla:  mean={a_tilde.mean().item():.4f} std={a_tilde.std().item():.4f} "
                            f"min={a_tilde.min().item():.4f} max={a_tilde.max().item():.4f}"
                        )
                        print(
                            f"  a_raw:   mean={a_demo.mean().item():.4f} std={a_demo.std().item():.4f} "
                            f"min={a_demo.min().item():.4f} max={a_demo.max().item():.4f}"
                        )
                        print(
                            f"  a_act:  mean={a_actor.mean().item():.4f} std={a_actor.std().item():.4f} "
                            f"min={a_actor.min().item():.4f} max={a_actor.max().item():.4f}"
                        )
                        print(
                            f"  diff:  mean={diff.mean().item():.4f} std={diff.std().item():.4f} "
                            f"min={diff.min().item():.4f} max={diff.max().item():.4f}"
                        )
                        print(
                            f"  z_rl:  mean={z_rl.mean().item():.4f} std={z_rl.std().item():.4f} "
                            f"norm_mean={z_rl.norm(dim=-1).mean().item():.2f}"
                        )
                        print(
                            f"  s_p:  mean={s_p.mean().item():.4f} std={s_p.std().item():.4f} "
                            f"norm_mean={s_p.norm(dim=-1).mean().item():.2f}"
                        )
            elif (step + 1) % 10 == 0:
                print(
                    f"[Actor Pretrain] step {step + 1}/{config.actor_pretrain_steps}  "
                    f"loss={loss.item():.6f}"
                )

        # Save pretrain checkpoint
        save_dir = Path(config.save_dir) / config.run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / "actor_pretrain.pt"
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "pretrain_steps": config.actor_pretrain_steps,
                "repo_id": config.repo_id,
                "final_loss": loss.item(),
            },
            ckpt_path,
        )
        print(f"[Actor Pretrain] Saved checkpoint to {ckpt_path}")
        print(f"[Actor Pretrain] Done. Final loss={loss.item():.6f}\n")

    def train(
        self,
        env: Any,
        intervention_mgr: InterventionManager | None = None,
        log_fn: Any | None = None,
        *,
        pretrain_data_iter: Any = None,
    ) -> None:
        """Run the full online RL training loop (Algorithm 1).

        Args:
            env: Chunk-level environment wrapper.
            intervention_mgr: Optional human intervention manager.
            log_fn: Optional callable ``log_fn(metrics_dict)`` for logging.
            pretrain_data_iter: Optional infinite iterator yielding
                ``(Observation, demo_actions)`` tuples for actor pre-training.
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

        # Phase 0: actor pre-training from --repo-id.  This is deliberately
        # independent of --obs-source, which only controls online rollouts.
        self._pretrain_actor(pretrain_data_iter, log_fn=log_fn)

        # NOTE: Warmup runs through RolloutWorker.collect_warmup() on the real env.
        # Phase 1: Warmup with VLA-only policy (skip if buffer already has data)
        if self.replay_buffer.size > 0:
            print(f"[Stage 2] Skipping warmup — replay buffer already has {self.replay_buffer.size} transitions (resumed)")
        elif cfg.warmup_buffer:
            self._load_warmup_buffer(cfg.warmup_buffer)
        else:
            display.warmup_start(cfg.warmup_steps)
            stored = 0
            for i in range(cfg.warmup_steps):
                stored += worker.collect_warmup(1)
                self._total_env_steps += cfg.chunk_length
                display.warmup_progress(i + 1, cfg.warmup_steps)
            display.warmup_done(stored, self.replay_buffer.size)
            self._save_warmup_buffer()

        # Phase 2: Online RL loop
        # NOTE: env._display_episode_num commented out (no real env)
        # if hasattr(env, '_display_episode_num'):
        #     env._display_episode_num = self._total_episodes + 1

        while self._total_env_steps < cfg.max_env_steps:
            # if hasattr(env, '_display_episode_num'):
            #     env._display_episode_num = self._total_episodes + 1

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
        print(f"[Stage 2] Saved warmup buffer ({self.replay_buffer.size} transitions) to {buf_path}")
        return buf_path

    def _load_warmup_buffer(self, path: str) -> None:
        """Load a standalone warmup buffer file into the replay buffer."""
        buf_state = torch.load(path, map_location="cpu", weights_only=False)
        self.replay_buffer.load_state_dict(buf_state)
        print(f"[Stage 2] Loaded warmup buffer ({self.replay_buffer.size} transitions) from {path}")

    def load_actor_pretrain(self, ckpt_path: str) -> None:
        """Load an actor-only BC pre-training checkpoint.

        This restores the actor and, when present, its optimizer state. It does
        not alter the critic, replay buffer, or online-RL counters.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "actor" not in ckpt:
            raise KeyError(f"Actor pretrain checkpoint has no 'actor' state: {ckpt_path}")

        self.actor.load_state_dict(ckpt["actor"])
        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])

        pretrain_steps = ckpt.get("pretrain_steps", "unknown")
        repo_id = ckpt.get("repo_id", "unknown")
        print(
            f"[Actor Pretrain] Loaded checkpoint from {ckpt_path} "
            f"(steps={pretrain_steps}, dataset={repo_id})"
        )

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
        print(f"[Stage 2] Saved checkpoint to {ckpt_path} (buffer={save_buffer})")
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
            print(f"[Stage 2] Restored replay buffer ({self.replay_buffer.size} transitions)")

        print(f"[Stage 2] Loaded checkpoint from {ckpt_path} (episode {self._total_episodes}, step {self._total_env_steps})")
