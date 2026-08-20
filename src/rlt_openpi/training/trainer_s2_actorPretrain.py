"""Standalone behavior-cloning pre-training for the Stage 2 actor."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from rlt_openpi.models.actor import Actor
from rlt_openpi.models.rl_token import RLTokenModel
from rlt_openpi.training.config import ActorPretrainConfig
from rlt_openpi.vla.vla_wrapper import VLAWrapper


class ActorPretrainTrainer:
    """Train and save an actor without constructing online-RL components."""

    def __init__(
        self,
        config: ActorPretrainConfig,
        vla: VLAWrapper,
        rl_token_model: RLTokenModel,
        device: torch.device | str = "cuda",
    ) -> None:
        self.config = config
        self.vla = vla
        self.rl_token_model = rl_token_model
        self.device = torch.device(device)

        self.rl_token_model.eval()
        for param in self.rl_token_model.parameters():
            param.requires_grad_(False)

        self.actor = Actor(
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            hidden_dim=config.mlp_hidden_dim,
            num_hidden_layers=config.mlp_num_hidden_layers,
            sigma=0.0,
            ref_dropout=0.0,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)

        action_stats = self.vla.norm_stats["actions"]
        if self.vla.use_quantile_norm:
            low = np.asarray(action_stats.q01[: config.action_dim], dtype=np.float32)
            high = np.asarray(action_stats.q99[: config.action_dim], dtype=np.float32)
            center = (low + high) / 2.0
            scale = (high - low) / 2.0
        else:
            center = np.asarray(action_stats.mean[: config.action_dim], dtype=np.float32)
            scale = np.asarray(action_stats.std[: config.action_dim], dtype=np.float32)

        self._action_center = torch.as_tensor(center, device=self.device).repeat(
            config.chunk_length
        )
        self._action_scale = (
            torch.as_tensor(scale, device=self.device)
            .clamp_min(1e-6)
            .repeat(config.chunk_length)
        )

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self._action_center) / self._action_scale

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        step: int,
        loss_value: float,
    ) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "pretrain_steps": step,
                "repo_id": self.config.repo_id,
                "final_loss": loss_value,
            },
            checkpoint_path,
        )
        print(f"[Actor Pretrain] Saved checkpoint to {checkpoint_path}")

    def train(self, data_iter: Any, log_fn: Any | None = None) -> Path:
        """Run BC pre-training and return the saved checkpoint path."""
        config = self.config
        self.actor.eval()

        print(f"\n[Actor Pretrain] Starting: {config.steps} steps")
        print(f"  Dataset: {config.repo_id}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Save every: {config.save_every} steps")

        save_dir = Path(config.save_dir) / config.run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(range(config.steps), desc="Actor Pretrain")
        for step in pbar:
            t0 = time.monotonic()
            observation, demo_actions = next(data_iter)
            t1 = time.monotonic()

            z, pad_mask, vla_actions = self.vla.extract_both(observation)
            t2 = time.monotonic()

            z_rl = self.rl_token_model.encode(
                z.to(self.device), pad_mask.to(self.device)
            )
            t3 = time.monotonic()

            state = torch.as_tensor(
                np.asarray(observation.state[:, : config.action_dim]),
                dtype=torch.float32,
                device=self.device,
            )
            actor_state = torch.cat([z_rl, state], dim=-1)

            batch_size = z_rl.shape[0]
            reference = vla_actions[
                :, : config.chunk_length, : config.action_dim
            ].to(device=self.device, dtype=torch.float32).reshape(batch_size, -1)
            target = torch.as_tensor(
                np.asarray(
                    demo_actions[:, : config.chunk_length, : config.action_dim]
                ),
                dtype=torch.float32,
                device=self.device,
            ).reshape(batch_size, -1)

            reference = self._normalize_action(reference)
            target = self._normalize_action(target)
            t4 = time.monotonic()

            prediction = self.actor(actor_state, reference)
            loss = F.mse_loss(prediction, target)
            t5 = time.monotonic()

            self.actor_optimizer.zero_grad()
            loss.backward()
            t6 = time.monotonic()

            step_num = step + 1
            is_log_step = (
                step_num == 1
                or step_num % config.log_every == 0
                or step_num == config.steps
            )
            is_checkpoint_step = (
                step_num % config.save_every == 0 and step_num < config.steps
            )
            grad_norm = None
            if is_log_step or is_checkpoint_step:
                grad_norm = sum(
                    param.grad.norm().item() ** 2
                    for param in self.actor.parameters()
                    if param.grad is not None
                ) ** 0.5
            self.actor_optimizer.step()
            t7 = time.monotonic()

            loss_value = loss.item()
            # if grad_norm is None:
            pbar.set_postfix(loss=f"{loss_value:.6f}")
            # else:
            #     pbar.set_postfix(loss=f"{loss_value:.6f}", grad=f"{grad_norm:.4f}")
            if log_fn is not None and is_log_step:
                assert grad_norm is not None
                log_fn(
                    {
                        "pretrain/actor_bc_loss": loss_value,
                        "pretrain/actor_grad_norm": grad_norm,
                        "pretrain/actor_lr": self.actor_optimizer.param_groups[0]["lr"],
                        "pretrain/step": step_num,
                    },
                    step=step_num,
                )

            if is_checkpoint_step:
                self._save_checkpoint(
                    save_dir / f"actor_pretrain_step{step_num}.pt",
                    step_num,
                    loss_value,
                )

            t8 = time.monotonic()
            # print(
            #     f"[DEBUG] step = {step_num} | "
            #     f"data_load={(t1 - t0) * 1000:.1f}ms | "
            #     f"vla_extract={(t2 - t1) * 1000:.1f}ms | "
            #     f"rl_encode={(t3 - t2) * 1000:.1f}ms | "
            #     f"batch_prepare={(t4 - t3) * 1000:.1f}ms | "
            #     f"actor_forward={(t5 - t4) * 1000:.1f}ms | "
            #     f"backward={(t6 - t5) * 1000:.1f}ms | "
            #     f"optimizer={(t7 - t6) * 1000:.1f}ms | "
            #     f"logging={(t8 - t7) * 1000:.1f}ms | "
            #     f"total={(t8 - t0) * 1000:.1f}ms"
            # )

        checkpoint_path = save_dir / "actor_pretrain.pt"
        self._save_checkpoint(checkpoint_path, config.steps, loss_value)
        return checkpoint_path
