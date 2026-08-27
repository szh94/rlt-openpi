"""Standalone behavior-cloning pre-training for the Stage 2 actor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from rlt_openpi.models.actor import Actor
from rlt_openpi.models.rl_token import RLTokenModel
from rlt_openpi.training.config import ActorPretrainConfig
from rlt_openpi.vla.jax_vla_wrapper import JaxVLAWrapper


def _format_array_2f(value: Any) -> str:
    """Format an array with two fixed decimal places."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array2string(
        np.asarray(value, dtype=np.float64),
        formatter={"float_kind": lambda x: f"{x:.3f}"},
        max_line_width=200,
    )


def _format_array_6f(value: Any) -> str:
    """Format an array with six fixed decimal places."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array2string(
        np.asarray(value, dtype=np.float64),
        formatter={"float_kind": lambda x: f"{x:.6f}"},
        max_line_width=200,
    )


class ActorPretrainTrainer:
    """Train and save an actor without constructing online-RL components."""

    def __init__(
        self,
        config: ActorPretrainConfig,
        vla: JaxVLAWrapper,
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
        self._use_quantile_norm = self.vla.use_quantile_norm
        if self._use_quantile_norm:
            self._action_q01 = torch.as_tensor(
                np.asarray(action_stats.q01[: config.action_dim], dtype=np.float32),
                device=self.device,
            ).repeat(config.chunk_length)
            self._action_q99 = torch.as_tensor(
                np.asarray(action_stats.q99[: config.action_dim], dtype=np.float32),
                device=self.device,
            ).repeat(config.chunk_length)
        else:
            self._action_mean = torch.as_tensor(
                np.asarray(action_stats.mean[: config.action_dim], dtype=np.float32),
                device=self.device,
            ).repeat(config.chunk_length)
            self._action_std = torch.as_tensor(
                np.asarray(action_stats.std[: config.action_dim], dtype=np.float32),
                device=self.device,
            ).repeat(config.chunk_length)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Apply the same action normalization formula as OpenPI Normalize."""
        if self._use_quantile_norm:
            return (
                (action - self._action_q01)
                / (self._action_q99 - self._action_q01 + 1e-6)
                * 2.0
                - 1.0
            )
        return (action - self._action_mean) / (self._action_std + 1e-6)

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Invert the action normalization applied by _normalize_action."""
        if self._use_quantile_norm:
            return (
                (action + 1.0)
                / 2.0
                * (self._action_q99 - self._action_q01 + 1e-6)
                + self._action_q01
            )
        return action * (self._action_std + 1e-6) + self._action_mean

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

        save_dir = Path(config.save_dir) / config.run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(range(config.steps), desc="Actor Pretrain")
        for step in pbar:
            vla_obs_batch, data_actions, data_actions_norm, raw_state = next(data_iter)

            z, pad_mask, vla_actions_norm, vla_actions = self.vla.extract_both(vla_obs_batch)

            z_rl = self.rl_token_model.encode(
                z.to(self.device), pad_mask.to(self.device)
            )

            state = torch.as_tensor(
                np.asarray(vla_obs_batch.state[:, : config.action_dim]),
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
                    data_actions[:, : config.chunk_length, : config.action_dim]
                ),
                dtype=torch.float32,
                device=self.device,
            ).reshape(batch_size, -1)

            if step % 500 == 0:
                print(f"[ActorPretrain] action.q01     = {_format_array_2f(self._action_q01[ : config.action_dim])}")
                print(f"[ActorPretrain] action.q99     = {_format_array_2f(self._action_q99[ : config.action_dim])}")
                print(f"[ActorPretrain] action.range   = {_format_array_2f((self._action_q99 - self._action_q01)[: config.action_dim])}")
                print(f"[ActorPretrain] raw obs.state  = {_format_array_2f(raw_state[0, : config.action_dim])}")
                print(f"[ActorPretrain] vla_obs.state  = {_format_array_2f(vla_obs_batch.state[0, : config.action_dim])}")
                print(f"[ActorPretrain] target[0]      = {_format_array_2f(data_actions[0, 0, : config.action_dim])}")
                print(f"[ActorPretrain] target[1]      = {_format_array_2f(data_actions[0, 1, : config.action_dim])}")
                print(f"[ActorPretrain] reference      = {_format_array_2f(reference[0, : 2 * config.action_dim])}")

            reference = self._normalize_action(reference)
            target = self._normalize_action(target)

            prediction = self.actor(actor_state, reference)
            if step % 500 == 0:
                print(f"[ActorPretrain] actor_state = {_format_array_2f(actor_state[0, : 5]), _format_array_2f(actor_state[0, -14 :])}")
            loss = F.mse_loss(prediction, target)

            if step % 500 == 0:
                print(f"[ActorPretrain] tar_norm[0] = {_format_array_2f(data_actions_norm[0, 0, : config.action_dim])}")
                print(f"[ActorPretrain] tar_norm[1] = {_format_array_2f(data_actions_norm[0, 1, : config.action_dim])}")
                print(f"[ActorPretrain] ref_norm    = {_format_array_2f(reference[0, : 2 * config.action_dim])}")
                print(f"[ActorPretrain] pre_norm    = {_format_array_2f(prediction[0, : 2 * config.action_dim])}")
                print(f"[ActorPretrain] pre_unnorm  = {_format_array_2f(self._unnormalize_action(prediction)[0, : 2 * config.action_dim])}")

            if (step + 1) % 50 == 0:
                ref_loss = F.mse_loss(reference, target)
                actor_mse_per_joint = (
                    (prediction - target)
                    .reshape(batch_size, config.chunk_length, config.action_dim)
                    .square()
                    .mean(dim=(0, 1))
                )
                ref_mse_per_joint = (
                    (reference - target)
                    .reshape(batch_size, config.chunk_length, config.action_dim)
                    .square()
                    .mean(dim=(0, 1))
                )
                print(f"[ActorPretrain] step={step + 1} ref_loss={ref_loss.item():.6f} actor_loss={loss.item():.6f}")
                print(
                    "[ActorPretrain] ref_mse_per_joint="
                    f"{_format_array_6f(ref_mse_per_joint)}"
                )
                print(
                    "[ActorPretrain] actor_mse_per_joint="
                    f"{_format_array_6f(actor_mse_per_joint)}"
                )
                print(
                    "[ActorPretrain] actor_better = "
                    f"{((ref_mse_per_joint - actor_mse_per_joint) > 0).detach().cpu().numpy()}"
                )

            self.actor_optimizer.zero_grad()
            loss.backward()

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

        checkpoint_path = save_dir / "actor_pretrain.pt"
        self._save_checkpoint(checkpoint_path, config.steps, loss_value)
        return checkpoint_path
