"""Stage 1 trainer: RL token encoder-decoder on demonstration data.

Trains the RLTokenModel (encoder-decoder) to compress VLA embeddings
into a single RL token z_rl via reconstruction loss.

Only frozen-VLA mode is supported (VLA joint training has been disabled).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import time

import jax
import torch
from torch import Tensor
from tqdm import tqdm

from rlt_openpi.models.rl_token import RLTokenModel
from rlt_openpi.training.config import RLTokenTrainConfig
from rlt_openpi.vla.vla_wrapper import VLAWrapper

class RLTokenTrainer:
    """Stage 1 trainer for the RL token encoder-decoder.

    Only frozen-VLA mode is supported. VLA is always frozen.

    Usage::

        trainer = RLTokenTrainer(config, device="cuda")
        trainer.train(vla, dataloader, log_fn=logger.log)

    Args:
        config: Stage 1 training hyperparameters.
        device: Torch device for training.
    """

    def __init__(
        self,
        config: RLTokenTrainConfig,
        device: torch.device | str = "cuda",
    ) -> None:
        self.config = config
        self.device = torch.device(device)

        # Build RL token model
        self.model = RLTokenModel(
            embedding_dim=config.embedding_dim,
            encoder_layers=config.encoder_layers,
            encoder_heads=config.encoder_heads,
            decoder_layers=config.decoder_layers,
            decoder_heads=config.decoder_heads,
        ).to(self.device)

        # Optimizer for the RL token model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = self._build_scheduler(self.optimizer)

        # --- VLA joint training state (disabled) ---
        # self._vla: VLAWrapper | None = None
        # self.vla_optimizer: torch.optim.Optimizer | None = None
        # self.vla_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        self._global_step = 0

    # --- VLA joint training disabled ---
    # @property
    # def joint(self) -> bool:
    #     """Whether the trainer is in joint training mode."""
    #     return self.config.vla_finetune_alpha > 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        vla: VLAWrapper,
        observations: Any,
        actions: Tensor,
    ) -> dict[str, float]:
        """Run one training step (frozen-VLA mode only).

        Args:
            vla: VLA wrapper.
            observations: Batched Observation (or dict) for the VLA.
            actions: Ground-truth demo actions [B, H, action_dim].

        Returns:
            Dict of logged metrics.
        """
        # VLA joint training disabled — always use frozen mode.
        return self._step_frozen(vla, observations)

    def train(
        self,
        vla: VLAWrapper,
        dataloader: Iterator[tuple[Any, Tensor]],
        log_fn: Any | None = None,
    ) -> None:
        """Run the full training loop.

        Always runs frozen-VLA mode. Joint training is disabled.

        Args:
            vla: VLA wrapper.
            dataloader: Infinite iterator yielding (observations, actions).
            log_fn: Optional callable ``log_fn(metrics_dict)`` for logging.
        """
        # VLA joint training disabled — always frozen-VLA mode.
        print(f"[Stage 1] Starting frozen-VLA training for {self.config.num_train_steps} steps")

        if self.config.resume_checkpoint:
            self.load(self.config.resume_checkpoint)
            print(f"[Stage 1] Resumed from step {self._global_step}")

        pbar = tqdm(range(1, self.config.num_train_steps + 1), desc="Stage 1")
        for step_idx in pbar:
            try:
                observations, actions = next(dataloader)
            except StopIteration:
                print(f"[Stage 1] WARNING: Dataloader exhausted at step {step_idx}")
                break

            metrics = self.step(vla, observations, actions)

            # Progress bar (VLA joint training disabled)
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

            # wandb logging (every log_every steps)
            if step_idx % self.config.log_every == 0 and log_fn is not None:
                log_fn(metrics, step=metrics.get("step"))

            if step_idx % self.config.save_every == 0:
                self.save()

        if self._global_step % self.config.save_every != 0:
            self.save()
        print(f"[Stage 1] Training complete ({self._global_step} steps)")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> Path:
        """Save model and optimizer state.

        Args:
            path: Override save path. Defaults to config.save_dir.

        Returns:
            Path to the saved checkpoint.
        """
        save_dir = Path(path or self.config.save_dir) / self.config.run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"rl_token_step{self._global_step}.pt"
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self._global_step,
            "config": self.config,
        }
        # --- VLA joint training checkpoint (disabled) ---
        # if self._vla is not None:
        #     state["vla_model"] = self._vla.extractor.pi0.state_dict()
        # if self.vla_optimizer is not None:
        #     state["vla_optimizer"] = self.vla_optimizer.state_dict()
        # if self.vla_scheduler is not None:
        #     state["vla_scheduler"] = self.vla_scheduler.state_dict()
        torch.save(state, ckpt_path)
        print(f"[Stage 1] Saved checkpoint to {ckpt_path}")
        return ckpt_path

    def load(self, ckpt_path: str) -> None:
        """Load model and optimizer state from checkpoint.

        Args:
            ckpt_path: Path to a saved checkpoint file.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self._global_step = ckpt["step"]
        # --- VLA joint training checkpoint load (disabled) ---
        # if "vla_model" in ckpt and self._vla is not None:
        #     self._vla.extractor.pi0.load_state_dict(ckpt["vla_model"])
        #     logger.info("Restored fine-tuned VLA weights from checkpoint")
        # if "vla_optimizer" in ckpt and self.vla_optimizer is not None:
        #     self.vla_optimizer.load_state_dict(ckpt["vla_optimizer"])
        # if "vla_scheduler" in ckpt and self.vla_scheduler is not None:
        #     self.vla_scheduler.load_state_dict(ckpt["vla_scheduler"])
        print(f"[Stage 1] Loaded checkpoint from {ckpt_path} (step {self._global_step})")

    # ------------------------------------------------------------------
    # Private: mode-specific steps
    # ------------------------------------------------------------------

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Build a linear warmup → constant LR scheduler."""
        warmup = self.config.warmup_steps
        if warmup <= 0:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup
        )

    # def _setup_joint_training(self, vla: VLAWrapper) -> None:
    #     """Unfreeze VLA and create its optimizer (called once by train())."""
    #     vla.unfreeze()
    #     if self.config.gradient_checkpointing:
    #         vla.extractor.pi0.gradient_checkpointing_enable()
    #         logger.info("Enabled gradient checkpointing on VLA")
    #     self._vla = vla
    #     vla_params = vla.trainable_parameters()
    #     logger.info("Unfroze VLA: %d trainable parameters", sum(p.numel() for p in vla_params))

    #     self.vla_optimizer = torch.optim.AdamW(
    #         vla_params,
    #         lr=self.config.vla_learning_rate,
    #         weight_decay=self.config.weight_decay,
    #     )
    #     self.vla_scheduler = self._build_scheduler(self.vla_optimizer)

    def _step_frozen(
        self,
        vla: VLAWrapper,
        observations: Any,
    ) -> dict[str, float]:
        """Frozen VLA step: extract embeddings (no grad) → L_ro only."""
        t0 = time.monotonic()
        self.model.train()

        observations = _obs_to_device(observations, self.device)
        t1 = time.monotonic()

        with torch.no_grad():
            z, pad_mask = vla.extract_embeddings(observations)
        t2 = time.monotonic()

        z = z.to(self.device)
        pad_mask = pad_mask.to(self.device)
        loss, _z_rl, _z_hat = self.model(z, pad_mask)
        t3 = time.monotonic()

        self.optimizer.zero_grad()
        loss.backward()
        t4 = time.monotonic()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        t5 = time.monotonic()

        self._global_step += 1

        # Timing breakout (ms)
        t_obs_to_device = (t1 - t0) * 1000
        t_vla_embed = (t2 - t1) * 1000
        t_rl_forward = (t3 - t2) * 1000
        t_backward = (t4 - t3) * 1000
        t_optimizer = (t5 - t4) * 1000
        t_total = (t5 - t0) * 1000

        print(
            f"[Step {self._global_step}] "
            f"obs_to_device={t_obs_to_device:.1f}ms | "
            f"vla_embed={t_vla_embed:.1f}ms | "
            f"rl_forward={t_rl_forward:.1f}ms | "
            f"backward={t_backward:.1f}ms | "
            f"optimizer={t_optimizer:.1f}ms | "
            f"total={t_total:.1f}ms"
        )

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "step": self._global_step,
        }

    # --- VLA joint training disabled ---
    # def _step_joint(
    #     self,
    #     vla: VLAWrapper,
    #     observations: Any,
    #     actions: Tensor,
    # ) -> dict[str, float]:
    #     """Joint step: single VLA forward → L_ro + alpha * L_vla."""
    #     alpha = self.config.vla_finetune_alpha
    #     self.model.train()
    #
    #     observations = _obs_to_device(observations, self.device)
    #     actions = actions.to(self.device)
    #
    #     # Single VLA forward: detached embeddings + flow-matching loss
    #     z, pad_mask, l_vla = vla.compute_vla_loss_with_embeddings(observations, actions)
    #
    #     # L_ro: RL token reconstruction loss
    #     z = z.to(self.device)
    #     pad_mask = pad_mask.to(self.device)
    #     l_ro, _z_rl, _z_hat = self.model(z, pad_mask)
    #
    #     # Combined backward (disjoint graphs: L_ro→φ, L_vla→θ_vla)
    #     total_loss = l_ro + alpha * l_vla
    #
    #     self.optimizer.zero_grad()
    #     if self.vla_optimizer is not None:
    #         self.vla_optimizer.zero_grad()
    #
    #     total_loss.backward()
    #
    #     grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
    #     self.optimizer.step()
    #     self.scheduler.step()
    #     if self.vla_optimizer is not None:
    #         vla_grad_norm = torch.nn.utils.clip_grad_norm_(self._vla.trainable_parameters(), max_norm=self.config.max_grad_norm)
    #         self.vla_optimizer.step()
    #     else:
    #         vla_grad_norm = torch.tensor(0.0)
    #     if self.vla_scheduler is not None:
    #         self.vla_scheduler.step()
    #
    #     self._global_step += 1
    #     return {
    #         "loss": total_loss.item(),
    #         "l_ro": l_ro.item(),
    #         "l_vla": l_vla.item(),
    #         "grad_norm": grad_norm.item(),
    #         "vla_grad_norm": vla_grad_norm.item(),
    #         "lr": self.optimizer.param_groups[0]["lr"],
    #         "step": self._global_step,
    #     }


def _obs_to_device(obs: Any, device: torch.device) -> Any:
    """Recursively move an Observation (or dict) of tensors to a device."""
    from openpi.models.model import Observation

    if isinstance(obs, Observation):
        return Observation(
            images={k: _obs_to_device(v, device) for k, v in obs.images.items()},
            image_masks={k: _obs_to_device(v, device) for k, v in obs.image_masks.items()},
            state=_obs_to_device(obs.state, device),
            tokenized_prompt=_obs_to_device(obs.tokenized_prompt, device) if obs.tokenized_prompt is not None else None,
            tokenized_prompt_mask=_obs_to_device(obs.tokenized_prompt_mask, device) if obs.tokenized_prompt_mask is not None else None,
            token_ar_mask=_obs_to_device(obs.token_ar_mask, device) if obs.token_ar_mask is not None else None,
            token_loss_mask=_obs_to_device(obs.token_loss_mask, device) if obs.token_loss_mask is not None else None,
        )
    if isinstance(obs, dict):
        return {k: _obs_to_device(v, device) for k, v in obs.items()}
    if isinstance(obs, jax.Array):
        return obs
    if isinstance(obs, torch.Tensor):
        return obs.to(device)
    return obs
