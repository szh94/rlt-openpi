"""Stage 1 trainer: RL token encoder-decoder on demonstration data.

Trains the RLTokenModel (encoder-decoder) to compress VLA embeddings
into a single RL token z_rl via reconstruction loss.

Supports two modes (selected automatically by ``vla_finetune_alpha``):
- **Frozen VLA** (alpha=0): Only trains the encoder-decoder on
  on-the-fly VLA embeddings.
- **Joint training** (alpha>0): Simultaneously trains the RL token
  encoder-decoder (L_ro) and fine-tunes the VLA (alpha * L_vla).
  The combined objective is: L = L_ro(phi) + alpha * L_vla(theta_vla).
  Gradients are independent — L_ro only updates phi (VLA embeddings
  are always detached in the encoder-decoder), and L_vla only updates
  theta_vla.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor
from tqdm import tqdm

from rlt_openpi.models.rl_token import RLTokenModel
from rlt_openpi.training.config import RLTokenTrainConfig
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logger = logging.getLogger(__name__)


class RLTokenTrainer:
    """Stage 1 trainer for the RL token encoder-decoder.

    Mode is selected automatically by ``config.vla_finetune_alpha``:
    - alpha == 0: frozen VLA, trains encoder-decoder only.
    - alpha > 0:  joint training, fine-tunes VLA alongside encoder-decoder.

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

        # VLA joint training state (created by _setup_joint_training)
        self._vla: VLAWrapper | None = None
        self.vla_optimizer: torch.optim.Optimizer | None = None
        self.vla_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        self._global_step = 0

    @property
    def joint(self) -> bool:
        """Whether the trainer is in joint training mode."""
        return self.config.vla_finetune_alpha > 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        vla: VLAWrapper,
        observations: Any,
        actions: Tensor,
    ) -> dict[str, float]:
        """Run one training step, auto-selecting frozen or joint mode.

        Args:
            vla: VLA wrapper.
            observations: Batched Observation (or dict) for the VLA.
            actions: Ground-truth demo actions [B, H, action_dim].

        Returns:
            Dict of logged metrics.
        """
        if self.joint:
            return self._step_joint(vla, observations, actions)
        return self._step_frozen(vla, observations)

    def train(
        self,
        vla: VLAWrapper,
        dataloader: Iterator[tuple[Any, Tensor]],
        log_fn: Any | None = None,
    ) -> None:
        """Run the full training loop.

        Automatically sets up joint training if alpha > 0.

        Args:
            vla: VLA wrapper.
            dataloader: Infinite iterator yielding (observations, actions).
            log_fn: Optional callable ``log_fn(metrics_dict)`` for logging.
        """
        alpha = self.config.vla_finetune_alpha
        if self.joint:
            self._setup_joint_training(vla)
            logger.info(
                "Starting Stage 1 joint training for %d steps (alpha=%.3f)",
                self.config.num_train_steps,
                alpha,
            )
        else:
            logger.info(
                "Starting Stage 1 frozen-VLA training for %d steps",
                self.config.num_train_steps,
            )

        if self.config.resume_checkpoint:
            self.load(self.config.resume_checkpoint)
            logger.info("Resumed from step %d", self._global_step)

        pbar = tqdm(range(1, self.config.num_train_steps + 1), desc="Stage 1")
        for step_idx in pbar:
            try:
                observations, actions = next(dataloader)
            except StopIteration:
                logger.warning("Dataloader exhausted at step %d", step_idx)
                break

            metrics = self.step(vla, observations, actions)

            # Progress bar
            if self.joint:
                pbar.set_postfix(l_ro=f"{metrics['l_ro']:.4f}", l_vla=f"{metrics['l_vla']:.4f}")
            else:
                pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

            # wandb logging (every log_every steps)
            if step_idx % self.config.log_every == 0 and log_fn is not None:
                log_fn(metrics, step=metrics.get("step"))

            if step_idx % self.config.save_every == 0:
                self.save()

        if self._global_step % self.config.save_every != 0:
            self.save()
        logger.info("Stage 1 training complete (%d steps)", self._global_step)

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
        if self._vla is not None:
            state["vla_model"] = self._vla.extractor.pi0.state_dict()
        if self.vla_optimizer is not None:
            state["vla_optimizer"] = self.vla_optimizer.state_dict()
        if self.vla_scheduler is not None:
            state["vla_scheduler"] = self.vla_scheduler.state_dict()
        torch.save(state, ckpt_path)
        logger.info("Saved checkpoint to %s", ckpt_path)
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
        if "vla_model" in ckpt and self._vla is not None:
            self._vla.extractor.pi0.load_state_dict(ckpt["vla_model"])
            logger.info("Restored fine-tuned VLA weights from checkpoint")
        if "vla_optimizer" in ckpt and self.vla_optimizer is not None:
            self.vla_optimizer.load_state_dict(ckpt["vla_optimizer"])
        if "vla_scheduler" in ckpt and self.vla_scheduler is not None:
            self.vla_scheduler.load_state_dict(ckpt["vla_scheduler"])
        logger.info("Loaded checkpoint from %s (step %d)", ckpt_path, self._global_step)

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

    def _setup_joint_training(self, vla: VLAWrapper) -> None:
        """Unfreeze VLA and create its optimizer (called once by train())."""
        vla.unfreeze()
        if self.config.gradient_checkpointing:
            vla.extractor.pi0.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on VLA")
        self._vla = vla
        vla_params = vla.trainable_parameters()
        logger.info("Unfroze VLA: %d trainable parameters", sum(p.numel() for p in vla_params))

        self.vla_optimizer = torch.optim.AdamW(
            vla_params,
            lr=self.config.vla_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.vla_scheduler = self._build_scheduler(self.vla_optimizer)

    def _step_frozen(
        self,
        vla: VLAWrapper,
        observations: Any,
    ) -> dict[str, float]:
        """Frozen VLA step: extract embeddings (no grad) → L_ro only."""
        self.model.train()

        observations = _obs_to_device(observations, self.device)
        with torch.no_grad():
            z, pad_mask = vla.extract_embeddings(observations)

        z = z.to(self.device)
        pad_mask = pad_mask.to(self.device)
        loss, _z_rl, _z_hat = self.model(z, pad_mask)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        self._global_step += 1
        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "step": self._global_step,
        }

    def _step_joint(
        self,
        vla: VLAWrapper,
        observations: Any,
        actions: Tensor,
    ) -> dict[str, float]:
        """Joint step: single VLA forward → L_ro + alpha * L_vla."""
        alpha = self.config.vla_finetune_alpha
        self.model.train()

        observations = _obs_to_device(observations, self.device)
        actions = actions.to(self.device)

        # Single VLA forward: detached embeddings + flow-matching loss
        z, pad_mask, l_vla = vla.compute_vla_loss_with_embeddings(observations, actions)

        # L_ro: RL token reconstruction loss
        z = z.to(self.device)
        pad_mask = pad_mask.to(self.device)
        l_ro, _z_rl, _z_hat = self.model(z, pad_mask)

        # Combined backward (disjoint graphs: L_ro→φ, L_vla→θ_vla)
        total_loss = l_ro + alpha * l_vla

        self.optimizer.zero_grad()
        if self.vla_optimizer is not None:
            self.vla_optimizer.zero_grad()

        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        if self.vla_optimizer is not None:
            vla_grad_norm = torch.nn.utils.clip_grad_norm_(self._vla.trainable_parameters(), max_norm=self.config.max_grad_norm)
            self.vla_optimizer.step()
        else:
            vla_grad_norm = torch.tensor(0.0)
        if self.vla_scheduler is not None:
            self.vla_scheduler.step()

        self._global_step += 1
        return {
            "loss": total_loss.item(),
            "l_ro": l_ro.item(),
            "l_vla": l_vla.item(),
            "grad_norm": grad_norm.item(),
            "vla_grad_norm": vla_grad_norm.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "step": self._global_step,
        }


def _obs_to_device(obs: Any, device: torch.device) -> Any:
    """Recursively move an Observation (or dict) of tensors to a device."""
    from openpi.models.model import Observation

    if isinstance(obs, Observation):
        return Observation(
            images={k: v.to(device) for k, v in obs.images.items()},
            image_masks={k: v.to(device) for k, v in obs.image_masks.items()},
            state=obs.state.to(device),
            tokenized_prompt=obs.tokenized_prompt.to(device) if obs.tokenized_prompt is not None else None,
            tokenized_prompt_mask=obs.tokenized_prompt_mask.to(device) if obs.tokenized_prompt_mask is not None else None,
            token_ar_mask=obs.token_ar_mask.to(device) if obs.token_ar_mask is not None else None,
            token_loss_mask=obs.token_loss_mask.to(device) if obs.token_loss_mask is not None else None,
        )
    if isinstance(obs, dict):
        return {k: _obs_to_device(v, device) for k, v in obs.items()}
    if isinstance(obs, torch.Tensor):
        return obs.to(device)
    return obs
