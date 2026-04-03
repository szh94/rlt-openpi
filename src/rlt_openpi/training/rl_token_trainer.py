"""Stage 1 trainer: RL token encoder-decoder on demonstration data.

Trains the RLTokenModel (encoder-decoder) to compress VLA embeddings
into a single RL token z_rl via reconstruction loss.

Supports two modes:
- **Frozen VLA** (alpha=0): Only trains the encoder-decoder on
  pre-extracted or on-the-fly VLA embeddings.
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

from rlt_openpi.models.rl_token import RLTokenModel
from rlt_openpi.training.config import RLTokenTrainConfig
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logger = logging.getLogger(__name__)


class RLTokenTrainer:
    """Stage 1 trainer for the RL token encoder-decoder.

    When ``config.vla_finetune_alpha == 0`` (default), only the RL token
    encoder-decoder is trained on pre-extracted VLA embeddings.

    When ``config.vla_finetune_alpha > 0``, the trainer also fine-tunes
    the VLA using its flow-matching loss on ground-truth demo actions.
    Call ``setup_joint_training(vla)`` before training to unfreeze the
    VLA and create its optimizer.

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

        # VLA optimizer (created by setup_joint_training)
        self.vla_optimizer: torch.optim.Optimizer | None = None

        self._global_step = 0

    def setup_joint_training(self, vla: VLAWrapper) -> None:
        """Prepare VLA for joint fine-tuning (alpha > 0).

        Unfreezes VLA parameters and creates a separate optimizer for
        them.  Must be called before ``train_step_joint`` or
        ``train_joint``.

        Args:
            vla: VLA wrapper whose parameters will be unfrozen.
        """
        if self.config.vla_finetune_alpha <= 0:
            logger.warning("setup_joint_training called but vla_finetune_alpha=%.2f", self.config.vla_finetune_alpha)
            return

        vla.unfreeze()
        vla_params = vla.trainable_parameters()
        logger.info("Unfroze VLA: %d trainable parameters", sum(p.numel() for p in vla_params))

        self.vla_optimizer = torch.optim.AdamW(
            vla_params,
            lr=self.config.vla_learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_step(self, z: Tensor, pad_mask: Tensor) -> dict[str, float]:
        """Run one training step on pre-extracted VLA embeddings (frozen VLA).

        Args:
            z: VLA embeddings [B, M, D] (from EmbeddingExtractor).
            pad_mask: Boolean mask [B, M] (True = valid token).

        Returns:
            Dict of logged metrics.
        """
        self.model.train()

        z = z.to(self.device)
        pad_mask = pad_mask.to(self.device)

        loss, _z_rl, _z_hat = self.model(z, pad_mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._global_step += 1

        return {"loss": loss.item(), "step": self._global_step}

    def train_step_from_obs(self, vla: VLAWrapper, observations: dict[str, Any]) -> dict[str, float]:
        """Extract VLA embeddings from observations, then train (frozen VLA).

        Convenience method that combines embedding extraction with a
        training step.  Use when iterating over a demonstration dataset
        of raw observations with alpha=0.

        Args:
            vla: Frozen VLA wrapper for embedding extraction.
            observations: Batched observation dict for the VLA.

        Returns:
            Dict of logged metrics.
        """
        with torch.no_grad():
            z, pad_mask = vla.extract_embeddings(observations)
        return self.train_step(z, pad_mask)

    def train_step_joint(
        self,
        vla: VLAWrapper,
        observations: dict[str, Any],
        actions: Tensor,
    ) -> dict[str, float]:
        """Run one joint training step: L_ro(phi) + alpha * L_vla(theta_vla).

        Both losses are computed and backpropagated in a single pass.
        Gradients are independent:
        - L_ro only updates encoder-decoder params (z is detached inside
          RLTokenModel.forward)
        - L_vla only updates VLA params (it doesn't touch the encoder-decoder)

        Requires ``setup_joint_training(vla)`` to have been called first.

        Args:
            vla: VLA wrapper (unfrozen for fine-tuning).
            observations: Batched observation dict for the VLA.
            actions: Ground-truth demo actions [B, H, action_dim].

        Returns:
            Dict of logged metrics.
        """
        alpha = self.config.vla_finetune_alpha
        self.model.train()

        # --- L_ro: RL token reconstruction loss ---
        # Extract embeddings without gradients (we detach inside the model
        # anyway, so no VLA gradient flows through L_ro)
        with torch.no_grad():
            z, pad_mask = vla.extract_embeddings(observations)
        z = z.to(self.device)
        pad_mask = pad_mask.to(self.device)
        l_ro, _z_rl, _z_hat = self.model(z, pad_mask)

        # --- L_vla: VLA flow-matching loss ---
        actions = actions.to(self.device)
        l_vla = vla.compute_vla_loss(observations, actions)

        # --- Combined backward ---
        total_loss = l_ro + alpha * l_vla

        self.optimizer.zero_grad()
        if self.vla_optimizer is not None:
            self.vla_optimizer.zero_grad()

        total_loss.backward()

        self.optimizer.step()
        if self.vla_optimizer is not None:
            self.vla_optimizer.step()

        self._global_step += 1

        return {
            "loss": total_loss.item(),
            "l_ro": l_ro.item(),
            "l_vla": l_vla.item(),
            "step": self._global_step,
        }

    def train(
        self,
        dataloader: Iterator[tuple[Tensor, Tensor]],
        log_fn: Any | None = None,
    ) -> None:
        """Run the full training loop with pre-extracted embeddings (frozen VLA).

        Args:
            dataloader: Iterator yielding (z, pad_mask) batches of
                pre-extracted VLA embeddings.
            log_fn: Optional callable ``log_fn(metrics_dict)`` for logging.
        """
        logger.info("Starting Stage 1 training for %d steps (frozen VLA)", self.config.num_train_steps)

        for step_idx in range(1, self.config.num_train_steps + 1):
            try:
                z, pad_mask = next(dataloader)
            except StopIteration:
                logger.warning("Dataloader exhausted at step %d", step_idx)
                break

            metrics = self.train_step(z, pad_mask)

            if step_idx % self.config.log_every == 0:
                logger.info("Step %d | loss=%.6f", step_idx, metrics["loss"])
                if log_fn is not None:
                    log_fn(metrics)

            if step_idx % self.config.save_every == 0:
                self.save()

        # Final save
        self.save()
        logger.info("Stage 1 training complete (%d steps)", self._global_step)

    def train_joint(
        self,
        vla: VLAWrapper,
        dataloader: Iterator[tuple[dict[str, Any], Tensor]],
        log_fn: Any | None = None,
    ) -> None:
        """Run the full joint training loop: L_ro + alpha * L_vla.

        Requires ``setup_joint_training(vla)`` to have been called first.

        Args:
            vla: VLA wrapper (unfrozen).
            dataloader: Iterator yielding (observations_dict, actions) batches
                of demonstration data.  ``observations_dict`` is a batched
                dict suitable for VLA inference; ``actions`` is
                [B, action_horizon, action_dim].
            log_fn: Optional callable ``log_fn(metrics_dict)`` for logging.
        """
        alpha = self.config.vla_finetune_alpha
        logger.info(
            "Starting Stage 1 joint training for %d steps (alpha=%.3f)",
            self.config.num_train_steps,
            alpha,
        )

        for step_idx in range(1, self.config.num_train_steps + 1):
            try:
                observations, actions = next(dataloader)
            except StopIteration:
                logger.warning("Dataloader exhausted at step %d", step_idx)
                break

            metrics = self.train_step_joint(vla, observations, actions)

            if step_idx % self.config.log_every == 0:
                logger.info(
                    "Step %d | total=%.6f | l_ro=%.6f | l_vla=%.6f",
                    step_idx,
                    metrics["loss"],
                    metrics["l_ro"],
                    metrics["l_vla"],
                )
                if log_fn is not None:
                    log_fn(metrics)

            if step_idx % self.config.save_every == 0:
                self.save()

        # Final save
        self.save()
        logger.info("Stage 1 joint training complete (%d steps)", self._global_step)

    def save(self, path: str | None = None) -> Path:
        """Save model and optimizer state.

        Args:
            path: Override save path. Defaults to config.save_dir.

        Returns:
            Path to the saved checkpoint.
        """
        save_dir = Path(path or self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"rl_token_step{self._global_step}.pt"
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._global_step,
            "config": self.config,
        }
        if self.vla_optimizer is not None:
            state["vla_optimizer"] = self.vla_optimizer.state_dict()
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
        self._global_step = ckpt["step"]
        if "vla_optimizer" in ckpt and self.vla_optimizer is not None:
            self.vla_optimizer.load_state_dict(ckpt["vla_optimizer"])
        logger.info("Loaded checkpoint from %s (step %d)", ckpt_path, self._global_step)
