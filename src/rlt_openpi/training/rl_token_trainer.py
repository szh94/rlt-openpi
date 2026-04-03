"""Stage 1 trainer: RL token encoder-decoder on demonstration data.

Trains the RLTokenModel (encoder-decoder) to compress VLA embeddings
into a single RL token z_rl via reconstruction loss.  The VLA backbone
is frozen; only the encoder, decoder, and projection head are trained.

Optionally applies a VLA fine-tuning term (alpha * VLA loss) for joint
training, though alpha=0 (frozen VLA) is the default.
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

    Loads a frozen VLA, extracts embeddings from demonstration
    observations, and trains the RLTokenModel on reconstruction loss.

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

        # Optimizer for the RL token model only
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self._global_step = 0

    def train_step(self, z: Tensor, pad_mask: Tensor) -> dict[str, float]:
        """Run one training step on a batch of VLA embeddings.

        Args:
            z: VLA embeddings [B, M, D] (from EmbeddingExtractor).
            pad_mask: Boolean mask [B, M] (True = valid token).

        Returns:
            Dict of logged metrics (loss, etc.).
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
        """Extract VLA embeddings from observations, then train.

        Convenience method that combines embedding extraction with
        a training step.  Use this when iterating over a demonstration
        dataset of raw observations.

        Args:
            vla: Frozen VLA wrapper for embedding extraction.
            observations: Batched observation dict for the VLA.

        Returns:
            Dict of logged metrics.
        """
        with torch.no_grad():
            z, pad_mask = vla.extract_embeddings(observations)
        return self.train_step(z, pad_mask)

    def train(
        self,
        dataloader: Iterator[tuple[Tensor, Tensor]],
        log_fn: Any | None = None,
    ) -> None:
        """Run the full training loop.

        Args:
            dataloader: Iterator yielding (z, pad_mask) batches of
                pre-extracted VLA embeddings.
            log_fn: Optional callable ``log_fn(metrics_dict)`` for logging.
        """
        logger.info("Starting Stage 1 training for %d steps", self.config.num_train_steps)

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
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self._global_step,
                "config": self.config,
            },
            ckpt_path,
        )
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
        logger.info("Loaded checkpoint from %s (step %d)", ckpt_path, self._global_step)
