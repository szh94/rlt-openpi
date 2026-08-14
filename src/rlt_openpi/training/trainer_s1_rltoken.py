"""Stage 1 trainer: RL token encoder-decoder on demonstration data.

Trains the RLTokenModel (encoder-decoder) to compress VLA embeddings
into a single RL token z_rl via reconstruction loss.

The VLA remains frozen while the RL token model is trained.
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

        self._global_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        vla: VLAWrapper,
        dataloader: Iterator[tuple[Any, Tensor]],
        log_fn: Any | None = None,
    ) -> None:
        """Run the full training loop.

        The VLA remains frozen throughout training.

        Args:
            vla: VLA wrapper.
            dataloader: Infinite iterator yielding (observations, actions).
            log_fn: Optional callable ``log_fn(metrics_dict)`` for logging.
        """
        print(f"[Stage 1] Starting frozen-VLA training for {self.config.num_train_steps} steps")

        if self.config.resume_checkpoint:
            self.load(self.config.resume_checkpoint)
            print(f"[Stage 1] Resumed from step {self._global_step}")

        pbar = tqdm(range(1, self.config.num_train_steps + 1), desc="Stage 1")
        for step_idx in pbar:
            t0 = time.monotonic()
            try:
                observations, actions = next(dataloader)
            except StopIteration:
                print(f"[Stage 1] WARNING: Dataloader exhausted at step {step_idx}")
                break
            t1 = time.monotonic()

            metrics = self.step_frozen(vla, observations)
            t2 = time.monotonic()

            # Progress bar
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}")
            t3 = time.monotonic()

            # wandb logging (every log_every steps)
            if step_idx % self.config.log_every == 0 and log_fn is not None:
                log_fn(metrics, step=metrics.get("step"))
            t4 = time.monotonic()

            if step_idx % self.config.save_every == 0:
                self.save()
            t5 = time.monotonic()

            print(
                f"[Train] step = {step_idx} | "
                f"data_load={(t1 - t0) * 1000:.1f}ms | "
                f"train_step={(t2 - t1) * 1000:.1f}ms | "
                f"progress={(t3 - t2) * 1000:.1f}ms | "
                f"logging={(t4 - t3) * 1000:.1f}ms | "
                f"checkpoint={(t5 - t4) * 1000:.1f}ms | "
                f"total={(t5 - t0) * 1000:.1f}ms"
            )

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
        print(f"[Stage 1] Loaded checkpoint from {ckpt_path} (step {self._global_step})")

    # ------------------------------------------------------------------
    # Training helpers
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

    def step_frozen(
        self,
        vla: VLAWrapper,
        observations: Any,
    ) -> dict[str, float]:
        """Frozen VLA step: extract embeddings (no grad) → L_ro only."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.monotonic()
        self.model.train()

        observations = _obs_to_device(observations, self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t1 = time.monotonic()

        with torch.no_grad():
            z, pad_mask = vla.extract_embeddings(observations)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t2 = time.monotonic()

        z = z.to(self.device)
        pad_mask = pad_mask.to(self.device)
        loss, _z_rl, _z_hat = self.model(z, pad_mask)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t3 = time.monotonic()

        self.optimizer.zero_grad()
        loss.backward()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t4 = time.monotonic()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t5 = time.monotonic()

        loss_value = loss.item()
        grad_norm_value = grad_norm.item()
        t6 = time.monotonic()

        self._global_step += 1

        # Timing breakout (ms)
        t_obs_to_device = (t1 - t0) * 1000
        t_vla_embed = (t2 - t1) * 1000
        t_rl_forward = (t3 - t2) * 1000
        t_backward = (t4 - t3) * 1000
        t_optimizer = (t5 - t4) * 1000
        t_metrics = (t6 - t5) * 1000
        t_total = (t6 - t0) * 1000

        print(
            f"[train_step] {self._global_step}] "
            f"obs_to_device={t_obs_to_device:.1f}ms | "
            f"vla_embed={t_vla_embed:.1f}ms | "
            f"rl_forward={t_rl_forward:.1f}ms | "
            f"backward={t_backward:.1f}ms | "
            f"optimizer={t_optimizer:.1f}ms | "
            f"metrics={t_metrics:.1f}ms | "
            f"total={t_total:.1f}ms"
        )

        return {
            "loss": loss_value,
            "grad_norm": grad_norm_value,
            "lr": self.optimizer.param_groups[0]["lr"],
            "step": self._global_step,
        }

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
