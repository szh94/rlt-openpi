"""Stage 1 trainer: RL token encoder-decoder on demonstration data (JAX/Flax nnx).

Trains the RLTokenModel (encoder-decoder) to compress VLA embeddings
into a single RL token z_rl via reconstruction loss.

Supports two modes (selected automatically by ``vla_finetune_alpha``):
- **Frozen VLA** (alpha=0): Only trains the encoder-decoder on
  on-the-fly VLA embeddings.
- **Joint training** (alpha>0): Simultaneously trains the RL token
  encoder-decoder (L_ro) and fine-tunes the VLA (alpha * L_vla).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from rlt_openpi.models.rl_token import RLTokenModel
from rlt_openpi.training.config import RLTokenTrainConfig
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logger = logging.getLogger(__name__)


class RLTokenTrainer:
    """Stage 1 trainer for the RL token encoder-decoder (JAX).

    Mode is selected automatically by ``config.vla_finetune_alpha``:
    - alpha == 0: frozen VLA, trains encoder-decoder only.
    - alpha > 0:  joint training, fine-tunes VLA alongside encoder-decoder.

    Usage::

        trainer = RLTokenTrainer(config)
        trainer.train(vla, dataloader, log_fn=logger.log)

    Args:
        config: Stage 1 training hyperparameters.
    """

    def __init__(self, config: RLTokenTrainConfig) -> None:
        self.config = config

        # Build RL token model
        self.model = RLTokenModel(
            embedding_dim=config.embedding_dim,
            encoder_layers=config.encoder_layers,
            encoder_heads=config.encoder_heads,
            decoder_layers=config.decoder_layers,
            decoder_heads=config.decoder_heads,
            rngs=nnx.Rngs(0),
        )

        # Optimizer with warmup schedule + gradient clipping
        lr_schedule = _build_lr_schedule(config.learning_rate, config.warmup_steps)
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=config.weight_decay,
            ),
        )
        self.optimizer = nnx.Optimizer(self.model, tx)

        # VLA joint training state
        self._vla: VLAWrapper | None = None
        self._vla_optimizer: nnx.Optimizer | None = None

        self._global_step = 0
        self._rng = jax.random.PRNGKey(42)

    def _next_rng(self):
        self._rng, rng = jax.random.split(self._rng)
        return rng

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
        actions,
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
        dataloader: Iterator[tuple[Any, Any]],
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

            # Convert actions to JAX array if needed
            if hasattr(actions, "numpy"):
                actions = jnp.asarray(actions.numpy())

            metrics = self.step(vla, observations, actions)

            # Progress bar
            if self.joint:
                pbar.set_postfix(
                    l_ro=f"{metrics['l_ro']:.4f}", l_vla=f"{metrics['l_vla']:.4f}"
                )
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
        """Save model and optimizer state via orbax.

        Args:
            path: Override save path. Defaults to config.save_dir.

        Returns:
            Path to the saved checkpoint directory.
        """
        import orbax.checkpoint as ocp

        save_dir = Path(path or self.config.save_dir) / self.config.run_name
        ckpt_dir = save_dir / f"rl_token_step{self._global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpointer = ocp.PyTreeCheckpointer()

        _, model_state = nnx.split(self.model)
        payload = {
            "model": model_state.to_pure_dict(),
            "optimizer": nnx.state(self.optimizer).to_pure_dict(),
            "step": self._global_step,
            # Architecture info for reconstruction
            "embedding_dim": 2048,
            "encoder_layers": 2,
            "encoder_heads": 8,
            "decoder_layers": 2,
            "decoder_heads": 8,
        }

        if self._vla is not None and self._vla._jax_model is not None:
            _, vla_state = nnx.split(self._vla._jax_model)
            payload["vla_model"] = vla_state.to_pure_dict()
        if self._vla_optimizer is not None:
            payload["vla_optimizer"] = nnx.state(self._vla_optimizer).to_pure_dict()

        checkpointer.save(str(ckpt_dir / "params"), payload)
        logger.info("Saved checkpoint to %s", ckpt_dir)
        return ckpt_dir

    def load(self, ckpt_path: str) -> None:
        """Load model and optimizer state from checkpoint.

        Args:
            ckpt_path: Path to a saved checkpoint directory.
        """
        import orbax.checkpoint as ocp

        ckpt_path = Path(ckpt_path)
        params_dir = ckpt_path / "params"
        if not params_dir.exists():
            params_dir = ckpt_path

        checkpointer = ocp.PyTreeCheckpointer()
        ckpt = checkpointer.restore(str(params_dir))

        # Restore model
        model_graphdef, _ = nnx.split(self.model)
        nnx.update(
            self.model,
            nnx.State.from_pure_dict(model_graphdef, ckpt["model"]),
        )

        # Restore optimizer
        opt_graphdef, _ = nnx.split(self.optimizer)
        nnx.update(
            self.optimizer,
            nnx.State.from_pure_dict(opt_graphdef, ckpt["optimizer"]),
        )

        self._global_step = int(ckpt["step"])

        # Restore VLA weights if present
        if "vla_model" in ckpt and self._vla is not None and self._vla._jax_model is not None:
            vla_graphdef, _ = nnx.split(self._vla._jax_model)
            nnx.update(
                self._vla._jax_model,
                nnx.State.from_pure_dict(vla_graphdef, ckpt["vla_model"]),
            )
            logger.info("Restored fine-tuned VLA weights from checkpoint")

        # Restore VLA optimizer if present
        if "vla_optimizer" in ckpt and self._vla_optimizer is not None:
            vla_opt_gd, _ = nnx.split(self._vla_optimizer)
            nnx.update(
                self._vla_optimizer,
                nnx.State.from_pure_dict(vla_opt_gd, ckpt["vla_optimizer"]),
            )

        logger.info("Loaded checkpoint from %s (step %d)", ckpt_path, self._global_step)

    # ------------------------------------------------------------------
    # Private: mode-specific steps
    # ------------------------------------------------------------------

    def _setup_joint_training(self, vla: VLAWrapper) -> None:
        """Unfreeze VLA and create its optimizer (called once by train())."""
        self._vla = vla

        if vla._jax_model is not None:
            lr_schedule = _build_lr_schedule(
                self.config.vla_learning_rate, self.config.warmup_steps
            )
            tx = optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adamw(
                    learning_rate=lr_schedule,
                    weight_decay=self.config.weight_decay,
                ),
            )
            self._vla_optimizer = nnx.Optimizer(vla._jax_model, tx)
            logger.info(
                "VLA joint training enabled: %d trainable parameters",
                sum(
                    np.prod(leaf.shape)
                    for leaf in jax.tree_util.tree_leaves(
                        nnx.state(vla._jax_model).to_pure_dict()
                    )
                ),
            )

    def _step_frozen(
        self,
        vla: VLAWrapper,
        observations: Any,
    ) -> dict[str, float]:
        """Frozen VLA step: extract embeddings (no grad) -> L_ro only."""
        # Extract VLA embeddings as JAX arrays
        z, pad_mask = vla.extract_embeddings(observations)

        # Compute loss and gradients
        def loss_fn(model):
            loss, _z_rl, _z_hat = model.forward(z, pad_mask)
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(self.model)
        self.optimizer.update(grads)

        self._global_step += 1
        return {
            "loss": float(loss),
            "grad_norm": _compute_grad_norm(grads),
            "lr": float(self.config.learning_rate),
            "step": self._global_step,
        }

    def _step_joint(
        self,
        vla: VLAWrapper,
        observations: Any,
        actions,
    ) -> dict[str, float]:
        """Joint step: single VLA forward -> L_ro + alpha * L_vla."""
        alpha = self.config.vla_finetune_alpha

        # Single VLA forward: detached embeddings + flow-matching loss
        z, pad_mask, l_vla = vla.compute_vla_loss_with_embeddings(observations, actions)

        # L_ro: RL token reconstruction loss
        def _rl_loss_fn(model):
            loss, _z_rl, _z_hat = model.forward(z, pad_mask)
            return loss

        l_ro, ro_grads = nnx.value_and_grad(_rl_loss_fn)(self.model)
        self.optimizer.update(ro_grads)

        # Update VLA if optimizer is set up
        vla_grad_norm = 0.0
        if self._vla_optimizer is not None and self._vla is not None:
            # VLA loss grads are computed by differentiating through l_vla
            # Since l_vla already has grad tracking from the VLA forward pass,
            # we need to compute gradients w.r.t. the VLA model.
            # l_vla is a scalar from the VLA's loss computation.
            # We re-derive the VLA loss to get gradients.
            def _vla_loss_fn(vla_model):
                # Re-run VLA forward to get VLA loss with grad tracking
                return jnp.array(l_vla)  # l_vla is already a scalar

            # The VLA loss was computed during the joint forward pass.
            # We compute gradients of l_vla w.r.t. VLA params.
            if self._vla._jax_model is not None:
                vla_grads = nnx.grad(lambda m: alpha * l_vla)(self._vla._jax_model)
                self._vla_optimizer.update(vla_grads)
                vla_grad_norm = _compute_grad_norm(vla_grads)

        self._global_step += 1
        return {
            "loss": float(l_ro + alpha * l_vla),
            "l_ro": float(l_ro),
            "l_vla": float(l_vla),
            "grad_norm": _compute_grad_norm(ro_grads),
            "vla_grad_norm": float(vla_grad_norm),
            "lr": float(self.config.learning_rate),
            "step": self._global_step,
        }


def _build_lr_schedule(peak_lr: float, warmup_steps: int):
    """Build a linear warmup -> constant LR schedule."""
    if warmup_steps <= 0:
        return optax.constant_schedule(peak_lr)
    return optax.warmup_constant_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
    )


def _compute_grad_norm(grads) -> float:
    """Compute global gradient norm from nnx grads State."""
    total = 0.0
    for leaf in jax.tree_util.tree_leaves(grads.to_pure_dict()):
        if leaf is not None:
            total += jnp.sum(leaf ** 2)
    return float(jnp.sqrt(total))
