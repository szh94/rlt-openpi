"""JAX-based VLA wrapper for RLT-OpenPI.

Provides :class:`JaxVLAWrapper` and :class:`JaxEmbeddingExtractor` that use
the native JAX Pi0 model (from ``openpi-main/src/openpi/models/pi0.py``)
instead of the PyTorch port.  All RLT components (RLTokenModel, Actor,
TwinQCritic) remain in PyTorch — the JAX↔PyTorch bridge converts arrays
at the output boundary.

Differences from the PyTorch :class:`VLAWrapper`:

* Loads Orbax checkpoints (directory) instead of ``.safetensors`` files.
* Joint training (``vla_finetune_alpha > 0``) is **not supported** — a JAX
  NNX model cannot be optimised by a PyTorch optimizer.
* :meth:`preprocess_obs` returns torch tensors on **CPU** so that
  ``RolloutWorker`` can call ``.to(device=...)`` without redundant copies.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from openpi.models import model as _model
from openpi.models.model import Observation
from openpi.models.pi0 import make_attn_mask
from openpi.training import checkpoints as _checkpoints
from openpi.transforms import InjectDefaultPrompt, Normalize, Unnormalize, compose
import openpi.transforms as _transforms
from torch import Tensor

from rlt_openpi.vla.config import load_vla_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SliceAction:
    """Simple transform that slices actions to the first N dimensions.

    Used as a replacement for DroidOutputs when the output action dimension
    differs from the default registered data config.
    """

    def __init__(self, action_dim: int) -> None:
        self._action_dim = action_dim

    def __call__(self, data: dict) -> dict:
        data["actions"] = data["actions"][:, :self._action_dim]
        return data


def _observation_to_numpy(obs: Observation) -> Observation:
    """Convert an Observation containing torch tensors to numpy arrays for JAX."""

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    images = {k: _to_np(v) for k, v in obs.images.items()}
    image_masks = {k: _to_np(v) for k, v in obs.image_masks.items()}
    state = _to_np(obs.state)
    tokenized_prompt = _to_np(obs.tokenized_prompt) if obs.tokenized_prompt is not None else None
    tokenized_prompt_mask = (
        _to_np(obs.tokenized_prompt_mask) if obs.tokenized_prompt_mask is not None else None
    )

    return Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


# ---------------------------------------------------------------------------
# JaxEmbeddingExtractor
# ---------------------------------------------------------------------------


class JaxEmbeddingExtractor:
    """Wraps a JAX Pi0 model for embedding extraction and action sampling.

    Unlike the PyTorch ``EmbeddingExtractor``, this is **not** a
    ``torch.nn.Module``.  It holds a reference to the native JAX NNX
    model and converts arrays at the boundary::

        torch → numpy → JAX → numpy → torch

    The JAX model runs on whatever device JAX is configured to use
    (typically GPU when available).
    """

    def __init__(self, pi0_model) -> None:
        self.pi0 = pi0_model
        # Per-instance RNG key for action sampling.
        self._rng = jax.random.PRNGKey(0)

    def _next_rng(self):
        self._rng, rng = jax.random.split(self._rng)
        return rng

    def extract_embeddings(self, observation: Observation) -> tuple[Tensor, Tensor]:
        """Extract post-transformer prefix embeddings from the frozen JAX VLA.

        Replicates the prefix-only forward pass from
        ``Pi0.sample_actions``::

            embed_prefix → make_attn_mask → PaliGemma.llm([prefix, None])

        Args:
            observation: Batched openpi Observation (torch tensors).

        Returns:
            z: Post-transformer prefix embeddings [B, M, 2048] (float32).
            pad_mask: Boolean padding mask [B, M] (True = valid token).
        """
        obs_np = _observation_to_numpy(observation)

        # JAX prefix embedding forward pass.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.pi0.embed_prefix(obs_np)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        (prefix_out, _), _kv_cache = self.pi0.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        # prefix_out: [B, M, embed_dim], _ is None (suffix slot)

        # Convert back to torch tensors.
        z = torch.from_numpy(np.array(prefix_out)).to(dtype=torch.float32)
        pad_mask = torch.from_numpy(np.array(prefix_mask)).to(dtype=torch.bool)

        return z, pad_mask

    def sample_actions(
        self,
        observation: Observation,
        num_steps: int = 10,
    ) -> Tensor:
        """Get reference actions from the JAX VLA via diffusion sampling.

        Delegates to ``Pi0.sample_actions``, which internally calls
        ``preprocess_observation`` (a near-no-op when the observation
        is already preprocessed by ``JaxVLAWrapper.preprocess_obs``).

        Args:
            observation: Batched openpi Observation (torch tensors).
            num_steps: Number of diffusion denoising steps (default 10).

        Returns:
            actions: VLA output [B, action_horizon, action_dim] (float32).
        """
        obs_np = _observation_to_numpy(observation)
        rng = self._next_rng()
        actions = self.pi0.sample_actions(rng, obs_np, num_steps=num_steps)
        return torch.from_numpy(np.array(actions)).to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# JaxVLAWrapper
# ---------------------------------------------------------------------------


class JaxVLAWrapper:
    """Loads and wraps a **JAX** VLA model for use by RLT components.

    Public interface mirrors :class:`VLAWrapper` exactly so that
    ``RolloutWorker`` and training scripts can use either wrapper
    interchangeably (duck-typing).

    Supported operations:
    * ``preprocess_obs`` — raw env obs → batched Observation
    * ``extract_embeddings`` — post-transformer prefix embeddings z_{1:M}
    * ``sample_reference_actions`` — full VLA action trajectory
    * ``get_rl_chunk_reference`` — first C steps as RL reference

    **Unsupported** (raise ``NotImplementedError``):
    * ``compute_vla_loss``
    * ``compute_vla_loss_with_embeddings``
    * ``unfreeze``
    * ``trainable_parameters``

    Args:
        checkpoint_dir: Path to the Orbax checkpoint directory (the
            directory containing ``params/`` and optionally ``assets/``).
        config_name: Registered openpi config name (e.g. ``"pi05_droid_finetune"``).
        device: Torch device where output tensors are placed.
        data_transforms: Optional override for the config's default
            ``data_transforms``.
        default_prompt: If set, injected into inputs that lack a ``prompt`` key.
        output_action_dim: If set, slices output actions to this dimension
            instead of using the default ``DroidOutputs`` transform.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        config_name: str,
        device: torch.device | str = "cuda",
        data_transforms: _transforms.Group | None = None,
        default_prompt: str | None = None,
        output_action_dim: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.train_config = load_vla_config(config_name)

        # Load the JAX model from an Orbax checkpoint.
        checkpoint_dir_path = pathlib.Path(checkpoint_dir)
        logger.info("Loading JAX VLA from Orbax checkpoint: %s", checkpoint_dir_path)
        params = _model.restore_params(checkpoint_dir_path, restore_type=np.ndarray)
        self._pi0_model = self.train_config.model.load(params)
        logger.info("JAX Pi0 model loaded successfully.")

        self.extractor = JaxEmbeddingExtractor(self._pi0_model)
        self.action_dim = self.train_config.model.action_dim
        self.action_horizon = self.train_config.model.action_horizon

        # Build transform chains (identical logic to VLAWrapper).
        data_config = self.train_config.data.create(
            self.train_config.assets_dirs, self.train_config.model
        )
        use_q = data_config.use_quantile_norm

        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = self._load_norm_stats(checkpoint_dir_path, data_config)

        dt = data_transforms or data_config.data_transforms

        output_transforms = [
            *data_config.model_transforms.outputs,
            Unnormalize(norm_stats, use_quantiles=use_q),
        ]
        if output_action_dim is not None:
            output_transforms.append(_SliceAction(output_action_dim))
            logger.info(
                "[JaxVLA] Replaced DroidOutputs with SliceAction(dim=%d)", output_action_dim
            )
        else:
            output_transforms.extend(dt.outputs)

        self._input_transform = compose([
            *dt.inputs,
            InjectDefaultPrompt(default_prompt),
            Normalize(norm_stats, use_quantiles=use_q),
            *data_config.model_transforms.inputs,
        ])
        self._output_transform = compose(output_transforms)

    # ------------------------------------------------------------------
    # Public interface (mirrors VLAWrapper)
    # ------------------------------------------------------------------

    def preprocess_obs(self, obs: dict[str, Any]) -> Observation:
        """Convert a raw environment observation into a batched Observation.

        Applies the full OpenPI input transform chain and returns torch
        tensors on **CPU** so that ``RolloutWorker`` can move them to the
        target device via ``.to(device=...)`` without a redundant copy.

        Args:
            obs: Raw observation dict from the environment.
        """
        transformed = self._input_transform(dict(obs))

        batched = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x))[None, ...],
            transformed,
        )
        return Observation.from_dict(batched)

    def extract_embeddings(self, observation: Observation) -> tuple[Tensor, Tensor]:
        """Extract post-transformer prefix embeddings from the frozen VLA.

        Returns:
            z: [B, M, embedding_dim] post-transformer prefix embeddings.
            pad_mask: [B, M] boolean mask (True = valid token).
        """
        z, pad_mask = self.extractor.extract_embeddings(observation)
        return z.to(self.device), pad_mask.to(self.device)

    def sample_reference_actions(self, observation: Observation) -> Tensor:
        """Get full VLA reference action trajectory, unnormalized to robot space.

        Mirrors OpenPI's ``Policy.infer`` output path.  The output
        transform chain (Unnormalize → DroidOutputs or SliceAction)
        converts actions from normalized space to robot space.

        Returns:
            actions: [B, H, output_action_dim] where H = action_horizon.
        """
        raw = self.extractor.sample_actions(observation)  # [B, H, action_dim] torch
        actions_np = raw.cpu().numpy()
        state_np = observation.state.cpu().numpy()

        out = []
        for i in range(actions_np.shape[0]):
            actions_in = actions_np[i]  # [H, action_dim_raw]
            t = self._output_transform({
                "state": state_np[i],
                "actions": actions_in,
            })
            out.append(t["actions"])
        return torch.as_tensor(np.stack(out), device=self.device)

    def get_rl_chunk_reference(
        self,
        observation: Observation,
        chunk_length: int = 10,
    ) -> Tensor:
        """Get the first C action steps from the VLA as the RL reference.

        Args:
            observation: Batched openpi Observation.
            chunk_length: Number of action steps to slice (C, default 10).

        Returns:
            a_tilde: [B, C, action_dim] reference actions for the RL chunk.
        """
        full_actions = self.sample_reference_actions(observation)
        return full_actions[:, :chunk_length, :]

    # ------------------------------------------------------------------
    # Unsupported operations
    # ------------------------------------------------------------------

    def compute_vla_loss(self, observation, actions) -> Tensor:
        """Not supported: JAX NNX model cannot be trained by PyTorch."""
        raise NotImplementedError(
            "compute_vla_loss is not supported for JAX VLA. "
            "Joint training requires the PyTorch VLAWrapper."
        )

    def compute_vla_loss_with_embeddings(self, observation, actions) -> tuple[Tensor, Tensor, Tensor]:
        """Not supported: JAX NNX model cannot be trained by PyTorch."""
        raise NotImplementedError(
            "compute_vla_loss_with_embeddings is not supported for JAX VLA. "
            "Joint training requires the PyTorch VLAWrapper."
        )

    def unfreeze(self) -> None:
        """Not supported: JAX NNX model cannot be trained by PyTorch."""
        raise NotImplementedError(
            "unfreeze is not supported for JAX VLA. "
            "Joint training requires the PyTorch VLAWrapper."
        )

    def trainable_parameters(self):
        """Not supported: JAX NNX model cannot be trained by PyTorch."""
        raise NotImplementedError(
            "trainable_parameters is not supported for JAX VLA. "
            "Joint training requires the PyTorch VLAWrapper."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_norm_stats(
        checkpoint_dir: pathlib.Path,
        data_config,
    ) -> dict[str, _transforms.NormStats]:
        """Load norm stats, preferring checkpoint-embedded assets over config assets.

        Mirrors OpenPI's ``create_trained_policy`` which loads from
        ``checkpoint_dir/assets/<asset_id>/`` to guarantee the stats
        match training.  Falls back to the config's ``norm_stats`` for
        checkpoints that don't bundle assets.
        """
        asset_id = data_config.asset_id
        try:
            norm_stats = _checkpoints.load_norm_stats(
                checkpoint_dir / "assets", asset_id
            )
            logger.info(
                "Loaded norm stats from checkpoint: %s/assets/%s",
                checkpoint_dir, asset_id,
            )
            return norm_stats
        except FileNotFoundError:
            pass

        if data_config.norm_stats is not None:
            logger.info(
                "Checkpoint has no embedded assets; using norm stats from config assets dir"
            )
            return data_config.norm_stats

        raise FileNotFoundError(
            f"No norm stats found in checkpoint ({checkpoint_dir}/assets/{asset_id}) "
            f"or config assets dir. Run compute_norm_stats.py first."
        )
