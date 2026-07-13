"""High-level wrapper for VLA inference, embedding extraction, and joint training (JAX).

Loads a PI0/PI0.5 model directly from an Orbax checkpoint (no PyTorch bridge),
and exposes the interface used by the rollout worker and trainers.

Builds the same input/output transform chains as OpenPI's
``create_trained_policy`` so that normalisation, camera layout, and
action slicing are fully config-driven.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from openpi.models import model as _model
from openpi.models.model import Observation
from openpi.training import checkpoints as _checkpoints
from openpi.transforms import InjectDefaultPrompt, Normalize, Unnormalize, compose
import openpi.transforms as _transforms

from rlt_openpi.vla.config import load_vla_config

logger = logging.getLogger(__name__)


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


class VLAWrapper:
    """Loads and wraps a JAX VLA model for use by RLT components.

    Provides:
    - preprocess_obs: raw env obs -> batched Observation (JAX arrays)
    - extract_embeddings: get post-transformer prefix embeddings z_{1:M}
    - sample_reference_actions: get full VLA action trajectory (H steps)
    - get_rl_chunk_reference: slice first C steps as RL reference actions
    - compute_vla_loss: VLA flow-matching loss (standalone)
    - compute_vla_loss_with_embeddings: single forward for joint training

    The input/output transform chains mirror OpenPI's
    ``create_trained_policy`` (``policy_config.py``).

    Args:
        checkpoint_path: Path to Orbax checkpoint directory (containing
            ``params/`` subdirectory).
        config_name: Registered openpi config name (e.g. "pi05_droid_finetune").
        data_transforms: Optional override for the config's default
            ``data_transforms``.
        default_prompt: If set, injected into inputs that lack a ``prompt`` key.
        output_action_dim: Optional override for action dimension slicing.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_name: str,
        data_transforms: _transforms.Group | None = None,
        default_prompt: str | None = None,
        output_action_dim: int | None = None,
    ) -> None:
        self.train_config = load_vla_config(config_name)

        # Load JAX model directly from Orbax checkpoint (no PyTorch bridge)
        checkpoint_path = pathlib.Path(checkpoint_path)

        # Detect checkpoint format: params/ dir = Orbax, .safetensors = PyTorch
        params_dir = checkpoint_path / "params"
        if params_dir.exists():
            params = _model.restore_params(params_dir)
        elif checkpoint_path.suffix == ".safetensors":
            # Fallback: safetensors in parent dir, look for params/ nearby
            alt_params = checkpoint_path.parent / "params"
            if alt_params.exists():
                params = _model.restore_params(alt_params)
            else:
                raise FileNotFoundError(
                    f"No Orbax params/ directory found at {checkpoint_path.parent}/params. "
                    "JAX-native loading requires an Orbax checkpoint. "
                    "Use convert_jax_model_to_pytorch.py to convert from PyTorch."
                )
        else:
            # Try loading checkpoint_path directly as params dir
            params = _model.restore_params(checkpoint_path)

        self._jax_model = self.train_config.model.load(params)
        logger.info("Loaded JAX VLA model from %s", checkpoint_path)

        self.action_dim = self.train_config.model.action_dim
        self.action_horizon = self.train_config.model.action_horizon

        checkpoint_dir = checkpoint_path.parent if checkpoint_path.suffix == ".safetensors" else checkpoint_path
        data_config = self.train_config.data.create(
            self.train_config.assets_dirs, self.train_config.model
        )
        use_q = data_config.use_quantile_norm

        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = self._load_norm_stats(checkpoint_dir, data_config)

        dt = data_transforms or data_config.data_transforms

        # Output transform chain
        output_transforms = [
            *data_config.model_transforms.outputs,
            Unnormalize(norm_stats, use_quantiles=use_q),
        ]
        if output_action_dim is not None:
            output_transforms.append(_SliceAction(output_action_dim))
        else:
            output_transforms.extend(dt.outputs)

        self._input_transform = compose([
            *dt.inputs,
            InjectDefaultPrompt(default_prompt),
            Normalize(norm_stats, use_quantiles=use_q),
            *data_config.model_transforms.inputs,
        ])
        self._output_transform = compose(output_transforms)

        # RNG state for action sampling
        self._rng = jax.random.PRNGKey(0)

        # Store model graphdef for serialization
        self._model_graphdef, _ = nnx.split(self._jax_model)

    def _next_rng(self):
        self._rng, rng = jax.random.split(self._rng)
        return rng

    @staticmethod
    def _load_norm_stats(
        checkpoint_dir: pathlib.Path, data_config
    ) -> dict[str, _transforms.NormStats]:
        """Load norm stats, preferring checkpoint-embedded assets over config assets."""
        asset_id = data_config.asset_id
        try:
            norm_stats = _checkpoints.load_norm_stats(
                checkpoint_dir / "assets", asset_id
            )
            logger.info("Loaded norm stats from checkpoint: %s/assets/%s", checkpoint_dir, asset_id)
            return norm_stats
        except FileNotFoundError:
            pass

        if data_config.norm_stats is not None:
            logger.info("Checkpoint has no embedded assets; using norm stats from config assets dir")
            return data_config.norm_stats

        raise FileNotFoundError(
            f"No norm stats found in checkpoint ({checkpoint_dir}/assets/{asset_id}) "
            f"or config assets dir. Run compute_norm_stats.py first."
        )

    def preprocess_obs(self, obs: dict[str, Any]) -> Observation:
        """Convert a raw environment observation into a batched Observation.

        Applies the full OpenPI input transform chain and converts to JAX arrays.
        """
        transformed = self._input_transform(dict(obs))

        batched = jax.tree.map(
            lambda x: jnp.asarray(x)[None, ...],
            transformed,
        )
        return Observation.from_dict(batched)

    def extract_embeddings(self, observation: Observation):
        """Extract post-transformer prefix embeddings from the frozen VLA.

        Returns:
            z: [B, M, embedding_dim] post-transformer prefix embeddings (JAX).
            pad_mask: [B, M] boolean mask (True = valid token) (JAX).
        """
        from rlt_openpi.vla.embedding_extractor import extract_embeddings

        return extract_embeddings(self._jax_model, observation)

    def sample_reference_actions(self, observation: Observation):
        """Get full VLA reference action trajectory, unnormalized to robot space.

        Returns:
            actions: [B, H, robot_action_dim] JAX array, where H = action_horizon.
        """
        from rlt_openpi.vla.embedding_extractor import sample_actions

        rng = self._next_rng()
        raw = sample_actions(self._jax_model, rng, observation)  # [B, H, action_dim]

        # Convert to numpy for the output transform chain (openpi transforms are numpy-based)
        actions_np = np.array(raw)
        state_np = np.array(observation.state)

        out = []
        for i in range(actions_np.shape[0]):
            t = self._output_transform({
                "state": state_np[i],
                "actions": actions_np[i],
            })
            out.append(t["actions"])
        return jnp.asarray(np.stack(out))

    def get_rl_chunk_reference(
        self,
        observation: Observation,
        chunk_length: int = 10,
    ):
        """Get the first C action steps from the VLA as the RL reference.

        Args:
            observation: Batched openpi Observation.
            chunk_length: Number of action steps to slice (C, default 10).

        Returns:
            a_tilde: [B, C, action_dim] reference actions (JAX array).
        """
        full_actions = self.sample_reference_actions(observation)
        return full_actions[:, :chunk_length, :]

    def compute_vla_loss(self, observation, actions):
        """Compute the VLA's flow-matching training loss on demo data.

        Args:
            observation: Batched Observation (JAX arrays).
            actions: Ground-truth demo actions [B, H, action_dim] (JAX array).

        Returns:
            Scalar mean VLA loss (JAX).
        """
        per_element_loss = self._jax_model.compute_loss(observation, actions)
        return per_element_loss.mean()

    def compute_vla_loss_with_embeddings(self, observation, actions):
        """Single VLA forward pass returning both embeddings and loss.

        Used by joint training (alpha > 0) to avoid the double forward
        pass of calling extract_embeddings() + compute_vla_loss()
        separately.

        Args:
            observation: Batched Observation (JAX arrays).
            actions: Ground-truth demo actions [B, H, action_dim] (JAX array).

        Returns:
            z: Detached prefix embeddings [B, M, D] (stop-grad from VLA).
            pad_mask: Boolean mask [B, M] (True = valid token).
            vla_loss: Scalar VLA flow-matching loss (with grad for VLA).
        """
        from rlt_openpi.vla.embedding_extractor import forward_joint

        return forward_joint(self._jax_model, observation, actions)

    # ------------------------------------------------------------------
    # Joint training helpers
    # ------------------------------------------------------------------

    def unfreeze(self) -> None:
        """No-op in JAX — gradients are controlled by nnx.grad/nnx.DiffState."""
        pass

    def trainable_parameters(self):
        """Return an iterator over trainable VLA parameters (for optimizer setup)."""
        # In JAX/nnx, parameters are accessible via nnx.state()
        return []  # Handled by nnx optimizer directly
