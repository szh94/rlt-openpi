from __future__ import annotations

import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from openpi.models import model as _model
from openpi.models.model import Observation
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.transforms import InjectDefaultPrompt, Normalize, Unnormalize, compose
import openpi.transforms as _transforms
from torch import Tensor

from rlt_openpi.vla.config import load_vla_config


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
    """Convert an Observation containing torch/jnp tensors to numpy arrays for JAX."""

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, jnp.ndarray):
            return np.array(x)
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
    """Wraps a JAX Pi0 model for embedding extraction.

    ``extract_embeddings`` goes through the single JIT-compiled
    :meth:`pi0_model.sample_actions`, which returns
    ``((prefix_out, prefix_mask), actions)`` — the prefix embeddings
    are kept and actions discarded.  This follows the same JIT strategy
    as :class:`openpi.policies.policy.Policy`.
    """

    def __init__(self, pi0_model) -> None:
        self.pi0 = pi0_model
        self._rng = jax.random.key(0)

        # JIT-compile: extract_prefix_embeddings runs only 1 LLM pass (prefix only,
        # no diffusion). ~11x faster than sample_actions for embedding extraction.
        self._extract_prefix = nnx_utils.module_jit(pi0_model.extract_prefix_embeddings)

        # sample_actions kept for extract_both (needs actions too).
        self._sample_actions = nnx_utils.module_jit(pi0_model.sample_actions)

    def _next_rng(self):
        self._rng, rng = jax.random.split(self._rng)
        return rng

    def extract_embeddings(self, observation: Observation) -> tuple[Tensor, Tensor]:
        """Extract post-transformer prefix embeddings via JIT-compiled Pi0.

        Uses ``extract_prefix_embeddings`` which runs a single LLM forward pass
        on the prefix tokens only — no diffusion denoising loop.

        Returns:
            z: [B, M, embedding_dim] float32.
            pad_mask: [B, M] bool.
        """
        t0 = time.monotonic()
        obs_np = _observation_to_numpy(observation)
        t1 = time.monotonic()

        prefix_out, prefix_mask = self._extract_prefix(obs_np)
        t2 = time.monotonic()

        # DLPack zero-copy: JAX GPU → PyTorch GPU, avoids ~945ms GPU→CPU roundtrip.
        # .clone() ensures the PyTorch tensor owns its memory independently from JAX.
        z = torch.utils.dlpack.from_dlpack(prefix_out.__dlpack__()).to(dtype=torch.float32)
        pad_mask = torch.utils.dlpack.from_dlpack(prefix_mask.__dlpack__()).to(dtype=torch.bool)
        t3 = time.monotonic()

        t_obs_to_numpy = (t1 - t0) * 1000
        t_extract_prefix = (t2 - t1) * 1000
        t_dlpack_to_torch = (t3 - t2) * 1000
        t_total = (t3 - t0) * 1000

        print(
            f"[JaxEmbeddingExtractor] "
            f"obs_to_numpy={t_obs_to_numpy:.1f}ms | "
            f"extract_prefix(JIT)={t_extract_prefix:.1f}ms | "
            f"dlpack_to_torch={t_dlpack_to_torch:.1f}ms | "
            f"total={t_total:.1f}ms"
        )

        return z, pad_mask



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
    * ``extract_both`` — single forward pass returning both embeddings and robot-space actions

    **Unsupported** (VLA joint training disabled — no stubs remain):
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
        checkpoint_params_path = pathlib.Path(checkpoint_dir + "/params")

        print(f"Loading JAX VLA from Orbax checkpoint: {checkpoint_params_path}")
        params = _model.restore_params(checkpoint_params_path, restore_type=np.ndarray)
        self._pi0_model = self.train_config.model.load(params)
        print("JAX Pi0 model loaded successfully.")

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
            print(
                f"[JaxVLA] Replaced DroidOutputs with SliceAction(dim={output_action_dim})"
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

        Applies the full OpenPI input transform chain and returns jnp
        arrays (same as the JAX path in ``policy.py``), so images stay in
        NHWC format expected by the JAX model.

        Args:
            obs: Raw observation dict from the environment.
        """
        print(f"[DEBUG] In preprocess_obs")
        print(f"[DEBUG] obs.state before trans = {obs['state']}")
        transformed = self._input_transform(dict(obs))
        print(f"[DEBUG] obs.state after trans = {transformed['state']}")

        # Use jnp (NOT torch) so Observation.from_dict keeps images in NHWC
        # format, matching JAX model expectations (same as policy.py JAX path).
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed,)
        return Observation.from_dict(batched)

    def extract_embeddings(self, observation: Observation) -> tuple[Tensor, Tensor]:
        """Extract post-transformer prefix embeddings from the frozen VLA.

        Returns:
            z: [B, M, embedding_dim] post-transformer prefix embeddings.
            pad_mask: [B, M] boolean mask (True = valid token).
        """
        t0 = time.monotonic()
        z, pad_mask = self.extractor.extract_embeddings(observation)
        t1 = time.monotonic()
        z = z.to(self.device)
        pad_mask = pad_mask.to(self.device)
        t2 = time.monotonic()

        t_extract = (t1 - t0) * 1000
        t_to_device = (t2 - t1) * 1000
        t_total = (t2 - t0) * 1000
        print(
            f"[JaxVLAWrapper.extract_embeddings] "
            f"extract={t_extract:.1f}ms | "
            f"to_device={t_to_device:.1f}ms | "
            f"total={t_total:.1f}ms"
        )

        return z, pad_mask

    def extract_both(
        self,
        observation: Observation,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single VLA forward pass returning embeddings, pad_mask, and robot-space actions.

        Combines ``extract_embeddings`` and ``sample_reference_actions``
        into one JIT-compiled forward pass, avoiding a redundant second
        call to the Pi0 model.

        Returns:
            z: [B, M, embedding_dim] post-transformer prefix embeddings.
            pad_mask: [B, M] boolean mask (True = valid token).
            actions: [B, H, output_action_dim] unnormalized robot-space actions.
        """
        obs_np = _observation_to_numpy(observation)
        rng = self.extractor._next_rng()
        (prefix_out, prefix_mask), raw_actions = self.extractor._sample_actions(rng, obs_np)

        # DLPack zero-copy: JAX GPU → PyTorch GPU (avoid GPU→CPU roundtrip).
        z = torch.utils.dlpack.from_dlpack(prefix_out.__dlpack__()).to(
            dtype=torch.float32, device=self.device
        )
        pad_mask = torch.utils.dlpack.from_dlpack(prefix_mask.__dlpack__()).to(
            dtype=torch.bool, device=self.device
        )

        # Apply output transform chain (Unnormalize → DroidOutputs / SliceAction)
        raw_t = torch.utils.dlpack.from_dlpack(raw_actions.__dlpack__()).to(dtype=torch.float32)
        actions_np = raw_t.cpu().numpy()
        state_np = obs_np.state  # already numpy from _observation_to_numpy

        out = []
        for i in range(actions_np.shape[0]):
            actions_in = actions_np[i]
            t = self._output_transform({
                "state": state_np[i],
                "actions": actions_in,
            })
            out.append(t["actions"])
        actions = torch.as_tensor(np.stack(out), device=self.device, dtype=torch.float32)

        return z, pad_mask, actions

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
            print(
                f"Loaded norm stats from checkpoint: {checkpoint_dir}/assets/{asset_id}"
            )
            return norm_stats
        except FileNotFoundError:
            pass

        if data_config.norm_stats is not None:
            return data_config.norm_stats

        raise FileNotFoundError(
            f"No norm stats found in checkpoint ({checkpoint_dir}/assets/{asset_id}) "
            f"or config assets dir. Run compute_norm_stats.py first."
        )
