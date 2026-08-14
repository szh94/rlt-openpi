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
from openpi.transforms import InjectDefaultPrompt, Normalize, Unnormalize
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


def _observation_to_jax(obs: Observation) -> Observation:
    """Convert observation leaves to JAX arrays without staging JAX data on CPU.

    ``preprocess_obs`` already returns device-backed JAX arrays.  Keeping those
    leaves unchanged avoids a JAX device -> NumPy host -> JAX device round trip
    immediately before the JIT-compiled model call.  Torch inputs still need to
    pass through CPU because PyTorch and JAX do not share them here.
    """

    def _to_jax(x):
        if isinstance(x, jax.Array):
            return x
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return jnp.asarray(x)

    images = {k: _to_jax(v) for k, v in obs.images.items()}
    image_masks = {k: _to_jax(v) for k, v in obs.image_masks.items()}
    state = _to_jax(obs.state)
    tokenized_prompt = _to_jax(obs.tokenized_prompt) if obs.tokenized_prompt is not None else None
    tokenized_prompt_mask = (
        _to_jax(obs.tokenized_prompt_mask) if obs.tokenized_prompt_mask is not None else None
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
    """JIT-compiled JAX Pi0 embedding extractor."""

    def __init__(self, pi0_model) -> None:
        self.pi0 = pi0_model
        self._rng = jax.random.key(0)

        # Prefix-only embedding extraction skips diffusion sampling.
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
        observation = _observation_to_jax(observation)
        t1 = time.monotonic()

        prefix_out, prefix_mask = self._extract_prefix(observation)
        jax.block_until_ready((prefix_out, prefix_mask))
        t2 = time.monotonic()

        # Share JAX device buffers with PyTorch through DLPack.
        z = torch.utils.dlpack.from_dlpack(prefix_out.__dlpack__()).to(dtype=torch.float32)
        pad_mask = torch.utils.dlpack.from_dlpack(prefix_mask.__dlpack__()).to(dtype=torch.bool)
        t3 = time.monotonic()

        t_obs_to_jax = (t1 - t0) * 1000
        t_extract_prefix = (t2 - t1) * 1000
        t_dlpack_to_torch = (t3 - t2) * 1000
        t_total = (t3 - t0) * 1000

        print(
            f"[JaxEmbeddingExtractor] "
            f"obs_to_jax={t_obs_to_jax:.1f}ms | "
            f"extract_prefix(JIT)={t_extract_prefix:.1f}ms | "
            f"dlpack_to_torch={t_dlpack_to_torch:.1f}ms | "
            f"total={t_total:.1f}ms"
        )

        return z, pad_mask



# ---------------------------------------------------------------------------
# JaxVLAWrapper
# ---------------------------------------------------------------------------


class JaxVLAWrapper:
    """Frozen JAX Pi0 adapter for RLT training and rollout.

    Raw environment observations use the OpenPI transform chain. Model outputs
    are exposed as PyTorch tensors for downstream RLT components.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        config_name: str,
        device: torch.device | str = "cuda",
        repack_transforms: _transforms.Group | None = None,
        default_prompt: str | None = None,
        output_action_dim: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.train_config = load_vla_config(config_name)
        repack_transforms = repack_transforms or _transforms.Group()

        # Load the JAX model from an Orbax checkpoint.
        checkpoint_dir_path = pathlib.Path(checkpoint_dir)
        checkpoint_params_path = pathlib.Path(checkpoint_dir + "/params")

        print(f"Loading JAX VLA from Orbax checkpoint: {checkpoint_params_path}")
        model_sharding = jax.sharding.SingleDeviceSharding(jax.devices("gpu")[0])
        params = _model.restore_params(
            checkpoint_params_path,
            dtype=jnp.bfloat16,
            sharding=model_sharding,
        )
        self._pi0_model = self.train_config.model.load(params)
        print("JAX Pi0 model loaded successfully.")

        self.extractor = JaxEmbeddingExtractor(self._pi0_model)
        self.action_dim = self.train_config.model.action_dim
        self.action_horizon = self.train_config.model.action_horizon

        # Build the data config and norm stats exactly as create_trained_policy does.
        data_config = self.train_config.data.create(
            self.train_config.assets_dirs, self.train_config.model
        )
        use_q = data_config.use_quantile_norm

        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = self._load_norm_stats(checkpoint_dir_path, data_config)
        # Expose the checkpoint-resolved stats for dataset consumers such as
        # actor pretraining.  Local training configs may not carry stats even
        # though the checkpoint bundles the authoritative assets.
        self.norm_stats = norm_stats
        self.use_quantile_norm = use_q

        dt = data_config.data_transforms

        # Output transform chain, same order as create_trained_policy:
        # model_transforms.outputs → Unnormalize → data_transforms.outputs
        # → repack_transforms.outputs.  When output_action_dim is set, it
        # replaces the data_transforms outputs with a plain slice (RLT).
        output_transforms: list[_transforms.DataTransformFn] = [
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
        output_transforms.extend(repack_transforms.outputs)

        # Match the input transform order used by create_trained_policy().
        transforms: list[_transforms.DataTransformFn] = [
            *repack_transforms.inputs,
            InjectDefaultPrompt(default_prompt),
            *dt.inputs,
            Normalize(norm_stats, use_quantiles=use_q),
            *data_config.model_transforms.inputs,
        ]
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

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
        transformed = self._input_transform(obs)
        print(f"[DEBUG] obs.state after trans = {transformed['state']}")

        # Preserve the NHWC layout expected by the JAX model.
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed,)
        return Observation.from_dict(batched)

    def extract_embeddings(self, observation: Observation) -> tuple[Tensor, Tensor]:
        """Return prefix embeddings and their padding mask as PyTorch tensors."""
        t0 = time.monotonic()
        z, pad_mask = self.extractor.extract_embeddings(observation)
        t1 = time.monotonic()
        z = z.to(self.device)
        pad_mask = pad_mask.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
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
        """Return prefix embeddings, padding mask, and robot-space actions."""
        t0 = time.monotonic()
        observation = _observation_to_jax(observation)
        jax.block_until_ready(observation)
        t1 = time.monotonic()

        rng = self.extractor._next_rng()
        jax.block_until_ready(rng)
        t2 = time.monotonic()

        (prefix_out, prefix_mask), raw_actions = self.extractor._sample_actions(rng, observation)
        jax.block_until_ready((prefix_out, prefix_mask, raw_actions))
        t3 = time.monotonic()

        # Share JAX device buffers with PyTorch through DLPack.
        z = torch.utils.dlpack.from_dlpack(prefix_out.__dlpack__()).to(
            dtype=torch.float32, device=self.device
        )
        pad_mask = torch.utils.dlpack.from_dlpack(prefix_mask.__dlpack__()).to(
            dtype=torch.bool, device=self.device
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t4 = time.monotonic()

        # Apply output transform chain (Unnormalize → DroidOutputs / SliceAction)
        raw_t = torch.utils.dlpack.from_dlpack(raw_actions.__dlpack__()).to(dtype=torch.float32)
        if raw_t.is_cuda:
            torch.cuda.synchronize(raw_t.device)
        t5 = time.monotonic()

        actions_np = raw_t.cpu().numpy()
        t6 = time.monotonic()

        # Output transforms are NumPy/Python based.  Copy only state to the host
        # here, after the model call; it is not fed back into JAX.
        state_np = np.asarray(jax.device_get(observation.state))
        t7 = time.monotonic()

        out = []
        for i in range(actions_np.shape[0]):
            actions_in = actions_np[i]
            t = self._output_transform({
                "state": state_np[i],
                "actions": actions_in,
            })
            out.append(t["actions"])
        t8 = time.monotonic()

        actions = torch.as_tensor(np.stack(out), device=self.device, dtype=torch.float32)
        if actions.is_cuda:
            torch.cuda.synchronize(actions.device)
        t9 = time.monotonic()

        print(
            f"[DEBUG] extract_both | "
            f"obs_to_jax={(t1 - t0) * 1000:.1f}ms | "
            f"rng={(t2 - t1) * 1000:.1f}ms | "
            f"vla_sample={(t3 - t2) * 1000:.1f}ms | "
            f"embedding_dlpack={(t4 - t3) * 1000:.1f}ms | "
            f"action_dlpack={(t5 - t4) * 1000:.1f}ms | "
            f"action_to_host={(t6 - t5) * 1000:.1f}ms | "
            f"state_to_host={(t7 - t6) * 1000:.1f}ms | "
            f"output_transform={(t8 - t7) * 1000:.1f}ms | "
            f"action_to_device={(t9 - t8) * 1000:.1f}ms | "
            f"total={(t9 - t0) * 1000:.1f}ms"
        )

        return z, pad_mask, actions

    @staticmethod
    def _load_norm_stats(
        checkpoint_dir: pathlib.Path,
        data_config,
    ) -> dict[str, _transforms.NormStats]:
        """Load checkpoint norm stats, falling back to the data config."""
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
