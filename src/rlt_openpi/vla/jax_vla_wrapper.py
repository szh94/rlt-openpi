from __future__ import annotations

import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from openpi.models import model as _model
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
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


def _observation_to_jax(obs: _model.Observation) -> _model.Observation:
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

    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def _copy_dict_structure(value: Any) -> Any:
    """Copy nested dictionaries while sharing their array leaves.

    OpenPI input transforms mutate dictionaries in place.  Rollout code may
    preprocess the same environment observation more than once (first as a
    replay-buffer next state, then as the next policy input), so the transform
    chain must not mutate the caller-owned observation.  Array leaves are only
    read or replaced by the inference transforms and do not need an expensive
    deep copy.
    """
    if isinstance(value, dict):
        return {key: _copy_dict_structure(item) for key, item in value.items()}
    return value


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
        checkpoint_params_path = checkpoint_dir_path / "params"

        print(f"Loading JAX VLA from Orbax checkpoint: {checkpoint_params_path}")
        model_sharding = jax.sharding.SingleDeviceSharding(jax.devices("gpu")[0])
        params = _model.restore_params(
            checkpoint_params_path,
            dtype=jnp.bfloat16,
            sharding=model_sharding,
        )
        pi0_model = self.train_config.model.load(params)

        self._rng = jax.random.key(0)
        self._extract_prefix = nnx_utils.module_jit(
            pi0_model.extract_prefix_embeddings
        )
        self._sample_actions = nnx_utils.module_jit(pi0_model.sample_actions)
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
            _transforms.Unnormalize(norm_stats, use_quantiles=use_q),
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
            _transforms.InjectDefaultPrompt(default_prompt),
            *dt.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=use_q),
            *data_config.model_transforms.inputs,
        ]
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def _next_rng(self) -> jax.Array:
        self._rng, rng = jax.random.split(self._rng)
        return rng

    # ------------------------------------------------------------------
    # Public interface (mirrors VLAWrapper)
    # ------------------------------------------------------------------

    def preprocess_obs(self, obs: dict[str, Any]) -> _model.Observation:
        """Convert a raw environment observation into a batched Observation.

        Applies the full OpenPI input transform chain and returns jnp
        arrays (same as the JAX path in ``policy.py``), so images stay in
        NHWC format expected by the JAX model.

        Args:
            obs: Raw observation dict from the environment.
        """
        # AlohaInputs converts images from CHW to HWC and mutates its input
        # dictionary.  Preserve the environment observation so repeated
        # preprocessing cannot transpose an already-transposed image again.
        transformed = self._input_transform(_copy_dict_structure(obs))

        # Preserve the NHWC layout expected by the JAX model.
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
        return _model.Observation.from_dict(batched)

    def extract_embeddings(
        self, observation: _model.Observation
    ) -> tuple[Tensor, Tensor]:
        """Extract prefix embeddings and return them as PyTorch tensors.

        This calls the prefix-only JIT path and skips diffusion sampling.
        """
        observation = _observation_to_jax(observation)

        prefix_out, prefix_mask = self._extract_prefix(observation)
        # Complete the JAX work before handing its buffers to PyTorch.
        jax.block_until_ready((prefix_out, prefix_mask))

        # Share JAX device buffers with PyTorch through DLPack.
        z = torch.utils.dlpack.from_dlpack(prefix_out.__dlpack__()).to(
            dtype=torch.float32
        )
        pad_mask = torch.utils.dlpack.from_dlpack(prefix_mask.__dlpack__()).to(
            dtype=torch.bool
        )

        z = z.to(self.device)
        pad_mask = pad_mask.to(self.device)

        return z, pad_mask

    def extract_both(
        self,
        observation: _model.Observation,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return prefix embeddings, padding mask, and robot-space actions."""
        observation = _observation_to_jax(observation)

        rng = self._next_rng()

        (prefix_out, prefix_mask), raw_actions = self._sample_actions(
            rng, observation
        )
        # Complete the JAX work before handing its buffers to PyTorch.
        jax.block_until_ready((prefix_out, prefix_mask, raw_actions))

        # Share JAX device buffers with PyTorch through DLPack.
        z = torch.utils.dlpack.from_dlpack(prefix_out.__dlpack__()).to(
            dtype=torch.float32, device=self.device
        )
        pad_mask = torch.utils.dlpack.from_dlpack(prefix_mask.__dlpack__()).to(
            dtype=torch.bool, device=self.device
        )

        # Apply output transform chain (Unnormalize → DroidOutputs / SliceAction)
        raw_t = torch.utils.dlpack.from_dlpack(raw_actions.__dlpack__()).to(
            dtype=torch.float32
        )

        actions_np = raw_t.cpu().numpy()

        # Output transforms are NumPy/Python based.  Copy only state to the host
        # here, after the model call; it is not fed back into JAX.
        state_np = np.asarray(jax.device_get(observation.state))

        out = []
        for i in range(actions_np.shape[0]):
            actions_in = actions_np[i]
            t = self._output_transform(
                {
                    "state": state_np[i],
                    "actions": actions_in,
                }
            )
            out.append(t["actions"])

        actions = torch.as_tensor(
            np.stack(out), device=self.device, dtype=torch.float32
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
