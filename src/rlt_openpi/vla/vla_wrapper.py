"""High-level wrapper for VLA inference, embedding extraction, and joint training.

Loads a PI0/PI0.5 model from checkpoint, wraps it with EmbeddingExtractor,
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
import numpy as np
import torch
from openpi.models.model import Observation
from openpi.training import checkpoints as _checkpoints
from openpi.transforms import InjectDefaultPrompt, Normalize, Unnormalize, compose
import openpi.transforms as _transforms
from torch import Tensor

from rlt_openpi.vla.config import load_vla_config
from rlt_openpi.vla.embedding_extractor import EmbeddingExtractor

logger = logging.getLogger(__name__)


class VLAWrapper:
    """Loads and wraps a VLA model for use by RLT components.

    Provides:
    - preprocess_obs: raw env obs → batched Observation
    - extract_embeddings: get post-transformer prefix embeddings z_{1:M}
    - sample_reference_actions: get full VLA action trajectory (H steps)
    - get_rl_chunk_reference: slice first C steps as RL reference actions
    - compute_vla_loss: VLA flow-matching loss (standalone)
    - compute_vla_loss_with_embeddings: single forward for joint training

    The input/output transform chains mirror OpenPI's
    ``create_trained_policy`` (``policy_config.py``):

    **Input** (applied by :meth:`preprocess_obs`)::

        data_transforms.inputs  (e.g. DroidInputs)
        → Normalize
        → model_transforms.inputs  (ResizeImages, TokenizePrompt, Pad)

    **Output** (applied by :meth:`sample_reference_actions`)::

        model_transforms.outputs
        → Unnormalize
        → data_transforms.outputs  (e.g. DroidOutputs)

    Args:
        checkpoint_path: Path to model.safetensors weight file.
        config_name: Registered openpi config name (e.g. "pi05_droid_finetune").
        device: Torch device for the model.
        data_transforms: Optional override for the config's default
            ``data_transforms``.  Must match whatever was used during
            training (e.g. ``ThreeCameraDroidInputs`` for 3-camera setups).
        default_prompt: If set, injected into inputs that lack a ``prompt``
            key (mirrors OpenPI's ``InjectDefaultPrompt``).
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_name: str,
        device: torch.device | str = "cuda",
        data_transforms: _transforms.Group | None = None,
        default_prompt: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.train_config = load_vla_config(config_name)

        pi0_model = self.train_config.model.load_pytorch(
            self.train_config,
            checkpoint_path,
        )
        pi0_model = pi0_model.to(self.device)

        self.extractor = EmbeddingExtractor(pi0_model)
        self.action_dim = self.train_config.model.action_dim
        self.action_horizon = self.train_config.model.action_horizon

        checkpoint_dir = pathlib.Path(checkpoint_path).parent
        data_config = self.train_config.data.create(
            self.train_config.assets_dirs, self.train_config.model
        )
        use_q = data_config.use_quantile_norm

        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = self._load_norm_stats(checkpoint_dir, data_config)

        dt = data_transforms or data_config.data_transforms

        self._norm_stats = norm_stats
        self._use_quantile_norm = use_q
        self._normalize = Normalize(norm_stats, use_quantiles=use_q)
        self._unnormalize = Unnormalize(norm_stats, use_quantiles=use_q)

        self._input_transform = compose([
            *dt.inputs,
            InjectDefaultPrompt(default_prompt),
            self._normalize,
            *data_config.model_transforms.inputs,
        ])
        self._output_transform = compose([
            *data_config.model_transforms.outputs,
            self._unnormalize,
            *dt.outputs,
        ])

    @staticmethod
    def _load_norm_stats(checkpoint_dir: pathlib.Path, data_config) -> dict[str, _transforms.NormStats]:
        """Load norm stats, preferring checkpoint-embedded assets over config assets.

        Mirrors OpenPI's ``create_trained_policy`` which loads from
        ``checkpoint_dir/assets/<asset_id>/`` to guarantee the stats match
        training.  Falls back to the config's ``norm_stats`` for checkpoints
        that don't bundle assets (e.g. rlt-openpi ``.pt`` checkpoints).
        """
        asset_id = data_config.asset_id
        try:
            norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", asset_id)
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

        Applies the full OpenPI input transform chain
        (DroidInputs → Normalize → ResizeImages → TokenizePrompt →
        PadStatesAndActions).

        Expects ``obs`` in DROID-schema keys as produced by the env
        factory (e.g. ``observation/joint_position``,
        ``observation/exterior_image_1_left``, ``prompt``).

        Args:
            obs: Raw observation dict from the environment.
        """
        transformed = self._input_transform(dict(obs))

        batched = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self.device)[None, ...],
            transformed,
        )
        return Observation.from_dict(batched)

    def extract_embeddings(
        self,
        observation: Observation,
    ) -> tuple[Tensor, Tensor]:
        """Extract post-transformer prefix embeddings from the frozen VLA.

        Returns:
            z: [B, M, embedding_dim] post-transformer prefix embeddings.
            pad_mask: [B, M] boolean mask (True = valid token).
        """
        return self.extractor.extract_embeddings(observation)

    def sample_reference_actions(
        self,
        observation: Observation,
    ) -> Tensor:
        """Get full VLA reference action trajectory, unnormalized to robot space.

        Mirrors OpenPI's ``Policy.infer`` output path: passes both
        ``state`` and ``actions`` through the output transform so that
        ``Unnormalize`` can operate on the full norm_stats.

        Returns:
            actions: [B, H, robot_action_dim] where H = action_horizon.
        """
        raw = self.extractor.sample_actions(observation, self.device)
        actions_np = raw.cpu().numpy()
        state_np = observation.state.cpu().numpy()

        out = []
        for i in range(actions_np.shape[0]):
            t = self._output_transform({
                "state": state_np[i],
                "actions": actions_np[i],
            })
            out.append(t["actions"])
        return torch.as_tensor(np.stack(out), device=self.device)

    def get_rl_chunk_reference(
        self,
        observation: Observation,
        chunk_length: int = 10,
    ) -> Tensor:
        """Get the first C action steps from the VLA as the RL reference.

        The RL actor conditions on these reference actions (a_tilde_{1:C}).

        Args:
            observation: Batched openpi Observation.
            chunk_length: Number of action steps to slice (C, default 10).

        Returns:
            a_tilde: [B, C, action_dim] reference actions for the RL chunk.
        """
        full_actions = self.sample_reference_actions(observation)
        return full_actions[:, :chunk_length, :]

    def get_rl_chunk_reference_normalized(
        self,
        observation: Observation,
        chunk_length: int = 10,
    ) -> Tensor:
        """Get first C action steps in **normalized** space (before Unnormalize).

        This is the version that should be fed to the Actor and stored in
        the replay buffer.  The Actor operates in normalized space [-1, 1].

        Args:
            observation: Batched openpi Observation.
            chunk_length: Number of action steps to slice (C).

        Returns:
            a_tilde: [B, C, action_dim] in normalized space.
        """
        raw = self.extractor.sample_actions(observation, self.device)
        return raw[:, :chunk_length, :]

    def unnormalize_actions(
        self,
        actions_normalized: Tensor,
        state: Tensor,
    ) -> Tensor:
        """Convert normalized actions back to robot space for execution.

        Applies the same Unnormalize transform used in
        ``sample_reference_actions``.

        Args:
            actions_normalized: [B, C, action_dim] in normalized space.
            state: [B, state_dim] normalized state tensor (from Observation.state).

        Returns:
            actions: [B, C, action_dim] in robot space.
        """
        actions_np = actions_normalized.cpu().numpy()
        state_np = state.cpu().numpy()
        out = []
        for i in range(actions_np.shape[0]):
            t = self._unnormalize({
                "state": state_np[i],
                "actions": actions_np[i],
            })
            out.append(t["actions"])
        return torch.as_tensor(np.stack(out), device=self.device)

    def normalize_actions(
        self,
        actions_robot: Tensor,
        state: Tensor,
    ) -> Tensor:
        """Convert robot-space actions to normalized space.

        Used to normalize human intervention actions before storing in
        the replay buffer.

        Args:
            actions_robot: [B, C, action_dim] in robot space.
            state: [B, state_dim] normalized state tensor (from Observation.state).

        Returns:
            actions: [B, C, action_dim] in normalized space.
        """
        actions_np = actions_robot.cpu().numpy()
        state_np = state.cpu().numpy()
        out = []
        for i in range(actions_np.shape[0]):
            t = self._normalize({
                "state": state_np[i],
                "actions": actions_np[i],
            })
            out.append(t["actions"])
        return torch.as_tensor(np.stack(out), device=self.device)

    def compute_vla_loss(
        self,
        observation: dict[str, Any] | Observation,
        actions: Tensor,
    ) -> Tensor:
        """Compute the VLA's flow-matching training loss on demo data.

        Calls PI0Pytorch.forward() which computes the denoising loss:
        noisy actions x_t are created from ground-truth actions + noise,
        the model predicts the velocity field v_t, and loss = MSE(u_t, v_t).

        Args:
            observation: Batched observation (dict or openpi Observation).
            actions: Ground-truth demo actions [B, action_horizon, action_dim].

        Returns:
            Scalar mean VLA loss.
        """
        per_element_loss = self.extractor.pi0.forward(observation, actions)
        return per_element_loss.mean()

    def compute_vla_loss_with_embeddings(
        self,
        observation: Observation,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single VLA forward pass returning both embeddings and loss.

        Used by joint training (alpha > 0) to avoid the double forward
        pass of calling extract_embeddings() + compute_vla_loss()
        separately.

        Args:
            observation: Batched openpi Observation.
            actions: Ground-truth demo actions [B, H, action_dim].

        Returns:
            z: Detached prefix embeddings [B, M, D] (stop-grad from VLA).
            pad_mask: Boolean mask [B, M] (True = valid token).
            vla_loss: Scalar VLA flow-matching loss (with grad for VLA).
        """
        return self.extractor.forward_joint(observation, actions)

    def unfreeze(self) -> None:
        """Re-enable gradients on VLA parameters for joint fine-tuning."""
        self.extractor.unfreeze()

    def trainable_parameters(self):
        """Return VLA parameters that require gradients (for optimizer)."""
        return [p for p in self.extractor.pi0.parameters() if p.requires_grad]
