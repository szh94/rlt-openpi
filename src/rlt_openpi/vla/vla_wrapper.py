"""High-level wrapper for VLA inference, embedding extraction, and joint training.

Loads a PI0/PI0.5 model from checkpoint, wraps it with EmbeddingExtractor,
and exposes the interface used by the rollout worker and trainers.

Also builds the OpenPI input transform chain so that raw environment
observations can be preprocessed for VLA inference via :meth:`preprocess_obs`.
"""

from typing import Any

import numpy as np
import torch
from openpi.models.model import Observation
from openpi.policies.droid_policy import DroidOutputs
from openpi.transforms import Normalize, Unnormalize, compose
from torch import Tensor

from rlt_openpi.vla.config import load_vla_config
from rlt_openpi.vla.embedding_extractor import EmbeddingExtractor

_CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class VLAWrapper:
    """Loads and wraps a VLA model for use by RLT components.

    Provides:
    - preprocess_obs: raw env obs → batched Observation
    - extract_embeddings: get post-transformer prefix embeddings z_{1:M}
    - sample_reference_actions: get full VLA action trajectory (H steps)
    - get_rl_chunk_reference: slice first C steps as RL reference actions
    - compute_vla_loss: VLA flow-matching loss (standalone)
    - compute_vla_loss_with_embeddings: single forward for joint training

    Args:
        checkpoint_path: Path to model.safetensors weight file.
        config_name: Registered openpi config name (e.g. "pi05_droid_finetune").
        device: Torch device for the model.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_name: str,
        device: torch.device | str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.train_config = load_vla_config(config_name)

        # Load PI0Pytorch from safetensors checkpoint
        pi0_model = self.train_config.model.load_pytorch(
            self.train_config,
            checkpoint_path,
        )
        pi0_model = pi0_model.to(self.device)

        self.extractor = EmbeddingExtractor(pi0_model)
        self.action_dim = self.train_config.model.action_dim
        self.action_horizon = self.train_config.model.action_horizon

        # Build OpenPI input/output transform chains.
        # Skips DroidInputs since rollout env obs already uses model-format keys.
        data_config = self.train_config.data.create(
            self.train_config.assets_dirs, self.train_config.model
        )
        self._input_transform = compose([
            Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ])
        # Output transform: unnormalize VLA actions → robot space, then
        # slice to real robot dims. DroidOutputs (actions[:, :8]) is explicit
        # here because the finetune config omits it from model_transforms.outputs.
        actions_only_stats = {"actions": data_config.norm_stats["actions"]}
        self._output_transform = compose([
            Unnormalize(actions_only_stats, use_quantiles=data_config.use_quantile_norm),
            DroidOutputs(),
        ])

    def preprocess_obs(self, obs: dict[str, Any]) -> Observation:
        """Convert a raw environment observation into a batched Observation.

        Applies the OpenPI input transform chain (Normalize → ResizeImages →
        TokenizePrompt → PadStatesAndActions) and returns an ``Observation``
        with batch size 1, ready for ``extract_embeddings`` or
        ``get_rl_chunk_reference``.

        Expects ``obs`` with keys:
            - ``"state"``: proprioceptive state array
            - ``"base_0_rgb"``, ``"left_wrist_0_rgb"``, ``"right_wrist_0_rgb"``:
              uint8 HWC images (missing keys are zero-padded)
            - ``"prompt"``: language instruction string

        Args:
            obs: Raw observation dict from the environment.
        """
        # Restructure flat env obs → nested model format
        sample: dict[str, Any] = {
            "state": np.asarray(obs["state"], dtype=np.float64),
            "prompt": obs.get("prompt", ""),
            # Dummy actions (required by Normalize / Pad but unused)
            "actions": np.zeros(
                (self.action_horizon, self.action_dim), dtype=np.float64
            ),
        }
        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, Any] = {}
        for key in _CAMERA_KEYS:
            if key in obs and obs[key] is not None:
                img = np.asarray(obs[key])
                if np.issubdtype(img.dtype, np.floating):
                    img = (img * 255).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[key] = img
                image_masks[key] = np.True_
            else:
                images[key] = np.zeros((224, 224, 3), dtype=np.uint8)
                image_masks[key] = np.False_
        sample["image"] = images
        sample["image_mask"] = image_masks

        # Apply OpenPI transforms
        transformed = self._input_transform(sample)

        # Batch (size 1) + convert to torch → Observation
        import jax

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

        Returns:
            actions: [B, H, robot_action_dim] where H = action_horizon.
        """
        raw = self.extractor.sample_actions(observation, self.device)
        actions_np = raw.cpu().numpy()  # [B, H, 32]
        # OpenPI transforms operate on unbatched samples, so process per-sample.
        out = []
        for i in range(actions_np.shape[0]):
            t = self._output_transform({"actions": actions_np[i]})  # [H, 32] → [H, 8]
            out.append(t["actions"])
        return torch.as_tensor(np.stack(out), device=self.device)  # [B, H, 8]

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
