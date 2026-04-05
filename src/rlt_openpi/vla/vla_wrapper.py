"""High-level wrapper for VLA inference, embedding extraction, and joint training.

Loads a PI0/PI0.5 model from checkpoint, wraps it with EmbeddingExtractor,
and exposes the interface used by the rollout worker and trainers.
"""

from typing import Any

import torch
from openpi.models.model import Observation
from torch import Tensor

from rlt_openpi.vla.config import load_vla_config
from rlt_openpi.vla.embedding_extractor import EmbeddingExtractor


class VLAWrapper:
    """Loads and wraps a VLA model for use by RLT components.

    Provides:
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
        """Get full VLA reference action trajectory.

        Returns:
            actions: [B, H, action_dim] where H = action_horizon.
        """
        return self.extractor.sample_actions(observation, self.device)

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
