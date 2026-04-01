"""High-level wrapper for frozen VLA inference and embedding extraction.

Loads a PI0/PI0.5 model from checkpoint, wraps it with EmbeddingExtractor,
and exposes the interface used by the rollout worker and trainers.
"""

import torch
from openpi.models.model import Observation
from torch import Tensor

from rlt_openpi.vla.config import load_vla_config
from rlt_openpi.vla.embedding_extractor import EmbeddingExtractor


class VLAWrapper:
    """Loads and wraps a frozen VLA model for use by RLT components.

    Provides three operations:
    - extract_embeddings: get post-transformer prefix embeddings z_{1:M}
    - sample_reference_actions: get full VLA action trajectory (H steps)
    - get_rl_chunk_reference: slice first C steps as RL reference actions

    Args:
        checkpoint_path: Path to model.safetensors weight file.
        config_name: Registered openpi config name (e.g. "pi0_aloha_sim").
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
            actions: [B, H, action_dim] where H = action_horizon (e.g. 50).
        """
        return self.extractor.get_vla_actions(observation, self.device)

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
