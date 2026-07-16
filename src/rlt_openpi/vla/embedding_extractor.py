"""Extract frozen VLA embeddings from a PI0/PI0.5 model.

Replicates the prefix-only forward pass from PI0Pytorch.sample_actions
to obtain post-transformer embeddings z_{1:M} without running the
diffusion denoising loop.
"""

import torch
from openpi.models.model import Observation
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from torch import Tensor, nn


class EmbeddingExtractor(nn.Module):
    """Wraps a PI0Pytorch model for embedding extraction and joint training.

    Capabilities:
    1. extract_embeddings(observation) → (z, pad_mask)
       Runs prefix-only forward pass to get post-transformer embeddings.
    2. sample_actions(observation) → action tensor
       Delegates to PI0Pytorch.sample_actions for reference actions.
    # VLA joint training (forward_joint, unfreeze) is disabled.
    """

    def __init__(self, pi0_model: PI0Pytorch, freeze: bool = True) -> None:
        super().__init__()
        self.pi0 = pi0_model
        if freeze:
            self.pi0.eval()
            for param in self.pi0.parameters():
                param.requires_grad_(False)

    @torch.no_grad()
    def extract_embeddings(
        self,
        observation: Observation,
    ) -> tuple[Tensor, Tensor]:
        """Extract post-transformer prefix embeddings from the frozen VLA.

        Mirrors the first half of PI0Pytorch.sample_actions:
        preprocess → embed_prefix → paligemma forward (prefix only).

        Args:
            observation: An openpi Observation (batched).

        Returns:
            z: Post-transformer prefix embeddings [B, M, embedding_dim].
            pad_mask: Boolean padding mask [B, M] (True = valid token).
        """
        images, img_masks, lang_tokens, lang_masks, _state = self.pi0._preprocess_observation(observation, train=False)

        # Embed prefix (images + language) → raw embeddings
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi0.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        )

        # Build 2D attention mask and position IDs (same as sample_actions)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Prepare 4D attention mask
        att_2d_masks_4d = self.pi0._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Ensure eager attention (no SDPA) for prefix-only pass
        self.pi0.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        # Cast to model dtype if needed
        model_dtype = self.pi0.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if model_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Prefix-only forward: inputs_embeds=[prefix, None] → runs PaliGemma LM only
        (prefix_out, _suffix_out), _kv_cache = self.pi0.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )

        # prefix_out is last_hidden_state [B, M, 2048], cast back to float32
        z = prefix_out.to(dtype=torch.float32)

        return z, prefix_pad_masks

    @torch.no_grad()
    def sample_actions(
        self,
        observation: Observation,
        device: torch.device,
    ) -> Tensor:
        """Get reference actions from the VLA via full diffusion sampling.

        Args:
            observation: An openpi Observation (batched).
            device: Device to run inference on.

        Returns:
            actions: VLA output [B, action_horizon, action_dim].
        """
        return self.pi0.sample_actions(device, observation)
