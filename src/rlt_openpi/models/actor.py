"""Actor network with VLA reference action conditioning.

The actor takes the RL state x = cat(z_rl, s^p) and a VLA reference action
chunk a_tilde, applies reference dropout during training, and outputs an
action chunk mu (plus optional Gaussian exploration noise).
"""

import torch
from torch import Tensor, nn

from rlt_openpi.models.networks import MLP


class Actor(nn.Module):
    """Actor with reference action dropout and exploration noise.

    Input: cat(x, a_tilde_masked) where a_tilde is zeroed for a fraction
    of the batch during training (ref_dropout probability).

    Output: mu + N(0, sigma^2) during training, mu during eval.

    Args:
        state_dim: Dimension of RL state (z_rl + s^p).
        action_chunk_dim: Dimension of flattened action chunk (C * d).
        hidden_dim: MLP hidden layer width.
        num_hidden_layers: Number of MLP hidden layers.
        sigma: Exploration noise std (applied during training only).
        ref_dropout: Probability of zeroing reference actions during training.
    """

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        sigma: float = 0.1,
        ref_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.action_chunk_dim = action_chunk_dim
        self.sigma = sigma
        self.ref_dropout = ref_dropout

        self.mlp = MLP(
            input_dim=state_dim + action_chunk_dim,
            output_dim=action_chunk_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(self, x: Tensor, a_tilde: Tensor) -> Tensor:
        """Compute action chunk.

        Args:
            x: RL state [B, state_dim].
            a_tilde: Flattened VLA reference action chunk [B, action_chunk_dim].

        Returns:
            Action chunk [B, action_chunk_dim], clamped to [-1, 1].
            During training: mu + noise, with ref dropout applied to a_tilde.
            During eval: mu, with full a_tilde.
        """
        a_tilde_input = self._apply_ref_dropout(a_tilde)
        mu = self.mlp(torch.cat([x, a_tilde_input], dim=-1))

        if self.training:
            noise = torch.randn_like(mu) * self.sigma
            return (mu + noise).clamp(-1.0, 1.0)
        return mu.clamp(-1.0, 1.0)

    def _apply_ref_dropout(self, a_tilde: Tensor) -> Tensor:
        """Zero out reference actions for a fraction of the batch during training."""
        if not self.training or self.ref_dropout == 0.0:
            return a_tilde

        B = a_tilde.shape[0]
        # Per-sample dropout mask: 1 = keep, 0 = drop
        keep_mask = torch.rand(B, 1, device=a_tilde.device) >= self.ref_dropout
        return a_tilde * keep_mask
