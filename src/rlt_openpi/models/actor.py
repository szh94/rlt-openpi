"""Actor network with VLA reference-action conditioning.

The actor takes the RL state x = cat(z_rl, s^p) and a VLA reference action
chunk a_tilde, applies reference dropout during training, and predicts the
final action distribution conditioned on both inputs.
"""

import torch
from torch import Tensor, nn

from rlt_openpi.models.networks import MLP


class Actor(nn.Module):
    """Actor with reference action dropout and exploration noise.

    Input: cat(x, a_tilde_masked) where a_tilde is zeroed for a fraction
    of the batch during training (ref_dropout probability).

    Output: mu + N(0, sigma^2) during training, mu during eval, where mu is
    the final action mean predicted by the MLP.

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
        """Predict an action chunk conditioned on the VLA reference.

        Args:
            x: RL state [B, state_dim].
            a_tilde: Flattened VLA reference action chunk [B, action_chunk_dim].

        Returns:
            Final action chunk [B, action_chunk_dim]. The MLP directly predicts
            the action mean; Gaussian exploration noise is added during
            training only.
        """
        a_tilde_partial = self._apply_ref_dropout(a_tilde)
        mu = self.mlp(torch.cat([x, a_tilde_partial], dim=-1))

        if self.training:
            noise = torch.randn_like(mu) * self.sigma
            return (mu + noise)
        return mu

    def _apply_ref_dropout(self, a_tilde: Tensor) -> Tensor:
        """Zero out reference actions for a fraction of the batch during training."""
        if not self.training or self.ref_dropout == 0.0:
            return a_tilde

        B = a_tilde.shape[0]
        # Per-sample dropout mask: 1 = keep, 0 = drop
        keep_mask = torch.rand(B, 1, device=a_tilde.device) >= self.ref_dropout
        return a_tilde * keep_mask
