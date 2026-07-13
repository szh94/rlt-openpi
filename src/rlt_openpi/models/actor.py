"""Actor network with VLA reference action conditioning (JAX/Flax nnx).

The actor takes the RL state x = cat(z_rl, s^p) and a VLA reference action
chunk a_tilde, applies reference dropout during training, and outputs an
action chunk mu (plus optional Gaussian exploration noise).
"""

import jax
import jax.numpy as jnp
from flax import nnx

from rlt_openpi.models.networks import MLP


class Actor(nnx.Module):
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
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.action_chunk_dim = action_chunk_dim
        self.sigma = sigma
        self.ref_dropout = ref_dropout

        # Zero-init the last linear layer so the residual starts at zero,
        # meaning the actor initially reproduces the VLA reference exactly.
        self.mlp = MLP(
            input_dim=state_dim + action_chunk_dim,
            output_dim=action_chunk_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            rngs=rngs,
            final_kernel_init=nnx.initializers.zeros,
            final_bias_init=nnx.initializers.zeros,
        )

    def __call__(self, x, a_tilde, *, training: bool = False, rng=None):
        """Compute action chunk as VLA reference + learned residual.

        Args:
            x: RL state [B, state_dim].
            a_tilde: Flattened VLA reference action chunk [B, action_chunk_dim].
            training: Whether to apply ref_dropout and exploration noise.
            rng: JAX random key (required when training=True).

        Returns:
            Action chunk [B, action_chunk_dim] = a_tilde + residual (+ noise).
            During training: a_tilde + residual + noise, with ref dropout.
            During eval: a_tilde + residual, with full a_tilde.
        """
        a_tilde_input = self._apply_ref_dropout(a_tilde, training, rng)
        residual = self.mlp(jnp.concatenate([x, a_tilde_input], axis=-1))
        mu = a_tilde + residual

        if training:
            if rng is None:
                raise ValueError("rng key required when training=True")
            noise = jax.random.normal(rng, mu.shape) * self.sigma
            return mu + noise
        return mu

    def _apply_ref_dropout(self, a_tilde, training: bool, rng):
        """Zero out reference actions for a fraction of the batch during training."""
        if not training or self.ref_dropout == 0.0:
            return a_tilde

        B = a_tilde.shape[0]
        keep_mask = jax.random.bernoulli(rng, 1.0 - self.ref_dropout, shape=(B, 1))
        return a_tilde * keep_mask
