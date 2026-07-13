"""TD3-style twin Q-critic with target networks (JAX/Flax nnx).

Each Q-network maps (state, action_chunk) -> scalar Q-value.
TwinQCritic maintains two online + two target copies with Polyak averaging.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from rlt_openpi.models.networks import MLP


class QNetwork(nnx.Module):
    """Single Q-network: (state, action_chunk) -> scalar.

    Args:
        state_dim: Dimension of RL state (z_rl + s^p).
        action_chunk_dim: Dimension of flattened action chunk (C * d).
        hidden_dim: MLP hidden layer width.
        num_hidden_layers: Number of MLP hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.mlp = MLP(
            input_dim=state_dim + action_chunk_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            rngs=rngs,
        )

    def __call__(self, x, a):
        """Compute Q-value.

        Args:
            x: RL state [B, state_dim].
            a: Flattened action chunk [B, action_chunk_dim].

        Returns:
            Q-value [B, 1].
        """
        return self.mlp(jnp.concatenate([x, a], axis=-1))


class TwinQCritic(nnx.Module):
    """Twin Q-networks with target copies for TD3.

    Args:
        state_dim: Dimension of RL state.
        action_chunk_dim: Dimension of flattened action chunk.
        hidden_dim: MLP hidden layer width.
        num_hidden_layers: Number of MLP hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.q1 = QNetwork(state_dim, action_chunk_dim, hidden_dim, num_hidden_layers, rngs=rngs)
        self.q2 = QNetwork(state_dim, action_chunk_dim, hidden_dim, num_hidden_layers, rngs=rngs)

        # Frozen target copies (separate nnx.Module instances)
        q1_graphdef, q1_state = nnx.split(self.q1)
        q2_graphdef, q2_state = nnx.split(self.q2)
        self.q1_target = nnx.merge(q1_graphdef, q1_state)
        self.q2_target = nnx.merge(q2_graphdef, q2_state)
        # Store graphdefs for potential serialization
        self._q1_graphdef = q1_graphdef
        self._q2_graphdef = q2_graphdef

    def __call__(self, x, a):
        """Compute Q-values from both online networks.

        Returns:
            (q1, q2): Each [B, 1].
        """
        return self.q1(x, a), self.q2(x, a)

    def q_min(self, x, a):
        """Min of online Q-values (used in actor loss).

        Returns:
            min(q1, q2) [B, 1].
        """
        q1, q2 = self(x, a)
        return jnp.minimum(q1, q2)

    def target_q_min(self, x, a):
        """Min of target Q-values (used in TD target computation).

        Returns:
            min(q1_target, q2_target) [B, 1].
        """
        q1_t = jax.lax.stop_gradient(self.q1_target(x, a))
        q2_t = jax.lax.stop_gradient(self.q2_target(x, a))
        return jnp.minimum(q1_t, q2_t)

    def update_targets(self, tau: float) -> None:
        """Polyak-average online params into target networks.

        theta_target = (1 - tau) * theta_target + tau * theta_online
        """
        for online, target in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
            _, online_state = nnx.split(online)
            target_graphdef, target_state = nnx.split(target)
            new_target_state = jax.tree.map(
                lambda t, o: (1.0 - tau) * t + tau * o,
                target_state.to_pure_dict(),
                online_state.to_pure_dict(),
            )
            target.replace_by_pure_dict(new_target_state)
