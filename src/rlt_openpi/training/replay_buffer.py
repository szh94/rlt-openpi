"""Chunked replay buffer with stride-2 subsampling for online RL.

Stores transitions (x, a, a_tilde, rewards, next_x, dones) where each
transition corresponds to one action chunk of length C.

Internal storage uses numpy arrays.  ``sample()`` returns JAX arrays.
"""

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


class ReplayBuffer:
    """Fixed-capacity circular replay buffer with pre-allocated storage.

    Args:
        capacity: Maximum number of transitions.
        state_dim: Dimension of RL state (z_rl + s^p).
        action_chunk_dim: Dimension of flattened action chunk (C * d).
        chunk_length: C, number of steps per chunk (for per-step rewards).
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_chunk_dim: int,
        chunk_length: int,
    ) -> None:
        self.capacity = capacity
        self.chunk_length = chunk_length
        self._ptr = 0
        self._size = 0

        # Pre-allocate storage as numpy arrays (float32)
        self._x = np.zeros((capacity, state_dim), dtype=np.float32)
        self._a = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self._a_tilde = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, chunk_length), dtype=np.float32)
        self._next_x = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self._size

    def add(
        self,
        x: NDArray,
        a: NDArray,
        a_tilde: NDArray,
        rewards: NDArray,
        next_x: NDArray,
        done: float,
    ) -> None:
        """Add a single chunk-level transition.

        Args:
            x: RL state [state_dim].
            a: Executed action chunk [action_chunk_dim].
            a_tilde: VLA reference action chunk [action_chunk_dim].
            rewards: Per-step rewards [chunk_length].
            next_x: Next RL state [state_dim].
            done: Episode termination flag (0.0 or 1.0).
        """
        self._x[self._ptr] = x
        self._a[self._ptr] = a
        self._a_tilde[self._ptr] = a_tilde
        self._rewards[self._ptr] = rewards
        self._next_x[self._ptr] = next_x
        self._dones[self._ptr] = done
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_episode_strided(
        self,
        xs: NDArray,
        actions: NDArray,
        a_tildes: NDArray,
        rewards: NDArray,
        next_xs: NDArray,
        dones: NDArray,
        stride: int = 2,
    ) -> int:
        """Add transitions from an episode with stride-based subsampling.

        The paper uses stride=2 to get ~25 samples/second from 50 Hz control.
        Every ``stride``-th transition is stored.

        Returns:
            Number of transitions actually stored.
        """
        indices = range(0, len(xs), stride)
        for i in indices:
            self.add(
                xs[i], actions[i], a_tildes[i],
                rewards[i], next_xs[i], dones[i].item(),
            )
        return len(list(indices))

    def state_dict(self) -> dict[str, object]:
        """Return buffer state for checkpointing (numpy arrays)."""
        n = self._size
        return {
            "ptr": self._ptr,
            "size": n,
            "x": self._x[:n].copy(),
            "a": self._a[:n].copy(),
            "a_tilde": self._a_tilde[:n].copy(),
            "rewards": self._rewards[:n].copy(),
            "next_x": self._next_x[:n].copy(),
            "dones": self._dones[:n].copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore buffer from a checkpoint produced by :meth:`state_dict`."""
        n = int(state["size"])
        self._ptr = int(state["ptr"])
        self._size = n
        self._x[:n] = state["x"]
        self._a[:n] = state["a"]
        self._a_tilde[:n] = state["a_tilde"]
        self._rewards[:n] = state["rewards"]
        self._next_x[:n] = state["next_x"]
        self._dones[:n] = state["dones"]

    def sample(self, batch_size: int) -> dict[str, jnp.ndarray]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict with keys: x, a, a_tilde, rewards, next_x, dones.
            Each value is a JAX array.
        """
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "x": jnp.asarray(self._x[indices]),
            "a": jnp.asarray(self._a[indices]),
            "a_tilde": jnp.asarray(self._a_tilde[indices]),
            "rewards": jnp.asarray(self._rewards[indices]),
            "next_x": jnp.asarray(self._next_x[indices]),
            "dones": jnp.asarray(self._dones[indices]),
        }
