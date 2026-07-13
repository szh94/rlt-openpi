"""TD3 utility functions for critic and actor updates (JAX).

Provides:
- compute_td_target: discounted chunk return + gamma^C * target Q(next state)
- critic_loss: MSE for twin Q-networks against the TD target
- actor_loss: -Q.mean() + beta * BC regularizer
"""

import jax
import jax.numpy as jnp


def compute_td_target(
    rewards,
    dones,
    next_x,
    next_a_tilde,
    actor,
    critic,
    gamma: float,
    chunk_length: int,
    rng,
    target_noise_sigma: float = 0.2,
    target_noise_clip: float = 0.5,
):
    """Compute TD target for critic training.

    y = sum_{k=0}^{C-1} gamma^k * r_k + gamma^C * (1 - done) * min Q_target(x', a')

    where a' = actor(x', a_tilde') + clip(N(0, sigma), -c, c)  (TD3 target smoothing).

    Args:
        rewards: Per-step rewards within the chunk [B, C].
        dones: Episode termination flags [B, 1].
        next_x: Next RL state [B, state_dim].
        next_a_tilde: Next VLA reference action chunk [B, action_chunk_dim].
        actor: Actor nnx.Module (used deterministically for target action).
        critic: TwinQCritic nnx.Module (uses target networks).
        gamma: Discount factor.
        chunk_length: C, number of steps per chunk.
        rng: JAX random key for target noise.
        target_noise_sigma: Std of TD3 target policy smoothing noise.
        target_noise_clip: Clamp range for target noise.

    Returns:
        TD target [B, 1].
    """
    # Discounted chunk return: sum_{k=0}^{C-1} gamma^k * r_k
    discount_powers = gamma ** jnp.arange(chunk_length, dtype=rewards.dtype)
    chunk_return = (rewards * discount_powers).sum(axis=-1, keepdims=True)  # [B, 1]

    # Target action from actor (deterministic — no noise, no ref_dropout)
    next_a = actor(next_x, next_a_tilde, training=False)

    # TD3 target policy smoothing: add clipped noise to target action
    noise = jax.random.normal(rng, next_a.shape) * target_noise_sigma
    noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    next_a = next_a + noise  # actions in robot space, no clamp(-1,1)

    # Bootstrap: gamma^C * (1 - done) * min Q_target(x', a')
    next_q = critic.target_q_min(next_x, next_a)
    next_q = jax.lax.stop_gradient(next_q)
    bootstrap = (gamma ** chunk_length) * (1.0 - dones) * next_q

    return chunk_return + bootstrap


def critic_loss(q1, q2, q_target):
    """MSE loss for both Q-networks against the TD target.

    L_critic = MSE(q1, y) + MSE(q2, y)

    Args:
        q1: Q-value from first network [B, 1].
        q2: Q-value from second network [B, 1].
        q_target: TD target [B, 1] (detached).

    Returns:
        Scalar loss.
    """
    return jnp.mean((q1 - q_target) ** 2) + jnp.mean((q2 - q_target) ** 2)


def actor_loss(q_value, a, a_tilde, beta: float):
    """Actor loss: maximize Q-value with BC regularizer.

    L_actor = -Q.mean() + beta * MSE(a, a_tilde)

    Args:
        q_value: Q-value of actor's action [B, 1].
        a: Actor's action chunk [B, action_chunk_dim].
        a_tilde: VLA reference action chunk [B, action_chunk_dim].
        beta: BC regularizer coefficient (0 = pure RL).

    Returns:
        Scalar loss.
    """
    policy_loss = -q_value.mean()
    bc_loss = jnp.mean((a - a_tilde) ** 2)
    return policy_loss + beta * bc_loss
