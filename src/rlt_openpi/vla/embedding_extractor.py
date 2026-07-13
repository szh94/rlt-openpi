"""Extract VLA embeddings from a JAX PI0/PI0.5 model.

Calls the JAX model's ``embed_prefix`` directly — no monkey-patching
needed since the JAX Pi0 exposes this as a public method.
"""

import logging

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def extract_embeddings(model, observation):
    """Extract post-transformer prefix embeddings from the VLA.

    Args:
        model: JAX Pi0/Pi05 nnx.Module.
        observation: An openpi Observation (batched, JAX arrays).

    Returns:
        z: Post-transformer prefix embeddings [B, M, embedding_dim] (stop-grad).
        pad_mask: Boolean padding mask [B, M] (True = valid token).
    """
    tokens, input_mask, _ar_mask = model.embed_prefix(observation)
    z = jax.lax.stop_gradient(tokens)
    return z, input_mask


def forward_joint(model, observation, actions):
    """Single VLA forward pass returning both prefix embeddings and loss.

    Args:
        model: JAX Pi0/Pi05 nnx.Module.
        observation: An openpi Observation (batched, JAX arrays).
        actions: Ground-truth demo actions [B, H, action_dim] (JAX array).

    Returns:
        z: Detached prefix embeddings [B, M, D] (stop-grad from VLA).
        pad_mask: Boolean padding mask [B, M] (True = valid token).
        vla_loss: Scalar VLA flow-matching loss (with grad for VLA).
    """
    # Prefix embeddings (stop-gradient so RL token loss doesn't backprop through VLA)
    tokens, input_mask, _ar_mask = model.embed_prefix(observation)
    z = jax.lax.stop_gradient(tokens)

    # VLA flow-matching loss (gradients flow to VLA params in joint training)
    per_element_loss = model.compute_loss(observation, actions)
    vla_loss = per_element_loss.mean()

    return z, input_mask, vla_loss


def sample_actions(model, rng, observation, *, num_steps: int = 10):
    """Get reference actions from the VLA via full diffusion/AR sampling.

    Args:
        model: JAX Pi0/Pi05 nnx.Module.
        rng: JAX PRNG key.
        observation: An openpi Observation (batched, JAX arrays).
        num_steps: Number of diffusion steps (for Pi0) or max decoding
            steps (for Pi0FAST).

    Returns:
        actions: VLA output [B, action_horizon, action_dim] (JAX array).
    """
    return model.sample_actions(rng, observation, num_steps=num_steps)
