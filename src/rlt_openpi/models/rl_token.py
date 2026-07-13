"""RL Token encoder-decoder model (Stage 1) (JAX/Flax nnx).

Compresses variable-length VLA prefix embeddings z_{1:M} into a single
fixed-size RL token z_rl via an information bottleneck, and reconstructs
the original embeddings to train the bottleneck via masked MSE + cosine loss.

Paper reference: "RL Token: Bootstrapping Online RL with VLA Models"
"""

import jax
import jax.numpy as jnp
from flax import nnx


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------


class _TransformerEncoderBlock(nnx.Module):
    """Single pre-norm transformer encoder block (self-attention + FFN)."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, *, rngs: nnx.Rngs):
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            qkv_features=d_model,
            out_features=d_model,
            dropout_rate=0.0,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = nnx.Linear(dim_feedforward, d_model, rngs=rngs)
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x, mask=None):
        # Self-attention (pre-norm)
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask=mask)
        x = x + residual

        # FFN (pre-norm)
        residual = x
        x = self.norm2(x)
        x = self.linear2(nnx.relu(self.linear1(x)))
        x = x + residual
        return x


class _TransformerDecoderBlock(nnx.Module):
    """Single pre-norm transformer decoder block (self-attn + cross-attn + FFN)."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, *, rngs: nnx.Rngs):
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            qkv_features=d_model,
            out_features=d_model,
            dropout_rate=0.0,
            rngs=rngs,
        )
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            qkv_features=d_model,
            out_features=d_model,
            dropout_rate=0.0,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = nnx.Linear(dim_feedforward, d_model, rngs=rngs)
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm3 = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x, memory, self_mask=None):
        # Self-attention (pre-norm, causal)
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask=self_mask)
        x = x + residual

        # Cross-attention to memory (pre-norm)
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, inputs_kv=memory)
        x = x + residual

        # FFN (pre-norm)
        residual = x
        x = self.norm3(x)
        x = self.linear2(nnx.relu(self.linear1(x)))
        x = x + residual
        return x


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class RLTokenEncoder(nnx.Module):
    """Encoder: appends a learnable [RL] token and runs bidirectional transformer."""

    def __init__(
        self,
        embedding_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
        max_position_embeddings: int = 1024,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        # Learnable RL token embedding
        self.e_rl = nnx.Param(jax.random.normal(rngs(), (1, 1, embedding_dim)) * 0.02)

        # Position embeddings for M + 1 positions
        self.pos_embeddings = nnx.Embed(max_position_embeddings, embedding_dim, rngs=rngs)

        # Stack of encoder blocks
        for i in range(num_layers):
            setattr(
                self, f"layer{i}",
                _TransformerEncoderBlock(
                    embedding_dim, num_heads, 4 * embedding_dim, rngs=rngs,
                ),
            )
        self.num_layers = num_layers

    def __call__(self, z, pad_mask):
        """Encode prefix embeddings into a single RL token.

        Args:
            z: VLA embeddings [B, M, D].
            pad_mask: Boolean mask [B, M] (True = valid token).

        Returns:
            z_rl: RL token [B, D].
        """
        B, M, D = z.shape

        # Append learnable e_rl token: [B, M+1, D]
        e_rl = jnp.broadcast_to(self.e_rl.value, (B, 1, D))
        tokens = jnp.concatenate([z, e_rl], axis=1)

        # Add position embeddings
        pos_indices = jnp.arange(M + 1)[None, :]  # [1, M+1]
        tokens = tokens + self.pos_embeddings(pos_indices)

        # Build additive padding mask for attention
        # True=valid → 0.0, False=padded → -1e9
        rl_mask = jnp.ones((B, 1), dtype=jnp.bool_)
        extended_pad = jnp.concatenate([pad_mask, rl_mask], axis=1)  # [B, M+1]
        attn_mask = jnp.where(extended_pad[:, None, None, :], 0.0, -1e9)  # [B, 1, 1, M+1]

        # Run through encoder layers
        x = tokens
        for i in range(self.num_layers):
            x = getattr(self, f"layer{i}")(x, mask=attn_mask)

        # Extract RL token output (last position)
        z_rl = x[:, -1, :]  # [B, D]
        return z_rl


class RLTokenDecoder(nnx.Module):
    """Decoder: autoregressively reconstructs embeddings from z_rl."""

    def __init__(
        self,
        embedding_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
        max_position_embeddings: int = 1024,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.pos_embeddings = nnx.Embed(max_position_embeddings, embedding_dim, rngs=rngs)

        for i in range(num_layers):
            setattr(
                self, f"layer{i}",
                _TransformerDecoderBlock(
                    embedding_dim, num_heads, 4 * embedding_dim, rngs=rngs,
                ),
            )
        self.num_layers = num_layers
        self.h_phi = nnx.Linear(embedding_dim, embedding_dim, rngs=rngs)

    def __call__(self, z_rl, z, pad_mask):
        """Decode from z_rl to reconstruct full embedding sequence.

        Args:
            z_rl: RL token [B, D].
            z: Target embeddings [B, M, D] (for teacher forcing).
            pad_mask: Boolean mask [B, M] (True = valid token).

        Returns:
            z_hat: Reconstructed embeddings [B, M, D].
        """
        B, M, D = z.shape

        # Teacher-forced input: [z_rl, z_1, ..., z_{M-1}]
        tgt = jnp.concatenate([z_rl[:, None, :], z[:, :-1, :]], axis=1)  # [B, M, D]

        # Position embeddings
        pos_indices = jnp.arange(M)[None, :]  # [1, M]
        tgt = tgt + self.pos_embeddings(pos_indices)

        # Causal mask: upper triangle = -1e9
        causal_mask = jnp.triu(jnp.full((M, M), -1e9), k=1)  # [M, M]

        # Combine with padding mask as additive mask
        # pad_mask: [B, M] True=valid → 0.0, False=pad → -1e9
        pad_additive = jnp.where(pad_mask[:, None, :], 0.0, -1e9)  # [B, 1, M]
        self_mask = causal_mask[None, None, :, :] + pad_additive[:, :, None, :]  # [B, 1, M, M]

        # Memory = z_rl as a single token for cross-attention
        memory = z_rl[:, None, :]  # [B, 1, D]

        x = tgt
        for i in range(self.num_layers):
            x = getattr(self, f"layer{i}")(x, memory, self_mask=self_mask)

        z_hat = self.h_phi(x)  # [B, M, D]
        return z_hat


# ---------------------------------------------------------------------------
# Combined RL Token Model
# ---------------------------------------------------------------------------


class RLTokenModel(nnx.Module):
    """Combined RL token encoder-decoder for Stage 1 training.

    Training: forward(z, pad_mask) → (loss, z_rl, z_hat)
    Inference: encode(z, pad_mask) → z_rl

    Args:
        embedding_dim: Dimension of VLA embeddings.
        encoder_layers: Number of encoder transformer layers.
        encoder_heads: Number of encoder attention heads.
        decoder_layers: Number of decoder transformer layers.
        decoder_heads: Number of decoder attention heads.
        cosine_weight: Weight for cosine similarity loss term.
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        cosine_weight: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.encoder = RLTokenEncoder(embedding_dim, encoder_layers, encoder_heads, rngs=rngs)
        self.decoder = RLTokenDecoder(embedding_dim, decoder_layers, decoder_heads, rngs=rngs)
        self.cosine_weight = cosine_weight
        self.norm = nnx.LayerNorm(embedding_dim, rngs=rngs)

    def forward(self, z, pad_mask):
        """Training forward pass: encode, decode, compute reconstruction loss.

        Args:
            z: VLA embeddings [B, M, D] (will be detached internally).
            pad_mask: Boolean mask [B, M] (True = valid token).

        Returns:
            loss: Masked MSE reconstruction loss (scalar).
            z_rl: Encoded RL token [B, D].
            z_hat: Reconstructed embeddings [B, M, D].
        """
        # Stop gradient on VLA embeddings, then normalize
        z = jax.lax.stop_gradient(z)
        z_norm = self.norm(z)  # LayerNorm across D dim

        # Save stats for potential denormalization
        self._last_mean = z.mean(axis=-1, keepdims=True)  # [B, M, 1]
        self._last_var = z.var(axis=-1, keepdims=True)   # [B, M, 1]

        # Encode → z_rl [B, D]
        z_rl = self.encoder(z_norm, pad_mask)

        # Decode → z_hat [B, M, D]
        z_hat = self.decoder(z_rl, z_norm, pad_mask)

        # Denormalize z_hat back to original space for comparison with z
        z_hat_denorm = self._denorm(z_hat)

        # Masked MSE in normalized space
        mse = (z_hat - z_norm).__pow__(2).mean(axis=-1)  # [B, M]
        masked_mse = mse * pad_mask.astype(jnp.float32)

        # Masked cosine similarity loss: 1 - cos(z_hat, z_norm)
        cos_sim = _cosine_similarity(z_hat, z_norm, axis=-1)  # [B, M]
        loss_cos = (1.0 - cos_sim) * pad_mask.astype(jnp.float32)

        # Average over valid tokens
        num_valid = pad_mask.astype(jnp.float32).sum().clip(min=1.0)
        loss = (masked_mse.sum() + self.cosine_weight * loss_cos.sum()) / num_valid

        return loss, z_rl, z_hat

    def _denorm(self, z_norm):
        """Inverse of LayerNorm: map z_norm back to original space.

        z_norm = (z - mean) / sqrt(var + eps) * gamma + beta
        => z = (z_norm - beta) / gamma * sqrt(var + eps) + mean
        """
        mean = self._last_mean
        var = self._last_var
        eps = self.norm.epsilon
        gamma = self.norm.scale
        beta = self.norm.bias
        return (z_norm - beta) / gamma * jnp.sqrt(var + eps) + mean

    def encode(self, z, pad_mask):
        """Inference-only: extract z_rl without decoding.

        Args:
            z: VLA embeddings [B, M, D].
            pad_mask: Boolean mask [B, M] (True = valid token).

        Returns:
            z_rl: RL token [B, D].
        """
        return self.encoder(self.norm(z), pad_mask)


def _cosine_similarity(a, b, axis=-1):
    """Compute cosine similarity along given axis."""
    a_norm = a / (jnp.linalg.norm(a, axis=axis, keepdims=True) + 1e-8)
    b_norm = b / (jnp.linalg.norm(b, axis=axis, keepdims=True) + 1e-8)
    return (a_norm * b_norm).sum(axis=axis)
