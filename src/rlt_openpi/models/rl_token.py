"""RL Token encoder-decoder model (Stage 1).

Compresses variable-length VLA prefix embeddings z_{1:M} into a single
fixed-size RL token z_rl via an information bottleneck, and reconstructs
the original embeddings to train the bottleneck via masked MSE + cosine loss.

Paper reference: "RL Token: Bootstrapping Online RL with VLA Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RLTokenEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
        max_position_embeddings: int = 1024, # 预设一个足够大的最大Token数
    ) -> None:
        super().__init__()
        self.e_rl = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        
        # Encoder position embeddings (M positions + 1 RL position)
        self.pos_embeddings = nn.Embedding(max_position_embeddings, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        B, M, D = z.shape

        # Append learnable e_rl token: [B, M+1, D]
        e_rl = self.e_rl.expand(B, -1, -1)
        tokens = torch.cat([z, e_rl], dim=1)

        # Add position embeddings to input tokens
        pos_indices = torch.arange(M + 1, device=z.device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.pos_embeddings(pos_indices)

        # Extend pad_mask for the RL token (always valid)
        rl_mask = torch.ones(B, 1, dtype=torch.bool, device=z.device)
        extended_pad_mask = torch.cat([pad_mask, rl_mask], dim=1)
        ignore_mask = ~extended_pad_mask

        out = self.transformer(tokens, src_key_padding_mask=ignore_mask)

        # Extract RL token output (last position)
        z_rl = out[:, -1, :]  # [B, D]
        return z_rl


class RLTokenDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
        max_position_embeddings: int = 1024,
    ) -> None:
        super().__init__()
        
        # Decoder position embeddings
        self.pos_embeddings = nn.Embedding(max_position_embeddings, embedding_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.h_phi = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, z_rl: Tensor, z: Tensor, pad_mask: Tensor) -> Tensor:
        B, M, D = z.shape
        
        # Teacher-forced input: [z_rl, z_1, ..., z_{M-1}]
        tgt = torch.cat([z_rl.unsqueeze(1), z[:, :-1, :]], dim=1)  # [B, M, D]

        # Inject explicit position embeddings into decoder target
        pos_indices = torch.arange(M, device=z.device).unsqueeze(0).expand(B, -1)
        tgt = tgt + self.pos_embeddings(pos_indices)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            M,
            device=tgt.device,
        )

        # Memory = z_rl as a single token for cross-attention
        memory = z_rl.unsqueeze(1)  # [B, 1, D]

        # The single memory token can also carry position information
        tgt_key_padding_mask = ~pad_mask

        out = self.transformer(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        z_hat = self.h_phi(out)  # [B, M, D]
        return z_hat


class RLTokenModel(nn.Module):
    """Combined RL token encoder-decoder for Stage 1 training.

    Training: forward(z, pad_mask) → (loss, z_rl, z_hat)
    Inference: encode(z, pad_mask) → z_rl

    Args:
        embedding_dim: Dimension of VLA embeddings.
        encoder_layers: Number of encoder transformer layers.
        encoder_heads: Number of encoder attention heads.
        decoder_layers: Number of decoder transformer layers.
        decoder_heads: Number of decoder attention heads.
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        cosine_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = RLTokenEncoder(embedding_dim, encoder_layers, encoder_heads)
        self.decoder = RLTokenDecoder(embedding_dim, decoder_layers, decoder_heads)
        self.cosine_weight = cosine_weight
        self.norm = nn.LayerNorm(embedding_dim)  # 输入归一化，稳定训练

    def forward(
        self,
        z: Tensor,
        pad_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        z = z.detach()
        z_norm = self.norm(z)  # LayerNorm across D dim

        # z_rl = self.encoder(z, pad_mask)
        z_rl = self.encoder(z_norm, pad_mask)
        # z_hat = self.decoder(z_rl, z, pad_mask)
        z_hat = self.decoder(z_rl, z_norm, pad_mask)

        # print(f"z_norm  min={z_norm.min().item():.4f} max={z_norm.max().item():.4f} mean={z_norm.mean().item():.4f}")
        # print(f"z_hat   min={z_hat.min().item():.4f} max={z_hat.max().item():.4f} mean={z_hat.mean().item():.4f}")

        # Masked MSE: only compute loss on valid (non-padded) positions
        # mse = (z_hat - z).pow(2).mean(dim=-1)  # [B, M]
        mse = (z_hat - z_norm).pow(2).mean(dim=-1)  # [B, M]

        masked_mse = mse * pad_mask.float()  # zero out padded positions

        # Masked cosine similarity loss: 1 - cos(z_hat, z_norm), direction alignment
        cos_sim = F.cosine_similarity(z_hat, z_norm, dim=-1)  # [B, M], range [-1, 1]
        loss_cos = (1.0 - cos_sim) * pad_mask.float()

        # Average over valid tokens
        # num_valid = pad_mask.float().sum().clamp(min=1.0)
        num_valid = pad_mask.float().sum().clamp(min=1.0)
        # loss = masked_mse.sum() / num_valid
        loss = (masked_mse.sum() + self.cosine_weight * loss_cos.sum()) / num_valid

        return loss, z_rl, z_hat

    @torch.no_grad()
    def encode(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        """Inference-only: extract z_rl without decoding.

        Args:
            z: VLA embeddings [B, M, D].
            pad_mask: Boolean mask [B, M] (True = valid token).

        Returns:
            z_rl: RL token [B, D].
        """
        return self.encoder(self.norm(z), pad_mask)
