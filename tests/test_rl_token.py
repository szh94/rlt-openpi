"""Tests for RL token encoder-decoder model."""

import torch

from rlt_openpi.models.rl_token import RLTokenDecoder, RLTokenEncoder, RLTokenModel

D = 64
B = 4
M = 10


def test_encoder_output_shape():
    encoder = RLTokenEncoder(embedding_dim=D, num_layers=1, num_heads=4)
    z = torch.randn(B, M, D)
    pad_mask = torch.ones(B, M, dtype=torch.bool)
    z_rl = encoder(z, pad_mask)
    assert z_rl.shape == (B, D)


def test_encoder_with_padding():
    encoder = RLTokenEncoder(embedding_dim=D, num_layers=1, num_heads=4)
    z = torch.randn(B, M, D)
    pad_mask = torch.ones(B, M, dtype=torch.bool)
    pad_mask[:, -3:] = False  # last 3 tokens padded
    z_rl = encoder(z, pad_mask)
    assert z_rl.shape == (B, D)


def test_decoder_output_shape():
    decoder = RLTokenDecoder(embedding_dim=D, num_layers=1, num_heads=4)
    z_rl = torch.randn(B, D)
    z = torch.randn(B, M, D)
    pad_mask = torch.ones(B, M, dtype=torch.bool)
    z_hat = decoder(z_rl, z, pad_mask)
    assert z_hat.shape == (B, M, D)


def test_model_forward_returns_loss_and_tensors():
    model = RLTokenModel(embedding_dim=D, encoder_layers=1, encoder_heads=4, decoder_layers=1, decoder_heads=4)
    z = torch.randn(B, M, D)
    pad_mask = torch.ones(B, M, dtype=torch.bool)
    loss, z_rl, z_hat = model(z, pad_mask)
    assert loss.shape == ()
    assert loss.item() > 0
    assert z_rl.shape == (B, D)
    assert z_hat.shape == (B, M, D)


def test_model_encode_inference():
    model = RLTokenModel(embedding_dim=D, encoder_layers=1, encoder_heads=4, decoder_layers=1, decoder_heads=4)
    model.eval()
    z = torch.randn(B, M, D)
    pad_mask = torch.ones(B, M, dtype=torch.bool)
    z_rl = model.encode(z, pad_mask)
    assert z_rl.shape == (B, D)


def test_model_loss_decreases():
    model = RLTokenModel(embedding_dim=D, encoder_layers=1, encoder_heads=4, decoder_layers=1, decoder_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    z = torch.randn(B, M, D)
    pad_mask = torch.ones(B, M, dtype=torch.bool)
    pad_mask[:, -2:] = False

    losses = []
    for _ in range(10):
        loss, _, _ = model(z, pad_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Average of last 3 should be less than average of first 3
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3, f"Loss did not decrease: {losses}"


def test_model_stop_gradient():
    """Verify that VLA embeddings are detached inside forward."""
    model = RLTokenModel(embedding_dim=D, encoder_layers=1, encoder_heads=4, decoder_layers=1, decoder_heads=4)
    z = torch.randn(B, M, D, requires_grad=True)
    pad_mask = torch.ones(B, M, dtype=torch.bool)
    loss, _, _ = model(z, pad_mask)
    loss.backward()
    # z should NOT have gradients (detached inside model)
    assert z.grad is None
