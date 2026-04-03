"""Tests for shared MLP building block."""

import torch

from rlt_openpi.models.networks import MLP


def test_mlp_output_shape():
    mlp = MLP(input_dim=100, output_dim=1, hidden_dim=64, num_hidden_layers=2)
    out = mlp(torch.randn(4, 100))
    assert out.shape == (4, 1)


def test_mlp_single_hidden_layer():
    mlp = MLP(input_dim=50, output_dim=10, hidden_dim=32, num_hidden_layers=1)
    out = mlp(torch.randn(8, 50))
    assert out.shape == (8, 10)


def test_mlp_no_hidden_layers():
    mlp = MLP(input_dim=20, output_dim=5, hidden_dim=64, num_hidden_layers=0)
    out = mlp(torch.randn(2, 20))
    assert out.shape == (2, 5)


def test_mlp_gradient_flow():
    mlp = MLP(input_dim=10, output_dim=1, hidden_dim=16, num_hidden_layers=2)
    x = torch.randn(4, 10, requires_grad=True)
    out = mlp(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == (4, 10)
