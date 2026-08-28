"""Tests for the RLT actor action parameterization."""

import torch

from rlt_openpi.models.actor import Actor


def test_zero_initialized_actor_matches_reference() -> None:
    actor = Actor(
        state_dim=2,
        action_chunk_dim=3,
        hidden_dim=4,
        num_hidden_layers=1,
        sigma=0.0,
        ref_dropout=0.0,
    )
    actor.eval()

    action = actor(torch.ones(1, 2), torch.tensor([[0.2, -0.4, 0.8]]))

    torch.testing.assert_close(action, torch.tensor([[0.2, -0.4, 0.8]]))


def test_reference_dropout_preserves_reference_skip_path() -> None:
    actor = Actor(
        state_dim=2,
        action_chunk_dim=3,
        hidden_dim=4,
        num_hidden_layers=1,
        sigma=0.0,
        ref_dropout=1.0,
    )
    actor.train()

    action = actor(torch.ones(1, 2), torch.tensor([[0.2, -0.4, 0.8]]))

    torch.testing.assert_close(action, torch.tensor([[0.2, -0.4, 0.8]]))
