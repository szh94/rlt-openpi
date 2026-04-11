"""Tests for actor, critic, and TD3 utility functions."""

import torch

from rlt_openpi.models.actor import Actor
from rlt_openpi.models.critic import TwinQCritic
from rlt_openpi.training.td3_utils import actor_loss, compute_td_target, critic_loss

STATE_DIM = 34  # D=32 + d=2
ACTION_CHUNK_DIM = 6  # C=3 * d=2
B = 8
C = 3


class TestActor:
    def test_output_shape(self):
        actor = Actor(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM)
        actor.eval()
        x = torch.randn(B, STATE_DIM)
        a_tilde = torch.randn(B, ACTION_CHUNK_DIM)
        out = actor(x, a_tilde)
        assert out.shape == (B, ACTION_CHUNK_DIM)

    def test_ref_dropout_zeros_some_samples(self):
        actor = Actor(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM, ref_dropout=1.0)
        actor.train()
        a_tilde = torch.ones(B, ACTION_CHUNK_DIM)
        masked = actor._apply_ref_dropout(a_tilde)
        # With dropout=1.0, all should be zeroed
        assert (masked == 0).all()

    def test_ref_dropout_keeps_all_when_zero(self):
        actor = Actor(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM, ref_dropout=0.0)
        actor.train()
        a_tilde = torch.ones(B, ACTION_CHUNK_DIM)
        masked = actor._apply_ref_dropout(a_tilde)
        assert torch.allclose(masked, a_tilde)

    def test_ref_dropout_disabled_in_eval(self):
        actor = Actor(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM, ref_dropout=1.0)
        actor.eval()
        a_tilde = torch.ones(B, ACTION_CHUNK_DIM)
        masked = actor._apply_ref_dropout(a_tilde)
        assert torch.allclose(masked, a_tilde)

    def test_noise_in_train_mode(self):
        torch.manual_seed(42)
        actor = Actor(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM, sigma=0.5, ref_dropout=0.0)
        actor.train()
        x = torch.randn(B, STATE_DIM)
        a_tilde = torch.randn(B, ACTION_CHUNK_DIM)
        out1 = actor(x, a_tilde)
        out2 = actor(x, a_tilde)
        # Two calls with noise should differ
        assert not torch.allclose(out1, out2)

    def test_deterministic_in_eval(self):
        actor = Actor(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM, ref_dropout=0.0)
        actor.eval()
        x = torch.randn(B, STATE_DIM)
        a_tilde = torch.randn(B, ACTION_CHUNK_DIM)
        out1 = actor(x, a_tilde)
        out2 = actor(x, a_tilde)
        assert torch.allclose(out1, out2)


class TestCritic:
    def test_twin_q_output_shapes(self):
        critic = TwinQCritic(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM)
        x = torch.randn(B, STATE_DIM)
        a = torch.randn(B, ACTION_CHUNK_DIM)
        q1, q2 = critic(x, a)
        assert q1.shape == (B, 1)
        assert q2.shape == (B, 1)

    def test_q_min(self):
        critic = TwinQCritic(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM)
        x = torch.randn(B, STATE_DIM)
        a = torch.randn(B, ACTION_CHUNK_DIM)
        q_min = critic.q_min(x, a)
        q1, q2 = critic(x, a)
        expected = torch.min(q1, q2)
        assert torch.allclose(q_min, expected)

    def test_target_q_min(self):
        critic = TwinQCritic(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM)
        x = torch.randn(B, STATE_DIM)
        a = torch.randn(B, ACTION_CHUNK_DIM)
        tq = critic.target_q_min(x, a)
        assert tq.shape == (B, 1)

    def test_polyak_update_changes_targets(self):
        critic = TwinQCritic(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM)
        # Manually change online params so they differ from targets
        with torch.no_grad():
            for p in critic.q1.parameters():
                p.add_(1.0)
        target_before = critic.q1_target.mlp.net[-1].weight.clone()
        critic.update_targets(tau=0.1)
        target_after = critic.q1_target.mlp.net[-1].weight
        assert not torch.allclose(target_before, target_after)

    def test_target_no_grad(self):
        critic = TwinQCritic(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM)
        for p in critic.q1_target.parameters():
            assert not p.requires_grad
        for p in critic.q2_target.parameters():
            assert not p.requires_grad


class TestTD3Utils:
    def test_compute_td_target_shape(self):
        actor = Actor(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM)
        critic = TwinQCritic(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM)
        rewards = torch.randn(B, C)
        dones = torch.zeros(B, 1)
        next_x = torch.randn(B, STATE_DIM)
        next_a_tilde = torch.randn(B, ACTION_CHUNK_DIM)
        td = compute_td_target(rewards, dones, next_x, next_a_tilde, actor, critic, gamma=0.99, chunk_length=C)
        assert td.shape == (B, 1)

    def test_td_target_zero_bootstrap_when_done(self):
        actor = Actor(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM)
        critic = TwinQCritic(state_dim=STATE_DIM, action_chunk_dim=ACTION_CHUNK_DIM)
        rewards = torch.zeros(B, C)
        dones = torch.ones(B, 1)  # all episodes done
        next_x = torch.randn(B, STATE_DIM)
        next_a_tilde = torch.randn(B, ACTION_CHUNK_DIM)
        td = compute_td_target(rewards, dones, next_x, next_a_tilde, actor, critic, gamma=0.99, chunk_length=C)
        # With zero rewards and done=1, target should be zero (no bootstrap)
        assert torch.allclose(td, torch.zeros(B, 1), atol=1e-6)

    def test_critic_loss_scalar(self):
        q1 = torch.randn(B, 1)
        q2 = torch.randn(B, 1)
        target = torch.randn(B, 1)
        loss = critic_loss(q1, q2, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_actor_loss_scalar(self):
        q_val = torch.randn(B, 1)
        a = torch.randn(B, ACTION_CHUNK_DIM)
        a_tilde = torch.randn(B, ACTION_CHUNK_DIM)
        loss = actor_loss(q_val, a, a_tilde, beta=1.0)
        assert loss.shape == ()

    def test_actor_loss_beta_zero_ignores_bc(self):
        q_val = torch.randn(B, 1)
        a = torch.randn(B, ACTION_CHUNK_DIM)
        a_tilde = torch.randn(B, ACTION_CHUNK_DIM)  # different from a
        loss_beta0 = actor_loss(q_val, a, a_tilde, beta=0.0)
        loss_beta1 = actor_loss(q_val, a, a_tilde, beta=1.0)
        # With beta=0, BC term is ignored so loss should just be -Q.mean()
        assert torch.allclose(loss_beta0, -q_val.mean())
        # With beta>0 and a != a_tilde, loss should be larger
        assert loss_beta1.item() > loss_beta0.item()
