"""Tests for replay buffer."""

import numpy as np
import torch

from rlt_openpi.training.replay_buffer import ReplayBuffer

STATE_DIM = 34
ACTION_CHUNK_DIM = 6
C = 3


def _make_buffer(capacity=100):
    return ReplayBuffer(
        capacity=capacity,
        state_dim=STATE_DIM,
        action_chunk_dim=ACTION_CHUNK_DIM,
        chunk_length=C,
    )


def test_add_and_size():
    buf = _make_buffer()
    assert buf.size == 0
    buf.add(
        x=np.zeros(STATE_DIM),
        a=np.zeros(ACTION_CHUNK_DIM),
        a_tilde=np.zeros(ACTION_CHUNK_DIM),
        rewards=np.zeros(C),
        next_x=np.zeros(STATE_DIM),
        done=0.0,
    )
    assert buf.size == 1


def test_sample_shapes():
    buf = _make_buffer()
    for i in range(10):
        buf.add(
            x=np.random.randn(STATE_DIM).astype(np.float32),
            a=np.random.randn(ACTION_CHUNK_DIM).astype(np.float32),
            a_tilde=np.random.randn(ACTION_CHUNK_DIM).astype(np.float32),
            rewards=np.random.randn(C).astype(np.float32),
            next_x=np.random.randn(STATE_DIM).astype(np.float32),
            done=0.0,
        )
    batch = buf.sample(batch_size=4, device="cpu")
    assert batch["x"].shape == (4, STATE_DIM)
    assert batch["a"].shape == (4, ACTION_CHUNK_DIM)
    assert batch["a_tilde"].shape == (4, ACTION_CHUNK_DIM)
    assert batch["rewards"].shape == (4, C)
    assert batch["next_x"].shape == (4, STATE_DIM)
    assert batch["dones"].shape == (4, 1)
    assert isinstance(batch["x"], torch.Tensor)


def test_capacity_wraps():
    buf = _make_buffer(capacity=5)
    for i in range(10):
        buf.add(
            x=np.full(STATE_DIM, i, dtype=np.float32),
            a=np.zeros(ACTION_CHUNK_DIM),
            a_tilde=np.zeros(ACTION_CHUNK_DIM),
            rewards=np.zeros(C),
            next_x=np.zeros(STATE_DIM),
            done=0.0,
        )
    assert buf.size == 5
    # Buffer should contain items 5-9 (wrapped around)
    batch = buf.sample(batch_size=5, device="cpu")
    values = batch["x"][:, 0].numpy()
    for v in values:
        assert v >= 5.0, f"Expected wrapped values >= 5, got {v}"


def test_add_episode_strided():
    buf = _make_buffer()
    N = 20
    xs = np.random.randn(N, STATE_DIM).astype(np.float32)
    actions = np.random.randn(N, ACTION_CHUNK_DIM).astype(np.float32)
    a_tildes = np.random.randn(N, ACTION_CHUNK_DIM).astype(np.float32)
    rewards = np.random.randn(N, C).astype(np.float32)
    next_xs = np.random.randn(N, STATE_DIM).astype(np.float32)
    dones = np.zeros((N, 1), dtype=np.float32)
    dones[-1] = 1.0

    stored = buf.add_episode_strided(xs, actions, a_tildes, rewards, next_xs, dones, stride=2)
    assert stored == 10  # 20 / 2 = 10
    assert buf.size == 10


def test_add_episode_strided_odd():
    buf = _make_buffer()
    N = 7
    xs = np.random.randn(N, STATE_DIM).astype(np.float32)
    actions = np.random.randn(N, ACTION_CHUNK_DIM).astype(np.float32)
    a_tildes = np.random.randn(N, ACTION_CHUNK_DIM).astype(np.float32)
    rewards = np.random.randn(N, C).astype(np.float32)
    next_xs = np.random.randn(N, STATE_DIM).astype(np.float32)
    dones = np.zeros((N, 1), dtype=np.float32)

    stored = buf.add_episode_strided(xs, actions, a_tildes, rewards, next_xs, dones, stride=2)
    # indices: 0, 2, 4, 6 → 4 transitions
    assert stored == 4
    assert buf.size == 4


def test_sample_returns_correct_data():
    buf = _make_buffer()
    x = np.ones(STATE_DIM, dtype=np.float32) * 42.0
    buf.add(
        x=x,
        a=np.zeros(ACTION_CHUNK_DIM),
        a_tilde=np.zeros(ACTION_CHUNK_DIM),
        rewards=np.zeros(C),
        next_x=np.zeros(STATE_DIM),
        done=1.0,
    )
    batch = buf.sample(batch_size=1, device="cpu")
    assert torch.allclose(batch["x"][0, 0], torch.tensor(42.0))
    assert batch["dones"][0, 0].item() == 1.0
