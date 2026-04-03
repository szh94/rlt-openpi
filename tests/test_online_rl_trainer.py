"""Integration test for the online RL trainer (Stage 2)."""

from unittest.mock import MagicMock

import gymnasium as gym
import torch

from rlt_openpi.models.rl_token import RLTokenModel
from rlt_openpi.rollout.env_wrapper import RLTEnv
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.online_rl_trainer import OnlineRLTrainer

# Small dims for fast tests
D = 32
ACTION_DIM = 2
C = 3
M = 5


def _make_mock_vla():
    mock_vla = MagicMock()
    mock_vla.extract_embeddings.return_value = (
        torch.randn(1, M, D),
        torch.ones(1, M, dtype=torch.bool),
    )
    mock_vla.get_rl_chunk_reference.return_value = torch.randn(1, C, ACTION_DIM)
    return mock_vla


def _make_config(**overrides):
    defaults = dict(
        embedding_dim=D,
        action_dim=ACTION_DIM,
        chunk_length=C,
        mlp_hidden_dim=32,
        mlp_num_hidden_layers=1,
        actor_noise_sigma=0.1,
        ref_action_dropout=0.0,
        gamma=0.99,
        tau=0.005,
        utd_ratio=2,
        bc_regularizer_beta=1.0,
        critic_updates_per_actor=2,
        actor_lr=1e-3,
        critic_lr=1e-3,
        buffer_capacity=100,
        warmup_steps=5,
        max_env_steps=30,
        save_every=99999,
        log_every=1,
        wandb_enabled=False,
    )
    defaults.update(overrides)
    return OnlineRLTrainConfig(**defaults)


def _make_env(max_episode_steps=10):
    raw_env = gym.make("MountainCarContinuous-v0", max_episode_steps=max_episode_steps)
    return RLTEnv(raw_env, action_dim=ACTION_DIM, chunk_length=C)


def test_trainer_construction():
    config = _make_config()
    vla = _make_mock_vla()
    rl_token = RLTokenModel(embedding_dim=D, encoder_layers=1, encoder_heads=4, decoder_layers=1, decoder_heads=4)
    trainer = OnlineRLTrainer(config, vla, rl_token, device="cpu")

    assert trainer.actor is not None
    assert trainer.critic is not None
    assert trainer.replay_buffer.size == 0


def test_full_training_loop():
    config = _make_config(warmup_steps=3, max_env_steps=20, utd_ratio=2)
    vla = _make_mock_vla()
    rl_token = RLTokenModel(embedding_dim=D, encoder_layers=1, encoder_heads=4, decoder_layers=1, decoder_heads=4)
    trainer = OnlineRLTrainer(config, vla, rl_token, device="cpu")
    env = _make_env(max_episode_steps=10)

    logged = []
    trainer.train(env=env, log_fn=logged.append)

    assert trainer._total_episodes > 0
    assert trainer._total_updates > 0
    assert trainer.replay_buffer.size >= config.warmup_steps
    assert len(logged) > 0


def test_losses_are_finite():
    config = _make_config(warmup_steps=3, max_env_steps=20, utd_ratio=3)
    vla = _make_mock_vla()
    rl_token = RLTokenModel(embedding_dim=D, encoder_layers=1, encoder_heads=4, decoder_layers=1, decoder_heads=4)
    trainer = OnlineRLTrainer(config, vla, rl_token, device="cpu")
    env = _make_env(max_episode_steps=10)

    logged = []
    trainer.train(env=env, log_fn=logged.append)

    for m in logged:
        assert torch.isfinite(torch.tensor(m.get("critic_loss", 0.0)))
        if "actor_loss" in m:
            assert torch.isfinite(torch.tensor(m["actor_loss"]))


def test_buffer_grows():
    config = _make_config(warmup_steps=5, max_env_steps=40)
    vla = _make_mock_vla()
    rl_token = RLTokenModel(embedding_dim=D, encoder_layers=1, encoder_heads=4, decoder_layers=1, decoder_heads=4)
    trainer = OnlineRLTrainer(config, vla, rl_token, device="cpu")
    env = _make_env(max_episode_steps=10)

    trainer.train(env=env)

    # Buffer should have warmup + episode transitions
    assert trainer.replay_buffer.size > config.warmup_steps


def test_save_and_load(tmp_path):
    config = _make_config(warmup_steps=2, max_env_steps=15)
    vla = _make_mock_vla()
    rl_token = RLTokenModel(embedding_dim=D, encoder_layers=1, encoder_heads=4, decoder_layers=1, decoder_heads=4)
    trainer = OnlineRLTrainer(config, vla, rl_token, device="cpu")
    env = _make_env(max_episode_steps=10)
    trainer.train(env=env)

    # Save
    ckpt_path = trainer.save(str(tmp_path))

    # Load into a new trainer
    trainer2 = OnlineRLTrainer(config, vla, rl_token, device="cpu")
    trainer2.load(str(ckpt_path))

    assert trainer2._total_episodes == trainer._total_episodes
    assert trainer2._total_updates == trainer._total_updates
    assert trainer2._total_env_steps == trainer._total_env_steps
