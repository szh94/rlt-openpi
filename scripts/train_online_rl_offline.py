"""Offline Stage 2 pipeline test with real VLA + real Stage 1 checkpoint.

Uses MockEnv (fake ALOHA-format observations) + real VLA checkpoint +
real Stage 1 RLTokenModel checkpoint, then calls the **exact same**
``OnlineRLTrainer.train()`` as ``train_online_rl.py`` — only the env
is different.

    test_stage2_offline.py          train_online_rl.py
    ─────────────────────           ──────────────────
    MockEnv (fake obs)      ←diff→  RobotEnv (real robot)
    OnlineRLTrainer         ←same→  OnlineRLTrainer

Usage::

    python scripts/train_online_rl_offline.py \
        --vla-checkpoint-dir ~/.cache/openpi/.../model.safetensors \
        --rl-token-checkpoint checkpoints/stage1_rlt_encoder/run_xxx/step_5000.pt \
        --episodes 5
"""

from __future__ import annotations

import logging

import torch
import tyro

from rlt_openpi.envs.intervention import InterventionManager
from rlt_openpi.envs.mock.mock_env import MockEnv
from rlt_openpi.policies.aloha.config import aloha_data_transforms
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.online_rl_trainer import OnlineRLTrainer
from rlt_openpi.utils.checkpoint import load_rl_token_model
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


def main(config: OnlineRLTrainConfig) -> None:
    """Run offline Stage 2 test — same pipeline, fake env."""
    log.info("=== Stage 2 Offline Test (MockEnv) ===")

    # Set up logger
    rl_logger = Logger.from_train_config(config)

    # ── 1. Load frozen VLA (same as train_online_rl.py) ────────────
    log.info("Loading VLA: config=%s, checkpoint=%s",
             config.vla_config_name, config.vla_checkpoint_dir)
    vla = VLAWrapper(
        checkpoint_path=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device="cuda",
        output_action_dim=config.action_dim,
        data_transforms=aloha_data_transforms(),
    )

    # ── 2. Load frozen RL token model (same as train_online_rl.py) ─
    log.info("Loading RL token model from %s", config.rl_token_checkpoint)
    rl_token_model = load_rl_token_model(config.rl_token_checkpoint, device="cuda")

    # Restore fine-tuned VLA weights from Stage 1 checkpoint (same as train_online_rl.py)
    stage1_ckpt = torch.load(config.rl_token_checkpoint, map_location="cpu", weights_only=False)
    if "vla_model" in stage1_ckpt:
        vla.extractor.pi0.load_state_dict(stage1_ckpt["vla_model"])
        log.info("Restored fine-tuned VLA weights from Stage 1 checkpoint")
    else:
        log.info("No fine-tuned VLA weights in checkpoint; using base VLA")
    del stage1_ckpt
    torch.cuda.empty_cache()

    # ── 3. Create trainer (same as train_online_rl.py) ──────────────
    trainer = OnlineRLTrainer(
        config=config,
        vla=vla,
        rl_token_model=rl_token_model,
        device="cuda",
    )

    if config.resume_checkpoint:
        log.info("Resuming from checkpoint: %s", config.resume_checkpoint)
        trainer.load(config.resume_checkpoint)

    # ── 4. Create MockEnv instead of real robot ─────────────────────
    # THIS is the ONLY line that differs from train_online_rl.py
    log.info("Creating MockEnv (fake observations, no robot)")
    env = MockEnv(
        action_dim=config.action_dim,
        chunk_length=config.chunk_length,
        task_prompt=config.task_prompt,
        max_episode_chunks=config.max_episode_chunks,
        image_size=config.mock_image_size,
    )

    # ── 5. Run training (same OnlineRLTrainer.train) ────────────────
    # No VR intervention for mock test
    intervention_mgr: InterventionManager | None = None

    trainer.train(env=env, intervention_mgr=intervention_mgr, log_fn=rl_logger.log)

    rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(OnlineRLTrainConfig))
