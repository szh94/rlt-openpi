"""Stage 2 with JAX VLA: Online RL training with frozen VLA + RL token.

Same as ``train_online_rl.py`` but loads the native JAX Pi0 model via
:class:`JaxVLAWrapper` instead of the PyTorch port.

Key difference: Skips VLA weight restoration from the Stage 1 checkpoint
because JAX NNX models do not use ``load_state_dict``.

Usage::

    uv run python scripts/train_online_rl_jax.py --help
    uv run python scripts/train_online_rl_jax.py \\
        --vla-config-name pi0_aloha_sim \\
        --vla-checkpoint-dir /path/to/orbax_checkpoint \\
        --rl-token-checkpoint /path/to/rl_token.pt \\
        --env-factory rlt_openpi.envs.sim.sim_env.make_sim_env
"""

from __future__ import annotations

import json
import logging

import tyro

from rlt_openpi.envs.factory import make_env, make_intervention
from rlt_openpi.envs.intervention import InterventionManager
from rlt_openpi.policies.aloha.config import aloha_data_transforms
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.online_rl_trainer import OnlineRLTrainer
from rlt_openpi.utils.checkpoint import load_rl_token_model
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.jax_vla_wrapper import JaxVLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


def main(config: OnlineRLTrainConfig) -> None:
    """Run online RL training with JAX VLA (Stage 2, Algorithm 1)."""
    log.info("Stage 2 (JAX) config: %s", config)

    # Set up logger
    rl_logger = Logger.from_train_config(config)

    # Load frozen JAX VLA
    log.info(
        "Loading JAX VLA: config=%s, checkpoint=%s",
        config.vla_config_name,
        config.vla_checkpoint_dir,
    )
    vla = JaxVLAWrapper(
        checkpoint_dir=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device="cuda",
        output_action_dim=config.action_dim,
        data_transforms=aloha_data_transforms(),
    )

    # Load frozen RL token model from Stage 1
    log.info("Loading RL token model from %s", config.rl_token_checkpoint)
    rl_token_model = load_rl_token_model(config.rl_token_checkpoint, device="cuda")

    # NOTE: VLA weight restoration from Stage 1 checkpoint is SKIPPED.
    # The JAX NNX model does not use load_state_dict, and Stage 1 with
    # JaxVLAWrapper cannot fine-tune the VLA (vla_finetune_alpha must be 0).
    # The base JAX VLA loaded above is used as-is.

    # Create trainer
    trainer = OnlineRLTrainer(
        config=config,
        vla=vla,
        rl_token_model=rl_token_model,
        device="cuda",
    )

    # Resume from checkpoint if provided
    if config.resume_checkpoint:
        log.info("Resuming from checkpoint: %s", config.resume_checkpoint)
        trainer.load(config.resume_checkpoint)

    # Create environment via pluggable factory.
    if not config.env_factory:
        log.error(
            "--env-factory is required. Provide a Python import path to an env factory function."
        )
        raise SystemExit(1)

    env_extra_kwargs = json.loads(config.env_kwargs)
    env = make_env(
        config.env_factory,
        action_dim=config.action_dim,
        chunk_length=config.chunk_length,
        task_prompt=config.task_prompt,
        max_episode_chunks=config.max_episode_chunks,
        dry_run=config.dry_run,
        **env_extra_kwargs,
    )
    log.info(
        "Environment created: action_dim=%d, chunk_length=%d",
        env.action_dim,
        env.chunk_length,
    )

    # Create intervention manager (VR teleoperation, etc.) if specified.
    intervention_mgr: InterventionManager | None = None
    if config.intervention_factory:
        intervention_mgr = make_intervention(config.intervention_factory, env=env)
        log.info("Intervention manager created via %s", config.intervention_factory)

    trainer.train(env=env, intervention_mgr=intervention_mgr, log_fn=rl_logger.log)

    rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(OnlineRLTrainConfig))
