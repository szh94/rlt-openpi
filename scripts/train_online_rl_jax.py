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

# import json
import importlib

import tyro

# NOTE: env imports commented out — using mock env (make_aloha_obs)
# from rlt_openpi.envs.factory import make_env, make_intervention
# from rlt_openpi.envs.intervention import InterventionManager
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.data_loader import build_data_loader
from rlt_openpi.training.online_rl_trainer import OnlineRLTrainer
from rlt_openpi.utils.checkpoint import load_rl_token_model
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.jax_vla_wrapper import JaxVLAWrapper


def _resolve_data_transforms(dotted_path: str | None, openpi_config_name: str):
    """Dynamically import and call a data-transforms factory (same as Stage 1)."""
    if dotted_path is None:
        return None

    from openpi.training.config import get_config

    module_path, func_name = dotted_path.rsplit(".", 1)
    factory_fn = getattr(importlib.import_module(module_path), func_name)
    return factory_fn(get_config(openpi_config_name).model)


def main(config: OnlineRLTrainConfig) -> None:
    """Run online RL training with JAX VLA (Stage 2, Algorithm 1)."""
    # Set up logger
    rl_logger = Logger.from_train_config(config)

    # Resolve data transforms (same pattern as Stage 1 JAX)
    data_transforms = _resolve_data_transforms(
        config.data_transforms_fn, config.vla_config_name
    )

    # Load frozen JAX VLA
    vla = JaxVLAWrapper(
        checkpoint_dir=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device="cuda",
        output_action_dim=config.action_dim,
        data_transforms=data_transforms,
    )

    # Load frozen RL token model from Stage 1
    rl_token_model = load_rl_token_model(config.rl_token_checkpoint, device="cuda")

    # NOTE: VLA weight restoration from Stage 1 checkpoint is SKIPPED.
    # The JAX NNX model does not use load_state_dict, and Stage 1 with
    # JaxVLAWrapper cannot fine-tune the VLA (vla_finetune_alpha must be 0).
    # The base JAX VLA loaded above is used as-is.

    # Build BC pre-training data loader (if configured)
    pretrain_data_iter = None
    if config.repo_id and config.actor_pretrain_steps > 0:
        print(f"[Data] Building BC pretrain data loader: {config.repo_id}")
        pretrain_dataloader = build_data_loader(
            openpi_config_name=config.vla_config_name,
            repo_id=config.repo_id,
            batch_size=config.actor_pretrain_batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            data_transforms=data_transforms,
        )
        pretrain_data_iter = iter(pretrain_dataloader)

    # Create trainer
    trainer = OnlineRLTrainer(
        config=config,
        vla=vla,
        rl_token_model=rl_token_model,
        device="cuda",
    )

    # Resume from checkpoint if provided
    if config.resume_checkpoint:
        print(f"Resuming from checkpoint: {config.resume_checkpoint}")
        trainer.load(config.resume_checkpoint)

    # NOTE: env-factory check commented out — using mock env (make_aloha_obs)
    # if not config.env_factory:
    #     print(
    #         "[ERROR] --env-factory is required. Provide a Python import path to an env factory function."
    #     )
    #     raise SystemExit(1)

    # NOTE: env/intervention creation commented out — using mock env (make_aloha_obs)
    # env_extra_kwargs = json.loads(config.env_kwargs)
    # env = make_env(
    #     config.env_factory,
    #     action_dim=config.action_dim,
    #     chunk_length=config.chunk_length,
    #     task_prompt=config.task_prompt,
    #     max_episode_chunks=config.max_episode_chunks,
    #     dry_run=config.dry_run,
    #     **env_extra_kwargs,
    # )
    # print(
    #     f"Environment created: action_dim={env.action_dim}, chunk_length={env.chunk_length}"
    # )

    # # Create intervention manager (VR teleoperation, etc.) if specified.
    # intervention_mgr: InterventionManager | None = None
    # if config.intervention_factory:
    #     intervention_mgr = make_intervention(config.intervention_factory, env=env)
    #     print(f"Intervention manager created via {config.intervention_factory}")

    trainer.train(env=None, intervention_mgr=None, log_fn=rl_logger.log, pretrain_data_iter=pretrain_data_iter)

    rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(OnlineRLTrainConfig))
