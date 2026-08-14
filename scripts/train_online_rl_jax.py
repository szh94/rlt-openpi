"""Stage 2 with JAX VLA: Online RL training with frozen VLA + RL token.

Same as ``train_online_rl.py`` but loads the native JAX Pi0 model via
:class:`JaxVLAWrapper` instead of the PyTorch port.

Key difference: Skips VLA weight restoration from the Stage 1 checkpoint
because JAX NNX models do not use ``load_state_dict``.

Usage::

    python scripts/train_online_rl_jax.py --help
    python scripts/train_online_rl_jax.py \\
        --vla-config-name pi0_aloha_sim \\
        --vla-checkpoint-dir /path/to/orbax_checkpoint \\
        --rl-token-checkpoint /path/to/rl_token.pt \\
        --actor-pretrain-checkpoint /path/to/actor_pretrain.pt \\
        --env-factory rlt_openpi.envs.sim.sim_env.make_sim_env
"""

from __future__ import annotations

import json
import warnings

import tyro

from rlt_openpi.envs.factory import make_env, make_intervention
from rlt_openpi.envs.intervention import InterventionManager
from rlt_openpi.envs.obs_source import make_obs_source
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.online_rl_trainer import OnlineRLTrainer
from rlt_openpi.utils.checkpoint import load_rl_token_model
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.jax_vla_wrapper import JaxVLAWrapper

# 屏蔽 lerobot 内部 torch.tensor(tensor) 触发的无害 UserWarning
# ("To copy construct from a tensor...")
warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")


def main(config: OnlineRLTrainConfig) -> None:
    """Run online RL training with JAX VLA (Stage 2, Algorithm 1)."""
    # Set up logger
    rl_logger = Logger.from_train_config(config)

    # Load frozen JAX VLA
    vla = JaxVLAWrapper(
        checkpoint_dir=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device="cuda",
        output_action_dim=config.action_dim,
    )

    # Load frozen RL token model from Stage 1
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
        print(f"Resuming from checkpoint: {config.resume_checkpoint}")
        trainer.load(config.resume_checkpoint)
    elif config.actor_pretrain_checkpoint:
        trainer.load_actor_pretrain(config.actor_pretrain_checkpoint)
    else:
        raise ValueError(
            "--actor-pretrain-checkpoint is required unless --resume-checkpoint is provided"
        )

    # Create environment via pluggable factory.
    if not config.env_factory:
        print(
            "[ERROR] --env-factory is required. Provide a Python import path to an env factory function."
        )
        raise SystemExit(1)

    env_extra_kwargs = json.loads(config.env_kwargs)

    # 黑盒 obs 来源：除 "robot" 外（mock/dataset），构建 ObsSource 并注入 env 工厂，
    # 工厂会跳过机器人初始化，obs 全部来自该黑盒。
    obs_source = None
    if config.obs_source and config.obs_source != "robot":
        obs_kwargs: dict = {"image_size": tuple(env_extra_kwargs.get("image_size", (224, 224)))}
        if "camera_names" in env_extra_kwargs:
            obs_kwargs["camera_names"] = env_extra_kwargs["camera_names"]
        if config.obs_source == "dataset":
            if not config.repo_id:
                print("[ERROR] --obs-source dataset requires --repo-id (LeRobot 数据集 ID)")
                raise SystemExit(1)
            obs_kwargs["repo_id"] = config.repo_id
            obs_kwargs["vla_config_name"] = config.vla_config_name
            obs_kwargs["task_prompt"] = config.task_prompt
        else:  # mock
            obs_kwargs["task_prompt"] = config.task_prompt
        obs_source = make_obs_source(config.obs_source, **obs_kwargs)
        print(
            f"[ObsSource] built {type(obs_source).__name__} from --obs-source={config.obs_source}",
        )

    env_kwargs = {
        "action_dim": config.action_dim,
        "chunk_length": config.chunk_length,
        "task_prompt": config.task_prompt,
        "max_episode_chunks": config.max_episode_chunks,
        "dry_run": config.dry_run,
        **env_extra_kwargs,
    }
    if obs_source is not None:
        env_kwargs["obs_source"] = obs_source
    env = make_env(config.env_factory, **env_kwargs)
    print(
        f"Environment created: action_dim={env.action_dim}, chunk_length={env.chunk_length}"
    )

    # Create intervention manager (VR teleoperation, etc.) if specified.
    intervention_mgr: InterventionManager | None = None
    if config.intervention_factory:
        intervention_mgr = make_intervention(config.intervention_factory, env=env)
        print(f"Intervention manager created via {config.intervention_factory}")

    trainer.train(
        env=env,
        intervention_mgr=intervention_mgr,
        log_fn=rl_logger.log,
    )

    rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(OnlineRLTrainConfig))
