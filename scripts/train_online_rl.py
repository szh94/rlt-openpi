"""Stage 2: Online RL training with frozen VLA + RL token (Algorithm 1).

Usage:
    uv run python scripts/train_online_rl.py --help
    uv run python scripts/train_online_rl.py --vla-config-name pi0_aloha_sim \
        --vla-checkpoint-dir /path/to/vla.safetensors \
        --rl-token-checkpoint /path/to/rl_token.pt
"""

from __future__ import annotations

import json

import torch
import tyro

from rlt_openpi.envs.factory import make_env, make_intervention
from rlt_openpi.envs.intervention import InterventionManager
from rlt_openpi.envs.obs_source import make_obs_source
from rlt_openpi.policies.aloha.config import aloha_data_transforms
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.trainer_s2_onlinerl import OnlineRLTrainer
from rlt_openpi.utils.checkpoint import load_rl_token_model
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.vla_wrapper import VLAWrapper

def main(config: OnlineRLTrainConfig) -> None:
    """Run online RL training (Stage 2, Algorithm 1)."""
    # Set up logger
    rl_logger = Logger.from_train_config(config)

    # Load frozen VLA
    vla = VLAWrapper(
        checkpoint_path=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device="cuda",
        output_action_dim=config.action_dim,
        data_transforms=aloha_data_transforms(),
    )

    # Load frozen RL token model from Stage 1
    rl_token_model = load_rl_token_model(config.rl_token_checkpoint, device="cuda")

    # Restore fine-tuned VLA weights from Stage 1 checkpoint (if available).
    # Load to CPU first to avoid OOM — the VLA + RL token already occupy most VRAM.
    stage1_ckpt = torch.load(config.rl_token_checkpoint, map_location="cpu", weights_only=False)
    if "vla_model" in stage1_ckpt:
        vla.extractor.pi0.load_state_dict(stage1_ckpt["vla_model"])
        print("Restored fine-tuned VLA weights from Stage 1 checkpoint")
    else:
        print("[WARNING] No fine-tuned VLA weights found in Stage 1 checkpoint; using base VLA")
    del stage1_ckpt
    torch.cuda.empty_cache()

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

    # Create environment via pluggable factory.
    # Pass --env-factory to specify a Python import path, e.g.:
    #   --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env
    #   --env-factory rlt_openpi.envs.sim.sim_env.make_sim_env
    if not config.env_factory:
        print("[ERROR] --env-factory is required. Provide a Python import path to an env factory function.")
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
    # Create intervention manager (VR teleoperation, etc.) if specified.
    intervention_mgr: InterventionManager | None = None
    if config.intervention_factory:
        intervention_mgr = make_intervention(config.intervention_factory, env=env)

    trainer.train(env=env, intervention_mgr=intervention_mgr, log_fn=rl_logger.log)

    rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(OnlineRLTrainConfig))
