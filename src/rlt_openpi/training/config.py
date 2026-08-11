"""Configuration dataclasses for RLT-OpenPI training stages."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated

import tyro


@dataclass
class RLTokenTrainConfig:
    """Stage 1: RL token encoder-decoder training hyperparameters."""

    # Architecture
    embedding_dim: int = 2048
    encoder_layers: int = 2
    encoder_heads: int = 8
    decoder_layers: int = 2
    decoder_heads: int = 8

    # Training
    num_train_steps: int = 5000
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500  # Linear LR warmup steps (matches OpenPI default)
    max_grad_norm: float = 1.0  # Global gradient norm clipping (matches OpenPI default)
    vla_finetune_alpha: float = 0.0  # VLA fine-tuning weight (always 0 = frozen VLA, joint training disabled)
    # --- VLA joint training fields (disabled) ---
    # vla_learning_rate: float = 1e-5  # VLA fine-tuning learning rate (used when alpha > 0)
    # gradient_checkpointing: bool = True  # Enable gradient checkpointing to reduce VRAM

    # Checkpoints
    vla_checkpoint_dir: str = ""
    vla_config_name: str = "pi05_droid_finetune"
    resume_checkpoint: str = ""  # Path to Stage 1 checkpoint to resume training from
    save_dir: str = "checkpoints/stage1_rlt_encoder"
    run_name: str = ""  # Subdirectory name for this run (auto-generated if empty)
    save_every: int = 10000
    log_every: int = 1000  # wandb logging interval (steps)
    print_every: int = 100  # stdout logging interval (steps)

    # wandb
    wandb_project: str = "rlt-openpi"
    wandb_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.run_name:
            self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")


@dataclass
class OnlineRLTrainConfig:
    """Stage 2: Online RL training hyperparameters (Algorithm 1)."""

    # Architecture (shared by RLTokenTrainConfig — must match the Stage 1 model)
    embedding_dim: int = 2048

    # Action space (ALOHA dual-arm: 6 DOF + 1 gripper per arm = 14)
    action_dim: int = 14
    chunk_length: int = 10  # C
    vla_action_horizon: int = 50  # H: number of action steps the VLA outputs

    # Actor-critic architecture
    mlp_hidden_dim: int = 256
    mlp_num_hidden_layers: int = 2
    actor_noise_sigma: float = 0.1  # actor exploration noise std
    ref_action_dropout: float = 0.5
    max_deviation: float = 3.0  # safety cap: max |a - a_tilde| per action element
    deviation_abort_threshold: float = 0.8  # abort if raw |a - a_tilde| exceeds this

    # RL hyperparameters
    gamma: float = 0.99
    tau: float = 0.005  # Polyak averaging coefficient
    utd_ratio: int = 5  # G: update-to-data ratio
    bc_regularizer_beta: float = 0.5  # BC regularizer coefficient
    critic_updates_per_actor: int = 2
    target_noise_sigma: float = 0.2  # TD3 target policy smoothing noise std
    target_noise_clip: float = 0.5  # clamp range for target noise

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Replay buffer
    buffer_capacity: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 1000

    # Actor BC pre-training (before warmup)
    repo_id: str = ""  # LeRobot dataset repo ID (e.g. "aloha_phone"). Empty = skip pre-training.
    actor_pretrain_steps: int = 1000  # Number of BC pre-training steps. 0 = skip.
    actor_pretrain_batch_size: int = 64  # Batch size for actor pre-training.
    num_workers: int = 4  # DataLoader worker processes for BC pre-training.
    data_transforms_fn: str | None = None  # Dotted path to data-transforms factory (same as Stage 1).

    # Environment
    env_factory: str = ""  # Python import path, e.g. "rlt_openpi.envs.franka.env_factory.make_franka_env"
    intervention_factory: str = ""  # Python import path, e.g. "rlt_openpi.envs.franka.intervention.make_vr_intervention"
    task_prompt: str = ""  # Task instruction for VLA (passed to env factory)
    max_episode_chunks: int = 150  # Max chunks per episode before forced termination
    env_kwargs: str = "{}"  # JSON string of extra kwargs forwarded to the env factory
    dry_run: Annotated[str, tyro.conf.arg(metavar="{true,false}")] = (
        "false"  # If True, print actions instead of sending to hardware
    )
    obs_source: str = "robot"  # 黑盒 obs 来源: "robot"=真实硬件 | "mock"=随机假obs | "dataset"=从数据集加载

    # MockEnv configuration (only used when env is MockEnv)
    mock_image_size: int = 224  # H=W of generated random camera images (ALOHA default)

    # Training loop
    max_env_steps: int = 100_000

    # Checkpoints
    rl_token_checkpoint: str = ""
    vla_checkpoint_dir: str = ""
    vla_config_name: str = "pi05_aloha"
    resume_checkpoint: str = ""  # Path to Stage 2 checkpoint to resume training from
    warmup_buffer: str = ""  # Path to a standalone warmup buffer .pt file (skips warmup if provided)
    save_dir: str = "checkpoints/stage2_ac_online"
    run_name: str = ""  # Subdirectory name for this run (auto-generated if empty)
    save_every: int = 50
    log_every: int = 1  # wandb logging interval (steps)
    print_every: int = 100  # stdout logging interval (steps)

    # wandb
    wandb_project: str = "rlt-openpi"
    wandb_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.run_name:
            self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        if isinstance(self.dry_run, str):
            self.dry_run = self.dry_run.lower() in ("true", "1", "yes")

    @property
    def state_dim(self) -> int:
        """RL state dimension: z_rl (embedding_dim) + s^p (action_dim)."""
        return self.embedding_dim + self.action_dim

    @property
    def action_chunk_dim(self) -> int:
        """Flattened action chunk dimension: C * d."""
        return self.chunk_length * self.action_dim
