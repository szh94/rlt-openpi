# Stage 2 指南

## 一、总览

### 1.1 前置条件

| # | 前置项 | 路径/说明 | 状态 |
|---|--------|----------|------|
| 1 | Stage 1 checkpoint | `checkpoints/rl_token/<run>/step_xxxx.pt` | **需要用户提供** |
| 2 | VLA pretrained 权重 | `~/.cache/openpi/.../model.safetensors` | **需要下载** |

**Stage 1 checkpoint 内容**（冻结 VLA 策略 `vla_finetune_alpha=0`）：

```
model: RLTokenModel 权重                              ← 加载到 rl_token_model
config: RLTokenTrainConfig                            ← 用于重建模型结构
optimizer / scheduler / step                          ← 不在 Stage 2 使用
vla_model: 无（因为 vla_finetune_alpha=0）              ← VLA 直接用原始权重
```

> Stage 1 使用冻结 VLA，checkpoint 中**只有** RLTokenModel 权重，**没有** `vla_model`。Stage 2 的 VLA 直接从原始 pretrained checkpoint 加载。

### 1.2 核心流程概要

1. 加载 Stage 1 的冻结 `RLTokenModel`（只用于提取 `z_rl`，不再训练）
2. 加载冻结的 VLA（提供参考动作 `a_tilde` 和 embedding）
3. 创建 Actor（MLP + 残差）和 Twin Q-Critic（TD3 框架）
4. 在真实/仿真环境中交互，收集 transition 到 Replay Buffer
5. 用 TD3 + BC Regularizer 更新 Actor 和 Critic

### 1.3 论文 Algorithm 1 伪代码

```
Algorithm 1: RLT — Train RL Actor and Critic (Stage 2)
═══════════════════════════════════════════════════════

输入: 冻结 VLA π_vla, 冻结 RL Token Encoder E_φ, 环境 env
参数: C, G, β, τ, γ, σ_target, c_target

初始化:
    Actor μ_ψ        (最后一层零初始化, 起始 = VLA 参考)
    Critic Q_θ1, Q_θ2  (双 Q 网络)
    Target Q_θ1', Q_θ2' (deepcopy Q, 冻结)
    Replay Buffer B   (容量 N)

1.  Warmup Phase:
    用 VLA 参考动作 a_tilde 收集若干 transition 填充 B

2.  for each episode do:
        # ─── 数据收集 ───
        环境 reset → obs
        repeat:
            z, mask = VLA.extract_embeddings(obs)
            z_rl = E_φ.encode(z, mask)                 # RL token
            a_tilde = VLA.sample_reference(obs)[:C]      # VLA 参考 chunk
            x = concat(z_rl, s_p)                        # RL 状态

            if human_intervention:
                a = human_action  +  存入 B
            else:
                a = μ_ψ(x, a_tilde) + N(0, σ²)          # 探索
                执行 a, 获得 rewards, next_obs
                存入 B: (x, a, a_tilde, r_{0..C-1}, next_x, done)

            obs = next_obs
        until done

        # ─── 梯度更新 (G 次) ───
        for g = 1 to G do:
            采样 batch {x, a, a_tilde, r, next_x, done} ~ B

            # Critic 更新
            chunk_return = Σ_{k=0}^{C-1} γ^k · r_k
            a_target = μ_ψ(next_x, a_tilde') + clip(N(0, σ_target²), ±c_target)
            y = chunk_return + γ^C · (1-done) · min(Q_θ1', Q_θ2')(next_x, a_target)
            L_critic = MSE(Q_θ1(x, a), y) + MSE(Q_θ2(x, a), y)
            梯度更新 Q_θ1, Q_θ2

            # Actor 更新 (延迟, critic_updates_per_actor 次 critic 才更新一次)
            if g % 2 == 0:
                a_actor = μ_ψ(x, a_tilde_masked)         # ref_dropout = 50%
                L_actor = -Q_min(x, a_actor).mean() + β · MSE(a_actor, a_tilde)
                梯度更新 μ_ψ

            # Polyak 目标网络更新
            θ1' = (1-τ)·θ1' + τ·θ1
            θ2' = (1-τ)·θ2' + τ·θ2

        end for
    end for
```

**算法符号 → 代码参数映射**：

| 伪代码符号 | 代码实现 | 对应 config 参数 | 默认值 |
|-----------|---------|-----------------|--------|
| C | `chunk_length` | `chunk_length` | 10 |
| G | `utd_ratio` | `utd_ratio` | 5 |
| β | `bc_regularizer_beta` | `bc_regularizer_beta` | 0.5 |
| τ | `tau` | `tau` | 0.005 |
| γ | `gamma` | `gamma` | 0.99 |
| σ (探索噪声) | `actor_noise_sigma` | `actor_noise_sigma` | 0.1 |
| σ_target (TD3 平滑噪声) | `target_noise_sigma` | `target_noise_sigma` | 0.2 |
| c_target (噪声裁剪) | `target_noise_clip` | `target_noise_clip` | 0.5 |
| ref_dropout | `ref_action_dropout` | `ref_action_dropout` | 0.5 |
| critic_updates_per_actor | `critic_updates_per_actor` | `critic_updates_per_actor` | 2 |
| a_tilde_masked | `Actor._apply_ref_dropout()` | — | — |
| Q_min | `TwinQCritic.q_min()` | — | — |
| chunk_return | `compute_td_target()` | — | — |

### 1.4 运行命令

**正式训练（真实机器人 / 仿真）**：

```bash
python scripts/train_online_rl.py \
    --rl-token-checkpoint checkpoints/rl_token/run_xxx/step_5000.pt \
    --vla-checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors \
    --vla-config-name pi05_droid_finetune \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --task-prompt "pick up the pen" \
    --max-env-steps 100000
```

**离线测试（MockEnv，无需硬件）**：

```bash
python scripts/train_online_rl_offline.py \
    --rl-token-checkpoint checkpoints/rl_token/run_xxx/step_5000.pt \
    --vla-checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors \
    --max-env-steps 5000
```

> 两个脚本共用完全相同的 `OnlineRLTrainer` 和 `trainer.train()`，唯一区别是 `env` 对象不同（`make_env()` vs `MockEnv()`）。

**关键参数简表**：

| 参数 | 必填 | 说明 |
|------|:--:|------|
| `--rl-token-checkpoint` | 是 | Stage 1 训练产出的 `.pt` 文件路径 |
| `--vla-checkpoint-dir` | 是 | 原始 VLA pretrained 权重路径 |
| `--vla-config-name` | 否 | VLA 配置名，默认 `pi05_droid_finetune` |
| `--env-factory` | 是 | 环境工厂函数路径（真实机器人或仿真） |
| `--task-prompt` | 否 | 任务指令文字 |
| `--max-env-steps` | 否 | 总交互步数，默认 100000 |
| `--intervention-factory` | 否 | VR 干预工厂路径（可选） |
| `--batch-size` | 否 | 训练批大小，默认 256 |
| `--warmup-steps` | 否 | 纯 VLA warmup chunk 数，默认 1000 |
| `--resume-checkpoint` | 否 | 中断恢复的 Stage 2 checkpoint 路径 |

### 1.5 主程序流程框图

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          Stage 2 入口脚本（二选一）                                  │
│                                                                                   │
│  ┌─ train_online_rl.py ─────────────┐   ┌─ train_online_rl_offline.py ───────┐    │
│  │          main()                  │   │          main()                    │   │
│  │                                  │   │                                    │   │
│  │  ① tyro.cli(...)  解析参数       │   │  ① tyro.cli(...)  解析参数         │   │
│  │  ② VLAWrapper(...)               │   │  ② VLAWrapper(...)                 │   │
│  │  ② load_rl_token_model(...)      │   │  ② load_rl_token_model(...)        │   │
│  │  ③ OnlineRLTrainer.__init__()    │   │  ③ OnlineRLTrainer.__init__()      │   │
│  │     ├── Actor(...)                │   │     ├── Actor(...)                  │   │
│  │     ├── TwinQCritic(...)          │   │     ├── TwinQCritic(...)            │   │
│  │     ├── Adam / Adam               │   │     ├── Adam / Adam                 │   │
│  │     └── ReplayBuffer(...)         │   │     └── ReplayBuffer(...)           │   │
│  │                                  │   │                                    │   │
│  │  ④ make_env(env_factory)  ←─ 唯一│   │  ④ MockEnv(...)         ←─ 唯一   │   │
│  │     真实机器人 / 仿真环境   差异  │   │     假观测, 无机器人硬件     差异  │   │
│  │                                  │   │                                    │   │
│  │  ⑤ trainer.train(env, ...) ──┐   │   │  ⑤ trainer.train(env, ...) ──┐    │   │
│  └──────────────────────────────│───┘   └──────────────────────────────│────┘   │
│                                 │                                       │        │
│                                 ▼                                       ▼        │
│                    ┌────────────────────────────────────────────┐                │
│                    │   完全相同的 OnlineRLTrainer.train()         │                │
│                    │   src/rlt_openpi/training/online_rl_trainer │                │
│                    │   唯一区别: env 类型不同 (real/sim vs mock)   │                │
│                    └────────────────────┬───────────────────────┘                │
│                                         │                                         │
└─────────────────────────────────────────│─────────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────│─────────────────────────────────────────┐
│  OnlineRLTrainer.train()                ▼                                          │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Phase 1: Warmup（预热阶段）                                           │    │
│  │                                                                       │    │
│  │  buffer已有数据? ──Yes── 跳过                                          │    │
│  │       │ No                                                            │    │
│  │       ▼                                                               │    │
│  │  指定了warmup_buffer? ──Yes── _load_warmup_buffer() 加载                │    │
│  │       │ No                                                            │    │
│  │       ▼                                                               │    │
│  │  for i in range(warmup_steps):   默认 1000 个 chunk                    │    │
│  │    ├── _get_warmup_action(obs)     纯 VLA 参考动作（无 Actor）          │    │
│  │    ├── _extract_rl_state(obs)      提取 x = cat(z_rl, s^p)             │    │
│  │    ├── env.step(action_chunk)      执行 C=10 步                        │    │
│  │    └── replay_buffer.add(...)      存储 transition                     │    │
│  │  └── _save_warmup_buffer()       保存 warmup_buffer.pt                 │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Phase 2: Online RL Loop（在线强化学习循环）                            │    │
│  │                                                                       │    │
│  │  while total_env_steps < max_env_steps:   ←── 主循环                   │    │
│  │    │                                                                  │    │
│  │    │  actor.eval()                   切换到评估模式                    │    │
│  │    │                                                                  │    │
│  │    ├── worker.collect_episode()     采集一个完整 episode                │    │
│  │    │   ┌──────────────────────────────────────────────────────────┐   │    │
│  │    │   │  for each chunk in episode:                              │   │    │
│  │    │   │    ├── _extract_rl_state(obs)     z_rl → x                │   │    │
│  │    │   │    ├── check_intervention()       人类接管?              │   │    │
│  │    │   │    │   ├── Yes → 用人类动作                               │   │    │
│  │    │   │    │   └── No  → _get_actor_action(x, a_tilde)           │   │    │
│  │    │   │    ├── env.step(action_chunk)      执行 C 步              │   │    │
│  │    │   │    │   └── 每步: HumanReward.check()  键盘标注奖励        │   │    │
│  │    │   │    └── replay_buffer.add(x, a, a_tilde, r, next_x, done) │   │    │
│  │    │   │  until done                                              │   │    │
│  │    │   └──────────────────────────────────────────────────────────┘   │    │
│  │    │                                                                  │    │
│  │    ├── 统计: total_reward, success, chunks, steps, interventions       │    │
│  │    │                                                                  │    │
│  │    ├── for g in range(utd_ratio):   G=5 次 TD3 更新                    │    │
│  │    │   ┌── _update_step(g) ───────────────────────────────────────┐   │    │
│  │    │   │                                                           │   │    │
│  │    │   │  ① batch ← ReplayBuffer.sample(256)                       │   │    │
│  │    │   │                                                           │   │    │
│  │    │   │  ② Critic 更新（每次）:                                    │   │    │
│  │    │   │     td_target = Σγᵏ·rₖ + γᶜ·(1-done)·min_Q_target(x',a') │   │    │
│  │    │   │     L_critic   = MSE(q1, target) + MSE(q2, target)        │   │    │
│  │    │   │     critic_optimizer.step()                                │   │    │
│  │    │   │                                                           │   │    │
│  │    │   │  ③ Actor 更新（延迟, 每 2 次 critic 才更新 1 次）:         │   │    │
│  │    │   │     a_actor = actor(x, a_tilde_masked)   ← ref_dropout    │   │    │
│  │    │   │     L_actor = -Q_min(x, a_actor).mean()                   │   │    │
│  │    │   │               + β·MSE(a_actor, a_tilde)   ← BC regularizer│   │    │
│  │    │   │     actor_optimizer.step()                                 │   │    │
│  │    │   │                                                           │   │    │
│  │    │   │  ④ Polyak 目标网络更新（每次都更新）:                       │   │    │
│  │    │   │     θ_target = (1-τ)·θ_target + τ·θ_online  (τ=0.005)     │   │    │
│  │    │   └───────────────────────────────────────────────────────────┘   │    │
│  │    │                                                                  │    │
│  │    ├── log_fn(metrics)              wandb / stdout 日志                │    │
│  │    │                                                                  │    │
│  │    └── if episode % save_every == 0:  每 50 episode                   │    │
│  │          save() → online_rl_ep{N}.pt                                   │    │
│  │                                                                       │    │
│  │  total_env_steps >= max_env_steps → 退出循环                           │    │
│  │  save()  最终保存                                                       │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

**关键函数调用链**：

```
train_online_rl.py                     train_online_rl_offline.py
  │                                      │
  ├── VLAWrapper(...)                    ├── VLAWrapper(...)                       ← 相同
  ├── load_rl_token_model(...)           ├── load_rl_token_model(...)              ← 相同
  ├── OnlineRLTrainer.__init__(...)      ├── OnlineRLTrainer.__init__(...)          ← 相同
  │     ├── Actor(...)                   │     ├── Actor(...)
  │     ├── TwinQCritic(...)             │     ├── TwinQCritic(...)
  │     ├── Adam / Adam                  │     ├── Adam / Adam
  │     └── ReplayBuffer(...)            │     └── ReplayBuffer(...)
  │                                      │
  ├── make_env(env_factory)              ├── MockEnv(...)                          ← 唯一区别
  │   真实机器人 / 仿真                    │   假观测, 无硬件依赖
  │                                      │
  └── trainer.train(env, ...)            └── trainer.train(env, ...)               ← 相同
        │                                      │
        └──────────────────┬───────────────────┘
                           │ 都进入同一方法: online_rl_trainer.py:186
                           ▼
  trainer.train(env, intervention_mgr, log_fn)
        │
        ├── Phase 1: Warmup (line 213-243)
        │     ├── worker._get_warmup_action(obs)   ← VLA-only
        │     ├── worker._extract_rl_state(obs)
        │     ├── env.step(action_chunk)
        │     └── replay_buffer.add(...)
        │
        └── Phase 2: Online RL Loop (line 250-315)
              while total_env_steps < max_env_steps:
                ├── actor.eval()
                ├── worker.collect_episode()        ← 采集一个 episode
                │     └── for each chunk:
                │           ├── worker._extract_rl_state(obs)
                │           ├── intervention_mgr.check_intervention()
                │           ├── worker._get_actor_action(x, a_tilde)
                │           ├── env.step(action_chunk)
                │           └── replay_buffer.add(...)
                │
                ├── for g in range(utd_ratio):     ← TD3 更新 (G=5)
                │     └── self._update_step(g)
                │           ├── compute_td_target(...)
                │           ├── critic_loss(...) → critic_optimizer.step()
                │           ├── actor_loss(...) → actor_optimizer.step()  (delayed)
                │           └── critic.update_targets(tau)
                │
                ├── log_fn(metrics)
                └── if episode % save_every == 0: save()
```

---

## 二、组件详解

### 2.1 冻结模型加载

#### 2.1.1 RLTokenModel

**涉及文件**：`src/rlt_openpi/utils/checkpoint.py` → `load_rl_token_model()`

**代码**（`train_online_rl.py:46`）：

```python
rl_token_model = load_rl_token_model(config.rl_token_checkpoint, device="cuda")
```

**内部流程**（`checkpoint.py:14-44`）：

1. 从 Stage 1 的 `.pt` 文件读取 `config` 和 `model` 权重
2. 根据 config 中的结构参数重建 `RLTokenModel`（`embedding_dim=2048`, `encoder_layers/heads=2/8`, `decoder_layers/heads=2/8`）
3. 加载权重，移到 GPU，设为 eval 模式，冻结所有参数

**对应 config 参数**：`rl_token_checkpoint`

#### 2.1.2 VLA

**涉及文件**：`src/rlt_openpi/vla/vla_wrapper.py` → `VLAWrapper`

**代码**（`train_online_rl.py:38-56`）：

```python
vla = VLAWrapper(
    checkpoint_path=config.vla_checkpoint_dir,
    config_name=config.vla_config_name,
    device="cuda",
)
```

**两种 Stage 1 策略的差异**：

| Stage 1 策略 | `vla_finetune_alpha` | checkpoint 中有 `vla_model`？ | Stage 2 VLA 来源 |
|-------------|----------------------|------------------------------|-------------------|
| 冻结 VLA（你的情况） | `0` | 否 | 原始 pretrained 权重 |
| Joint Training | `>0`（如 1.0） | 是 | Stage 1 微调后的权重 |

**冻结 VLA 路径**（`train_online_rl.py:50-56`）：

```python
stage1_ckpt = torch.load(config.rl_token_checkpoint, ...)
if "vla_model" in stage1_ckpt:              # ← False，跳过
    vla.extractor.pi0.load_state_dict(...)
else:
    log.warning("No fine-tuned VLA weights... using base VLA")  # ← 走这里
```

**VLA 在 Stage 2 中提供的方法**：

| 方法 | 作用 | 调用位置 |
|------|------|----------|
| `extract_embeddings(obs)` | 提取 prefix embedding `z` [B, M, D] | `RolloutWorker._extract_rl_state()` |
| `get_rl_chunk_reference(obs, C)` | 取前 C 步 VLA 动作 `a_tilde` [B, C, d] | `RolloutWorker._extract_rl_state()` / `_get_warmup_action()` |
| `preprocess_obs(obs)` | 原始 obs → OpenPI transform → Observation | `RolloutWorker._obs_to_vla_input()` |

**对应 config 参数**：`vla_checkpoint_dir`、`vla_config_name`

### 2.2 可训练网络

**涉及文件**：

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `src/rlt_openpi/models/networks.py` | `MLP` | 通用 MLP 基类 |
| `src/rlt_openpi/models/actor.py` | `Actor` | 策略网络（残差 + VLA 参考） |
| `src/rlt_openpi/models/critic.py` | `TwinQCritic`, `QNetwork` | 双 Q 值网络 |

**初始化代码**（`online_rl_trainer.py:62-78`）：

```python
self.actor = Actor(
    state_dim=embedding_dim + action_dim,       # 2048 + 8 = 2056
    action_chunk_dim=chunk_length * action_dim, # 10 * 8 = 80
    hidden_dim=256, num_hidden_layers=2,
    sigma=0.1, ref_dropout=0.5,
)
self.critic = TwinQCritic(
    state_dim=embedding_dim + action_dim,
    action_chunk_dim=chunk_length * action_dim,
    hidden_dim=256, num_hidden_layers=2,
)
```

#### 2.2.1 MLP 基类

Actor 和 Critic 都使用 `MLP`（`networks.py:7`）作为骨干：

```
输入 x
  → LayerNorm(input_dim)          ← 输入归一化
  → Linear(input_dim, hidden)     ← 全连接层 1
  → ReLU()                        ← 激活
  → Linear(hidden, hidden)        ← 重复 num_hidden_layers 次
  → ReLU()
  → Linear(hidden, output_dim)    ← 输出层，无激活函数
输出
```

| 设计点 | 说明 |
|--------|------|
| **输入 LayerNorm** | 非标准 MLP 设计。输入中 z_rl（~2048维）和 s^p（~8维）尺度差异大，LayerNorm 消除此影响 |
| **最后一层无激活** | 回归任务标准做法。Actor 输出残差、Critic 输出 Q 值，都需要预测任意实数 |
| **无 Dropout / BatchNorm** | 极简设计，当前场景不需要额外正则化 |

以 Actor 为例（`input_dim=2136, hidden_dim=256, output_dim=80`）：

```
[B,2136] → LayerNorm(2136) → Linear(2136,256) → ReLU → Linear(256,256) → ReLU → Linear(256,80) → [B,80]
```

纯顺序结构，用 `nn.Sequential(*layers)` 打包。

#### 2.2.2 Actor

```
x [B, 2056]    a_tilde [B, 80]
     │                │
     │     ┌──────────┘
     │     │ _apply_ref_dropout: 训练时 50% 样本置零, 评估时原样
     │     ▼
     │  a_tilde_input [B, 80]
     │     │
     ├─────┤  torch.cat([x, a_tilde_input], dim=-1)
     │     ▼
     │  [B, 2136]  →  MLP  →  residual [B, 80]
     │                         │
     │     ┌───────────────────┘
     ▼     ▼
  mu = a_tilde + residual   [B, 80]
     │
     ▼
  training?  →  Yes  →  mu + N(0, σ²)  →  clamp(-1, 1)  →  [B, 80]
                No  →  clamp(-1, 1)  →  [B, 80]
```

**关键设计**：

| 设计 | 说明 |
|------|------|
| **最后一层零初始化** | `Linear(256,80)` 的 weight/bias 全置 0，训练起点 Actor = 纯 VLA 参考 |
| **ref_dropout** | 50% 概率丢参考动作，迫使 Actor 独立决策 |
| **探索噪声** | 训练时加 N(0, 0.1²)，评估时不加 |
| **clamp(-1,1)** | 动作空间已归一化，确保输出合法 |

#### 2.2.3 TwinQCritic

- 两个独立的 `QNetwork`（Q1, Q2），结构相同，各含一个 MLP
- 输入 `cat(x, a)` [B, 2136] → 输出标量 Q 值 [B, 1]
- `q_min(x, a) = min(Q1(x, a), Q2(x, a))`，取较小值防止过估计
- 维护 target 网络（deepcopy 创建，不参与梯度更新），Polyak 软更新

### 2.3 环境与交互

**涉及文件**：

| 文件 | 关键类/函数 | 作用 |
|------|-------------|------|
| `src/rlt_openpi/rollout/rollout_worker.py` | `RolloutWorker` | 编排 VLA → 编码 → Actor → 环境 → Buffer 交互 |
| `src/rlt_openpi/rollout/robot_env.py` | `RobotEnv` | 真实机器人环境封装 |
| `src/rlt_openpi/rollout/sim_env.py` | `SimEnv` | 仿真环境封装 |
| `src/rlt_openpi/rollout/mock_env.py` | `MockEnv` | 假观测环境（离线测试用） |
| `src/rlt_openpi/rollout/reward.py` | `HumanReward` | 键盘奖励标注 |
| `src/rlt_openpi/rollout/intervention.py` | `InterventionManager` | 人工干预基类 |
| `src/rlt_openpi/envs/franka/intervention.py` | `VRInterventionManager` | VR 手柄干预 |
| `src/rlt_openpi/training/replay_buffer.py` | `ReplayBuffer` | 经验回放缓存 |

#### 2.3.1 Transition 结构

每次存储到 ReplayBuffer 的 transition 包含 6 个字段：

| 字段 | 形状 | 说明 |
|------|------|------|
| `x` | [state_dim] | 当前 RL 状态 = cat(z_rl, s^p) |
| `a` | [C\*d] | 执行的 flat 动作 chunk |
| `a_tilde` | [C\*d] | VLA 参考动作 chunk |
| `rewards` | [C] | C 个 per-step 奖励 |
| `next_x` | [state_dim] | 下一步 RL 状态 |
| `done` | [1] | 是否终止 |

#### 2.3.2 RolloutWorker 核心函数

**`_extract_rl_state(obs)`**（`rollout_worker.py:98-128`）：

```python
# 1. VLA 预处理
vla_input = self.vla.preprocess_obs(obs)
# 2. 提取 embedding → z_rl
z, pad_mask = self.vla.extract_embeddings(vla_input)
z_rl = self.rl_token_model.encode(z, pad_mask)  # [1, D]
# 3. 获取 VLA 参考动作
a_tilde = self.vla.get_rl_chunk_reference(vla_input, C)  # [1, C, d]
# 4. 本体状态
s_p = vla_input.state[:, :action_dim]  # [1, d]
# 5. 拼接
x = cat(z_rl, s_p)  # [1, state_dim]
```

**`_get_warmup_action(obs)`**（`rollout_worker.py:130-139`）：纯 VLA 参考动作，不经 Actor。

**`_get_actor_action(x, a_tilde_flat)`**（`rollout_worker.py:141-156`）：Actor 推理，输出动作 chunk。

**`collect_episode()`**（`rollout_worker.py:205-280`）：完整 episode 循环：

```
for each chunk:
  1. _extract_rl_state(obs) → x, a_tilde_flat
  2. check_intervention()
     - 有干预 → 用人类动作
     - 无干预 → _get_actor_action(x, a_tilde_flat)
  3. env.step(action_chunk)  执行 C 步
  4. _extract_rl_state(next_obs) → next_x
  5. replay_buffer.add(x, a, a_tilde, rewards, next_x, done)
until done
```

### 2.4 模型文件完整度检查

| # | 组件 | 文件路径 | 状态 |
|---|------|---------|------|
| 1 | `RLTokenModel`（encoder+decoder） | `src/rlt_openpi/models/rl_token.py` | 已有 |
| 2 | `RLTokenModel` 加载函数 | `src/rlt_openpi/utils/checkpoint.py` | 已有 |
| 3 | `VLAWrapper`（VLA 封装） | `src/rlt_openpi/vla/vla_wrapper.py` | 已有 |
| 4 | `EmbeddingExtractor`（提取 prefix embedding） | `src/rlt_openpi/vla/embedding_extractor.py` | 已有 |
| 5 | `Actor`（策略网络） | `src/rlt_openpi/models/actor.py` | 已有 |
| 6 | `TwinQCritic`（双 Q 网络） | `src/rlt_openpi/models/critic.py` | 已有 |
| 7 | `MLP`（通用网络组件） | `src/rlt_openpi/models/networks.py` | 已有 |
| 8 | `OnlineRLTrainer`（TD3 训练循环） | `src/rlt_openpi/training/online_rl_trainer.py` | 已有 |
| 9 | `ReplayBuffer`（经验回放） | `src/rlt_openpi/training/replay_buffer.py` | 已有 |
| 10 | TD3 工具函数（target/loss） | `src/rlt_openpi/training/td3_utils.py` | 已有 |
| 11 | `OnlineRLTrainConfig`（参数配置） | `src/rlt_openpi/training/config.py` | 已有 |
| 12 | `RolloutWorker`（环境交互） | `src/rlt_openpi/rollout/rollout_worker.py` | 已有 |
| 13 | `RobotEnv`（真实机器人） | `src/rlt_openpi/rollout/robot_env.py` | 已有 |
| 14 | `SimEnv`（仿真环境） | `src/rlt_openpi/rollout/sim_env.py` | 已有 |
| 15 | `HumanReward`（键盘奖励标注） | `src/rlt_openpi/rollout/reward.py` | 已有 |
| 16 | `InterventionManager`（人工干预） | `src/rlt_openpi/rollout/intervention.py` | 已有 |
| 17 | 环境工厂（Franka / 仿真） | `src/rlt_openpi/envs/franka/env_factory.py` | 已有 |
| 18 | VR 干预（可选） | `src/rlt_openpi/envs/franka/intervention.py` | 已有 |
| 19 | Stage 2 入口脚本 | `scripts/train_online_rl.py` | 已有 |
| 20 | Stage 2 离线测试脚本 | `scripts/train_online_rl_offline.py` | 已有 |

> Stage 2 所需的模型、训练、环境、交互代码均已完整定义，无需新增文件。

---

## 三、训练循环详解

训练主循环在 `OnlineRLTrainer.train()`（`online_rl_trainer.py:186`）中实现，分为两个 Phase。

### 3.1 Phase 1: Warmup（预热阶段）

**代码位置**：`online_rl_trainer.py:213-243`

**目的**：用纯 VLA 策略采集 transition 预热 replay buffer，使 Actor/Critic 一开始就有数据可训练。

**流程**：

```
buffer已有数据? ──Yes── 跳过 (从 checkpoint 恢复)
     │ No
     ▼
指定了warmup_buffer? ──Yes── _load_warmup_buffer() 加载预采集数据
     │ No
     ▼
for i in range(warmup_steps):   默认 1000 个 chunk
  ├── _get_warmup_action(obs)     纯 VLA 参考动作（无 Actor 参与）
  ├── _extract_rl_state(obs)      提取 x = cat(z_rl, s^p)
  ├── env.step(action_chunk)      执行 C=10 步
  └── replay_buffer.add(...)      存储 transition
_save_warmup_buffer()            保存 warmup_buffer.pt，下次可跳过
```

**关键点**：Actor 和 Critic **完全不参与** warmup，只是用 VLA 参考策略收集数据。

### 3.2 Phase 2: Online RL Loop（在线强化学习循环）

**代码位置**：`online_rl_trainer.py:250-315`

```
while total_env_steps < max_env_steps:

  ├── actor.eval()
  │
  ├── collect_episode()                      采集一个完整 episode
  │     └── 详见 §2.3.2 collect_episode() 流程
  │
  ├── 统计: total_reward, success, chunks, steps, interventions
  │
  ├── for g in range(utd_ratio):             G=5 次 TD3 更新
  │     └── _update_step(g)                  详见 §3.2.1
  │
  ├── log_fn(metrics)                        wandb / stdout 日志
  │
  └── if episode % save_every == 0:         每 50 episode
        save() → online_rl_ep{N}.pt

total_env_steps >= max_env_steps → 最终 save() → 训练结束
```

#### 3.2.1 TD3 更新 `_update_step()`

**代码位置**：`online_rl_trainer.py:115-184`

**涉及文件**：`src/rlt_openpi/training/td3_utils.py` → `compute_td_target()`, `critic_loss()`, `actor_loss()`

每个 episode 结束后运行 G=5 次：

```
① batch ← ReplayBuffer.sample(256)                    随机采样

② Critic 更新（每次都做）:
   td_target = Σᵏγᵏ·rₖ + γᶜ·(1-done)·min_Q_target(next_x, a')
   L_critic = MSE(q1, td_target) + MSE(q2, td_target)
   critic_optimizer.step()

③ Actor 更新（延迟, 每 2 次 critic 更新才做 1 次）:
   a_actor = actor(x, a_tilde_masked)                 ref_dropout=50%
   L_actor = -Q_min(x, a_actor).mean()                最大化 Q 值
             + β·MSE(a_actor, a_tilde)                BC 正则项 (β=0.5)
   actor_optimizer.step()

④ Polyak 目标网络更新（每次都做）:
   θ_target = (1-τ)·θ_target + τ·θ_online             τ=0.005
```

**损失函数总结**：

| 损失 | 公式 | 含义 |
|------|------|------|
| TD Target | `y = Σγᵏ·rₖ + γᶜ·(1-done)·min Q_target(x', a')` | 折扣 chunk 回报 + 自举 |
| Critic Loss | `MSE(Q₁, y) + MSE(Q₂, y)` | 双 Q 逼近 TD target |
| Actor Loss | `-Q_min(x, a).mean() + β·MSE(a, a_tilde)` | 最大化 Q 值 + 贴近 VLA 参考 |

### 3.3 人工介入节点

> **注意**：以下仅在真实机器人（`RobotEnv`）中需要；仿真环境（`SimEnv`）自动返回奖励。VR 干预仅在配置 `--intervention-factory` 时启用。

Stage 2 中有 **3 处需要人为介入**：

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 2 训练主循环                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Warmup Phase   │
                    │  (VLA-only,      │
                    │   无需人为介入)    │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │      Episode 开始        │
              └──────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │   👤 人工介入 1:          │
              │   env.reset()            │
              │   摆放场景 → 按 Enter      │
              │   ("Set up the scene")   │
              └──────────┬───────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│                   Episode 循环 (每个 chunk)                  │
│                                                            │
│   ┌─────────────────────┐                                  │
│   │ _extract_rl_state() │  VLA → z_rl + a_tilde → x       │
│   └──────────┬──────────┘                                  │
│              │                                              │
│              ▼                                              │
│   ┌──────────────────────────┐                             │
│   │  👤 人工介入 3 (可选):    │                             │
│   │  intervention_mgr        │                             │
│   │  .check_intervention()   │                             │
│   │  VR手柄按下 → 人类操作    │                             │
│   │  松开 → 恢复Actor控制    │                             │
│   └───┬──────────┬───────────┘                             │
│       │          │                                          │
│   人类操作    Actor推理                                      │
│       │          │                                          │
│       └────┬─────┘                                          │
│            ▼                                                │
│   ┌──────────────────────────┐                             │
│   │  env.step(action_chunk)  │  执行 C 步                   │
│   │  ┌────────────────────┐  │                             │
│   │  │ 👤 人工介入 2:      │  │  每步检测键盘:              │
│   │  │ HumanReward.check() │  │  S=成功 F=失败 P=进度      │
│   │  └────────────────────┘  │                             │
│   └──────────┬───────────────┘                             │
│              │                                              │
│              ▼                                              │
│   ┌──────────────────────────┐                             │
│   │ ReplayBuffer.add()       │  存储 transition             │
│   └──────────┬───────────────┘                             │
│              │                                              │
│              ▼                                              │
│         done? ── No ──→ 继续下一个 chunk                    │
│              │                                              │
└──────────────┼──────────────────────────────────────────────┘
               │ Yes
               ▼
┌──────────────────────────────────────────────────────────┐
│              TD3 更新 (G 次, 无需人为介入)                 │
│  for g in 1..G:                                          │
│    batch ← ReplayBuffer.sample()                         │
│    y ← chunk_return + γ^C·(1-done)·min_Q_target(x', a')  │
│    update Q_θ1, Q_θ2    (Critic)                         │
│    update μ_ψ           (Actor, 延迟)                     │
│    update Q_target      (Polyak τ)                       │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  env_steps < max ?     │
              │  Yes → 下一个 Episode  │
              │  No  → 训练结束        │
              └────────────────────────┘
```

**三处人为介入汇总**：

| # | 节点 | 触发条件 | 操作 |
|---|------|---------|------|
| 👤1 | Episode 开始 | 每个 episode 的 `reset()` | 摆放场景 + 按 **Enter** |
| 👤2 | 单步执行中 | 每个 action step 之后 | 按 **S**(成功) / **F**(失败) / **P**(进度) 标注奖励 |
| 👤3 | Chunk 边界 | VR 手柄按钮按下（可选） | 人类接管控制，松开后恢复 Actor |

---

## 四、参考

### 4.1 完整参数表（OnlineRLTrainConfig）

| 分类 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| **架构** | `embedding_dim` | 2048 | 与 Stage 1 保持一致 |
| **动作空间** | `action_dim` | 8 | 单步动作维度 |
| | `chunk_length` (C) | 10 | 每 chunk 步数 |
| | `vla_action_horizon` (H) | 16 | VLA 输出的动作 horizon |
| **Actor-Critic** | `mlp_hidden_dim` | 256 | MLP 隐藏层宽度 |
| | `mlp_num_hidden_layers` | 2 | MLP 深度 |
| | `actor_noise_sigma` | 0.1 | 探索噪声标准差 |
| | `ref_action_dropout` | 0.5 | 参考动作 dropout 比例 |
| **RL 超参** | `gamma` | 0.99 | 折扣因子 |
| | `tau` | 0.005 | 目标网络软更新系数 |
| | `utd_ratio` (G) | 5 | 每 episode 的梯度更新次数 |
| | `bc_regularizer_beta` | 0.5 | BC 正则化权重 |
| | `critic_updates_per_actor` | 2 | Actor 延迟更新频率 |
| | `target_noise_sigma` | 0.2 | TD3 目标平滑噪声 |
| | `target_noise_clip` | 0.5 | 目标噪声裁剪范围 |
| **学习率** | `actor_lr` | 3e-4 | Actor 学习率 |
| | `critic_lr` | 3e-4 | Critic 学习率 |
| **Replay Buffer** | `buffer_capacity` | 100000 | 最大 transition 数 |
| | `batch_size` | 256 | 训练 batch size |
| | `warmup_steps` | 1000 | VLA 纯 warmup chunk 数 |
| **环境** | `env_factory` | "" | 环境工厂函数路径 |
| | `intervention_factory` | "" | 人工干预工厂函数路径 |
| | `task_prompt` | "" | 任务指令 |
| | `max_episode_chunks` | 150 | 每 episode 最大 chunk 数 |
| **训练循环** | `max_env_steps` | 100000 | 总环境交互步数 |
| **Checkpoint** | `rl_token_checkpoint` | "" | Stage 1 checkpoint 路径（必填） |
| | `vla_checkpoint_dir` | "" | VLA 模型路径 |
| | `save_dir` | "checkpoints/online_rl" | 保存目录 |
| | `save_every` | 50 | 每 N 个 episode 保存一次 |

### 4.2 文件清单

#### 入口脚本

| 文件 | 作用 |
|------|------|
| `scripts/train_online_rl.py` | Stage 2 正式训练入口 |
| `scripts/train_online_rl_offline.py` | Stage 2 离线测试入口（MockEnv） |

#### 训练核心

| 文件 | 关键类/函数 | 作用 |
|------|-------------|------|
| `src/rlt_openpi/training/config.py` | `OnlineRLTrainConfig` | Stage 2 参数配置 |
| `src/rlt_openpi/training/online_rl_trainer.py` | `OnlineRLTrainer` | TD3 训练循环、checkpoint |
| `src/rlt_openpi/training/td3_utils.py` | `compute_td_target()`, `critic_loss()`, `actor_loss()` | TD3 损失计算 |
| `src/rlt_openpi/training/replay_buffer.py` | `ReplayBuffer` | 经验回放缓存 |

#### 模型

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `src/rlt_openpi/models/actor.py` | `Actor` | 策略网络（残差 + VLA 参考） |
| `src/rlt_openpi/models/critic.py` | `TwinQCritic`, `QNetwork` | 双 Q 值网络 |
| `src/rlt_openpi/models/networks.py` | `MLP` | 通用 MLP 组件 |
| `src/rlt_openpi/models/rl_token.py` | `RLTokenModel` | Stage 1 训练好的 encoder（冻结） |

#### 环境与交互

| 文件 | 关键类/函数 | 作用 |
|------|-------------|------|
| `src/rlt_openpi/rollout/rollout_worker.py` | `RolloutWorker` | 环境交互、RL state 提取 |
| `src/rlt_openpi/rollout/robot_env.py` | `RobotEnv` | 真实机器人环境封装 |
| `src/rlt_openpi/rollout/sim_env.py` | `SimEnv` | 仿真环境封装 |
| `src/rlt_openpi/rollout/mock_env.py` | `MockEnv` | 假观测环境（离线测试） |
| `src/rlt_openpi/rollout/intervention.py` | `InterventionManager` | 人工干预基类 |
| `src/rlt_openpi/rollout/reward.py` | `HumanReward` | 键盘奖励标注 |
| `src/rlt_openpi/rollout/factory.py` | `make_env()`, `make_intervention()` | 动态工厂函数 |
| `src/rlt_openpi/envs/franka/env_factory.py` | `make_franka_env()` | Franka 机器人环境 |
| `src/rlt_openpi/envs/franka/intervention.py` | `VRInterventionManager` | VR 手柄干预 |

#### VLA 接口

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `src/rlt_openpi/vla/vla_wrapper.py` | `VLAWrapper` | VLA 调用封装（提取 embedding、参考动作） |
| `src/rlt_openpi/vla/embedding_extractor.py` | `EmbeddingExtractor` | Prefix embedding 提取 |

#### 工具

| 文件 | 关键函数 | 作用 |
|------|----------|------|
| `src/rlt_openpi/utils/checkpoint.py` | `load_rl_token_model()` | 加载 Stage 1 模型 |
| `src/rlt_openpi/utils/logging.py` | `Logger` | wandb + stdout 日志 |

### 4.3 数据流

#### 变量说明

| 符号 | 含义 | 来源 |
|------|------|------|
| `z_rl` | RL token，prefix embedding 压缩成的单向量 | VLA 提取 `z` → RLTokenModel.encode(z) |
| `s^p` | proprioceptive state = 关节位置 + 夹爪 | VLA 预处理后 `Observation.state` 取前 `action_dim` 维 |
| `a_tilde` | VLA 参考动作 chunk = H 步中取前 C 步展平 | `VLA.get_rl_chunk_reference(obs, C)` |
| `x` | RL 状态 = cat(z_rl, s^p) | 拼接 |
| `a` | Actor 输出的动作 chunk = a_tilde + 残差 + 噪声 | Actor 前向 |

#### 完整 shape 流

```
obs → VLA.extract_embeddings        → z        [B, M, 2048]
    → RLTokenModel.encode(z)        → z_rl     [B, 2048]
    → VLA.preprocess_obs → state    → s^p      [B, 8]
    → cat(z_rl, s^p)                → x        [B, 2056]

    → VLA.get_rl_chunk_ref(obs, C)  → a_tilde  [B, C*d]   = [B, 80]

Actor:
  cat(x, a_tilde)                    [B, 2136]
    → LayerNorm(2136)                [B, 2136]
    → Linear(2136, 256)              [B, 256]
    → ReLU                          [B, 256]
    → Linear(256, 256)              [B, 256]
    → ReLU                          [B, 256]
    → Linear(256, 80)               [B, 80]    ← residual (zero-init)
  a = a_tilde + residual + noise    [B, 80]    ← clamp(-1,1)

Critic (Twin, 单个 Q):
  cat(x, a)                          [B, 2136]
    → LayerNorm(2136)                [B, 2136]
    → Linear(2136, 256)              [B, 256]
    → ReLU                          [B, 256]
    → Linear(256, 256)              [B, 256]
    → ReLU                          [B, 256]
    → Linear(256, 1)                 [B, 1]
  Q_min = min(Q1, Q2)               [B, 1]

ReplayBuffer 存储:
  x [B, 2056], a [B, 80], a_tilde [B, 80], r [B, C], next_x [B, 2056], done [B, 1]

TD3 更新 (每 episode 后 G 次):
    batch = ReplayBuffer.sample(B=256)
    td_target = Σγᵏ·rₖ + γ^C·(1-done)·min_Q_target(next_x, a')
    critic_loss = MSE(q1, target) + MSE(q2, target)
    actor_loss = -Q_min(x, actor(x, a_tilde)).mean() + β·MSE(a, a_tilde)
```

#### Actor/Critic 参数量

| 组件 | 层 | shape | 参数量 |
|------|----|-------|--------|
| **Actor** | LayerNorm | 2136→2136 | 4,272 |
| | Linear 1 | 2136→256 | 547,072 |
| | Linear 2 | 256→256 | 65,792 |
| | Linear 3 | 256→80 | 20,560 |
| | **小计** | | **~637K** |
| **Q-Network (×2)** | LayerNorm | 2136→2136 | 4,272 |
| | Linear 1 | 2136→256 | 547,072 |
| | Linear 2 | 256→256 | 65,792 |
| | Linear 3 | 256→1 | 257 |
| | 单 Q 小计 | | ~617K |
| | **双 Q 合计** | | **~1.27M** |
| **AC 总计** | | | **~1.9M** |

### 4.4 Checkpoint 产物

**保存路径**：
```
{save_dir}/{run_name}/online_rl_ep{N}.pt
```

默认示例：`checkpoints/online_rl/run_20250615_143052/online_rl_ep50.pt`

**保存时机**：每 `save_every` 个 episode（默认 50）保存一次 + 训练结束最终保存。

**每个 `.pt` 文件包含的 key**：

| key | 内容 |
|-----|------|
| `actor` | Actor 网络 `state_dict`（可单独取出来部署推理） |
| `critic` | TwinQCritic 网络 `state_dict` |
| `actor_optimizer` | Actor 优化器状态（恢复训练用） |
| `critic_optimizer` | Critic 优化器状态（恢复训练用） |
| `replay_buffer` | ReplayBuffer 全部 transition 数据（恢复训练用） |
| `total_env_steps` | 总交互步数 |
| `total_updates` | 总梯度更新次数 |
| `total_episodes` | 总 episode 数 |
| `config` | `OnlineRLTrainConfig` 完整参数 |

**只取 Actor/Critic 权重用于推理**：

```python
ckpt = torch.load("checkpoints/online_rl/run_xxx/online_rl_ep200.pt", weights_only=False)
actor_state = ckpt["actor"]
critic_state = ckpt["critic"]
```
