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
| max_deviation | `max_deviation` | `max_deviation` | 0.3 |
| critic_updates_per_actor | `critic_updates_per_actor` | `critic_updates_per_actor` | 2 |
| a_tilde_masked | `Actor._apply_ref_dropout()` | — | — |
| Q_min | `TwinQCritic.q_min()` | — | — |
| chunk_return | `compute_td_target()` | — | — |

> **C=10 选取依据**：VLA 预测 H=50 步动作，C=10 将决策频率从 50 Hz 压缩约 5 倍到 ~5 Hz，在 VLA 推理开销和 RL 修正粒度之间折中：
> - C 太大 → RL 修正不够及时，VLA 错误无法快速纠正
> - C 太小 → 每个 chunk 都要跑一次 VLA 前向，推理开销过大
>
> 可通过 `--chunk-length` 调整。

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

#### 2.3.0 env 接口约定

`trainer.train()` 不对 `env` 做具体类型假设，只依赖鸭子类型的 **3 个方法 + 2 个属性**：

| 接口 | 类型 | 调用形式 | 方向 | 作用 |
|------|------|---------|------|------|
| `env.action_dim` | `int` | 属性读取 | — | 单步动作维度，用于切 s^p 宽度、reshape action_chunk |
| `env.chunk_length` | `int` | 属性读取 | — | 每 chunk 步数 C，用于取 a_tilde 前 C 步、step 循环次数 |
| `env.reset()` | 方法 | `→ dict[str, Any]` | **机器人→模型** | 复位，返回初始观测（图像+关节+指令） |
| `env.step(chunk)` | 方法 | `[C, d] → (obs, rewards, done, info)` | **模型→机器人** | 执行 C 步动作，返回新观测 + per-step 奖励 `[C]` |
| `env.step()` 返回值 | — | `rewards [C], done bool, info dict` | **机器人→模型** | 驱动的下一轮推理的输入 |

三种实现的差异：

| | `RobotEnv` | `SimEnv` | `MockEnv` |
|---|---|---|---|
| `reset()` | 机器人复位 + 等待 Enter 键 | `gym.env.reset()` | 返回随机假观测 |
| `step()` | 逐步发机器人指令 + `HumanReward` 键盘检测 | 逐步 `gym.env.step()` | 动作被丢弃，返回零奖励 |
| `obs` 格式 | DROID schema（3 相机 + joint state） | `{"state": np.array}` | DROID schema（随机图像） |
| 奖励来源 | 人工键盘标注 (S/F/P) | 环境自动返回 | 固定零值 |
| `_display_episode_num` | 有 | 无 | 无 |

> 唯一非必需的接口是 `_display_episode_num` 属性，只 `RobotEnv` 有，`trainer.train()` 通过 `hasattr` 做可选注入，用于终端显示当前 episode 编号。

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

> **`s^p` 的提取细节**：`vla_input.state` 是 VLA 预处理后的状态张量，包含 7 维关节位置 + 1 维夹爪 = 8 维实际本体数据。但 VLA 的 `PadStatesAndActions` 变换会将其零填充到 VLA 内部宽度（如 32 维）。切片 `[:, :action_dim]` 只取前 8 维实际数据，丢弃后面的零填充。

**`_get_warmup_action(obs)`**（`rollout_worker.py:130-139`）：纯 VLA 参考动作，不经 Actor。

**`_get_actor_action(x, a_tilde_flat)`**（`rollout_worker.py:144-164`）：

```python
# rollout_worker.py:154-163
x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
a_tilde_t = torch.as_tensor(a_tilde_flat, dtype=torch.float32, device=self.device).unsqueeze(0)

a_flat = self.actor(x_t, a_tilde_t)  # ← rollout_worker.py:157, Actor 原始输出

# Safety: cap deviation from VLA reference  ← rollout_worker.py:159-163
deviation = a_flat - a_tilde_t
deviation = torch.clamp(deviation, -self.max_deviation, self.max_deviation)
a_flat = a_tilde_t + deviation
a_flat = a_flat.clamp(-1.0, 1.0)
return a_flat.squeeze(0).cpu().numpy().reshape(self.chunk_length, self.action_dim)
```

> `self.actor()` 的最终实现在 `src/rlt_openpi/models/actor.py:58` → `Actor.forward(x, a_tilde)`。

**Deviation Cap 安全设计**：

`a_flat` 是传给机械臂控制的最终动作（经 `env.step(action_chunk)` → `robot.move_to(action)` 执行），Actor 仅在 VLA 基模周围做小修正：

```
a_output = a_tilde(VLA基模) + clamp( Actor残差 , ±max_deviation )
```

| 设计要素 | 说明 |
|---------|------|
| **a_tilde 来源** | `VLAWrapper.get_rl_chunk_reference(obs, C)` → VLA 冻结基模预测 H=50 步，取前 C 步。受过大规模预训练，大概率不会输出危险动作 |
| **Actor 角色** | 只学习残差修正，初始化为零（最后层零初始化），训练起点 = 纯 VLA 参考 |
| **deviation cap** | `max_deviation=0.3`（可配置），强制每个动作元素偏离 VLA 参考不超过 ±0.3（动作已归一化到 [-1,1]） |
| **双重 clamp** | 残差 clamp 后，最终输出再做一次 `clamp(-1.0, 1.0)` 确保在合法动作范围内 |
| **配置参数** | `--max-deviation`（`OnlineRLTrainConfig.max_deviation`，默认 0.3） |

> 这是模型输出到机械臂之前的最后一道关卡。无论 Actor 训练结果如何，最终动作永远在 VLA 基模 ±0.3 范围内，物理上杜绝了大偏离导致机械臂失控的风险。

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

#### 3.2.2 Critic Update 详解

Critic 更新是 TD3 的核心，在 `_update_step()` 中分三步执行（`online_rl_trainer.py:137-159`）：

**Step 1: 计算 TD Target**（`td3_utils.py:compute_td_target`）

```
y = Σ_{k=0}^{C-1} γᵏ·rₖ   +   γ^C · (1 - done) · min Q_target(x', a')
     \_________________/       \___________________________________/
        chunk 内贴现回报                下一状态 bootstrap
```

关键点：
- **rewards** shape `[B, C]` — 一个 chunk 里 C=10 步各自的 reward，不是单个标量
- **chunk_return** = `γ⁰·r₀ + γ¹·r₁ + ... + γ⁹·r₉` → `[B, 1]`
- **next_a** = `actor(next_x, next_a_tilde)` + `clip(N(0, σ_target), ±c_target)` — TD3 的 target policy smoothing，防止 Q 网络过拟合到 sharp 峰值
- **bootstrap** = `γ^C · (1 - done) · min(Q1_target, Q2_target)` — clipped double Q 减少 overestimation bias
- 整个计算在 `@torch.no_grad()` 下，`td_target` 不参与梯度回传

**Step 2: 算当前 Q 值**

```python
q1, q2 = self.critic(x, a)   # 每个 [B, 1]
```

`x` = 当前 RL state，`a` = 实际执行的动作 chunk（buffer 里存的）。问的是"当时在这个状态下做这个动作，值多少？"

**Step 3: MSE Loss + 更新**

```python
c_loss = MSE(q1, y) + MSE(q2, y)
```

两个 Q 网络独立向同一个 target 回归。梯度只流过 `q1` 和 `q2`，不流过 `y`。

**Critic 更新数据流图**：

```
rewards [B,C]                   next_x [B,2056]
    │                                │
    ▼                                ▼
discounted sum          actor(next_x, next_a_tilde)
    │                     + clipped noise
    ▼                     + clamp[-1,1]
chunk_return [B,1]           │
    │                         ▼
    │              min Q_target(x', a') [B,1]
    │                         │
    │                     × γ^C·(1-done)
    │                         │
    ▼                         ▼
    └─────── + ───────────────┘
                 │
                 ▼
           td_target [B,1]  ←── detached (no grad)

x [B,2056]  ─┬─→  Q1  ─→ q1 [B,1]  ─→ MSE(q1, y)  ─┐
             │                                        ├─→ c_loss → backward → step
a [B,80]  ───┴─→  Q2  ─→ q2 [B,1]  ─→ MSE(q2, y)  ─┘
```

#### 3.2.2.1 HumanReward 得分与训练信号

**HumanReward 按键映射**（`src/rlt_openpi/rollout/reward.py`）：

| 按键 | 含义 | reward 值 | episode 行为 | 信号状态 |
|:----:|------|:---------:|-------------|---------|
| `S` / `Space` | 成功 | **+1.0** | 立即终止 (`done=True`) | 锁定（按一次后每次都返回 `"s"`） |
| `F` | 失败 | **0.0** | 立即终止 (`done=True`) | 锁定（按一次后每次都返回 `"f"`） |
| `P` | 进展（进度） | **+0.5**（可配置 `progress_reward`） | 继续 | 瞬时（返回一次后清空，可按多次） |
| 不操作 | 无信号 | **0** | 继续 | — |

> **为什么 S/F 锁定、P 不锁定？** 成功/失败是 episode 的最终结局，一旦发生就不应再改变，所以锁住信号保证所有后续步都能检测到终止条件。P 是进度标记，每次按键代表感受到机器人又往前推进了一步，消耗后等待下一次按键，可以在一个 episode 中多次标注。

**全零 Reward Chunk 的 TD Target 信号传播**：

大部分 chunk 人类不会按键（reward 全是 0），此时项①（chunk 贴现回报）= 0，TD target 退化为纯 bootstrap：

```
y = 0 + γ^C · (1 - done) · min Q_target(x', a')
  = γ^C · min Q_target(x', a')
```

信号完全依赖 `γ^C ≈ 0.904` 的折扣因子逐 chunk 反向传播。以一个 20 chunk 的 episode 为例，假设只在最后一个 chunk 人类按了 `S`（reward=1.0）：

```
chunk 19 (最后): y = 1.0 + 0                = 1.000  ← 直接拿到成功信号
chunk 18:        y = 0   + 0.904 × 1.000   = 0.904
chunk 17:        y = 0   + 0.904 × 0.904   = 0.817
chunk 16:        y = 0   + 0.904 × 0.817   = 0.739
...
chunk 0:         y = 0.904²⁰               ≈ 0.133  ← 仅剩 13%
```

第一个 chunk 的 TD target 只剩成功信号的 **13%**。chunk 越多，越早的 chunk 信号越弱。这是稀疏奖励 + chunk 级 MDP 的结构性特征，不是 bug：

| 场景 | 项①（chunk 回报） | 项②（bootstrap） | 信号来源 |
|------|:---:|:---:|------|
| 人类按了 S/F/P | 非零 | 叠加 bootstrap | 项① + 项② |
| 人类没按键（**大多数情况**） | **0** | 全靠 `γ^C·Q_target(x', a')` | 纯项② |
| episode 最后 chunk（done=1） | 可能有最后一步的 reward | **0**（done 截断） | 纯项① |

**缓解手段**：

| 手段 | 效果 | 代价 |
|------|------|------|
| **多用 P 键标注进度** | 中间 chunk 的项①不再为零，打断纯 bootstrap 链，信号更直接回传 | 需人工持续关注并标注 |
| **增大 γ**（如 0.99→0.995） | `γ^10 ≈ 0.951`，20 chunk 后剩 37%（vs 13%） | 策略更"远视"，价值估计方差增大 |
| **减小 C**（如 C=10→5） | `γ^5 ≈ 0.951`，同样 episode chunk 数加倍但每次衰减更小 | VLA 推理开销翻倍 |
| **增大 G（UTD ratio）** | 同样 episode 数做更多次更新，加速 Q 值反向传播 | 训练时间增加 |

> **P 键是最有效的缓解手段**——它在中间 chunk 注入非零 reward（默认 +0.5），直接打断纯 bootstrap 链。不需要改任何参数，只需要操作者在机器人有实质进展时随手按一下。

**当前已知问题**：

1. **`next_a_tilde` 近似**（`online_rl_trainer.py:145`）：用当前 transition 的 `a_tilde` 代替真正的 `next_a_tilde`（下一个 chunk 的 VLA 参考动作）。ReplayBuffer 只存了 `(x, a, a_tilde, rewards, next_x, done)`，没存 `next_a_tilde`。在 episode 边界附近或时序不连续时误差较大。**修复方案**：采集 episode 时顺手存 `next_a_tilde`（不需要额外 VLA 推理，收集时已经有了），或在 buffer 里加一个字段。

2. **Metrics 覆盖**（`online_rl_trainer.py:284-286`）：`for g in range(utd_ratio)` 循环中每次 `update_metrics = step_metrics` 会覆盖上一次结果，只保留最后一次 G 的指标。应改为累加取平均：

   ```python
   for g in range(cfg.utd_ratio):
       step_metrics = self._update_step(g)
       for k, v in step_metrics.items():
           update_metrics[k] = update_metrics.get(k, 0.0) + v / cfg.utd_ratio
   ```

#### 3.2.3 与标准 TD3 的对比

对比 canonical TD3 (Fujimoto et al., 2018)：

| TD3 组件 | 标准做法 | 你的实现 | 是否标准 |
|---------|---------|---------|:---:|
| Twin Q (Clipped Double Q) | `y = r + γ·min(Q1', Q2')(s', a')` | `y = Σγᵏrₖ + γ^C·min(Q1', Q2')(x', a')` | ✓ 标准 |
| Target Policy Smoothing | `a' = π'(s') + clip(N, ±c)` | `a' = actor(x', a_tilde') + clip(N, ±c)` | ✓ 标准 |
| Delayed Actor Updates | 每 d 次 critic 更新 1 次 actor | `update_idx % critic_updates_per_actor == 0` | ✓ 标准 |
| Polyak Target Update | `θ' = (1-τ)θ' + τθ` | 每个 `_update_step` 都更新 | ✓ 标准 |
| 单步 MDP | `y = r + γ·Q(s', a')` | Chunk 级: `y = Σγᵏrₖ + γ^C·Q(x', a')` | 标准推导* |
| 存 true next state | buffer 存完整 transition | `next_a_tilde ≈ a_tilde`（近似） | ✗ 非标准 |
| 每步更新 | 每 env step 做 1 次 update | 每 episode 后做 G=5 次 | off-policy 下合法 |

> \*Chunk 级 MDP 推导：将 C 步原始动作打包为一个 macro-step，discount 变为 γ^C，reward 为 Σγᵏ·rₖ。数学上等价于将 chunk 视为一个抽象时间步。

核心 TD3 三件套（twin Q + target smoothing + delayed actor）全部标准。唯一非标准的是 `next_a_tilde` 近似和 per-episode UTD 节奏。

#### 3.2.4 调参空间

按优先级和影响面分三个梯队：

**第一梯队：直接影响 Q 值估计质量**

| 参数 | 当前值 | 作用 | 调参方向 |
|------|--------|------|---------|
| `target_noise_sigma` | 0.2 | 目标策略平滑强度 | 太小→Q 值过拟合尖峰；太大→bootstrap 信号太弱。范围 0.1~0.3 |
| `target_noise_clip` | 0.5 | 噪声裁剪 | 跟 action 范围 [-1,1] 配合。范围 0.3~0.5 |
| `gamma` | 0.99 | 远期回报权重 | `γ^C = 0.99^10 ≈ 0.904`，每 chunk bootstrap 权重只剩 90%，有效 horizon ~10 chunk。减小=更短视，增大=更远视 |
| `tau` | 0.005 | Polyak 更新速率 | **关键：跟 UTD ratio 联调。** G=5 次后 `θ_target` 偏离约 `(1-(1-τ)^G) ≈ 2.5%`。如果 G 增大需相应减小 τ。等效 tau ≈ τ × G |

**第二梯队：Actor-Critic 平衡**

| 参数 | 当前值 | 作用 | 调参方向 |
|------|--------|------|---------|
| `utd_ratio` | 5 | 每个 episode 的更新次数 | 增大→训练更快但可能过拟合当前 episode。收集慢就增大 G |
| `critic_updates_per_actor` | 2 | Actor 更新延迟 | TD3 论文推荐 2。增大→Actor 更稳定但学得更慢 |
| `bc_regularizer_beta` | 0.5 | BC 正则项权重 | 控制偏离 VLA 的程度。初期大→安全，后期小→更多探索。可考虑退火 |
| `actor_lr` / `critic_lr` | 3e-4 / 3e-4 | 学习率 | 通常 `critic_lr > actor_lr`（如 1e-3 vs 1e-4）。当前相同可试试放大 critic |

**第三梯队：探索与稳健性**

| 参数 | 当前值 | 作用 | 调参方向 |
|------|--------|------|---------|
| `actor_noise_sigma` | 0.1 | 探索噪声标准差 | 0.1 相对保守。增大→更多探索但可能不稳定 |
| `ref_action_dropout` | 0.5 | 训练时丢弃 VLA 参考 | 50% 不用 VLA 参考。增大→更独立，减小→更依赖 VLA |
| `batch_size` | 256 | 每次更新采样数 | 对于 off-policy RL 偏小，可试 512 或 1024 |
| `mlp_hidden_dim` | 256 | 网络宽度 | 2048 维 z_rl 压缩后进 256 宽 MLP，可适当加宽到 512 |

**建议优先关注**：

1. **`tau` 和 UTD ratio 联动** — 当前 `tau=0.005`、G=5 时等效更新量 ≈2.5%，合理。如果增大 G，需相应减小 tau
2. **`next_a_tilde` 近似影响** — 在 buffer 里加 `next_a_tilde` 字段消除近似误差
3. **Q 值监控** — logger 里关注 `q1_mean`、`q2_mean`，如果 Q 值随时间单调膨胀，说明 overestimation 没被控住
4. **`bc_regularizer_beta` 退火** — 训练初期 beta 大（多依赖 VLA 参考），后期逐步减小（多信任自己的 Q 值估计）

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
                    │   无 Actor 参与)  │
                    │ → trainer:213-243│
                    │ → worker:158-203 │
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
│                   → 对应 rollout_worker.py:205-280          │
│                      collect_episode()                      │
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
│              → 主循环: online_rl_trainer.py:284-286        │
│              → 单步:  online_rl_trainer.py:115-184          │
│                      _update_step()                         │
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
| | `vla_action_horizon` (H) | 50 | VLA 输出的动作 horizon |
| **Actor-Critic** | `mlp_hidden_dim` | 256 | MLP 隐藏层宽度 |
| | `mlp_num_hidden_layers` | 2 | MLP 深度 |
| | `actor_noise_sigma` | 0.1 | 探索噪声标准差 |
| | `ref_action_dropout` | 0.5 | 参考动作 dropout 比例 |
| | `max_deviation` | 0.3 | Actor 输出偏离 VLA 参考的安全上限 |
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

#### 模型 ←→ 环境 交互循环

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        单次 chunk 交互循环                                │
│                                                                          │
│  ┌─────────────────────┐              ┌──────────────────────────────┐   │
│  │    模型侧 (GPU)      │              │       环境侧 (机器人/仿真)    │   │
│  │                      │              │                              │   │
│  │  VLA (冻结)          │              │                              │   │
│  │  RLTokenModel (冻结)  │              │                              │   │
│  │  Actor (可训)        │              │                              │   │
│  │  Critic (可训)       │              │                              │   │
│  └──────────┬───────────┘              └──────────┬───────────────────┘   │
│             │                                     │                       │
│             │  ① env.reset()                      │                       │
│             │◄────────────────────────────────────│                       │
│             │  obs = {                            │                       │
│             │    joint_position:  [7] float32      │  相机拍照 + 读取编码器  │
│             │    gripper_position: [1] float32      │                       │
│             │    exterior_image_1: [H,W,3] uint8   │                       │
│             │    wrist_image:      [H,W,3] uint8   │                       │
│             │    exterior_image_2: [H,W,3] uint8   │                       │
│             │    prompt: "pick up the pen"         │                       │
│             │  }                                   │                       │
│             │                                     │                       │
│             │  ② VLA 编码:                         │                       │
│             │  VLA.preprocess_obs(obs)              │                       │
│             │    → VLA.extract_embeddings()        │                       │
│             │      → z [1, M, 2048]                │                       │
│             │    → RLTokenModel.encode(z)          │                       │
│             │      → z_rl [1, 2048]                │                       │
│             │    → VLA.get_rl_chunk_ref(C=10)      │                       │
│             │      → a_tilde [1, 10, 8] → [1, 80]  │                       │
│             │    → vla_input.state[:, :action_dim] │                       │
│             │      → s^p [1, 8]                    │                       │
│             │  x = cat(z_rl, s^p) → [1, 2056]      │                       │
│             │     ▲ env.action_dim=8  决定 s^p 宽   │                       │
│             │     ▲ env.chunk_length=10 决定 C      │                       │
│             │                                     │                       │
│             │  ③ Actor 推理:                        │                       │
│             │  Actor(x, a_tilde)                    │                       │
│             │    = a_tilde + MLP残差 + 探索噪声      │                       │
│             │    → [1, 80]                          │                       │
│             │  reshape(env.chunk_length,             │                       │
│             │          env.action_dim)               │                       │
│             │    → action_chunk [10, 8]              │                       │
│             │                                     │                       │
│             │  ④ env.step(action_chunk)             │                       │
│             ├────────────────────────────────────→│                       │
│             │                                     │  for k in 0..9:       │
│             │                                     │    robot.step(a[k])   │
│             │                                     │    ├─ 关节运动         │
│             │                                     │    └─ HumanReward      │
│             │                                     │       S→+1 F→0 P→+0.5│
│             │                                     │                       │
│             │  ⑤ 返回: next_obs, rewards[10],     │                       │
│             │◄────────────────────────────────────│      done, info       │
│             │                                     │                       │
│             │  ⑥ next_obs → _extract_rl_state()    │                       │
│             │     → next_x [2056]                  │                       │
│             │                                     │                       │
│  ──────────┼─────────────────────────────────────┼───────────────────────  │
│             │                                     │                       │
│  transition = (x, a, a_tilde, rewards[0..9],       │                       │
│                next_x, done)                      │                       │
│  → replay_buffer.add(transition)                  │                       │
│  → obs = next_obs, 回到 ②                          │                       │
│             │                                     │                       │
│  ═══════════════════════════════════════════════════════════════════════   │
│  每 chunk 循环: obs → ②编码 → ③推理 → ④执行 → ⑤⑥存储 → 回到②             │
│  ═══════════════════════════════════════════════════════════════════════   │
└──────────────────────────────────────────────────────────────────────────┘
```

**env 接口在交互循环中的角色总结**：

| 接口 | 对应步骤 | 方向 | 一句话 |
|------|---------|------|--------|
| `env.reset()` | ①→② | **机器人→模型** | 获取初始观测，是整个推理链的起点 |
| `env.action_dim` | ②③ | — | 决定 s^p 的切片宽度和 action_chunk 的 reshape 列数 |
| `env.chunk_length` | ②③④ | — | 决定取 VLA 前几步动作、step 循环执行次数 |
| `env.step(action)` | ③→④ | **模型→机器人** | 发送 Actor 输出的 10 步动作，机器人逐步执行 |
| `env.step()` 返回值 | ④→② | **机器人→模型** | 执行后返回新观测 + per-step 奖励，驱动下一轮推理 |

#### 变量说明

| 符号 | 含义 | 来源 |
|------|------|------|
| `z_rl` | RL token，prefix embedding 压缩成的单向量 | VLA 提取 `z` → RLTokenModel.encode(z) |
| `s^p` | proprioceptive state = 7 维关节位置 + 1 维夹爪 = 8 维（上标 p 表示 proprioceptive，本体感受） | VLA 预处理后 `Observation.state` 取前 `action_dim` 维 |
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
