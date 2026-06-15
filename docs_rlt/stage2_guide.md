# Stage 2 下一步指南

## 模型文件完整度检查

在运行 Stage 2 之前，确认以下所有模块代码已存在且完整：

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
| 20 | Stage 1 训练产物 | `checkpoints/rl_token/<run>/step_xxxx.pt` | **需要用户提供** |
| 21 | VLA pretrained 权重 | `~/.cache/openpi/.../model.safetensors` | **需要下载** |

> **结论**：Stage 2 所需的模型、训练、环境、交互代码均已完整定义，无需新增文件。

---

## 运行 Stage 2 的命令

### 前置条件

1. Stage 1 训练完成，得到 checkpoint 文件（如 `checkpoints/rl_token/run_xxx/step_5000.pt`）
2. VLA pretrained 权重已下载（如 `~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors`）

### 命令

```bash
python scripts/train_online_rl.py \
    --rl-token-checkpoint checkpoints/rl_token/run_xxx/step_5000.pt \
    --vla-checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors \
    --vla-config-name pi05_droid_finetune \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --task-prompt "pick up the pen" \
    --max-env-steps 100000
```

### 关键参数简表

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

### 你的情况（Stage 1 冻结 VLA）

```
Stage 1 checkpoint 内容：
  model: RLTokenModel 权重                              ← 加载到 rl_token_model
  config: RLTokenTrainConfig                            ← 用于重建模型结构
  optimizer / scheduler / step                          ← 不在 Stage 2 使用
  vla_model: 无（因为 vla_finetune_alpha=0）              ← VLA 直接用原始权重
```

---

## 假定 Stage 1 已完成

Stage 1 训练完成后，你会得到一个 checkpoint 文件（如 `checkpoints/rl_token/run_xxx/step_5000.pt`），包含：
- `RLTokenModel` 权重（encoder + decoder）

> **你的情况**：Stage 1 使用冻结 VLA（`vla_finetune_alpha=0`），checkpoint 中**只有** RLTokenModel 权重，**没有** `vla_model`。这意味着 Stage 2 的 VLA 直接从原始 pretrained checkpoint 加载，不经过 Stage 1 微调。

Stage 2 的核心流程：
1. 加载 Stage 1 的冻结 `RLTokenModel`（只用于提取 `z_rl`，不再训练）
2. 加载冻结的 VLA（提供参考动作 `a_tilde` 和 embedding）
3. 创建 Actor（MLP + 残差）和 Twin Q-Critic（TD3 框架）
4. 在真实/仿真环境中交互，收集 transition 到 Replay Buffer
5. 用 TD3 + BC Regularizer 更新 Actor 和 Critic

---

## 论文 Algorithm 1: Train RL Actor and Critic 伪代码

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

**算法 vs 代码映射**：

| 伪代码符号 | 代码实现 | 对应 config 参数 |
|-----------|---------|-----------------|
| C | `chunk_length` | 10 |
| G | `utd_ratio` | 5 |
| β | `bc_regularizer_beta` | 0.5 |
| τ | `tau` | 0.005 |
| γ | `gamma` | 0.99 |
| σ (探索噪声) | `actor_noise_sigma` | 0.1 |
| σ_target (TD3 平滑噪声) | `target_noise_sigma` | 0.2 |
| c_target (噪声裁剪) | `target_noise_clip` | 0.5 |
| ref_dropout | `ref_action_dropout` | 0.5 |
| critic_updates_per_actor | `critic_updates_per_actor` | 2 |
| a_tilde_masked | `Actor._apply_ref_dropout()` | — |
| Q_min | `TwinQCritic.q_min()` | — |
| chunk_return | `compute_td_target()` | — |

---

## 五步流程详细说明

### 步骤 1：加载冻结的 RLTokenModel

**涉及文件**：`src/rlt_openpi/utils/checkpoint.py` → `load_rl_token_model()`

**执行的代码**（`scripts/train_online_rl.py:46`）：
```python
rl_token_model = load_rl_token_model(config.rl_token_checkpoint, device="cuda")
```

**内部流程**（`checkpoint.py:14-44`）：
1. 从 Stage 1 的 `.pt` 文件读取 `config` 和 `model` 权重
2. 根据 config 中的结构参数重建 `RLTokenModel`：
   - `embedding_dim`（默认 2048）
   - `encoder_layers` / `encoder_heads`（默认 2/8）
   - `decoder_layers` / `decoder_heads`（默认 2/8）
3. 加载权重，移到 GPU，设为 eval 模式，冻结所有参数

**对应 config 参数**：`rl_token_checkpoint`（必填，Stage 1 checkpoint 路径）

---

### 步骤 2：加载冻结的 VLA

**涉及文件**：`src/rlt_openpi/vla/vla_wrapper.py` → `VLAWrapper`

**执行的代码**（`scripts/train_online_rl.py:38-56`）：
```python
# 从原始 pretrained 模型加载 VLA（你的情况：vla_finetune_alpha=0）
vla = VLAWrapper(
    checkpoint_path=config.vla_checkpoint_dir,  # 原始 pi05 权重路径
    config_name=config.vla_config_name,         # 如 "pi05_droid_finetune"
    device="cuda",
)
```

**两种 Stage 1 策略的差异**：

| Stage 1 策略 | `vla_finetune_alpha` | checkpoint 中有 `vla_model`？ | Stage 2 VLA 来源 |
|-------------|----------------------|------------------------------|-------------------|
| 冻结 VLA（你的情况） | `0` | 否 | 原始 pretrained 权重 |
| Joint Training | `>0`（如 1.0） | 是 | Stage 1 微调后的权重 |

**你的情况代码走这条路径**（`train_online_rl.py:50-56`）：
```python
stage1_ckpt = torch.load(config.rl_token_checkpoint, ...)
if "vla_model" in stage1_ckpt:              # ← False，跳过
    vla.extractor.pi0.load_state_dict(...)
else:
    log.warning("No fine-tuned VLA weights... using base VLA")  # ← 走这里
```
即 VLA 直接用 `--vla-checkpoint-dir` 指定的原始 pretrained 模型，不受 Stage 1 影响。

**VLA 在 Stage 2 中的作用**（通过 `VLAWrapper` 提供）：
| 方法 | 作用 | 调用位置 |
|------|------|----------|
| `extract_embeddings(obs)` | 提取 prefix embedding `z` [B, M, D] | `RolloutWorker._extract_rl_state()` |
| `get_rl_chunk_reference(obs, C)` | 取前 C 步 VLA 动作 `a_tilde` [B, C, d] | `RolloutWorker._extract_rl_state()` / `_get_warmup_action()` |
| `preprocess_obs(obs)` | 原始 obs → OpenPI transform → Observation | `RolloutWorker._obs_to_vla_input()` |

**恢复 Stage 1 微调权重**（line 50-56）：
```python
stage1_ckpt = torch.load(config.rl_token_checkpoint, ...)
if "vla_model" in stage1_ckpt:
    vla.extractor.pi0.load_state_dict(stage1_ckpt["vla_model"])
```
如果 Stage 1 是 joint training（`vla_finetune_alpha > 0`），checkpoint 中会包含微调后的 VLA 权重 `vla_model`，加载到 `vla.extractor.pi0` 上。

**对应 config 参数**：`vla_checkpoint_dir`、`vla_config_name`

---

### 步骤 3：创建 Actor 和 Twin Q-Critic

**涉及文件**：
- `src/rlt_openpi/models/actor.py` → `Actor`
- `src/rlt_openpi/models/critic.py` → `TwinQCritic`
- `src/rlt_openpi/models/networks.py` → `MLP`

**执行的代码**（`online_rl_trainer.py:62-78`）：
```python
# Actor: state_dim + action_chunk_dim → action_chunk_dim
self.actor = Actor(
    state_dim=embedding_dim + action_dim,    # z_rl(D) + s^p(d) = 2048 + 8
    action_chunk_dim=chunk_length * action_dim,  # C * d = 10 * 8 = 80
    hidden_dim=256,
    num_hidden_layers=2,
    sigma=0.1,       # 探索噪声
    ref_dropout=0.5,  # 参考动作 dropout
)

# Critic: 两个 Q-Network
self.critic = TwinQCritic(
    state_dim=embedding_dim + action_dim,
    action_chunk_dim=chunk_length * action_dim,
    hidden_dim=256,
    num_hidden_layers=2,
)
```

**Actor 关键设计**（`actor.py:31-87`）：
- 输入：`cat(x, a_tilde_masked)` — 其中 `x = cat(z_rl, s^p)`，`a_tilde` 以 50% 概率随机置零
- 输出：`a = (a_tilde + residual + noise).clamp(-1, 1)` — VLA 参考 + 学习残差
- **零初始化**：最后一层 Linear 的 weight/bias 全置零，所以训练开始时 Actor = VLA 参考
- **探索**：训练时加高斯噪声 N(0, 0.1²)，评估时不加

**Twin Q-Critic 设计**（`critic.py`）：
- 两个独立的 Q-Network（Q1, Q2），取 min 防止过估计
- 每个 Q-Network 输入 `cat(x, a)`，输出标量 Q 值
- 维护 target 网络（通过 deepcopy 创建），Polyak 更新：`θ_target = (1-τ)·θ_target + τ·θ_online`

**对应 config 参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_dim` | 2048 | z_rl 维度 |
| `action_dim` | 8 | 单步动作维度 |
| `chunk_length` | 10 | C，每 chunk 步数 |
| `mlp_hidden_dim` | 256 | MLP 隐藏层宽度 |
| `mlp_num_hidden_layers` | 2 | MLP 深度 |
| `actor_noise_sigma` | 0.1 | 探索噪声 |
| `ref_action_dropout` | 0.5 | 参考动作 dropout |

---

### 步骤 4：环境交互收集 Transition

**涉及文件**：
- `src/rlt_openpi/rollout/rollout_worker.py` → `RolloutWorker`
- `src/rlt_openpi/rollout/robot_env.py` → `RobotEnv`（真实机器人）
- `src/rlt_openpi/rollout/sim_env.py` → `SimEnv`（仿真）
- `src/rlt_openpi/rollout/reward.py` → `HumanReward`（键盘奖励标注）
- `src/rlt_openpi/training/replay_buffer.py` → `ReplayBuffer`

#### 4a. Warmup 阶段（Phase 1）

**代码**（`online_rl_trainer.py:214-243`）：
```
默认 warmup_steps=1000 个 chunk，使用纯 VLA 动作（无 Actor）。
每个 chunk 经历：
  提取 RL state → VLA 参考动作 → 执行动作 → 收集 per-step reward → 存储 transition
```

每次存储的 transition 包含 6 个字段：

| 字段 | 形状 | 说明 |
|------|------|------|
| `x` | [state_dim] | 当前 RL 状态 = cat(z_rl, s^p) |
| `a` | [C*d] | 执行的 flat 动作 chunk |
| `a_tilde` | [C*d] | VLA 参考动作 chunk |
| `rewards` | [C] | C 个 per-step 奖励 |
| `next_x` | [state_dim] | 下一步 RL 状态 |
| `done` | [1] | 是否终止 |

**核心函数：`RolloutWorker._extract_rl_state(obs)`**（`rollout_worker.py:98-128`）：
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

#### 4b. Online RL 阶段（Phase 2）

**代码**（`online_rl_trainer.py:250-256`）：
```
每个 episode:
  1. Actor 设为 eval 模式
  2. worker.collect_episode() 运行一个完整 episode
  3. 统计 reward、chunks、steps、interventions
```

**`RolloutWorker.collect_episode()` 详细步骤**（`rollout_worker.py:205-280`）：

```
对于每个 chunk boundary 循环：
  1. _extract_rl_state(obs) → x, a_tilde_flat
  2. 检查人工干预：intervention_mgr.check_intervention()
     - 如果有干预 → 用人类动作
     - 否则 → _get_actor_action(x, a_tilde_flat)
  3. env.step(action_chunk) 执行 C 步
     - 真实机器人：每一步后检查 HumanReward 键盘信号
     - 仿真：直接用 gymnasium env.step()
  4. _extract_rl_state(next_obs) → next_x
  5. replay_buffer.add(x, a, a_tilde, rewards, next_x, done)

  直到 done=True（成功/失败/超时）
```

---

### 【需要人为介入的步骤】

Stage 2 中有 **3 处需要人为介入**（标注 👤）：

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
| 👤3 | Chunk 边界 | VR 手柄按钮按下（可选）| 人类接管控制，松开后恢复 Actor |

> **注意**：👤2 只在真实机器人（`RobotEnv`）中需要；仿真环境（`SimEnv`）自动返回奖励。👤3 仅在配置 `--intervention-factory` 时启用。

---

---

### 步骤 5：TD3 + BC Regularizer 更新

**涉及文件**：
- `src/rlt_openpi/training/td3_utils.py` → `compute_td_target()`, `critic_loss()`, `actor_loss()`
- `src/rlt_openpi/training/online_rl_trainer.py` → `OnlineRLTrainer._update_step()`

**执行时机**：每个 episode 结束后，运行 **G = utd_ratio = 5** 次更新

**代码**（`online_rl_trainer.py:284-286`）：
```python
for g in range(cfg.utd_ratio):  # 5 次
    step_metrics = self._update_step(g)
```

**`_update_step()` 内部流程**（`online_rl_trainer.py:115-184`）：

```
1. 从 ReplayBuffer 随机采样 batch_size=256 条 transition
2. 计算 TD Target（td3_utils.py:16-70）：
   td_target = Σ_{k=0}^{C-1} γ^k * r_k + γ^C * (1 - done) * min(Q1_target, Q2_target)(next_x, a')
   其中 a' = actor(next_x, a_tilde') + clipped_noise（TD3 目标平滑）

3. 计算 Critic Loss（td3_utils.py:73-90）：
   L_critic = MSE(q1, td_target) + MSE(q2, td_target)
   更新 critic_optimizer

4. 每 critic_updates_per_actor=2 次更新一次 Actor（td3_utils.py:93-114）：
   L_actor = -Q_min(x, actor(x, a_tilde)).mean() + β * MSE(a, a_tilde)
                                         ↑ BC regularizer (β = 0.5)
   更新 actor_optimizer

5. Polyak 目标网络更新（每次都更新）：
   θ_target = (1 - τ) * θ_target + τ * θ_online   (τ = 0.005)
```

**损失函数公式总结**：

| 损失 | 公式 | 含义 |
|------|------|------|
| TD Target | `y = Σγᵏ·rₖ + γᶜ·(1-done)·min Q_target(x', a')` | 折扣 chunk 回报 + 自举 |
| Critic Loss | `MSE(Q₁, y) + MSE(Q₂, y)` | 双 Q 逼近 TD target |
| Actor Loss | `-Q_min(x, a).mean() + β·MSE(a, a_tilde)` | 最大化 Q + 贴近 VLA 参考 |

**对应 config 参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 0.005 | 目标网络 Polyak 系数 |
| `utd_ratio` | 5 | 每 episode 更新次数 |
| `bc_regularizer_beta` | 0.5 | BC 正则化强度 |
| `critic_updates_per_actor` | 2 | Actor 延迟更新比例 |
| `target_noise_sigma` | 0.2 | TD3 目标平滑噪声 |
| `target_noise_clip` | 0.5 | 噪声裁剪范围 |
| `actor_lr` | 3e-4 | Actor 学习率 |
| `critic_lr` | 3e-4 | Critic 学习率 |
| `batch_size` | 256 | 训练 batch size |

---

### Stage 2 训练产物：Actor-Critic 权重保存位置

**代码**：`online_rl_trainer.py:332-360`

**保存路径**：
```
{save_dir}/{run_name}/online_rl_ep{N}.pt
```

**默认路径示例**（`save_dir="checkpoints/online_rl"`，`run_name` 自动生成时间戳）：
```
checkpoints/online_rl/run_20250615_143052/online_rl_ep50.pt
checkpoints/online_rl/run_20250615_143052/online_rl_ep100.pt
...
```

**保存时机**：每 `save_every` 个 episode（默认 50）保存一次，训练结束时再做一次最终保存。

**每个 `.pt` 文件包含的 key**：

| key | 内容 |
|-----|------|
| `actor` | Actor 网络 `state_dict`（可单独取出来部署推理） |
| `critic` | TwinQCritic 网络 `state_dict`（推理时用 Q 值评估） |
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

---

## Stage 2 关键参数（OnlineRLTrainConfig）

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

## 涉及的文件

### 入口脚本
| 文件 | 作用 |
|------|------|
| `scripts/train_online_rl.py` | Stage 2 训练入口 |

### 训练核心
| 文件 | 关键类/函数 | 作用 |
|------|-------------|------|
| `src/rlt_openpi/training/config.py` | `OnlineRLTrainConfig` | Stage 2 参数配置 |
| `src/rlt_openpi/training/online_rl_trainer.py` | `OnlineRLTrainer` | TD3 训练循环、checkpoint |
| `src/rlt_openpi/training/td3_utils.py` | `compute_td_target()`, `critic_loss()`, `actor_loss()` | TD3 损失计算 |
| `src/rlt_openpi/training/replay_buffer.py` | `ReplayBuffer` | 经验回放缓存 |

### 模型
| 文件 | 关键类 | 作用 |
|------|--------|------|
| `src/rlt_openpi/models/actor.py` | `Actor` | 策略网络（残差+VLA 参考） |
| `src/rlt_openpi/models/critic.py` | `TwinQCritic`, `QNetwork` | 双 Q 值网络 |
| `src/rlt_openpi/models/networks.py` | `MLP` | 通用 MLP 组件 |
| `src/rlt_openpi/models/rl_token.py` | `RLTokenModel` | Stage 1 训练好的 encoder（冻结） |

### 环境与交互
| 文件 | 关键类/函数 | 作用 |
|------|-------------|------|
| `src/rlt_openpi/rollout/rollout_worker.py` | `RolloutWorker` | 环境交互、RL state 提取 |
| `src/rlt_openpi/rollout/robot_env.py` | `RobotEnv` | 真实机器人环境封装 |
| `src/rlt_openpi/rollout/sim_env.py` | `SimEnv` | 仿真环境封装 |
| `src/rlt_openpi/rollout/intervention.py` | `InterventionManager` | 人工干预基类 |
| `src/rlt_openpi/rollout/reward.py` | `HumanReward` | 键盘奖励标注 |
| `src/rlt_openpi/rollout/factory.py` | `make_env()`, `make_intervention()` | 动态工厂函数 |
| `src/rlt_openpi/envs/franka/env_factory.py` | `make_franka_env()` | Franka 机器人环境 |
| `src/rlt_openpi/envs/franka/intervention.py` | `VRInterventionManager` | VR 手柄干预 |

### VLA 接口
| 文件 | 关键类 | 作用 |
|------|--------|------|
| `src/rlt_openpi/vla/vla_wrapper.py` | `VLAWrapper` | VLA 调用封装（提取 embedding、参考动作） |
| `src/rlt_openpi/vla/embedding_extractor.py` | `EmbeddingExtractor` | Prefix embedding 提取 |

### 工具
| 文件 | 关键函数 | 作用 |
|------|----------|------|
| `src/rlt_openpi/utils/checkpoint.py` | `load_rl_token_model()` | 加载 Stage 1 模型 |
| `src/rlt_openpi/utils/logging.py` | `Logger` | wandb + stdout 日志 |

## Stage 2 数据流

```
环境观测 obs
    --> VLAWrapper.extract_embeddings(obs) --> z [B, M, D]
    --> RLTokenModel.encode(z)              --> z_rl [B, D]        (冻结)
    --> VLAWrapper.get_rl_chunk_reference() --> a_tilde [B, C*d]   (冻结)
    --> cat(z_rl, s_p)                      --> x [state_dim]

Actor(x, a_tilde) --> a (动作 chunk)
env.step(a)        --> next_obs, rewards, done

ReplayBuffer.add(x, a, a_tilde, rewards, next_x, done)

TD3 更新 (每 episode 后 G 次):
    batch = ReplayBuffer.sample()
    td_target = 折扣 chunk 回报 + γ^C * target_Q(next_x, actor(next_x, a_tilde) + noise)
    critic_loss = MSE(q1, target) + MSE(q2, target)
    actor_loss = -Q_min(x, actor(x, a_tilde)).mean() + β * MSE(a, a_tilde)
```
