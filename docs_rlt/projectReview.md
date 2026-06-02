# RLT-OpenPI 项目文档

## 1. 项目简介

**RLT-OpenPI** 是论文 *RL Token: Bootstrapping Online RL with Vision-Language-Action Models* (Xu et al., Physical Intelligence) 的非官方 PyTorch 实现。该项目基于 [OpenPI](https://github.com/Physical-Intelligence/openpi) 公开的 VLA (Vision-Language-Action) 模型检查点，在行为克隆 (Behavior Cloning, BC) 预训练策略之上引入在线强化学习（Online RL），使机器人能够通过与环境交互和人类反馈不断改进策略。

### 核心思想

- **Stage 1 (RL Token 训练)**：在演示数据集上训练一个信息瓶颈编码器-解码器，将 VLA 模型生成的变长前缀嵌入 `z_{1:M}` 压缩为一个定长的 **RL Token** `z_rl`，通过掩码 MSE 重构损失进行训练。
- **Stage 2 (在线强化学习)**：固定 VLA 和 RL Token 编码器，训练轻量级 Actor-Critic 网络。Actor 以 `(z_rl, VLA 参考动作)` 为条件输出残差动作，Critic 采用 TD3 风格的双 Q 网络进行评估。人类操作员可通过键盘或 VR 控制器提供成功/失败/进展奖励及干预示范。

### 论文与参考

- 论文：https://pi.website/research/rlt
- OpenPI 基础库：https://github.com/Physical-Intelligence/openpi

---

## 2. PI0.5 vs RLT 架构对比

RLT 基于 PI0.5 VLA 模型构建。两者的本质区别在于：
- **PI0.5** 是一个纯行为克隆（BC）模型——输入观测，输出动作轨迹（流匹配去噪），无反馈学习能力。
- **RLT** 在冻结的 PI0.5 之上外挂轻量级 Actor-Critic 模块，通过在线 RL 微调动作，使其超越演示数据质量。

### 2.1 PI0.5 完整架构（基线模型）

PI0.5（pi0.5）是 Physical Intelligence 开发的视觉-语言-动作（VLA）模型，核心是 **PaliGemma 2B + Action Expert 300M 的双专家 Transformer**。

#### 架构图

```
Observation 输入
├── images: [B, 224, 224, 3] × 3 (三摄像头)
├── tokenized_prompt: [B, ≤200] (文本指令 token IDs)
└── state: [B, 32] (机器人关节角度 + 夹爪, PI0.5 已离散化为文本 token 嵌入 prefix)

embed_prefix 阶段
├── SigLIP ViT: images → image_tokens [B, 768, 2048]    ← 每图 16×16=256 patch, 3 图共 768
├── Gemma embedder: text → text_tokens [B, L_text, 2048] ← L_text ≤ 200
└── concat → prefix_tokens [B, ~968, 2048]               ← 768 + L_text, 双向注意力

embed_suffix 阶段
├── noisy_actions [B, 50, 32]                            ← 扩散去噪起点 (推理时=纯噪声)
├── action_in_proj Linear(32→1024) → action_embeds [B, 50, 1024]
├── time_embedding (posemb_sincos + time_mlp) → time_emb [B, 1024]
└── suffix_tokens [B, 50, 1024]                           ← 因果注意力, adaRMSNorm 受 time_emb 调制

PaliGemma Transformer（18 层，双专家）
├── Expert 0 (PaliGemma 2B 权重) ← 处理 prefix_tokens, 双向注意力
├── Expert 1 (Action Expert 300M 权重) ← 处理 suffix_tokens, 因果注意力
├── 两个专家共享 Attention 层 (QKV 拼在一起算注意力再拆开)
├── prefix_out: [B, ~968, 2048]      ← Expert 0 输出
└── suffix_out: [B, 50, 1024]        ← Expert 1 输出

Flow Matching 去噪 (推理)
├── action_out_proj Linear(1024→32) → v_t [B, 50, 32]
├── Euler 积分 x_t + dt·v_t, 循环 10 步 (t=1.0 → t=0.0)
└── 预测动作 x_0 [B, 50, 32]

训练方式: 行为克隆 (BC)
├── 损失: Flow Matching MSE L_vla = E[||u_t - v_t||²]
├── u_t = (x_0 - x_1) 是真实速度场 (由噪声→数据的直线路径解析计算)
└── v_t = model(x_t, t, cond) 是模型预测的速度场
```

#### 训练方式

PI0.5 采用 **Flow Matching** 作为动作生成范式：
1. **前向扩散**：从数据分布 x_0 到噪声分布 x_1 = N(0,1)，线性插值 x_t = (1-t)·x_0 + t·x_1
2. **速度场预测**：模型学习预测 v_t = dx/dt = x_1 - x_0
3. **损失函数**：L_vla = E[||u_t - v_t||²]，其中 u_t = x_1 - x_0 是真实速度
4. **推理去噪**：从 x_1 = N(0,1) 开始，Euler 积分 10 步到 x_0

**关键特征**：
- 全部 ~5B 参数端到端训练
- 无强化学习、无值函数、无策略梯度
- 动作质量受限于演示数据质量

#### 推理流程

```
观测 → SigLIP + Gemma → prefix tokens → PaliGemma LM → prefix_out (丢弃)
                                              ↓
纯噪声 → action_in_proj → suffix tokens → PaliGemma LM → suffix_out
                                              ↓
                                    action_out_proj → v_t
                                              ↓
                                    重复 10 步 Euler 积分 → 最终动作 x_0
                                              ↓
                                    执行 50 步 (action_horizon=50)
```

### 2.2 RLT 完整架构

RLT 在 PI0.5 的基础上新增了轻量级模块，实现从纯 BC 到在线 RL 的跨越。

#### 架构对比：模块级

```
PI0.5 (纯 BC)                           RLT (在线 RL)
┌────────────────────────┐              ┌────────────────────────────────┐
│                        │              │                                │
│ Observation (images +  │              │ Observation (images + text)    │
│ text + state)          │              │                                │
│         │              │              │         │                      │
│         ▼              │              │         ▼                      │
│  ┌──────────────┐      │              │  ┌──────────────┐              │
│  │ PaliGemma LM │      │              │  │ PaliGemma LM │ (冻结 ✅)    │
│  │  (~5B params)│      │              │  │  (~5B params)│             │
│  │              │      │              │  │              │              │
│  │  输出动作    │      │              │  │  输出:       │              │
│  │  [50, 32]    │      │              │  │  ① 参考动作 [50,32]        │
│  └──────────────┘      │              │  │  ② 中间 token embeddings   │
│         │              │              │  │     [~968, 2048]           │
│         ▼              │              │  └──────┬────────────────────┘
│  ┌──────────────┐      │              │         │
│  │ 执行 50 步   │      │              │  ┌──────┴────────────────────┐
│  └──────────────┘      │              │  │ ③ RL Token Encoder (冻结) │
│                        │              │  │   [~968, 2048] → [2048]   │
│  ✅ 直接执行 VLA       │              │  └──────┬────────────────────┘
│     预测的动作         │              │         │
│                        │              │  ┌──────┴────────────────────┐
│                        │              │  │ ④ State = z_rl + s_p     │
│                        │              │  │   [2048] + [8] = [2056]   │
│                        │              │  └──────┬────────────────────┘
│                        │              │         │
│                        │              │  ┌──────┴────────────────────┐
│                        │              │  │ ⑤ Actor + Critic (训练)   │
│                        │              │  │  训练: TD3 + BC 正则化    │
│                        │              │  │  推理: Actor → 修正动作   │
│                        │              │  │   [10, 8] 残差于 VLA     │
│                        │              │  └──────┬────────────────────┘
│                        │              │         │
│                        │              │  ┌──────┴────────────────────┐
│                        │              │  │ ⑥ 执行 C=10 步修正动作   │
│                        │              │  │    人类键盘/VR 收集奖励   │
│                        │              │  └───────────────────────────┘
│                        │              │
│  无学习能力            │              │  ✅ VLA 不动，外挂小模块
│  动作受限于演示数据    │              │     通过在线 RL 持续改进
└────────────────────────┘              └────────────────────────────────┘
```

#### 关键架构差异汇总

| 维度 | PI0.5 | RLT |
|------|-------|-----|
| **方法学** | 行为克隆 (Flow Matching BC) | 在线 Actor-Critic (TD3 + BC 正则) |
| **VLA 参数** | 全参数训练 | **冻结**（Stage 2 完全冻结） |
| **动作生成** | 扩散去噪 10 步，预测 50 步 | VLA 参考动作 + Actor 残差修正 |
| **状态表示** | 无独立状态向量 | RL Token: VLA 所有 token 压缩为 1 个 2048 维向量 |
| **学习信号** | 仅监督学习 (MSE) | 环境奖励 + Critic Q 值 + BC 正则 |
| **训练数据** | 离线演示数据集 | 在线环境交互（15min-2h） |
| **额外参数量** | 0 | ~2-5M (Actor + Critic 2-3 层 MLP) |
| **控制频率** | 50 Hz 连续推理 | 块级控制 (C=10, ~5 Hz 决策) |
| **人类反馈** | 无 | 键盘成功/失败/进展 + VR 干预 |
| **动作 horizon** | H=50 (单次推理预测 50 步) | C=10 (每次推理 10 步，含残差修正) |
| **训练成本** | 极大量 GPU (数百卡) | 轻量级（小 MLP + 冻结 VLA） |

#### RLT 六步推理流程详解

```
Step ①: VLA 前向传播 (冻结)
  ┌──────────────────────────────────────────────────────────────┐
  │ Observation → embed_prefix → PaliGemma LM → prefix_out       │
  │                                            [B, ~968, 2048]   │
  │                                            (所有 token 嵌入)  │
  │                                            + 参考动作 [50,32] │
  └──────────────────────────────────────────────────────────────┘

Step ②: RL Token 编码 (冻结, Stage 1 训练)
  ┌──────────────────────────────────────────────────────────────┐
  │ prefix_out [B, ~968, 2048] + 可学习 [RL] token              │
  │   → Transformer Encoder (2 层, 双向注意力)                   │
  │   → 取 [RL] 位置输出 = z_rl [B, 2048]                       │
  └──────────────────────────────────────────────────────────────┘

Step ③: 状态构建
  ┌──────────────────────────────────────────────────────────────┐
  │ x = cat(z_rl [2048], s_p [8]) = [B, 2056]                  │
  │ 对应: rollout_worker.py:120-123                              │
  └──────────────────────────────────────────────────────────────┘

Step ④: Actor 推理 (在线训练)
  ┌──────────────────────────────────────────────────────────────┐
  │ 输入: x [2056] + flat(ref_actions[:C]) [80] → [2136]       │
  │ 网络: LayerNorm → Linear(2136→256) → ReLU →                 │
  │       Linear(256→256) → ReLU → Linear(256→80) ◄─ 零初始化   │
  │       对应: actor.py:58-77                                   │
  │ 输出: a = ref_actions[:C] + residual  [80 → 10×8]            │
  │ 推理时: 直接取 mu (无探索噪声)                                 │
  └──────────────────────────────────────────────────────────────┘

Step ⑤: 环境交互
  ┌──────────────────────────────────────────────────────────────┐
  │ 机器人执行 C=10 步修正动作                                    │
  │ 人类观察并给出键盘信号: s(成功+1), f(失败+0), p(进展+0.5)      │
  │ 可选 VR 干预: 人类直接接管机器人                                │
  │ transition = (x_t, a_t, r_1:C, x_{t+C}, done) 存入 buffer    │
  └──────────────────────────────────────────────────────────────┘

Step ⑥: TD3 学习更新 (每 episode 后, UTD=5)
  ┌──────────────────────────────────────────────────────────────┐
  │ Critic 更新: TD target = 折扣块回报 + γ^C · min Q_target    │
  │              L_Q = MSE(Q_pred, TD_target) × 2 (双 Critic)   │
  │ Actor 更新: L_π = -Q.mean() + β · MSE(a, ref_actions)      │
  │            每 2 步 Critic 更新 → 1 步 Actor 更新             │
  │ Polyak 更新: θ_target ← τ·θ_online + (1-τ)·θ_target        │
  └──────────────────────────────────────────────────────────────┘
```

---

## 3. RLT 模块与代码映射

下表展示了 RLT 论文中的每个模块在本项目代码中的具体实现位置。

### 3.1 Stage 1: RL Token 训练 (离线)

| RLT 论文模块 | 代码文件 | 核心类/函数 | 说明 |
|-------------|---------|------------|------|
| VLA 模型 (冻结) | `src/rlt_openpi/vla/embedding_extractor.py` | `EmbeddingExtractor` | 包装 PI0Pytorch, 默认 freeze=True, 提供 `extract_embeddings()` |
| VLA 前向 (嵌入提取) | `src/rlt_openpi/vla/embedding_extractor.py:45-99` | `EmbeddingExtractor.extract_embeddings()` | 只运行 prefix 前向，不跑扩散循环，输出 [B, M, D] 嵌入 |
| VLA 配置加载 | `src/rlt_openpi/vla/config.py` | `load_vla_config()` | 加载 OpenPI 注册的模型配置 (pi05_droid_finetune 等) |
| VLA 高级接口 | `src/rlt_openpi/vla/vla_wrapper.py` | `VLAWrapper` | 封装 VLA 推理、变换链、嵌入提取、动作采样 |
| RL Token 编码器 | `src/rlt_openpi/models/rl_token.py:15-75` | `RLTokenEncoder` | 追加 e_rl token → TransformerEncoder → 取最后位置 |
| RL Token 解码器 | `src/rlt_openpi/models/rl_token.py:78-151` | `RLTokenDecoder` | 教师强制 + 因果掩码 + cross-attention → h_phi 投影 |
| RL Token 联合模型 | `src/rlt_openpi/models/rl_token.py:154-223` | `RLTokenModel` | 组合 enc+dec, 提供 forward() (训练) 和 encode() (推理) |
| 掩码 MSE 损失 | `src/rlt_openpi/models/rl_token.py:196-210` | `RLTokenModel.forward()` | MSE 只在有效位置计算, pad_mask 屏蔽填充 |
| Stage 1 训练器 | `src/rlt_openpi/training/rl_token_trainer.py` | `RLTokenTrainer` | 冻结模式(alpha=0) + 联合微调模式(alpha>0) |
| 联合损失 L = L_ro + α·L_vla | `src/rlt_openpi/training/rl_token_trainer.py:283-333` | `RLTokenTrainer._step_joint()` | 单次前向同时获取嵌入和流匹配损失 |
| VLA 流匹配损失 | `src/rlt_openpi/vla/vla_wrapper.py:219-238` | `VLAWrapper.compute_vla_loss()` | 代理到 PI0Pytorch.forward() |
| 联合前向 (嵌入+损失) | `src/rlt_openpi/vla/embedding_extractor.py:101-163` | `EmbeddingExtractor.forward_joint()` | 函数补丁捕获中间 prefix_out + 同时计算 VLA 损失 |
| 配置 | `src/rlt_openpi/training/config.py:7-45` | `RLTokenTrainConfig` | embedding_dim, encoder_layers, vla_finetune_alpha 等 |
| 数据加载 | `src/rlt_openpi/training/data_loader.py` | `build_data_loader()` | 复用 OpenPI 变换链, 返回无限迭代器 |
| 训练入口 | `scripts/train_rl_token.py` | `main()` | CLI tyro 解析 → RLTokenTrainer.train() |

### 3.2 Stage 2: 在线 RL 训练

| RLT 论文模块 | 代码文件 | 核心类/函数 | 说明 |
|-------------|---------|------------|------|
| VLA (冻结) | `src/rlt_openpi/vla/vla_wrapper.py` | `VLAWrapper` | freeze=True, 仅用于提取嵌入+参考动作 |
| RL Token 编码器 (冻结) | `src/rlt_openpi/models/rl_token.py:212-223` | `RLTokenModel.encode()` | `@torch.no_grad()`, 仅推理模式 |
| Actor π_θ | `src/rlt_openpi/models/actor.py` | `Actor` | 输入 (state + ref_actions), 输出残差动作, 最后线性层零初始化 |
| Actor 参考动作 Dropout | `src/rlt_openpi/models/actor.py:79-87` | `Actor._apply_ref_dropout()` | 训练时 50% 概率置零 ref_actions |
| Actor 探索噪声 | `src/rlt_openpi/models/actor.py:74-76` | `Actor.forward()` 训练分支 | 添加 N(0, σ²), σ=0.1 |
| Critic Q_ψ | `src/rlt_openpi/models/critic.py:15-50` | `QNetwork` | (state+action) → Q 值, 2 层 MLP |
| Twin Q Critic | `src/rlt_openpi/models/critic.py:53-125` | `TwinQCritic` | 双 Q 网络 + 冻结目标副本 + Polyak 更新 |
| 目标网络软更新 | `src/rlt_openpi/models/critic.py:111-125` | `TwinQCritic.update_targets()` | θ_target ← lerp(θ_online, τ) |
| 共享 MLP 组件 | `src/rlt_openpi/models/networks.py` | `MLP` | LayerNorm → [Linear → ReLU]^N → Linear |
| Stage 2 训练器 | `src/rlt_openpi/training/online_rl_trainer.py` | `OnlineRLTrainer` | Warmup + 主循环 + 更新 |
| Warmup 阶段 | `src/rlt_openpi/training/online_rl_trainer.py:213-243` | `OnlineRLTrainer.train()` Phase 1 | 纯 VLA 策略收集数据填充 replay buffer |
| 主训练循环 | `src/rlt_openpi/training/online_rl_trainer.py:245-309` | `OnlineRLTrainer.train()` Phase 2 | 收集 episode → UTD 更新 |
| TD3 更新步骤 | `src/rlt_openpi/training/online_rl_trainer.py:115-184` | `OnlineRLTrainer._update_step()` | Critic 更新 + 延迟 Actor 更新 + Polyak |
| TD target 计算 | `src/rlt_openpi/training/td3_utils.py:16-70` | `compute_td_target()` | 折扣块回报 + γ^C·minQ_target |
| Critic 损失 | `src/rlt_openpi/training/td3_utils.py:73-90` | `critic_loss()` | MSE(q1, y) + MSE(q2, y) |
| Actor 损失 | `src/rlt_openpi/training/td3_utils.py:93-114` | `actor_loss()` | -Q.mean() + β·MSE(a, a_tilde) |
| 配置 | `src/rlt_openpi/training/config.py:48-121` | `OnlineRLTrainConfig` | gamma, tau, utd_ratio, bc_regularizer_beta 等 |
| 训练入口 | `scripts/train_online_rl.py` | `main()` | CLI tyro → OnlineRLTrainer.train() |

### 3.3 Rollout 与环境交互

| RLT 论文模块 | 代码文件 | 核心类/函数 | 说明 |
|-------------|---------|------------|------|
| Rollout Worker | `src/rlt_openpi/rollout/rollout_worker.py` | `RolloutWorker` | 编排 VLA+编码+Actor+环境+Buffer 交互 |
| 提取 RL 状态 x | `src/rlt_openpi/rollout/rollout_worker.py:98-128` | `RolloutWorker._extract_rl_state()` | z_rl + s^p → x, 同时获取 VLA 参考动作 |
| Actor 动作推理 | `src/rlt_openpi/rollout/rollout_worker.py:141-156` | `RolloutWorker._get_actor_action()` | x + a_tilde → Actor → 修正动作块 |
| Warmup 动作 | `src/rlt_openpi/rollout/rollout_worker.py:130-139` | `RolloutWorker._get_warmup_action()` | 纯 VLA 参考动作 (无 Actor) |
| Episode 收集 | `src/rlt_openpi/rollout/rollout_worker.py:205-280` | `RolloutWorker.collect_episode()` | 支持人类干预, stride=2 存储 |
| 真实机器人环境 | `src/rlt_openpi/rollout/robot_env.py` | `RobotEnv` | 通过三个回调(step/reset/get_obs)解耦机器人硬件 |
| 模拟环境 | `src/rlt_openpi/rollout/sim_env.py` | `SimEnv` | Gymnasium 包装器, 块级接口 |
| 人类键盘奖励 | `src/rlt_openpi/rollout/reward.py` | `HumanReward` | 非阻塞键盘监听, s=成功/f=失败/p=进展 |
| 干预基类 | `src/rlt_openpi/rollout/intervention.py` | `InterventionManager` | 可扩展的人类干预接口 |
| VR 干预 (Franka) | `src/rlt_openpi/envs/franka/intervention.py` | `VRInterventionManager` | Oculus VR 控制器读写机器人 |
| Franka 环境工厂 | `src/rlt_openpi/envs/franka/env_factory.py` | `make_franka_env()` | Franka + DROID + ZED 三摄像头 |
| 数据变换 | `src/rlt_openpi/policies/franka/config.py` | `three_camera_droid` | 三摄像头 DROID 输入配置 |

### 3.4 数据管理与存储

| RLT 论文模块 | 代码文件 | 核心类/函数 | 说明 |
|-------------|---------|------------|------|
| Replay Buffer | `src/rlt_openpi/training/replay_buffer.py` | `ReplayBuffer` | 固定容量环形缓冲区, 预分配 numpy |
| stride=2 子采样存储 | `src/rlt_openpi/training/replay_buffer.py:75-105` | `ReplayBuffer.add_episode_strided()` | 50Hz → ~25 样本/秒 |
| Buffer 采样 | `src/rlt_openpi/training/replay_buffer.py:137-156` | `ReplayBuffer.sample()` | 随机均匀采样 |
| 检查点保存 Stage 1 | `src/rlt_openpi/training/rl_token_trainer.py:176-203` | `RLTokenTrainer.save()` | model+optimizer+scheduler+可选的 VLA |
| 检查点加载 Stage 1 | `src/rlt_openpi/training/rl_token_trainer.py:205-223` | `RLTokenTrainer.load()` | 恢复训练 |
| 检查点保存 Stage 2 | `src/rlt_openpi/training/online_rl_trainer.py:332-360` | `OnlineRLTrainer.save()` | actor+critic+optimizers+buffer |
| 检查点加载 Stage 2 | `src/rlt_openpi/training/online_rl_trainer.py:362-388` | `OnlineRLTrainer.load()` | 恢复训练 |
| 日志 | `src/rlt_openpi/utils/logging.py` | `Logger` | wandb + stdout 统一封装 |
| 终端 UI | `src/rlt_openpi/utils/display.py` | `TrainingDisplay` | Rich 风格进度/结果展示 |

### 3.5 工具与评估

| 模块 | 代码文件 | 说明 |
|------|---------|------|
| 统一评估 | `scripts/evaluate.py` | 自动检测 Stage 1 / Stage 2 检查点类型 |
| JAX→PyTorch 转换 | `scripts/tools/convert_jax_to_pytorch.py` | 转换 OpenPI JAX/Orbax → PyTorch safetensors |
| 归一化统计量 | `scripts/tools/compute_norm_stats.py` | 计算 LeRobot 数据集的归一化统计量 |
| LeRobot 转换 | `scripts/tools/convert_to_lerobot.py` | 原始演示 → LeRobot 格式 |
| 视频→图像转换 | `scripts/tools/convert_video_to_image_dataset.py` | 避免对 torchcodec 的运行时依赖 |

---

## 4. 项目结构 (完整)

```
rlt-openpi/
├── src/rlt_openpi/                 # 核心源代码
│   ├── __init__.py
│   │
│   ├── models/                     # 神经网络模型定义
│   │   ├── rl_token.py             # RL Token 编码器/解码器
│   │   ├── actor.py                # Actor (残差策略网络)
│   │   ├── critic.py               # Twin Q-Critic (双 Q 网络)
│   │   └── networks.py             # 共享 MLP 组件
│   │
│   ├── training/                   # 训练逻辑
│   │   ├── config.py               # 训练配置 (Stage 1 & 2)
│   │   ├── rl_token_trainer.py     # Stage 1 训练器
│   │   ├── online_rl_trainer.py    # Stage 2 在线 RL 训练器
│   │   ├── data_loader.py          # 数据加载（基于 OpenPI 流水线）
│   │   ├── replay_buffer.py        # 经验回放缓冲区
│   │   └── td3_utils.py            # TD3 算法工具函数
│   │
│   ├── vla/                        # VLA 模型封装
│   │   ├── config.py               # OpenPI 配置加载
│   │   ├── vla_wrapper.py          # VLA 模型高级封装
│   │   └── embedding_extractor.py  # 嵌入提取钩子
│   │
│   ├── rollout/                    # 环境交互与数据收集
│   │   ├── rollout_worker.py       # 数据收集工作器
│   │   ├── robot_env.py            # 真实机器人环境
│   │   ├── sim_env.py              # 模拟器环境
│   │   ├── reward.py               # 人类键盘奖励
│   │   ├── intervention.py         # 人类干预接口
│   │   └── factory.py              # 动态工厂函数
│   │
│   ├── envs/franka/                # Franka Panda 机器人环境示例
│   │   ├── env_factory.py          # Franka 环境工厂
│   │   └── intervention.py         # VR 干预管理器
│   │
│   ├── policies/franka/            # Franka DROID 数据变换
│   │   ├── config.py               # 三摄像头数据变换配置
│   │   └── policy.py               # 三摄像头 DROID 输入变换
│   │
│   └── utils/                      # 工具模块
│       ├── checkpoint.py           # 模型检查点加载
│       ├── logging.py              # wandb + stdout 日志
│       └── display.py              # Rich 终端 UI 显示
│
├── scripts/                        # 入口脚本
│   ├── train_rl_token.py           # Stage 1 训练入口
│   ├── train_online_rl.py          # Stage 2 训练入口
│   ├── evaluate.py                 # 统一评估入口
│   └── tools/                      # 工具脚本
│       ├── convert_jax_to_pytorch.py       # JAX → PyTorch 模型转换
│       ├── convert_video_to_image_dataset.py # 视频 → 图像数据集转换
│       ├── convert_to_lerobot.py           # 转 LeRobot 格式
│       └── compute_norm_stats.py           # 计算归一化统计量
│
├── exp/                            # 实验脚本
│   ├── stage1.sh                   # Stage 1 训练示例命令
│   ├── stage2.sh                   # Stage 2 训练示例命令
│   ├── eval_vla.sh                 # VLA 评估示例命令
│   └── eval_full.sh                # 完整评估示例命令
│
├── tests/                          # 单元测试
│   ├── test_actor_critic.py
│   ├── test_embedding_extractor.py
│   ├── test_networks.py
│   ├── test_online_rl_trainer.py
│   ├── test_replay_buffer.py
│   └── test_rl_token.py
│
├── docs/                           # 文档与图片
├── setup_env.sh                    # 环境安装脚本
├── pyproject.toml                  # 项目配置文件
└── README.md                       # 项目说明
```

### 4.1 Stage 1: RL Token 训练

**目标**：训练编码器-解码器将 VLA 前缀嵌入压缩为单一定长 RL Token。

**文件**：`src/rlt_openpi/models/rl_token.py`

| 组件 | 功能 |
|------|------|
| `RLTokenEncoder` | 在输入末尾附加可学习 `e_rl` 标记，通过 Transformer 编码器处理，提取 RL Token 位置输出 |
| `RLTokenDecoder` | 以 `(z_rl, z_1, ..., z_{M-1})` 为教师强制输入，使用因果掩码的 Transformer 解码器重构嵌入 |
| `RLTokenModel` | 组合编码器+解码器，提供 `forward()`（训练）和 `encode()`（推理）接口 |

#### 两种训练模式

1. **冻结 VLA (alpha=0)**：仅训练编码器-解码器，VLA 嵌入无梯度
2. **联合微调 (alpha>0)**：联合损失 `L = L_ro(phi) + alpha * L_vla(theta)`，编码器-解码器参数 phi 和 VLA 参数 theta 分别用各自的优化器更新

**文件**：`src/rlt_openpi/training/rl_token_trainer.py`

#### 配置文件

`src/rlt_openpi/training/config.py::RLTokenTrainConfig`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_dim` | 2048 | VLA 嵌入维度 |
| `encoder_layers` | 2 | 编码器 Transformer 层数 |
| `decoder_layers` | 2 | 解码器 Transformer 层数 |
| `vla_finetune_alpha` | 0.0 | VLA 微调权重 (0=冻结) |
| `num_train_steps` | 5000 | 训练步数 |
| `batch_size` | 32 | 批次大小 |

---

### 4.2 Stage 2: 在线强化学习

**目标**：在真实机器人上通过在线 RL 微调策略，利用人类反馈学习超出演示数据范围的技能。

#### 算法流程（对应论文 Algorithm 1）

1. **Warmup**：使用基础 VLA 策略收集数据填充回放缓冲区
2. **主循环**：收集一个 episode → UTD 比率 G 次梯度更新 → TD3 风格更新

#### 模型组件

**Actor** (`src/rlt_openpi/models/actor.py`)：
- 输入：`concat(z_rl, s^p, a_tilde)` → 残差动作
- 最后线性层零初始化，确保起始策略等同于 VLA
- 训练时添加高斯探索噪声 + 参考动作 Dropout

**Twin Q-Critic** (`src/rlt_openpi/models/critic.py`)：
- 两个独立 Q 网络 + 冻结目标副本 + Polyak 平均
- Target policy smoothing (TD3)

**MLP** (`src/rlt_openpi/models/networks.py`)：
- `LayerNorm → [Linear → ReLU] × N → Linear`

#### 训练器

`src/rlt_openpi/training/online_rl_trainer.py::OnlineRLTrainer`

| 方法 | 功能 |
|------|------|
| `_update_step()` | 单步 TD3 更新（TD target → Critic → 延迟 Actor → Polyak） |
| `train()` | Warmup + 主训练循环 |

#### TD3 工具函数 (`src/rlt_openpi/training/td3_utils.py`)

| 函数 | 公式 | 说明 |
|------|------|------|
| `compute_td_target` | `y = sum(gamma^k * r_k) + gamma^C * (1-done) * min Q_target(x', a')` | 折扣块回报 + 目标 Q 值 |
| `critic_loss` | `MSE(q1, y) + MSE(q2, y)` | 双 Q 网络 MSE 损失 |
| `actor_loss` | `-Q.mean() + beta * MSE(a, a_tilde)` | Q 最大化 + BC 正则化 |

#### 配置文件 (`OnlineRLTrainConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `action_dim` | 8 | 动作维度 |
| `chunk_length` | 10 | 动作块长度 C |
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 0.005 | Polyak 平均系数 |
| `utd_ratio` | 5 | 更新-数据比率 G |
| `bc_regularizer_beta` | 0.5 | BC 正则化系数 |

---

### 4.3 VLA 模型封装

`src/rlt_openpi/vla/vla_wrapper.py::VLAWrapper`

| 方法 | 功能 |
|------|------|
| `preprocess_obs()` | 原始环境观测 → 批量化 Observation |
| `extract_embeddings()` | 提取 VLA 后变换器前缀嵌入 `z_{1:M}` |
| `sample_reference_actions()` | 完整 VLA 扩散采样生成参考动作轨迹 |
| `get_rl_chunk_reference()` | 取前 C 步作为 RL 参考动作 |
| `compute_vla_loss()` | 计算 VLA 流匹配损失 |
| `compute_vla_loss_with_embeddings()` | 单次前向同时返回嵌入和损失 |

`src/rlt_openpi/vla/embedding_extractor.py::EmbeddingExtractor`
- `extract_embeddings()`：前缀专用前向传播，截取 PaliGemma 中间隐藏状态
- `forward_joint()`：函数补丁捕获中间嵌入 + 同时计算 VLA 损失

---

### 4.4 Rollout 与环境交互

**RolloutWorker** (`rollout/rollout_worker.py`)：
- `_extract_rl_state()`: x = cat(z_rl, s^p) + VLA 参考动作
- `_get_warmup_action()`: 纯 VLA 策略动作
- `_get_actor_action()`: Actor 网络动作
- `collect_episode()`: 收集完整 episode，支持人类干预

**环境系统**：
| 环境 | 文件 | 说明 |
|------|------|------|
| `RobotEnv` | `rollout/robot_env.py` | 通过三个回调与任意机器人栈解耦 |
| `SimEnv` | `rollout/sim_env.py` | Gymnasium 块级包装器 |
| `FrankaEnv` | `envs/franka/env_factory.py` | Franka + DROID 三摄像头 |

**人类反馈**：
| 按键 | 含义 | 效果 |
|------|------|------|
| `s` / `Space` | 成功 | 奖励 +1.0，结束 episode |
| `f` | 失败 | 奖励 0.0，结束 episode |
| `p` | 进步 | 奖励 +0.5，继续 episode |

**VR 干预** (`envs/franka/intervention.py`)：Oculus 控制器接管机器人，干预数据写入 buffer

---

### 4.5 数据加载 & 回放缓冲区

- `training/data_loader.py::build_data_loader()`：复用 OpenPI 变换链，无限迭代器
- `training/replay_buffer.py::ReplayBuffer`：固定容量循环缓冲区，stride=2 子采样，存储 `(x, a, a_tilde, rewards, next_x, dones)`

---

### 4.6 策略与数据变换

`policies/franka/`：三摄像头 DROID 配置，`ThreeCameraDroidInputs` 将三路 ZED 映射到模型输入

---

### 4.7 工具脚本 & 工具模块

| 脚本 | 说明 |
|------|------|
| `convert_jax_to_pytorch.py` | JAX/Orbax → PyTorch safetensors |
| `convert_video_to_image_dataset.py` | 视频 → 图像 LeRobot 格式 |
| `convert_to_lerobot.py` | 原始演示 → LeRobot 格式 |
| `compute_norm_stats.py` | 计算归一化统计量 |
| `rollout_vla.py` | VLA 模型 rollout 工具 |

| 模块 | 说明 |
|------|------|
| `utils/checkpoint.py` | 检查点加载（Stage 1/2） |
| `utils/logging.py` | wandb + stdout 日志 |
| `utils/display.py` | Rich 终端 UI |

---

## 5. Stage 2 神经网络与机械臂数据交换详解

Stage 2 的核心是从"摄像头图像 → GPU 推理 → 机器人关节指令 → 传感器反馈 → 训练更新"的完整闭环。下面逐层拆解数据如何通过网络流到物理机械臂。

### 5.1 单次 Chunk 推理→执行数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│ RolloutWorker.collect_episode() 中的一个 chunk 循环                        │
│                                                                          │
│  ① env.reset() / obs = next_obs                                          │
│     └── RobotEnv 调用 get_obs_fn()（用户提供的回调）                       │
│         └── DROID 机器人栈读取:                                            │
│             ├── 三台 ZED 摄像头 → 3× [480×640×3] uint8 图像               │
│             ├── 关节编码器 → 7× float32 (关节角度)                         │
│             └── 夹爪编码器 → 1× float32 (夹爪宽度)                        │
│         └── 返回 dict: {"observation/joint_position": [...],              │
│                         "observation/gripper_position": [...],            │
│                         "observation/exterior_image_1_left": ndarray,     │
│                         "observation/wrist_image_left": ndarray,          │
│                         "observation/exterior_image_2_left": ndarray,     │
│                         "prompt": "task instruction"}                     │
│                                                                          │
│  ② RolloutWorker._obs_to_vla_input(obs)                                   │
│     └── code: rollout_worker.py:80-96                                     │
│     └── VLAWrapper.preprocess_obs(obs)                                    │
│         ├── DroidInputs: 按 DROID schema 提取命名图像到模型槽位           │
│         ├── 注入默认 prompt（如果没有）                                    │
│         ├── Normalize: 使用数据集 norm_stats 标准化图像/state              │
│         ├── ResizeImages: [480×640] → [224×224]                          │
│         ├── TokenizePrompt: 文本指令 → token IDs [1, ≤200]                │
│         └── PadStatesAndActions: 补齐到固定长度                            │
│     └── → Observation (包含 images, state, tokenized_prompt 等)           │
│                                                                          │
│  ③ RolloutWorker._extract_rl_state(obs) ← 核心 RL 状态构建                │
│     ├── code: rollout_worker.py:98-128                                    │
│     ├── vla.extract_embeddings(vla_input)                                 │
│     │   └── EmbeddingExtractor: prefix-only forward pass                  │
│     │       └── code: embedding_extractor.py:46-99                       │
│     │       ├── SigLIP ViT: 3×[224,224,3] → [1, 768, 2048]               │
│     │       ├── Gemma embedder: token IDs → [1, ≤200, 2048]              │
│     │       ├── concat → [1, ~968, 2048]                                 │
│     │       └── PaliGemma LM 前向 (不跑 action expert, 不跑扩散)           │
│     │   → z: [1, ~968, 2048] (VLA 所有前缀 token 的最终层嵌入)             │
│     │                                                                     │
│     ├── rl_token_model.encode(z, pad_mask)                                │
│     │   └── code: rl_token.py:212-223                                     │
│     │   └── RLTokenEncoder:                                               │
│     │       ├── concat(z, e_rl) → [1, ~969, 2048]                        │
│     │       ├── TransformerEncoder ×2 层 (双向注意力)                      │
│     │       └── 取最后 [RL] 位置 → z_rl: [1, 2048]                        │
│     │                                                                     │
│     ├── vla.get_rl_chunk_reference(vla_input, C)                          │
│     │   └── code: vla_wrapper.py:200-217                                  │
│     │   └── sample_actions → flow matching 10 步去噪 → a_tilde [1, H, d] │
│     │   └── 取前 C 步 → a_tilde_chunk: [1, C, d]                          │
│     │   └── reshape → a_tilde_flat: [1, 80]  (C×d=80)               │
│     │                                                                     │
│     └── s_p = vla_input.state[:, :d] → [1, 8] (本体感觉, 去掉 VLA 填充)   │
│     └── x = torch.cat([z_rl, s_p], dim=-1) → [1, 2056]                   │
│                                                                          │
│  ④ RolloutWorker._get_actor_action(x, a_tilde_flat)                       │
│     └── code: rollout_worker.py:141-156                                   │
│     └── Actor.forward(x_t, a_tilde_t)                                     │
│         ├── code: actor.py:70-72                                          │
│         ├── a_tilde_input = a_tilde (eval 模式, 无 dropout)               │
│         ├── residual = MLP(cat([x, a_tilde_input])) → MLP([1, 2136])     │
│         │   ├── LayerNorm(2136)        ← shape: [2136]                    │
│         │   ├── Linear(2136→256) → ReLU                                   │
│         │   ├── Linear(256→256) → ReLU                                    │
│         │   └── Linear(256→80) ← 零初始化 → residual = 0 (初始)          │
│         └── mu = a_tilde + residual → [1, 80]                            │
│     └── reshape(C, d) → action_chunk [10, 8]                             │
│                                                                          │
│  ⑤ env.step(action_chunk) ← 发送到物理机器人                               │
│     └── RobotEnv.step()                                                   │
│         for k in range(C=10):                                            │
│           ├── self._step_fn(action_chunk[k])                              │
│           │   └── DROID droid.step(action_7d)                             │
│           │       └── [关节速度指令] → 电机驱动器 → 物理运动              │
│           ├── HumanReward.check() ← 非阻塞键盘监听                        │
│           │   ├── 's' → reward=1.0, done=True, success=True              │
│           │   ├── 'f' → done=True, success=False                         │
│           │   └── 'p' → reward=0.5, episode 继续                          │
│           └── time.sleep(control_period) ← 保持 15Hz 控制频率             │
│     └── 返回: next_obs (新传感器读数), rewards [10], done, info           │
│                                                                          │
│  ⑥ next_x, _ = RolloutWorker._extract_rl_state(next_obs)                 │
│     └── 对新观测重复步骤②③ → next_x: [2056], a_tilde_flat_new: [80]      │
│                                                                          │
│  ⑦ ReplayBuffer.add(x, a_flat, a_tilde_flat, rewards, next_x, done)     │
│     └── x:         [2056]  ← 当前 RL 状态                                │
│     └── a_flat:    [80]    ← Actor 实际执行的动作                        │
│     └── a_tilde:   [80]    ← VLA 参考动作 (用于 BC 正则)                 │
│     └── rewards:   [10]    ← 人类反馈的每步奖励                          │
│     └── next_x:    [2056]  ← 执行 C 步后的 RL 状态                      │
│     └── done:      [1]     ← 是否终止                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Warmup 阶段数据流

Warmup 与上述流程的区别仅在第④步：不使用 Actor，直接用 VLA 参考动作：

```
④ Warmup: action_chunk = VLA 参考动作 [C, d]  (无需 Actor 推理)
           code: rollout_worker.py:_get_warmup_action():130-139
           a_flat = action_chunk.reshape(-1)   [80]
           其余步骤完全相同
```

Warmup 的目的是用 VLA 基础策略收集初始数据填充 ReplayBuffer（通常 250-1000 个 chunk），为后续 Actor-Critic 训练提供足够的"经验"。

### 5.3 训练更新数据流（每 Episode 后）

```
RolloutWorker 收集完一个完整 episode 后（可能包含 N 个 chunk 循环）:

OnlineRLTrainer.train() 内循环 (UTD=G, 默认 5 次):

for g in range(G):                           ← 每条 episode 数据被复用 G 次
    ┌───────────────────────────────────────────────────────────────────┐
    │ ① ReplayBuffer.sample(B=256)                                      │
    │   └── code: replay_buffer.py:137-156                              │
    │   └── 从 buffer 随机均匀采样 256 条 transition:                    │
    │       x:       [256, 2056]  ← RL 状态                             │
    │       a:       [256, 80]    ← Actor 动作                          │
    │       a_tilde: [256, 80]    ← VLA 参考动作                         │
    │       rewards: [256, 10]    ← chunk 内每步奖励                     │
    │       next_x:  [256, 2056]  ← 下一状态                            │
    │       dones:   [256, 1]     ← 终止标志                            │
    │                                                                   │
    │ ② compute_td_target: TD target = r_discounted + γ^C·minQ_target  │
    │   └── code: td3_utils.py:16-70                                    │
    │   ┌── 折扣块回报: discount = γ^[0:C] → [0.99, 0.9801, ...]       │
    │   │   discounted_return = (rewards × discount).sum()  [256, 1]   │
    │   ├── next_a = target_actor(next_x, a_tilde)    [256, 80]        │
    │   ├── target smoothing: noise = clip(N(0,0.2), -0.5, 0.5)        │
    │   │   next_a = next_a + noise                     [256, 80]      │
    │   └── next_q = target_critic_q_min(next_x, next_a) [256, 1]       │
    │       td_target = discounted_return + γ^C·(1-done)·next_q [256,1] │
    │                                                                   │
    │ ③ Critic 更新 (每次执行)                                            │
    │   ├── q1, q2 = critic(x, a)                    ← 在线 Q 网络      │
    │   │      code: critic.py:82-88 (forward)                           │
    │   ├── c_loss = critic_loss(q1, q2, td_target)                      │
    │   │      code: td3_utils.py:73-90                                  │
    │   ├── c_loss.backward() → 梯度流入 critic.q1/q2.MLP 的所有参数    │
    │   └── critic_optimizer.step() → 更新 Critic 权重                   │
    │                                                                   │
    │ ④ Actor 更新 (每 2 次 Critic 更新执行 1 次, Delayed Policy)        │
    │   ├── 代码: online_rl_trainer.py:_update_step():168-176           │
    │   ├── a_actor = actor(x, a_tilde)               [256, 80]         │
    │   │   注意: 这里训练模式 → ref_dropout 50% 可能置零 a_tilde       │
    │   ├── q = critic.q_min(x, a_actor)              [256, 1]          │
    │   ├── a_loss = actor_loss(q, a_actor, a_tilde, β)                  │
    │   │      code: td3_utils.py:93-114                                  │
    │   ├── a_loss.backward() → 梯度流入 actor.MLP 的所有参数           │
    │   └── actor_optimizer.step() → 更新 Actor 权重                    │
    │                                                                   │
    │ ⑤ Polyak 目标网络更新 (每次执行)                                    │
    │   └── critic.update_targets(tau=0.005)                            │
    │       code: critic.py:110-125                                      │
    │       θ_target ← τ·θ_online + (1-τ)·θ_target                     │
    └───────────────────────────────────────────────────────────────────┘
```

### 5.4 时序细节：控制频率 vs 训练频率

```
时间轴 (物理世界):
                    ┌─── C 步执行 (~0.67s) ──┐
  Episode: reset → | chunk1 | chunk2 | ... | chunk_N | → episode_done
                    └─── 每步 ~67ms (15Hz) ──┘

时间轴 (GPU 推理):
  每 chunk 边界:
  ├── VLA 前向 (embed + PaliGemma prefix):          ~50-100ms  (取决于 GPU)
  ├── RL Token encode:                              ~1-2ms
  ├── VLA 扩散采样 10 步 (参考动作):                  ~100-200ms
  └── Actor 推理:                                    <1ms

时间轴 (训练):
  每 episode 收集完毕 (可能 30-150 chunks, 20-100 秒):
  └── G=5 次 UTD 更新 (每次 ~5ms):                   ~25ms
```

关键点：推理与机器人执行是**串行**的（推理期间机器人等待），但训练是**异步**的（episode 收集完成后批量更新）。

### 5.5 人类反馈集成

```
键盘奖励 (HumanReward):
  物理世界: 人类看到机器人动作 → 按下 's'/'f'/'p'
  代码世界: RobotEnv.step() 中每步调用 .check()
           → 信号存入 rewards[C] 数组
           → 's'/'f' 导致 done=True 结束 episode
  训练信号: TD target = sum(γ^k·r_k) + ...  ← rewards 直接参与计算

VR 干预 (VRInterventionManager):
  物理世界: 人类握持 Oculus 手柄 → 机器人被接管
  代码世界: check_intervention() 返回 True
           → get_human_action() 返回 human_action
           → env.step() 被跳过, human_action 直接作为 transition
  训练信号: 干预动作正常存入 buffer
           → Actor 的 BC 正则项 MSE(a, a_tilde) 拉向人类动作
```

---

## 6. 网络参数详解

### 6.1 Stage 1: RL Token 网络参数

#### RLTokenEncoder

| 层 | 变量名 | 输入维度 | 输出维度 | 参数量 | 公式 |
|----|--------|---------|---------|--------|------|
| e_rl (可学习 token) | `e_rl` | — | [1, 1, 2048] | 2,048 | 随机初始化 N(0,0.02) |
| **Layer 1** | | | | | |
| MultiheadAttention QKV | `transformer.layers.0.self_attn.in_proj_weight` | 2048 | 3×2048 | 12,582,912 | W_q, W_k, W_v 拼接 |
| Attention 输出投影 | `transformer.layers.0.self_attn.out_proj.weight` | 2048 | 2048 | 4,194,304 | concat(heads)W_O |
| LayerNorm1 (norm_first) | `transformer.layers.0.norm1.weight/bias` | 2048 | 2048 | 4,096 | γ, β |
| FFN Linear1 | `transformer.layers.0.linear1.weight/bias` | 2048 | 8192 | 16,785,408 | W₁x + b₁ |
| FFN Linear2 | `transformer.layers.0.linear2.weight/bias` | 8192 | 2048 | 16,779,264 | W₂·ReLU(·) + b₂ |
| LayerNorm2 | `transformer.layers.0.norm2.weight/bias` | 2048 | 2048 | 4,096 | γ, β |
| **Layer 1 小计** | | | | **50,350,080** | |
| **Layer 2** (与 Layer 1 结构相同) | `transformer.layers.1.*` | 2048 | 2048 | 50,350,080 | |
| **编码器总计** | | | | **100,702,208** | ≈ 100.7M |

**每层注意力机制细化（nhead=8, head_dim=256）**:
```
QKV 拆分:  W_q ∈ ℝ^{2048×2048}, W_k ∈ ℝ^{2048×2048}, W_v ∈ ℝ^{2048×2048}
           每个 head: 仅处理 256 维子空间
           并行计算: score = Q_i·K_i^T/√256, output = softmax(score)·V_i

FFN 扩展比: dim_feedforward / d_model = 8192 / 2048 = 4.0
```

#### RLTokenDecoder

| 层 | 变量名 | 输入维度 | 输出维度 | 参数量 |
|----|--------|---------|---------|--------|
| **Layer 1** | | | | |
| Self-Attention QKV | `transformer.layers.0.self_attn.in_proj_weight` | 2048 | 3×2048 | 12,582,912 |
| Self-Attention 输出 | `transformer.layers.0.self_attn.out_proj.weight` | 2048 | 2048 | 4,194,304 |
| Cross-Attention Q | `transformer.layers.0.multihead_attn.in_proj_weight` (q部分) | 2048 | 2048 | 4,194,304 |
| Cross-Attention KV | `transformer.layers.0.multihead_attn.in_proj_weight` (k,v部分) | 2048 | 2×2048 | 8,388,608 |
| Cross-Attention 输出 | `transformer.layers.0.multihead_attn.out_proj.weight` | 2048 | 2048 | 4,194,304 |
| LayerNorm1 (self-attn) | `transformer.layers.0.norm1.weight/bias` | 2048 | 2048 | 4,096 |
| LayerNorm2 (cross-attn) | `transformer.layers.0.norm2.weight/bias` | 2048 | 2048 | 4,096 |
| FFN Linear1 | `transformer.layers.0.linear1.weight/bias` | 2048 | 8192 | 16,785,408 |
| FFN Linear2 | `transformer.layers.0.linear2.weight/bias` | 8192 | 2048 | 16,779,264 |
| LayerNorm3 | `transformer.layers.0.norm3.weight/bias` | 2048 | 2048 | 4,096 |
| **Layer 1 小计** | | | | **67,131,392** |
| **Layer 2** (与 Layer 1 结构相同) | `transformer.layers.1.*` | 2048 | 2048 | 67,131,392 |
| h_phi 投影 | `h_phi.weight/bias` | 2048 | 2048 | 4,196,352 |
| **解码器总计** | | | | **138,459,136** | ≈ 138.5M |

#### RLTokenModel 总计

| 组件 | 参数量 |
|------|--------|
| 编码器 (2 层 Transformer) | 100,702,208 |
| e_rl 可学习 token | 2,048 |
| 解码器 (2 层 Transformer + h_phi) | 138,459,136 |
| **Stage 1 总参数量** | **239,163,392** ≈ **239.2M** |
| *(其中推理时仅用编码器)* | *(100.7M)* |

**训练成本**:
- Frozen VLA 模式: 仅训练 239.2M 参数 (编码器+解码器)
- 联合微调模式 (alpha>0): 额外训练 VLA 的所有 ~5B 参数
- 默认配置 (batch=32, steps=5000): 单卡 A100 约 2-4 小时

### 6.2 Stage 2: Actor 网络参数

#### Actor MLP 结构（默认配置） — `actor.py:45-56`

```
输入: x = cat(z_rl[2048], s_p[8], a_tilde_flat[80]) = [2136]   ← dim 来自 config.py:116,121
                                             ↓
                              ┌─────────────────────────┐
                              │ LayerNorm(2136)         │  ← γ, β: 4,272 参数
                              │ weight=[2136], bias=[2136]│
                              └──────────┬──────────────┘
                                         ↓
                              ┌─────────────────────────┐
                              │ Linear(2136 → 256)      │  ← 547,072 参数
                              │ weight=[256, 2136]      │
                              │ bias=[256]              │
                              └──────────┬──────────────┘
                                         ↓
                                      ReLU
                                         ↓
                              ┌─────────────────────────┐
                              │ Linear(256 → 256)       │  ← 65,792 参数
                              │ weight=[256, 256]       │
                              │ bias=[256]              │
                              └──────────┬──────────────┘
                                         ↓
                                      ReLU
                                         ↓
                              ┌─────────────────────────┐
                              │ Linear(256 → 80) ◄── 零初始化!   ← C×d = 80
                              │ weight=[80, 256]        │  ← 20,560 参数
                              │ bias=[80]               │
                              └──────────┬──────────────┘
                                         ↓
                              残差连接: mu = a_tilde + output  [B, 80]
                                         ↓
                              ┌─────────────────────────┐
                              │ Clamp [-1, 1]            │
                              └─────────────────────────┘
```

| 层 | 变量名 | 输入 | 输出 | 参数量 | 初始化 | 源代码 |
|----|--------|------|------|--------|--------|--------|
| LayerNorm | `mlp.net.0.weight/bias` | 2136 | 2136 | 4,272 | γ=1.0, β=0.0 | networks.py:15-30 |
| Linear1 | `mlp.net.1.weight/bias` | 2136 | 256 | 547,072 | Xavier uniform | networks.py:26 |
| Linear2 | `mlp.net.3.weight/bias` | 256 | 256 | 65,792 | Xavier uniform | networks.py:26 |
| Linear3 (输出) | `mlp.net.5.weight/bias` | 256 | 80 | 20,560 | **零初始化** | actor.py:54-56 |
| **Actor 总计** | | | | **637,696** | ≈ **638K** | |

**零初始化输出的意义** (actor.py:54-56):
- 训练开始时 residual=0 → mu = a_tilde → 策略 = VLA 参考动作
- 避免初始阶段 Actor 输出随机动作导致 Critic Q 值崩塌
- 随训练进行，残差逐渐学习到"修正量"

**参考动作 Dropout** (actor.py:79-87):
- 训练时以 `ref_dropout=0.5` 概率将整条 sample 的 a_tilde 置零
- 迫使 Actor 不依赖 VLA 参考也能产生合理动作
- 推理时 `self.training=False`, dropout 自动关闭

### 6.3 Stage 2: Critic 网络参数

#### 单个 QNetwork 结构 — `critic.py:25-38`

```
输入: cat(state[2056], action_chunk[80]) = [2136]        ← dim 来自 config.py:116,121
                                             ↓
                              ┌─────────────────────────┐
                              │ LayerNorm(2136)         │  ← 4,272 参数
                              │ weight=[2136], bias=[2136]│
                              └──────────┬──────────────┘
                                         ↓
                              ┌─────────────────────────┐
                              │ Linear(2136 → 256)      │  ← 547,072 参数
                              │ weight=[256, 2136]      │
                              │ bias=[256]              │
                              └──────────┬──────────────┘
                                         ↓
                                      ReLU
                                         ↓
                              ┌─────────────────────────┐
                              │ Linear(256 → 256)       │  ← 65,792 参数
                              │ weight=[256, 256]       │
                              │ bias=[256]              │
                              └──────────┬──────────────┘
                                         ↓
                                      ReLU
                                         ↓
                              ┌─────────────────────────┐
                              │ Linear(256 → 1)         │  ← 257 参数
                              │ weight=[256], bias=[1]  │
                              └──────────┬──────────────┘
                                         ↓
                                     Q 值 [B, 1] (标量)
```

| 层 | 变量名 | 输入 | 输出 | 参数量 | 源代码 |
|----|--------|------|------|--------|--------|
| LayerNorm | `q1.mlp.net.0.weight/bias` | 2136 | 2136 | 4,272 | networks.py:23 |
| Linear1 | `q1.mlp.net.1.weight/bias` | 2136 | 256 | 547,072 | networks.py:26 |
| Linear2 | `q1.mlp.net.3.weight/bias` | 256 | 256 | 65,792 | networks.py:26 |
| Linear3 (输出) | `q1.mlp.net.5.weight/bias` | 256 | 1 | 257 | networks.py:29 |
| **单 QNetwork** | | | | **617,393** | |

#### TwinQCritic 总计

| 组件 | 参数量 | 是否训练 |
|------|--------|---------|
| q1 (online) | 617,393 | ✅ | critic.py:71 |
| q2 (online) | 617,393 | ✅ | critic.py:72 |
| q1_target (frozen) | 617,393 | ❌ (深拷贝, 冻结) | critic.py:75 |
| q2_target (frozen) | 617,393 | ❌ (深拷贝, 冻结) | critic.py:76 |
| **Critic 在线参数量** | **1,234,786** | ≈ **1.23M** | |
| **Critic 总计 (含 target)** | **2,469,572** | ≈ **2.47M** | |

### 6.4 Stage 2 训练参数量汇总

| 组件 | 参数量 | 优化器 | 学习率 | 源代码 |
|------|--------|--------|--------|--------|
| Actor | 637,696 | Adam | 3e-4 | actor.py:31-56, online_rl_trainer.py:81 |
| Critic (online ×2) | 1,234,786 | Adam | 3e-4 | critic.py:53-80, online_rl_trainer.py:82 |
| **在线训练总计** | **1,872,482** ≈ **1.87M** | | | |
| 冻结 VLA + RL Token | ~5.24B | 不更新 | — | rollout 时推理 |
| **全系统总计 (推理时)** | **~5.24B** | — | — | VLA 前向为主 (~98%) |

**对比**:
- PI0.5 训练: ~5B 参数全部训练, 需数百 GPU
- RLT Stage 2: 仅 1.87M 参数训练 (Actor + Critic), 单 GPU 足够
- 训练/推理瓶颈在 VLA 前向 (embedding extraction + diffusion sampling), 占 ~98% 计算时间

### 6.5 输入/输出张量维度总表

#### Stage 1 (RL Token 训练)

| 变量 | 符号 | Shape (B=1) | Shape (B=32) | 说明 |
|------|------|-------------|--------------|------|
| VLA 前缀嵌入 | z | [1, ~968, 2048] | [32, ~968, 2048] | 图像768 + 文本~200 |
| 可学习 RL token | e_rl | [1, 1, 2048] | — | 与 batch 无关 |
| 编码器输入 | concat(z, e_rl) | [1, ~969, 2048] | [32, ~969, 2048] | 每 batch 拼接 |
| RL Token 输出 | z_rl | [1, 2048] | [32, 2048] | 取最后位置 |
| 解码器输入 | tgt | [1, ~968, 2048] | [32, ~968, 2048] | [z_rl, z_1, ..., z_{M-1}] |
| 重构嵌入 | z_hat | [1, ~968, 2048] | [32, ~968, 2048] | h_phi(decoder_out) |
| 掩码 MSE 损失 | loss | scalar | scalar | 仅有效位置 |

#### Stage 2 (在线 RL, Actor/Critic)

| 变量 | 符号 | Shape (单样本) | Shape (Batch=256) | 来自 | 源代码 |
|------|------|----------------|-------------------|------|--------|
| RL Token | z_rl | [2048] | [256, 2048] | RLTokenEncoder | rl_token.py:212-223 |
| 本体感觉 | s_p | [8] | [256, 8] | preprocessed obs state[:d] | rollout_worker.py:120 |
| RL 状态 | x | [2056] | [256, 2056] | cat(z_rl, s_p) | rollout_worker.py:123 |
| VLA 参考动作块 | a_tilde | [80] | [256, 80] | get_rl_chunk_reference, flatten | vla_wrapper.py:200-217 |
| **Actor 输入** | cat(x, a_tilde) | **[2136]** | **[256, 2136]** | | actor.py:71 |
| Actor 残差 | residual | [80] | [256, 80] | MLP 输出 | actor.py:71 |
| **Actor 输出** | mu = a_tilde + residual | **[80]** | **[256, 80]** | C×d=80 | actor.py:72 |
| **Critic 输入** | cat(x, a) | **[2136]** | **[256, 2136]** | | critic.py:50 |
| Critic 输出 (Q1) | q1 | [1] | [256, 1] | | critic.py:50 |
| Critic 输出 (Q2) | q2 | [1] | [256, 1] | |
| TD target | y | [1] | [256, 1] | r + γ^C·minQ |
| 折扣块奖励 | discounted_return | [1] | [256, 1] | sum(γ^k·r_k) |
| Actor 损失 | L_π | scalar | scalar | -Q + β·MSE |
| Critic 损失 | L_Q | scalar | scalar | MSE(q1,y)+MSE(q2,y) |

### 6.6 内存占用估算

| 组件 | 参数量 | 精度 | 显存占用 | 说明 |
|------|--------|------|---------|------|
| VLA 模型 (PI0.5) | ~5.0B | bfloat16 | ~9.4 GB | 冻结, 不存梯度 |
| RL Token Encoder | 100.7M | float32 | ~384 MB | 推理时冻结 |
| RL Token Decoder | 138.5M | float32 | ~529 MB | Stage 1 训练, Stage 2 丢弃 |
| Actor | 0.64M | float32 | ~2.4 MB | |
| Critic (online ×2) | 1.23M | float32 | ~4.7 MB | |
| Critic (target ×2) | 1.23M | float32 | ~4.7 MB | 冻结 |
| 优化器状态 (Actor) | 2×0.64M | float32 | ~4.8 MB | Adam: m + v |
| 优化器状态 (Critic) | 2×1.23M | float32 | ~9.4 MB | Adam: m + v |
| **Stage 2 总计** | | | **~9.8 GB** | 主要被 VLA 占据 |
| Batch 数据 (256条) | — | float32 | ~5.6 MB | x, a, a_tilde 等 |

---

## 7. 关键变量 ↔ 代码行映射

### 7.1 Stage 1 (RL Token 训练) 变量映射

| 变量 | 符号 | 维度 | 定义位置 | 使用位置 |
|------|------|------|---------|---------|
| VLA 前缀嵌入 | `z` | [B, M, 2048] | `embedding_extractor.py:97` | `rl_token.py:182, rl_token_trainer.py:268` |
| 填充掩码 | `pad_mask` | [B, M] | `embedding_extractor.py:99` | `rl_token.py:183, rl_token_trainer.py:268` |
| 可学习 RL token | `e_rl` | [1, 1, 2048] | `rl_token.py:34` | `rl_token.py:60` |
| 编码器输入 | `tokens` | [B, M+1, 2048] | — | `rl_token.py:61 (cat)` |
| RL Token (压缩) | `z_rl` | [B, 2048] | `rl_token.py:74` | `rl_token.py:199, rl_token.py:221` |
| 解码器输入 (教师强制) | `tgt` | [B, M, 2048] | — | `rl_token.py:128 (cat)` |
| 重构嵌入 | `z_hat` | [B, M, 2048] | `rl_token.py:150` | `rl_token.py:200` |
| 逐 token MSE | `mse` | [B, M] | `rl_token.py:203` | `rl_token.py:204` |
| 掩码 MSE 损失 | `loss` | scalar | `rl_token.py:208` | `rl_token_trainer.py:272` |
| VLA 流匹配损失 | `l_vla` | scalar | `embedding_extractor.py:159` | `rl_token_trainer.py:305 (alpha>0)` |

### 7.2 Stage 2 (在线 RL) 变量映射

#### Rollout 阶段 (数据收集)

| 变量 | 符号 | 维度 | 定义位置 | 使用位置 |
|------|------|------|---------|---------|
| 预处理观测 | `vla_input` | Observation | `rollout_worker.py:80-96` | `rollout_worker.py:109,113` |
| VLA 前缀嵌入 | `z` | [1, M, 2048] | `embedding_extractor.py:97` | `rollout_worker.py:109` |
| 填充掩码 | `pad_mask` | [1, M] bool | `embedding_extractor.py:99` | `rollout_worker.py:110` |
| RL Token | `z_rl` | [1, 2048] | `rl_token.py:221` | `rollout_worker.py:110` |
| 本体感觉 | `s_p` | [1, 8] | `rollout_worker.py:120` | `rollout_worker.py:123` |
| **RL 状态** | **`x`** | **[1, 2056]** | `rollout_worker.py:123` | `rollout_worker.py:152,229` |
| VLA 参考动作 | `a_tilde` | [1, H, d] | `vla_wrapper.py:186-198` | `vla_wrapper.py:215` |
| RL 参考动作块 | `a_tilde_chunk` | [1, C, d] | `vla_wrapper.py:216` | — |
| 展平参考动作 | `a_tilde_flat` | **[1, 80]** | `rollout_worker.py:114` | `rollout_worker.py:153` |
| Actor 动作 (展平) | `a_flat` | **[1, 80]** | `actor.py:76-77` | `rollout_worker.py:155` |
| 动作块 (env 输入) | `action_chunk` | [C, d] | `rollout_worker.py:156` | `robot_env.py:139` |
| 每步奖励 | `rewards` | [C] | `robot_env.py:140-171` | `rollout_worker.py:233` |
| 下一 RL 状态 | `next_x` | **[1, 2056]** | `rollout_worker.py:256` | `rollout_worker.py:258` |

#### Buffer 存储

| 变量 | 维度 | 存储数组 | 代码位置 |
|------|------|---------|---------|
| `x` | [capacity, 2056] | `self._x` | `replay_buffer.py:35` |
| `a` (flattened) | [capacity, 80] | `self._a` | `replay_buffer.py:36` |
| `a_tilde` (flattened) | [capacity, 80] | `self._a_tilde` | `replay_buffer.py:37` |
| `rewards` | [capacity, 10] | `self._rewards` | `replay_buffer.py:38` |
| `next_x` | [capacity, 2056] | `self._next_x` | `replay_buffer.py:39` |
| `dones` | [capacity, 1] | `self._dones` | `replay_buffer.py:40` |
| 采样 batch: `x, a, a_tilde, rewards, next_x, dones` | [B=256, ...] | — | `replay_buffer.py:137-156` |

#### 训练更新阶段 (_update_step)

| 变量 | 符号 | 维度 (B=256) | 定义位置 | 使用位置 |
|------|------|-------------|---------|---------|
| sampled batch | `x` | [256, 2056] | `replay_buffer.py:149` | `online_rl_trainer.py:130-135` |
| executed action | `a` | [256, 80] | `replay_buffer.py:150` | `online_rl_trainer.py:131,154` |
| VLA reference | `a_tilde` | [256, 80] | `replay_buffer.py:151` | `online_rl_trainer.py:132,145,170` |
| chunk rewards | `rewards` | [256, 10] | `replay_buffer.py:152` | `online_rl_trainer.py:133,141` |
| next RL state | `next_x` | [256, 2056] | `replay_buffer.py:153` | `online_rl_trainer.py:134,144` |
| dones | `dones` | [256, 1] | `replay_buffer.py:154` | `online_rl_trainer.py:135,143` |
| discount powers | `discount_powers` | [10] | `td3_utils.py:51` | `td3_utils.py:52` |
| discounted chunk return | `chunk_return` | [256, 1] | `td3_utils.py:52` | `td3_utils.py:68` |
| next target action | `next_a` | [256, 80] | `td3_utils.py:57` | `td3_utils.py:64,67` |
| target smoothing noise | `noise` | [256, 80] | `td3_utils.py:62` | `td3_utils.py:63-64` |
| target Q min | `next_q` | [256, 1] | `td3_utils.py:67` | `td3_utils.py:68` |
| **TD target** | **`td_target`** | **[256, 1]** | `td3_utils.py:70` | `online_rl_trainer.py:141-152,155` |
| online Q1/Q2 | `q1, q2` | [256, 1] | `critic.py:88` | `online_rl_trainer.py:154-155` |
| critic loss | `c_loss` | scalar | `td3_utils.py:89` | `online_rl_trainer.py:155` |
| actor action | `a_actor` | [256, 80] | `actor.py:72` | `online_rl_trainer.py:170-171` |
| min Q for actor | `q` | [256, 1] | `critic.py:97` | `online_rl_trainer.py:171-172` |
| actor loss | `a_loss` | scalar | `td3_utils.py:113` | `online_rl_trainer.py:172` |

---

## 8. 训练流程

### 前置条件

1. **下载 VLA 检查点**：从 [OpenPI Model Zoo](https://github.com/Physical-Intelligence/openpi#checkpoints) 下载 JAX 格式检查点
2. **转换模型**：`convert_jax_to_pytorch.py` → PyTorch safetensors
3. **准备数据集**：LeRobot 格式的演示数据

### Stage 1: RL Token 训练

```bash
python scripts/train_rl_token.py \
    --train.vla-config-name pi05_droid_finetune \
    --train.vla-checkpoint-dir /path/to/model.safetensors \
    --train.vla-finetune-alpha 1.0 \
    --train.batch-size 32 \
    --train.num-train-steps 5000 \
    --repo-id local/stack_the_blocks \
    --data-transforms-fn rlt_openpi.policies.franka.config.three_camera_droid
```

输出：
- `checkpoints/rl_token/<run_name>/rl_token_step<N>.pt`

### Stage 2: 在线 RL

```bash
python scripts/train_online_rl.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --intervention-factory rlt_openpi.envs.franka.intervention.make_vr_intervention \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir /path/to/model.safetensors \
    --rl-token-checkpoint checkpoints/rl_token/rl_token_step3000.pt \
    --task-prompt "stack the three blocks on the tray" \
    --warmup-steps 250 \
    --chunk-length 5 \
    --max-episode-chunks 150
```

### 评估

```bash
python scripts/evaluate.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir /path/to/model.safetensors \
    --checkpoint checkpoints/online_rl/run_latest/online_rl_ep100.pt \
    --task-prompt "stack the three blocks on the tray" \
    --num-episodes 50
```

评估脚本自动检测检查点类型（Stage 1 VLA 仅评估 vs Stage 2 完整评估），结果以 JSON 格式保存。

---

## 9. 测试

```bash
pytest tests/
```

涵盖 RL Token 编码器/解码器、VLA 嵌入提取、Actor + Critic 前向/反向传播、回放缓冲区、以及端到端 Stage 2 训练器的冒烟测试。

---

## 10. 环境依赖

- Python ≥ 3.11
- PyTorch 2.7.1
- OpenPI（从 GitHub 源码安装，commit `fdc03f5`）
- HuggingFace Transformers 4.53.2
- LeRobot（HuggingFace 数据集）
- wandb（实验日志）
- tyro（配置解析）
- rich（终端 UI）

可选依赖（机器人运行）：
- DROID 机器人栈
- ZED SDK（摄像头驱动）
- Oculus VR SDK（VR 干预）

---

## 11. 设计特点

1. **环境无关性**：RLT 核心算法与具体机器人硬件解耦，通过工厂函数模式支持任意机器人
2. **可插拔组件**：环境工厂、干预管理器、数据变换均可通过 CLI 参数指定导入路径
3. **基于 OpenPI**：复用 OpenPI 完整的变换链和模型配置，确保数据预处理与预训练 VLA 完全一致
4. **人类友好**：终端键盘奖励（无需 Enter）、VR 干预、Rich UI 显示，降低真实机器人实验的操作负担
5. **渐进式训练**：从离线行为克隆 → RL Token 训练 → 在线 RL 微调，逐步提升策略性能
