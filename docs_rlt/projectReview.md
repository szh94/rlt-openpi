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
│                        │              │  │ ④ State = z_rl + proprio │
│                        │              │  │   [2048] + [32] = [2080]  │
│                        │              │  └──────┬────────────────────┘
│                        │              │         │
│                        │              │  ┌──────┴────────────────────┐
│                        │              │  │ ⑤ Actor + Critic (训练)   │
│                        │              │  │  训练: TD3 + BC 正则化    │
│                        │              │  │  推理: Actor → 修正动作   │
│                        │              │  │   [10, 32] 残差于 VLA    │
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
  │ x = cat(z_rl [2048], proprio_s_p [32]) = [B, 2080]         │
  └──────────────────────────────────────────────────────────────┘

Step ④: Actor 推理 (在线训练)
  ┌──────────────────────────────────────────────────────────────┐
  │ 输入: x [2080] + flat(ref_actions[:C]) [1600] → [3680]      │
  │ 网络: LayerNorm → Linear(3680→256) → ReLU →                 │
  │       Linear(256→256) → ReLU → Linear(256→320)               │
  │ 输出: a = ref_actions[:C] + residual  [320 → 10×32]          │
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

## 5. 训练流程

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

## 6. 测试

```bash
pytest tests/
```

涵盖 RL Token 编码器/解码器、VLA 嵌入提取、Actor + Critic 前向/反向传播、回放缓冲区、以及端到端 Stage 2 训练器的冒烟测试。

---

## 7. 环境依赖

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

## 8. 设计特点

1. **环境无关性**：RLT 核心算法与具体机器人硬件解耦，通过工厂函数模式支持任意机器人
2. **可插拔组件**：环境工厂、干预管理器、数据变换均可通过 CLI 参数指定导入路径
3. **基于 OpenPI**：复用 OpenPI 完整的变换链和模型配置，确保数据预处理与预训练 VLA 完全一致
4. **人类友好**：终端键盘奖励（无需 Enter）、VR 干预、Rich UI 显示，降低真实机器人实验的操作负担
5. **渐进式训练**：从离线行为克隆 → RL Token 训练 → 在线 RL 微调，逐步提升策略性能
