# RLT Encoder-Decoder Training (Stage 1)

## 核心动机

对在线RL来说，状态表示（state representation）的选择至关重要。直接对整个VLA（数十亿参数）做RL存在两个问题：

1. 表示维度太高（transformer每层的embedding都是高维的）
2. 在线更新整个模型在计算上和样本效率上都不现实

但同时，预训练VLA内部已经包含了丰富的、与任务相关的知识。核心挑战是：**如何从VLA的transformer特征中提取一个紧凑的、保留任务相关信息的表示，同时小到足够让轻量级actor-critic做在线RL？**

## RL Token 的 Encoder-Decoder 架构

RLT的做法是在预训练VLA上附加一个**小型的encoder-decoder transformer**，以encoder-decoder方式训练。

### 流程

1. **获取VLA的final-layer token embeddings**：
   - 给定状态 `s` 和语言指令 `ℓ`，VLA输出 token embeddings `z = f(s, ℓ; θ_vla)`
   - 这些embeddings分解为 `z_{1:M} = {z_1, ..., z_M}`，每个 `z_i` 对应一个输入token的embedding

2. **添加RL Token**：
   - 在序列末尾追加一个**可学习的embedding** `e_rl = e_ϕ(<rl>)`
   - 将增强后的序列送入一个轻量级 **encoder transformer `g_ϕ`**
   - encoder在特殊token位置（最后一个位置）的输出就是**RL token**：

   ```
   z_rl = g_ϕ([z_{1:M}, e_rl])_{M+1}
   ```

   这个 `z_rl` 就是最终用作RL状态的**压缩表示**（文中提到维度是 `1×2048`）

3. **Decoder 与重构损失**：
   - **decoder transformer `d_ϕ`** + 线性输出投影 `h_ϕ` 被训练来**自回归地重构原始VLA embeddings**
   - 设 `¯z_i = sg(z_i)` 表示对VLA embedding施加**stop-gradient**操作
   - 自回归重构损失（在演示数据集D上）：

   ```
   L_ro = E_D [ Σ_{i=1}^M || h_ϕ(d_ϕ([z_rl, ¯z_{1:i-1}]))_i - ¯z_i ||^2 ]
   ```

   即：decoder以 RL token `z_rl` 和之前已重构的embedding `¯z_{1:i-1}` 为条件，预测第i个embedding，与原始VLA的embedding做MSE损失。

4. **训练方式**：
   - 参数 `ϕ` 在**小规模任务特定演示数据集**上训练
   - 训练时VLA相对于 `L_ro` **保持冻结**（frozen）
   - 可以**可选地**同时结合对VLA的监督微调（即同时更新 `θ_vla`）
   - 训练完成后，**`θ_vla` 和 `ϕ` 都被冻结**，在线RL只使用 `z_rl` 作为状态表示

## 为什么这样设计？

- **瓶颈效应（Bottleneck）**：RL token 作为 encoder-decoder 之间的信息瓶颈，必须保留足够的信息才能让decoder成功重构所有输入embeddings。这强制 `z_rl` 学习到一个**紧凑且有信息量的压缩表示**。
- **保留预训练知识**：由于重构目标迫使RL token编码VLA的全部输入信息，它自然地保留了VLA中与任务相关的感知和行为知识。
- **小而高效**：`z_rl` 是一个固定大小的低维向量（如 `1×2048`），使得轻量级 actor-critic（2-3层MLP）可以高效地在其上进行在线RL，而不需要更新数十亿参数的VLA。

---

## 代码实现分析

### 关键文件

| 文件 | 作用 |
|------|------|
| `scripts/train_rl_token.py` | 训练入口脚本，组装各个组件 |
| `src/rlt_openpi/training/config.py` | 训练超参数配置 (`RLTokenTrainConfig`) |
| `src/rlt_openpi/training/data_loader.py` | 数据加载管道，复用OpenPI的transform链 |
| `src/rlt_openpi/training/rl_token_trainer.py` | 训练器 (`RLTokenTrainer`): 训练循环 + step逻辑 |
| `src/rlt_openpi/models/rl_token.py` | 模型定义: `RLTokenEncoder`, `RLTokenDecoder`, `RLTokenModel` |
| `src/rlt_openpi/vla/vla_wrapper.py` | VLA封装: embedding提取、action采样、joint forward |
| `src/rlt_openpi/vla/embedding_extractor.py` | 底层 embedding 提取与 joint forward hook 实现 |

---

### 训练入口: `scripts/train_rl_token.py`

```
TrainConfig (tyro CLI)
  ├── train: RLTokenTrainConfig    ← 模型架构 + 训练超参
  ├── repo_id: str                 ← LeRobot数据集 ID
  ├── data_transforms_fn: str|null ← 可选的数据变换工厂函数
  └── num_workers: int             ← 数据加载进程数

main(config):
  1. _resolve_data_transforms(...)  → 解析自定义数据变换
  2. VLAWrapper(checkpoint, config) → 加载VLA模型到CUDA
  3. RLTokenTrainer(config, cuda)   → 构建 encoder-decoder + optimizer
  4. Logger.from_train_config(...)  → wandb 日志
  5. build_data_loader(...)         → 构建无限数据迭代器
  6. trainer.train(vla, dataloader) → 执行训练
```

**两种训练模式**（通过 `vla_finetune_alpha` 控制）:
- **alpha=0** (默认): VLA冻结，仅训练 encoder-decoder (ϕ)
- **alpha>0**: 联合训练，同时训练 encoder-decoder (L_ro) 和微调 VLA (alpha * L_vla)

---

### 配置: `RLTokenTrainConfig`

```python
@dataclass
class RLTokenTrainConfig:
    # 架构参数
    embedding_dim: int = 2048      # VLA embedding 维度
    encoder_layers: int = 2        # encoder transformer 层数
    encoder_heads: int = 8         # encoder attention heads
    decoder_layers: int = 2        # decoder transformer 层数
    decoder_heads: int = 8         # decoder attention heads

    # 训练超参数
    num_train_steps: int = 5000    # 总训练步数
    batch_size: int = 32           # 批次大小
    learning_rate: float = 1e-4    # encoder-decoder 学习率
    weight_decay: float = 1e-5     # 权重衰减
    warmup_steps: int = 500        # 线性预热步数
    max_grad_norm: float = 1.0     # 梯度裁剪阈值

    # 联合训练
    vla_finetune_alpha: float = 0.0     # VLA 微调权重 (0=冻结)
    vla_learning_rate: float = 1e-5     # VLA 微调学习率
    gradient_checkpointing: bool = True # VLA 梯度检查点

    # 检查点
    vla_checkpoint_dir: str = ""         # VLA 权重路径
    vla_config_name: str = "pi05_droid_finetune"
    resume_checkpoint: str = ""          # Stage 1 恢复训练路径
    save_dir: str = "checkpoints/rl_token"
    save_every: int = 1000
```

---

### 数据管道: `build_data_loader()`

数据管道完全复用 OpenPI 的 transform 链，保证数据预处理与预训练VLA一致。

```
LeRobot Dataset (repo_id)
  → OpenPI data_config.repack_transforms    ← 重映射字段名（如 "action" → "actions"）
  → OpenPI data_config.data_transforms      ← 数据级变换（DroidInputs等）
  → OpenPI data_config.model_transforms     ← 模型级变换（ResizeImages, TokenizePrompt, Pad）
  → PyTorch DataLoader (collate_fn)         ← JAX tree collate
  → _InfiniteLoader                         ← 包装为无限迭代器
  → yield (Observation, actions)
```

关键实现细节:

- **自动检测action列名**: 兼容 `"action"` (标准LeRobot) 和 `"actions"` (OpenPI DROID)
- **_patch_repack_action_key**: 动态修改 OpenPI 的 repack transform 使其读取正确的 action 列
- **_InfiniteLoader**: 包装 DataLoader 为无限迭代器，并将 batch 转为 `Observation.from_dict()` + actions tensor
- **actions tensor shape**: `[B, action_horizon, action_dim]` (如 `[32, 16, 8]`)
- **支持自定义 camera layout**: 通过 `data_transforms` 参数传入自定义 transforms group

---

### 模型定义: `rl_token.py`

#### `RLTokenEncoder`

```
输入: z [B, M, D], pad_mask [B, M]
                        │
  e_rl = Parameter(1,1,D)  ← 可学习
                        │
  tokens = concat([z, e_rl], dim=1)  → [B, M+1, D]
  ignore_mask = ~extended_pad_mask    ← PyTorch: True=忽略
                        │
  TransformerEncoder(tokens, src_key_padding_mask=ignore_mask)
                        │
  z_rl = out[:, -1, :]   ← 取RL token位置（最后一个位置）
```

- TransformerEncoder: `batch_first=True, norm_first=True`, feedforward = 4 * D
- e_rl 初始化: `randn * 0.02`
- pad_mask 语义: `True = valid token`（内部取反适配 PyTorch 的 `src_key_padding_mask`）

#### `RLTokenDecoder`

```
输入: z_rl [B, D], z [B, M, D] (stop-grad), pad_mask [B, M]
                        │
  教师强迫输入: tgt = [z_rl, z_1, ..., z_{M-1}]  → [B, M, D]
  位置0输入=z_rl, 位置i输入=z_{i-1}, 位置i输出预测z_i
                        │
  causal_mask = generate_square_subsequent_mask(M)  ← 因果掩码
  memory = z_rl.unsqueeze(1)  → [B, 1, D]   ← 跨注意力memory
                        │
  TransformerDecoder(tgt, memory, tgt_mask=causal_mask,
                     tgt_key_padding_mask=~pad_mask)
                        │
  z_hat = h_phi(out)    ← Linear(D, D) 投影回embedding空间
```

- 教师强制: 使用前一个真实embedding作为当前输入（而非使用模型自己的预测）
- 跨注意力: decoder每一层都通过cross-attention关注 `z_rl`

#### `RLTokenModel`（组合）

```
forward(z, pad_mask):
  z = z.detach()                           ← stop-gradient on VLA embeddings
  z_rl = encoder(z, pad_mask)              ← [B, D]
  z_hat = decoder(z_rl, z, pad_mask)       ← [B, M, D]

  mse = (z_hat - z).pow(2).mean(-1)        ← [B, M] 逐token MSE
  masked_mse = mse * pad_mask.float()      ← 填充位置置零
  loss = masked_mse.sum() / num_valid      ← 仅对有效位置平均

  return loss, z_rl, z_hat

encode(z, pad_mask):                       ← 推理模式，无decoder
  return encoder(z, pad_mask)
```

**损失函数细节**:
- 先对每个 token 的 D 维取 MSE mean → `[B, M]`
- 再用 `pad_mask` 将填充位置置零
- 最终 loss = 所有有效位置的 MSE 之和 / 有效 token 总数
- VLA embedding 通过 `.detach()` 完全从计算图分离

---

### 训练器: `RLTokenTrainer`

#### 初始化

```python
RLTokenTrainer.__init__(config, device):
  self.model = RLTokenModel(...)        ← encoder-decoder 模型
  self.optimizer = AdamW(...)           ← 仅 ϕ 参数
  self.scheduler = LinearLR(warmup)     ← 线性预热 → 常数
  self._vla = None                      ← 联合训练时使用
```

#### 训练循环

```python
train(vla, dataloader, log_fn):
  if joint: _setup_joint_training(vla)
    1. vla.unfreeze()                   ← 启用VLA梯度
    2. 创建 vla_optimizer (AdamW, lr=1e-5)
    3. 创建 vla_scheduler

  if resume: load(resume_checkpoint)    ← 恢复模型和优化器状态

  for step in 1..num_train_steps:
    observations, actions = next(dataloader)
    metrics = step(vla, observations, actions)  ← _step_frozen 或 _step_joint
    log(metrics)
    save_checkpoint (每 save_every 步)
```

#### Frozen Step (`_step_frozen`)

```
observations → to(device)
                        │
  with torch.no_grad():                          ← VLA 完全冻结
    z, pad_mask = vla.extract_embeddings(obs)    ← [B, M, D], [B, M]
                        │
  loss, z_rl, z_hat = self.model(z, pad_mask)    ← L_ro
                        │
  optimizer.zero_grad()
  loss.backward()
  clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()
  scheduler.step()
```

#### Joint Step (`_step_joint`)

```
observations → to(device)
actions → to(device)
                        │
  z, pad_mask, l_vla = vla.compute_vla_loss_with_embeddings(obs, actions)
  ├── z: [B, M, D] (detached, stop-grad)         ← z 不参与VLA梯度
  ├── pad_mask: [B, M]
  └── l_vla: scalar (with grad for VLA)          ← VLA flow-matching loss
                        │
  l_ro, z_rl, z_hat = self.model(z, pad_mask)    ← L_ro (仅更新 ϕ)
                        │
  total_loss = l_ro + alpha * l_vla               ← 联合损失
                        │
  model_optimizer.zero_grad()
  vla_optimizer.zero_grad()
  total_loss.backward()                           ← 一次 backward
  clip_grad_norm_(model.parameters())             ← 裁剪 ϕ 梯度
  model_optimizer.step()                          ← 更新 ϕ
  clip_grad_norm_(vla.parameters())               ← 裁剪 θ_vla 梯度
  vla_optimizer.step()                            ← 更新 θ_vla
```

**关键: 计算图分离机制**
- `l_ro` 计算路径上的 `z` 是 detached 的，因此 `l_ro.backward()` 不会影响 VLA
- `l_vla` 来自 VLA 的 flow-matching loss，梯度自然流向 VLA
- 两者通过 `total_loss = l_ro + alpha * l_vla` 组合，backward 时各更新各的参数

---

### VLA Embedding 提取: `embedding_extractor.py`

#### `EmbeddingExtractor`

##### `extract_embeddings()` (frozen mode)

```
observation
  → pi0._preprocess_observation(obs, train=False)  ← 预处理图像+语言token
  → pi0.embed_prefix(images, masks, lang, masks)    ← 获取 prefix embeddings
      → images + language → token embeddings [B, M, D]
      → prefix_pad_masks [B, M]
      → prefix_att_masks [B, M] (0=prefix can see all)
  → make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
  → position_ids = cumsum(prefix_pad_masks) - 1
  → paligemma_with_expert.forward(                 ← 仅 prefix forward
      inputs_embeds=[prefix_embs, None])            ← suffix=None
  → 返回 prefix_out [B, M, D] (float32)
```

- 仅在 PaliGemma LM 上做 prefix-only forward（不运行 diffusion decoder）
- 强制使用 eager attention（避免 SDPA 与 prefix-only 不兼容）

##### `forward_joint()` (joint mode)

通过 **monkey-patch** 在 VLA 一次前向传播中同时获取 prefix embeddings 和 flow-matching loss:

1. 临时替换 `pi0.embed_prefix` → 捕获 `prefix_pad_masks`
2. 临时替换 `pi0.paligemma_with_expert.forward` → 捕获 `prefix_out`（detach + float32）
3. 执行 `pi0.forward(observation, actions)` → 获得 `per_element_loss`
4. 恢复原始方法

**为什么可行？** PI0 注意力模式中 prefix tokens 不 attend suffix tokens（prefix_att_masks = 0, suffix 以 1 开头），所以 prefix 的输出在完整 forward 和 prefix-only forward 中是相同的。

---

### 检查点系统

#### 保存 (`save()`)

```
checkpoint = {
  "model":         self.model.state_dict(),        ← encoder-decoder ϕ
  "optimizer":     self.optimizer.state_dict(),
  "scheduler":     self.scheduler.state_dict(),
  "step":          self._global_step,
  "config":        self.config,
  # 以下仅在联合训练时存在:
  "vla_model":     self._vla.extractor.pi0.state_dict(),    ← fine-tuned θ_vla
  "vla_optimizer": self.vla_optimizer.state_dict(),
  "vla_scheduler": self.vla_scheduler.state_dict(),
}
```

保存路径: `{save_dir}/{run_name}/rl_token_step{step}.pt`

#### 加载 (`load()`)

- 加载时自动检测 checkpoint 中是否包含 VLA 权重，若当前处于联合训练模式则恢复 VLA 权重
- `weights_only=False` 以支持 dataclass 对象的 pickle 加载

---

### 关键数据流总结

```
                     ┌──────────────────────────────────────┐
                     │           LeRobot Dataset             │
                     │     (human demonstration data)        │
                     └──────────┬───────────────────────────┘
                                │ OpenPI transform chain
                                ▼
                     ┌──────────────────────────────────────┐
                     │     (Observation, actions)           │
                     │     obs: images, state, prompt       │
                     │     actions: [B, H, action_dim]      │
                     └──────────┬───────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │ frozen mode       │                    │ joint mode
            ▼                   ▼                    ▼
┌───────────────────────┐  ┌──────────────────────────────┐
│ vla.extract_          │  │ vla.compute_vla_loss_with_   │
│ embeddings(obs)       │  │ embeddings(obs, actions)     │
│ → z [B, M, D]         │  │ → z [B, M, D] (detached)    │
│ → pad_mask [B, M]     │  │ → pad_mask [B, M]            │
│                       │  │ → l_vla (flow-matching loss) │
└──────────┬────────────┘  └──────────┬───────────────────┘
           │                          │
           ▼                          ▼
┌───────────────────────┐  ┌──────────────────────────────┐
│ RLTokenModel(z,       │  │ RLTokenModel(z, pad_mask)    │
│           pad_mask)   │  │ → l_ro + alpha * l_vla      │
│ → loss (L_ro)         │  │ → update ϕ AND θ_vla        │
│ → z_rl [B, D]         │  │ → z_rl [B, D]              │
│ → z_hat [B, M, D]     │  │ → z_hat [B, M, D]          │
└──────────┬────────────┘  └──────────┬───────────────────┘
           │                          │
           ▼                          ▼
         z_rl: 最终 RL 状态表示 (冻结, 用于 Stage 2 在线RL)
```

---

### 关键数据结构维度速查

| 符号 | shape | 说明 |
|------|-------|------|
| `z` | `[B, M, 2048]` | VLA prefix embeddings（M 可变） |
| `pad_mask` | `[B, M]` | 有效 token 掩码 (True = valid) |
| `z_rl` | `[B, 2048]` | 压缩后的 RL token |
| `z_hat` | `[B, M, 2048]` | decoder 重构的 embeddings |
| `e_rl` | `[1, 1, 2048]` | 可学习的 RL token embedding |
| `actions` | `[B, H, action_dim]` | ground-truth 演示动作 |
| `loss` | scalar | masked MSE + cosine 联合重构损失 |
| `l_vla` | scalar | VLA flow-matching 损失 (joint mode) |
| `ignore_mask` | `[B, M+1]` | ~extended_pad_mask (PyTorch convention) |
| `causal_mask` | `[M, M]` | decoder 因果注意力掩码 |

---

### 小结

RL Token 的 encoder-decoder 训练本质上是一个**自监督的表示压缩**过程：encoder将VLA的高维token embeddings压缩为一个紧凑的readout向量，decoder试图从该向量重构原始embeddings。训练后的RL token作为 **VLA内部知识与在线RL策略之间的桥梁**，使得冻结VLA能提供丰富的状态表示，而轻量级actor-critic能高效地进行样本效率的在线强化学习。

代码层面实现了两种训练模式：
1. **Frozen模式**: VLA完全冻结，只训练 encoder-decoder，适用于快速原型验证
2. **Joint模式**: 通过精巧的计算图分离设计，在一次backward中同时更新 encoder-decoder 和 VLA，适用于需要调优VLA表示的场景

---

## 附：Encoder-Decoder 架构的范式与变动空间

### 问题

Encoder-Decoder 架构有什么固定的范式吗？还是说有比较大的修改变动空间？

### 回答

#### 一、唯一"固定"的核心范式

```
输入 z_{1:M} (变长, D=2048)
  → Encoder (自注意力) → z_rl (单向量, D=2048)  ← 信息瓶颈
  → Decoder (自注意力 + 交叉注意力) → z_hat (重建)
  → Masked MSE Loss
```

这个范式是固定的，但**除此之外几乎全是可变的**。

#### 二、巨大的改动空间

##### 1. 瓶颈的构建方式（最核心的设计选择）

| 方式 | 当前项目做法 | 其他可能方案 |
|------|------------|------------|
| **如何得到隐变量** | 追加可学习 `e_rl` token，用 encoder 输出 | 平均池化、注意力池化、Perceiver Resampler、Q-Former 式查询 |
| **隐变量维度** | 单向量 (1×2048) | 多向量 (N×D)、更低维 (如 512)、更高维 |
| **隐变量性质** | 连续确定向量 | 离散编码 (VQ-VAE)、随机变量 (VAE, 加 KL 散度)、扩散潜变量 |

举例 — 用 Perceiver Resampler（Flamingo 架构）替代单 token 瓶颈：
```python
# 用 N 个可学习查询向量通过交叉注意力从 z 中提取信息
queries = nn.Parameter(torch.randn(1, N, D))  # e_rl 的推广版本
z_rl = cross_attn(queries, z, z)  # [B, N, D]
```
这样能得到多个 RL token，信息容量更大。

##### 2. Encoder 架构

| 当前 | 可替换为 |
|------|---------|
| `nn.TransformerEncoder` (2层, 8头, FFN=4D) | 更深/更浅、MLP-Mixer、Mamba (状态空间模型)、RWKV、甚至 CNN |
| 双向自注意力 | 因果自注意力、带相对位置编码 |

##### 3. Decoder 架构

| 当前 | 可替换为 |
|------|---------|
| `nn.TransformerDecoder` + causal mask | 双向解码（非自回归）、迭代精炼、扩散解码 |
| Teacher forcing（右移输入） | 从头生成（用起始符）、MLP 直接映射 |
| 交叉注意力到 `z_rl` | 把 `z_rl` concat 到 decoder 输入、自适应层归一化 (AdaLN) |

##### 4. 损失函数

| 当前 | 可替换为 |
|------|---------|
| Masked MSE + Cosine Similarity（已实现） | 对比损失 (InfoNCE)、感知损失、加 KL 散度做 VAE、GAN 式判别器 |

> **已更新**: 当前代码已将纯 MSE 升级为 **MSE + Cosine 联合损失**，详见下文「附：MSE + Cosine 联合训练范式」。

##### 5. 训练方式

| 当前 | 可替换为 |
|------|---------|
| 仅重建 loss | 联合 VLA loss 微调（`vla_finetune_alpha` 已支持）、对比学习、next-token prediction |

#### 三、对 RL Token 项目最有意义的改动方向

**1. 瓶颈容量** — 当前 `z_rl` 是单个 2048 维向量。对于复杂任务，1 个 token 可能不够。可以用 4-8 个 token 做瓶颈（类似 Perceiver 或 Q-Former），decoder 通过交叉注意力读取。这会增加 actor 的输入维度，但信息更丰富。

**2. Encoder/Decoder 层数** — 当前各 2 层。如果 VLA 的 prefix embedding 序列很长（如 256+ tokens），增加 encoder 层数（如 4-6 层）有助于更好地压缩。如果序列短，1 层可能就够。

**3. 瓶颈结构** — 如果用 VAE 式（预测均值+方差，加 KL loss），`z_rl` 就变成一个分布采样，RL 训练时天然有随机性，可能比 actor 加探索噪声更稳定。

**4. Decoder 结构** — 当前 decoder 用 teacher forcing 自回归地重建。但重建质量对下游 RL 真的重要吗？可能直接用一个简单的 MLP 把 `z_rl` 映射回 `z_hat`（去掉 decoder 中的自注意力），训练更快，效果未必差。

#### 四、总结

```
Encoder-Decoder 的"固定"核心:  输入 → 瓶颈 → 重建
Encoder-Decoder 的"变动"空间:  瓶颈形式 / 架构 / 损失 / 训练方式 → 几乎无限
```

当前项目实现的是**一种经典且可靠的方案**（标准 Transformer + MSE + Cosine），但瓶颈如何构造、损失如何设计、要不要重建，都是可以大改的点。

---

## 附：MSE + Cosine 联合训练范式

### 为什么需要 Cosine Loss？

在高维 embedding 空间（D=2048）中：

| | MSE | Cosine |
|---|---|---|
| 关注点 | 逐元素数值大小 + 方向 | 仅方向（向量夹角） |
| 对 magnitude 敏感 | 是 | 否 |
| 对方向偏差敏感 | 中等 | 非常敏感 |

- **MSE 约束幅度**：要求 `z_hat` 每个维度值和 `z` 接近，保证数值范围一致
- **Cosine 对齐方向**：在高维空间中"指向同一方向"比"每个坐标都对"更重要，cosine 更能捕捉语义层面的相似性

**纯 MSE 的问题**：可能得到一个"模长偏小但数值接近"的向量
**纯 Cosine 的问题**：可以重建出一个方向正确但模长完全不对的向量
**组合使用**：既保证方向正确（语义保留），又保证数值范围匹配（便于下游 actor 使用）

### 代码实现

文件 `src/rlt_openpi/models/rl_token.py`，核心改动：

```python
# RLTokenModel.__init__ 新增参数
cosine_weight: float = 0.1   # 0 = 退化为纯 MSE，向后兼容

# forward 中的联合 loss 计算
mse = (z_hat - z).pow(2).mean(dim=-1)                     # [B, M] MSE
cos_sim = F.cosine_similarity(z_hat, z, dim=-1)            # [B, M], range [-1,1]
loss_cos = (1.0 - cos_sim) * pad_mask.float()              # [B, M], cos=1时loss=0

num_valid = pad_mask.float().sum().clamp(min=1.0)
loss = (masked_mse.sum() + self.cosine_weight * loss_cos.sum()) / num_valid
```

### 使用方式

- `cosine_weight=0` → 退化为纯 MSE（向后兼容旧行为）
- `cosine_weight=0.1` → 默认值，MSE 为主，cosine 提供方向辅助信号
- `cosine_weight=1.0` → MSE 和 cosine 等权重

---

## 附：MAE vs VAE vs RL Token 架构对比

三者都遵循 encode → bottleneck → decode 的结构，但**瓶颈设计**和**训练目标**完全不同。

```
输入 ──► Encoder ──► Bottleneck ──► Decoder ──► 输出
                        │
                        ▼
                 核心差异在这里
```

### MAE (Masked Autoencoder)

```
输入图像 → Encoder(只看25%可见patch) → Decoder(可见latent + 75%mask token) → 重建完整图像
                                              ↑
                                       只算被mask位置的loss
```

- Encoder 只处理**未被 mask 的 patch**（高效）
- Decoder 输入 = encoder 输出的 latent + 可学习的 `[MASK]` token 填充被 mask 的位置
- **只对被 mask 的位置算 loss**，迫使模型从少量可见信息推断全局

### VAE (Variational Autoencoder)

```
输入 → Encoder → μ, σ → 采样 z = μ + ε·σ → Decoder → 重建输出
                     ↑                    ↑
                  N(0,1)先验          KL( N(μ,σ²) || N(0,1) ) 约束
```

- 瓶颈不是固定向量，而是一个**概率分布**
- 训练时从分布采样 `z`（reparameterization trick: `z = μ + ε·σ`）
- Loss = **重建 loss + KL 散度**，KL 项迫使隐空间接近标准正态分布
- 推理时可以从 N(0,1) 随机采样生成新样本

### 对比总结

| | MAE | VAE | RL Token |
|---|---|---|---|
| **瓶颈形式** | 可见 patch 的 latent + mask token | 均值 μ 和方差 σ（分布） | 单个 learnable token `e_rl` |
| **瓶颈维度** | 同输入维度（部分 masked） | 压缩到隐变量 z ~ N(μ, σ²) | 1×D 固定向量 |
| **Decoder 输入** | 可见 latent + 被 mask 的位置 | 从分布中采样 z | `z_rl` 作为 memory（交叉注意力） |
| **训练目标** | 只重建被 mask 的区域 (MSE) | 重建 + KL 散度约束 | 重建全部 token（MSE + cosine） |
| **瓶颈性质** | 确定性（部分缺失） | 随机性（分布采样） | 确定性（固定压缩） |

### RL Token 与 MAE/VAE 的关系

| 借鉴 MAE 的 | 借鉴 VAE 的 |
|---|---|
| Encoder-Decoder 结构 | 信息瓶颈压缩思想 |
| 重建 loss 作为训练信号 | 把变长序列压缩成固定表示 |
| 用 pad_mask 忽略无效位置（类似 MAE 只算 masked 区域） | — |

关键区别：RL Token 的瓶颈是**确定性的**（单个 `z_rl` 向量），不是 MAE 的 mask-then-predict，也不是 VAE 的概率采样。如果未来要加探索，可以把 VAE 的采样机制引入——让 `z_rl` 从分布中采样，RL 时天然带随机性。

---

## 附：关键代码行解读

### `pos_indices = torch.arange(M + 1, device=z.device).unsqueeze(0).expand(B, -1)`

逐步骤拆解（见于 `RLTokenEncoder.forward`）：

```python
# 1. torch.arange(M + 1, device=z.device)
#    创建 1D tensor: [0, 1, 2, ..., M]
#    M+1 是序列长度（M 个 VLA token + 1 个 RL token）

# 2. .unsqueeze(0)
#    [M+1] → [1, M+1]，增加 batch 维度
#    内容: [[0, 1, 2, ..., M]]

# 3. .expand(B, -1)
#    [1, M+1] → [B, M+1]
#    -1 表示该维度保持不变
#    在 batch 维度上重复 B 次（不复制内存，只是 view）

# 最终结果: shape [B, M+1]
# [[0, 1, 2, ..., M],
#  [0, 1, 2, ..., M],
#  ...
#  [0, 1, 2, ..., M]]   # B 行完全一致
```

**用途**：每个样本在 batch 中使用相同的位置索引 `[0..M]`，通过 `self.pos_embeddings(pos_indices)` 查表得到位置编码，注入到 token 序列中。对应的 Decoder 版本使用 `torch.arange(M)`（不包含 RL token 位置）。

**为什么不用 `torch.arange(M+1).expand(B, -1)`直接写？**
- `.expand()` 需要源 tensor 的第一个维度为 1 才能广播，`torch.arange(M+1)` 的 shape 是 `[M+1]`，没有 batch 维度，直接 expand 会报错。必须先 `unsqueeze(0)` 变成 `[1, M+1]`。
