# VLA (PI0 / PI0.5) 完整前向流程

> 基于 `openpi/models_pytorch/pi0_pytorch.py` 和 `rlt-openpi/src/rlt_openpi/vla/embedding_extractor.py` 的代码追踪。
> 主图以 PI0.5 为准（rlt-openpi 默认使用 `pi05_droid_finetune`），PI0 差异在文末说明。

---

## 总览

PI0.5 没有传统的 encoder-decoder 结构。prefix 和 suffix 共享 18 层 Transformer 的 Self-Attention，然后在 FFN 处分叉：prefix 走 PaliGemma FFN（观测理解），suffix 走 **Action Expert FFN**（动作预测）。时间步通过 **adaRMS** 调制 Action Expert 的 RMSNorm 层。

```
                         observation
              (image)   (language prompt)   (state: 关节角等)
                 │            │                  │
                 │            │    ┌─────────────┘
                 │            │    │
                 │            ▼    ▼
                 │   ①  离散化 state (仅 PI0.5)
                 │      np.digitize(state, 256 bins)
                 │      → "0 128 255 64 ..."
                 │      → f"Task: {prompt}, State: {state_str};\n"
                 │      → PaliGemma tokenizer encode
                 │            │
                 ▼            ▼
        ①  _preprocess_observation
                 │
                 ├── images       [B, N_img, P²·C]
                 └── lang_tokens  [B, L]   (含 state token)
                 │
                 ▼
        ②  embed_prefix
           ├─ image → SigLIP ViT → img_emb
           └─ language → Embedding → lang_emb
                 │
          concat → prefix_embs [B, M, 2048]
                 │
                 │    ┌─── ②  embed_suffix ──────────────┐
                 │    │  x_t  [B,H,ad] → action_in_proj  │
                 │    │    → action_emb = suffix_embs     │
                 │    │    shape [B, H, 2048]             │
                 │    │                                   │
                 │    │  time [B] → sinusoidal → time_mlp │
                 │    │    → SiLU → SiLU                  │
                 │    │    → adarms_cond [B, 2048]        │
                 │    │    (调制 Expert RMSNorm)           │
                 │    └───────────────────────────────────┘
                 │           │
                 ▼           ▼
  ╔══════════════════════════════════════════════════════════════════════╗
  ║     ③  PaligemmaWithExpert Transformer (18 层, 共享, 每层如下)       ║
  ║                                                                      ║
  ║    prefix_embs [B,M,D]       suffix_embs [B,H,D]                     ║
  ║         │                          │                                 ║
  ║    QKV 投影 (PaliGemma)      QKV 投影 (Expert)                       ║
  ║    q_proj, k_proj, v_proj    q_proj, k_proj, v_proj                  ║
  ║    (各自独立权重)              (各自独立权重)                          ║
  ║         │                          │                                 ║
  ║         └─────────┬────────────────┘                                 ║
  ║                   ▼                                                  ║
  ║      共享 Self-Attention + RoPE (Q,K,V 沿 seq 维拼起来)               ║
  ║      prefix ↔ prefix (双向)                                          ║
  ║      suffix → prefix + suffix (因果)                                 ║
  ║      prefix → suffix (✗ 阻断: cumsum 规则)                           ║
  ║                   │                                                  ║
  ║      ┌────────────┴────────────┐                                     ║
  ║      ▼                         ▼                                     ║
  ║  Split att_output           Split att_output                         ║
  ║  [:, 0:M, :]                [:, M:, :]                               ║
  ║      │                         │                                     ║
  ║      ▼                         ▼                                     ║
  ║  O-proj (PaliGemma)        O-proj (Expert)                           ║
  ║  各自独立 Linear             各自独立 Linear                          ║
  ║      │                         │                                     ║
  ║      ▼                         ▼                                     ║
  ║  + residual                 + residual                               ║
  ║  (+ prefix_embs)            (+ suffix_embs)                          ║
  ║      │                         │                                     ║
  ║      ▼                         ▼                                     ║
  ║  post_attn_layernorm       post_attn_layernorm                       ║
  ║      │                         │                                     ║
  ║      │  包含: prefix token       │  包含: suffix token                 ║
  ║      │  之间互相 attend 后       │  attend 了全部 prefix              ║
  ║      │  的语义融合结果           │  (含 PI0.5 的离散 state)          ║
  ║      │  不含任何 suffix 信息     │  + 之前的 suffix 的动作上下文      ║
  ║      │                         │                                     ║
  ║      ▼                         ▼                                     ║
  ║  ┌──────────────┐    ┌──────────────────────────────┐                ║
  ║  │ PaliGemma FFN │    │  Action Expert FFN            │                ║
  ║  │ (标准权重)     │    │  (动作专用权重)               │                ║
  ║  │               │    │  RMSNorm ← adarms_cond       │                ║
  ║  └──────┬───────┘    └──────────────┬───────────────┘                ║
  ║         │                           │                                ║
  ║         ▼                           ▼                                ║
  ║    + residual                   + residual                           ║
  ║    (after_first_residual)       (after_first_residual)               ║
  ║         │                           │                                ║
  ║         ▼                           ▼                                ║
  ║    prefix_hidden               suffix_hidden                         ║
  ║    [B, M, 2048]                [B, H, 2048]                          ║
  ║         │                           │                                ║
  ║         └────── 重复 ×18 层 ─────────┘                                ║
  ║                    │                                                  ║
  ║                    ▼                                                  ║
  ║  最终输出:  prefix_out [B,M,2048]=z    suffix_out [B,H,2048]           ║
  ╚════════════════════════════════════════╪═════════════════════════════╝
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
            prefix_out (z)      ┌── Action Head ──────────────────────┐
            [B, M, 2048]        │  suffix_out [B, H, 2048]             │
                  │             │           │                          │
                  │  ★ Stage 1  │           ▼                          │
                  │   截取      │  ④  action_out_proj                  │
                  │             │      Linear(2048 → action_dim)       │
                  │             │           │                          │
                  │             │           ▼                          │
                  │             │      v_t [B, H, action_dim]          │
                  │             │       速度场预测 (Flow Matching)      │
                  │             │           │                          │
                  │             │           ▼                          │
                  │             │  ⑤  Diffusion Denoising              │
                  │             │      x_{t-Δt} = x_t + v_t·Δt         │
                  │             │      迭代 10 步                       │
                  │             │           │                          │
                  │             │           ▼                          │
                  │             │      action chunk                    │
                  │             │      [B, H, action_dim]              │
                  │             └──────────────────────────────────────┘
                  ▼
            语义向量序列 z
            进入 Stage 1 (RL Token 训练) 或 Stage 2 (作为 RL state 的一部分)

  ═══════════════════════════════════════════════════════════
  图示总结:

  ①② embed → ③ 共享 Transformer ─┬─ PaliGemma FFN → z  (观测理解)
                                 │
                                 └─ Action Expert FFN → ④ proj → ⑤ diffusion → action
                                    (adaRMS ← time_mlp)   └── Action Head ──┘
```

### 模块归属速查

| 步骤 | 模块 | 属于 |
|------|------|------|
| ① | `_preprocess_observation` | 预处理（共享） |
| ② | `embed_prefix` | 观测嵌入 |
| ② | `embed_suffix`（action_emb + time_mlp→adaRMS） | Action Head 输入准备 |
| ③ | 共享 Self-Attention | 共享 Transformer |
| ③ | PaliGemma FFN | 观测理解 |
| ③ | **Action Expert FFN**（RMSNorm 受 adarms_cond 调制） | **Action Head** |
| ④ | `action_out_proj` | **Action Head** |
| ⑤ | Diffusion Denoising | **Action Head** |

### PI0 与 PI0.5 的关键差异

> 来自 `pi0_config.py:28-30`：*"Pi05 has two differences from Pi0: the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix; the action expert uses adaRMSNorm to inject the flow matching timestep."*

| | PI0 | PI0.5（本图） |
|---|---|---|
| state 处理 | `state_proj` → 连续嵌入 → suffix_embs 第 0 号 token | 离散化 → 文本 → tokenize → lang_tokens → **prefix_embs** |
| state 形式 | 1 个 2048 维连续向量 | 若干个离散 token（例如 14 维 joint → 14 个 token） |
| 时间步注入 | 与 action concat → MLP 融合 | time_mlp → adarms_cond 调制 Expert RMSNorm |
| suffix_embs shape | `[B, 1+H, 2048]`（含 state token） | `[B, H, 2048]`（纯 action） |
| state token 切片 | ④ `suffix_out[:, -H:]` 去掉 state | 不需要 |
| adaRMS | 无 | Expert 模型 `[False, True]` |
| max_token_len | 48 | 200（因为 state 占了更多 token） |

**核心区别一句话：PI0 的 state 是 suffix 里的连续向量；PI0.5 的 state 是 prefix 里的离散文本 token。** 所以对于 prefix_out (z)，PI0 的 z 不含 state，PI0.5 的 z 含 state。

---

## 各阶段 Tensor 速查

| 阶段 | Tensor 名 | Shape | 含义 |
|------|----------|-------|------|
| ① 预处理 | `images` | `[B, N_img, P²·C]` | patch 化后的多视角图像 |
| | `lang_tokens` | `[B, L]` | tokenized 任务指令（PI0.5 含离散 state） |
| | `state` | `[B, action_dim]` | 本体感知（PI0: 入 suffix; PI0.5: 离散化入 lang_tokens） |
| ② prefix 嵌入 | `prefix_embs` | `[B, M, 2048]` | 图像+语言混合嵌入序列（PI0.5 含 state 信息） |
| | `prefix_pad_masks` | `[B, M]` | 有效位置标记 |
| | `prefix_att_masks` | `[B, M]` | 注意力类型标记 |
| ② suffix 嵌入 | `suffix_embs` | `[B, H, 2048]` (PI0.5) / `[B, 1+H, 2048]` (PI0) | 噪声动作嵌入（PI0 多一个 state token） |
| | `x_t` | `[B, H, action_dim]` | 扩散过程中的带噪动作 |
| ③ Transformer | `prefix_out` = `z` | `[B, M, 2048]` | PaliGemma FFN 输出（观测理解） |
| | `suffix_out` | `[B, H, 2048]` | Expert FFN 输出（动作倾向） |
| ④ 截取 | `suffix_out[:, -H:]` | `[B, H, 2048]` | 取动作对应位置 |
| ⑤ 动作投影 | `v_t` | `[B, H, action_dim]` | Flow Matching 速度场 |
| ⑥ 去噪 | `actions` | `[B, H, action_dim]` | 最终动作序列 |

---

## ③ Transformer 层的详细机制

### 双模型架构

```python
# paligemma_with_expert.py:234
models = [self.paligemma.language_model.model,   # 模型 0: 处理 prefix（观测）
          self.gemma_expert.model]                # 模型 1: 处理 suffix（动作）
```

### 单层流程（第 247-324 行）

```
inputs_embeds = [prefix_embs, suffix_embs]
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    模型 0: QKV 投影              模型 1: QKV 投影
    q_proj,k_proj,v_proj         q_proj,k_proj,v_proj
    (各自独立的 Linear 层)         (各自独立的 Linear 层)
            │                           │
            └──────────┬────────────────┘
                       ▼
              Q,K,V = cat(所有 Q, 所有 K, 所有 V) 沿 seq 维拼接
                       ▼
              共享 Self-Attention + RoPE
              - prefix ↔ prefix: 双向
              - suffix → prefix+suffix: 因果
                       ▼
              Split 回 prefix/suffix 各自长度
                       ▼
            各自 O-proj (独立 Linear)
                       ▼
                各自 + residual
            ┌──────────┴──────────┐
            ▼                     ▼
      PaliGemma FFN         Action Expert FFN
      (标准 Transformer FFN)  (动作专用 FFN, 不同权重)
            │                     │
            ▼                     ▼
        + residual            + residual
            │                     │
      prefix_hidden          suffix_hidden
```

### Attention 模式

由 `make_att_2d_masks` 的 `cumsum[j] <= cumsum[i]` 规则决定（token i 能看 token j 当且仅当 j 的 cumsum 不大于 i 的 cumsum）：

**PI0.5（本图默认）**：suffix 只有 action token，无 state token。

```
att_masks:
  prefix (M个)              action (H个)
  [0, 0, ..., 0,            1, 0, 0, ..., 0]
                            
cumsum:
  [0, 0, ..., 0,            1, 1, 1, ..., 1]
   └─ prefix ─┘             └── suffix ───┘
   (含 language + image      (纯 action, 因果组)
    + PI0.5 离散 state)

Attention 矩阵 (↓i 看 j→):
              prefix_j      action_j
  prefix_i   [  ✓  ]       [  ✗  ]      ← prefix 看不到 suffix
  action_i   [  ✓  ]       [✓causal]    ← suffix 看到全部 prefix + 因果
```

**PI0**：suffix 多一个 state token，attention mask 变为 `[1, 1, 0, ..., 0]`：

```
att_masks:
  prefix (M个)       state  action (H个)
  [0, 0, ..., 0,      1,    1, 0, 0, ..., 0]
                            
cumsum:
  [0, 0, ..., 0,      1,    2, 2, 2, ..., 2]
   └─ prefix ─┘       └──── suffix ────────┘
```

**关键结论：prefix 的 attention 始终只看 prefix（cumsum=0），suffix 侧的任何 token 都不会影响 prefix_out (z)。PI0.5 的 state 在 prefix 内部（离散文本 token），所以被 prefix attention 看到；PI0 的 state 在 suffix 侧，prefix 看不到。**

### 非对称信息流

Attention mask 的非对称设计导致 prefix_out 和 suffix_out 包含的信息完全不同。

**PI0.5（本图默认）**：state 在 prefix 内部。

```
         prefix_out              suffix_out
              ↑                       ↑
              │                       │
     PaliGemma FFN           Action Expert FFN
              ↑                       ↑
              │                       │
    ┌─ Self-Attention ──── Self-Attention ─┐
    │  prefix ↔ prefix      suffix 看到:    │
    │  (双向, 含 image +    ├─ prefix (全部, │
    │   language + 离散      │   含 state)   │
    │   state)              └─ 之前的 suffix │
    │                       │               │
    │  prefix 看不到:        │               │
    │  suffix 的全部         │               │
    └───────────────────────────────────────┘
              ↑                       ↑
        prefix_embs             suffix_embs
    (image+lang+离散state)    (action_emb only)
```

**PI0**：state 在 suffix 内部，作为单独的 suffix token。

```
    prefix_embs                 suffix_embs
    (image+lang)           (state+action+time)

    ┌─ Self-Attention ──── Self-Attention ─┐
    │  prefix ↔ prefix      suffix 看到:    │
    │  (双向, 只看自己)      ├─ prefix (全部) │
    │                       ├─ state token  │
    │                       └─ 之前的 suffix │
    └───────────────────────────────────────┘
```

| | prefix_out (z) | suffix_out |
|---|---|---|
| 包含 (PI0.5) | image + language + **离散 state** 的语义理解 | prefix 全部 + 之前的 suffix 的动作上下文 |
| 包含 (PI0) | image + language 的语义理解（**不含 state**） | prefix 全部 + **state token** + 之前的 suffix |
| 不受影响于 | 噪声动作 x_t, 扩散时间步（两种版本一致） | — |
| 为什么 | attention mask 阻断 prefix→suffix | attention mask 允许 suffix→prefix |
| 下游用途 | Stage 1 RL Token, Stage 2 RL state | action_out_proj → 去噪 → 最终动作 |

**设计意图**：prefix 充当"只读"的观测编码器，suffix 是动作预测器，两者通过单向 attention（suffix→prefix）传递场景信息。阻断反向（prefix→suffix）确保观测表示不被动作噪声污染。

### 当 suffix=None 时

```python
# embedding_extractor.py:91
inputs_embeds = [prefix_embs, None]

# paligemma_with_expert.py:252
for i, hidden_states in enumerate(inputs_embeds):
    if hidden_states is None:
        continue    # ← 模型 1 (Expert) 整层跳过
```

结果：只有 PaliGemma 的 QKV 投影 + FFN 工作。attention 只在 prefix token 之间做。`prefix_out` 的值与完整 forward 中相同（因为 prefix 不 attend suffix，FFN 也不混入 suffix 信息）。

---

## extract_embeddings vs 完整 forward 对比

| | extract_embeddings (Stage 1) | PI0 forward (训练/推理) |
|---|---|---|
| 代码位置 | `embedding_extractor.py:46-99` | `pi0_pytorch.py:317-374` |
| `inputs_embeds` | `[prefix_embs, None]` | `[prefix_embs, suffix_embs]` |
| 模型 0 (PaliGemma) | ✓ 运行 | ✓ 运行 |
| 模型 1 (Expert) | ✗ 跳过 (`continue`) | ✓ 运行 |
| 取哪个输出 | `prefix_out` = z | `suffix_out` |
| 是否过 `action_out_proj` | ✗ | ✓ |
| 产出 | `z` [B, M, 2048] | `v_t` [B, H, action_dim] |

---

## 为什么 Stage 1 不提取 suffix_out

1. **代码层**：`inputs_embeds=[prefix, None]` → Expert 模型无输入，不产生 suffix_out
2. **架构层**：即使有 suffix，prefix 的 FFN 也是 PaliGemma 原版，不走 Action Expert
3. **语义层**：suffix_out 依赖当前噪声动作 `x_t` 和扩散时间步 `time`，不是观测的稳定编码。Stage 1 要的是场景语义理解，不是特定噪声状态下的动作倾向
4. **PI0.5 特有**：state 已经离散化进入 prefix（lang_tokens），prefix_out 直接包含 state 信息，不需要从 suffix 获取

---

## suffix_out 详解

### H 是什么

`H` = `action_horizon`，即一次前向预测的动作块长度。PI0 中 H = 10，每个动作步间隔约 0.3s，一次前向覆盖约 3 秒的未来轨迹。

### suffix_embs 的构成

从 `pi0_pytorch.py:238-315` (`embed_suffix`) 追踪。**PI0.5** 无 state token，结构更简单：

```
      noisy_actions [B, H, action_dim]     timestep [B]
                │                                  │
                ▼                                  ▼
         action_in_proj                   sinusoidal embedding
         Linear(ad→2048)                   sin/cos → 2048-dim
                │                                  │
                ▼                                  ▼
         [B, H, 2048]                    time_mlp (SiLU → SiLU)
                │                                  │
                │                                  ▼
                │                           adarms_cond [B, 2048]
                │                           (调制 Expert RMSNorm,
                │                            不进入 token 序列)
                ▼
    suffix_embs = action_emb  [B, H, 2048]
```

**PI0** 多一个 state token，且时间步与 action 融合：

```
state [B, action_dim]     noisy_actions [B, H, action_dim]     timestep [B]
      │                          │                                  │
      ▼                          ▼                                  ▼
 state_proj                action_in_proj                   sinusoidal embedding
 Linear(ad→2048)           Linear(ad→2048)                   sin/cos → 2048-dim
      │                          │                                  │
      ▼                          ▼                                  ▼
 [B, 2048]                  [B, H, 2048]                    [B, 2048]
      │                          │                                  │
      │ unsqueeze                │                                  │ expand_as
      ▼                          ▼                                  ▼
 [B, 1, 2048]              [B, H, 2048]                     [B, H, 2048]
      │                          └──────────┬───────────────────────┘
      │                                     ▼
      │                          cat(dim=-1) → [B, H, 4096]
      │                                     │
      │                          action_time_mlp_in → SiLU → action_time_mlp_out
      │                                     │
      │                                     ▼
      │                              [B, H, 2048]
      │                                     │
      └──────────────┬──────────────────────┘
                     ▼
              cat(dim=1) → suffix_embs [B, 1+H, 2048]

              位置 0:      state token（本体感知编码）
              位置 1..H:   action_time token（噪声动作 + 扩散时间步融合）
```

### 为什么要过 Transformer

suffix_embs 此时只是**独立投影**——每个 token 孤立产生，不知道场景、不知道彼此。

#### 1. 场景感知：suffix → prefix attention

```
suffix token ──attention──→ prefix token 中编码的"杯子在桌上""请拿起杯子"
```

不看 prefix，suffix 完全不知道机器人面前是什么。attention 是视觉/语言信息流入动作预测的**唯一通路**。

#### 2. 时序一致性：suffix → suffix causal attention

```
动作步 0 ──→ 动作步 1 ──→ 动作步 2 ──→ ... ──→ 动作步 9
```

保证整条轨迹平滑连贯。如果各步独立预测，可能第一步前、第二步猛退——没有连续性。

#### 3. Action Expert FFN：动作空间专用变换

PaliGemma 原版 FFN 是通用视觉/语言知识。Action Expert FFN 是专门为去噪动作预测训练的独立权重——动作专用的归纳偏置。

### 对比：有 Transformer vs 没有

| | 没有 Transformer | 有 Transformer |
|---|---|---|
| suffix 见过 prefix 吗 | ❌ | ✓（suffix→prefix attention） |
| 动作步之间有关联吗 | ❌ 各自独立 | ✓（causal attention） |
| FFN | 无 | Action Expert FFN |
| 产物 | 噪声动作的独立投影（无用） | 融合场景+时序信息的去噪方向 `v_t` |

去掉 Transformer，suffix 通路等于盲猜——它根本不知道自己在什么场景里。

### suffix_out 的最终去路

**PI0.5**：suffix_embs 无 state token，直接投影。

```
suffix_embs [B, H, 2048]
      │
      ▼ (经过 Transformer)
suffix_out [B, H, 2048]
      │
      ▼ action_out_proj        ← Linear(2048 → action_dim)
 v_t [B, H, action_dim]        ← Flow Matching 速度场
      │
      ▼ Euler 积分             ← x_{t-Δt} = x_t + v_t · Δt，迭代 10 步
 actions [B, H, action_dim]    ← 最终动作序列
```

**PI0**：suffix_embs 多一个 state token，需先切掉。

```
suffix_embs [B, 1+H, 2048]
      │
      ▼ (经过 Transformer)
suffix_out [B, 1+H, 2048]
      │
      ▼ suffix_out[:, -H:]    ← 去掉 state token 位置 0
 [B, H, 2048]
      │
      ▼ action_out_proj
 v_t [B, H, action_dim]
      │
      ▼ Euler 积分
 actions [B, H, action_dim]
```

---

## 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `pi0_pytorch.py` | 62-187 | `embed_prefix` — 图像+语言嵌入 |
| `pi0_pytorch.py` | 238-315 | `embed_suffix` — 噪声动作+状态+时间嵌入 |
| `pi0_pytorch.py` | 317-374 | `forward` — 完整训练前向 |
| `pi0_pytorch.py` | 377-460 | `sample_actions` — 推理（含扩散去噪循环） |
| `paligemma_with_expert.py` | 225-324 | `PaligemmaWithExpert.forward` — 双模型 Transformer |
| `embedding_extractor.py` | 46-99 | `extract_embeddings` — Stage 1 的 prefix-only 前向 |
| `embedding_extractor.py` | 101-163 | `forward_joint` — 单次前向同时出 prefix_out + VLA loss |
