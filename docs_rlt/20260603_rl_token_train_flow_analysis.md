# RL Token Stage 1 训练调用链分析

分析 `trainer.train(vla, iter(data_loader), log_fn=rl_logger.log)` 的完整调用链。

---

## 入口

```python
print("Start to train RL token encoder-decoder...")
trainer.train(vla, iter(data_loader), log_fn=rl_logger.log)
```

---

## 1. `RLTokenTrainer.train()` — `rl_token_trainer.py:112-170`

- 默认 `alpha=0`，走 frozen-VLA 模式（**不**调用 `_setup_joint_training`）
- 循环从 dataloader 取 `(observations, actions)`，每步调用 `self.step(vla, observations, actions)`

```python
def train(self, vla, dataloader, log_fn=None):
    alpha = self.config.vla_finetune_alpha
    if self.joint:                              # alpha > 0
        self._setup_joint_training(vla)
    else:
        logger.info("Starting Stage 1 frozen-VLA training...")

    if self.config.resume_checkpoint:
        self.load(self.config.resume_checkpoint)

    pbar = tqdm(range(1, self.config.num_train_steps + 1), desc="Stage 1")
    for step_idx in pbar:
        observations, actions = next(dataloader)
        metrics = self.step(vla, observations, actions)   # ← 核心调用
        # ... 日志、保存 ...
```

---

## 2. `RLTokenTrainer.step()` — `rl_token_trainer.py:92-110`

- `alpha=0` → 调用 `self._step_frozen(vla, observations)`
- **注意 `actions` 没有传入 `_step_frozen`**（frozen 模式不需要动作标签）

```python
def step(self, vla, observations, actions):
    if self.joint:
        return self._step_joint(vla, observations, actions)
    return self._step_frozen(vla, observations)    # alpha=0 走这里
```

---

## 3. `_step_frozen()` — `rl_token_trainer.py:258-281` ← 核心训练步

```python
def _step_frozen(self, vla, observations):
    self.model.train()
    observations = _obs_to_device(observations, self.device)

    with torch.no_grad():                          # VLA 完全冻结，不计算梯度
        z, pad_mask = vla.extract_embeddings(observations)

    z = z.to(self.device)
    pad_mask = pad_mask.to(self.device)

    loss, _z_rl, _z_hat = self.model(z, pad_mask) # 只训练 encoder-decoder

    self.optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), ...)
    self.optimizer.step()
    self.scheduler.step()
```

**关键点**：
- VLA 完全在 `torch.no_grad()` 下运行，不占显存计算图
- `_obs_to_device`（:336-354）递归将 `Observation` 中的所有 tensor 移到 GPU
- 反向传播**只更新** `RLTokenModel`（encoder + decoder）的参数

### `_obs_to_device()` — `rl_token_trainer.py:336-354`

```python
def _obs_to_device(obs, device):
    if isinstance(obs, Observation):
        return Observation(
            images={k: v.to(device) for k, v in obs.images.items()},
            image_masks={k: v.to(device) for k, v in obs.image_masks.items()},
            state=obs.state.to(device),
            tokenized_prompt=obs.tokenized_prompt.to(device) if ... else None,
            tokenized_prompt_mask=obs.tokenized_prompt_mask.to(device) if ... else None,
            token_ar_mask=obs.token_ar_mask.to(device) if ... else None,
            token_loss_mask=obs.token_loss_mask.to(device) if ... else None,
        )
    if isinstance(obs, dict):
        return {k: _obs_to_device(v, device) for k, v in obs.items()}
    if isinstance(obs, torch.Tensor):
        return obs.to(device)
    return obs
```

---

## 4. `vla.extract_embeddings()` — `vla_wrapper.py:162-172`

直接委托给 `self.extractor.extract_embeddings(observations)`：

```python
def extract_embeddings(self, observation):
    return self.extractor.extract_embeddings(observation)
```

---

## 5. `EmbeddingExtractor.extract_embeddings()` — `embedding_extractor.py:46-99` ← VLA 前向

这是实际执行 VLA 模型前向的地方，分为 6 个步骤：

```python
@torch.no_grad()
def extract_embeddings(self, observation):
    # Step A: 预处理 → 图像、语言 token、state
    images, img_masks, lang_tokens, lang_masks, _state = \
        self.pi0._preprocess_observation(observation, train=False)

    # Step B: 将图像 + 语言嵌入为 prefix embeddings
    prefix_embs, prefix_pad_masks, prefix_att_masks = \
        self.pi0.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    #   prefix_embs: [B, M, 2048]       ← 视觉 token + 语言 token 的嵌入
    #   prefix_pad_masks: [B, M] bool   ← True = 有效 token
    #   prefix_att_masks: [B, M]        ← 0 = prefix, 1 = suffix

    # Step C: 构建 2D/4D attention mask
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    att_2d_masks_4d = self.pi0._prepare_attention_masks_4d(prefix_att_2d_masks)

    # Step D: 强制 eager attention（不用 SDPA）
    self.pi0.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    # Step E: prefix-only 前向 — 关键：只跑 PaliGemma LM，不跑 diffusion
    (prefix_out, _suffix_out), _kv_cache = \
        self.pi0.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],     # ← suffix = None
            use_cache=False,
        )

    # Step F: 返回 [B, M, 2048] 的后 transformer 嵌入
    z = prefix_out.to(dtype=torch.float32)
    return z, prefix_pad_masks                     # [B, M, 2048], [B, M] bool
```

**核心**：`inputs_embeds=[prefix_embs, None]` — suffix 传 `None`，意味着**只跑 PaliGemma 的 prefix 部分**，不做 diffusion 采样。这是为什么它比 `sample_actions` 快很多的原因。

---

## 6. `RLTokenModel.forward()` — `rl_token.py:180-210` ← 训练目标

```python
def forward(self, z, pad_mask):
    z = z.detach()                               # stop-grad（双重保险）
    z_rl = self.encoder(z, pad_mask)             # [B, M, 2048] → [B, 2048]
    z_hat = self.decoder(z_rl, z, pad_mask)      # [B, 2048] → [B, M, 2048]

    # 掩码 MSE：只计算有效（非 padding）位置
    mse = (z_hat - z).pow(2).mean(dim=-1)        # [B, M]
    masked_mse = mse * pad_mask.float()           # padding 位置置零
    num_valid = pad_mask.float().sum()
    loss = masked_mse.sum() / num_valid.clamp(min=1.0)
    return loss, z_rl, z_hat
```

### Encoder — `rl_token.py:47-75`

在 `z` 末尾拼接一个可学习的 `e_rl token`（`nn.Parameter`），过 TransformerEncoder，取最后位置输出作为 `z_rl`：

```python
def forward(self, z, pad_mask):
    B = z.shape[0]
    e_rl = self.e_rl.expand(B, -1, -1)            # [1, 1, D] → [B, 1, D]
    tokens = torch.cat([z, e_rl], dim=1)          # [B, M+1, D]
    rl_mask = torch.ones(B, 1, dtype=torch.bool, device=z.device)
    extended_pad_mask = torch.cat([pad_mask, rl_mask], dim=1)

    ignore_mask = ~extended_pad_mask               # True = IGNORE (pytorch 约定)
    out = self.transformer(tokens, src_key_padding_mask=ignore_mask)
    z_rl = out[:, -1, :]                           # 取 RL token 位置 → [B, D]
    return z_rl
```

### Decoder — `rl_token.py:111-151`

Teacher-forcing 重建，输入 `[z_rl, z_1, ..., z_{M-1}]`，causal mask 确保 autoregressive，cross-attend 到 `z_rl`，最后 `h_phi` 线性投影：

```python
def forward(self, z_rl, z, pad_mask):
    tgt = torch.cat([z_rl.unsqueeze(1), z[:, :-1, :]], dim=1)  # [B, M, D]
    M = tgt.shape[1]
    causal_mask = nn.Transformer.generate_square_subsequent_mask(M, ...)
    memory = z_rl.unsqueeze(1)                                   # [B, 1, D]
    tgt_key_padding_mask = ~pad_mask

    out = self.transformer(tgt, memory, tgt_mask=causal_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask)
    z_hat = self.h_phi(out)                                      # Linear(2048→2048)
    return z_hat                                                  # [B, M, D]
```

---

## 完整数据流图

```
dataloader → (observations, actions)
                  │
                  ▼
    _step_frozen(observations)              ← actions 被忽略（frozen 模式）
                  │
                  ▼  [torch.no_grad()]
    EmbeddingExtractor.extract_embeddings(obs)
       │
       ├─ A. _preprocess_observation
       │      → images, img_masks, lang_tokens, lang_masks, state
       │
       ├─ B. embed_prefix
       │      → prefix_embs [B, M, 2048]
       │      → prefix_pad_masks [B, M] (bool)
       │      → prefix_att_masks [B, M]
       │
       ├─ C-D. make_att_2d_masks + prepare_4d
       │      → att_2d_masks_4d
       │
       ├─ E. paligemma_with_expert.forward
       │      inputs_embeds=[prefix_embs, None]   ← suffix = None 跳过 diffusion
       │      → prefix_out [B, M, 2048]
       │
       └─ F. to float32
              → z [B, M, 2048], pad_mask [B, M]
                  │
                  ▼
    RLTokenModel.forward(z, pad_mask)
       │
       ├─ z = z.detach()                        stop-grad
       │
       ├─ Encoder: token [+e_rl] → Transformer → z_rl [B, 2048]
       │
       ├─ Decoder: [z_rl, z_1..z_{M-1}] → causal Transformer + h_phi → z_hat [B, M, 2048]
       │
       └─ Masked MSE: mean((z_hat - z)², dim=-1) × pad_mask / sum(pad_mask)
              → loss (标量)
                  │
                  ▼
    loss.backward()           ← 只更新 encoder + decoder 的梯度
    clip_grad_norm_()
    optimizer.step()          ← 只更新 RLTokenModel 参数
    scheduler.step()
```

---

## 关键结论

| 环节 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 训练循环入口 | `rl_token_trainer.py` | 112-170 | `train()` 循环取数据 |
| 单步调度 | `rl_token_trainer.py` | 92-110 | `step()` 根据 alpha 选 frozen/joint |
| Frozen 步 | `rl_token_trainer.py` | 258-281 | VLA no_grad + encoder-decoder 训练 |
| VLA 前向 | `embedding_extractor.py` | 46-99 | prefix-only PaliGemma，不做 diffusion |
| VLA 封装 | `vla_wrapper.py` | 162-172 | 委托给 extractor |
| RL Token 模型 | `rl_token.py` | 180-210 | encoder + decoder + masked MSE |
| Encoder | `rl_token.py` | 47-75 | 拼接 e_rl → Transformer → z_rl |
| Decoder | `rl_token.py` | 111-151 | teacher-forcing causal 重建 |
