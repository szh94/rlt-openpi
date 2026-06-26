# Stage 2 训练 — 人工操作指南

本文档说明 Stage 2 训练过程中需要人工执行的操作，以及每项操作对应的代码模块。不涉及算法原理。

---

## 一、启动前一次性准备

| # | 操作 | 对应模块 |
|---|------|---------|
| 1 | 确认 Stage 1 checkpoint 存在 | `--rl-token-checkpoint` 参数 → `load_rl_token_model()` (`utils/checkpoint.py`) |
| 2 | 确认 VLA 权重已下载 | `--vla-checkpoint-dir` 参数 → `VLAWrapper` (`vla/vla_wrapper.py`) |
| 3 | 配置 `example/keyPara.sh`（checkpoint 路径等） | 所有示例脚本共用 |
| 4 | 清空机器人工作空间，确保安全 | `RobotEnv` (`envs/robot_base/robot_env.py`) |
| 5 | 启动训练脚本 | `scripts/train_online_rl.py` |

**推荐方式**：直接使用 `example/` 目录下的 shell 脚本启动，脚本已包含完整的硬件参数配置。例如：

```bash
bash example/stage2_alicd.sh       # Alicia-D 机械臂
bash example/stage2_franka.sh      # Franka 机械臂（如有）
bash example/stage2_aloha.sh       # ALOHA 双臂（如有）
```

也可以直接调用 Python 脚本。不同机械臂通过 `--env-factory` 切换：

| 机械臂 | `--env-factory` |
|--------|----------------|
| Alicia-D | `rlt_openpi.envs.alicd.env_factory.make_alicd_env` |
| Franka | `rlt_openpi.envs.franka.env_factory.make_franka_env` |
| ALOHA 双臂 | `rlt_openpi.envs.aloha.env_factory.make_aloha_env` |

通用命令模板：

```bash
python scripts/train_online_rl.py \
    --env-factory <按上表选择> \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir <VLA 权重目录> \
    --rl-token-checkpoint <Stage1 checkpoint .pt 路径> \
    --save-dir <Stage2 输出目录> \
    --task-prompt "pick up the cup" \
    --action-dim <机械臂动作维度> \
    --chunk-length 10 \
    --warmup-steps 150 \
    --max-episode-chunks 150 \
    --env-kwargs '<硬件参数 JSON>'
```

其中 `--env-kwargs` 按机械臂类型传递不同的硬件参数（端口、相机 ID、控制频率等），具体写法参考对应的示例脚本。

---

## 二、训练中需要人工操作的节点

Stage 2 训练分为两个阶段：Warmup（预热）和 Online RL（在线强化学习）。两个阶段的人工操作完全相同，只有 3 种操作：

### 操作速查表

| 操作 | 按键 | 时机 | 对应模块 | 备注 |
|------|:---:|------|---------|------|
| 摆放场景 + 继续 | **Enter** | 每个 episode 开始 | `RobotEnv.reset()` → `robot_env.py` | **必须做**，程序会阻塞等待 |
| 标注奖励 | **S / F / P** | 机器人执行动作时 | `HumanReward.check()` → `reward.py` | 非阻塞，按需操作 |

---

### 操作 1：Enter 键 — 开始 episode

**触发场景**：每个 episode 开始时，机器人已自动复位到初始位姿，终端提示等待。

**你要做的**：
1. 将任务物品摆放好（如把笔放回起始位置）
2. 确认机器人周围安全
3. 按 **Enter** 键

**对应模块**：`robot_env.py` 中 `RobotEnv.reset()` 方法。程序在 `input("")` 处**阻塞等待**，没有超时，按 Enter 前不会继续。

**频率**：每个 episode 1 次，约每 1-2 分钟。

---

### 操作 2：S / F / P 键 — 标注奖励信号

**触发场景**：机器人执行动作的过程中（每个 action step 之后），终端会检测键盘输入。

**你要做的**：根据观察到的情况按下对应按键：

| 按键 | 含义 | 按下去之后 | 何时按 |
|:---:|------|-----------|--------|
| **S** 或 **Space** | 成功 | Episode **立即终止**，信号**锁定**（后续步持续返回成功） | 任务明确完成时 |
| **F** | 失败 | Episode **立即终止**，信号**锁定** | 任务失败或出现不安全动作时 |
| **P** | 进展 | Episode **继续**，信号**消耗**（只生效一次） | 机器人有实质推进时（如抓住物体、到达关键位置） |
| 不按 | 无信号 | reward = 0，episode 继续 | 默认状态 |

**对应模块**：`reward.py` 中 `HumanReward.check()` 方法。使用**非阻塞轮询**（`select` timeout=0），在机器人控制循环的每个 step 末尾调用。

**重要提示**：
- S/F 一旦按下无法撤销，确认清楚再按
- **P 键可以按多次**（每个 episode 内），尽量在机器人有实质进展时按，对训练质量帮助最大
- 按键检测窗口很窄（~67ms），按下去不要犹豫

**频率**：S/F 每个 episode 至多 1 次，P 可按任意次。

---

## 三、完整 Episode 操作流程

```
┌─ Episode 开始 ─────────────────────────────────────────────┐
│                                                              │
│  1. 机器人自动复位                                           │
│  2. 终端显示 "Episode N ready"                               │
│  3. 【你】摆放场景，按 Enter ──────→ robot_env.reset()       │
│                                                              │
├─ Chunk 循环（每个 episode 约 10-20 个 chunk）───────────────┤
│                                                              │
│  每个 chunk:                                                 │
│    ├─ [自动] VLA 推理 + Actor 推理                           │
│    ├─ [自动] 执行 C=10 步动作                                │
│    │    └─ 每步: 【你·可选】按 S/F/P ─→ reward.py           │
│    └─ [自动] 数据存入 ReplayBuffer                           │
│                                                              │
├─ Episode 结束 (done=True 或按了 S/F) ────────────────────────┤
│                                                              │
│  [自动] TD3 梯度更新 × 5 次（无需你做任何事）                 │
│                                                              │
└─ 下一 Episode 自动开始 ──────────────────────────────────────┘
```

---

## 四、你需要做的事 vs 不需要你做的事

### 你需要做的（全部人工介入）

| 操作 | 按键 | 对应模块 | 代码文件 |
|------|:---:|---------|---------|
| 开始每个 episode | Enter | `RobotEnv.reset()` | `robot_env.py` |
| 标注成功 | S / Space | `HumanReward.check()` | `reward.py` |
| 标注失败 | F | `HumanReward.check()` | `reward.py` |
| 标注进展 | P | `HumanReward.check()` | `reward.py` |

### 不需要你做的（全自动）

| 自动步骤 | 对应模块 |
|---------|---------|
| VLA 提取视觉特征 + 参考动作 | `VLAWrapper` (`vla_wrapper.py`) + `RLTokenModel` (`rl_token.py`) |
| Actor 推理输出动作 | `Actor` (`actor.py`) + `RolloutWorker._get_actor_action()` (`rollout_worker.py`) |
| 执行动作 | `RobotEnv.step()` (`robot_env.py`) |
| 数据存入缓冲区 | `ReplayBuffer.add()` (`replay_buffer.py`) |
| TD3 网络更新（Critic + Actor + Target） | `OnlineRLTrainer._update_step()` (`online_rl_trainer.py`) |
| Checkpoint 保存（每 50 episode） | `OnlineRLTrainer.save()` (`online_rl_trainer.py`) |
| 日志输出 | `Logger` (`logging.py`) |

---

## 五、安全保护（自动）

以下安全机制**不需要你操作**，但你需要知道它们存在：

| 保护机制 | 作用 | 对应模块 |
|---------|------|---------|
| Deviation Cap | Actor 输出被限制在 VLA 参考 ±0.3 内 | `rollout_worker.py` → `_get_actor_action()` |
| 动作 Clamp | 最终动作裁剪到 [-1, 1] | `rollout_worker.py` → `_get_actor_action()` |
| Episode 超时 | 超过 `max_episode_chunks`（默认 150）后强制终止 | `robot_env.py` → `step()` |

---

## 六、操作技巧

1. **多用 P 键**：大多数 chunk 没有奖励信号，P 键注入的中间奖励是训练信号最有效的来源
2. **S/F 按准**：一旦按下无法撤销，务必确认状态再按
3. **观察终端日志**：关注 `total_reward`（episode 得分）、`success`（是否成功）
4. **决定停止时机**：连续多个 episode 成功率稳定、或达到 `max_env_steps` 时停止
