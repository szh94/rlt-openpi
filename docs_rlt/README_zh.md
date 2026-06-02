# RLT-OpenPI 中文使用指南

基于 **RL Token: Bootstrapping Online RL with Vision-Language-Action Models** (Xu et al., Physical Intelligence) 的非官方实现，构建在 [OpenPI](https://github.com/Physical-Intelligence/openpi) 公开检查点之上。

论文：https://pi.website/research/rlt

![RLT 方法概览](docs/rlt_overview.png)

> **关于示例环境。** 本文中的命令、`exp/` 下的脚本以及硬件相关章节均以 **Franka Panda + DROID + Oculus VR** 为例——这是本仓库开发时使用的硬件平台。RLT 本身与环境无关：环境、干预管理器、数据变换和 VLA 检查点均可插拔。如果使用不同的机器人、模拟器、数据集或 VLA 配置，请替换对应的 `--env-factory`、`--intervention-factory`、`--data-transforms-fn` 和 `--vla-config-name` 参数。

---

## 仓库结构

```
src/rlt_openpi/
  models/          RL Token 编码器/解码器、Actor（残差策略）、TwinQCritic
  training/        Stage 1 + Stage 2 训练器、配置、回放缓冲区、TD3 工具函数
  vla/             OpenPI VLA 封装、嵌入提取钩子
  rollout/         RolloutWorker、环境/干预/奖励基类、工厂函数
  envs/franka/     Franka+DROID 环境工厂、VR 干预管理器示例
  policies/franka/ 三摄像头 DROID 数据变换示例
  utils/           检查点读写、wandb 日志、终端 UI
scripts/
  train_rl_token.py    Stage 1 入口
  train_online_rl.py   Stage 2 入口
  evaluate.py          统一评估（自动检测 Stage 1 / Stage 2 检查点）
exp/
  stage1.sh, stage2.sh, eval_vla.sh, eval_full.sh   示例运行命令
tests/               模型、缓冲区、训练循环的单元测试
```

---

## 安装

### 前置条件

- Python ≥ 3.11
- conda（推荐使用 [Miniconda](https://docs.anaconda.com/miniconda/)）
- 如果需要运行 Stage 2 真实机器人实验：Linux 系统（DROID 栈和 ZED SDK 需要）

---

### 方式一：在线安装（能访问 GitHub）

```bash
git clone https://github.com/yknxh/rlt-openpi.git
cd rlt-openpi
bash setup_env.sh        # 创建名为 'rl_token' 的 conda 环境
conda activate rl_token
```

脚本会自动从 GitHub 安装 OpenPI。可以指定自定义环境名：`bash setup_env.sh myenvname`。

---

### 方式二：脱机部署（无网络，推荐）

如果你的机器无法访问 GitHub（完全离线环境），按以下步骤操作。

#### 第 1 步：准备源码

在有网络的机器上，将两个仓库打包：

```bash
# 1. 克隆 rlt-openpi（本仓库）
git clone https://github.com/yknxh/rlt-openpi.git
tar czf rlt-openpi.tar.gz rlt-openpi

# 2. 克隆 openpi（依赖基础库）
git clone https://github.com/Physical-Intelligence/openpi.git
tar czf openpi.tar.gz openpi
```

也可以通过 U 盘、`scp`、内部 Git 服务等方式将两个仓库目录传送到目标机器。

#### 第 2 步：配置 setup_env.sh

目标机器上，解压两个仓库：

```bash
tar xzf rlt-openpi.tar.gz -o /path/to/rlt-openpi
tar xzf openpi.tar.gz -o /path/to/openpi
```

编辑 `setup_env.sh`，修改 `OPENPI_DIR` 默认路径指向本地 openpi：

```bash
# 找到这一行（约第 50 行）：
OPENPI_DIR="${OPENPI_DIR:-/home/path/to/openpi}"   # ← 修改此路径！

# 改为你的真实路径，例如：
OPENPI_DIR="${OPENPI_DIR:-/home/user/code/openpi}"   # ← 改为你的真实路径
```

> 或者不修改脚本，每次运行时通过环境变量传入：
> ```bash
> OPENPI_DIR=/home/user/code/openpi bash setup_env.sh
> ```

#### 第 3 步：运行安装

```bash
cd /path/to/rlt-openpi
bash setup_env.sh        # 使用 OPENPI_DIR 中的本地 openpi，跳过 GitHub
conda activate rl_token
```

安装流程：
1. 创建 conda 环境（如已存在则跳过）
2. 安装 `uv`（如环境已存在则跳过）
3. 使用本地 openpi 路径安装（如环境已存在则跳过）
4. 安装 rlt-openpi
5. 修补 transformers

#### 更新已有环境

如果 conda 环境已存在（例如已装好 openpi），脚本会自动检测并跳过 conda 创建和 openpi 安装，只更新 rlt-openpi 自身代码和 transformers 补丁：

```bash
cd /path/to/rlt-openpi     # 先更新代码（重新拉取或覆盖目录）
bash setup_env.sh          # 只装 rlt-openpi，不碰已有的 openpi
```

这对需要频繁更新 rlt-openpi 代码但不想每次重装 openpi 的场景特别有用。

---

### 方式三：机器人机器安装（带 DROID 硬件栈）

运行 Stage 2 需要 [DROID](https://github.com/droid-dataset/droid) 机器人栈。添加 `--robot` 并设置 `DROID_DIR`：

```bash
DROID_DIR=/path/to/droid bash setup_env.sh --robot
conda activate rl_token
```

这会额外安装：
- DROID 遥操作栈
- Oculus 控制器驱动
- ZED 摄像头绑定
- OpenCV/protobuf 兼容性修复

需要 ZED SDK 位于 `/usr/local/zed`（如果不存在会跳过）。

---

## 模型检查点

### 下载

从 [OpenPI 模型库](https://github.com/Physical-Intelligence/openpi#checkpoints) 下载检查点（托管在 GCS `gs://openpi-assets/checkpoints/`）。

### 转换为 PyTorch 格式

下载的 JAX/Orbax 需要转换为 PyTorch 格式：

```bash
python scripts/tools/convert_jax_to_pytorch.py \
    --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
    --config-name pi05_droid_finetune \
    --output-path checkpoints/pi05_droid_pytorch
```

转换后得到一个 `model.safetensors` 文件。在训练命令中通过 `--train.vla-checkpoint-dir` / `--vla-checkpoint-dir` 指向它。

---

## 数据准备

Stage 1 训练使用 [LeRobot](https://github.com/huggingface/lerobot) 格式的演示数据。数据集由 `repo_id` 定位，LeRobot 会解析到 `$HF_LEROBOT_HOME/<repo_id>/`（默认 `~/.cache/huggingface/lerobot/<repo_id>/`）。

使用**本地数据集**时，在启动训练前设置环境变量：

```bash
export HF_LEROBOT_HOME="/path/to/your/data"
```

例如，如果数据集位于 `/data/my_task_lerobot/`，则设置 `HF_LEROBOT_HOME=/data`，传入 `--repo-id my_task_lerobot`。

原始演示数据需要先转换为 LeRobot 格式并计算归一化统计量，详细步骤参考 [OpenPI 数据准备脚本](https://github.com/Physical-Intelligence/openpi/tree/main/scripts/data_prep)。

---

## Stage 1：训练 RL Token

训练一个小型编码器-解码器，将 VLA 的内部逐 token 嵌入 `z_{1:M}` 压缩为单一的 **RL token** `z_rl`，通过 LeRobot 演示数据集上的掩码 MSE 重构损失进行训练。

- `--train.vla-finetune-alpha 0`：VLA 冻结（仅训练编码器-解码器）
- `--train.vla-finetune-alpha > 0`：联合微调 VLA（损失 = `L_ro + α · L_vla`）

### 示例命令

```bash
CHECKPOINT_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors"

python scripts/train_rl_token.py \
    --train.vla-config-name pi05_droid_finetune \
    --train.vla-checkpoint-dir "$CHECKPOINT_DIR" \
    --train.vla-finetune-alpha 1.0 \
    --train.batch-size 32 \
    --train.num-train-steps 5000 \
    --repo-id local/stack_the_blocks_100 \
    --data-transforms-fn rlt_openpi.policies.franka.config.three_camera_droid
```

请根据你的数据集和机器人替换 `--repo-id`、`--data-transforms-fn` 以及 VLA 配置/检查点。

### 关键参数

| 参数 | 说明 |
| --- | --- |
| `--train.vla-finetune-alpha` | `0.0` = 冻结 VLA（仅训练编码器-解码器）；`> 0` = 联合微调 |
| `--train.num-train-steps` | 训练步数（默认 5000） |
| `--train.batch-size` | 批次大小（默认 32） |
| `--train.warmup-steps` | 学习率线性预热步数（默认 500） |
| `--train.resume-checkpoint` | 从之前的 `rl_token_step<N>.pt` 恢复训练 |
| `--train.save-every` | 检查点保存间隔（默认 1000） |
| `--repo-id` | LeRobot 数据集 ID |
| `--data-transforms-fn` | 数据变换工厂函数的导入路径（如 `rlt_openpi.policies.franka.config.three_camera_droid`） |

输出保存在 `checkpoints/rl_token/<run_name>/rl_token_step<N>.pt`，`run_name` 默认为 `run_YYYYMMDD_HHMMSS`。

---

## Stage 2：在线强化学习

冻结 VLA + RL Token 编码器，训练轻量级 **Actor** 和 **Twin-Q Critic**。

- Actor 以 `(z_rl, VLA 参考动作块)` 为条件，输出 VLA 动作的**残差修正**（最后一层零初始化，因此初始策略 = VLA）
- 先运行 **Warmup 阶段**：用基础 VLA 策略收集 episodes
- 然后交替执行：rollout 收集 → 离策略 TD3 更新（UTD = 5）
- BC 正则化项将 Actor 拉向 VLA 参考动作 + 参考动作 Dropout
- 人类通过键盘提供稀疏的成功/失败/进展奖励，可以通过 VR 控制器接管机器人

### 示例硬件（Franka + DROID + VR）

- Franka Panda 机器人（DROID 栈驱动，关节速度控制）
- 三台 ZED 摄像头（匹配 `three_camera_droid` 数据变换布局）
- Oculus VR 控制器（集成在 `src/rlt_openpi/envs/franka/intervention.py`）

如果使用其他机器人或模拟器，请实现自己的 `make_env` / `make_intervention` 函数并传入 `--env-factory` 和 `--intervention-factory`。

### 示例命令

```bash
python scripts/train_online_rl.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --intervention-factory rlt_openpi.envs.franka.intervention.make_vr_intervention \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir checkpoints/pi05_droid_pytorch/model.safetensors \
    --rl-token-checkpoint checkpoints/rl_token/rl_token_step3000.pt \
    --task-prompt "stack the three blocks on the tray" \
    --warmup-steps 250 \
    --chunk-length 5 \
    --max-episode-chunks 150 \
    --save-dir checkpoints/online_rl
```

### 关键参数

| 参数 | 说明 |
| --- | --- |
| `--env-factory` | 环境工厂函数的导入路径 |
| `--intervention-factory` | 干预管理器的导入路径（可选） |
| `--rl-token-checkpoint` | Stage 1 检查点路径 |
| `--task-prompt` | 传入 VLA 的文本指令 |
| `--warmup-steps` | 纯 VLA 数据收集的步数（预热阶段） |
| `--chunk-length` | 动作块长度 C |
| `--max-episode-chunks` | 每个 episode 的最大块数（安全限制） |
| `--warmup-buffer` | 加载之前保存的预热缓冲区，跳过预热阶段 |
| `--resume-checkpoint` | 恢复 Stage 2 训练 |

其他默认值：`gamma=0.99`、`tau=0.005`、`utd_ratio=5`、`bc_regularizer_beta=0.5`、`actor_noise_sigma=0.1`、`ref_action_dropout=0.5`、`batch_size=256`。

---

## 人工交互控制（Stage 2）

### 键盘奖励

`src/rlt_openpi/rollout/reward.py` 中的非阻塞按键监听器，支持单键输入（无需回车）：

| 按键 | 含义 |
| --- | --- |
| `s` 或 `Space` | 成功 — 奖励 `+1.0`，结束 episode |
| `f` | 失败 — 奖励 `0.0`，结束 episode |
| `p` | 进展 — 奖励 `+0.5`，episode 继续 |

成功/失败信号是锁存的（一旦触发持续返回）；进展信号每次读取后消耗。无 TTY 的无头运行会自动回退到行缓冲输入。

### VR 干预

`src/rlt_openpi/envs/franka/intervention.py`（Franka 示例）。操作员通过 VR 控制器接管机器人时，`VRInterventionManager` 会覆盖当前动作块；执行的人类动作写入回放缓冲区，下游的 BC 正则化项将 Actor 拉向人类动作。在其他硬件上，请提供自己的 `InterventionManager` 子类。

### 终端 UI

`src/rlt_openpi/utils/display.py` 使用 [`rich`](https://github.com/Textualize/rich) 库显示预热进度、每 episode 统计和操作员指令。

---

## 评估

`scripts/evaluate.py` 自动检测检查点是 Stage 1（仅 VLA）还是 Stage 2（VLA + RL Token + Actor），并运行相应的 rollout 循环。

```bash
python scripts/evaluate.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir checkpoints/pi05_droid_pytorch/model.safetensors \
    --rl-token-checkpoint checkpoints/rl_token/rl_token_step5000.pt \
    --checkpoint checkpoints/online_rl/run_latest/online_rl_ep100.pt \
    --task-prompt "stack the three blocks on the tray" \
    --num-episodes 50
```

结果（每 episode 的成功率、奖励、长度）以 JSON 格式保存在运行目录下。

---

## 测试

```bash
pytest tests/
```

覆盖：RL Token 编码器/解码器维度、VLA 嵌入提取钩子、Actor + TwinQCritic 前向/反向传播、回放缓冲区、Stage 2 训练器端到端冒烟测试。

---

## 状态与限制

这是一个**非官方实现**，与 Physical Intelligence 无关。目前仍在活跃开发中，可能仍有缺陷。

### 已实现

- Stage 1 RL Token 训练（冻结 VLA 和联合微调两种模式）
- Stage 2 TD3 在线 RL（双 Q、延迟 Actor、Polyak 目标、BC 正则化、参考动作 Dropout、Chunk Stride 子采样）
- Franka/DROID 环境包装器（三台 ZED 摄像头）
- Oculus VR 干预（修正动作写入缓冲区）
- 键盘人工奖励整形
- 终端 UI（预热 + rollout 进度）
- 自动检测 Stage 1 / Stage 2 检查点的评估脚本

### 尚未验证

论文中四个任务（螺丝安装、扎带固定、以太网插入、充电器插入）的端到端验证尚未完成。开发过程中仅测试了 Franka + DROID + VR 方案；其他机器人、模拟器和 VLA 配置原则上支持但未经过测试。

---

## 引用

```
@article{xu2025rltoken,
  title   = {RL Token: Bootstrapping Online RL with Vision-Language-Action Models},
  author  = {Xu, Charles and Springenberg, Jost Tobias and Equi, Michael and Amin, Ali
             and Esmail, Adnan and Levine, Sergey and Ke, Liyiming},
  year    = {2025},
  journal = {Physical Intelligence},
  url     = {https://pi.website/research/rlt}
}
```

VLA 骨架和训练基础设施来自 [OpenPI](https://github.com/Physical-Intelligence/openpi)。RLT 方法归论文作者所有，本实现中的任何错误均由本人负责。
