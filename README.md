# RLT-OpenPI

A research implementation of **RL Token: Bootstrapping Online RL with Vision-Language-Action Models** (Xu et al., Physical Intelligence) built on top of [OpenPI](https://github.com/Physical-Intelligence/openpi)'s public checkpoints.

Paper: https://pi.website/research/rlt

> **Note on the example environment.** The end-to-end commands, scripts under `exp/`, and the hardware sections below use a **Franka Panda + DROID + Oculus VR** setup as a concrete example — that's the rig this repo was developed against. RLT itself is environment-agnostic: the env, intervention manager, data transforms, and VLA checkpoint are all pluggable. If you are running against a different robot, simulator, dataset, or VLA configuration, substitute your own `--env-factory`, `--intervention-factory`, `--data-transforms-fn`, and `--vla-config-name` accordingly.

---

## Repository Layout

```
src/rlt_openpi/
  models/          RLTokenEncoder/Decoder, Actor (residual), TwinQCritic
  training/        Stage 1 + Stage 2 trainers, configs, replay buffer, TD3 utils
  vla/             OpenPI VLA wrapper, embedding extractor hooks
  rollout/         RolloutWorker, base env/intervention/reward interfaces, factory
  envs/franka/     Example Franka+DROID env factory, VR intervention manager
  policies/franka/ Example three-camera DROID data transforms
  utils/           Checkpoint I/O, wandb logger, rich terminal UI
scripts/
  train_rl_token.py    Stage 1 entry point
  train_online_rl.py   Stage 2 entry point
  evaluate.py          Unified Stage 1 / Stage 2 evaluation
exp/
  stage1.sh, stage2.sh, eval_vla.sh, eval_full.sh   Example run commands
tests/               Unit tests for models, buffers, and training loop
```

---

## Installation

Two install paths are supported. Pick based on which machine you're setting up.

Common requirements:

- Python ≥ 3.11
- PyTorch 2.7.1 with CUDA
- OpenPI (pinned to a specific GitHub rev — see `pyproject.toml`)

### Path A: `uv` (development / Stage 1 training box)

Use this on any plain GPU box where you just need to train the RL token or evaluate checkpoints — no real-robot dependencies involved.

```bash
git clone https://github.com/yknxh/rlt-openpi.git
cd rlt-openpi
uv sync
source .venv/bin/activate   # so the exp/ scripts can call `python` directly
```

### Path B: conda (robot machine with DROID)

On the robot host, Stage 2 needs the [DROID](https://github.com/droid-dataset/droid) teleop stack, the Oculus reader, ZED camera bindings, and an opencv/protobuf fixup — none of which fit cleanly into a pure-`uv` project. A helper script wires everything up inside a conda env:

```bash
git clone https://github.com/yknxh/rlt-openpi.git
cd rlt-openpi

# Expects DROID already cloned at $HOME/franka_teleop.
# Override the env name with: bash setup_conda_env.sh <env_name>
bash setup_conda_env.sh

conda activate rlt
```

The script creates the env with Python 3.11, installs OpenPI + rlt-openpi + DROID + oculus_reader, pins opencv-contrib and numpy < 2.0, and patches `transformers` with OpenPI's `transformers_replace` files. Requires the ZED SDK at `/usr/local/zed` for pyzed bindings.

### Running the `exp/` scripts

All scripts under `exp/` call plain `python`, not `uv run python`, so that the same script works in either install path. **Activate your environment first** (`source .venv/bin/activate` or `conda activate rlt`), then run e.g. `bash exp/stage1.sh`.

---

## Checkpoints

The example commands assume the π₀.₅ DROID PyTorch checkpoint at:

```
$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors
```

Any OpenPI checkpoint compatible with your chosen `--vla-config-name` will work — point `--train.vla-checkpoint-dir` / `--vla-checkpoint-dir` at whichever safetensors file you have. Converting OpenPI's JAX/Orbax checkpoints to PyTorch is handled by `scripts/convert_jax_to_pytorch.py` (out of scope for this README).

---

## Stage 1: Train the RL Token

Trains a small encoder/decoder to compress the VLA's internal per-token embeddings `z_{1:M}` into a single **RL token** `z_rl`, via masked MSE reconstruction on a LeRobot demonstration dataset. With `--train.vla-finetune-alpha 0` the VLA is frozen; with `α > 0` the VLA is co-finetuned using a weighted flow-matching loss (matching the paper's `L_ro + α · L_vla` objective).

Example command (see `exp/stage1.sh`):

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

Swap `--repo-id`, `--data-transforms-fn`, and the VLA config/checkpoint for your own dataset and robot.

Key flags (full list in `src/rlt_openpi/training/config.py::RLTokenTrainConfig`):

| Flag | Purpose |
| --- | --- |
| `--train.vla-finetune-alpha` | `0.0` = frozen VLA (only encoder/decoder trained); `> 0` = joint finetune. |
| `--train.num-train-steps` | Default `5000`. |
| `--train.batch-size` | Default `32`. |
| `--train.warmup-steps` | Linear LR warmup, default `500`. |
| `--train.resume-checkpoint` | Resume from a previous `rl_token_step<N>.pt`. |
| `--train.save-every` | Checkpoint interval (default `1000`). |
| `--repo-id` | LeRobot dataset ID. |
| `--data-transforms-fn` | Import path to a data-transform factory (e.g. `rlt_openpi.policies.franka.config.three_camera_droid`). |

Outputs land under `checkpoints/rl_token/<run_name>/rl_token_step<N>.pt`, where `run_name` defaults to `run_YYYYMMDD_HHMMSS`.

---

## Stage 2: Online RL

With VLA + encoder frozen, a lightweight **Actor** and **Twin-Q Critic** are trained online. The actor conditions on `(z_rl, VLA reference action chunk)` and outputs a **residual** over the VLA's proposal (zero-initialized last layer, so the actor starts as a copy of the VLA). The loop first runs a **warmup phase** collecting episodes with the base VLA policy, then alternates between rollout collection and off-policy TD3-style updates at UTD = 5, with a BC regularizer pulling the actor toward the VLA reference and reference-action dropout. A human supervisor provides sparse success/failure/progress rewards and can take over the robot via a VR controller mid-episode; interventions are stored in the replay buffer as corrective labels.

### Example hardware (Franka + DROID + VR)

The example `--env-factory` and `--intervention-factory` target:

- Franka Panda driven by the DROID stack (joint-velocity control).
- Three ZED cameras matching the layout expected by `three_camera_droid`.
- Oculus/VR controller wired into `src/rlt_openpi/envs/franka/intervention.py` (`make_vr_intervention`).

To run against a different robot or simulator, implement your own `make_env` / `make_intervention` callables (see the Franka example as a template) and pass their import paths via `--env-factory` and `--intervention-factory`.

### Example command (see `exp/stage2.sh`)

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

Key flags (full list in `src/rlt_openpi/training/config.py::OnlineRLTrainConfig`):

| Flag | Purpose |
| --- | --- |
| `--env-factory` | Import path to a `make_env` callable. |
| `--intervention-factory` | Import path to a `make_intervention` callable. Optional. |
| `--rl-token-checkpoint` | Stage 1 checkpoint (encoder + optional finetuned VLA). |
| `--task-prompt` | Language instruction passed to the VLA each step. |
| `--warmup-steps` | Number of env steps of pure base-VLA data collection before RL updates start. |
| `--chunk-length` | Action chunk length `C`. |
| `--max-episode-chunks` | Safety cap per episode. |
| `--warmup-buffer` | Load a previously saved warmup buffer and skip the warmup phase. |
| `--resume-checkpoint` | Resume a Stage 2 run. |

Defaults worth knowing: `gamma=0.99`, `tau=0.005`, `utd_ratio=5`, `bc_regularizer_beta=0.5`, `actor_noise_sigma=0.1`, `ref_action_dropout=0.5`, `batch_size=256`.

---

## Human-in-the-Loop Controls (Stage 2)

**Keyboard rewards** — non-blocking listener in `src/rlt_openpi/rollout/reward.py` reads single keypresses (no Enter needed):

| Key | Meaning |
| --- | --- |
| `s` or `Space` | Success — reward `+1.0`, episode ends. |
| `f` | Failure — reward `0.0`, episode ends. |
| `p` | Progress — reward `+0.5`, episode continues. |

Success/failure are latched; progress is consumed on read. Headless runs (no TTY) degrade gracefully to line-buffered input.

**VR intervention** (example, Franka-specific) — `src/rlt_openpi/envs/franka/intervention.py`. When the operator engages the VR controller, `VRInterventionManager` takes over the current action chunk; the executed human action is written to the replay buffer and downstream BC regularization pulls the actor toward it. On a different rig, provide your own `InterventionManager` subclass.

**Terminal UI** — `src/rlt_openpi/utils/display.py` renders warmup progress, per-episode stats, and operator instructions using [`rich`](https://github.com/Textualize/rich).

---

## Evaluation

`scripts/evaluate.py` auto-detects whether a checkpoint is a Stage 1 (VLA-only) or Stage 2 (VLA + RL token + actor) artifact and runs the appropriate rollout loop on whatever env factory you pass in. Example command (Franka rig):

```bash
# exp/eval_full.sh
python scripts/evaluate.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir checkpoints/pi05_droid_pytorch/model.safetensors \
    --rl-token-checkpoint checkpoints/rl_token/rl_token_step5000.pt \
    --checkpoint checkpoints/online_rl/run_latest/online_rl_ep100.pt \
    --task-prompt "stack the three blocks on the tray" \
    --num-episodes 50
```

Results (per-episode success, reward, length) are written to JSON under the run's save dir.

---

## Tests

```bash
pytest tests/
```

Covers: RL token encoder/decoder shapes, VLA embedding extraction hooks, Actor + TwinQCritic forward/backward, replay buffer, and an end-to-end Stage 2 trainer smoke test.

---

## Status & Limitations

This is a **research reimplementation** — unofficial and not affiliated with Physical Intelligence. It is under active development and may still contain bugs.

Currently implemented:

- Stage 1 RL token training with both frozen-VLA and joint VLA-finetune modes.
- Stage 2 TD3-style online RL (twin Q, delayed actor, Polyak targets, BC regularizer, reference-action dropout, subsampled chunk stride).
- Example Franka/DROID env wrapper with three ZED cameras.
- Example VR intervention via an Oculus controller (corrective actions written to the buffer).
- Keyboard-based human reward shaping.
- Rich terminal UI for warmup + rollout progress.
- Evaluation script that auto-detects Stage 1 vs Stage 2 checkpoints.

Not yet validated end-to-end on the four paper tasks (screw installation, zip-tie fastening, Ethernet insertion, charger insertion). Only the Franka + DROID + VR path has been exercised during development; other robots, simulators, and VLA configs are supported in principle but untested here.

---

## Citation

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

The VLA backbone and training infrastructure come from [OpenPI](https://github.com/Physical-Intelligence/openpi). All credit for the RLT method belongs to the paper's authors; any errors in this reimplementation are mine.
