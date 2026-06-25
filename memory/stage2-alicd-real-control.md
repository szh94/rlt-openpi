---
name: stage2-alicd-real-control
description: Stage 2 Alicia-D real-robot VLA+RL closed-loop control milestone (2026-06-25)
metadata:
  type: project
---

# Stage 2 Alicia-D Real-Robot Control

On 2026-06-25, the VLA + RL closed-loop pipeline was successfully verified on the real Alicia-D robot arm.

## Key Fixes

- **Action dim mapping**: `step_fn` uses dim 7 (not dim 6) as gripper — DROID format is 7 joints + 1 gripper, Alicia-D has 6 joints + 1 gripper. DROID's 7th joint (dim 6) is discarded.
- **Gripper scaling**: DROID gripper is in [0,1], scaled ×1000 to [0,1000] for Alicia-D SDK.
- **Joint safety override**: `joint_override` parameter (via `JOINT_OVERRIDE` env var) locks specific joints to fixed radian values.

## Torch 2.8.0

- Upgraded from 2.7.1 to 2.8.0+cu128
- Fixed openpi METADATA: `torch==2.7.1` → `torch>=2.7.1` to prevent pip/uv from downgrading
- torchvision locked to 0.23.0 (compatible with torch 2.8)
- `pyproject.toml` has double lock: `dependencies` + `override-dependencies`
- `uv.lock` rebuilt with local openpi source (`[tool.uv.sources]`)

## Dry-Run Display

- `print_actions` parameter decoupled from `dry_run` — allows printing actions while driving real hardware
- Output format: joints in degrees, gripper in [0,1000], chunk separators
- `[dry_run]` prefix when dry_run=true, `[action]` when dry_run=false+print_actions=true

## Camera & USB

- Live images turned black when dry_run was disabled due to USB bus contention from robot serial comms
- Added frame read stats `[ok=N fail=N]` for debugging
- Issue resolved — normal operation confirmed

**Why:** Real-robot closed-loop control is the primary goal of the project. These fixes made the pipeline work end-to-end.
**How to apply:** Use `example/stage2_alicd.sh` as the entry point. Set `JOINT_OVERRIDE='{"0":0.0,...}'` to lock joints for safety. Set `PRINT_ACTIONS=true` to monitor actions on real hardware.
