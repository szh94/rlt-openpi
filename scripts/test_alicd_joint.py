#!/usr/bin/env python3
"""Minimal test: read joint state from Alicia-D and move joint 2 by +10°.

This script:
1. Connects to the robot via ``make_alicd_env`` (with default cameras).
2. Reads the initial ``observation/joint_position``.
3. Constructs an action chunk where every single-step action targets the
   initial joint angles with joint-2 offset by +10 degrees.
4. Executes ``env.step(action_chunk)`` and prints the new joint state.

Camera images from observations are saved to ``--image-dir`` (default ``image/``).

Usage::

    cd /home/shenzh/Robot/rlt-openpi
    source .venv/bin/activate
    python scripts/test_alicd_joint.py [--port /dev/ttyACM0] [--speed 30]
"""

from __future__ import annotations

import argparse
import atexit
import logging
import math
import os
import sys
import termios
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Ensure the rlt_openpi package is importable (editable install should handle
# this, but add the src dir just in case)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from rlt_openpi.envs.alicd.env_factory import make_alicd_env
from rlt_openpi.envs.alicd.safe_pose import move_to_safe_pose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
log = logging.getLogger("test_alicd_joint")

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

# ── helpers ────────────────────────────────────────────────────────────────

def _save_obs_images(obs: dict, image_dir: Path, prefix: str) -> None:
    """Save all camera images from an observation dict to *image_dir*.

    Images are expected under keys like ``observation/<cam_name>``, stored as
    ``{prefix}_{cam_name}.png``.  RGB → BGR for cv2.imwrite.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    for key, val in obs.items():
        if key.startswith("observation/") and isinstance(val, np.ndarray) and val.ndim == 3:
            cam_name = key.split("/", 1)[1]
            out_path = image_dir / f"{prefix}_{cam_name}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(val, cv2.COLOR_RGB2BGR))
            log.info("Saved image: %s", out_path)


def _safe_close(env, timeout: float = 3.0) -> None:
    """Call *env.close()* in a daemon thread, wait at most *timeout* seconds.

    If ``env.close()`` hangs (e.g. pyserial's internal thread won't exit),
    we move on — ``os._exit`` will still fire.
    """
    done = threading.Event()

    def _close() -> None:
        try:
            env.close()
        except Exception:
            pass
        finally:
            done.set()

    t = threading.Thread(target=_close, daemon=True)
    t.start()
    if not done.wait(timeout=timeout):
        log.warning("env.close() did not finish within %.1f s — forcing exit anyway", timeout)


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test Alicia-D joint control")
    parser.add_argument("--port", default="", help="Serial port (auto-detect if empty)")
    parser.add_argument("--speed", type=float, default=30.0, help="Joint speed in deg/s")
    parser.add_argument("--control-hz", type=float, default=15.0, help="Control loop frequency in Hz (default: 15)")
    parser.add_argument("--chunk-length", type=int, default=10, help="Chunk length C")
    parser.add_argument("--j2-step-deg", type=float, default=3.0, help="Joint-2 increment per chunk step in degrees")
    parser.add_argument("--image-dir", default="image", help="Directory to save camera images (default: image/)")
    parser.add_argument("--log-dir", default="", help="Directory to save log file (default: no log file)")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera capture")
    parser.add_argument(
        "--camera", nargs=2, action="append", metavar=("NAME", "DEVICE_ID"),
        default=None,
        help="Add a camera (repeatable). Overrides defaults. "
             "Example: --camera exterior_image_1_left 0 --camera wrist_image_left 2",
    )
    args = parser.parse_args()

    # Setup log file if requested
    if args.log_dir:
        log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "test_alicd_joint.log"
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
        logging.getLogger().addHandler(fh)
        # Print to stderr so the shell banner still shows even if log goes to file
        print(f"Log saved to: {log_path}", file=sys.stderr)

    # Build camera_ids dict
    if args.no_camera:
        camera_ids: dict[str, int] | None = None
    elif args.camera is not None:
        camera_ids = {name: int(dev_id) for name, dev_id in args.camera}
    else:
        camera_ids = None  # no --camera args → no cameras

    image_dir = Path(args.image_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")

    j2_step_rad = args.j2_step_deg * DEG_TO_RAD
    log.info("Joint-2 step: %.2f° = %.4f rad per chunk step", args.j2_step_deg, j2_step_rad)

    # ------------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------------
    log.info("Creating Alicia-D env (port=%s, speed=%.1f deg/s, chunk_length=%d, cameras=%s)...",
             args.port or "<auto>", args.speed, args.chunk_length,
             list(camera_ids.keys()) if camera_ids else "none")

    env = make_alicd_env(
        port=args.port,
        action_dim=7,
        chunk_length=args.chunk_length,
        task_prompt="joint position test",
        control_hz=args.control_hz,
        max_episode_chunks=10,
        speed_deg_s=args.speed,
        camera_ids=camera_ids,
        image_size=(224, 224),
    )

    try:
        # ------------------------------------------------------------------
        # Save terminal settings — env.reset() will switch to cbreak mode
        # via HumanReward.start().  We MUST restore on exit or the terminal
        # will appear "hung" (no prompt echo).
        # ------------------------------------------------------------------
        try:
            _saved_termios = termios.tcgetattr(sys.stdin)
            _termios_saved = True
        except (termios.error, OSError):
            _termios_saved = False

        def _restore_terminal() -> None:
            if _termios_saved:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _saved_termios)
                except Exception:
                    pass

        atexit.register(_restore_terminal)

        # ------------------------------------------------------------------
        # Reset → home position
        # ------------------------------------------------------------------
        log.info("Resetting robot to home position...")
        obs = env.reset()
        log.info("Reset complete.")

        # Save post-reset images
        if camera_ids:
            _save_obs_images(obs, image_dir, "reset")

        # ------------------------------------------------------------------
        # Read initial joint state
        # ------------------------------------------------------------------
        joint_positions = obs["observation/joint_position"]  # [6] radians
        gripper_pos = obs["observation/gripper_position"]     # [1]
        log.info("Initial joint positions (deg): %s",
                 [f"{a * RAD_TO_DEG:.2f}" for a in joint_positions])
        log.info("Initial joint positions (rad): %s",
                 [f"{a:.4f}" for a in joint_positions])
        log.info("Initial gripper: %.1f", gripper_pos[0])

        # ------------------------------------------------------------------
        # Build action_chunk: j2 increases by j2_step_deg each step
        # Step 1: j2 + step, Step 2: j2 + 2*step, ..., Step C: j2 + C*step
        # ------------------------------------------------------------------
        base_joints = joint_positions.copy().astype(np.float64)  # [6]
        action_chunk = np.zeros((args.chunk_length, 7), dtype=np.float64)
        for i in range(args.chunk_length):
            action_chunk[i, :6] = base_joints
            action_chunk[i, 1] += j2_step_rad * (i + 1)  # joint-2 = index 1
            action_chunk[i, :6] = np.clip(action_chunk[i, :6], -math.pi, math.pi)
            action_chunk[i, 6] = gripper_pos[0]  # keep gripper open for most steps

        # Last step: close gripper halfway (0 = closed, 997 ≈ open)
        action_chunk[-1, 6] = 500.0

        log.info("Action chunk shape: %s", action_chunk.shape)
        log.info("Joint-2 target per step (deg): %s",
                 [f"{a[1] * RAD_TO_DEG:.1f}" for a in action_chunk])
        log.info("Joint-2 delta per step (deg): %s",
                 [f"{(action_chunk[i, 1] - base_joints[1]) * RAD_TO_DEG:+.1f}" for i in range(args.chunk_length)])

        # ------------------------------------------------------------------
        # Execute action chunk
        # ------------------------------------------------------------------
        input("\nPress Enter to execute the action chunk...")
        log.info("Executing action_chunk (%d steps)...", args.chunk_length)
        next_obs, rewards, done, info = env.step(action_chunk)
        log.info("Step complete. rewards=%s, done=%s, info=%s", rewards, done, info)

        # Save post-step images
        if camera_ids:
            _save_obs_images(next_obs, image_dir, "step")

        # ------------------------------------------------------------------
        # Read resulting joint state
        # ------------------------------------------------------------------
        new_joint_positions = next_obs["observation/joint_position"]
        new_gripper_pos = next_obs["observation/gripper_position"]

        log.info("Final joint positions (deg): %s",
                 [f"{a * RAD_TO_DEG:.2f}" for a in new_joint_positions])
        log.info("Joint deltas (deg): %s",
                 [f"{(n - o) * RAD_TO_DEG:+.2f}" for n, o in zip(new_joint_positions, joint_positions)])
        log.info("  → joint-2 moved by: %+.2f°", (new_joint_positions[1] - joint_positions[1]) * RAD_TO_DEG)

    finally:
        _restore_terminal()  # undo HumanReward's cbreak mode
        # Move to safe pose before disconnecting
        try:
            move_to_safe_pose(env.alicd_robot, speed=args.speed, torque_off=True)
        except Exception:
            log.exception("Failed to move to safe pose")
        _safe_close(env, timeout=3.0)
        log.info("Env closed (or timed out).")

    # os._exit: work around pyserial's non-daemon internal thread.
    log.info("Exiting.")
    os._exit(0)


if __name__ == "__main__":
    main()
