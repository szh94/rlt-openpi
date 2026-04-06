"""Real-robot environment for online RL training.

Provides ``RobotEnv``, a chunk-level environment that connects to any
robot through three user-supplied callables (``step_fn``, ``reset_fn``,
``get_obs_fn``).  No dependency on any specific robot stack (DROID,
polymetis, ROS, etc.) — the wiring happens in the user's launch script.

Human feedback (success/failure) is collected via instant keypress
detection (no Enter needed) during episodes.

Usage (with DROID)::

    from droid.robot_env import RobotEnv as DroidEnv

    droid = DroidEnv(action_space="cartesian_velocity", control_hz=15)

    def get_obs():
        obs = droid.get_observation()
        return {
            "state": np.concatenate([
                obs["robot_state"]["cartesian_position"],
                [obs["robot_state"]["gripper_position"]],
            ]).astype(np.float32),
            "base_0_rgb": obs["39790647_left"],
            "left_wrist_0_rgb": obs["15850436_left"],
            "right_wrist_0_rgb": obs["35840217_left"],
            "prompt": "stack the three blocks on the tray",
        }

    env = RobotEnv(
        step_fn=droid.step,
        reset_fn=droid.reset,
        get_obs_fn=get_obs,
        action_dim=7,
        chunk_length=10,
        control_hz=15,
    )
"""

from __future__ import annotations

import logging
import select
import sys
import termios
import time
import tty
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class HumanFeedback:
    """Non-blocking keyboard listener for human reward signals.

    During an episode, the human presses a single key (no Enter needed):
        - ``s`` → success (reward +1, episode ends)
        - ``f`` → failure (reward  0, episode ends)
        - ``Space`` → success (alternative, easier to hit)

    Uses terminal raw mode for instant keypress detection.
    Falls back to line-buffered input() if raw mode is unavailable
    (e.g., running without a TTY).
    """

    def __init__(self) -> None:
        self._signal: str | None = None
        self._old_settings: list | None = None
        self._raw_mode = False

    def start(self) -> None:
        """Enter raw terminal mode for instant keypress detection."""
        self._signal = None
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._raw_mode = True
        except (termios.error, OSError):
            self._raw_mode = False
            logger.warning("Raw terminal mode unavailable, falling back to line input (type + Enter)")

    def stop(self) -> None:
        """Restore original terminal settings."""
        if self._raw_mode and self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._raw_mode = False
            self._old_settings = None

    def check(self) -> str | None:
        """Poll for keypress. Returns 's', 'f', or None."""
        if self._signal is not None:
            return self._signal

        if self._raw_mode:
            # Non-blocking check: is there a character waiting?
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1).lower()
                if ch in ("s", " "):
                    self._signal = "s"
                elif ch == "f":
                    self._signal = "f"
        return self._signal


class RobotEnv:
    """Chunk-level environment for real robot online RL.

    Robot-agnostic: connects to any robot through three callables.
    Provides the same interface as ``SimEnv`` (``reset``, ``step``,
    ``action_dim``, ``chunk_length``) so it works with ``RolloutWorker``
    and ``OnlineRLTrainer``.

    Args:
        step_fn: Callable that sends a single action ``[action_dim]`` to
            the robot.  Signature: ``step_fn(action: np.ndarray) -> Any``.
        reset_fn: Callable that resets the robot to a home pose.
            Signature: ``reset_fn() -> Any``.
        get_obs_fn: Callable that returns an observation dict with at
            least a ``"state"`` key (proprioceptive, ``np.float32``).
            Camera images and ``"prompt"`` should also be included for
            VLA embedding extraction.
            Signature: ``get_obs_fn() -> dict[str, Any]``.
        action_dim: Dimension of a single-step action (e.g. 7 for
            cartesian velocity + gripper).
        chunk_length: C, number of single-step actions per chunk.
        control_hz: Robot control frequency in Hz.
        max_episode_chunks: Maximum chunks per episode before forced
            termination.
    """

    def __init__(
        self,
        step_fn: Callable[[NDArray], Any],
        reset_fn: Callable[[], Any],
        get_obs_fn: Callable[[], dict[str, Any]],
        action_dim: int = 7,
        chunk_length: int = 10,
        control_hz: int = 15,
        max_episode_chunks: int = 50,
    ) -> None:
        self._step_fn = step_fn
        self._reset_fn = reset_fn
        self._get_obs_fn = get_obs_fn
        self._action_dim = action_dim
        self._chunk_length = chunk_length
        self._control_period = 1.0 / control_hz
        self._max_episode_chunks = max_episode_chunks

        self._chunk_count = 0
        self._feedback = HumanFeedback()

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        """Reset robot to home position and wait for scene setup.

        Returns:
            Observation dict with ``"state"``, camera images, and ``"prompt"``.
        """
        # Restore terminal before blocking input (in case previous episode left raw mode)
        self._feedback.stop()
        self._reset_fn()
        self._chunk_count = 0

        input("\n[RESET] Set up the scene, then press Enter to start...")

        # Enter raw mode for instant keypress detection during episode
        self._feedback.start()
        print("[EPISODE] Running — press [S]/[Space]=success  [F]=failure")

        return self._get_obs_fn()

    def step(
        self, action_chunk: NDArray
    ) -> tuple[dict[str, Any], NDArray, bool, dict[str, Any]]:
        """Execute C single-step actions on the robot.

        Args:
            action_chunk: Actions to execute, shape ``[C, action_dim]``.

        Returns:
            next_obs: Observation dict after the last step.
            rewards: Per-step rewards ``[C]``.  Sparse: only the last step
                of a successful episode gets +1.
            done: Whether the episode ended.
            info: Contains ``"success"`` key on termination.
        """
        C = self._chunk_length
        rewards = np.zeros(C, dtype=np.float32)
        done = False
        info: dict[str, Any] = {}

        for k in range(C):
            t_start = time.time()

            self._step_fn(action_chunk[k])

            # Check for human signal between steps
            signal = self._feedback.check()
            if signal is not None:
                if signal == "s":
                    rewards[k] = 1.0
                    done = True
                    info["success"] = True
                    logger.info("Human signal: SUCCESS")
                elif signal == "f":
                    done = True
                    info["success"] = False
                    logger.info("Human signal: FAILURE")
                break

            # Enforce control frequency
            elapsed = time.time() - t_start
            sleep_time = self._control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        info["steps_executed"] = k + 1
        self._chunk_count += 1

        # Timeout: force episode end after max chunks
        if not done and self._chunk_count >= self._max_episode_chunks:
            done = True
            info["success"] = False
            info["timeout"] = True
            logger.info("Episode timed out after %d chunks", self._chunk_count)

        # Restore terminal when episode ends
        if done:
            self._feedback.stop()

        obs = self._get_obs_fn()
        return obs, rewards, done, info
