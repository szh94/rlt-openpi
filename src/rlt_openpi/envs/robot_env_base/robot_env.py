"""Real-robot environment for online RL training.

Provides ``RobotEnv``, a chunk-level environment that connects to any
robot through three user-supplied callables (``step_fn``, ``reset_fn``,
``get_obs_fn``).  No dependency on any specific robot stack (DROID,
polymetis, ROS, etc.) — the wiring happens in the user's launch script.

Human reward (success/failure) is collected via
:class:`~rlt_openpi.envs.robot_env_base.reward.HumanReward` using instant
keypress detection (no Enter needed) during episodes.

Usage (with DROID)::

    from droid.robot_env import RobotEnv as DroidEnv

    droid = DroidEnv(action_space="cartesian_velocity", control_hz=15)

    def get_obs():
        obs = droid.get_observation()
        return {
            "observation/joint_position": np.array(
                obs["robot_state"]["joint_positions"], dtype=np.float32,
            ),
            "observation/gripper_position": np.array(
                [obs["robot_state"]["gripper_position"]], dtype=np.float32,
            ),
            "observation/exterior_image_1_left": obs["image"]["39790647_left"],
            "observation/wrist_image_left": obs["image"]["15850436_left"],
            "observation/exterior_image_2_left": obs["image"]["35840217_left"],
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
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from rlt_openpi.envs.robot_env_base.reward import HumanReward
from rlt_openpi.utils import display

logger = logging.getLogger(__name__)


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
        dry_run: If True, actions are printed to stdout instead of
            being sent to the robot hardware.  Useful for inspecting
            action values before running on a real arm.
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
        dry_run: bool = False,
        print_actions: bool = False,
    ) -> None:
        self._step_fn = step_fn
        self._reset_fn = reset_fn
        self._get_obs_fn = get_obs_fn
        self._action_dim = action_dim
        self._chunk_length = chunk_length
        self._control_period = 1.0 / control_hz
        self._max_episode_chunks = max_episode_chunks
        self._dry_run = dry_run
        self._print_actions = print_actions

        self._chunk_count = 0
        self._feedback = HumanReward()
        self._display_episode_num: int = 0

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
        self._feedback.stop()
        self._reset_fn()
        self._chunk_count = 0

        display.episode_reset(self._display_episode_num)
        input("")

        self._feedback.start()
        display.episode_running(self._display_episode_num)

        return self._get_obs_fn()

    def step(
        self, action_chunk: NDArray
    ) -> tuple[dict[str, Any], NDArray, bool, dict[str, Any]]:
        """Execute C single-step actions on the robot.

        Args:
            action_chunk: Actions to execute, shape ``[C, action_dim]``.

        Returns:
            next_obs: Observation dict after the last step.
            rewards: Per-step rewards ``[C]``.  Terminal success gives +1;
                progress keypresses (``p``) give +0.5 without ending
                the episode.
            done: Whether the episode ended.
            info: Contains ``"success"`` key on termination.
        """
        C = self._chunk_length
        rewards = np.zeros(C, dtype=np.float32)
        done = False
        info: dict[str, Any] = {}

        for k in range(C):
            t_start = time.time()

            # Print all action values (generic, works for any action_dim)
            if self._print_actions or self._dry_run:
                a = action_chunk[k]
                vals = " ".join(f"[{i}]{a[i]:+.6f}" for i in range(len(a)))
                tag = "dry_run" if self._dry_run else "action"
                print(f"[{tag}] step {k:>2d}: {vals}")

            # Send to hardware (dry_run skips this)
            if not self._dry_run:
                self._step_fn(action_chunk[k])
            else:
                # Still sleep to simulate control period
                time.sleep(self._control_period)

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
                elif signal == "p":
                    rewards[k] = self._feedback.progress_reward
                    logger.info("Human signal: PROGRESS (+%.2f)", self._feedback.progress_reward)
                if done:
                    break

            # Enforce control frequency
            elapsed = time.time() - t_start
            sleep_time = self._control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        info["steps_executed"] = k + 1
        self._chunk_count += 1

        # Chunk separator (printed when action logging is on)
        if self._print_actions or self._dry_run:
            print(f"{'─' * 60}")

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
