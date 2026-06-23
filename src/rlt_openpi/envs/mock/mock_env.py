"""Mock environment for offline Stage 2 testing with REAL VLA checkpoint.

Generates fake DROID-format observations (images + joint state) that the
real VLA can process, but does NOT control any robot.  Actions passed to
``step()`` are discarded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def make_mock_obs(
    prompt: str = "do the task",
    num_joints: int = 7,
    num_arms: int = 1,
    image_size: int = 64,
    cameras: tuple[str, ...] = ("exterior_image_1_left", "wrist_image_left", "exterior_image_2_left"),
) -> dict[str, Any]:
    """Generate one fake observation dict matching the configured layout.

    Args:
        prompt: Task instruction string.
        num_joints: Per-arm joint position dimension (default 7).
        num_arms: Number of arms (1 = single, 2 = dual).
        image_size: H=W of generated random images.
        cameras: Camera key suffixes (without ``observation/`` prefix).
    """
    obs: dict[str, Any] = {"prompt": prompt}

    if num_arms == 1:
        obs["observation/joint_position"] = np.random.randn(num_joints).astype(np.float32)
        obs["observation/gripper_position"] = np.array([np.random.rand()], dtype=np.float32)
    else:
        for side in ("left", "right"):
            obs[f"observation/joint_position_{side}"] = np.random.randn(num_joints).astype(np.float32)
            obs[f"observation/gripper_position_{side}"] = np.array([np.random.rand()], dtype=np.float32)

    for cam in cameras:
        obs[f"observation/{cam}"] = np.random.randint(
            0, 255, (image_size, image_size, 3), dtype=np.uint8
        )

    return obs


class MockEnv:
    """Chunk-level env that feeds fake observations through the real VLA.

    Each ``step()`` ignores the action and returns a fresh random observation.
    Rewards are synthetic (zeros).  This tests the full production pipeline:
    real VLA → real RLToken → Actor → Critic, without a robot.

    Args:
        action_dim: Single-step action dimension.
        chunk_length: C, steps per action chunk.
        task_prompt: Task instruction for VLA.
        max_episode_chunks: Auto-reset after this many chunks.
    """

    def __init__(
        self,
        action_dim: int = 8,
        chunk_length: int = 10,
        task_prompt: str = "do the task",
        max_episode_chunks: int = 50,
        num_joints: int = 7,
        num_arms: int = 1,
        image_size: int = 64,
        cameras: tuple[str, ...] = ("exterior_image_1_left", "wrist_image_left", "exterior_image_2_left"),
    ) -> None:
        self._action_dim = action_dim
        self._chunk_length = chunk_length
        self._task_prompt = task_prompt
        self._max_chunks = max_episode_chunks
        self._chunk_count = 0
        self._num_joints = num_joints
        self._num_arms = num_arms
        self._image_size = image_size
        self._cameras = cameras

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    def _make_obs(self) -> dict[str, Any]:
        return make_mock_obs(
            prompt=self._task_prompt,
            num_joints=self._num_joints,
            num_arms=self._num_arms,
            image_size=self._image_size,
            cameras=self._cameras,
        )

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        """Return a new fake observation."""
        self._chunk_count = 0
        return self._make_obs()

    def step(
        self, action_chunk: NDArray
    ) -> tuple[dict[str, Any], NDArray, bool, dict[str, Any]]:
        """Discard action, return new fake observation.

        Returns:
            next_obs: New fake observation.
            rewards: Zeros [C].
            done: True after ``max_episode_chunks``.
            info: dict with ``steps_executed``.
        """
        self._chunk_count += 1
        C = self._chunk_length
        rewards = np.zeros(C, dtype=np.float32)
        done = self._chunk_count >= self._max_chunks
        info: dict[str, Any] = {"steps_executed": C}
        return self._make_obs(), rewards, done, info


def make_mock_env(
    action_dim: int = 8,
    chunk_length: int = 10,
    task_prompt: str = "do the task",
    max_episode_chunks: int = 50,
    num_joints: int = 7,
    num_arms: int = 1,
    image_size: int = 64,
    cameras: tuple[str, ...] = ("exterior_image_1_left", "wrist_image_left", "exterior_image_2_left"),
    **kwargs,
) -> MockEnv:
    """Factory for ``--env-factory`` CLI argument.

    Usage::

        --env-factory rlt_openpi.envs.mock.mock_env.make_mock_env
    """
    return MockEnv(
        action_dim=action_dim,
        chunk_length=chunk_length,
        task_prompt=task_prompt,
        max_episode_chunks=max_episode_chunks,
        num_joints=num_joints,
        num_arms=num_arms,
        image_size=image_size,
        cameras=cameras,
    )
