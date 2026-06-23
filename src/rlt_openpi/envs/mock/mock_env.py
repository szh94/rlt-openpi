"""Mock environment for offline Stage 2 testing with real VLA checkpoint.

Generates fake ALOHA-format observations (images + joint state) that the
real VLA can process, but does NOT control any robot.  Actions passed to
``step()`` are discarded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def make_mock_obs(
    prompt: str = "do the task",
    image_size: int = 224,
) -> dict[str, Any]:
    """Generate one fake observation in ALOHA schema.

    Args:
        prompt: Task instruction string.
        image_size: H=W of generated random images (produced as channel-first
            ``(3, H, W)`` to match AlohaInputs).
    """
    obs: dict[str, Any] = {"prompt": prompt}

    # Dual-arm state: [6 joints + 1 gripper] * 2 = 14
    obs["state"] = np.random.randn(14).astype(np.float32)

    # ALOHA expects four cameras in channel-first (C, H, W) format.
    obs["images"] = {
        "cam_high": np.random.randint(
            0, 255, (3, image_size, image_size), dtype=np.uint8
        ),
        "cam_low": np.random.randint(
            0, 255, (3, image_size, image_size), dtype=np.uint8
        ),
        "cam_left_wrist": np.random.randint(
            0, 255, (3, image_size, image_size), dtype=np.uint8
        ),
        "cam_right_wrist": np.random.randint(
            0, 255, (3, image_size, image_size), dtype=np.uint8
        ),
    }

    return obs


class MockEnv:
    """Chunk-level env that feeds fake ALOHA-format observations through the VLA.

    Each ``step()`` ignores the action and returns a fresh random observation.
    Rewards are synthetic (zeros).  This tests the full production pipeline:
    VLA → RLToken → Actor → Critic, without a robot.

    Args:
        action_dim: Single-step action dimension (14 for ALOHA dual-arm).
        chunk_length: C, steps per action chunk.
        task_prompt: Task instruction for VLA.
        max_episode_chunks: Auto-reset after this many chunks.
        image_size: H=W of generated random camera images.
    """

    def __init__(
        self,
        action_dim: int = 14,
        chunk_length: int = 10,
        task_prompt: str = "do the task",
        max_episode_chunks: int = 150,
        image_size: int = 224,
    ) -> None:
        self._action_dim = action_dim
        self._chunk_length = chunk_length
        self._task_prompt = task_prompt
        self._max_chunks = max_episode_chunks
        self._chunk_count = 0
        self._image_size = image_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    def _make_obs(self) -> dict[str, Any]:
        return make_mock_obs(
            prompt=self._task_prompt,
            image_size=self._image_size,
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
            next_obs: New fake observation (ALOHA format).
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
    action_dim: int = 14,
    chunk_length: int = 10,
    task_prompt: str = "do the task",
    max_episode_chunks: int = 150,
    image_size: int = 224,
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
        image_size=image_size,
    )
