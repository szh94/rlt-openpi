"""Mock environment for offline Stage 2 testing with REAL VLA checkpoint.

Generates fake DROID-format observations (images + joint state) that the
real VLA can process, but does NOT control any robot.  Actions passed to
``step()`` are discarded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def make_mock_obs(prompt: str = "do the task") -> dict[str, Any]:
    """Generate one fake observation in DROID-schema format.

    Produces random images and joint positions that ``VLAWrapper.preprocess_obs``
    (via ``DroidInputs``/``ThreeCameraDroidInputs``) can consume.  Image size
    is deliberately small (64×64) to keep VRAM usage down — the VLA's
    ``ResizeImages`` transform will resize to the expected model input size.

    Args:
        prompt: Task instruction string.

    Returns:
        Dict with DROID-schema keys ready for VLA preprocess_obs.
    """
    return {
        "observation/joint_position": np.random.randn(7).astype(np.float32),
        "observation/gripper_position": np.array([np.random.rand()], dtype=np.float32),
        "observation/exterior_image_1_left": np.random.randint(
            0, 255, (64, 64, 3), dtype=np.uint8
        ),
        "observation/wrist_image_left": np.random.randint(
            0, 255, (64, 64, 3), dtype=np.uint8
        ),
        "observation/exterior_image_2_left": np.random.randint(
            0, 255, (64, 64, 3), dtype=np.uint8
        ),
        "prompt": prompt,
    }


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
    ) -> None:
        self._action_dim = action_dim
        self._chunk_length = chunk_length
        self._task_prompt = task_prompt
        self._max_chunks = max_episode_chunks
        self._chunk_count = 0

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        """Return a new fake observation."""
        self._chunk_count = 0
        return make_mock_obs(self._task_prompt)

    def step(
        self, action_chunk: NDArray
    ) -> tuple[dict[str, Any], NDArray, bool, dict[str, Any]]:
        """Discard action, return new fake observation.

        Returns:
            next_obs: New fake DROID-format observation.
            rewards: Zeros [C].
            done: True after ``max_episode_chunks``.
            info: dict with ``steps_executed``.
        """
        self._chunk_count += 1
        C = self._chunk_length
        rewards = np.zeros(C, dtype=np.float32)
        done = self._chunk_count >= self._max_chunks
        info: dict[str, Any] = {"steps_executed": C}
        return make_mock_obs(self._task_prompt), rewards, done, info


def make_mock_env(
    action_dim: int = 8,
    chunk_length: int = 10,
    task_prompt: str = "do the task",
    max_episode_chunks: int = 50,
    **kwargs,
) -> MockEnv:
    """Factory for ``--env-factory`` CLI argument.

    Usage::

        --env-factory rlt_openpi.rollout.mock_env.make_mock_env
    """
    return MockEnv(
        action_dim=action_dim,
        chunk_length=chunk_length,
        task_prompt=task_prompt,
        max_episode_chunks=max_episode_chunks,
    )
