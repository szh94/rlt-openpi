"""Mock environment for testing the RL pipeline without hardware.

``MockEnv`` is a :class:`~rlt_openpi.envs.envbase.robot_env.RobotEnv`
subclass whose observations come from a
:class:`~rlt_openpi.envs.mock.mock_obs_source.MockObsSource` (random
ALOHA-schema obs).  ``step_fn`` / ``reset_fn`` are no-ops, so the
pipeline (VLA embedding → RL token → actor → replay buffer) can be
exercised end-to-end with zero hardware.

This decouples mock-obs support from the training scripts: the script
just builds the env via ``make_env`` and never sees where obs come from.

Usage::

    env = make_env(
        "rlt_openpi.envs.mock.mock_env.make_mock_env",
        action_dim=14,
        chunk_length=10,
        task_prompt="place phone",
    )
"""

from __future__ import annotations

from typing import Any

from rlt_openpi.envs.envbase.robot_env import RobotEnv
from rlt_openpi.envs.mock.mock_obs_source import MockObsSource
from rlt_openpi.envs.obs_source import ObsSource


class MockEnv(RobotEnv):
    """RobotEnv subclass backed by a :class:`MockObsSource`.

    All observations are random ALOHA-schema obs; steps and resets are
    no-ops.  Exposes ``.obs_source`` for inspection.
    """

    def __init__(
        self,
        action_dim: int = 14,
        chunk_length: int = 10,
        control_hz: int = 15,
        max_episode_chunks: int = 150,
        task_prompt: str = "",
        image_size: tuple[int, int] = (224, 224),
        camera_names: list[str] | None = None,
        state: Any = None,
        **kwargs: Any,
    ) -> None:
        obs_source: ObsSource = MockObsSource(
            task_prompt=task_prompt,
            image_size=image_size,
            camera_names=camera_names,
            state=state,
        )
        super().__init__(
            step_fn=lambda action: None,
            reset_fn=lambda: None,
            get_obs_fn=obs_source.get_obs,
            action_dim=action_dim,
            chunk_length=chunk_length,
            control_hz=control_hz,
            max_episode_chunks=max_episode_chunks,
            **kwargs,
        )
        self.obs_source = obs_source  # type: ignore[attr-defined]


def make_mock_env(**kwargs: Any) -> MockEnv:
    """Create a :class:`MockEnv` for pipeline testing (no hardware)."""
    print("[MockEnv] Using random mock obs — no robot hardware connected")
    return MockEnv(**kwargs)
