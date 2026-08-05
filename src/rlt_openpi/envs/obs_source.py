"""黑盒 obs 获取：统一 ``get_obs()`` 接口（:class:`ObsSource`），按来源分发。

三种来源分别实现在独立子包中（不在本文件内混在一起）:

  - ``rlt_openpi.envs.real.robot_obs_source``      真实机械臂硬件（``RobotObsSource``）
  - ``rlt_openpi.envs.mock.mock_obs_source``       随机假 obs（``MockObsSource``）
  - ``rlt_openpi.envs.dataset.dataset_obs_source`` 从数据集加载（``DatasetObsSource``）

统一 ALOHA schema（三种来源都产出）::

    state:   float32[14]  (弧度 + gripper [0,1])
    images:  {cam_high, cam_left_wrist, cam_right_wrist} uint8[3,H,W] CHW
    prompt:  str
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ObsSource(ABC):
    """Black-box obs source.

    所有来源返回**单个未 batch 的原始 obs dict**，遵循 ALOHA schema。
    训练/评估脚本只通过 :meth:`get_obs` 拿 obs，不需要关心 obs 的来源。
    """

    @abstractmethod
    def get_obs(self) -> dict[str, Any]:
        """Return a single unbatched raw obs dict in ALOHA schema."""


def make_obs_source(kind: str, **kwargs: Any) -> ObsSource:
    """Create an :class:`ObsSource` by kind.

    Args:
        kind: ``"robot"`` / ``"mock"`` / ``"dataset"``。
        **kwargs: Forwarded to the concrete source constructor.

    Returns:
        Configured ``ObsSource``.

    Raises:
        ValueError: Unknown kind.
    """
    if kind == "robot":
        from rlt_openpi.envs.real.robot_obs_source import RobotObsSource

        return RobotObsSource(**kwargs)
    if kind == "mock":
        from rlt_openpi.envs.mock.mock_obs_source import MockObsSource

        return MockObsSource(**kwargs)
    if kind == "dataset":
        from rlt_openpi.envs.dataset.dataset_obs_source import DatasetObsSource

        return DatasetObsSource(**kwargs)
    raise ValueError(f"Unknown obs source kind: {kind!r}")
