"""真实机械臂硬件 obs 来源。

``read_obs_fn`` 负责"读硬件原始数据"，``build_obs_fn`` 负责"把原始数据
转换成 ALOHA schema obs"。两步拆开，便于复用 build 逻辑或替换读来源。
"""

from __future__ import annotations

from typing import Any, Callable

from rlt_openpi.envs.obs_source import ObsSource


class RobotObsSource(ObsSource):
    """Obs source backed by real robot hardware."""

    def __init__(
        self,
        read_obs_fn: Callable[[], dict],
        build_obs_fn: Callable[[dict], dict],
    ) -> None:
        self._read_obs_fn = read_obs_fn
        self._build_obs_fn = build_obs_fn

    def get_obs(self) -> dict[str, Any]:
        return self._build_obs_fn(self._read_obs_fn())
