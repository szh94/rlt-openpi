"""随机假 obs，用于测试代码（结构同 ``aloha_policy.make_aloha_example()``）。

每次 ``get_obs`` 返回一组新的随机 ALOHA schema 观测，不连接硬件、不读数据集。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rlt_openpi.envs.obs_source import ObsSource


class MockObsSource(ObsSource):
    """随机假 obs, 用于测试代码(结构同 ``aloha_policy.make_aloha_example()``)。"""

    def __init__(
        self,
        task_prompt: str = "",
        image_size: tuple[int, int] = (224, 224),
        camera_names: list[str] | None = None,
        state: np.ndarray | None = None,
    ) -> None:
        self._task_prompt = task_prompt
        self._image_size = tuple(image_size)
        self._camera_names = list(camera_names or ["cam_high", "cam_left_wrist", "cam_right_wrist"])
        self._state = state

    def get_obs(self) -> dict[str, Any]:
        # if self._state is not None:
        #     state = np.asarray(self._state, dtype=np.float32)
        # else:
        #     # 关节弧度 + gripper [0,1] 的量级大约都在 [-1,1]，直接用均匀分布。
        #     state = np.random.uniform(-1.0, 1.0, size=(14,)).astype(np.float32)
        # h, w = self._image_size
        # images = {
        #     cam: np.random.randint(256, size=(3, h, w), dtype=np.uint8)
        #     for cam in self._camera_names
        # }
        # return {
        #     "state": state,
        #     "images": images,
        #     "prompt": self._task_prompt,
        # }
        return {
            "state": np.ones((14,)),
            "images": {
                "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            },
            "prompt": self._task_prompt,
        }