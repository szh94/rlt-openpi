"""从 LeRobot 数据集加载 obs，统一为 ALOHA schema。

复用 OpenPI 的 ``create_torch_dataset`` 读取数据集**原始帧**（不应用
Normalize / model_transforms，那些由 ``VLAWrapper.preprocess_obs`` 的
变换链负责），再按 ALOHA schema 组装 ``{state, images{cam_*}, prompt}``。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rlt_openpi.envs.obs_source import ObsSource


class DatasetObsSource(ObsSource):
    """从 LeRobot 数据集加载 obs（ALOHA schema）。

    要求数据集使用 ALOHA 式键名：``observation.state`` 与
    ``observation.images.<cam_name>``。
    """

    def __init__(
        self,
        repo_id: str,
        vla_config_name: str,
        task_prompt: str = "",
        image_size: tuple[int, int] = (224, 224),
        camera_names: list[str] | None = None,
    ) -> None:
        if not repo_id:
            raise ValueError("DatasetObsSource requires repo_id (LeRobot 数据集 ID)")
        self._repo_id = repo_id
        self._task_prompt = task_prompt
        self._image_size = tuple(image_size)
        self._camera_names = list(camera_names or ["cam_high", "cam_left_wrist", "cam_right_wrist"])
        self.last_action_chunk: np.ndarray | None = None
        self._printed_image_debug = False
        self._iterator = self._build_iterator(vla_config_name)

    def _build_iterator(self, vla_config_name: str):
        """Build an infinite cyclic iterator over dataset frames (ALOHA schema)."""
        from openpi.training.data_loader import create_torch_dataset

        from rlt_openpi.training.data_loader import build_data_config

        openpi_config, data_config = build_data_config(vla_config_name, self._repo_id)
        action_key = data_config.action_sequence_keys[0]
        self._camera_key_map = self._resolve_camera_key_map(data_config)

        dataset = create_torch_dataset(
            data_config,
            openpi_config.model.action_horizon,
            openpi_config.model,
        )

        def _iter():
            while True:
                for raw in dataset:
                    yield self._build_aloha_obs(raw, action_key)

        return _iter()

    @staticmethod
    def _resolve_camera_key_map(data_config: Any) -> dict[str, str]:
        """Map model camera names to raw keys using the VLA repack config."""
        camera_key_map: dict[str, str] = {}
        for transform in data_config.repack_transforms.inputs:
            structure = getattr(transform, "structure", None)
            if not isinstance(structure, Mapping):
                continue
            images = structure.get("images")
            if isinstance(images, Mapping):
                camera_key_map.update(
                    {str(camera): str(raw_key) for camera, raw_key in images.items()}
                )
        return camera_key_map

    def _build_aloha_obs(
        self,
        raw: dict[str, Any],
        action_key: str,
    ) -> dict[str, Any]:
        state = np.asarray(raw["observation.state"], dtype=np.float32)
        action_chunk = np.asarray(raw[action_key], dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        self.last_action_chunk = action_chunk.copy()

        if not getattr(self, "_printed_image_debug", False):
            raw_image_keys = sorted(
                key for key in raw if key.startswith("observation.images.")
            )
            print(f"[DatasetObsSource] raw image keys = {raw_image_keys}")
            for key in raw_image_keys:
                image = np.asarray(raw[key])
                print(
                    f"[DatasetObsSource] raw {key}: shape={image.shape}, "
                    f"dtype={image.dtype}, min={image.min()}, max={image.max()}"
                )
            for cam in self._camera_names:
                key = self._camera_key_map.get(cam, f"observation.images.{cam}")
                print(
                    f"[DatasetObsSource] output {cam} reads {key}: "
                    f"present={key in raw and raw[key] is not None}"
                )
            self._printed_image_debug = True

        images: dict[str, np.ndarray] = {}
        for cam in self._camera_names:
            key = self._camera_key_map.get(cam, f"observation.images.{cam}")
            if key in raw and raw[key] is not None:
                images[cam] = self._to_chw_uint8(raw[key])
            else:
                images[cam] = np.zeros((3, *self._image_size), dtype=np.uint8)
        return {
            "state": state,
            "images": images,
            "prompt": self._task_prompt,
        }

    @staticmethod
    def _to_chw_uint8(img: Any) -> np.ndarray:
        img = np.asarray(img)
        if img.ndim == 4:  # batched [B, H, W, C] / [B, C, H, W]
            img = img[0]
        if img.ndim == 3 and img.shape[-1] == 3:  # HWC -> CHW
            img = img.transpose(2, 0, 1)
        if np.issubdtype(img.dtype, np.floating):
            img = (255.0 * np.clip(img, 0.0, 1.0)).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        return img

    def get_obs(self) -> dict[str, Any]:
        print(f"[DatasetObsSource.get_obs] entered; repo_id={self._repo_id}")
        return next(self._iterator)
