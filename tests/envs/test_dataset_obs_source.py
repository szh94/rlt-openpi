import unittest
from types import SimpleNamespace

import numpy as np

from rlt_openpi.envs.dataset.dataset_obs_source import DatasetObsSource


class DatasetObsSourceTest(unittest.TestCase):
    def test_resolve_camera_key_map_uses_repack_config(self) -> None:
        data_config = SimpleNamespace(
            repack_transforms=SimpleNamespace(
                inputs=[
                    SimpleNamespace(
                        structure={
                            "images": {
                                "cam_high": "observation.images.head_image",
                                "cam_left_wrist": "observation.images.left_wrist_image",
                                "cam_right_wrist": "observation.images.right_wrist_image",
                            }
                        }
                    )
                ]
            )
        )

        self.assertEqual(
            DatasetObsSource._resolve_camera_key_map(data_config),
            {
                "cam_high": "observation.images.head_image",
                "cam_left_wrist": "observation.images.left_wrist_image",
                "cam_right_wrist": "observation.images.right_wrist_image",
            },
        )

    def test_build_aloha_obs_keeps_action_chunk_outside_observation(self) -> None:
        source = DatasetObsSource.__new__(DatasetObsSource)
        source._task_prompt = "test task"
        source._image_size = (2, 3)
        source._camera_names = ["cam_high"]
        source._camera_key_map = {
            "cam_high": "observation.images.head_image",
        }
        source._printed_image_debug = True
        source.last_action_chunk = None

        action_chunk = np.arange(28, dtype=np.float32).reshape(2, 14)
        raw = {
            "observation.state": np.arange(14, dtype=np.float32),
            "observation.images.head_image": np.arange(24, dtype=np.uint8).reshape(3, 2, 4),
            "action": action_chunk,
        }

        obs = source._build_aloha_obs(raw, "action")

        self.assertEqual(set(obs), {"state", "images", "prompt"})
        np.testing.assert_array_equal(
            obs["images"]["cam_high"], raw["observation.images.head_image"]
        )
        np.testing.assert_array_equal(source.last_action_chunk, action_chunk)
        self.assertIsNot(source.last_action_chunk, action_chunk)


if __name__ == "__main__":
    unittest.main()
