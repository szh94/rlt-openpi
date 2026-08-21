import unittest

import numpy as np

from rlt_openpi.envs.dataset.dataset_obs_source import DatasetObsSource


class DatasetObsSourceTest(unittest.TestCase):
    def test_build_aloha_obs_keeps_action_chunk_outside_observation(self) -> None:
        source = DatasetObsSource.__new__(DatasetObsSource)
        source._task_prompt = "test task"
        source._image_size = (2, 3)
        source._camera_names = ["cam_high"]
        source.last_action_chunk = None

        action_chunk = np.arange(28, dtype=np.float32).reshape(2, 14)
        raw = {
            "observation.state": np.arange(14, dtype=np.float32),
            "observation.images.cam_high": np.zeros((2, 3, 3), dtype=np.uint8),
            "action": action_chunk,
        }

        obs = source._build_aloha_obs(raw, "action")

        self.assertEqual(set(obs), {"state", "images", "prompt"})
        np.testing.assert_array_equal(source.last_action_chunk, action_chunk)
        self.assertIsNot(source.last_action_chunk, action_chunk)


if __name__ == "__main__":
    unittest.main()
