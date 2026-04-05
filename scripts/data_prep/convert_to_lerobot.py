"""Convert DROID-style HDF5 demo data to LeRobot format.

Reads raw HDF5 demo files (with images stored directly, not as MP4 videos)
and writes them into a LeRobot dataset compatible with OpenPI's DROID pipeline.

Usage:
    uv run python scripts/utils/convert_demo_to_lerobot.py \
        --data-dir /path/to/demo_hdf5s \
        --repo-name your_hf_username/my_dataset

Camera mapping (hardcoded for the current setup):
    39790647_left  -> exterior_image_1_left  (left exterior)
    35840217_left  -> exterior_image_2_left  (right exterior)
    15850436_left  -> wrist_image_left       (wrist / top-down)
"""

from __future__ import annotations

import dataclasses
import glob
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

# Camera ID -> LeRobot feature name
CAMERA_MAP = {
    "39790647_left": "exterior_image_1_left",
    "35840217_left": "exterior_image_2_left",
    "15850436_left": "wrist_image_left",
}

IMAGE_SIZE = (320, 180)  # (width, height) — DROID LeRobot convention


def resize_image(image: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    return np.array(Image.fromarray(image).resize(size, resample=Image.BICUBIC))


@dataclasses.dataclass
class ConvertConfig:
    data_dir: str
    """Directory containing demo*.hdf files."""

    repo_name: str = "local/stack_the_blocks"
    """LeRobot repo name (local path under $LEROBOT_HOME)."""

    action_key: str = "joint_velocity"
    """Which action to use: 'joint_velocity' or 'joint_position'."""

    fps: int = 15
    """Recording FPS."""

    overwrite: bool = False
    """If True, delete existing dataset before converting."""


def main(config: ConvertConfig) -> None:
    data_dir = Path(config.data_dir)
    demo_files = sorted(glob.glob(str(data_dir / "demo*.hdf")))
    if not demo_files:
        raise FileNotFoundError(f"No demo*.hdf files found in {data_dir}")
    log.info("Found %d demo files in %s", len(demo_files), data_dir)

    output_path = HF_LEROBOT_HOME / config.repo_name
    if output_path.exists():
        if config.overwrite:
            log.info("Removing existing dataset at %s", output_path)
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"Dataset already exists at {output_path}. Use --overwrite to replace.")

    dataset = LeRobotDataset.create(
        repo_id=config.repo_name,
        robot_type="panda",
        fps=config.fps,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_frames = 0
    for demo_path in tqdm(demo_files, desc="Converting episodes"):
        with h5py.File(demo_path, "r") as f:
            instruction = f.attrs.get("instruction", "do something")
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8")

            n_steps = f[f"observations/robot_state/joint_positions"].shape[0]

            for t in range(n_steps):
                # Images
                images = {}
                for cam_id, feature_name in CAMERA_MAP.items():
                    img = f[f"observations/image/{cam_id}"][t]
                    if img.shape[:2] != (IMAGE_SIZE[1], IMAGE_SIZE[0]):
                        img = resize_image(img)
                    images[feature_name] = img

                # State
                joint_pos = np.asarray(f["observations/robot_state/joint_positions"][t], dtype=np.float32)
                gripper_pos = np.asarray(f["observations/robot_state/gripper_position"][t], dtype=np.float32)
                if gripper_pos.ndim == 0:
                    gripper_pos = gripper_pos[np.newaxis]

                # Actions
                action_joints = np.asarray(f[f"action/{config.action_key}"][t], dtype=np.float32)
                action_gripper = np.asarray(f["action/target_gripper_position"][t], dtype=np.float32)
                if action_gripper.ndim == 0:
                    action_gripper = action_gripper[np.newaxis]
                actions = np.concatenate([action_joints, action_gripper])

                dataset.add_frame(
                    {
                        **images,
                        "joint_position": joint_pos,
                        "gripper_position": gripper_pos,
                        "actions": actions,
                        "task": instruction,
                    }
                )
            dataset.save_episode()
            total_frames += n_steps

    log.info("Done. %d episodes, %d total frames saved to %s", len(demo_files), total_frames, output_path)


if __name__ == "__main__":
    config = tyro.cli(ConvertConfig)
    main(config)
