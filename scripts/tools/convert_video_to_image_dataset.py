"""Convert a video-based LeRobot dataset to image-based DROID format.

Reads an existing LeRobot v2 dataset that stores camera observations as
MP4 videos and re-writes it as a new dataset with image storage.  This
avoids the torchcodec / FFmpeg dependency at training time.

The script also remaps feature names to match the DROID convention that
OpenPI's repack transforms expect:

    observation.images.exterior_image_1_left  ->  exterior_image_1_left
    observation.images.exterior_image_2_left  ->  exterior_image_2_left
    observation.images.wrist_image_left       ->  wrist_image_left
    observation.state[0:7]                    ->  joint_position
    observation.state[-1]                     ->  gripper_position
    action[0:N]                               ->  actions  (unchanged dim)

Usage:
    python scripts/tools/convert_video_to_image_dataset.py \
        --src-repo-id Pick_up_the_pen_and_place_it_in_the_trashbin_lerobot \
        --dst-repo-id local/pick_pen_image
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

# LeRobot video key -> flat DROID feature name
CAMERA_MAP = {
    "observation.images.exterior_image_1_left": "exterior_image_1_left",
    "observation.images.exterior_image_2_left": "exterior_image_2_left",
    "observation.images.wrist_image_left": "wrist_image_left",
}


def decode_episode_video(video_path: Path) -> list[np.ndarray]:
    """Decode all frames from a video file using pyav."""
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return frames


@dataclasses.dataclass
class ConvertConfig:
    src_repo_id: str
    """Source LeRobot dataset repo ID (video-based)."""

    dst_repo_id: str = ""
    """Destination repo ID.  Defaults to ``local/<src_repo_id>_image``."""

    image_size: tuple[int, int] | None = None
    """Optional (width, height) to resize images.  None keeps original size."""

    overwrite: bool = False
    """Delete existing destination dataset before converting."""


def main(config: ConvertConfig) -> None:
    src_root = HF_LEROBOT_HOME / config.src_repo_id
    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")

    # Load source metadata
    with open(src_root / "meta" / "info.json") as f:
        src_info = json.load(f)

    src_features = src_info["features"]
    fps = src_info["fps"]
    total_episodes = src_info["total_episodes"]
    chunks_size = src_info["chunks_size"]

    # Determine camera keys and image shape
    video_keys = [k for k, v in src_features.items() if v.get("dtype") == "video"]
    if not video_keys:
        log.info("Source dataset has no video keys — nothing to convert.")
        return

    sample_shape = tuple(src_features[video_keys[0]]["shape"])  # (H, W, 3)

    if config.image_size is not None:
        img_h, img_w = config.image_size[1], config.image_size[0]
    else:
        img_h, img_w = sample_shape[0], sample_shape[1]

    dst_repo_id = config.dst_repo_id or f"local/{config.src_repo_id}_image"
    dst_root = HF_LEROBOT_HOME / dst_repo_id
    if dst_root.exists():
        if config.overwrite:
            log.info("Removing existing dataset at %s", dst_root)
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(f"Dataset already exists at {dst_root}. Use --overwrite.")

    # Determine state decomposition
    has_obs_state = "observation.state" in src_features
    if has_obs_state:
        state_dim = src_features["observation.state"]["shape"][0]
        # Convention: first 7 dims = joint_position, last dim = gripper_position
        joint_dim = 7
        log.info(
            "observation.state (%d-dim) -> joint_position (%d) + gripper_position (1)",
            state_dim,
            joint_dim,
        )

    # Determine action key
    action_key = "actions" if "actions" in src_features else "action"
    action_dim = src_features[action_key]["shape"][0]

    # Build destination feature spec
    features = {}
    for vk in video_keys:
        dst_name = CAMERA_MAP.get(vk, vk.replace("observation.images.", ""))
        features[dst_name] = {
            "dtype": "image",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channel"],
        }

    if has_obs_state:
        features["joint_position"] = {
            "dtype": "float32",
            "shape": (joint_dim,),
            "names": ["joint_position"],
        }
        features["gripper_position"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_position"],
        }

    features["actions"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": ["actions"],
    }

    # Load tasks
    tasks = []
    with open(src_root / "meta" / "tasks.jsonl") as f:
        for line in f:
            tasks.append(json.loads(line)["task"])

    # Load episodes metadata
    episodes_meta = []
    with open(src_root / "meta" / "episodes.jsonl") as f:
        for line in f:
            episodes_meta.append(json.loads(line))

    # Create destination dataset
    dataset = LeRobotDataset.create(
        repo_id=dst_repo_id,
        robot_type=src_info.get("robot_type", "unknown"),
        fps=fps,
        features=features,
        image_writer_threads=4,
        image_writer_processes=4,
    )

    total_frames = 0
    for ep_idx in tqdm(range(total_episodes), desc="Converting episodes"):
        ep_meta = episodes_meta[ep_idx]
        chunk_idx = ep_idx // chunks_size

        # Read parquet data
        parquet_path = src_root / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet_path)
        n_frames = len(df)

        # Decode video frames for each camera
        camera_frames = {}
        for vk in video_keys:
            dst_name = CAMERA_MAP.get(vk, vk.replace("observation.images.", ""))
            video_path = src_root / f"videos/chunk-{chunk_idx:03d}/{vk}/episode_{ep_idx:06d}.mp4"
            frames = decode_episode_video(video_path)
            if len(frames) != n_frames:
                log.warning(
                    "Episode %d: video %s has %d frames but parquet has %d rows. Using min.",
                    ep_idx, vk, len(frames), n_frames,
                )
                n_frames = min(n_frames, len(frames))
            camera_frames[dst_name] = frames

        # Determine task string for this episode
        task_indices = df["task_index"].values
        task_str = tasks[task_indices[0]] if len(tasks) > 0 else "do something"

        for t in range(n_frames):
            frame_data = {}

            # Images
            for cam_name, frames in camera_frames.items():
                img = frames[t]
                if config.image_size is not None and (img.shape[1], img.shape[0]) != config.image_size:
                    from PIL import Image

                    img = np.array(
                        Image.fromarray(img).resize(config.image_size, resample=Image.BICUBIC)
                    )
                frame_data[cam_name] = img

            # State decomposition
            if has_obs_state:
                state = np.asarray(df["observation.state"].iloc[t], dtype=np.float32)
                frame_data["joint_position"] = state[:joint_dim]
                frame_data["gripper_position"] = state[-1:]

            # Actions
            action = np.asarray(df[action_key].iloc[t], dtype=np.float32)
            frame_data["actions"] = action

            frame_data["task"] = task_str
            dataset.add_frame(frame_data)

        # LeRobot bug: shape=(1,) features are mapped to scalar
        # datasets.Value, but validate_frame requires (1,) ndarrays.
        for key in dataset.features:
            if dataset.features[key].get("shape") == (1,):
                buf = dataset.episode_buffer.get(key)
                if isinstance(buf, list):
                    dataset.episode_buffer[key] = [
                        v.item() if isinstance(v, np.ndarray) else v for v in buf
                    ]

        dataset.save_episode()
        total_frames += n_frames

    log.info(
        "Done. %d episodes, %d total frames saved to %s",
        total_episodes,
        total_frames,
        dst_root,
    )


if __name__ == "__main__":
    config = tyro.cli(ConvertConfig)
    main(config)
