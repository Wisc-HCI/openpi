"""Convert RealSense DROID-GUI trajectories to OpenPI's DROID LeRobot format."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize(size, resample=Image.BICUBIC))


def hdf5_length(group: h5py.Group) -> int:
    for value in group.values():
        if isinstance(value, h5py.Group):
            return hdf5_length(value)
        return len(value)
    raise ValueError("Could not infer trajectory length from empty HDF5 file")


def lerobot_hdf5_length(trajectory: h5py.File) -> int:
    required_paths = (
        "observation/robot_state/joint_positions",
        "observation/robot_state/gripper_position",
        "observation/controller_info/movement_enabled",
        "action/joint_velocity",
        "action/gripper_position",
    )
    return min(len(trajectory[path]) for path in required_paths)


def load_hdf5_step(group: h5py.Group, index: int) -> dict:
    result = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            result[key] = load_hdf5_step(value, index)
        else:
            result[key] = value[index]
    return result


def load_lerobot_step(trajectory: h5py.File, index: int) -> dict:
    controller_info = {}
    if "controller_info" in trajectory["observation"]:
        controller_info = {key: value[index] for key, value in trajectory["observation"]["controller_info"].items()}

    return {
        "observation": {
            "controller_info": controller_info,
            "robot_state": {
                "joint_positions": trajectory["observation"]["robot_state"]["joint_positions"][index],
                "gripper_position": trajectory["observation"]["robot_state"]["gripper_position"][index],
            },
        },
        "action": {
            "joint_velocity": trajectory["action"]["joint_velocity"][index],
            "gripper_position": trajectory["action"]["gripper_position"][index],
        },
    }


def read_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr[..., :3], cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_language_instruction(episode_path: Path, attrs: h5py.AttributeManager, fallback_task: str) -> str:
    for key in ("current_task", "language_instruction"):
        if attrs.get(key):
            value = attrs[key]
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    metadata_path = episode_path.parent / "metadata_openpi.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        for key in ("language_instruction", "current_task"):
            if metadata.get(key):
                return metadata[key]

    return fallback_task


def camera_ids_from_step(step: dict) -> tuple[str, str]:
    camera_type = step["observation"]["camera_type"]
    wrist_ids = [cam_id for cam_id, camera_type_id in camera_type.items() if int(camera_type_id) == 0]
    exterior_ids = [cam_id for cam_id, camera_type_id in camera_type.items() if int(camera_type_id) != 0]
    if not wrist_ids:
        raise ValueError(f"No wrist camera found in camera_type: {camera_type}")
    if not exterior_ids:
        raise ValueError(f"No exterior camera found in camera_type: {camera_type}")
    return exterior_ids[0], wrist_ids[0]


def validate_episode(episode_path: Path) -> None:
    recording_dir = episode_path.parent / "recordings" / "MP4"
    if not recording_dir.exists():
        raise FileNotFoundError(recording_dir)

    with h5py.File(episode_path, "r") as trajectory:
        if "action" not in trajectory:
            raise KeyError(f"{episode_path}: missing action group")
        for key in ("joint_velocity", "gripper_position"):
            if key not in trajectory["action"]:
                raise KeyError(f"{episode_path}: missing action/{key}")

        first_step = load_hdf5_step(trajectory, 0)
        exterior_id, wrist_id = camera_ids_from_step(first_step)
        for camera_id in (exterior_id, wrist_id):
            if not (recording_dir / f"{camera_id}.mp4").exists():
                raise FileNotFoundError(recording_dir / f"{camera_id}.mp4")


def main(
    data_dir: Path,
    repo_id: str = "kindred/realsense_droid",
    *,
    fps: int = 15,
    image_width: int = 320,
    image_height: int = 180,
    fallback_task: str = "Do the task",
    include_failures: bool = False,
    drop_skipped: bool = False,
    skip_invalid: bool = False,
    overwrite: bool = False,
    push_to_hub: bool = False,
    private: bool = True,
    dry_run: bool = False,
) -> None:
    episode_paths = sorted((data_dir / "success").glob("**/trajectory.h5"))
    if include_failures:
        episode_paths.extend(sorted((data_dir / "failure").glob("**/trajectory.h5")))
    if not episode_paths:
        raise FileNotFoundError(f"No trajectory.h5 files found under {data_dir}")

    print(f"Found {len(episode_paths)} trajectories")
    if skip_invalid:
        valid_episode_paths = []
        skipped_count = 0
        for episode_path in episode_paths:
            try:
                validate_episode(episode_path)
            except Exception as err:
                skipped_count += 1
                print(f"Skipping invalid trajectory {episode_path}: {err}")
            else:
                valid_episode_paths.append(episode_path)
        episode_paths = valid_episode_paths
        print(f"Using {len(episode_paths)} valid trajectories; skipped {skipped_count}")
        if not episode_paths:
            raise ValueError("No valid trajectories found")

    if dry_run:
        valid_count = 0
        for episode_path in episode_paths[:5]:
            validate_episode(episode_path)
            with h5py.File(episode_path, "r") as trajectory:
                print(episode_path, "length=", lerobot_hdf5_length(trajectory))
            valid_count += 1
        print(f"Validated {valid_count} sample trajectories")
        return

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass overwrite=True to replace it.")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (image_height, image_width, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (image_height, image_width, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (image_height, image_width, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for episode_path in tqdm(episode_paths, desc="Converting trajectories"):
        recording_dir = episode_path.parent / "recordings" / "MP4"
        if not recording_dir.exists():
            raise FileNotFoundError(recording_dir)

        videos = {path.stem: read_video(path) for path in recording_dir.glob("*.mp4")}
        with h5py.File(episode_path, "r") as trajectory:
            horizon = lerobot_hdf5_length(trajectory)
            task = load_language_instruction(episode_path, trajectory.attrs, fallback_task)

            first_step = load_hdf5_step(trajectory, 0)
            exterior_id, wrist_id = camera_ids_from_step(first_step)
            if exterior_id not in videos:
                raise KeyError(f"Missing exterior video {exterior_id}.mp4 in {recording_dir}")
            if wrist_id not in videos:
                raise KeyError(f"Missing wrist video {wrist_id}.mp4 in {recording_dir}")

            horizon = min(horizon, len(videos[exterior_id]), len(videos[wrist_id]))
            frames_written = 0
            for index in range(horizon):
                step = load_lerobot_step(trajectory, index)
                controller_info = step["observation"].get("controller_info", {})
                if drop_skipped and not bool(controller_info.get("movement_enabled", True)):
                    continue

                exterior_image = resize_image(videos[exterior_id][index], (image_width, image_height))
                wrist_image = resize_image(videos[wrist_id][index], (image_width, image_height))
                robot_state = step["observation"]["robot_state"]
                action = step["action"]

                dataset.add_frame(
                    {
                        "exterior_image_1_left": exterior_image,
                        "exterior_image_2_left": exterior_image,
                        "wrist_image_left": wrist_image,
                        "joint_position": np.asarray(robot_state["joint_positions"], dtype=np.float32),
                        "gripper_position": np.asarray([robot_state["gripper_position"]], dtype=np.float32),
                        "actions": np.concatenate(
                            [
                                np.asarray(action["joint_velocity"], dtype=np.float32),
                                np.asarray([action["gripper_position"]], dtype=np.float32),
                            ]
                        ),
                        "task": task,
                    }
                )
                frames_written += 1

            if frames_written == 0:
                raise ValueError(f"No frames written for {episode_path}")
            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["droid", "panda", "realsense"], private=private, push_videos=True, license="apache-2.0"
        )

    metadata = {
        "source_data_dir": str(data_dir),
        "episode_count": len(episode_paths),
        "include_failures": include_failures,
        "drop_skipped": drop_skipped,
        "skip_invalid": skip_invalid,
    }
    (output_path / "realsense_droid_conversion_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved LeRobot dataset to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
