"""Load raw RealSense-DROID trajectories for instruction-compatibility evaluation.

The loader mirrors ``convert_realsense_droid_data_to_lerobot.py`` without writing a
LeRobot dataset.  This is useful for held-out trajectories: evaluation should not
need to add them to (or overwrite) the dataset that was used for fine-tuning.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image

LEFT_INSTRUCTION = "pick up the left block"
RIGHT_INSTRUCTION = "pick up the right block"
CANDIDATE_INSTRUCTIONS = (LEFT_INSTRUCTION, RIGHT_INSTRUCTION)

FAILURE_CONDITIONS = (
    "C0_natural_direct",
    "C1_mild_exaggeration",
    "C2_strong_exaggeration",
    "C3_extreme_exaggeration",
)


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    path: Path
    split: str
    episode_id: str
    instruction: str
    target_side: str
    pair_index: int
    condition: str


@dataclasses.dataclass(frozen=True)
class EpisodeArrays:
    joint_positions: np.ndarray
    gripper_positions: np.ndarray
    actions: np.ndarray
    cartesian_positions: np.ndarray
    movement_enabled: np.ndarray
    motion_start: int
    grasp_start: int

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


@dataclasses.dataclass(frozen=True)
class DroidActionChunk:
    episode: EpisodeSpec
    chunk_index: int
    phase: str
    start: int
    stop: int
    executed_steps: int
    progress_pregrasp: float
    progress_full: float
    exterior_image: np.ndarray
    wrist_image: np.ndarray
    joint_position: np.ndarray
    gripper_position: np.ndarray
    actions: np.ndarray
    cartesian_position: np.ndarray


def _target_side(instruction: str) -> str:
    if instruction == LEFT_INSTRUCTION:
        return "left"
    if instruction == RIGHT_INSTRUCTION:
        return "right"
    raise ValueError(f"Unexpected instruction {instruction!r}; expected one of {CANDIDATE_INSTRUCTIONS}")


def discover_episode_pairs(root: Path | str, split: str) -> list[EpisodeSpec]:
    """Discover chronologically adjacent left/right pairs below one date directory."""
    root = Path(root)
    paths = sorted(root.glob("*/trajectory.h5"))
    if not paths:
        raise FileNotFoundError(f"No trajectory.h5 files found below {root}")
    if len(paths) % 2:
        raise ValueError(f"Expected an even number of paired trajectories below {root}, got {len(paths)}")

    specs: list[EpisodeSpec] = []
    for pair_index in range(len(paths) // 2):
        pair_paths = paths[2 * pair_index : 2 * pair_index + 2]
        pair_instructions = []
        for path in pair_paths:
            metadata_path = path.parent / "metadata_openpi.json"
            metadata = json.loads(metadata_path.read_text())
            instruction = str(metadata.get("language_instruction") or metadata.get("current_task") or "")
            pair_instructions.append(instruction)
        if set(pair_instructions) != set(CANDIDATE_INSTRUCTIONS):
            raise ValueError(
                f"Pair {pair_index} below {root} must contain one left and one right instruction, "
                f"got {pair_instructions}"
            )

        if split == "train":
            condition = "train_natural"
        elif split == "test":
            if len(paths) != 8:
                raise ValueError(f"Expected exactly 8 held-out trajectories below {root}, got {len(paths)}")
            condition = FAILURE_CONDITIONS[pair_index]
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        for path, instruction in zip(pair_paths, pair_instructions, strict=True):
            specs.append(
                EpisodeSpec(
                    path=path,
                    split=split,
                    episode_id=path.parent.name,
                    instruction=instruction,
                    target_side=_target_side(instruction),
                    pair_index=pair_index,
                    condition=condition,
                )
            )
    return specs


def load_episode_arrays(episode: EpisodeSpec, *, grasp_threshold: float = 0.1) -> EpisodeArrays:
    with h5py.File(episode.path, "r") as trajectory:
        required_paths = (
            "observation/robot_state/joint_positions",
            "observation/robot_state/gripper_position",
            "observation/robot_state/cartesian_position",
            "observation/controller_info/movement_enabled",
            "action/joint_velocity",
            "action/gripper_position",
        )
        length = min(len(trajectory[path]) for path in required_paths)
        joint_positions = np.asarray(trajectory["observation/robot_state/joint_positions"][:length], dtype=np.float32)
        gripper_positions = np.asarray(
            trajectory["observation/robot_state/gripper_position"][:length], dtype=np.float32
        )
        cartesian_positions = np.asarray(
            trajectory["observation/robot_state/cartesian_position"][:length], dtype=np.float32
        )
        movement_enabled = np.asarray(trajectory["observation/controller_info/movement_enabled"][:length], dtype=bool)
        joint_velocity = np.asarray(trajectory["action/joint_velocity"][:length], dtype=np.float32)
        gripper_action = np.asarray(trajectory["action/gripper_position"][:length], dtype=np.float32)

    actions = np.concatenate([joint_velocity, gripper_action[:, None]], axis=1)
    enabled_indices = np.flatnonzero(movement_enabled)
    if not len(enabled_indices):
        raise ValueError(f"Trajectory has no movement-enabled steps: {episode.path}")
    motion_start = int(enabled_indices[0])

    grasp_indices = np.flatnonzero(gripper_action[motion_start:] > grasp_threshold)
    if not len(grasp_indices):
        raise ValueError(
            f"Could not find gripper closing onset above {grasp_threshold} after frame {motion_start}: {episode.path}"
        )
    grasp_start = motion_start + int(grasp_indices[0])
    if grasp_start <= motion_start:
        raise ValueError(f"Invalid motion/grasp interval [{motion_start}, {grasp_start}) in {episode.path}")

    return EpisodeArrays(
        joint_positions=joint_positions,
        gripper_positions=gripper_positions,
        actions=actions,
        cartesian_positions=cartesian_positions,
        movement_enabled=movement_enabled,
        motion_start=motion_start,
        grasp_start=grasp_start,
    )


def _camera_ids(episode: EpisodeSpec) -> tuple[str, str]:
    with h5py.File(episode.path, "r") as trajectory:
        camera_type = {
            camera_id: int(np.asarray(values[0]).item())
            for camera_id, values in trajectory["observation/camera_type"].items()
        }
    wrist_ids = [camera_id for camera_id, type_id in camera_type.items() if type_id == 0]
    exterior_ids = [camera_id for camera_id, type_id in camera_type.items() if type_id != 0]
    if len(wrist_ids) != 1 or len(exterior_ids) != 1:
        raise ValueError(f"Expected one wrist and one exterior camera, got {camera_type} in {episode.path}")
    return exterior_ids[0], wrist_ids[0]


def _read_and_resize_video(path: Path, *, width: int = 320, height: int = 180) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr[..., :3], cv2.COLOR_BGR2RGB)
        # Match the conversion script used to create peopleandrobots/spatial.
        frames.append(np.asarray(Image.fromarray(frame_rgb).resize((width, height), resample=Image.BICUBIC)))
    capture.release()
    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    return frames


def _phase_boundaries(arrays: EpisodeArrays, horizon: int) -> list[tuple[str, int, int]]:
    boundaries: list[tuple[str, int, int]] = []
    for phase, begin, end in (
        ("pregrasp", arrays.motion_start, arrays.grasp_start),
        ("postgrasp", arrays.grasp_start, arrays.length),
    ):
        boundaries.extend((phase, start, min(start + horizon, end)) for start in range(begin, end, horizon))
    return boundaries


def iter_action_chunks(
    episode: EpisodeSpec,
    *,
    action_horizon: int,
    arrays: EpisodeArrays | None = None,
) -> Iterator[DroidActionChunk]:
    """Yield non-overlapping chunks, splitting exactly at the grasp boundary.

    The last chunk of each phase is padded with its final action. ``executed_steps``
    masks this padding inside ``Pi0.score_actions``.  A chunk's evidence is assigned
    to ``stop`` so it never leaks unseen future actions into an earlier prefix.
    """
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    arrays = arrays or load_episode_arrays(episode)
    exterior_id, wrist_id = _camera_ids(episode)
    recording_dir = episode.path.parent / "recordings" / "MP4"
    exterior_frames = _read_and_resize_video(recording_dir / f"{exterior_id}.mp4")
    wrist_frames = _read_and_resize_video(recording_dir / f"{wrist_id}.mp4")
    available_length = min(arrays.length, len(exterior_frames), len(wrist_frames))
    if available_length != arrays.length:
        raise ValueError(
            f"HDF5/video length mismatch for {episode.path}: HDF5={arrays.length}, "
            f"exterior={len(exterior_frames)}, wrist={len(wrist_frames)}"
        )

    pregrasp_duration = arrays.grasp_start - arrays.motion_start
    full_duration = arrays.length - arrays.motion_start
    for chunk_index, (phase, start, stop) in enumerate(_phase_boundaries(arrays, action_horizon)):
        executed_steps = stop - start
        actions = arrays.actions[start:stop]
        if executed_steps < action_horizon:
            actions = np.pad(actions, ((0, action_horizon - executed_steps), (0, 0)), mode="edge")
        if actions.shape != (action_horizon, 8):
            raise ValueError(f"Expected action chunk {(action_horizon, 8)}, got {actions.shape}")

        progress_pregrasp = min(max((stop - arrays.motion_start) / pregrasp_duration, 0.0), 1.0)
        progress_full = min(max((stop - arrays.motion_start) / full_duration, 0.0), 1.0)
        yield DroidActionChunk(
            episode=episode,
            chunk_index=chunk_index,
            phase=phase,
            start=start,
            stop=stop,
            executed_steps=executed_steps,
            progress_pregrasp=float(progress_pregrasp),
            progress_full=float(progress_full),
            exterior_image=exterior_frames[start],
            wrist_image=wrist_frames[start],
            joint_position=arrays.joint_positions[start],
            gripper_position=np.asarray([arrays.gripper_positions[start]], dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            cartesian_position=arrays.cartesian_positions[min(stop - 1, arrays.length - 1), :3],
        )


def validate_expected_dataset(train: Sequence[EpisodeSpec], test: Sequence[EpisodeSpec]) -> None:
    if len(train) != 32:
        raise ValueError(f"Expected 32 training trajectories, got {len(train)}")
    if len(test) != 8:
        raise ValueError(f"Expected 8 held-out trajectories, got {len(test)}")
    if {episode.condition for episode in test} != set(FAILURE_CONDITIONS):
        raise ValueError("Held-out trajectories do not cover all four expected conditions")
