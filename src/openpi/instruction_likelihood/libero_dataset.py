"""Read observation/action chunks from standard LIBERO demonstration HDF5 files."""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import json
import pathlib

import h5py
import numpy as np


@dataclasses.dataclass(frozen=True)
class DemoRef:
    path: pathlib.Path
    episode: str
    task_id: str
    language: str | None
    length: int

    @property
    def demo_id(self) -> str:
        return f"{self.task_id}:{self.episode}"


@dataclasses.dataclass(frozen=True)
class ActionChunk:
    start: int
    stop: int
    base_image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    actions: np.ndarray


def _decode_problem_info(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        info = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return None
    instruction = info.get("language_instruction")
    if isinstance(instruction, list):
        instruction = "".join(instruction)
    if instruction is None:
        return None
    return str(instruction).strip().strip('"')


def task_id_from_hdf5(path: pathlib.Path, data_group: h5py.Group) -> str:
    bddl_name = data_group.attrs.get("bddl_file_name")
    if bddl_name is not None:
        if isinstance(bddl_name, bytes):
            bddl_name = bddl_name.decode("utf-8")
        return pathlib.Path(str(bddl_name)).stem
    stem = path.stem
    for suffix in ("_demo", "_demonstration"):
        stem = stem.removesuffix(suffix)
    return stem


def discover_demos(root: pathlib.Path | str) -> list[DemoRef]:
    root = pathlib.Path(root)
    paths = [root] if root.is_file() else sorted({*root.rglob("*.hdf5"), *root.rglob("*.h5")})
    demos: list[DemoRef] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                continue
            data = handle["data"]
            task_id = task_id_from_hdf5(path, data)
            language = _decode_problem_info(data.attrs.get("problem_info"))
            for episode in sorted(data.keys(), key=_episode_sort_key):
                group = data[episode]
                if isinstance(group, h5py.Group) and "actions" in group:
                    demos.append(DemoRef(path, episode, task_id, language, int(group["actions"].shape[0])))
    return demos


def _episode_sort_key(name: str) -> tuple[int, str]:
    tail = name.rsplit("_", 1)[-1]
    return (int(tail) if tail.isdigit() else 10**9, name)


def _first_dataset(group: h5py.Group, names: tuple[str, ...]) -> h5py.Dataset:
    for name in names:
        if name in group:
            value = group[name]
            if isinstance(value, h5py.Dataset):
                return value
    raise KeyError(f"None of {names} found below {group.name}; available keys: {sorted(group.keys())}")


def _state_at(obs: h5py.Group, index: int) -> np.ndarray:
    if "ee_states" in obs and "gripper_states" in obs:
        state = np.concatenate([np.asarray(obs["ee_states"][index]), np.asarray(obs["gripper_states"][index])])
    elif "ee_pos" in obs and "ee_ori" in obs and "gripper_states" in obs:
        state = np.concatenate(
            [
                np.asarray(obs["ee_pos"][index]),
                np.asarray(obs["ee_ori"][index]),
                np.asarray(obs["gripper_states"][index]),
            ]
        )
    elif "state" in obs:
        state = np.asarray(obs["state"][index])
    else:
        raise KeyError(
            f"Cannot construct checkpoint-compatible 8-D state below {obs.name}; available keys: {sorted(obs.keys())}"
        )
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (8,):
        raise ValueError(f"Expected 8-D [eef position, axis-angle, gripper qpos] state, got {state.shape}")
    return state


def iter_action_chunks(
    demo: DemoRef,
    *,
    action_horizon: int,
    stride: int | None = None,
    rotate_images_180: bool = True,
) -> Iterator[ActionChunk]:
    """Yield only complete native-horizon chunks; no synthetic tail padding is used."""
    # OpenPI's LeRobot loader constructs a future action sequence for every frame
    # index, so stride 1 is the native training-sample granularity.
    stride = 1 if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive")
    with h5py.File(demo.path, "r") as handle:
        group = handle[f"data/{demo.episode}"]
        obs = group["obs"]
        base = _first_dataset(obs, ("agentview_rgb", "agentview_image", "image"))
        wrist = _first_dataset(obs, ("eye_in_hand_rgb", "robot0_eye_in_hand_image", "wrist_image"))
        actions = group["actions"]
        length = min(len(actions), len(base), len(wrist))
        for start in range(0, length - action_horizon + 1, stride):
            base_image = np.asarray(base[start])
            wrist_image = np.asarray(wrist[start])
            if rotate_images_180:
                base_image = np.ascontiguousarray(base_image[::-1, ::-1])
                wrist_image = np.ascontiguousarray(wrist_image[::-1, ::-1])
            action_chunk = np.asarray(actions[start : start + action_horizon], dtype=np.float32)
            if action_chunk.shape != (action_horizon, 7):
                raise ValueError(f"Expected action chunk {(action_horizon, 7)}, got {action_chunk.shape}")
            yield ActionChunk(
                start=start,
                stop=start + action_horizon,
                base_image=base_image,
                wrist_image=wrist_image,
                state=_state_at(obs, start),
                actions=action_chunk,
            )


def load_demo_actions(demo: DemoRef) -> np.ndarray:
    """Load a demo's complete raw 7-D action sequence."""
    with h5py.File(demo.path, "r") as handle:
        actions = np.asarray(handle[f"data/{demo.episode}/actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected [time,7] actions for {demo.demo_id}, got {actions.shape}")
    return actions
