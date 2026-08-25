"""On-disk contract for scripted LIBERO spatial-legibility rollouts.

The simulator and the pi0.5 observer intentionally run in different Python
environments.  This module is the narrow boundary between them: the simulator
writes the exact observations and controller commands seen online, and the
observer reconstructs ten non-overlapping, checkpoint-compatible action chunks
without importing LIBERO or robosuite.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import dataclasses
import hashlib
import json
import os
import pathlib
from typing import Any

import h5py
import numpy as np

from openpi.instruction_likelihood.libero_dataset import ActionChunk

SCHEMA_VERSION = 1
PREGRASP_STEPS = 100
ACTION_HORIZON = 10

_TIME_ARRAY_WIDTHS = {
    "actions": 7,
    "states": 8,
    "desired_eef_pos": 3,
    "target_pose": 7,
    "distractor_pose": 7,
}

_REQUIRED_METADATA = {
    "episode_id",
    "geometry_id",
    "layout_id",
    "target_identity",
    "distractor_identity",
    "target_side",
    "target_prompt",
    "distractor_prompt",
    "target_object",
    "distractor_object",
    "legibility_level",
    "separation_m",
    "simulator_seed",
    "initial_sim_state_sha256",
    "frozen_geometry_sha256",
    "pregrasp_steps",
    "task_success",
    "collision_free",
}


@dataclasses.dataclass(frozen=True)
class RecordedEpisode:
    path: pathlib.Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def episode_id(self) -> str:
        return str(self.metadata["episode_id"])


def _json_default(value: object) -> object:
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return deterministic JSON suitable for hashes and HDF5 attributes."""
    return json.dumps(payload, default=_json_default, sort_keys=True, separators=(",", ":"), allow_nan=False)


def frozen_manifest_digest(payload: Mapping[str, Any]) -> str:
    """Hash a frozen-geometry manifest, excluding its self-referential digest."""
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def load_frozen_manifest(path: pathlib.Path | str) -> dict[str, Any]:
    """Load a pre-observer manifest and fail closed on mutation or incompleteness."""
    path = pathlib.Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "frozen_before_pi05",
        "candidate_identities",
        "candidate_prompts",
        "candidate_objects",
        "geometries",
        "legibility_levels",
        "pregrasp_steps",
        "sha256",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Frozen manifest is missing keys: {missing}")
    if payload["frozen_before_pi05"] is not True:
        raise ValueError("Manifest does not assert frozen_before_pi05=true")
    if int(payload["pregrasp_steps"]) != PREGRASP_STEPS:
        raise ValueError("Frozen manifest must specify exactly 100 pre-grasp steps")
    if len(payload["candidate_identities"]) != 2 or len(payload["candidate_prompts"]) != 2:
        raise ValueError("Frozen manifest must define exactly two canonical candidates")
    expected = frozen_manifest_digest(payload)
    if payload["sha256"] != expected:
        raise ValueError(f"Frozen manifest digest mismatch: recorded={payload['sha256']} computed={expected}")
    return payload


def _validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    missing = sorted(_REQUIRED_METADATA - result.keys())
    if missing:
        raise ValueError(f"Episode metadata is missing required keys: {missing}")
    if int(result["pregrasp_steps"]) != PREGRASP_STEPS:
        raise ValueError(f"pregrasp_steps must be exactly {PREGRASP_STEPS}")
    if float(result["separation_m"]) <= 0.0:
        raise ValueError("separation_m must be positive")
    if result["target_identity"] == result.get("distractor_identity"):
        raise ValueError("target_identity and distractor_identity must differ")
    initial_digest = str(result["initial_sim_state_sha256"])
    if len(initial_digest) != 64 or any(character not in "0123456789abcdef" for character in initial_digest):
        raise ValueError("initial_sim_state_sha256 must be a lowercase SHA256 digest")
    return result


def _validate_arrays(
    arrays: Mapping[str, np.ndarray], *, pregrasp_steps: int, require_images: bool = True
) -> dict[str, np.ndarray]:
    result = {name: np.asarray(value) for name, value in arrays.items()}
    required = set(_TIME_ARRAY_WIDTHS) | {
        "progress",
        "sim_states",
        "joint_positions",
        "actual_eef_pos",
        "actual_eef_quat",
    }
    if require_images:
        required |= {"base_images", "wrist_images"}
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(f"Episode arrays are missing required keys: {missing}")

    time_steps = int(result["actions"].shape[0])
    if time_steps < pregrasp_steps:
        raise ValueError(f"Episode has {time_steps} actions, fewer than {pregrasp_steps} pre-grasp steps")
    for name, width in _TIME_ARRAY_WIDTHS.items():
        if result[name].shape != (time_steps, width):
            raise ValueError(f"{name} must have shape {(time_steps, width)}, got {result[name].shape}")
    for name in ("base_images", "wrist_images"):
        if name not in result:
            continue
        value = result[name]
        if value.ndim != 4 or value.shape[0] != time_steps or value.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [time,height,width,3], got {value.shape}")
        if value.dtype != np.uint8:
            raise ValueError(f"{name} must use uint8 pixels, got {value.dtype}")
    if result["progress"].shape != (time_steps,):
        raise ValueError(f"progress must have shape {(time_steps,)}, got {result['progress'].shape}")
    if not np.all(np.isfinite(result["progress"])):
        raise ValueError("progress contains non-finite values")

    for name in ("actual_eef_pos", "actual_eef_quat"):
        if name not in result:
            raise ValueError(f"Episode arrays are missing required key: {name}")
    if result["actual_eef_pos"].shape != (time_steps + 1, 3):
        raise ValueError("actual_eef_pos must contain the pre-action pose plus every post-action pose")
    if result["actual_eef_quat"].shape != (time_steps + 1, 4):
        raise ValueError("actual_eef_quat must contain the pre-action pose plus every post-action pose")
    for name in ("sim_states", "joint_positions"):
        if result[name].ndim != 2 or result[name].shape[0] != time_steps or result[name].shape[1] == 0:
            raise ValueError(f"{name} must have shape [time,nonzero_width], got {result[name].shape}")
    if result["sim_states"].dtype != np.float64:
        raise ValueError(f"sim_states must use float64 for exact MuJoCo replay, got {result['sim_states'].dtype}")

    for name, value in result.items():
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_initial_state_digest(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    initial_state = np.asarray(arrays["sim_states"][0], dtype="<f8")
    computed = hashlib.sha256(initial_state.tobytes(order="C")).hexdigest()
    recorded = str(metadata["initial_sim_state_sha256"])
    if recorded != computed:
        raise ValueError(
            f"initial_sim_state_sha256 does not match sim_states[0]: recorded={recorded}, computed={computed}"
        )


def write_episode(
    path: pathlib.Path | str,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> pathlib.Path:
    """Atomically write one compressed rollout.

    A ``.partial`` sibling is used so an interrupted simulator job never looks
    like a complete episode to a resumed scoring job.
    """
    path = pathlib.Path(path)
    if path.suffix not in {".h5", ".hdf5"}:
        raise ValueError("Recorded episodes must use a .h5 or .hdf5 suffix")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    metadata_dict = _validate_metadata(metadata)
    array_dict = _validate_arrays(arrays, pregrasp_steps=int(metadata_dict["pregrasp_steps"]))
    _validate_initial_state_digest(metadata_dict, array_dict)

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    try:
        with h5py.File(partial, "w") as handle:
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["metadata_json"] = canonical_json(metadata_dict)
            for name, value in array_dict.items():
                kwargs: dict[str, object] = {}
                if value.ndim > 0 and value.size > 0:
                    kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
                handle.create_dataset(name, data=value, **kwargs)
            handle.flush()
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return path


def read_episode(path: pathlib.Path | str, *, load_images: bool = True) -> RecordedEpisode:
    path = pathlib.Path(path)
    with h5py.File(path, "r") as handle:
        schema_version = int(handle.attrs.get("schema_version", -1))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported schema version {schema_version}")
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        image_specs = {
            name: (tuple(value.shape), value.dtype)
            for name, value in handle.items()
            if name in {"base_images", "wrist_images"}
        }
        arrays = {
            name: np.asarray(value)
            for name, value in handle.items()
            if load_images or name not in {"base_images", "wrist_images"}
        }
    metadata = _validate_metadata(metadata)
    arrays = _validate_arrays(
        arrays,
        pregrasp_steps=int(metadata["pregrasp_steps"]),
        require_images=load_images,
    )
    _validate_initial_state_digest(metadata, arrays)
    if not load_images:
        time_steps = int(arrays["actions"].shape[0])
        for name in ("base_images", "wrist_images"):
            if name not in image_specs:
                raise ValueError(f"{path}: missing required image dataset {name}")
            shape, dtype = image_specs[name]
            if len(shape) != 4 or shape[0] != time_steps or shape[-1] != 3 or dtype != np.dtype(np.uint8):
                raise ValueError(f"{path}: invalid {name} dataset shape/dtype {shape}/{dtype}")
    return RecordedEpisode(path=path.resolve(), metadata=metadata, arrays=arrays)


def read_metadata(path: pathlib.Path | str) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported schema version")
        return _validate_metadata(json.loads(str(handle.attrs["metadata_json"])))


def discover_episodes(root: pathlib.Path | str) -> list[pathlib.Path]:
    root = pathlib.Path(root)
    paths = [root] if root.is_file() else sorted({*root.rglob("*.h5"), *root.rglob("*.hdf5")})
    return [path for path in paths if not path.name.endswith(".partial")]


def iter_observer_chunks(
    episode: RecordedEpisode,
    *,
    rotate_images_180: bool = True,
) -> Iterator[ActionChunk]:
    """Yield the ten pre-registered, non-overlapping 10-action chunks."""
    arrays = episode.arrays
    if "base_images" not in arrays or "wrist_images" not in arrays:
        raise ValueError("Episode was loaded without images")
    pregrasp_steps = int(episode.metadata["pregrasp_steps"])
    if pregrasp_steps != PREGRASP_STEPS or pregrasp_steps % ACTION_HORIZON:
        raise ValueError("The observer contract requires exactly 100 pre-grasp steps")
    for start in range(0, pregrasp_steps, ACTION_HORIZON):
        stop = start + ACTION_HORIZON
        base = arrays["base_images"][start]
        wrist = arrays["wrist_images"][start]
        if rotate_images_180:
            base = np.ascontiguousarray(base[::-1, ::-1])
            wrist = np.ascontiguousarray(wrist[::-1, ::-1])
        yield ActionChunk(
            start=start,
            stop=stop,
            base_image=base,
            wrist_image=wrist,
            state=np.asarray(arrays["states"][start], dtype=np.float32),
            actions=np.asarray(arrays["actions"][start:stop], dtype=np.float32),
        )
