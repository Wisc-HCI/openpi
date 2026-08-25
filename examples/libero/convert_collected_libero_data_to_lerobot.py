"""Convert SpaceMouse LIBERO demonstrations to OpenPI's LeRobot format.

The raw files produced by ``/workspace/LIBERO/scripts/collect_demonstration.py``
contain MuJoCo simulator states and actions, but no camera observations.  This
converter replays each recorded simulator state in the LIBERO Python 3.8
environment, renders the two camera views expected by OpenPI, constructs the
8-D LIBERO proprioceptive state, and writes a LeRobot v2 dataset.

The script orchestrates two Python environments:

* the current OpenPI environment writes the LeRobot dataset;
* ``--libero-python`` renders observations in the legacy LIBERO environment.

Example:

    uv run examples/libero/convert_collected_libero_data_to_lerobot.py \
      --data-dir /workspace/LIBERO/demonstration_data/legibility_v1 \
      --repo-id Wisc-HCI/libero_legibility_v1 \
      --overwrite

Run ``--dry-run`` first to audit source episode and frame counts without
rendering or writing a dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import h5py
import numpy as np

DEFAULT_LIBERO_PYTHON = Path("/workspace/miniconda3_data/envs/libero/bin/python")
DEFAULT_LIBERO_ROOT = Path("/workspace/LIBERO")
VALID_STYLES = frozenset({"direct", "legible", "free"})


def _episode_sort_key(name: str) -> tuple[int, str]:
    tail = name.rsplit("_", 1)[-1]
    return (int(tail) if tail.isdigit() else 10**9, name)


def _decode_json_attr(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _language_from_data_group(data: h5py.Group) -> str:
    info = _decode_json_attr(data.attrs["problem_info"])
    language = info["language_instruction"]
    if isinstance(language, list):
        language = "".join(str(part) for part in language)
    return str(language).strip().strip('"')


def discover_source_files(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.rglob("demo.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No demo.hdf5 files found under {data_dir}")
    return paths


def source_labels(data_dir: Path, source: Path) -> tuple[str, str]:
    relative_parts = source.relative_to(data_dir).parts
    object_name = relative_parts[0] if relative_parts else "unknown"
    style = next((part for part in relative_parts if part in VALID_STYLES), "unknown")
    return object_name, style


def audit_sources(data_dir: Path, source_files: list[Path]) -> dict[str, Any]:
    summary: dict[str, Any] = {"files": len(source_files), "episodes": 0, "frames": 0, "groups": {}}
    for source in source_files:
        object_name, style = source_labels(data_dir, source)
        key = f"{object_name}/{style}"
        group_summary = summary["groups"].setdefault(key, {"files": 0, "episodes": 0, "frames": 0})
        group_summary["files"] += 1
        with h5py.File(source, "r") as handle:
            if "data" not in handle:
                raise KeyError(f"{source}: missing data group")
            for episode_name in handle["data"]:
                episode = handle["data"][episode_name]
                if "states" not in episode or "actions" not in episode:
                    raise KeyError(f"{source}:{episode_name}: missing states or actions")
                states_shape = episode["states"].shape
                actions_shape = episode["actions"].shape
                if len(states_shape) != 2 or len(actions_shape) != 2 or actions_shape[1] != 7:
                    raise ValueError(
                        f"{source}:{episode_name}: expected states [T,D] and actions [T,7], "
                        f"got {states_shape} and {actions_shape}"
                    )
                if states_shape[0] != actions_shape[0]:
                    raise ValueError(
                        f"{source}:{episode_name}: state/action length mismatch {states_shape[0]} != {actions_shape[0]}"
                    )
                frames = int(actions_shape[0])
                summary["episodes"] += 1
                summary["frames"] += frames
                group_summary["episodes"] += 1
                group_summary["frames"] += frames
    return summary


def keep_action_indices(actions: np.ndarray, no_op_threshold: float) -> np.ndarray:
    """Keep arm motion and gripper transitions while removing stationary holds."""
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions with shape [T,7], got {actions.shape}")
    if len(actions) == 0:
        return np.empty((0,), dtype=np.int64)
    arm_moves = np.max(np.abs(actions[:, :6]), axis=1) > no_op_threshold
    gripper_changes = np.zeros(len(actions), dtype=bool)
    gripper_changes[0] = True
    gripper_changes[1:] = np.abs(np.diff(actions[:, 6])) > no_op_threshold
    keep = arm_moves | gripper_changes
    return np.flatnonzero(keep)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_source_worker(
    source: Path,
    output: Path,
    libero_root: Path,
    *,
    image_size: int,
    no_op_threshold: float,
    keep_no_ops: bool,
    max_episodes: int | None,
    max_frames_per_episode: int | None,
) -> None:
    """Render one raw HDF5 batch. This runs inside the LIBERO environment."""
    sys.path.insert(0, str(libero_root))

    from libero.libero.envs import TASK_MAPPING
    import libero.libero.utils.utils as libero_utils
    import robosuite.utils.transform_utils as transform_utils

    with h5py.File(source, "r") as raw:
        data = raw["data"]
        env_kwargs = _decode_json_attr(data.attrs["env_info"])
        problem_info = _decode_json_attr(data.attrs["problem_info"])
        problem_name = str(problem_info["problem_name"])
        bddl_file_name = Path(str(data.attrs["bddl_file_name"]))
        if not bddl_file_name.exists():
            raise FileNotFoundError(f"BDDL file recorded in {source} does not exist: {bddl_file_name}")

        libero_utils.update_env_kwargs(
            env_kwargs,
            bddl_file_name=str(bddl_file_name),
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            camera_depths=False,
            camera_names=["robot0_eye_in_hand", "agentview"],
            reward_shaping=True,
            control_freq=20,
            camera_heights=image_size,
            camera_widths=image_size,
            camera_segmentations=None,
        )
        env = TASK_MAPPING[problem_name](**env_kwargs)

        try:
            with h5py.File(output, "w") as rendered:
                rendered.attrs["source"] = str(source)
                rendered.attrs["language_instruction"] = _language_from_data_group(data)
                rendered.attrs["bddl_file_name"] = str(bddl_file_name)
                rendered.attrs["image_size"] = image_size

                episode_names = sorted(data.keys(), key=_episode_sort_key)
                if max_episodes is not None:
                    episode_names = episode_names[:max_episodes]

                for output_episode_index, episode_name in enumerate(episode_names):
                    episode = data[episode_name]
                    states = np.asarray(episode["states"], dtype=np.float64)
                    actions = np.asarray(episode["actions"], dtype=np.float32)
                    if len(states) != len(actions):
                        raise ValueError(f"{source}:{episode_name}: state/action length mismatch")

                    if keep_no_ops:
                        indices = np.arange(len(actions), dtype=np.int64)
                    else:
                        indices = keep_action_indices(actions, no_op_threshold)
                    if max_frames_per_episode is not None:
                        indices = indices[:max_frames_per_episode]
                    if len(indices) == 0:
                        raise ValueError(f"{source}:{episode_name}: no frames remain after filtering")

                    reset_succeeded = False
                    while not reset_succeeded:
                        try:
                            env.reset()
                            reset_succeeded = True
                        except Exception as error:  # LIBERO can raise placement randomization errors.
                            print(f"Retrying reset after {type(error).__name__}: {error}", flush=True)

                    model_xml = str(episode.attrs["model_file"])
                    model_xml = libero_utils.postprocess_model_xml(model_xml, {})
                    env.reset_from_xml_string(model_xml)
                    env.sim.reset()

                    images: list[np.ndarray] = []
                    wrist_images: list[np.ndarray] = []
                    proprio: list[np.ndarray] = []
                    selected_actions: list[np.ndarray] = []

                    for state_index in indices:
                        env.sim.set_state_from_flattened(states[state_index])
                        env.sim.forward()
                        env._post_process()  # noqa: SLF001 - required to regenerate observations from MuJoCo state.
                        env._update_observables(force=True)  # noqa: SLF001
                        observation = env._get_observations()  # noqa: SLF001

                        # OpenPI's LIBERO runtime applies this same 180-degree rotation
                        # before policy inference. Training data must use the same convention.
                        image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
                        wrist_image = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
                        state = np.concatenate(
                            [
                                np.asarray(observation["robot0_eef_pos"]),
                                transform_utils.quat2axisangle(observation["robot0_eef_quat"]),
                                np.asarray(observation["robot0_gripper_qpos"]),
                            ]
                        ).astype(np.float32)
                        if image.shape != (image_size, image_size, 3):
                            raise ValueError(f"Unexpected agent-view image shape {image.shape}")
                        if wrist_image.shape != (image_size, image_size, 3):
                            raise ValueError(f"Unexpected wrist image shape {wrist_image.shape}")
                        if state.shape != (8,):
                            raise ValueError(f"Unexpected LIBERO state shape {state.shape}")

                        images.append(image.astype(np.uint8, copy=False))
                        wrist_images.append(wrist_image.astype(np.uint8, copy=False))
                        proprio.append(state)
                        selected_actions.append(actions[state_index])

                    group = rendered.create_group(f"episode_{output_episode_index:06d}")
                    group.attrs["source_episode"] = episode_name
                    group.attrs["original_length"] = len(actions)
                    group.attrs["filtered_length"] = len(indices)
                    group.create_dataset("original_indices", data=indices)
                    group.create_dataset(
                        "image",
                        data=np.stack(images),
                        compression="gzip",
                        compression_opts=1,
                        shuffle=True,
                        chunks=(1, image_size, image_size, 3),
                    )
                    group.create_dataset(
                        "wrist_image",
                        data=np.stack(wrist_images),
                        compression="gzip",
                        compression_opts=1,
                        shuffle=True,
                        chunks=(1, image_size, image_size, 3),
                    )
                    group.create_dataset("state", data=np.stack(proprio))
                    group.create_dataset("actions", data=np.stack(selected_actions))
                    print(
                        f"Rendered {source.name}:{episode_name}: {len(actions)} -> {len(indices)} frames",
                        flush=True,
                    )
        finally:
            env.close()


def _safe_remove_dataset(output_path: Path) -> None:
    resolved = output_path.resolve()
    if resolved in {Path("/"), Path.home().resolve()} or len(resolved.parts) < 4:
        raise ValueError(f"Refusing to remove unsafe output path: {resolved}")
    shutil.rmtree(resolved)


def convert_dataset(args: argparse.Namespace) -> None:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    data_dir = args.data_dir.resolve()
    source_files = discover_source_files(data_dir)
    audit = audit_sources(data_dir, source_files)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.dry_run:
        return

    output_path = args.output_root.resolve() if args.output_root else HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
        _safe_remove_dataset(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_path,
        robot_type="panda",
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=0,
    )

    manifest: dict[str, Any] = {
        "format_version": 1,
        "repo_id": args.repo_id,
        "source_data_dir": str(data_dir),
        "fps": args.fps,
        "image_size": args.image_size,
        "rotate_images_180": True,
        "keep_no_ops": args.keep_no_ops,
        "no_op_threshold": args.no_op_threshold,
        "source_audit": audit,
        "episodes": [],
    }
    converted_episodes = 0

    with tempfile.TemporaryDirectory(prefix="libero_lerobot_render_") as temporary_directory:
        temporary_directory_path = Path(temporary_directory)
        for source_index, source in enumerate(source_files):
            if args.max_episodes is not None and converted_episodes >= args.max_episodes:
                break
            remaining = None if args.max_episodes is None else args.max_episodes - converted_episodes
            stage_path = temporary_directory_path / f"rendered_{source_index:06d}.hdf5"
            command = [
                str(args.libero_python),
                str(Path(__file__).resolve()),
                "--render-worker",
                "--source",
                str(source),
                "--stage-output",
                str(stage_path),
                "--libero-root",
                str(args.libero_root),
                "--image-size",
                str(args.image_size),
                "--no-op-threshold",
                str(args.no_op_threshold),
            ]
            if args.keep_no_ops:
                command.append("--keep-no-ops")
            if remaining is not None:
                command.extend(["--max-episodes", str(remaining)])
            if args.max_frames_per_episode is not None:
                command.extend(["--max-frames-per-episode", str(args.max_frames_per_episode)])

            worker_environment = os.environ.copy()
            worker_environment.setdefault("MUJOCO_GL", "egl")
            worker_environment.setdefault("PYOPENGL_PLATFORM", "egl")
            worker_environment.setdefault("NUMBA_CACHE_DIR", "/tmp/libero-numba")
            worker_environment.setdefault("MPLCONFIGDIR", "/tmp/libero-matplotlib")
            subprocess.run(
                command,
                check=True,
                cwd=args.libero_root,
                env=worker_environment,
            )

            object_name, style = source_labels(data_dir, source)
            with h5py.File(stage_path, "r") as rendered:
                task = str(rendered.attrs["language_instruction"])
                bddl_file_name = Path(str(rendered.attrs["bddl_file_name"]))
                for episode_name in sorted(rendered.keys()):
                    episode = rendered[episode_name]
                    episode_length = len(episode["actions"])
                    for frame_index in range(episode_length):
                        dataset.add_frame(
                            {
                                "image": np.asarray(episode["image"][frame_index]),
                                "wrist_image": np.asarray(episode["wrist_image"][frame_index]),
                                "state": np.asarray(episode["state"][frame_index], dtype=np.float32),
                                "actions": np.asarray(episode["actions"][frame_index], dtype=np.float32),
                                "task": task,
                            }
                        )
                    dataset.save_episode()
                    manifest["episodes"].append(
                        {
                            "lerobot_episode_index": converted_episodes,
                            "source_file": str(source),
                            "source_sha256": _sha256(source),
                            "source_episode": str(episode.attrs["source_episode"]),
                            "object": object_name,
                            "style": style,
                            "task": task,
                            "bddl_file": str(bddl_file_name),
                            "bddl_sha256": _sha256(bddl_file_name),
                            "original_length": int(episode.attrs["original_length"]),
                            "converted_length": episode_length,
                        }
                    )
                    converted_episodes += 1
                    print(
                        f"Saved LeRobot episode {converted_episodes}: {object_name}/{style}, "
                        f"{task!r}, {episode_length} frames",
                        flush=True,
                    )

    manifest["converted_episodes"] = converted_episodes
    manifest["converted_frames"] = sum(item["converted_length"] for item in manifest["episodes"])
    (output_path / "libero_conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "spacemouse"],
            private=args.private,
            push_videos=True,
            license="apache-2.0",
        )
    print(f"Saved {converted_episodes} episodes to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--repo-id", default="Wisc-HCI/libero_legibility_v1")
    parser.add_argument("--libero-python", type=Path, default=DEFAULT_LIBERO_PYTHON)
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--no-op-threshold", type=float, default=1e-6)
    parser.add_argument("--keep-no-ops", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-frames-per-episode", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", dest="private", action="store_true", default=True)
    parser.add_argument("--public", dest="private", action="store_false")
    parser.add_argument("--dry-run", action="store_true")

    # Internal worker arguments. Users normally do not invoke these directly.
    parser.add_argument("--render-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stage-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.render_worker:
        if args.source is None or args.stage_output is None:
            raise ValueError("--render-worker requires --source and --stage-output")
        render_source_worker(
            source=args.source,
            output=args.stage_output,
            libero_root=args.libero_root,
            image_size=args.image_size,
            no_op_threshold=args.no_op_threshold,
            keep_no_ops=args.keep_no_ops,
            max_episodes=args.max_episodes,
            max_frames_per_episode=args.max_frames_per_episode,
        )
        return
    if args.data_dir is None:
        raise ValueError("--data-dir is required")
    if not args.libero_python.exists():
        raise FileNotFoundError(args.libero_python)
    convert_dataset(args)


if __name__ == "__main__":
    main()
