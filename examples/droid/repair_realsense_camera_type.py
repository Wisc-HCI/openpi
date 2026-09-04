#!/usr/bin/env python3
"""Safely rename a stale DROID camera_type key in trajectory HDF5 files.

The script is a dry run unless --apply is passed. Before modifying an episode,
it verifies that the replacement camera has matching MP4, intrinsics, and
timestamp data. By default it also creates a full backup beside each HDF5 file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import h5py

CAMERA_TYPE_PATH = "observation/camera_type"
CAMERA_INTRINSICS_PATH = "observation/camera_intrinsics"
CAMERA_TIMESTAMPS_PATH = "observation/timestamp/cameras"


class PreflightError(RuntimeError):
    """Raised when an episode is not safe to modify automatically."""


@dataclass(frozen=True)
class Migration:
    episode_path: Path
    shape: tuple[int, ...]
    dtype: str
    content_hash: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="DROID data root containing success/.")
    parser.add_argument("--old-camera-id", required=True, help="Stale camera_type serial to replace.")
    parser.add_argument("--new-camera-id", required=True, help="Serial that actually produced the MP4.")
    parser.add_argument(
        "--expected-camera-type",
        type=int,
        default=1,
        help="Expected value stored in the old camera_type dataset (default: 1, external camera).",
    )
    parser.add_argument("--include-failures", action="store_true", help="Also scan data-dir/failure.")
    parser.add_argument("--apply", action="store_true", help="Apply the migration. Without this flag, only inspect.")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Modify without full trajectory.h5 backups. Not recommended.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".camera-type-backup",
        help="Suffix appended to trajectory.h5 for backups (default: .camera-type-backup).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print every inspected or modified trajectory.")
    return parser.parse_args()


def dataset_hash(dataset: h5py.Dataset) -> str:
    return hashlib.sha256(dataset[...].tobytes(order="C")).hexdigest()


def trajectory_paths(data_dir: Path, *, include_failures: bool) -> list[Path]:
    paths = sorted((data_dir / "success").glob("**/trajectory.h5"))
    if include_failures:
        paths.extend(sorted((data_dir / "failure").glob("**/trajectory.h5")))
    return paths


def check_camera_evidence(episode_path: Path, trajectory: h5py.File, old_id: str, new_id: str) -> None:
    recording_dir = episode_path.parent / "recordings" / "MP4"
    new_video = recording_dir / f"{new_id}.mp4"
    old_video = recording_dir / f"{old_id}.mp4"
    if not new_video.is_file():
        raise PreflightError(f"replacement video is missing: {new_video}")
    if old_video.exists():
        raise PreflightError(f"both old and replacement videos exist; camera mapping is ambiguous: {old_video}")

    if CAMERA_INTRINSICS_PATH not in trajectory:
        raise PreflightError(f"missing HDF5 group: {CAMERA_INTRINSICS_PATH}")
    intrinsics_keys = tuple(trajectory[CAMERA_INTRINSICS_PATH].keys())
    if not any(key == new_id or key.startswith(f"{new_id}_") for key in intrinsics_keys):
        raise PreflightError(f"no intrinsics found for replacement camera {new_id}")

    if CAMERA_TIMESTAMPS_PATH not in trajectory:
        raise PreflightError(f"missing HDF5 group: {CAMERA_TIMESTAMPS_PATH}")
    timestamp_keys = tuple(trajectory[CAMERA_TIMESTAMPS_PATH].keys())
    if not any(key == new_id or key.startswith(f"{new_id}_") for key in timestamp_keys):
        raise PreflightError(f"no timestamps found for replacement camera {new_id}")

    metadata_path = episode_path.parent / "metadata_openpi.json"
    if not metadata_path.is_file():
        raise PreflightError(f"missing episode metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"could not read episode metadata: {error}") from error
    camera_ids = metadata.get("camera_ids")
    if not isinstance(camera_ids, list):
        raise PreflightError("metadata_openpi.json has no camera_ids list")
    metadata_serials = {str(camera_id).split("_", maxsplit=1)[0] for camera_id in camera_ids}
    if new_id not in metadata_serials:
        raise PreflightError(f"replacement camera {new_id} is absent from metadata_openpi.json")
    if old_id in metadata_serials:
        raise PreflightError(f"old camera {old_id} is still declared in metadata_openpi.json")


def preflight_episode(
    episode_path: Path,
    old_id: str,
    new_id: str,
    expected_camera_type: int,
) -> Migration | None:
    with h5py.File(episode_path, "r") as trajectory:
        if CAMERA_TYPE_PATH not in trajectory:
            raise PreflightError(f"missing HDF5 group: {CAMERA_TYPE_PATH}")
        camera_types = trajectory[CAMERA_TYPE_PATH]
        has_old = old_id in camera_types
        has_new = new_id in camera_types

        if has_old and has_new:
            raise PreflightError("both old and replacement camera_type keys already exist")
        if not has_old and not has_new:
            raise PreflightError("neither old nor replacement camera_type key exists")

        check_camera_evidence(episode_path, trajectory, old_id, new_id)

        camera_type_dataset = camera_types[new_id if has_new else old_id]
        values = {int(value) for value in camera_type_dataset[...].reshape(-1)}
        if values != {expected_camera_type}:
            raise PreflightError(
                f"camera_type values are {sorted(values)}, expected only {expected_camera_type}"
            )

        if has_new:
            return None

        return Migration(
            episode_path=episode_path,
            shape=tuple(camera_type_dataset.shape),
            dtype=camera_type_dataset.dtype.str,
            content_hash=dataset_hash(camera_type_dataset),
        )


def backup_path(episode_path: Path, suffix: str) -> Path:
    return episode_path.with_name(episode_path.name + suffix)


def apply_migration(
    migration: Migration,
    old_id: str,
    new_id: str,
    expected_camera_type: int,
    *,
    make_backup: bool,
    backup_suffix: str,
) -> Path | None:
    episode_path = migration.episode_path
    destination = backup_path(episode_path, backup_suffix) if make_backup else None
    if destination is not None:
        shutil.copy2(episode_path, destination)

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{episode_path.name}.",
        suffix=".camera-type-repair",
        dir=episode_path.parent,
    )
    os.close(temporary_fd)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(episode_path, temporary_path)
        with h5py.File(temporary_path, "r+") as trajectory:
            camera_types = trajectory[CAMERA_TYPE_PATH]
            if old_id not in camera_types or new_id in camera_types:
                raise RuntimeError("camera_type keys changed after preflight")
            camera_types.move(old_id, new_id)
            trajectory.flush()

        with h5py.File(temporary_path, "r") as trajectory:
            camera_types = trajectory[CAMERA_TYPE_PATH]
            if old_id in camera_types or new_id not in camera_types:
                raise RuntimeError("camera_type rename did not persist")
            new_dataset = camera_types[new_id]
            values = {int(value) for value in new_dataset[...].reshape(-1)}
            if tuple(new_dataset.shape) != migration.shape:
                raise RuntimeError(f"dataset shape changed from {migration.shape} to {tuple(new_dataset.shape)}")
            if new_dataset.dtype.str != migration.dtype:
                raise RuntimeError(f"dataset dtype changed from {migration.dtype} to {new_dataset.dtype.str}")
            if dataset_hash(new_dataset) != migration.content_hash:
                raise RuntimeError("dataset contents changed during rename")
            if values != {expected_camera_type}:
                raise RuntimeError(f"replacement camera_type values are unexpectedly {sorted(values)}")
        os.replace(temporary_path, episode_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if args.old_camera_id == args.new_camera_id:
        print("ERROR: old and replacement camera IDs must differ", file=sys.stderr)
        return 2
    if not data_dir.is_dir():
        print(f"ERROR: data directory does not exist: {data_dir}", file=sys.stderr)
        return 2
    if args.apply and not args.no_backup and not args.backup_suffix:
        print("ERROR: backup suffix cannot be empty unless --no-backup is used", file=sys.stderr)
        return 2

    paths = trajectory_paths(data_dir, include_failures=args.include_failures)
    if not paths:
        print(f"ERROR: no trajectory.h5 files found below {data_dir}", file=sys.stderr)
        return 2

    migrations: list[Migration] = []
    already_migrated: list[Path] = []
    errors: list[tuple[Path, Exception]] = []
    for episode_path in paths:
        try:
            migration = preflight_episode(
                episode_path,
                args.old_camera_id,
                args.new_camera_id,
                args.expected_camera_type,
            )
        except Exception as error:
            errors.append((episode_path, error))
        else:
            if migration is None:
                already_migrated.append(episode_path)
            else:
                migrations.append(migration)
            if args.verbose:
                state = "already migrated" if migration is None else "ready"
                print(f"{state}: {episode_path}")

    print(
        f"Preflight: found={len(paths)}, ready={len(migrations)}, "
        f"already_migrated={len(already_migrated)}, errors={len(errors)}"
    )
    if errors:
        for episode_path, error in errors:
            print(f"ERROR: {episode_path}: {error}", file=sys.stderr)
        print("No files were modified because preflight failed.", file=sys.stderr)
        return 1

    if args.apply and not args.no_backup:
        existing_backups = [backup_path(migration.episode_path, args.backup_suffix) for migration in migrations]
        existing_backups = [path for path in existing_backups if path.exists()]
        if existing_backups:
            for path in existing_backups:
                print(f"ERROR: refusing to overwrite existing backup: {path}", file=sys.stderr)
            print("No files were modified because backup preflight failed.", file=sys.stderr)
            return 1

    if not args.apply:
        print("Dry run only; pass --apply to modify the ready trajectories.")
        return 0

    modified = 0
    backups: list[Path] = []
    for migration in migrations:
        try:
            destination = apply_migration(
                migration,
                args.old_camera_id,
                args.new_camera_id,
                args.expected_camera_type,
                make_backup=not args.no_backup,
                backup_suffix=args.backup_suffix,
            )
        except Exception as error:
            print(f"ERROR after modifying {modified} trajectories: {migration.episode_path}: {error}", file=sys.stderr)
            return 1
        modified += 1
        if destination is not None:
            backups.append(destination)
        if args.verbose:
            print(f"modified: {migration.episode_path}")

    print(f"Applied and verified: modified={modified}, already_migrated={len(already_migrated)}")
    if backups:
        print(f"Created {len(backups)} backups using suffix {args.backup_suffix!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
