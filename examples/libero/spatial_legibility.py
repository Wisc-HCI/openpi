"""Score and audit the frozen pi0.5 observer for LIBERO spatial legibility.

The simulator and observer intentionally live in separate environments.  This
entry point consumes only the recorded HDF5 contract and a geometry manifest
that was frozen before pi0.5 was queried.  It never modifies the checkpoint and
never silently repairs incomplete or incompatible artifacts.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import hashlib
import json
import os
import pathlib
import re
from typing import Any

import numpy as np

from openpi.instruction_likelihood import legibility_dataset
from openpi.instruction_likelihood import legibility_statistics
from openpi.instruction_likelihood import scorer as scorer_lib

DEFAULT_CHECKPOINT = pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero"
DEFAULT_FLOW_TIMESTEPS = (0.1, 0.3, 0.5, 0.7, 0.9)
DEFAULT_INITIAL_SEEDS = tuple(range(10))
EXPANDED_SEEDS = tuple(range(50))
RAW_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1
SANITY_SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 1
CHUNK_COUNT = legibility_dataset.PREGRASP_STEPS // legibility_dataset.ACTION_HORIZON

_RAW_ARRAY_KEYS = {
    "chunk_energy",
    "chunk_starts",
    "chunk_stops",
    "flow_timesteps",
    "noise",
    "normalized_actions",
    "residual",
    "seeds",
}


@dataclasses.dataclass(frozen=True)
class RawArtifact:
    path: pathlib.Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _write_json_atomic(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _save_npz_atomic(path: pathlib.Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial.npz")
    try:
        np.savez_compressed(
            partial,
            **arrays,
            metadata_json=np.asarray(legibility_dataset.canonical_json(metadata)),
        )
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _load_raw(path: pathlib.Path) -> RawArtifact:
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = sorted((_RAW_ARRAY_KEYS | {"metadata_json"}) - set(payload.files))
            unexpected = sorted(set(payload.files) - (_RAW_ARRAY_KEYS | {"metadata_json"}))
            if missing or unexpected:
                raise ValueError(f"{path}: raw NPZ keys missing={missing}, unexpected={unexpected}")
            arrays = {key: np.asarray(payload[key]) for key in _RAW_ARRAY_KEYS}
            metadata = json.loads(str(payload["metadata_json"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read raw artifact {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata_json must decode to an object")
    return RawArtifact(path=path.resolve(), arrays=arrays, metadata=metadata)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def _checkpoint_fingerprint(checkpoint: pathlib.Path) -> str:
    """Cheaply fingerprint an Orbax checkpoint without rereading multi-GB blobs.

    Orbax data blobs are content-addressed.  The fingerprint hashes their paths
    and sizes, while hashing every small metadata/config/normalization file in
    full.  This detects a changed checkpoint tree without adding a second model
    load worth of disk traffic to every command.
    """
    checkpoint = checkpoint.resolve()
    params = checkpoint / "params"
    if not params.is_dir():
        raise FileNotFoundError(f"Checkpoint params not found below {checkpoint}")
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Checkpoint contains no files: {checkpoint}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(checkpoint).as_posix()
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0".encode())
        if size <= 16 * 1024 * 1024:
            with path.open("rb") as handle:
                for piece in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(piece)
    return digest.hexdigest()


def _manifest_values(manifest: Mapping[str, Any], key: str) -> tuple[str, str]:
    value = manifest.get(key)
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Frozen manifest {key!r} must contain exactly two non-empty strings")
    return value[0], value[1]


def _level_names(manifest: Mapping[str, Any]) -> set[str]:
    levels = manifest.get("legibility_levels")
    if isinstance(levels, Mapping):
        return {str(key) for key in levels}
    if isinstance(levels, list):
        names: set[str] = set()
        for item in levels:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, Mapping) and "name" in item:
                names.add(str(item["name"]))
            elif isinstance(item, Mapping) and "level" in item:
                names.add(str(item["level"]))
            else:
                raise ValueError("Frozen manifest has an invalid legibility_levels entry")
        return names
    raise ValueError("Frozen manifest legibility_levels must be a list or object")


def _geometry_ids(manifest: Mapping[str, Any]) -> set[str]:
    geometries = manifest.get("geometries")
    if isinstance(geometries, Mapping):
        return {str(key) for key in geometries}
    if isinstance(geometries, list):
        result: set[str] = set()
        for item in geometries:
            if isinstance(item, str):
                result.add(item)
            elif isinstance(item, Mapping) and "geometry_id" in item:
                result.add(str(item["geometry_id"]))
            elif isinstance(item, Mapping) and "id" in item:
                result.add(str(item["id"]))
            else:
                raise ValueError("Frozen manifest has an invalid geometries entry")
        return result
    raise ValueError("Frozen manifest geometries must be a list or object")


def _geometry_separations(manifest: Mapping[str, Any]) -> dict[str, float]:
    geometries = manifest.get("geometries")
    if not isinstance(geometries, list):
        raise ValueError("Frozen spatial-legibility manifest geometries must be a list")
    result = {}
    for item in geometries:
        if not isinstance(item, Mapping) or "geometry_id" not in item or "separation_m" not in item:
            raise ValueError("Every frozen geometry must define geometry_id and separation_m")
        result[str(item["geometry_id"])] = float(item["separation_m"])
    return result


def _validate_episode_manifest(
    metadata: Mapping[str, Any], manifest: Mapping[str, Any], *, source: pathlib.Path
) -> int:
    digest = str(manifest["sha256"])
    if metadata.get("frozen_geometry_sha256") != digest:
        raise ValueError(
            f"{source}: episode frozen_geometry_sha256={metadata.get('frozen_geometry_sha256')!r} "
            f"does not match manifest {digest!r}"
        )
    identities = _manifest_values(manifest, "candidate_identities")
    prompts = _manifest_values(manifest, "candidate_prompts")
    objects = _manifest_values(manifest, "candidate_objects")
    target_identity = str(metadata["target_identity"])
    distractor_identity = str(metadata["distractor_identity"])
    if target_identity not in identities:
        raise ValueError(f"{source}: target identity {target_identity!r} is not a canonical candidate")
    target_index = identities.index(target_identity)
    other_index = 1 - target_index
    expected = {
        "distractor_identity": identities[other_index],
        "target_prompt": prompts[target_index],
        "distractor_prompt": prompts[other_index],
        "target_object": objects[target_index],
        "distractor_object": objects[other_index],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{source}: {key}={metadata.get(key)!r}, expected frozen value {value!r}")
    if distractor_identity != identities[other_index]:
        raise ValueError(f"{source}: distractor identity is inconsistent with the canonical candidates")
    geometry_ids = _geometry_ids(manifest)
    if str(metadata["geometry_id"]) not in geometry_ids:
        raise ValueError(f"{source}: unknown frozen geometry_id {metadata['geometry_id']!r}")
    expected_separation = _geometry_separations(manifest)[str(metadata["geometry_id"])]
    if not np.isclose(float(metadata["separation_m"]), expected_separation, atol=1e-9, rtol=0.0):
        raise ValueError(f"{source}: episode separation does not match its frozen geometry")
    level_names = _level_names(manifest)
    if str(metadata["legibility_level"]) not in level_names:
        raise ValueError(f"{source}: unknown frozen legibility_level {metadata['legibility_level']!r}")
    return target_index


def _validate_complete_inventory(
    discovered: Sequence[tuple[pathlib.Path, Mapping[str, Any], int]], manifest: Mapping[str, Any]
) -> None:
    geometry_ids = _geometry_ids(manifest)
    identities = set(_manifest_values(manifest, "candidate_identities"))
    levels = _level_names(manifest)
    seeds = {int(metadata["simulator_seed"]) for _, metadata, _ in discovered}
    expected = {
        (geometry, layout, seed, target, level)
        for geometry in geometry_ids
        for layout in ("A", "B")
        for seed in seeds
        for target in identities
        for level in levels
    }
    observed = {
        (
            str(metadata["geometry_id"]),
            str(metadata["layout_id"]),
            int(metadata["simulator_seed"]),
            str(metadata["target_identity"]),
            str(metadata["legibility_level"]),
        )
        for _, metadata, _ in discovered
    }
    if len(observed) != len(discovered):
        raise ValueError("Duplicate experimental conditions were found in the episode corpus")
    if expected != observed:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "Audit requires a complete G1/G2 x A/B x targets x A1/L0-L3 design for every seed; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def _discover_metadata(
    episode_root: pathlib.Path, manifest: Mapping[str, Any]
) -> list[tuple[pathlib.Path, dict[str, Any], int]]:
    paths = legibility_dataset.discover_episodes(episode_root)
    if not paths:
        raise ValueError(f"No complete HDF5 episodes found below {episode_root}")
    result: list[tuple[pathlib.Path, dict[str, Any], int]] = []
    seen_ids: dict[str, pathlib.Path] = {}
    for path in paths:
        metadata = legibility_dataset.read_metadata(path)
        target_index = _validate_episode_manifest(metadata, manifest, source=path)
        episode_id = str(metadata["episode_id"])
        if episode_id in seen_ids:
            raise ValueError(f"Duplicate episode_id {episode_id!r}: {seen_ids[episode_id]} and {path}")
        seen_ids[episode_id] = path
        result.append((path.resolve(), metadata, target_index))
    return result


def _result_name(episode_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id).strip("._") or "episode"
    suffix = hashlib.sha256(episode_id.encode()).hexdigest()[:12]
    return f"{slug[:96]}__{suffix}.npz"


def _initial_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if len(seeds) != 10:
        raise ValueError(f"The pre-registered initial run requires exactly 10 flow seeds, got {len(seeds)}")
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Initial flow seeds must be unique non-negative integers")
    return seeds


def _scoring_config(args: argparse.Namespace) -> scorer_lib.ResidualConfig:
    return scorer_lib.ResidualConfig(
        flow_timesteps=tuple(args.flow_timesteps),
        noise_samples=args.noise_samples,
        eval_batch_size=args.eval_batch_size,
    )


def _expected_chunk_bounds() -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, legibility_dataset.PREGRASP_STEPS, legibility_dataset.ACTION_HORIZON, dtype=np.int32)
    return starts, starts + legibility_dataset.ACTION_HORIZON


def _raw_metadata(
    *,
    episode: legibility_dataset.RecordedEpisode,
    target_candidate_index: int,
    manifest_path: pathlib.Path,
    manifest: Mapping[str, Any],
    checkpoint: pathlib.Path,
    checkpoint_fingerprint: str,
    source_sha256: str,
    config: scorer_lib.ResidualConfig,
    initial_seeds: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "artifact_kind": "pi05_libero_spatial_legibility_raw",
        "episode_id": episode.episode_id,
        "source_hdf5": str(episode.path),
        "source_hdf5_sha256": source_sha256,
        "frozen_geometry_path": str(manifest_path.resolve()),
        "frozen_geometry_sha256": str(manifest["sha256"]),
        "candidate_identities": list(_manifest_values(manifest, "candidate_identities")),
        "candidate_prompts": list(_manifest_values(manifest, "candidate_prompts")),
        "candidate_objects": list(_manifest_values(manifest, "candidate_objects")),
        "target_candidate_index": target_candidate_index,
        "target_identity": episode.metadata["target_identity"],
        "geometry_id": episode.metadata["geometry_id"],
        "layout_id": episode.metadata["layout_id"],
        "legibility_level": episode.metadata["legibility_level"],
        "simulator_seed": episode.metadata["simulator_seed"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_manifest_sha256": checkpoint_fingerprint,
        "checkpoint_config": "pi05_libero",
        "flow_timesteps": list(config.flow_timesteps),
        "noise_samples": config.noise_samples,
        "eval_batch_size": config.eval_batch_size,
        "initial_seeds": list(initial_seeds),
        "completed_seeds": [],
        "stability_expanded": False,
        "stability_trigger": None,
        "chunk_count": CHUNK_COUNT,
        "action_horizon": legibility_dataset.ACTION_HORIZON,
        "physical_action_dim": scorer_lib.LIBERO_ACTION_DIM,
        "model_action_dim": 32,
        "image_rotation_180": True,
        "common_noise": "one [seed,chunk,F,N,H,model_dim] tensor shared bit-identically by both candidates",
        "noise_seed_rule": "numpy SeedSequence([flow_seed, chunk_index]) with float32 standard_normal",
        "action_normalization": "pi05_libero checkpoint q01/q99 to [-1,1], then model padding to 32 dimensions",
        "residual_definition": "v_theta(x_t,t,o,l) - (epsilon - normalized_action), physical action dims 0:7",
        "energy_aggregation": "mean squared raw residual over flow timestep, noise sample, action step, dims 0:7",
    }


def _assert_resume_compatible(
    artifact: RawArtifact,
    expected: Mapping[str, Any],
    *,
    flow_timesteps: Sequence[float],
) -> None:
    stable_keys = (
        "schema_version",
        "artifact_kind",
        "episode_id",
        "source_hdf5",
        "source_hdf5_sha256",
        "frozen_geometry_sha256",
        "candidate_identities",
        "candidate_prompts",
        "candidate_objects",
        "target_candidate_index",
        "checkpoint",
        "checkpoint_manifest_sha256",
        "noise_samples",
        "eval_batch_size",
        "initial_seeds",
        "chunk_count",
        "action_horizon",
        "physical_action_dim",
        "model_action_dim",
        "image_rotation_180",
    )
    for key in stable_keys:
        if artifact.metadata.get(key) != expected.get(key):
            raise ValueError(
                f"{artifact.path}: cannot resume because metadata {key!r} changed "
                f"from {artifact.metadata.get(key)!r} to {expected.get(key)!r}"
            )
    if not np.array_equal(artifact.arrays["flow_timesteps"], np.asarray(flow_timesteps, dtype=np.float32)):
        raise ValueError(f"{artifact.path}: cannot resume with different flow timesteps")


def _score_seed_batch(
    scorer: scorer_lib.Pi05InstructionLikelihoodScorer,
    chunks: Sequence[Any],
    candidate_prompts: Sequence[str],
    seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(chunks) != CHUNK_COUNT:
        raise ValueError(f"Expected exactly {CHUNK_COUNT} non-overlapping chunks, got {len(chunks)}")
    scores = [
        scorer.score_chunk_seeds(chunk, candidate_prompts, chunk_index=index, seeds=seeds)
        for index, chunk in enumerate(chunks)
    ]
    for index, (chunk, result) in enumerate(zip(chunks, scores, strict=True)):
        expected_start = index * legibility_dataset.ACTION_HORIZON
        if chunk.start != expected_start or chunk.stop != expected_start + legibility_dataset.ACTION_HORIZON:
            raise ValueError(f"Observer chunk {index} is not the registered non-overlapping interval")
        if not np.array_equal(result.seeds, np.asarray(seeds, dtype=np.int64)):
            raise AssertionError("Scorer changed the requested flow-seed order")
    residual = np.stack([result.residual for result in scores], axis=1)
    noise = np.stack([result.noise for result in scores], axis=1)
    normalized_actions = np.stack([result.normalized_actions for result in scores])
    return residual, noise, normalized_actions


def _merge_seed_batch(
    existing: RawArtifact | None,
    *,
    residual: np.ndarray,
    noise: np.ndarray,
    normalized_actions: np.ndarray,
    seeds: Sequence[int],
) -> dict[str, np.ndarray]:
    starts, stops = _expected_chunk_bounds()
    new_seeds = np.asarray(seeds, dtype=np.int64)
    if existing is None:
        merged_residual = residual
        merged_noise = noise
        merged_seeds = new_seeds
    else:
        old_seeds = np.asarray(existing.arrays["seeds"], dtype=np.int64)
        overlap = set(old_seeds.tolist()) & set(new_seeds.tolist())
        if overlap:
            raise ValueError(f"{existing.path}: attempted to append existing seeds {sorted(overlap)}")
        if not np.array_equal(existing.arrays["normalized_actions"], normalized_actions):
            raise ValueError(f"{existing.path}: normalized demonstrated actions changed while resuming")
        merged_residual = np.concatenate([existing.arrays["residual"], residual], axis=0)
        merged_noise = np.concatenate([existing.arrays["noise"], noise], axis=0)
        merged_seeds = np.concatenate([old_seeds, new_seeds])
    order = np.argsort(merged_seeds)
    merged_seeds = merged_seeds[order]
    merged_residual = merged_residual[order]
    merged_noise = merged_noise[order]
    return {
        "residual": np.asarray(merged_residual, dtype=np.float32),
        "normalized_actions": np.asarray(normalized_actions, dtype=np.float32),
        "noise": np.asarray(merged_noise, dtype=np.float32),
        "seeds": merged_seeds,
        "flow_timesteps": np.asarray(
            existing.arrays["flow_timesteps"] if existing is not None else (), dtype=np.float32
        ),
        "chunk_starts": starts,
        "chunk_stops": stops,
        "chunk_energy": legibility_statistics.residual_energy_by_seed(merged_residual),
    }


def _new_artifact_arrays(
    *,
    residual: np.ndarray,
    noise: np.ndarray,
    normalized_actions: np.ndarray,
    seeds: Sequence[int],
    flow_timesteps: Sequence[float],
) -> dict[str, np.ndarray]:
    starts, stops = _expected_chunk_bounds()
    return {
        "residual": np.asarray(residual, dtype=np.float32),
        "normalized_actions": np.asarray(normalized_actions, dtype=np.float32),
        "noise": np.asarray(noise, dtype=np.float32),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "flow_timesteps": np.asarray(flow_timesteps, dtype=np.float32),
        "chunk_starts": starts,
        "chunk_stops": stops,
        "chunk_energy": legibility_statistics.residual_energy_by_seed(residual),
    }


def _append_missing_seeds(
    *,
    raw_path: pathlib.Path,
    artifact: RawArtifact | None,
    metadata: dict[str, Any],
    chunks: Sequence[Any],
    candidate_prompts: Sequence[str],
    requested_seeds: Sequence[int],
    scorer_factory: Callable[[], scorer_lib.Pi05InstructionLikelihoodScorer],
    flow_timesteps: Sequence[float],
) -> RawArtifact:
    completed = set() if artifact is None else set(np.asarray(artifact.arrays["seeds"], dtype=np.int64).tolist())
    missing = tuple(seed for seed in requested_seeds if seed not in completed)
    if not missing:
        assert artifact is not None
        return artifact
    scorer = scorer_factory()
    residual, noise, normalized_actions = _score_seed_batch(scorer, chunks, candidate_prompts, missing)
    if artifact is None:
        arrays = _new_artifact_arrays(
            residual=residual,
            noise=noise,
            normalized_actions=normalized_actions,
            seeds=missing,
            flow_timesteps=flow_timesteps,
        )
    else:
        arrays = _merge_seed_batch(
            artifact,
            residual=residual,
            noise=noise,
            normalized_actions=normalized_actions,
            seeds=missing,
        )
        arrays["flow_timesteps"] = np.asarray(flow_timesteps, dtype=np.float32)
    metadata["completed_seeds"] = arrays["seeds"].tolist()
    _save_npz_atomic(raw_path, arrays, metadata)
    return _load_raw(raw_path)


def _episode_summary(artifact: RawArtifact) -> dict[str, Any]:
    summary = legibility_statistics.summarize_episode_residuals(
        artifact.arrays["residual"], target_candidate_index=int(artifact.metadata["target_candidate_index"])
    )
    physical_actions = np.asarray(artifact.arrays["normalized_actions"][:, :, : scorer_lib.LIBERO_ACTION_DIM])
    return {
        "episode_id": artifact.metadata["episode_id"],
        "raw_path": str(artifact.path),
        "target_identity": artifact.metadata["target_identity"],
        "geometry_id": artifact.metadata["geometry_id"],
        "layout_id": artifact.metadata["layout_id"],
        "legibility_level": artifact.metadata["legibility_level"],
        "simulator_seed": artifact.metadata["simulator_seed"],
        "stability_expanded": bool(artifact.metadata["stability_expanded"]),
        "normalized_action_out_of_range_fraction": float(np.mean(np.abs(physical_actions) > 1.0)),
        "normalized_action_max_absolute_value": float(np.max(np.abs(physical_actions))),
        **summary,
    }


def score(args: argparse.Namespace) -> None:
    manifest_path = args.frozen_geometry.resolve()
    manifest = legibility_dataset.load_frozen_manifest(manifest_path)
    discovered = _discover_metadata(args.episode_root, manifest)
    initial_seeds = _initial_seeds(args.seeds)
    if args.expand_unstable_l3 and not set(initial_seeds).issubset(EXPANDED_SEEDS):
        raise ValueError("Stability expansion requires the initial seeds to be drawn from 0..49")
    if not np.isclose(args.stability_threshold, 0.8, atol=0.0, rtol=0.0):
        raise ValueError("The pre-registered stability threshold is fixed at 0.8")
    config = _scoring_config(args)
    checkpoint = args.checkpoint.resolve()
    checkpoint_fingerprint = _checkpoint_fingerprint(checkpoint)
    candidate_prompts = _manifest_values(manifest, "candidate_prompts")
    raw_dir = args.output_dir / "raw"
    scorer: scorer_lib.Pi05InstructionLikelihoodScorer | None = None

    def loaded_scorer() -> scorer_lib.Pi05InstructionLikelihoodScorer:
        nonlocal scorer
        if scorer is None:
            scorer = scorer_lib.Pi05InstructionLikelihoodScorer(checkpoint, config=config)
            if scorer.action_horizon != legibility_dataset.ACTION_HORIZON:
                raise ValueError(
                    f"Expected pi05_libero action horizon {legibility_dataset.ACTION_HORIZON}, "
                    f"got {scorer.action_horizon}"
                )
            if scorer.model_action_dim != 32:
                raise ValueError(f"Expected pi05_libero model action dimension 32, got {scorer.model_action_dim}")
        return scorer

    summaries: list[dict[str, Any]] = []
    for episode_path, discovered_metadata, target_index in discovered:
        episode = legibility_dataset.read_episode(episode_path)
        if episode.metadata != discovered_metadata:
            raise ValueError(f"{episode_path}: metadata changed between discovery and scoring")
        chunks = list(legibility_dataset.iter_observer_chunks(episode))
        raw_path = raw_dir / _result_name(episode.episode_id)
        source_sha256 = _sha256_file(episode_path)
        metadata = _raw_metadata(
            episode=episode,
            target_candidate_index=target_index,
            manifest_path=manifest_path,
            manifest=manifest,
            checkpoint=checkpoint,
            checkpoint_fingerprint=checkpoint_fingerprint,
            source_sha256=source_sha256,
            config=config,
            initial_seeds=initial_seeds,
        )
        artifact: RawArtifact | None = None
        if raw_path.exists() and not args.overwrite:
            artifact = _load_raw(raw_path)
            _assert_resume_compatible(artifact, metadata, flow_timesteps=config.flow_timesteps)
            _validate_raw_shapes(artifact, allow_unexpanded_l3=True)
            metadata = dict(artifact.metadata)
        artifact = _append_missing_seeds(
            raw_path=raw_path,
            artifact=artifact,
            metadata=metadata,
            chunks=chunks,
            candidate_prompts=candidate_prompts,
            requested_seeds=initial_seeds,
            scorer_factory=loaded_scorer,
            flow_timesteps=config.flow_timesteps,
        )

        stored_seeds = np.asarray(artifact.arrays["seeds"], dtype=np.int64)
        initial_indices = [int(np.flatnonzero(stored_seeds == seed)[0]) for seed in initial_seeds]
        initial_summary = legibility_statistics.summarize_episode_residuals(
            artifact.arrays["residual"][initial_indices], target_candidate_index=target_index
        )
        agreement = float(initial_summary["C30"]["mean_sign_agreement"])
        if (
            args.expand_unstable_l3
            and episode.metadata["legibility_level"] == "L3"
            and agreement < args.stability_threshold
        ):
            metadata = dict(artifact.metadata)
            metadata["stability_expanded"] = True
            metadata["stability_trigger"] = {
                "endpoint": "C30",
                "initial_sign_agreement": agreement,
                "threshold": args.stability_threshold,
                "expanded_seed_count": len(EXPANDED_SEEDS),
            }
            artifact = _append_missing_seeds(
                raw_path=raw_path,
                artifact=artifact,
                metadata=metadata,
                chunks=chunks,
                candidate_prompts=candidate_prompts,
                requested_seeds=EXPANDED_SEEDS,
                scorer_factory=loaded_scorer,
                flow_timesteps=config.flow_timesteps,
            )
            # The flag/trigger must also be persisted when all 50 seeds were already present.
            if artifact.metadata.get("stability_trigger") != metadata["stability_trigger"]:
                _save_npz_atomic(raw_path, artifact.arrays, metadata)
                artifact = _load_raw(raw_path)
        summaries.append(_episode_summary(artifact))
        print(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "seed_count": int(artifact.arrays["residual"].shape[0]),
                    "raw_path": str(raw_path.resolve()),
                },
                sort_keys=True,
            )
        )

    summary_payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "artifact_kind": "pi05_libero_spatial_legibility_score_summary",
        "episode_root": str(args.episode_root.resolve()),
        "episode_count": len(summaries),
        "frozen_geometry_path": str(manifest_path),
        "frozen_geometry_sha256": manifest["sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": checkpoint_fingerprint,
        "candidate_identities": list(_manifest_values(manifest, "candidate_identities")),
        "candidate_prompts": list(candidate_prompts),
        "initial_seeds": list(initial_seeds),
        "flow_timesteps": list(config.flow_timesteps),
        "noise_samples": config.noise_samples,
        "stability_expansion_enabled": bool(args.expand_unstable_l3),
        "stability_rule": "only L3 C30 mean-sign agreement < threshold expands to seeds 0..49",
        "stability_threshold": args.stability_threshold,
        "episodes": summaries,
    }
    _write_json_atomic(args.output_dir / "score_summary.json", summary_payload)


def _semantic_support(residual: np.ndarray, target_index: int, identities: Sequence[str]) -> dict[str, Any]:
    energy = np.mean(np.square(residual, dtype=np.float64), axis=(2, 3, 4, 5))
    canonical_margin = energy[:, 1] - energy[:, 0]
    truth_margin = canonical_margin if target_index == 0 else -canonical_margin
    canonical_mean = float(np.mean(canonical_margin))
    if canonical_mean > 0.0:
        preferred = identities[0]
    elif canonical_mean < 0.0:
        preferred = identities[1]
    else:
        preferred = "tie"
    return {
        "candidate_mean_energy": np.mean(energy, axis=0).tolist(),
        "mean_canonical_margin": canonical_mean,
        "preferred_identity": preferred,
        "truth_margin": dataclasses.asdict(legibility_statistics.summarize_seed_values(truth_margin)),
    }


def _sanity_episode(
    *,
    scorer: scorer_lib.Pi05InstructionLikelihoodScorer,
    episode: legibility_dataset.RecordedEpisode,
    target_index: int,
    prompts: Sequence[str],
    identities: Sequence[str],
    seeds: Sequence[int],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    chunks = list(legibility_dataset.iter_observer_chunks(episode))
    endpoint_rows: list[dict[str, Any]] = []
    all_same_prompt = True
    all_candidate_order = True
    all_common_noise = True
    for label, chunk_index in (("initial", 0), ("final", CHUNK_COUNT - 1)):
        chunk = chunks[chunk_index]
        baseline = scorer.score_chunk_seeds(chunk, prompts, chunk_index=chunk_index, seeds=seeds)
        reversed_scores = scorer.score_chunk_seeds(
            chunk, tuple(reversed(prompts)), chunk_index=chunk_index, seeds=seeds
        )
        same_results = [
            scorer.score_chunk_seeds(chunk, (prompt, prompt), chunk_index=chunk_index, seeds=seeds)
            for prompt in prompts
        ]
        same_differences = [
            float(np.max(np.abs(result.residual[:, 0] - result.residual[:, 1]))) for result in same_results
        ]
        same_passed = all(
            np.allclose(result.residual[:, 0], result.residual[:, 1], atol=atol, rtol=rtol) for result in same_results
        )
        order_difference = float(np.max(np.abs(baseline.residual - reversed_scores.residual[:, ::-1])))
        order_passed = bool(np.allclose(baseline.residual, reversed_scores.residual[:, ::-1], atol=atol, rtol=rtol))
        expected_noise = np.stack(
            [
                np.random.default_rng(np.random.SeedSequence([int(seed), chunk_index])).standard_normal(
                    baseline.noise.shape[1:], dtype=np.float32
                )
                for seed in seeds
            ]
        )
        common_passed = bool(
            np.array_equal(baseline.noise, expected_noise)
            and np.array_equal(baseline.noise, reversed_scores.noise)
            and all(np.array_equal(baseline.noise, result.noise) for result in same_results)
            and np.array_equal(baseline.normalized_actions, reversed_scores.normalized_actions)
            and all(np.array_equal(baseline.normalized_actions, result.normalized_actions) for result in same_results)
        )
        all_same_prompt &= same_passed
        all_candidate_order &= order_passed
        all_common_noise &= common_passed
        endpoint_rows.append(
            {
                "endpoint": label,
                "chunk_index": chunk_index,
                "same_prompt_passed": same_passed,
                "same_prompt_max_absolute_differences": same_differences,
                "candidate_order_passed": order_passed,
                "candidate_order_max_absolute_difference": order_difference,
                "common_noise_passed": common_passed,
                "semantic_support": _semantic_support(baseline.residual, target_index, identities),
            }
        )
    return {
        "episode_id": episode.episode_id,
        "source_hdf5": str(episode.path),
        "source_hdf5_sha256": _sha256_file(episode.path),
        "target_identity": episode.metadata["target_identity"],
        "target_candidate_index": target_index,
        "same_prompt_passed": all_same_prompt,
        "candidate_order_passed": all_candidate_order,
        "common_noise_passed": all_common_noise,
        "passed": all_same_prompt and all_candidate_order and all_common_noise,
        "endpoints": endpoint_rows,
    }


def _select_sanity_episodes(
    discovered: Sequence[tuple[pathlib.Path, dict[str, Any], int]], max_episodes: int
) -> list[tuple[pathlib.Path, dict[str, Any], int]]:
    if max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    selected: list[tuple[pathlib.Path, dict[str, Any], int]] = []
    selected_paths: set[pathlib.Path] = set()
    for target_index in (0, 1):
        # L0 is the cleanest initial-prior diagnostic: A1 and L1--L3 encode
        # target-specific curvature from their first actions.
        match = next(
            (row for row in discovered if row[2] == target_index and str(row[1].get("legibility_level")) == "L0"),
            None,
        )
        if match is None:
            match = next((row for row in discovered if row[2] == target_index), None)
        if match is not None and match[0] not in selected_paths and len(selected) < max_episodes:
            selected.append(match)
            selected_paths.add(match[0])
    for row in discovered:
        if len(selected) >= max_episodes:
            break
        if row[0] not in selected_paths:
            selected.append(row)
            selected_paths.add(row[0])
    return selected


def sanity(args: argparse.Namespace) -> None:
    manifest_path = args.frozen_geometry.resolve()
    manifest = legibility_dataset.load_frozen_manifest(manifest_path)
    discovered = _discover_metadata(args.episode_root, manifest)
    selected = _select_sanity_episodes(discovered, args.max_episodes)
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Sanity seeds must be a non-empty set of unique non-negative integers")
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("Sanity tolerances must be non-negative")
    config = _scoring_config(args)
    checkpoint = args.checkpoint.resolve()
    checkpoint_fingerprint = _checkpoint_fingerprint(checkpoint)
    scorer = scorer_lib.Pi05InstructionLikelihoodScorer(checkpoint, config=config)
    if scorer.action_horizon != legibility_dataset.ACTION_HORIZON or scorer.model_action_dim != 32:
        raise ValueError("The selected checkpoint does not have the frozen pi05_libero action contract")
    prompts = _manifest_values(manifest, "candidate_prompts")
    identities = _manifest_values(manifest, "candidate_identities")
    rows = []
    cache_validation: scorer_lib.CacheValidation | None = None
    for selected_index, (path, metadata, target_index) in enumerate(selected):
        episode = legibility_dataset.read_episode(path)
        if episode.metadata != metadata:
            raise ValueError(f"{path}: metadata changed between sanity discovery and scoring")
        if selected_index == 0:
            first_chunk = next(iter(legibility_dataset.iter_observer_chunks(episode)))
            cache_validation = scorer.validate_prefix_cache(first_chunk, prompts[target_index])
        rows.append(
            _sanity_episode(
                scorer=scorer,
                episode=episode,
                target_index=target_index,
                prompts=prompts,
                identities=identities,
                seeds=seeds,
                atol=args.atol,
                rtol=args.rtol,
            )
        )
    initial_truth = np.concatenate(
        [
            np.asarray(row["endpoints"][0]["semantic_support"]["truth_margin"]["true_support_fraction"])[None]
            for row in rows
        ]
    )
    final_truth = np.concatenate(
        [
            np.asarray(row["endpoints"][1]["semantic_support"]["truth_margin"]["true_support_fraction"])[None]
            for row in rows
        ]
    )
    final_semantic_grounding_passed = all(
        float(row["endpoints"][1]["semantic_support"]["truth_margin"]["mean"]) > 0.0 for row in rows
    )
    assert cache_validation is not None
    passed = all(bool(row["passed"]) for row in rows) and final_semantic_grounding_passed and cache_validation.allclose
    payload = {
        "schema_version": SANITY_SCHEMA_VERSION,
        "artifact_kind": "pi05_libero_spatial_legibility_sanity",
        "passed": passed,
        "sanity_gate": "methodological invariances plus positive final pre-grasp truth margin for both target identities",
        "episode_root": str(args.episode_root.resolve()),
        "frozen_geometry_path": str(manifest_path),
        "frozen_geometry_sha256": manifest["sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": checkpoint_fingerprint,
        "candidate_identities": list(identities),
        "candidate_prompts": list(prompts),
        "flow_timesteps": list(config.flow_timesteps),
        "noise_samples": config.noise_samples,
        "seeds": list(seeds),
        "same_prompt_passed": all(bool(row["same_prompt_passed"]) for row in rows),
        "candidate_order_passed": all(bool(row["candidate_order_passed"]) for row in rows),
        "common_noise_passed": all(bool(row["common_noise_passed"]) for row in rows),
        "prefix_cache_validation_passed": cache_validation.allclose,
        "prefix_cache_validation": dataclasses.asdict(cache_validation),
        "final_semantic_grounding_passed": final_semantic_grounding_passed,
        "initial_semantic_true_support_fraction_mean": float(np.mean(initial_truth)),
        "final_semantic_true_support_fraction_mean": float(np.mean(final_truth)),
        "episodes": rows,
    }
    output = args.output or args.output_dir / "sanity_summary.json"
    _write_json_atomic(output, payload)
    print(json.dumps({"passed": passed, "output": str(output.resolve())}, sort_keys=True))
    if not passed:
        raise SystemExit("Spatial-legibility sanity checks failed")


def _validate_noise_reproducibility(
    noise: np.ndarray, seeds: np.ndarray, *, flow_count: int, noise_samples: int, action_horizon: int, model_dim: int
) -> None:
    expected_shape = (len(seeds), CHUNK_COUNT, flow_count, noise_samples, action_horizon, model_dim)
    if noise.shape != expected_shape:
        raise ValueError(f"noise must have shape {expected_shape}, got {noise.shape}")
    for seed_axis, seed in enumerate(seeds):
        for chunk_index in range(CHUNK_COUNT):
            expected = np.random.default_rng(np.random.SeedSequence([int(seed), chunk_index])).standard_normal(
                (flow_count, noise_samples, action_horizon, model_dim), dtype=np.float32
            )
            if not np.array_equal(noise[seed_axis, chunk_index], expected):
                raise ValueError(
                    f"Stored noise is not reproducible for seed={int(seed)}, chunk={chunk_index}; "
                    "paired/common-noise provenance is invalid"
                )


def _validate_raw_shapes(artifact: RawArtifact, *, allow_unexpanded_l3: bool = False) -> None:
    arrays = artifact.arrays
    metadata = artifact.metadata
    if metadata.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError(f"{artifact.path}: unsupported raw schema version")
    if metadata.get("artifact_kind") != "pi05_libero_spatial_legibility_raw":
        raise ValueError(f"{artifact.path}: invalid artifact kind")
    seeds = np.asarray(arrays["seeds"])
    flow = np.asarray(arrays["flow_timesteps"])
    if seeds.ndim != 1 or len(seeds) == 0 or seeds.dtype.kind not in "iu":
        raise ValueError(f"{artifact.path}: seeds must be a non-empty integer vector")
    if len(np.unique(seeds)) != len(seeds) or np.any(seeds < 0) or np.any(np.diff(seeds) <= 0):
        raise ValueError(f"{artifact.path}: seeds must be sorted, unique, and non-negative")
    initial_seeds = tuple(int(seed) for seed in metadata.get("initial_seeds", []))
    _initial_seeds(initial_seeds)
    expanded = bool(metadata.get("stability_expanded"))
    expected_seeds = EXPANDED_SEEDS if expanded else tuple(sorted(initial_seeds))
    if tuple(seeds.tolist()) != expected_seeds:
        raise ValueError(f"{artifact.path}: completed seed set is inconsistent with stability metadata")
    trigger = metadata.get("stability_trigger")
    if expanded:
        if metadata.get("legibility_level") != "L3" or not isinstance(trigger, Mapping):
            raise ValueError(f"{artifact.path}: only L3 may carry a stability expansion trigger")
        threshold = float(trigger.get("threshold", float("nan")))
        initial_agreement = float(trigger.get("initial_sign_agreement", float("nan")))
        if (
            trigger.get("endpoint") != "C30"
            or trigger.get("expanded_seed_count") != len(EXPANDED_SEEDS)
            or not np.isfinite(threshold)
            or not np.isfinite(initial_agreement)
            or threshold != 0.8
            or not 0.0 <= initial_agreement <= 1.0
            or initial_agreement >= threshold
        ):
            raise ValueError(f"{artifact.path}: invalid L3 stability expansion trigger")
    elif trigger is not None:
        raise ValueError(f"{artifact.path}: non-expanded artifacts cannot carry a stability trigger")
    if metadata.get("completed_seeds") != seeds.tolist():
        raise ValueError(f"{artifact.path}: completed_seeds metadata does not match the seed axis")
    if flow.ndim != 1 or len(flow) == 0 or not np.all(np.isfinite(flow)) or np.any((flow <= 0) | (flow >= 1)):
        raise ValueError(f"{artifact.path}: invalid flow timesteps")
    if not np.array_equal(flow, np.asarray(metadata.get("flow_timesteps"), dtype=np.float32)):
        raise ValueError(f"{artifact.path}: flow timestep metadata does not match the stored array")
    flow_count = len(flow)
    noise_samples = int(metadata.get("noise_samples", -1))
    action_horizon = int(metadata.get("action_horizon", -1))
    physical_dim = int(metadata.get("physical_action_dim", -1))
    model_dim = int(metadata.get("model_action_dim", -1))
    if int(metadata.get("chunk_count", -1)) != CHUNK_COUNT:
        raise ValueError(f"{artifact.path}: chunk_count metadata must be exactly {CHUNK_COUNT}")
    if noise_samples <= 0:
        raise ValueError(f"{artifact.path}: noise_samples must be positive")
    if action_horizon != legibility_dataset.ACTION_HORIZON:
        raise ValueError(f"{artifact.path}: action_horizon must be {legibility_dataset.ACTION_HORIZON}")
    expected_residual = (len(seeds), CHUNK_COUNT, 2, flow_count, noise_samples, action_horizon, physical_dim)
    if arrays["residual"].shape != expected_residual or physical_dim != scorer_lib.LIBERO_ACTION_DIM:
        raise ValueError(
            f"{artifact.path}: residual must have shape {expected_residual}, got {arrays['residual'].shape}"
        )
    expected_actions = (CHUNK_COUNT, action_horizon, model_dim)
    if arrays["normalized_actions"].shape != expected_actions or model_dim != 32:
        raise ValueError(
            f"{artifact.path}: normalized_actions must have shape {expected_actions}, "
            f"got {arrays['normalized_actions'].shape}"
        )
    expected_dtypes = {
        "residual": np.dtype(np.float32),
        "normalized_actions": np.dtype(np.float32),
        "noise": np.dtype(np.float32),
        "seeds": np.dtype(np.int64),
        "flow_timesteps": np.dtype(np.float32),
        "chunk_starts": np.dtype(np.int32),
        "chunk_stops": np.dtype(np.int32),
        "chunk_energy": np.dtype(np.float64),
    }
    for key, expected_dtype in expected_dtypes.items():
        if arrays[key].dtype != expected_dtype:
            raise ValueError(f"{artifact.path}: {key} must use {expected_dtype}, got {arrays[key].dtype}")
    starts, stops = _expected_chunk_bounds()
    if not np.array_equal(arrays["chunk_starts"], starts) or not np.array_equal(arrays["chunk_stops"], stops):
        raise ValueError(f"{artifact.path}: chunks are not the ten registered non-overlapping intervals")
    for key in ("residual", "normalized_actions", "noise", "chunk_energy"):
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"{artifact.path}: {key} contains non-finite values")
    _validate_noise_reproducibility(
        arrays["noise"],
        seeds,
        flow_count=flow_count,
        noise_samples=noise_samples,
        action_horizon=action_horizon,
        model_dim=model_dim,
    )
    reproduced_energy = legibility_statistics.residual_energy_by_seed(arrays["residual"])
    if arrays["chunk_energy"].shape != reproduced_energy.shape or not np.array_equal(
        arrays["chunk_energy"], reproduced_energy
    ):
        max_difference = (
            float(np.max(np.abs(arrays["chunk_energy"] - reproduced_energy)))
            if arrays["chunk_energy"].shape == reproduced_energy.shape
            else float("inf")
        )
        raise ValueError(f"{artifact.path}: raw energy is not exactly reproducible (max difference {max_difference})")
    if expanded:
        initial_indices = [int(np.flatnonzero(seeds == seed)[0]) for seed in initial_seeds]
        initial_summary = legibility_statistics.summarize_episode_residuals(
            arrays["residual"][initial_indices],
            target_candidate_index=int(metadata.get("target_candidate_index", -1)),
        )
        reproduced_agreement = float(initial_summary["C30"]["mean_sign_agreement"])
        if reproduced_agreement != float(trigger["initial_sign_agreement"]):
            raise ValueError(f"{artifact.path}: L3 expansion trigger does not reproduce from the initial ten seeds")
    elif metadata.get("legibility_level") == "L3" and not allow_unexpanded_l3:
        initial_summary = legibility_statistics.summarize_episode_residuals(
            arrays["residual"],
            target_candidate_index=int(metadata.get("target_candidate_index", -1)),
        )
        if float(initial_summary["C30"]["mean_sign_agreement"]) < 0.8:
            raise ValueError(f"{artifact.path}: unstable L3 C30 was not expanded to 50 flow seeds")


def _load_required_json(path: pathlib.Path, *, kind: str, artifact_kind: str, schema_version: int) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing required {kind} artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse {kind} artifact {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != schema_version
        or payload.get("artifact_kind") != artifact_kind
    ):
        raise ValueError(f"{path}: invalid {kind} schema")
    return payload


def _validate_freeze_fields(
    payload: Mapping[str, Any],
    *,
    source: pathlib.Path,
    manifest: Mapping[str, Any],
    checkpoint: pathlib.Path,
    checkpoint_sha: str,
) -> None:
    expected = {
        "frozen_geometry_sha256": manifest["sha256"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_manifest_sha256": checkpoint_sha,
        "candidate_identities": list(_manifest_values(manifest, "candidate_identities")),
        "candidate_prompts": list(_manifest_values(manifest, "candidate_prompts")),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{source}: frozen field {key!r} is {payload.get(key)!r}, expected {value!r}")


def audit(args: argparse.Namespace) -> None:
    manifest_path = args.frozen_geometry.resolve()
    manifest = legibility_dataset.load_frozen_manifest(manifest_path)
    discovered = _discover_metadata(args.episode_root, manifest)
    if not args.allow_subset:
        _validate_complete_inventory(discovered, manifest)
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha = _checkpoint_fingerprint(checkpoint)
    score_summary_path = args.output_dir / "score_summary.json"
    score_summary = _load_required_json(
        score_summary_path,
        kind="score summary",
        artifact_kind="pi05_libero_spatial_legibility_score_summary",
        schema_version=SUMMARY_SCHEMA_VERSION,
    )
    _validate_freeze_fields(
        score_summary,
        source=score_summary_path,
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_sha=checkpoint_sha,
    )
    sanity_path = args.sanity or args.output_dir / "sanity_summary.json"
    sanity_payload = _load_required_json(
        sanity_path,
        kind="sanity",
        artifact_kind="pi05_libero_spatial_legibility_sanity",
        schema_version=SANITY_SCHEMA_VERSION,
    )
    _validate_freeze_fields(
        sanity_payload, source=sanity_path, manifest=manifest, checkpoint=checkpoint, checkpoint_sha=checkpoint_sha
    )
    required_sanity_flags = (
        "passed",
        "same_prompt_passed",
        "candidate_order_passed",
        "common_noise_passed",
        "prefix_cache_validation_passed",
        "final_semantic_grounding_passed",
    )
    if not all(sanity_payload.get(key) is True for key in required_sanity_flags):
        raise ValueError(f"{sanity_path}: sanity gate did not pass every required methodological check")
    if not isinstance(sanity_payload.get("episodes"), list) or not sanity_payload["episodes"]:
        raise ValueError(f"{sanity_path}: sanity artifact contains no checked episodes")
    if sanity_payload.get("episode_root") != str(args.episode_root.resolve()):
        raise ValueError(f"{sanity_path}: sanity artifact belongs to a different episode corpus")
    discovered_by_id = {str(metadata["episode_id"]): path for path, metadata, _ in discovered}
    sanity_targets = set()
    for row in sanity_payload["episodes"]:
        episode_id = str(row.get("episode_id", "")) if isinstance(row, Mapping) else ""
        if episode_id not in discovered_by_id:
            raise ValueError(f"{sanity_path}: unknown sanity episode {episode_id!r}")
        source = discovered_by_id[episode_id].resolve()
        if row.get("source_hdf5") != str(source) or row.get("source_hdf5_sha256") != _sha256_file(source):
            raise ValueError(f"{sanity_path}: sanity source provenance changed for {episode_id}")
        sanity_targets.add(str(row.get("target_identity")))
    if sanity_targets != set(_manifest_values(manifest, "candidate_identities")):
        raise ValueError(f"{sanity_path}: sanity did not cover both frozen semantic target identities")

    raw_paths = sorted((args.output_dir / "raw").glob("*.npz"))
    if len(raw_paths) != len(discovered):
        raise ValueError(f"Expected one raw NPZ for each of {len(discovered)} episodes, found {len(raw_paths)}")
    artifacts: dict[str, RawArtifact] = {}
    for path in raw_paths:
        artifact = _load_raw(path)
        _validate_raw_shapes(artifact)
        episode_id = str(artifact.metadata.get("episode_id"))
        if episode_id in artifacts:
            raise ValueError(f"Duplicate raw result for episode_id {episode_id!r}")
        artifacts[episode_id] = artifact

    audited_rows: list[dict[str, Any]] = []
    paired_initial_states: dict[tuple[object, ...], str] = {}
    for episode_path, metadata, target_index in discovered:
        episode_id = str(metadata["episode_id"])
        if episode_id not in artifacts:
            raise ValueError(f"Missing raw artifact for episode_id {episode_id!r}")
        artifact = artifacts[episode_id]
        _validate_freeze_fields(
            artifact.metadata,
            source=artifact.path,
            manifest=manifest,
            checkpoint=checkpoint,
            checkpoint_sha=checkpoint_sha,
        )
        expected_source = str(episode_path.resolve())
        if artifact.metadata.get("source_hdf5") != expected_source:
            raise ValueError(f"{artifact.path}: source_hdf5 does not identify {episode_path}")
        source_sha = _sha256_file(episode_path)
        if artifact.metadata.get("source_hdf5_sha256") != source_sha:
            raise ValueError(f"{artifact.path}: source HDF5 changed after observer scoring")
        if int(artifact.metadata.get("target_candidate_index", -1)) != target_index:
            raise ValueError(f"{artifact.path}: target candidate index is inconsistent with frozen semantics")
        for key in ("target_identity", "geometry_id", "layout_id", "legibility_level", "simulator_seed"):
            if artifact.metadata.get(key) != metadata.get(key):
                raise ValueError(f"{artifact.path}: {key} does not match its source HDF5 metadata")
        # This reads and validates the full episode schema, including both RGB streams.
        episode = legibility_dataset.read_episode(episode_path)
        chunks = list(legibility_dataset.iter_observer_chunks(episode))
        if len(chunks) != CHUNK_COUNT:
            raise ValueError(f"{episode_path}: did not reconstruct exactly ten observer chunks")
        actual_separation = float(
            np.linalg.norm(
                np.asarray(episode.arrays["target_pose"][0, :2], dtype=np.float64)
                - np.asarray(episode.arrays["distractor_pose"][0, :2], dtype=np.float64)
            )
        )
        if not np.isclose(actual_separation, float(metadata["separation_m"]), atol=0.003, rtol=0.0):
            raise ValueError(
                f"{episode_path}: initial physical target separation {actual_separation:.6f} m "
                f"does not match frozen {float(metadata['separation_m']):.6f} m"
            )
        initial_state_sha = hashlib.sha256(
            np.asarray(episode.arrays["sim_states"][0], dtype="<f8").tobytes(order="C")
        ).hexdigest()
        if metadata.get("initial_sim_state_sha256") != initial_state_sha:
            raise ValueError(f"{episode_path}: initial simulator-state digest does not match the recorded array")
        paired_key = (metadata["geometry_id"], metadata["layout_id"], metadata["simulator_seed"])
        previous_sha = paired_initial_states.setdefault(paired_key, initial_state_sha)
        if previous_sha != initial_state_sha:
            raise ValueError(
                f"{episode_path}: matched targets/levels for {paired_key} do not share an identical initial state"
            )
        audited_rows.append(
            {
                "episode_id": episode_id,
                "raw_path": str(artifact.path),
                "source_hdf5_sha256": source_sha,
                "seed_count": len(artifact.arrays["seeds"]),
                "residual_shape": list(artifact.arrays["residual"].shape),
                "energy_reproduced_exactly": True,
                "paired_noise_reproduced_exactly": True,
                "paired_initial_state_sha256": initial_state_sha,
                "initial_target_separation_m": actual_separation,
            }
        )

    summary_rows = score_summary.get("episodes")
    if not isinstance(summary_rows, list) or score_summary.get("episode_count") != len(discovered):
        raise ValueError(f"{score_summary_path}: invalid episode count or rows")
    summary_ids = [str(row.get("episode_id")) for row in summary_rows if isinstance(row, Mapping)]
    if len(summary_ids) != len(discovered) or set(summary_ids) != set(artifacts):
        raise ValueError(f"{score_summary_path}: episode inventory does not match raw artifacts")
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifact_kind": "pi05_libero_spatial_legibility_audit",
        "passed": True,
        "fail_closed": True,
        "episode_root": str(args.episode_root.resolve()),
        "episode_count": len(audited_rows),
        "frozen_geometry_path": str(manifest_path),
        "frozen_geometry_sha256": manifest["sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": checkpoint_sha,
        "candidate_identities": list(_manifest_values(manifest, "candidate_identities")),
        "candidate_prompts": list(_manifest_values(manifest, "candidate_prompts")),
        "sanity_path": str(sanity_path.resolve()),
        "sanity_passed": True,
        "validated": [
            "HDF5 and frozen-manifest schemas",
            "canonical frozen-geometry SHA through manifest, episodes, raw, score summary, and sanity",
            "checkpoint tree fingerprint through raw, score summary, and sanity",
            "source HDF5 byte digests",
            "raw tensor shapes and finite values",
            "exact residual-energy reproducibility",
            "exactly ten non-overlapping 10-step chunks",
            "identical initial simulator state within every geometry/layout/seed matched set",
            "deterministic paired/common noise for every seed and chunk",
            "passed prefix-cache/native-forward, same-prompt, candidate-order, common-noise, and final-grounding checks",
        ],
        "episodes": audited_rows,
    }
    output = args.output or args.output_dir / "audit.json"
    _write_json_atomic(output, report)
    print(
        json.dumps(
            {"passed": True, "episode_count": len(audited_rows), "output": str(output.resolve())}, sort_keys=True
        )
    )


def _add_observer_arguments(parser: argparse.ArgumentParser, *, sanity_defaults: bool = False) -> None:
    parser.add_argument("--episode-root", type=pathlib.Path, required=True, help="HDF5 file or rollout directory")
    parser.add_argument("--frozen-geometry", type=pathlib.Path, required=True, help="Pre-observer frozen_geometry.json")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--flow-timesteps", type=float, nargs="+", default=list(DEFAULT_FLOW_TIMESTEPS))
    parser.add_argument("--noise-samples", type=int, default=2, help="Independent common-noise samples per t and seed")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0] if sanity_defaults else list(DEFAULT_INITIAL_SEEDS))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score all frozen HDF5 episodes and resume raw NPZs")
    _add_observer_arguments(score_parser)
    stability_group = score_parser.add_mutually_exclusive_group()
    stability_group.add_argument(
        "--expand-unstable-l3",
        dest="expand_unstable_l3",
        action="store_true",
        help="Expand L3 episodes below 0.8 C30 sign agreement to seeds 0..49 (default)",
    )
    stability_group.add_argument(
        "--no-expand-unstable-l3",
        dest="expand_unstable_l3",
        action="store_false",
        help="Developer smoke only; a formal audit rejects any unstable unexpanded L3 result",
    )
    score_parser.add_argument("--stability-threshold", type=float, default=0.8)
    score_parser.add_argument("--overwrite", action="store_true")
    score_parser.set_defaults(func=score, expand_unstable_l3=True)

    sanity_parser = subparsers.add_parser("sanity", help="Run matched-noise methodological checks")
    _add_observer_arguments(sanity_parser, sanity_defaults=True)
    sanity_parser.add_argument("--max-episodes", type=int, default=2, help="Prefer one episode per target identity")
    sanity_parser.add_argument("--atol", type=float, default=1e-5)
    sanity_parser.add_argument("--rtol", type=float, default=1e-5)
    sanity_parser.add_argument("--output", type=pathlib.Path)
    sanity_parser.set_defaults(func=sanity)

    audit_parser = subparsers.add_parser("audit", help="Fail-closed audit of source, raw, and sanity artifacts")
    audit_parser.add_argument("--episode-root", type=pathlib.Path, required=True)
    audit_parser.add_argument("--frozen-geometry", type=pathlib.Path, required=True)
    audit_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    audit_parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    audit_parser.add_argument("--sanity", type=pathlib.Path, help="Defaults to OUTPUT_DIR/sanity_summary.json")
    audit_parser.add_argument("--output", type=pathlib.Path, help="Defaults to OUTPUT_DIR/audit.json")
    audit_parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Permit a filtered smoke-test corpus; formal reports must omit this flag",
    )
    audit_parser.set_defaults(func=audit)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
