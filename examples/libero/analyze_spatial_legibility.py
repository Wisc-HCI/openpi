"""Analyze the frozen pi05-LIBERO spatial-legibility experiment.

This command never runs the policy or changes trajectory parameters.  It reads
raw, seed-resolved residual tensors and simulator rollouts, writes the required
episode/aggregate tables, and generates the pre-registered plots and pass/fail
report.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import os
import pathlib
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openpi-matplotlib")
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.instruction_likelihood import legibility
from openpi.instruction_likelihood import legibility_dataset
from openpi.instruction_likelihood import legibility_statistics

LEVELS = ("A1", "L0", "L1", "L2", "L3")
MAIN_LEVELS = ("L0", "L1", "L2", "L3")
LEVEL_COLORS = {
    "A1": "#7f7f7f",
    "L0": "#1f77b4",
    "L1": "#2ca02c",
    "L2": "#ff7f0e",
    "L3": "#d62728",
}


def _load_npz(path: pathlib.Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as handle:
        if "metadata_json" not in handle:
            raise ValueError(f"{path}: missing metadata_json")
        arrays = {name: np.asarray(handle[name]) for name in handle.files if name != "metadata_json"}
        metadata = json.loads(str(handle["metadata_json"]))
    return arrays, metadata


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _episode_lookup(root: pathlib.Path) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for path in legibility_dataset.discover_episodes(root):
        metadata = legibility_dataset.read_metadata(path)
        episode_id = str(metadata["episode_id"])
        if episode_id in result:
            raise ValueError(f"Duplicate episode_id {episode_id!r}: {result[episode_id]} and {path}")
        result[episode_id] = path
    if not result:
        raise ValueError(f"No rollout HDF5 files found below {root}")
    return result


def _score_paths(root: pathlib.Path) -> list[pathlib.Path]:
    paths = [path for path in root.rglob("*.npz") if "sanity" not in path.parts and not path.name.startswith("sanity")]
    if not paths:
        raise ValueError(f"No raw score NPZ files found below {root}")
    return sorted(paths)


def _target_index(metadata: dict[str, Any], frozen: dict[str, Any], episode: dict[str, Any]) -> int:
    expected = list(frozen["candidate_identities"]).index(episode["target_identity"])
    if "target_candidate_index" not in metadata:
        raise ValueError("Raw score metadata is missing target_candidate_index")
    index = int(metadata["target_candidate_index"])
    if index not in (0, 1):
        raise ValueError(f"Invalid target candidate index {index}")
    if index != expected:
        raise ValueError(f"Raw target candidate index {index} disagrees with frozen semantics ({expected})")
    return index


def _trajectory_measurements(
    episode: legibility_dataset.RecordedEpisode,
    *,
    geometric_beta: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    arrays = episode.arrays
    positions = np.asarray(arrays["actual_eef_pos"][:101], dtype=np.float64)
    desired = np.concatenate([positions[:1], np.asarray(arrays["desired_eef_pos"][:100], dtype=np.float64)], axis=0)
    start = positions[0]
    true_goal = desired[-1]
    target_xyz = np.asarray(arrays["target_pose"][0, :3], dtype=np.float64)
    distractor_xyz = np.asarray(arrays["distractor_pose"][0, :3], dtype=np.float64)
    distractor_goal = distractor_xyz + (true_goal - target_xyz)
    away = true_goal - distractor_goal
    actual_metrics = legibility.compute_trajectory_metrics(
        positions,
        start,
        true_goal,
        distractor_goal,
        away,
    )
    planned_metrics = legibility.compute_trajectory_metrics(
        desired,
        start,
        true_goal,
        distractor_goal,
        away,
    )
    observer = legibility.geometric_goal_observer(
        positions,
        true_goal,
        distractor_goal,
        beta=geometric_beta,
    )
    tracking_error = np.linalg.norm(positions[1:] - desired[1:], axis=1)
    result = {
        "actual_path_length_m": actual_metrics.path_length,
        "planned_path_length_m": planned_metrics.path_length,
        "direct_distance_m": float(np.linalg.norm(true_goal - start)),
        "actual_max_lateral_deviation_m": actual_metrics.max_lateral_deviation,
        "actual_abs_max_lateral_deviation_m": abs(actual_metrics.max_lateral_deviation),
        "planned_max_lateral_deviation_m": planned_metrics.max_lateral_deviation,
        "max_deviation_progress": actual_metrics.max_deviation_progress,
        "min_distractor_distance_m": actual_metrics.min_distractor_distance,
        "mean_tracking_error_m": float(np.mean(tracking_error)),
        "max_tracking_error_m": float(np.max(tracking_error)),
        "geometric_confidence_10": float(observer.true_goal_confidence[10]),
        "geometric_confidence_20": float(observer.true_goal_confidence[20]),
        "geometric_confidence_30": float(observer.true_goal_confidence[30]),
        "geometric_confidence_40": float(observer.true_goal_confidence[40]),
    }
    curves = {
        "actual_positions": positions,
        "desired_positions": desired,
        "true_goal": true_goal,
        "distractor_goal": distractor_goal,
        "geometric_progress": observer.progress,
        "geometric_confidence": observer.true_goal_confidence,
    }
    return result, curves


def _build_rows(
    episode_root: pathlib.Path,
    score_root: pathlib.Path,
    frozen: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    episodes = _episode_lookup(episode_root)
    episode_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    seen: set[str] = set()
    checkpoint_fingerprints: set[str] = set()
    try:
        geometric_beta = float(frozen["simulation"]["observer_beta_per_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Frozen manifest is missing simulation.observer_beta_per_m") from exc
    if not np.isfinite(geometric_beta) or geometric_beta <= 0.0:
        raise ValueError("Frozen geometric observer beta must be positive and finite")

    for score_path in _score_paths(score_root):
        arrays, score_metadata = _load_npz(score_path)
        episode_id = str(score_metadata.get("episode_id", ""))
        if not episode_id:
            raise ValueError(f"{score_path}: score metadata has no episode_id")
        if episode_id in seen:
            raise ValueError(f"Duplicate score file for {episode_id}")
        seen.add(episode_id)
        if episode_id not in episodes:
            raise ValueError(f"{score_path}: no matching rollout for episode {episode_id}")
        episode = legibility_dataset.read_episode(episodes[episode_id], load_images=False)
        metadata = episode.metadata
        if metadata["frozen_geometry_sha256"] != frozen["sha256"]:
            raise ValueError(f"{episode_id}: rollout was not generated from the supplied frozen manifest")
        if score_metadata.get("frozen_geometry_sha256") != frozen["sha256"]:
            raise ValueError(f"{episode_id}: score digest does not match the supplied frozen manifest")
        if score_metadata.get("candidate_identities") != frozen["candidate_identities"]:
            raise ValueError(f"{episode_id}: raw candidate identities differ from the frozen manifest")
        if score_metadata.get("candidate_prompts") != frozen["candidate_prompts"]:
            raise ValueError(f"{episode_id}: raw candidate prompts differ from the frozen manifest")
        expected_source = str(episodes[episode_id].resolve())
        if score_metadata.get("source_hdf5") != expected_source:
            raise ValueError(f"{episode_id}: raw source_hdf5 does not identify its matched rollout")
        if score_metadata.get("source_hdf5_sha256") != _sha256_file(episodes[episode_id]):
            raise ValueError(f"{episode_id}: rollout HDF5 changed after residual scoring")
        checkpoint_fingerprint = str(score_metadata.get("checkpoint_manifest_sha256", ""))
        if len(checkpoint_fingerprint) != 64:
            raise ValueError(f"{episode_id}: raw score has no valid checkpoint fingerprint")
        checkpoint_fingerprints.add(checkpoint_fingerprint)
        for field in ("target_identity", "geometry_id", "layout_id", "legibility_level", "simulator_seed"):
            if score_metadata.get(field) != metadata.get(field):
                raise ValueError(f"{episode_id}: raw {field} does not match the rollout")
        if "residual" not in arrays:
            raise ValueError(f"{score_path}: missing raw residual tensor")
        residual = arrays["residual"]
        target_index = _target_index(score_metadata, frozen, metadata)
        energy = legibility_statistics.residual_energy_by_seed(residual)
        canonical, truth = legibility_statistics.canonical_and_truth_margins(
            energy, target_candidate_index=target_index
        )
        seed_scores = legibility_statistics.prefix_scores(truth)
        seeds = np.asarray(arrays.get("seeds"), dtype=np.int64)
        initial_seeds = tuple(int(value) for value in score_metadata.get("initial_seeds", ()))
        if seeds.ndim != 1 or len(initial_seeds) != 10 or not set(initial_seeds).issubset(set(seeds.tolist())):
            raise ValueError(f"{episode_id}: raw score does not preserve the registered initial ten seeds")
        initial_indices = [int(np.flatnonzero(seeds == seed)[0]) for seed in initial_seeds]
        score_summary = legibility_statistics.summarize_episode_residuals(residual, target_candidate_index=target_index)
        trajectory, curve = _trajectory_measurements(episode, geometric_beta=geometric_beta)
        curves[episode_id] = {
            **curve,
            "truth_margin_by_seed": truth,
            "canonical_margin_by_seed": canonical,
            "C30_by_seed": seed_scores["C30"],
            "C30_initial_by_seed": seed_scores["C30"][initial_indices],
        }

        row: dict[str, Any] = {
            "episode_id": episode_id,
            "rollout_path": str(episodes[episode_id].resolve()),
            "score_path": str(score_path.resolve()),
            "checkpoint_manifest_sha256": checkpoint_fingerprint,
            "geometry_id": metadata["geometry_id"],
            "layout_id": metadata["layout_id"],
            "target_identity": metadata["target_identity"],
            "distractor_identity": metadata["distractor_identity"],
            "target_side": metadata["target_side"],
            "target_prompt": metadata["target_prompt"],
            "distractor_prompt": metadata["distractor_prompt"],
            "separation_m": float(metadata["separation_m"]),
            "legibility_level": metadata["legibility_level"],
            "level_order": legibility_statistics.LEVEL_ORDER[metadata["legibility_level"]],
            "simulator_seed": int(metadata["simulator_seed"]),
            "task_success": bool(metadata["task_success"]),
            "collision_free": bool(metadata["collision_free"]),
            "seed_count": int(residual.shape[0]),
            "normalized_action_out_of_range_fraction": float(
                np.mean(np.abs(np.asarray(arrays["normalized_actions"])[..., :7]) > 1.0)
            ),
            **trajectory,
        }
        for score_name in (*legibility_statistics.PREFIX_CHUNKS, "flat_score", "linear_score", "exponential_score"):
            summary = score_summary[score_name]
            row[score_name] = float(summary["mean"])
            row[f"{score_name}_seed_sd"] = float(summary["standard_deviation"])
            row[f"{score_name}_ci95_low"] = float(summary["ci95_low"])
            row[f"{score_name}_ci95_high"] = float(summary["ci95_high"])
            row[f"{score_name}_seed_sign_agreement"] = float(summary["mean_sign_agreement"])
            row[f"{score_name}_true_seed_fraction"] = float(summary["true_support_fraction"])
        for name in legibility_statistics.PREFIX_CHUNKS:
            row[f"prediction_{name[1:]}"] = (
                metadata["target_identity"] if row[name] > 0.0 else metadata["distractor_identity"]
            )
            row[f"correct_{name[1:]}"] = bool(row[name] > 0.0)
        episode_rows.append(row)

        for chunk_index in range(10):
            summary = score_summary["local_truth_margin"][chunk_index]
            chunk_rows.append(
                {
                    "episode_id": episode_id,
                    "geometry_id": metadata["geometry_id"],
                    "layout_id": metadata["layout_id"],
                    "target_identity": metadata["target_identity"],
                    "target_side": metadata["target_side"],
                    "legibility_level": metadata["legibility_level"],
                    "simulator_seed": int(metadata["simulator_seed"]),
                    "chunk_index": chunk_index + 1,
                    "progress_end": (chunk_index + 1) / 10.0,
                    "truth_margin": float(summary["mean"]),
                    "margin_seed_sd": float(summary["standard_deviation"]),
                    "ci95_low": float(summary["ci95_low"]),
                    "ci95_high": float(summary["ci95_high"]),
                    "mean_sign_agreement": float(summary["mean_sign_agreement"]),
                    "true_seed_fraction": float(summary["true_support_fraction"]),
                }
            )

    missing_scores = sorted(set(episodes) - seen)
    if missing_scores:
        raise ValueError(f"Missing score files for {len(missing_scores)} rollouts; first: {missing_scores[:5]}")
    if len(checkpoint_fingerprints) != 1:
        raise ValueError("Raw score files were not all produced by one frozen checkpoint tree")

    l0_lengths: dict[tuple[object, ...], float] = {}
    group_fields = ("geometry_id", "layout_id", "simulator_seed", "target_identity")
    for row in episode_rows:
        if row["legibility_level"] == "L0":
            key = tuple(row[field] for field in group_fields)
            if key in l0_lengths:
                raise ValueError(f"Duplicate L0 in matched group {key}")
            l0_lengths[key] = float(row["actual_path_length_m"])
    for row in episode_rows:
        key = tuple(row[field] for field in group_fields)
        if key not in l0_lengths:
            raise ValueError(f"Missing L0 baseline for matched group {key}")
        baseline = l0_lengths[key]
        row["excess_path_length"] = (float(row["actual_path_length_m"]) - baseline) / baseline

    episode_rows.sort(
        key=lambda row: (
            row["geometry_id"],
            row["layout_id"],
            row["simulator_seed"],
            row["target_identity"],
            row["level_order"],
        )
    )
    chunk_rows.sort(key=lambda row: (row["episode_id"], row["chunk_index"]))
    return episode_rows, chunk_rows, curves


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _validate_complete_design(episodes: list[dict[str, Any]], frozen: dict[str, Any]) -> list[int]:
    geometries = {str(item["geometry_id"]): float(item["separation_m"]) for item in frozen["geometries"]}
    identities = tuple(str(value) for value in frozen["candidate_identities"])
    seeds = sorted({int(row["simulator_seed"]) for row in episodes})
    expected = {
        (geometry, layout, seed, target, level)
        for geometry in geometries
        for layout in ("A", "B")
        for seed in seeds
        for target in identities
        for level in LEVELS
    }
    observed = {
        (
            str(row["geometry_id"]),
            str(row["layout_id"]),
            int(row["simulator_seed"]),
            str(row["target_identity"]),
            str(row["legibility_level"]),
        )
        for row in episodes
    }
    if len(observed) != len(episodes):
        raise ValueError("Formal analysis found duplicate experimental condition rows")
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            "Formal report requires the complete frozen G1/G2 x A/B x targets x A1/L0-L3 design "
            f"for every simulator seed; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for row in episodes:
        expected_separation = geometries[str(row["geometry_id"])]
        if not np.isclose(float(row["separation_m"]), expected_separation, atol=1e-9, rtol=0.0):
            raise ValueError(f"{row['episode_id']}: separation differs from its frozen geometry")
    return seeds


def _aggregate_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for level in LEVELS:
        selected = [row for row in episodes if row["legibility_level"] == level]
        if not selected:
            continue
        output.append(
            {
                "scope": "level",
                "value": level,
                "episode_count": len(selected),
                "task_success_rate": _mean(selected, "task_success"),
                "mean_C20": _mean(selected, "C20"),
                "mean_C30": _mean(selected, "C30"),
                "mean_C40": _mean(selected, "C40"),
                "accuracy_20": _mean(selected, "correct_20"),
                "accuracy_30": _mean(selected, "correct_30"),
                "accuracy_40": _mean(selected, "correct_40"),
                "mean_path_length_m": _mean(selected, "actual_path_length_m"),
                "mean_excess_path_length": _mean(selected, "excess_path_length"),
                "mean_abs_lateral_deviation_m": _mean(selected, "actual_abs_max_lateral_deviation_m"),
                "mean_C30_seed_sign_agreement": _mean(selected, "C30_seed_sign_agreement"),
            }
        )
    for target in sorted({str(row["target_identity"]) for row in episodes}):
        selected = [row for row in episodes if row["target_identity"] == target]
        output.append(
            {
                "scope": "target_identity",
                "value": target,
                "episode_count": len(selected),
                "task_success_rate": _mean(selected, "task_success"),
                "mean_C20": _mean(selected, "C20"),
                "mean_C30": _mean(selected, "C30"),
                "mean_C40": _mean(selected, "C40"),
                "accuracy_20": _mean(selected, "correct_20"),
                "accuracy_30": _mean(selected, "correct_30"),
                "accuracy_40": _mean(selected, "correct_40"),
            }
        )
    for side in sorted({str(row["target_side"]) for row in episodes}):
        selected = [row for row in episodes if row["target_side"] == side]
        output.append(
            {
                "scope": "physical_side",
                "value": side,
                "episode_count": len(selected),
                "task_success_rate": _mean(selected, "task_success"),
                "mean_C20": _mean(selected, "C20"),
                "mean_C30": _mean(selected, "C30"),
                "mean_C40": _mean(selected, "C40"),
                "accuracy_20": _mean(selected, "correct_20"),
                "accuracy_30": _mean(selected, "correct_30"),
                "accuracy_40": _mean(selected, "correct_40"),
            }
        )
    return output


def _complete_group(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        grouped[(row["geometry_id"], row["layout_id"], row["simulator_seed"], row["target_identity"])].append(row)
    for rows in grouped.values():
        if {row["legibility_level"] for row in rows} == set(LEVELS):
            return rows
    raise ValueError("No complete A1/L0-L3 group is available for representative trajectory plots")


def _save_figure(path: pathlib.Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_trajectories(
    episodes: list[dict[str, Any]], curves: dict[str, dict[str, np.ndarray]], output: pathlib.Path
) -> None:
    representative = _complete_group(episodes)
    for early_only, name in ((False, "01_top_down_trajectories.png"), (True, "02_first_40pct_trajectories.png")):
        plt.figure(figsize=(7, 6))
        for row in sorted(representative, key=lambda item: item["level_order"]):
            curve = curves[row["episode_id"]]
            count = 41 if early_only else 101
            xyz = curve["actual_positions"][:count]
            plt.plot(xyz[:, 0], xyz[:, 1], color=LEVEL_COLORS[row["legibility_level"]], label=row["legibility_level"])
        curve = curves[representative[0]["episode_id"]]
        plt.scatter(*curve["true_goal"][:2], marker="*", s=180, color="black", label="true pre-grasp")
        plt.scatter(*curve["distractor_goal"][:2], marker="X", s=100, color="#9467bd", label="distractor")
        plt.xlabel("world x (m)")
        plt.ylabel("world y (m)")
        plt.title("Executed pre-grasp trajectories" + (" (first 40%)" if early_only else ""))
        plt.axis("equal")
        plt.legend(ncol=2)
        _save_figure(output / name)


def _mean_curve(items: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([np.interp(grid, x, y) for x, y in items])
    mean = np.mean(values, axis=0)
    error = np.std(values, axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros_like(mean)
    return mean, 1.96 * error


def _plot_geometric_observer(
    episodes: list[dict[str, Any]], curves: dict[str, dict[str, np.ndarray]], output: pathlib.Path
) -> None:
    grid = np.linspace(0.0, 1.0, 101)
    plt.figure(figsize=(8, 5))
    for level in LEVELS:
        items = [
            (curves[row["episode_id"]]["geometric_progress"], curves[row["episode_id"]]["geometric_confidence"])
            for row in episodes
            if row["legibility_level"] == level
        ]
        if not items:
            continue
        mean, error = _mean_curve(items, grid)
        plt.plot(grid, mean, color=LEVEL_COLORS[level], label=level)
        plt.fill_between(grid, mean - error, mean + error, color=LEVEL_COLORS[level], alpha=0.15)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.axvline(0.3, color="black", linestyle=":", linewidth=1)
    plt.xlabel("normalized pre-grasp progress")
    plt.ylabel("geometric observer P(true goal)")
    plt.ylim(0.0, 1.0)
    plt.legend(ncol=3)
    _save_figure(output / "03_geometric_observer_confidence.png")


def _plot_local_margin(chunks: list[dict[str, Any]], output: pathlib.Path) -> None:
    plt.figure(figsize=(8, 5))
    for level in LEVELS:
        selected = [row for row in chunks if row["legibility_level"] == level]
        if not selected:
            continue
        grouped = defaultdict(list)
        for row in selected:
            grouped[int(row["chunk_index"])].append(float(row["truth_margin"]))
        x = np.asarray(sorted(grouped), dtype=np.float64) / 10.0
        y = np.asarray([np.mean(grouped[index]) for index in sorted(grouped)])
        plt.plot(x, y, marker="o", color=LEVEL_COLORS[level], label=level)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.axvline(0.3, color="black", linestyle=":", linewidth=1)
    plt.xlabel("chunk end progress")
    plt.ylabel("local truth-aligned residual margin")
    plt.legend(ncol=3)
    _save_figure(output / "04_pi05_local_margin.png")


def _plot_prefix_scores(episodes: list[dict[str, Any]], output: pathlib.Path) -> None:
    x = np.arange(len(LEVELS))
    plt.figure(figsize=(8, 5))
    for name, marker in (("C20", "o"), ("C30", "s"), ("C40", "^")):
        means, errors = [], []
        for level in LEVELS:
            values = np.asarray([row[name] for row in episodes if row["legibility_level"] == level], dtype=float)
            means.append(np.mean(values))
            errors.append(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0)
        plt.errorbar(x, means, yerr=errors, marker=marker, capsize=3, label=name)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, LEVELS)
    plt.xlabel("legibility level")
    plt.ylabel("truth-aligned prefix margin")
    plt.legend()
    _save_figure(output / "05_prefix_scores_by_level.png")


def _plot_efficiency_scatters(episodes: list[dict[str, Any]], output: pathlib.Path) -> None:
    for field, xlabel, name in (
        ("actual_abs_max_lateral_deviation_m", "actual max lateral deviation (m)", "06_C30_vs_deviation.png"),
        ("excess_path_length", "excess path length relative to matched L0", "07_C30_vs_path_cost.png"),
    ):
        plt.figure(figsize=(7, 5))
        for level in LEVELS:
            selected = [row for row in episodes if row["legibility_level"] == level]
            plt.scatter(
                [row[field] for row in selected],
                [row["C30"] for row in selected],
                color=LEVEL_COLORS[level],
                alpha=0.75,
                label=level,
            )
        plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
        plt.xlabel(xlabel)
        plt.ylabel("C30")
        plt.legend(ncol=3)
        _save_figure(output / name)


def _plot_paired_monotonicity(episodes: list[dict[str, Any]], output: pathlib.Path) -> None:
    grouped: dict[tuple[object, ...], dict[str, float]] = defaultdict(dict)
    for row in episodes:
        if row["legibility_level"] in MAIN_LEVELS:
            key = (row["geometry_id"], row["layout_id"], row["simulator_seed"], row["target_identity"])
            grouped[key][row["legibility_level"]] = float(row["C30"])
    plt.figure(figsize=(8, 5))
    x = np.arange(4)
    for levels in grouped.values():
        if set(levels) == set(MAIN_LEVELS):
            plt.plot(x, [levels[level] for level in MAIN_LEVELS], color="#777777", alpha=0.25)
    means = [
        np.mean([levels[level] for levels in grouped.values() if set(levels) == set(MAIN_LEVELS)])
        for level in MAIN_LEVELS
    ]
    plt.plot(x, means, color="black", linewidth=3, marker="o", label="mean")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, MAIN_LEVELS)
    plt.xlabel("legibility level")
    plt.ylabel("matched C30")
    plt.legend()
    _save_figure(output / "08_paired_monotonicity.png")


def _plot_accuracy(episodes: list[dict[str, Any]], output: pathlib.Path) -> None:
    x = np.arange(len(LEVELS))
    width = 0.25
    plt.figure(figsize=(9, 5))
    for offset, endpoint in zip((-width, 0.0, width), (20, 30, 40), strict=True):
        values = [
            np.mean([row[f"correct_{endpoint}"] for row in episodes if row["legibility_level"] == level])
            for level in LEVELS
        ]
        plt.bar(x + offset, values, width=width, label=f"{endpoint}%")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, LEVELS)
    plt.ylabel("accuracy")
    plt.ylim(0.0, 1.05)
    plt.legend()
    _save_figure(output / "09_accuracy_20_30_40.png")


def _plot_seed_distributions(
    episodes: list[dict[str, Any]], curves: dict[str, dict[str, np.ndarray]], output: pathlib.Path
) -> None:
    values = [
        np.concatenate(
            [curves[row["episode_id"]]["C30_initial_by_seed"] for row in episodes if row["legibility_level"] == level]
        )
        for level in LEVELS
    ]
    plt.figure(figsize=(8, 5))
    plt.boxplot(values, tick_labels=LEVELS, showfliers=False)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("legibility level")
    plt.ylabel("per-flow-seed C30")
    _save_figure(output / "10_flow_seed_distributions.png")


def _plot_bias(episodes: list[dict[str, Any]], output: pathlib.Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, field, title in (
        (axes[0], "target_identity", "semantic identity"),
        (axes[1], "target_side", "physical side"),
    ):
        values = sorted({str(row[field]) for row in episodes})
        means = [_mean([row for row in episodes if row[field] == value], "C30") for value in values]
        accuracy = [_mean([row for row in episodes if row[field] == value], "correct_30") for value in values]
        x = np.arange(len(values))
        axis.bar(x - 0.18, means, width=0.36, label="mean C30")
        axis.bar(x + 0.18, accuracy, width=0.36, label="accuracy@30")
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(x, values, rotation=15)
        axis.set_title(title)
        axis.legend()
    _save_figure(output / "11_identity_physical_side_bias.png")


def _matched_geometry_checks(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[object, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in episodes:
        key = (row["geometry_id"], row["layout_id"], row["simulator_seed"], row["target_identity"])
        grouped[key][str(row["legibility_level"])] = row
    rows = []
    for key, levels in sorted(grouped.items()):
        if set(levels) != set(LEVELS):
            raise ValueError(f"Incomplete matched geometry group {key}")

        def increasing(
            field: str, *, absolute: bool = False, matched_levels: dict[str, dict[str, Any]] = levels
        ) -> bool:
            values = [float(matched_levels[level][field]) for level in MAIN_LEVELS]
            if absolute:
                values = [abs(value) for value in values]
            return all(right > left for left, right in itertools.pairwise(values))

        checks = {
            "deviation_L0_to_L3_increasing": increasing("actual_max_lateral_deviation_m", absolute=True),
            "path_length_L0_to_L3_increasing": increasing("actual_path_length_m"),
            "geometric_confidence30_L0_to_L3_increasing": increasing("geometric_confidence_30"),
            "A1_geometric_confidence30_below_L0": float(levels["A1"]["geometric_confidence_30"])
            < float(levels["L0"]["geometric_confidence_30"]),
            "L0_not_near_certain_at_10pct": float(levels["L0"]["geometric_confidence_10"]) < 0.90,
            "all_collision_free": all(bool(row["collision_free"]) for row in levels.values()),
        }
        rows.append(
            {
                "geometry_id": key[0],
                "layout_id": key[1],
                "simulator_seed": key[2],
                "target_identity": key[3],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {
        "matched_group_count": len(rows),
        "all_groups_passed": bool(rows and all(row["passed"] for row in rows)),
        "all_deviation_monotone": all(row["checks"]["deviation_L0_to_L3_increasing"] for row in rows),
        "all_path_length_monotone": all(row["checks"]["path_length_L0_to_L3_increasing"] for row in rows),
        "all_geometric_confidence_monotone": all(
            row["checks"]["geometric_confidence30_L0_to_L3_increasing"] for row in rows
        ),
        "all_A1_below_L0": all(row["checks"]["A1_geometric_confidence30_below_L0"] for row in rows),
        "all_L0_ambiguous_at_10pct": all(row["checks"]["L0_not_near_certain_at_10pct"] for row in rows),
        "all_collision_free": all(row["checks"]["all_collision_free"] for row in rows),
        "groups": rows,
    }


def _summary(
    episodes: list[dict[str, Any]],
    monotonicity_by_endpoint: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    monotonicity = monotonicity_by_endpoint["C30"]
    accuracy = {
        str(endpoint): float(np.mean([row[f"correct_{endpoint}"] for row in episodes])) for endpoint in (20, 30, 40)
    }
    main_episodes = [row for row in episodes if row["legibility_level"] in MAIN_LEVELS]
    main_accuracy = {
        str(endpoint): float(np.mean([row[f"correct_{endpoint}"] for row in main_episodes]))
        for endpoint in (20, 30, 40)
    }
    balanced = {
        str(endpoint): legibility_statistics.balanced_accuracy(episodes, f"C{endpoint}") for endpoint in (20, 30, 40)
    }
    main_balanced = {
        str(endpoint): legibility_statistics.balanced_accuracy(main_episodes, f"C{endpoint}")
        for endpoint in (20, 30, 40)
    }
    level_means = {
        level: {
            name: _mean([row for row in episodes if row["legibility_level"] == level], name)
            for name in ("C20", "C30", "C40", "actual_path_length_m", "excess_path_length")
        }
        for level in LEVELS
    }
    task_success = float(np.mean([row["task_success"] for row in episodes]))
    matched_geometry = _matched_geometry_checks(episodes)
    collision_free = bool(matched_geometry["all_collision_free"])
    l3_stabilities = [float(row["C30_seed_sign_agreement"]) for row in episodes if row["legibility_level"] == "L3"]
    mean_l3_stability = float(np.mean(l3_stabilities))
    minimum_l3_stability = float(np.min(l3_stabilities))
    geometric_confidence = {
        level: _mean([row for row in episodes if row["legibility_level"] == level], "geometric_confidence_30")
        for level in LEVELS
    }
    l0_confidence_10 = _mean([row for row in episodes if row["legibility_level"] == "L0"], "geometric_confidence_10")
    criteria = {
        "all_executed_trajectories_collision_free": collision_free,
        "geometry_L0_to_L3_deviation_increasing_in_every_matched_group": matched_geometry["all_deviation_monotone"],
        "geometry_L0_to_L3_confidence_increasing_in_every_matched_group": matched_geometry[
            "all_geometric_confidence_monotone"
        ],
        "geometry_A1_confidence_below_L0_in_every_matched_group": matched_geometry["all_A1_below_L0"],
        "geometry_L0_not_near_certain_at_10pct_in_every_matched_group": matched_geometry["all_L0_ambiguous_at_10pct"],
        "geometry_L0_to_L3_path_cost_increasing_in_every_matched_group": matched_geometry["all_path_length_monotone"],
        "scripted_task_success_at_least_95pct": task_success >= 0.95,
        "main_L0_L3_balanced_accuracy_20_at_least_70pct": main_balanced["20"] >= 0.70,
        "main_L0_L3_balanced_accuracy_30_at_least_80pct": main_balanced["30"] >= 0.80,
        "main_L0_L3_balanced_accuracy_40_at_least_80pct": main_balanced["40"] >= 0.80,
        "mean_spearman_rho_at_least_0_6": monotonicity["mean_spearman_rho"] >= 0.6,
        "L3_above_L0_at_least_75pct": monotonicity["L3_above_L0_fraction"] >= 0.75,
        "adjacent_order_success_at_least_75pct": monotonicity["adjacent_order_success_fraction"] >= 0.75,
        "every_L3_condition_seed_sign_agreement_at_least_80pct": minimum_l3_stability >= 0.80,
    }
    return {
        "schema_version": 1,
        "frozen_geometry_sha256": frozen["sha256"],
        "checkpoint_manifest_sha256": str(episodes[0]["checkpoint_manifest_sha256"]),
        "episode_count": len(episodes),
        "task_success_rate": task_success,
        "all_collision_free": collision_free,
        "mean_geometric_confidence_30_by_level": geometric_confidence,
        "mean_L0_geometric_confidence_10": l0_confidence_10,
        "accuracy": accuracy,
        "main_L0_L3_accuracy": main_accuracy,
        "balanced_accuracy": balanced,
        "main_L0_L3_balanced_accuracy": main_balanced,
        "level_means": level_means,
        "monotonicity": {key: value for key, value in monotonicity.items() if key != "groups"},
        "monotonicity_by_endpoint": {
            endpoint: {key: value for key, value in result.items() if key != "groups"}
            for endpoint, result in monotonicity_by_endpoint.items()
        },
        "matched_geometry": matched_geometry,
        "mean_L3_C30_seed_sign_agreement": mean_l3_stability,
        "minimum_L3_C30_seed_sign_agreement": minimum_l3_stability,
        "preregistered_criteria": criteria,
        "all_preregistered_criteria_passed": all(criteria.values()),
        "interpretation_guard": "Failures are reported as failures; geometry and levels were not retuned from pi05 scores.",
    }


def _write_report(path: pathlib.Path, summary: dict[str, Any]) -> None:
    criteria = summary["preregistered_criteria"]
    lines = [
        "# pi05-LIBERO spatial-legibility report",
        "",
        f"Frozen geometry digest: `{summary['frozen_geometry_sha256']}`",
        f"Episodes: {summary['episode_count']}",
        f"Scripted task success: {summary['task_success_rate']:.1%}",
        "",
        "## Early target inference",
        "",
        f"- Main L0-L3 balanced accuracy at 20%: {summary['main_L0_L3_balanced_accuracy']['20']:.1%}",
        f"- Main L0-L3 balanced accuracy at 30% (primary): {summary['main_L0_L3_balanced_accuracy']['30']:.1%}",
        f"- Main L0-L3 balanced accuracy at 40%: {summary['main_L0_L3_balanced_accuracy']['40']:.1%}",
        f"- Including anti-legible A1 at 30% (diagnostic): {summary['balanced_accuracy']['30']:.1%}",
        "",
        "## Monotonicity",
        "",
        f"- Mean matched Spearman rho(level, C30): {summary['monotonicity']['mean_spearman_rho']:.3f}",
        f"- Fraction L3 > L0: {summary['monotonicity']['L3_above_L0_fraction']:.1%}",
        f"- Adjacent-order success: {summary['monotonicity']['adjacent_order_success_fraction']:.1%}",
        f"- Mean L3 seed sign agreement: {summary['mean_L3_C30_seed_sign_agreement']:.1%}",
        f"- Minimum per-condition L3 seed sign agreement: {summary['minimum_L3_C30_seed_sign_agreement']:.1%}",
        "",
        "## Preregistered criteria",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in criteria.items())
    lines.extend(
        [
            "",
            "No trajectory parameter or weighting function is selected from these pi05 results. Any failed criterion remains reported as a failure.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> None:
    frozen = legibility_dataset.load_frozen_manifest(args.frozen_geometry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes, chunks, curves = _build_rows(args.episode_root, args.score_root, frozen)
    simulator_seeds = _validate_complete_design(episodes, frozen)
    monotonicity_by_endpoint = {
        endpoint: legibility_statistics.matched_monotonicity(episodes, score_key=endpoint)
        for endpoint in ("C20", "C30", "C40")
    }
    monotonicity = monotonicity_by_endpoint["C30"]
    aggregates = _aggregate_rows(episodes)
    summary = _summary(episodes, monotonicity_by_endpoint, frozen)
    summary["simulator_seeds"] = simulator_seeds
    aggregates.append(
        {
            "scope": "preregistered_overall",
            "value": "L0-L3",
            "episode_count": len([row for row in episodes if row["legibility_level"] in MAIN_LEVELS]),
            "balanced_accuracy_20": summary["main_L0_L3_balanced_accuracy"]["20"],
            "balanced_accuracy_30": summary["main_L0_L3_balanced_accuracy"]["30"],
            "balanced_accuracy_40": summary["main_L0_L3_balanced_accuracy"]["40"],
            "mean_spearman_rho_C20": monotonicity_by_endpoint["C20"]["mean_spearman_rho"],
            "mean_spearman_rho_C30": monotonicity_by_endpoint["C30"]["mean_spearman_rho"],
            "mean_spearman_rho_C40": monotonicity_by_endpoint["C40"]["mean_spearman_rho"],
            "mean_L3_minus_L0_C30": monotonicity["mean_L3_minus_L0"],
            "L3_above_L0_fraction_C30": monotonicity["L3_above_L0_fraction"],
            "adjacent_order_success_fraction_C30": monotonicity["adjacent_order_success_fraction"],
            "mean_A1_minus_L0_C30": monotonicity["mean_A1_minus_L0"],
        }
    )

    _write_csv(args.output_dir / "episode_metrics.csv", episodes)
    _write_csv(args.output_dir / "chunk_metrics.csv", chunks)
    _write_csv(args.output_dir / "aggregate_metrics.csv", aggregates)
    monotonicity_rows = [
        {"endpoint": endpoint, **row}
        for endpoint, endpoint_summary in monotonicity_by_endpoint.items()
        for row in endpoint_summary["groups"]
    ]
    _write_csv(args.output_dir / "matched_monotonicity.csv", monotonicity_rows)
    _write_json(args.output_dir / "summary.json", summary)
    _write_report(args.output_dir / "report.md", summary)

    _plot_trajectories(episodes, curves, args.output_dir)
    _plot_geometric_observer(episodes, curves, args.output_dir)
    _plot_local_margin(chunks, args.output_dir)
    _plot_prefix_scores(episodes, args.output_dir)
    _plot_efficiency_scatters(episodes, args.output_dir)
    _plot_paired_monotonicity(episodes, args.output_dir)
    _plot_accuracy(episodes, args.output_dir)
    _plot_seed_distributions(episodes, curves, args.output_dir)
    _plot_bias(episodes, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=pathlib.Path, required=True)
    parser.add_argument("--score-root", type=pathlib.Path, required=True)
    parser.add_argument("--frozen-geometry", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    return parser


if __name__ == "__main__":
    analyze(_parser().parse_args())
