"""Analyze residual scores from ``evaluate_spatial_legibility.py``."""

from __future__ import annotations

from collections import defaultdict
import csv
import dataclasses
import json
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi.instruction_likelihood import droid_dataset


@dataclasses.dataclass(frozen=True)
class Args:
    input_dir: Path = Path("artifacts/spatial_legibility_eval_39999")
    recognition_threshold: float = 0.75


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def _binary_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    # Stable logistic cross entropy: max(x, 0) - x*y + log(1 + exp(-abs(x))).
    return float(np.mean(np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))))


def _fit_positive_multiplier(raw_logits: np.ndarray, labels: np.ndarray) -> float:
    raw_logits = np.asarray(raw_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    candidates = np.logspace(-4, 4, 4001)
    losses = np.asarray([_binary_nll(raw_logits * candidate, labels) for candidate in candidates])
    return float(candidates[int(np.argmin(losses))])


def _mean(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(values)) if len(values) else float("nan")


def _sample_std(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _aggregate_seed_rows(score_rows: list[dict[str, str]]) -> list[dict]:
    grouped = defaultdict(list)
    key_fields = (
        "split",
        "condition",
        "pair_index",
        "episode_id",
        "target_side",
        "instruction",
        "chunk_index",
        "phase",
        "start",
        "stop",
        "executed_steps",
        "progress_pregrasp",
        "progress_full",
        "num_samples",
        "tau_min",
        "tau_max",
        "action_dims",
        "cartesian_x",
        "cartesian_y",
        "cartesian_z",
    )
    for row in score_rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)

    result = []
    for key, rows in grouped.items():
        base = dict(zip(key_fields, key, strict=True))
        margins = np.asarray([float(row["margin_alternative_minus_true"]) for row in rows])
        mean_margin = float(np.mean(margins))
        mean_sign = 1 if mean_margin >= 0 else -1
        result.append(
            {
                **base,
                "pair_index": int(base["pair_index"]),
                "chunk_index": int(base["chunk_index"]),
                "start": int(base["start"]),
                "stop": int(base["stop"]),
                "executed_steps": int(base["executed_steps"]),
                "progress_pregrasp": float(base["progress_pregrasp"]),
                "progress_full": float(base["progress_full"]),
                "num_samples": int(base["num_samples"]),
                "tau_min": float(base["tau_min"]),
                "tau_max": float(base["tau_max"]),
                "action_dims": int(base["action_dims"]),
                "cartesian_x": float(base["cartesian_x"]),
                "cartesian_y": float(base["cartesian_y"]),
                "cartesian_z": float(base["cartesian_z"]),
                "num_seeds": len(rows),
                "energy_left": _mean(float(row["energy_left"]) for row in rows),
                "energy_right": _mean(float(row["energy_right"]) for row in rows),
                "energy_true": _mean(float(row["energy_true"]) for row in rows),
                "energy_alternative": _mean(float(row["energy_alternative"]) for row in rows),
                "margin_alternative_minus_true": mean_margin,
                "margin_seed_sd": _sample_std(margins),
                "seed_sign_agreement": float(np.mean(np.sign(margins + 1e-15) == mean_sign)),
            }
        )
    return sorted(result, key=lambda row: (row["split"], row["episode_id"], row["chunk_index"]))


def _manifest_lookup(manifest_rows: list[dict[str, str]]) -> dict[str, dict]:
    numeric_fields = {
        "pair_index": int,
        "length": int,
        "motion_start": int,
        "grasp_start": int,
        "pregrasp_frames": int,
        "pregrasp_seconds": float,
        "full_active_frames": int,
        "full_active_seconds": float,
        "motion_start_x": float,
        "motion_start_y": float,
        "motion_start_z": float,
        "grasp_x": float,
        "grasp_y": float,
        "grasp_z": float,
        "pregrasp_path_length_m": float,
        "pregrasp_direct_distance_m": float,
        "pregrasp_path_ratio": float,
        "pregrasp_max_line_deviation_m": float,
    }
    output = {}
    for source in manifest_rows:
        row = dict(source)
        for field, cast in numeric_fields.items():
            row[field] = cast(row[field])
        output[row["episode_id"]] = row
    return output


def _target_lookup(manifest: dict[str, dict]) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    result: dict[tuple[str, int], dict[str, np.ndarray]] = defaultdict(dict)
    for row in manifest.values():
        result[(row["split"], row["pair_index"])][row["target_side"]] = np.asarray(
            [row["grasp_x"], row["grasp_y"], row["grasp_z"]], dtype=np.float64
        )
    for pair, targets in result.items():
        if set(targets) != {"left", "right"}:
            raise ValueError(f"Missing target for pair {pair}: {targets}")
    return result


def _assign_geometry_logits(chunk_rows: list[dict], targets: dict) -> None:
    for row in chunk_rows:
        xyz = np.asarray([row["cartesian_x"], row["cartesian_y"], row["cartesian_z"]])
        pair_targets = targets[(row["split"], row["pair_index"])]
        distance_left = float(np.linalg.norm(xyz - pair_targets["left"]))
        distance_right = float(np.linalg.norm(xyz - pair_targets["right"]))
        # Positive means the current end-effector position supports the left target.
        row["geometry_logit_left"] = distance_right - distance_left


def _calibration_arrays(chunk_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_episode = defaultdict(list)
    for row in chunk_rows:
        if row["split"] == "train" and row["phase"] == "pregrasp":
            by_episode[row["episode_id"]].append(row)
    model_logits, geometry_logits, labels = [], [], []
    for rows in by_episode.values():
        rows.sort(key=lambda row: row["chunk_index"])
        cumulative_left = 0.0
        cumulative_right = 0.0
        for row in rows:
            cumulative_left += row["energy_left"]
            cumulative_right += row["energy_right"]
            model_logits.append(cumulative_right - cumulative_left)
            geometry_logits.append(row["geometry_logit_left"])
            labels.append(1.0 if row["target_side"] == "left" else 0.0)
    return np.asarray(model_logits), np.asarray(geometry_logits), np.asarray(labels)


def _recognition_progress(progress: np.ndarray, belief: np.ndarray, threshold: float) -> float:
    for index in range(len(belief)):
        if belief[index] >= threshold and np.all(belief[index:] >= threshold):
            return float(progress[index])
    return float("nan")


def _auc(progress: np.ndarray, belief: np.ndarray) -> float:
    x = np.concatenate([[0.0], np.asarray(progress, dtype=np.float64)])
    y = np.concatenate([[0.5], np.asarray(belief, dtype=np.float64)])
    return float(np.trapz(y, x))


def _episode_metrics(
    chunk_rows: list[dict],
    manifest: dict[str, dict],
    *,
    model_multiplier: float,
    geometry_multiplier: float,
    recognition_threshold: float,
) -> list[dict]:
    by_episode = defaultdict(list)
    for row in chunk_rows:
        by_episode[row["episode_id"]].append(row)

    output = []
    for episode_id, rows in by_episode.items():
        rows.sort(key=lambda row: row["chunk_index"])
        cumulative_left = 0.0
        cumulative_right = 0.0
        for row in rows:
            cumulative_left += row["energy_left"]
            cumulative_right += row["energy_right"]
            left_belief = float(_sigmoid(np.asarray([(cumulative_right - cumulative_left) * model_multiplier]))[0])
            row["belief_left"] = left_belief
            row["belief_true"] = left_belief if row["target_side"] == "left" else 1.0 - left_belief
            geometry_left = float(_sigmoid(np.asarray([row["geometry_logit_left"] * geometry_multiplier]))[0])
            row["geometry_belief_true"] = geometry_left if row["target_side"] == "left" else 1.0 - geometry_left

        pregrasp = [row for row in rows if row["phase"] == "pregrasp"]
        progress = np.asarray([row["progress_pregrasp"] for row in pregrasp])
        model_belief = np.asarray([row["belief_true"] for row in pregrasp])
        geometry_belief = np.asarray([row["geometry_belief_true"] for row in pregrasp])
        metadata = manifest[episode_id]
        train_energy_values = [row["energy_true"] for row in pregrasp]
        output.append(
            {
                "split": metadata["split"],
                "condition": metadata["condition"],
                "pair_index": metadata["pair_index"],
                "episode_id": episode_id,
                "target_side": metadata["target_side"],
                "pregrasp_chunk_count": len(pregrasp),
                "all_chunk_count": len(rows),
                "pregrasp_local_accuracy": _mean(row["margin_alternative_minus_true"] > 0 for row in pregrasp),
                "pregrasp_final_correct": int(pregrasp[-1]["belief_true"] > 0.5),
                "full_final_correct": int(rows[-1]["belief_true"] > 0.5),
                "model_early_auc": _auc(progress, model_belief),
                "geometry_early_auc": _auc(progress, geometry_belief),
                "model_pregrasp_final_belief": float(model_belief[-1]),
                "geometry_pregrasp_final_belief": float(geometry_belief[-1]),
                "geometry_pregrasp_final_correct": int(geometry_belief[-1] > 0.5),
                "model_full_final_belief": float(rows[-1]["belief_true"]),
                "recognition_progress": _recognition_progress(progress, model_belief, recognition_threshold),
                "geometry_recognition_progress": _recognition_progress(
                    progress, geometry_belief, recognition_threshold
                ),
                "mean_pregrasp_true_energy": _mean(train_energy_values),
                "mean_pregrasp_alternative_energy": _mean(row["energy_alternative"] for row in pregrasp),
                "mean_pregrasp_local_margin": _mean(row["margin_alternative_minus_true"] for row in pregrasp),
                "cumulative_pregrasp_margin": float(np.sum([row["margin_alternative_minus_true"] for row in pregrasp])),
                "mean_margin_seed_sd": _mean(row["margin_seed_sd"] for row in pregrasp),
                "mean_seed_sign_agreement": _mean(row["seed_sign_agreement"] for row in pregrasp),
                "pregrasp_seconds": metadata["pregrasp_seconds"],
                "pregrasp_path_length_m": metadata["pregrasp_path_length_m"],
                "pregrasp_direct_distance_m": metadata["pregrasp_direct_distance_m"],
                "pregrasp_path_ratio": metadata["pregrasp_path_ratio"],
                "pregrasp_max_line_deviation_m": metadata["pregrasp_max_line_deviation_m"],
            }
        )
    return sorted(output, key=lambda row: (row["split"], row["pair_index"], row["episode_id"]))


def _condition_metrics(episode_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in episode_rows:
        if row["split"] == "test":
            grouped[row["condition"]].append(row)
    result = []
    for condition in droid_dataset.FAILURE_CONDITIONS:
        rows = grouped[condition]
        if len(rows) != 2:
            raise ValueError(f"Expected two held-out trajectories for {condition}, got {len(rows)}")
        result.append(
            {
                "condition": condition,
                "n_trajectories": len(rows),
                "model_early_auc_mean": _mean(row["model_early_auc"] for row in rows),
                "model_early_auc_min": min(row["model_early_auc"] for row in rows),
                "model_early_auc_max": max(row["model_early_auc"] for row in rows),
                "model_pregrasp_final_belief_mean": _mean(row["model_pregrasp_final_belief"] for row in rows),
                "model_full_final_belief_mean": _mean(row["model_full_final_belief"] for row in rows),
                "pregrasp_final_accuracy": _mean(row["pregrasp_final_correct"] for row in rows),
                "pregrasp_local_accuracy": _mean(row["pregrasp_local_accuracy"] for row in rows),
                "geometry_early_auc_mean": _mean(row["geometry_early_auc"] for row in rows),
                "geometry_pregrasp_final_accuracy": _mean(row["geometry_pregrasp_final_correct"] for row in rows),
                "mean_true_energy": _mean(row["mean_pregrasp_true_energy"] for row in rows),
                "mean_local_margin": _mean(row["mean_pregrasp_local_margin"] for row in rows),
                "path_ratio_mean": _mean(row["pregrasp_path_ratio"] for row in rows),
                "max_line_deviation_m_mean": _mean(row["pregrasp_max_line_deviation_m"] for row in rows),
                "seed_sign_agreement_mean": _mean(row["mean_seed_sign_agreement"] for row in rows),
                "action_quantile_outside_fraction_mean": _mean(row["action_quantile_outside_fraction"] for row in rows),
                "state_quantile_outside_fraction_mean": _mean(row["state_quantile_outside_fraction"] for row in rows),
                "joint_velocity_l2_mean": _mean(row["joint_velocity_l2_mean"] for row in rows),
            }
        )
    return result


def _split_summary(episode_rows: list[dict]) -> dict:
    output = {}
    for split in ("train", "test"):
        rows = [row for row in episode_rows if row["split"] == split]
        output[split] = {
            "n_trajectories": len(rows),
            "pregrasp_local_accuracy": _mean(row["pregrasp_local_accuracy"] for row in rows),
            "pregrasp_final_accuracy": _mean(row["pregrasp_final_correct"] for row in rows),
            "full_final_accuracy": _mean(row["full_final_correct"] for row in rows),
            "model_early_auc_mean": _mean(row["model_early_auc"] for row in rows),
            "model_pregrasp_final_belief_mean": _mean(row["model_pregrasp_final_belief"] for row in rows),
            "model_full_final_belief_mean": _mean(row["model_full_final_belief"] for row in rows),
            "geometry_early_auc_mean": _mean(row["geometry_early_auc"] for row in rows),
            "geometry_pregrasp_final_accuracy": _mean(row["geometry_pregrasp_final_correct"] for row in rows),
            "mean_pregrasp_true_energy": _mean(row["mean_pregrasp_true_energy"] for row in rows),
            "mean_seed_sign_agreement": _mean(row["mean_seed_sign_agreement"] for row in rows),
            "action_quantile_outside_fraction": _mean(row["action_quantile_outside_fraction"] for row in rows),
            "state_quantile_outside_fraction": _mean(row["state_quantile_outside_fraction"] for row in rows),
            "joint_velocity_l2_mean": _mean(row["joint_velocity_l2_mean"] for row in rows),
        }
    return output


def _assign_train_energy_percentiles(episode_rows: list[dict]) -> None:
    train_energy = np.sort([row["mean_pregrasp_true_energy"] for row in episode_rows if row["split"] == "train"])
    median = float(np.median(train_energy))
    q25, q75 = np.quantile(train_energy, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-12)
    for row in episode_rows:
        value = row["mean_pregrasp_true_energy"]
        row["train_energy_percentile"] = float(100 * np.mean(train_energy <= value))
        row["train_energy_robust_z"] = float((value - median) / scale)


def _assign_input_distribution_diagnostics(
    episode_rows: list[dict],
    manifest: dict[str, dict],
    checkpoint_dir: Path,
) -> None:
    stats = json.loads((checkpoint_dir / "assets" / "droid" / "norm_stats.json").read_text())["norm_stats"]
    action_q01 = np.asarray(stats["actions"]["q01"][:8], dtype=np.float64)
    action_q99 = np.asarray(stats["actions"]["q99"][:8], dtype=np.float64)
    state_q01 = np.asarray(stats["state"]["q01"][:8], dtype=np.float64)
    state_q99 = np.asarray(stats["state"]["q99"][:8], dtype=np.float64)

    for row in episode_rows:
        metadata = manifest[row["episode_id"]]
        start, stop = metadata["motion_start"], metadata["grasp_start"]
        with h5py.File(metadata["path"], "r") as trajectory:
            joint_velocity = np.asarray(trajectory["action/joint_velocity"][start:stop], dtype=np.float64)
            gripper_action = np.asarray(trajectory["action/gripper_position"][start:stop], dtype=np.float64)
            joint_position = np.asarray(
                trajectory["observation/robot_state/joint_positions"][start:stop], dtype=np.float64
            )
            gripper_position = np.asarray(
                trajectory["observation/robot_state/gripper_position"][start:stop], dtype=np.float64
            )
        actions = np.concatenate([joint_velocity, gripper_action[:, None]], axis=1)
        states = np.concatenate([joint_position, gripper_position[:, None]], axis=1)
        normalized_actions = (actions - action_q01) / (action_q99 - action_q01 + 1e-6) * 2 - 1
        normalized_states = (states - state_q01) / (state_q99 - state_q01 + 1e-6) * 2 - 1
        row["action_quantile_outside_fraction"] = float(np.mean(np.abs(normalized_actions) > 1))
        row["state_quantile_outside_fraction"] = float(np.mean(np.abs(normalized_states) > 1))
        row["normalized_action_abs_mean"] = float(np.mean(np.abs(normalized_actions)))
        row["normalized_state_abs_mean"] = float(np.mean(np.abs(normalized_states)))
        velocity_norm = np.linalg.norm(joint_velocity, axis=1)
        row["joint_velocity_l2_mean"] = float(np.mean(velocity_norm))
        row["joint_velocity_l2_p95"] = float(np.quantile(velocity_norm, 0.95))


def _assign_target_distribution_diagnostics(episode_rows: list[dict], manifest: dict[str, dict]) -> dict:
    train_metadata = [row for row in manifest.values() if row["split"] == "train"]
    train_points = np.asarray(
        [[row["grasp_x"], row["grasp_y"], row["grasp_z"]] for row in train_metadata],
        dtype=np.float64,
    )
    lower = np.min(train_points, axis=0)
    upper = np.max(train_points, axis=0)
    train_by_side = {
        side: np.asarray(
            [[row["grasp_x"], row["grasp_y"], row["grasp_z"]] for row in train_metadata if row["target_side"] == side],
            dtype=np.float64,
        )
        for side in ("left", "right")
    }
    test_rows = []
    for row in episode_rows:
        metadata = manifest[row["episode_id"]]
        point = np.asarray([metadata["grasp_x"], metadata["grasp_y"], metadata["grasp_z"]])
        box_excess = np.maximum(lower - point, 0) + np.maximum(point - upper, 0)
        row["train_target_box_excess_m"] = float(np.linalg.norm(box_excess))
        row["nearest_same_side_train_target_m"] = float(
            np.min(np.linalg.norm(train_by_side[row["target_side"]] - point, axis=1))
        )
        if row["split"] == "test":
            test_rows.append(row)
    return {
        "heldout_targets_inside_global_train_box": sum(row["train_target_box_excess_m"] == 0 for row in test_rows),
        "heldout_target_count": len(test_rows),
        "max_target_box_excess_m": max(row["train_target_box_excess_m"] for row in test_rows),
        "nearest_same_side_train_target_m_min": min(row["nearest_same_side_train_target_m"] for row in test_rows),
        "nearest_same_side_train_target_m_max": max(row["nearest_same_side_train_target_m"] for row in test_rows),
    }


def _rank_correlation(x, y) -> float:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    if len(x) < 3:
        return float("nan")

    # Average ranks for ties.
    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        output = np.empty(len(values), dtype=np.float64)
        begin = 0
        while begin < len(values):
            end = begin + 1
            while end < len(values) and values[order[end]] == values[order[begin]]:
                end += 1
            output[order[begin:end]] = (begin + end - 1) / 2
            begin = end
        return output

    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def _plot_belief_curves(chunk_rows: list[dict], output_dir: Path) -> None:
    test = [row for row in chunk_rows if row["split"] == "test" and row["phase"] == "pregrasp"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, condition in zip(axes.flat, droid_dataset.FAILURE_CONDITIONS, strict=True):
        condition_rows = [row for row in test if row["condition"] == condition]
        for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
            rows = sorted(
                [row for row in condition_rows if row["target_side"] == side],
                key=lambda row: row["progress_pregrasp"],
            )
            progress = [0.0, *[row["progress_pregrasp"] for row in rows]]
            belief = [0.5, *[row["belief_true"] for row in rows]]
            geometry = [0.5, *[row["geometry_belief_true"] for row in rows]]
            axis.plot(progress, belief, marker="o", color=color, label=f"VLA {side}")
            axis.plot(progress, geometry, linestyle="--", color=color, alpha=0.55, label=f"geometry {side}")
        axis.axhline(0.5, color="black", linewidth=0.8, alpha=0.4)
        axis.set_title(condition.replace("_", " "))
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Pre-grasp trajectory progress")
    axes[1, 1].set_xlabel("Pre-grasp trajectory progress")
    axes[0, 0].set_ylabel("Belief in true instruction")
    axes[1, 0].set_ylabel("Belief in true instruction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    figure.suptitle("Held-out intent belief over time (solid: VLA, dashed: geometric baseline)")
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    figure.savefig(output_dir / "heldout_belief_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_tradeoff(episode_rows: list[dict], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    train = [row for row in episode_rows if row["split"] == "train"]
    axis.scatter(
        [row["mean_pregrasp_true_energy"] for row in train],
        [row["model_early_auc"] for row in train],
        color="0.65",
        alpha=0.7,
        label="fine-tuning trajectories",
    )
    colors = ("tab:green", "tab:blue", "tab:orange", "tab:red")
    for condition, color in zip(droid_dataset.FAILURE_CONDITIONS, colors, strict=True):
        rows = [row for row in episode_rows if row["condition"] == condition]
        axis.scatter(
            [row["mean_pregrasp_true_energy"] for row in rows],
            [row["model_early_auc"] for row in rows],
            color=color,
            s=70,
            label=condition,
        )
        for row in rows:
            axis.annotate(row["target_side"][0].upper(), (row["mean_pregrasp_true_energy"], row["model_early_auc"]))
    axis.set_xlabel("True-instruction residual energy (lower = more policy-compatible)")
    axis.set_ylabel("VLA Early-AUC (higher = more legible)")
    axis.set_title("Predictability proxy versus model-derived legibility")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "predictability_legibility_tradeoff.png", dpi=180)
    plt.close(figure)


def _plot_condition_summary(condition_rows: list[dict], episode_rows: list[dict], output_dir: Path) -> None:
    labels = [row["condition"].split("_")[0] for row in condition_rows]
    x = np.arange(len(labels))
    train_energy = [row["mean_pregrasp_true_energy"] for row in episode_rows if row["split"] == "train"]
    train_median = float(np.median(train_energy))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(x, [row["model_early_auc_mean"] for row in condition_rows], marker="o", label="VLA Early-AUC")
    axes[0].plot(
        x,
        [row["model_pregrasp_final_belief_mean"] for row in condition_rows],
        marker="s",
        label="VLA final pre-grasp belief",
    )
    axes[0].plot(
        x,
        [row["geometry_early_auc_mean"] for row in condition_rows],
        marker="^",
        linestyle="--",
        label="geometric Early-AUC",
    )
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Intent evidence by designed condition")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    axes[1].plot(x, [row["mean_true_energy"] for row in condition_rows], marker="o", color="tab:red")
    axes[1].axhline(train_median, color="0.4", linestyle="--", label="training median")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("True-instruction residual energy")
    axes[1].set_title("Policy compatibility (lower is better)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "condition_summary.png", dpi=180)
    plt.close(figure)


def _markdown_table(rows: list[dict], fields: list[tuple[str, str]], digits: int = 3) -> str:
    lines = ["| " + " | ".join(label for _, label in fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        values = []
        for field, _ in fields:
            value = row[field]
            if isinstance(value, float):
                values.append("NA" if math.isnan(value) else f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    output_dir: Path,
    episode_rows: list[dict],
    condition_rows: list[dict],
    summary: dict,
) -> None:
    test_rows = [row for row in episode_rows if row["split"] == "test"]
    condition_table = _markdown_table(
        condition_rows,
        [
            ("condition", "Condition"),
            ("model_early_auc_mean", "VLA Early-AUC"),
            ("model_pregrasp_final_belief_mean", "Pre-grasp belief"),
            ("pregrasp_final_accuracy", "Pre-grasp acc."),
            ("mean_true_energy", "True energy"),
            ("geometry_early_auc_mean", "Geometry AUC"),
            ("geometry_pregrasp_final_accuracy", "Geometry acc."),
            ("path_ratio_mean", "Path ratio"),
            ("action_quantile_outside_fraction_mean", "Action outside"),
        ],
    )
    episode_table = _markdown_table(
        test_rows,
        [
            ("condition", "Condition"),
            ("target_side", "Target"),
            ("model_early_auc", "VLA Early-AUC"),
            ("model_pregrasp_final_belief", "Pre-grasp belief"),
            ("model_full_final_belief", "Full belief"),
            ("mean_pregrasp_true_energy", "True energy"),
            ("train_energy_percentile", "Train-energy pct."),
            ("action_quantile_outside_fraction", "Action outside"),
            ("state_quantile_outside_fraction", "State outside"),
            ("nearest_same_side_train_target_m", "Nearest train target (m)"),
            ("mean_seed_sign_agreement", "Seed sign agree."),
        ],
    )
    text = f"""# Spatial legibility evaluation: pi0.5-DROID checkpoint 39999

## Evaluation protocol

- Checkpoint: `{summary["checkpoint_dir"]}`.
- Data: 32 fine-tuning trajectories from `success/2026-08-09` and 8 held-out trajectories from
  `failure/2026-08-09`; all other dates were excluded.
- Candidate set: *pick up the left block* versus *pick up the right block*.
- Evidence uses non-overlapping 16-action chunks. Chunks are split at grasp onset and assigned to their
  end time, so a prefix never receives evidence from unseen future actions.
- Each energy is averaged over {summary["num_samples"]} common-noise flow samples for each of
  {summary["num_seeds"]} independent seeds, with flow time sampled in
  [{summary["tau_min"]:.1f}, {summary["tau_max"]:.1f}].
- Posterior temperature is calibrated descriptively on the 32 fine-tuning trajectories. Raw energy margin
  and rank accuracy do not depend on this calibration.

## Aggregate result

Fine-tuning trajectories: pre-grasp final accuracy **{summary["split_summary"]["train"]["pregrasp_final_accuracy"]:.3f}**,
mean Early-AUC **{summary["split_summary"]["train"]["model_early_auc_mean"]:.3f}**.

Held-out trajectories: pre-grasp final accuracy **{summary["split_summary"]["test"]["pregrasp_final_accuracy"]:.3f}**,
full-trajectory final accuracy **{summary["split_summary"]["test"]["full_final_accuracy"]:.3f}**, and mean
Early-AUC **{summary["split_summary"]["test"]["model_early_auc_mean"]:.3f}**.
The geometric observer reaches pre-grasp final accuracy
**{summary["split_summary"]["test"]["geometry_pregrasp_final_accuracy"]:.3f}** and mean Early-AUC
**{summary["split_summary"]["test"]["geometry_early_auc_mean"]:.3f}** on those same trajectories.

Mean true-instruction residual energy is **{summary["heldout_to_train_true_energy_ratio"]:.1f}x** higher on the
held-out set than on the fine-tuning trajectories. The fraction of raw action values outside the checkpoint's
1st--99th percentile interval is **{summary["split_summary"]["train"]["action_quantile_outside_fraction"]:.3f}**
for fine-tuning trajectories and **{summary["split_summary"]["test"]["action_quantile_outside_fraction"]:.3f}**
for held-out trajectories; the corresponding state fractions are
**{summary["split_summary"]["train"]["state_quantile_outside_fraction"]:.3f}** and
**{summary["split_summary"]["test"]["state_quantile_outside_fraction"]:.3f}**.
Of the 8 held-out grasp locations, **{summary["target_distribution"]["heldout_targets_inside_global_train_box"]}**
fall inside the training set's axis-aligned grasp-position box; the remaining maximum box excess is only
**{summary["target_distribution"]["max_target_box_excess_m"] * 1000:.1f} mm**. Each held-out target is
**{summary["target_distribution"]["nearest_same_side_train_target_m_min"] * 1000:.1f}--{summary["target_distribution"]["nearest_same_side_train_target_m_max"] * 1000:.1f} mm**
from a same-side training target.

Across the 8 held-out trajectories, Spearman correlation between geometric path ratio and VLA Early-AUC is
**{summary["test_spearman_path_ratio_vs_early_auc"]:.3f}**. Correlation between geometric-baseline Early-AUC
and VLA Early-AUC is **{summary["test_spearman_geometry_vs_vla_auc"]:.3f}**.

## Condition-level descriptive results

{condition_table}

Each condition contains only one left/right pair (`n=2`) collected in a different layout and in a fixed time
order. These numbers are pilot estimates, not an inferential test of a curvature-level effect.

## Held-out trajectories

{episode_table}

`Train-energy pct.` is the percentile of true-instruction residual energy relative to the 32 fine-tuning
trajectories; high values indicate weaker policy compatibility. Residual energy is a flow-matching
compatibility proxy, not a normalized likelihood. `Action outside` and `State outside` are the fractions of
pre-grasp values outside the checkpoint's DROID 1st--99th percentile normalization interval.

## Interpretation boundaries

1. A higher relative posterior can occur even when both candidate instructions have high absolute residual
   energy. Interpret Early-AUC together with `True energy` and the train-energy percentile.
2. C0-C3 were manually designed exaggeration levels; measured legibility need not increase monotonically.
3. The 32 fine-tuning trajectories are an in-distribution reference, not a generalization test.
4. Human legibility has not yet been measured. Agreement with people must be evaluated with matched video
   prefixes before claiming that this VLA is a human observer model.

## Artifacts

- `chunk_scores.csv`: raw per-seed residual energies.
- `chunk_scores_aggregated.csv`: seed-averaged energies and evidence.
- `episode_metrics.csv`: trajectory-level outcomes.
- `condition_metrics.csv`: C0-C3 descriptive summaries.
- `heldout_belief_curves.png`: temporal model and geometry beliefs.
- `predictability_legibility_tradeoff.png`: absolute compatibility versus relative legibility.
"""
    (output_dir / "REPORT.md").write_text(text)


def main(args: Args) -> None:
    output_dir = args.input_dir.resolve()
    score_rows = _read_csv(output_dir / "chunk_scores.csv")
    manifest = _manifest_lookup(_read_csv(output_dir / "episode_manifest.csv"))
    config = json.loads((output_dir / "scoring_config.json").read_text())
    chunk_rows = _aggregate_seed_rows(score_rows)
    targets = _target_lookup(manifest)
    _assign_geometry_logits(chunk_rows, targets)
    model_logits, geometry_logits, labels = _calibration_arrays(chunk_rows)
    model_multiplier = _fit_positive_multiplier(model_logits, labels)
    geometry_multiplier = _fit_positive_multiplier(geometry_logits, labels)
    episode_rows = _episode_metrics(
        chunk_rows,
        manifest,
        model_multiplier=model_multiplier,
        geometry_multiplier=geometry_multiplier,
        recognition_threshold=args.recognition_threshold,
    )
    _assign_train_energy_percentiles(episode_rows)
    _assign_input_distribution_diagnostics(
        episode_rows,
        manifest,
        Path(config["checkpoint_dir"]),
    )
    target_distribution = _assign_target_distribution_diagnostics(episode_rows, manifest)
    condition_rows = _condition_metrics(episode_rows)
    split_summary = _split_summary(episode_rows)
    test_rows = [row for row in episode_rows if row["split"] == "test"]
    summary = {
        "checkpoint_dir": config["checkpoint_dir"],
        "num_samples": config["num_samples"],
        "num_seeds": len(config["seeds"]),
        "tau_min": config["tau_min"],
        "tau_max": config["tau_max"],
        "action_dims": config["action_dims"],
        "posterior_temperature": 1.0 / model_multiplier,
        "geometry_logit_multiplier": geometry_multiplier,
        "split_summary": split_summary,
        "heldout_to_train_true_energy_ratio": (
            split_summary["test"]["mean_pregrasp_true_energy"] / split_summary["train"]["mean_pregrasp_true_energy"]
        ),
        "target_distribution": target_distribution,
        "test_spearman_path_ratio_vs_early_auc": _rank_correlation(
            (row["pregrasp_path_ratio"] for row in test_rows),
            (row["model_early_auc"] for row in test_rows),
        ),
        "test_spearman_line_deviation_vs_early_auc": _rank_correlation(
            (row["pregrasp_max_line_deviation_m"] for row in test_rows),
            (row["model_early_auc"] for row in test_rows),
        ),
        "test_spearman_geometry_vs_vla_auc": _rank_correlation(
            (row["geometry_early_auc"] for row in test_rows),
            (row["model_early_auc"] for row in test_rows),
        ),
    }
    _write_csv(output_dir / "chunk_scores_aggregated.csv", chunk_rows)
    _write_csv(output_dir / "episode_metrics.csv", episode_rows)
    _write_csv(output_dir / "condition_metrics.csv", condition_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot_belief_curves(chunk_rows, output_dir)
    _plot_tradeoff(episode_rows, output_dir)
    _plot_condition_summary(condition_rows, episode_rows, output_dir)
    _write_report(output_dir, episode_rows, condition_rows, summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
