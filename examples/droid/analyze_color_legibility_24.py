"""Analyze early color legibility and spatial/color factorial effects for 24 trajectories."""

from __future__ import annotations

from collections import defaultdict
import csv
import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro

PREFIXES = (0.1, 0.2, 0.3, 0.4, 1.0)
METHODS = ("flat", "linear", "exponential")


@dataclasses.dataclass(frozen=True)
class Args:
    input_dir: Path = Path("artifacts/spatial_legibility_color24_20260818_pi05_droid_h16/ten_seed")
    output_dir: Path = Path("artifacts/spatial_legibility_color24_20260818_pi05_droid_h16")
    exponential_alpha: float = 3.0

    def __post_init__(self) -> None:
        if self.exponential_alpha <= 0:
            raise ValueError("exponential_alpha must be positive")


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


def _prediction_color(margin: float, *, tolerance: float = 1e-12) -> str:
    if margin > tolerance:
        return "red"
    if margin < -tolerance:
        return "blue"
    return "-"


def _prediction_side(prediction_color: str, row: dict) -> str:
    if prediction_color == "blue":
        return row["blue_block_side"]
    if prediction_color == "red":
        return row["red_block_side"]
    return "-"


def _is_correct(prediction_color: str, target_color: str) -> bool:
    return prediction_color == target_color


def _weights(method: str, count: int, alpha: float) -> np.ndarray:
    time = np.arange(count, dtype=np.float64) / count
    if method == "flat":
        return np.ones(count, dtype=np.float64)
    if method == "linear":
        return 1.0 - time
    if method == "exponential":
        return np.exp(-alpha * time)
    raise ValueError(method)


def _prefix_value(progress: np.ndarray, cumulative_margin: np.ndarray, prefix: float) -> float:
    x = np.concatenate([[0.0], progress])
    y = np.concatenate([[0.0], cumulative_margin])
    return float(np.interp(prefix, x, y))


def _aggregate_chunks(
    score_rows: list[dict[str, str]],
) -> tuple[list[dict], dict[str, dict[int, list[dict]]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    raw_by_episode_chunk: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for row in score_rows:
        if row["phase"] == "pregrasp":
            grouped[(row["episode_id"], int(row["chunk_index"]))].append(row)

    chunks = []
    for (episode_id, chunk_index), rows in grouped.items():
        rows.sort(key=lambda row: int(row["seed"]))
        margins = np.asarray([float(row["margin_blue_minus_red"]) for row in rows], dtype=np.float64)
        mean_margin = float(np.mean(margins))
        mean_prediction = _prediction_color(mean_margin)
        target_color = rows[0]["target_color"]
        target_support = margins < 0 if target_color == "blue" else margins > 0
        chunk = {
            "episode_id": episode_id,
            "layout_id": rows[0]["layout_id"],
            "condition": rows[0]["condition"],
            "alpha_normalized": float(rows[0]["alpha_normalized"]),
            "alpha_m": float(rows[0]["alpha_m"]),
            "target_color": target_color,
            "target_side": rows[0]["target_side"],
            "blue_block_side": rows[0]["blue_block_side"],
            "red_block_side": rows[0]["red_block_side"],
            "instruction": rows[0]["instruction"],
            "chunk_index": chunk_index,
            "start": int(rows[0]["start"]),
            "stop": int(rows[0]["stop"]),
            "executed_steps": int(rows[0]["executed_steps"]),
            "progress": float(rows[0]["progress_pregrasp"]),
            "num_seeds": len(rows),
            "energy_blue_mean": float(np.mean([float(row["energy_blue"]) for row in rows])),
            "energy_red_mean": float(np.mean([float(row["energy_red"]) for row in rows])),
            "local_margin_blue_minus_red_mean": mean_margin,
            "local_margin_seed_sd": float(np.std(margins, ddof=1)) if len(rows) > 1 else 0.0,
            "red_support_rate": float(np.mean(margins > 0)),
            "mean_sign_seed_agreement": float(
                np.mean([_prediction_color(value) == mean_prediction for value in margins])
            ),
            "true_target_seed_support_rate": float(np.mean(target_support)),
            "cartesian_x": float(rows[0]["cartesian_x"]),
            "cartesian_y": float(rows[0]["cartesian_y"]),
            "cartesian_z": float(rows[0]["cartesian_z"]),
        }
        chunks.append(chunk)
        raw_by_episode_chunk[episode_id][chunk_index] = [
            {"seed": int(row["seed"]), "margin": margin}
            for row, margin in zip(rows, margins, strict=True)
        ]
    chunks.sort(key=lambda row: (row["episode_id"], row["chunk_index"]))
    return chunks, raw_by_episode_chunk


def _episode_analysis(
    chunks: list[dict],
    raw_by_episode_chunk: dict[str, dict[int, list[dict]]],
    alpha: float,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in chunks:
        grouped[row["episode_id"]].append(row)

    episode_rows = []
    curve_rows = []
    curves = {}
    for episode_id, rows in grouped.items():
        rows.sort(key=lambda row: row["chunk_index"])
        target_color = rows[0]["target_color"]
        progress = np.asarray([row["progress"] for row in rows], dtype=np.float64)
        local_margin = np.asarray(
            [row["local_margin_blue_minus_red_mean"] for row in rows], dtype=np.float64
        )
        count = len(rows)
        cumulative_by_method = {}
        score_by_method = {}
        for method in METHODS:
            weights = _weights(method, count, alpha)
            numerator = np.cumsum(weights * local_margin)
            denominator = np.cumsum(weights)
            cumulative_by_method[method] = numerator
            score_by_method[method] = numerator / denominator

        seeds = sorted({item["seed"] for values in raw_by_episode_chunk[episode_id].values() for item in values})
        seed_matrix = np.asarray(
            [
                [
                    next(
                        item["margin"]
                        for item in raw_by_episode_chunk[episode_id][row["chunk_index"]]
                        if item["seed"] == seed
                    )
                    for row in rows
                ]
                for seed in seeds
            ],
            dtype=np.float64,
        )
        episode_row = {
            key: rows[0][key]
            for key in (
                "episode_id",
                "layout_id",
                "condition",
                "alpha_normalized",
                "alpha_m",
                "target_color",
                "target_side",
                "blue_block_side",
                "red_block_side",
                "instruction",
            )
        }
        episode_row.update({"pregrasp_chunks": count, "num_seeds": len(seeds)})

        for prefix in PREFIXES:
            name = round(prefix * 100)
            mean_value = _prefix_value(progress, np.cumsum(local_margin), prefix)
            seed_values = np.asarray(
                [_prefix_value(progress, np.cumsum(seed_margins), prefix) for seed_margins in seed_matrix]
            )
            prediction = _prediction_color(mean_value)
            seed_predictions = [_prediction_color(value) for value in seed_values]
            episode_row[f"margin_{name}pct"] = mean_value
            episode_row[f"prediction_color_{name}pct"] = prediction
            episode_row[f"prediction_side_{name}pct"] = _prediction_side(prediction, rows[0])
            episode_row[f"correct_{name}pct"] = int(_is_correct(prediction, target_color))
            episode_row[f"seed_correct_rate_{name}pct"] = float(
                np.mean([_is_correct(value, target_color) for value in seed_predictions])
            )
            episode_row[f"seed_prediction_agreement_{name}pct"] = float(
                np.mean([value == prediction for value in seed_predictions])
            )

        early_correct = any(episode_row[f"correct_{name}pct"] for name in (20, 30, 40))
        episode_row["early_correct_20_to_40pct"] = int(early_correct)
        episode_row["late_reversal"] = int(early_correct and not episode_row["correct_100pct"])

        for method in METHODS:
            weights = _weights(method, count, alpha)
            mean_value = float(score_by_method[method][-1])
            seed_values = np.average(seed_matrix, axis=1, weights=weights)
            prediction = _prediction_color(mean_value)
            seed_predictions = [_prediction_color(value) for value in seed_values]
            episode_row[f"{method}_final_score"] = mean_value
            episode_row[f"{method}_final_prediction_color"] = prediction
            episode_row[f"{method}_final_prediction_side"] = _prediction_side(prediction, rows[0])
            episode_row[f"{method}_correct"] = int(_is_correct(prediction, target_color))
            episode_row[f"{method}_seed_correct_rate"] = float(
                np.mean([_is_correct(value, target_color) for value in seed_predictions])
            )
            episode_row[f"{method}_seed_prediction_agreement"] = float(
                np.mean([value == prediction for value in seed_predictions])
            )

        episode_row["flat_final_prediction_stable_80pct"] = int(
            episode_row["flat_seed_prediction_agreement"] >= 0.8
        )
        episode_row["mean_chunk_sign_seed_agreement"] = float(
            np.mean([row["mean_sign_seed_agreement"] for row in rows])
        )
        episode_row["minimum_chunk_sign_seed_agreement"] = float(
            np.min([row["mean_sign_seed_agreement"] for row in rows])
        )
        episode_rows.append(episode_row)

        for index, row in enumerate(rows):
            row.update(
                {
                    "flat_cumulative_margin": float(cumulative_by_method["flat"][index]),
                    "flat_cumulative_score": float(score_by_method["flat"][index]),
                    "linear_cumulative_score": float(score_by_method["linear"][index]),
                    "exponential_cumulative_score": float(score_by_method["exponential"][index]),
                }
            )
            curve_rows.append(dict(row))
        curves[episode_id] = {
            **{key: rows[0][key] for key in ("layout_id", "condition", "target_color", "target_side", "blue_block_side")},
            "progress": progress,
            "local_margin": local_margin,
            "flat_cumulative_margin": cumulative_by_method["flat"],
            "linear_score": score_by_method["linear"],
            "exponential_score": score_by_method["exponential"],
        }

    episode_rows.sort(key=lambda row: row["episode_id"])
    curve_rows.sort(key=lambda row: (row["episode_id"], row["chunk_index"]))
    return episode_rows, curve_rows, curves


def _subset(rows: list[dict], field: str, value: str) -> list[dict]:
    return [row for row in rows if row[field] == value]


def _summary_metrics(
    rows: list[dict], prediction_field: str, seed_correct_field: str, score_field: str
) -> dict[str, float]:
    return {
        "accuracy": float(np.mean([_is_correct(row[prediction_field], row["target_color"]) for row in rows])),
        "seed_pooled_accuracy": float(np.mean([row[seed_correct_field] for row in rows])),
        "predicted_blue_rate": float(np.mean([row[prediction_field] == "blue" for row in rows])),
        "predicted_right_rate": float(
            np.mean([_prediction_side(row[prediction_field], row) == "right" for row in rows])
        ),
        "mean_score_blue_minus_red": float(np.mean([row[score_field] for row in rows])),
    }


def _wide_summary(episodes: list[dict], decisions: list[tuple[str, str, str, str]]) -> list[dict]:
    output = []
    for decision, score_field, prediction_field, seed_field in decisions:
        row = {"decision": decision, **_summary_metrics(episodes, prediction_field, seed_field, score_field)}
        for field, levels, prefix in (
            ("target_color", ("blue", "red"), "target"),
            ("target_side", ("left", "right"), "side"),
            ("layout_id", ("L1", "L3"), "layout"),
            ("blue_block_side", ("left", "right"), "blue_side"),
            ("condition", ("C0", "C1", "C2"), "condition"),
        ):
            for level in levels:
                selected = _subset(episodes, field, level)
                metrics = _summary_metrics(selected, prediction_field, seed_field, score_field)
                row[f"{prefix}_{level}_accuracy"] = metrics["accuracy"]
                row[f"{prefix}_{level}_seed_pooled_accuracy"] = metrics["seed_pooled_accuracy"]
                row[f"{prefix}_{level}_predicted_blue_rate"] = metrics["predicted_blue_rate"]
                row[f"{prefix}_{level}_predicted_right_rate"] = metrics["predicted_right_rate"]
        output.append(row)
    return output


def _stratified_summary(episodes: list[dict], decisions: list[tuple[str, str, str, str]]) -> list[dict]:
    groupings = (
        ("overall", ()),
        ("target_color", ("target_color",)),
        ("target_side", ("target_side",)),
        ("layout", ("layout_id",)),
        ("blue_side", ("blue_block_side",)),
        ("condition", ("condition",)),
        ("target_color_x_side", ("target_color", "target_side")),
        ("layout_x_side", ("layout_id", "target_side")),
        ("layout_x_blue_side", ("layout_id", "blue_block_side")),
        ("condition_x_side", ("condition", "target_side")),
        ("condition_x_target_color", ("condition", "target_color")),
    )
    output = []
    for decision, score_field, prediction_field, seed_field in decisions:
        for grouping_name, fields in groupings:
            grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
            for row in episodes:
                grouped[tuple(str(row[field]) for field in fields)].append(row)
            for key, selected in sorted(grouped.items()):
                metrics = _summary_metrics(selected, prediction_field, seed_field, score_field)
                output.append(
                    {
                        "decision": decision,
                        "grouping": grouping_name,
                        "level": "all" if not key else "/".join(key),
                        "num_trajectories": len(selected),
                        **metrics,
                    }
                )
    return output


def _paired_summaries(
    episodes: list[dict], decisions: list[tuple[str, str, str, str]]
) -> tuple[list[dict], list[dict]]:
    pair_rows = []
    aggregate_rows = []
    for decision, score_field, prediction_field, _ in decisions:
        color_swap_rows = []
        grouped_color: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in episodes:
            grouped_color[(row["layout_id"], row["target_side"], row["condition"])].append(row)
        for (layout, target_side, condition), rows in sorted(grouped_color.items()):
            by_color = {row["target_color"]: row for row in rows}
            if set(by_color) != {"blue", "red"}:
                raise ValueError(f"Missing color-swap pair for {(layout, target_side, condition)}")
            blue = by_color["blue"]
            red = by_color["red"]
            result = {
                "decision": decision,
                "pair_type": "same_geometry_side_color_swap",
                "layout_id": layout,
                "condition": condition,
                "fixed_value": target_side,
                "first_episode": blue["episode_id"],
                "second_episode": red["episode_id"],
                "first_prediction": blue[prediction_field],
                "second_prediction": red[prediction_field],
                "first_score": blue[score_field],
                "second_score": red[score_field],
                "both_correct": int(
                    _is_correct(blue[prediction_field], "blue")
                    and _is_correct(red[prediction_field], "red")
                ),
                "desired_relation": int(blue[prediction_field] != red[prediction_field]),
                "oriented_score_difference": red[score_field] - blue[score_field],
            }
            pair_rows.append(result)
            color_swap_rows.append(result)

        side_swap_rows = []
        grouped_side: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in episodes:
            grouped_side[(row["layout_id"], row["target_color"], row["condition"])].append(row)
        for (layout, target_color, condition), rows in sorted(grouped_side.items()):
            by_side = {row["target_side"]: row for row in rows}
            if set(by_side) != {"left", "right"}:
                raise ValueError(f"Missing side-swap pair for {(layout, target_color, condition)}")
            left = by_side["left"]
            right = by_side["right"]
            result = {
                "decision": decision,
                "pair_type": "same_target_color_side_swap",
                "layout_id": layout,
                "condition": condition,
                "fixed_value": target_color,
                "first_episode": left["episode_id"],
                "second_episode": right["episode_id"],
                "first_prediction": left[prediction_field],
                "second_prediction": right[prediction_field],
                "first_score": left[score_field],
                "second_score": right[score_field],
                "both_correct": int(
                    _is_correct(left[prediction_field], target_color)
                    and _is_correct(right[prediction_field], target_color)
                ),
                "desired_relation": int(left[prediction_field] == right[prediction_field]),
                "oriented_score_difference": (
                    right[score_field] - left[score_field]
                    if target_color == "red"
                    else left[score_field] - right[score_field]
                ),
            }
            pair_rows.append(result)
            side_swap_rows.append(result)

        aggregate_rows.extend(
            (
                {
                    "decision": decision,
                    "pair_type": "same_geometry_side_color_swap",
                    "num_pairs": len(color_swap_rows),
                    "both_correct_rate": float(np.mean([row["both_correct"] for row in color_swap_rows])),
                    "desired_prediction_relation_rate": float(
                        np.mean([row["desired_relation"] for row in color_swap_rows])
                    ),
                    "mean_oriented_score_difference": float(
                        np.mean([row["oriented_score_difference"] for row in color_swap_rows])
                    ),
                },
                {
                    "decision": decision,
                    "pair_type": "same_target_color_side_swap",
                    "num_pairs": len(side_swap_rows),
                    "both_correct_rate": float(np.mean([row["both_correct"] for row in side_swap_rows])),
                    "desired_prediction_relation_rate": float(
                        np.mean([row["desired_relation"] for row in side_swap_rows])
                    ),
                    "mean_oriented_score_difference": float(
                        np.mean([row["oriented_score_difference"] for row in side_swap_rows])
                    ),
                },
            )
        )
    return pair_rows, aggregate_rows


def _geometry_summary(manifest: list[dict[str, str]]) -> list[dict]:
    output = []
    for grouping, field, levels in (
        ("layout", "layout_id", ("L1", "L3")),
        ("target_side", "target_side", ("left", "right")),
        ("condition", "condition", ("C0", "C1", "C2")),
    ):
        for level in levels:
            selected = [row for row in manifest if row[field] == level]
            output.append(
                {
                    "grouping": grouping,
                    "level": level,
                    "num_trajectories": len(selected),
                    "grasp_x_mean": float(np.mean([float(row["grasp_x"]) for row in selected])),
                    "grasp_y_mean": float(np.mean([float(row["grasp_y"]) for row in selected])),
                    "grasp_z_mean": float(np.mean([float(row["grasp_z"]) for row in selected])),
                    "pregrasp_frames_mean": float(
                        np.mean([float(row["pregrasp_frames"]) for row in selected])
                    ),
                    "path_length_mean_m": float(
                        np.mean([float(row["pregrasp_path_length_m"]) for row in selected])
                    ),
                    "path_ratio_mean": float(np.mean([float(row["pregrasp_path_ratio"]) for row in selected])),
                    "max_line_deviation_mean_m": float(
                        np.mean([float(row["pregrasp_max_line_deviation_m"]) for row in selected])
                    ),
                }
            )
    return output


def _plot_curves(curves: dict[str, dict], output_dir: Path) -> None:
    plot_dir = output_dir / "trajectory_curves"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for episode_id, curve in curves.items():
        progress = curve["progress"] * 100
        figure, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        width = max(2.0, 70.0 / len(progress))
        axes[0].bar(progress, curve["local_margin"], width=width, alpha=0.8)
        axes[0].set_ylabel("local E_blue-E_red")
        axes[0].set_title(
            f"{episode_id}\n{curve['layout_id']} {curve['condition']}, blue-{curve['blue_block_side']}, "
            f"GT {curve['target_color']} / {curve['target_side']}"
        )
        axes[1].plot(progress, curve["flat_cumulative_margin"], marker="o", color="tab:blue")
        axes[1].set_ylabel("flat cumulative")
        axes[2].plot(progress, curve["linear_score"], marker="o", color="tab:orange")
        axes[2].set_ylabel("linear score")
        axes[3].plot(progress, curve["exponential_score"], marker="o", color="tab:green")
        axes[3].set_ylabel("exp score")
        axes[3].set_xlabel("pre-grasp trajectory progress (%)")
        for axis in axes:
            axis.axhline(0, color="black", linewidth=0.8)
            axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(plot_dir / f"{episode_id}.png", dpi=160)
        plt.close(figure)


def _plot_aggregate(method_summary: list[dict], prefix_summary: list[dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    prefix_x = np.asarray([10, 20, 30, 40, 100])
    for field, label in (
        ("accuracy", "overall"),
        ("target_blue_accuracy", "target blue"),
        ("target_red_accuracy", "target red"),
        ("side_left_accuracy", "target left"),
        ("side_right_accuracy", "target right"),
    ):
        axes[0].plot(prefix_x, [row[field] for row in prefix_summary], marker="o", label=label)
    axes[0].set_title("Prefix accuracy")
    axes[0].set_xlabel("pre-grasp prefix (%)")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=8)

    x = np.arange(len(METHODS))
    width = 0.25
    for index, (field, label) in enumerate(
        (("accuracy", "overall"), ("target_blue_accuracy", "target blue"), ("target_red_accuracy", "target red"))
    ):
        axes[1].bar(x + (index - 1) * width, [row[field] for row in method_summary], width, label=label)
    axes[1].set_xticks(x, METHODS)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Final method accuracy")
    axes[1].legend(fontsize=8)

    axes[2].bar(x - 0.18, [row["predicted_blue_rate"] for row in method_summary], 0.36, label="blue")
    axes[2].bar(x + 0.18, [row["predicted_right_rate"] for row in method_summary], 0.36, label="base-right")
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1, label="balanced target rate")
    axes[2].set_xticks(x, METHODS)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Prediction bias")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "aggregate_accuracy_and_bias.png", dpi=170)
    plt.close(figure)


def main(args: Args) -> None:
    args = dataclasses.replace(
        args,
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    scoring_config = json.loads((args.input_dir / "scoring_config.json").read_text())
    score_rows = _read_csv(args.input_dir / "chunk_scores.csv")
    manifest = _read_csv(args.input_dir / "episode_manifest.csv")
    chunks, raw_by_episode_chunk = _aggregate_chunks(score_rows)
    episodes, curve_rows, curves = _episode_analysis(chunks, raw_by_episode_chunk, args.exponential_alpha)

    method_decisions = [
        (
            method,
            f"{method}_final_score",
            f"{method}_final_prediction_color",
            f"{method}_seed_correct_rate",
        )
        for method in METHODS
    ]
    prefix_decisions = [
        (
            f"{round(prefix * 100)}pct",
            f"margin_{round(prefix * 100)}pct",
            f"prediction_color_{round(prefix * 100)}pct",
            f"seed_correct_rate_{round(prefix * 100)}pct",
        )
        for prefix in PREFIXES
    ]
    all_decisions = prefix_decisions + method_decisions
    method_summary = _wide_summary(episodes, method_decisions)
    prefix_summary = _wide_summary(episodes, prefix_decisions)
    stratified_summary = _stratified_summary(episodes, all_decisions)
    pair_rows, pair_summary = _paired_summaries(episodes, all_decisions)
    geometry_summary = _geometry_summary(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "color24_chunk_curves.csv", curve_rows)
    _write_csv(args.output_dir / "color24_episode_summary.csv", episodes)
    _write_csv(args.output_dir / "color24_method_summary.csv", method_summary)
    _write_csv(args.output_dir / "color24_prefix_summary.csv", prefix_summary)
    _write_csv(args.output_dir / "color24_stratified_summary.csv", stratified_summary)
    _write_csv(args.output_dir / "color24_matched_pairs.csv", pair_rows)
    _write_csv(args.output_dir / "color24_matched_pair_summary.csv", pair_summary)
    _write_csv(args.output_dir / "color24_geometry_summary.csv", geometry_summary)
    _plot_curves(curves, args.output_dir)
    _plot_aggregate(method_summary, prefix_summary, args.output_dir)

    first_chunks = [row for row in curve_rows if row["chunk_index"] == 0]
    final_chunks = [
        max((row for row in curve_rows if row["episode_id"] == episode_id), key=lambda row: row["chunk_index"])
        for episode_id in sorted({row["episode_id"] for row in curve_rows})
    ]
    diagnostics = {
        "checkpoint_dir": scoring_config["checkpoint_dir"],
        "num_trajectories": len(episodes),
        "num_seeds": len(scoring_config["seeds"]),
        "seeds": scoring_config["seeds"],
        "num_samples_per_seed": scoring_config["num_samples"],
        "exponential_alpha": args.exponential_alpha,
        "margin_definition": "energy_blue - energy_red; positive supports red",
        "analysis_phase": "pregrasp only",
        "prefix_interpolation": "piecewise-linear cumulative margin between chunk-stop progress points",
        "early_correct_definition": "correct at any of 20%, 30%, or 40%",
        "late_reversal_definition": "early_correct and incorrect at 100%",
        "early_correct_count": int(sum(row["early_correct_20_to_40pct"] for row in episodes)),
        "late_reversal_count": int(sum(row["late_reversal"] for row in episodes)),
        "flat_stable_prediction_count_80pct": int(
            sum(row["flat_final_prediction_stable_80pct"] for row in episodes)
        ),
        "first_chunk_mean_margin_blue_minus_red": float(
            np.mean([row["local_margin_blue_minus_red_mean"] for row in first_chunks])
        ),
        "first_chunk_predicted_blue_count": int(
            sum(row["local_margin_blue_minus_red_mean"] < 0 for row in first_chunks)
        ),
        "first_chunk_correct_count": int(
            sum(
                _is_correct(
                    _prediction_color(row["local_margin_blue_minus_red_mean"]), row["target_color"]
                )
                for row in first_chunks
            )
        ),
        "final_chunk_predicted_blue_count": int(
            sum(row["local_margin_blue_minus_red_mean"] < 0 for row in final_chunks)
        ),
        "final_chunk_correct_count": int(
            sum(
                _is_correct(
                    _prediction_color(row["local_margin_blue_minus_red_mean"]), row["target_color"]
                )
                for row in final_chunks
            )
        ),
        "balanced_design": {
            "target_blue": sum(row["target_color"] == "blue" for row in episodes),
            "target_red": sum(row["target_color"] == "red" for row in episodes),
            "target_left": sum(row["target_side"] == "left" for row in episodes),
            "target_right": sum(row["target_side"] == "right" for row in episodes),
            "blue_side_left": sum(row["blue_block_side"] == "left" for row in episodes),
            "blue_side_right": sum(row["blue_block_side"] == "right" for row in episodes),
            "layout_L1": sum(row["layout_id"] == "L1" for row in episodes),
            "layout_L3": sum(row["layout_id"] == "L3" for row in episodes),
        },
    }
    (args.output_dir / "color24_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps({"methods": method_summary, "prefixes": prefix_summary, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
