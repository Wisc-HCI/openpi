"""Analyze the exterior-camera flip intervention on the balanced 24 trajectories."""

from __future__ import annotations

from collections import defaultdict
import csv
import dataclasses
import json
from pathlib import Path

import analyze_color_legibility_24 as color_analysis
import analyze_early_weighted_legibility as side_analysis
import evaluate_color_legibility_24 as color_evaluation
import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi.instruction_likelihood import droid_dataset

PREFIXES = (10, 20, 30, 40, 100)
METHODS = ("flat", "linear", "exponential")
DECISIONS = tuple(f"{value}pct" for value in PREFIXES) + METHODS
FAMILIES = ("color", "side")
VISUAL_CONDITIONS = ("original", "exterior_hflip")


@dataclasses.dataclass(frozen=True)
class Args:
    root_dir: Path = Path("artifacts/spatial_legibility_color24_20260818_pi05_droid_h16")
    exponential_alpha: float = 3.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _key(row: dict[str, str]) -> tuple[str, int, str, int]:
    return row["episode_id"], int(row["chunk_index"]), row["phase"], int(row["seed"])


def _index_unique(rows: list[dict[str, str]], name: str) -> dict[tuple[str, int, str, int], dict[str, str]]:
    output = {}
    for row in rows:
        key = _key(row)
        if key in output:
            raise ValueError(f"Duplicate {name} row: {key}")
        output[key] = row
    return output


def _prepare_condition_inputs(root: Path) -> tuple[dict[str, dict], list[dict[str, str]]]:
    """Validate exact row pairing and translate flipped scores into existing analyzer schemas."""
    original_paths = {
        "color": root / "ten_seed" / "chunk_scores.csv",
        "side": root / "side_prompts" / "ten_seed" / "chunk_scores.csv",
    }
    original = {family: _read_csv(path) for family, path in original_paths.items()}
    flip_rows = _read_csv(root / "exterior_hflip" / "ten_seed" / "chunk_scores.csv")
    flip_by_family = {
        family: [row for row in flip_rows if row["prompt_family"] == family] for family in FAMILIES
    }
    pairing = {}
    normalized = {}
    for family in FAMILIES:
        original_index = _index_unique(original[family], f"original/{family}")
        flip_index = _index_unique(flip_by_family[family], f"exterior_hflip/{family}")
        if original_index.keys() != flip_index.keys():
            missing = sorted(original_index.keys() - flip_index.keys())[:5]
            extra = sorted(flip_index.keys() - original_index.keys())[:5]
            raise ValueError(f"Pairing keys differ for {family}: missing={missing}, extra={extra}")
        translated = []
        for key in sorted(original_index):
            source = original_index[key]
            flipped = flip_index[key]
            for source_field, flip_field in (
                ("layout_id", "layout_id"),
                ("condition", "condition"),
                ("target_color", "target_color"),
                ("target_side", "target_side_base"),
                ("blue_block_side", "blue_block_side_base"),
                ("red_block_side", "red_block_side_base"),
                ("start", "start"),
                ("stop", "stop"),
                ("executed_steps", "executed_steps"),
                ("progress_pregrasp", "progress_pregrasp"),
            ):
                if source[source_field] != flipped[flip_field]:
                    raise ValueError(
                        f"Metadata mismatch for {family}/{key}: "
                        f"{source_field}={source[source_field]} vs {flip_field}={flipped[flip_field]}"
                    )
            row = dict(source)
            first = float(flipped["energy_first"])
            second = float(flipped["energy_second"])
            if family == "color":
                row["energy_blue"] = first
                row["energy_red"] = second
                row["margin_blue_minus_red"] = first - second
                if row["target_color"] == "blue":
                    energy_true, energy_alternative = first, second
                else:
                    energy_true, energy_alternative = second, first
            else:
                row["energy_left"] = first
                row["energy_right"] = second
                if row["target_side"] == "left":
                    energy_true, energy_alternative = first, second
                else:
                    energy_true, energy_alternative = second, first
            row["energy_true"] = energy_true
            row["energy_alternative"] = energy_alternative
            row["margin_alternative_minus_true"] = energy_alternative - energy_true
            translated.append(row)
        normalized[family] = translated
        pairing[family] = {
            "original_rows": len(original_index),
            "flipped_rows": len(flip_index),
            "paired_keys_identical": True,
            "pregrasp_rows": sum(row["phase"] == "pregrasp" for row in translated),
            "episodes": len({row["episode_id"] for row in translated}),
            "seeds": sorted({int(row["seed"]) for row in translated}),
        }

    color_input = root / "exterior_hflip" / "color_ten_seed"
    side_input = root / "exterior_hflip" / "side_ten_seed"
    color_input.mkdir(parents=True, exist_ok=True)
    side_input.mkdir(parents=True, exist_ok=True)
    _write_csv(color_input / "chunk_scores.csv", normalized["color"])
    _write_csv(side_input / "chunk_scores.csv", normalized["side"])
    (color_input / "episode_manifest.csv").write_text(
        (root / "ten_seed" / "episode_manifest.csv").read_text()
    )
    for family, input_dir, source_config in (
        ("color", color_input, root / "ten_seed" / "scoring_config.json"),
        ("side", side_input, root / "side_prompts" / "ten_seed" / "scoring_config.json"),
    ):
        config = json.loads(source_config.read_text())
        config.update(
            {
                "visual_condition": "exterior_hflip",
                "normalized_from": str(root / "exterior_hflip" / "ten_seed" / "chunk_scores.csv"),
                "prompt_family": family,
            }
        )
        (input_dir / "scoring_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return pairing, flip_rows


def _run_condition_analyzers(root: Path, alpha: float) -> None:
    color_analysis.main(
        color_analysis.Args(
            input_dir=root / "exterior_hflip" / "color_ten_seed",
            output_dir=root / "exterior_hflip" / "color_analysis",
            exponential_alpha=alpha,
        )
    )
    side_analysis.main(
        side_analysis.Args(
            input_dir=root / "exterior_hflip" / "side_ten_seed",
            output_dir=root / "exterior_hflip" / "side_analysis",
            exponential_alpha=alpha,
        )
    )


def _prediction_fields(family: str, decision: str) -> tuple[str, str, str]:
    if decision.endswith("pct"):
        value = decision.removesuffix("pct")
        if family == "color":
            return f"margin_{value}pct", f"prediction_color_{value}pct", f"seed_correct_rate_{value}pct"
        return f"margin_{value}pct", f"prediction_{value}pct", f"seed_correct_rate_{value}pct"
    if family == "color":
        return (
            f"{decision}_final_score",
            f"{decision}_final_prediction_color",
            f"{decision}_seed_correct_rate",
        )
    return f"{decision}_final_score", f"{decision}_final_prediction", f"{decision}_seed_correct_rate"


def _normalize_prediction(family: str, prediction: str) -> str:
    if family == "color":
        return prediction
    return {"L": "left", "R": "right", "-": "-"}[prediction]


def _episode_comparison(root: Path) -> list[dict]:
    metadata_rows = _read_csv(root / "ten_seed" / "episode_manifest.csv")
    metadata = {row["episode_id"]: row for row in metadata_rows}
    summary_paths = {
        ("color", "original"): root / "color24_episode_summary.csv",
        ("color", "exterior_hflip"): root / "exterior_hflip" / "color_analysis" / "color24_episode_summary.csv",
        ("side", "original"): root / "side_prompts" / "early_weighted_episode_summary.csv",
        ("side", "exterior_hflip"): (
            root / "exterior_hflip" / "side_analysis" / "early_weighted_episode_summary.csv"
        ),
    }
    summaries = {
        key: {row["episode_id"]: row for row in _read_csv(path)} for key, path in summary_paths.items()
    }
    episode_ids = sorted(metadata)
    for rows in summaries.values():
        if set(rows) != set(episode_ids):
            raise ValueError("Episode summary sets do not match the manifest")

    output = []
    for family in FAMILIES:
        target_field = "target_color" if family == "color" else "target_side"
        for episode_id in episode_ids:
            meta = metadata[episode_id]
            original = summaries[(family, "original")][episode_id]
            flipped = summaries[(family, "exterior_hflip")][episode_id]
            target = meta[target_field]
            for decision in DECISIONS:
                score_field, prediction_field, seed_field = _prediction_fields(family, decision)
                original_prediction = _normalize_prediction(family, original[prediction_field])
                flipped_prediction = _normalize_prediction(family, flipped[prediction_field])
                original_correct = original_prediction == target
                flipped_correct = flipped_prediction == target
                output.append(
                    {
                        "episode_id": episode_id,
                        "layout_id": meta["layout_id"],
                        "trajectory_condition": meta["condition"],
                        "target_color": meta["target_color"],
                        "target_side_base": meta["target_side"],
                        "blue_block_side_base": meta["blue_block_side"],
                        "red_block_side_base": meta["red_block_side"],
                        "prompt_family": family,
                        "decision": decision,
                        "target_label": target,
                        "original_score": float(original[score_field]),
                        "flipped_score": float(flipped[score_field]),
                        "score_delta_flip_minus_original": float(flipped[score_field])
                        - float(original[score_field]),
                        "original_prediction": original_prediction,
                        "flipped_prediction": flipped_prediction,
                        "prediction_changed": int(original_prediction != flipped_prediction),
                        "original_correct": int(original_correct),
                        "flipped_correct": int(flipped_correct),
                        "accuracy_change": int(flipped_correct) - int(original_correct),
                        "original_seed_correct_rate": float(original[seed_field]),
                        "flipped_seed_correct_rate": float(flipped[seed_field]),
                        "seed_accuracy_change": float(flipped[seed_field]) - float(original[seed_field]),
                    }
                )
    return output


def _summary_metrics(rows: list[dict]) -> dict[str, float | int]:
    family = rows[0]["prompt_family"]
    first_label = "blue" if family == "color" else "left"
    return {
        "num_trajectories": len(rows),
        "original_accuracy": float(np.mean([row["original_correct"] for row in rows])),
        "flipped_accuracy": float(np.mean([row["flipped_correct"] for row in rows])),
        "accuracy_change": float(np.mean([row["accuracy_change"] for row in rows])),
        "original_seed_pooled_accuracy": float(
            np.mean([row["original_seed_correct_rate"] for row in rows])
        ),
        "flipped_seed_pooled_accuracy": float(
            np.mean([row["flipped_seed_correct_rate"] for row in rows])
        ),
        "seed_pooled_accuracy_change": float(np.mean([row["seed_accuracy_change"] for row in rows])),
        "original_first_label_rate": float(
            np.mean([row["original_prediction"] == first_label for row in rows])
        ),
        "flipped_first_label_rate": float(
            np.mean([row["flipped_prediction"] == first_label for row in rows])
        ),
        "prediction_changed_count": int(sum(row["prediction_changed"] for row in rows)),
        "improved_count": int(sum(row["accuracy_change"] > 0 for row in rows)),
        "worsened_count": int(sum(row["accuracy_change"] < 0 for row in rows)),
        "mean_score_delta_flip_minus_original": float(
            np.mean([row["score_delta_flip_minus_original"] for row in rows])
        ),
    }


def _aggregate_comparison(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    summary = []
    for family in FAMILIES:
        for decision in DECISIONS:
            selected = [
                row for row in rows if row["prompt_family"] == family and row["decision"] == decision
            ]
            summary.append({"prompt_family": family, "decision": decision, **_summary_metrics(selected)})

    groupings = (
        ("overall", ()),
        ("layout", ("layout_id",)),
        ("target_color", ("target_color",)),
        ("target_side", ("target_side_base",)),
        ("blue_side", ("blue_block_side_base",)),
        ("trajectory_condition", ("trajectory_condition",)),
        ("layout_x_target_side", ("layout_id", "target_side_base")),
        ("condition_x_target_side", ("trajectory_condition", "target_side_base")),
        ("target_color_x_target_side", ("target_color", "target_side_base")),
    )
    stratified = []
    for family in FAMILIES:
        for decision in DECISIONS:
            decision_rows = [
                row for row in rows if row["prompt_family"] == family and row["decision"] == decision
            ]
            for grouping, fields in groupings:
                grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
                for row in decision_rows:
                    grouped[tuple(row[field] for field in fields)].append(row)
                for key, selected in sorted(grouped.items()):
                    stratified.append(
                        {
                            "prompt_family": family,
                            "decision": decision,
                            "grouping": grouping,
                            "level": "all" if not key else "/".join(key),
                            **_summary_metrics(selected),
                        }
                    )
    return summary, stratified


def _raw_paired_effects(root: Path, flip_rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    original = {
        "color": _read_csv(root / "ten_seed" / "chunk_scores.csv"),
        "side": _read_csv(root / "side_prompts" / "ten_seed" / "chunk_scores.csv"),
    }
    flip_by_family = {
        family: _index_unique(
            [row for row in flip_rows if row["prompt_family"] == family], f"flip/{family}"
        )
        for family in FAMILIES
    }
    episode_effects = []
    chunk_effects = []
    for family in FAMILIES:
        original_index = _index_unique(original[family], f"original/{family}")
        grouped_episode: dict[str, list[tuple[float, float]]] = defaultdict(list)
        grouped_chunk: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
        for key, row in original_index.items():
            if row["phase"] != "pregrasp":
                continue
            original_margin = (
                float(row["margin_blue_minus_red"])
                if family == "color"
                else float(row["energy_left"]) - float(row["energy_right"])
            )
            flipped_margin = float(flip_by_family[family][key]["margin_first_minus_second"])
            grouped_episode[row["episode_id"]].append((original_margin, flipped_margin))
            grouped_chunk[(row["episode_id"], int(row["chunk_index"]))].append(
                (original_margin, flipped_margin)
            )

        for episode_id, values in sorted(grouped_episode.items()):
            array = np.asarray(values, dtype=np.float64)
            original_values, flipped_values = array[:, 0], array[:, 1]
            episode_effects.append(
                {
                    "prompt_family": family,
                    "episode_id": episode_id,
                    "num_paired_chunk_seeds": len(values),
                    "mean_original_margin": float(np.mean(original_values)),
                    "mean_flipped_margin": float(np.mean(flipped_values)),
                    "mean_margin_delta_flip_minus_original": float(
                        np.mean(flipped_values - original_values)
                    ),
                    "mean_absolute_margin_delta": float(np.mean(np.abs(flipped_values - original_values))),
                    "same_sign_rate": float(np.mean(np.sign(flipped_values) == np.sign(original_values))),
                    "opposite_sign_rate": float(
                        np.mean(np.sign(flipped_values) == -np.sign(original_values))
                    ),
                    "correlation_with_original": float(np.corrcoef(original_values, flipped_values)[0, 1]),
                }
            )
        for (episode_id, chunk_index), values in sorted(grouped_chunk.items()):
            array = np.asarray(values, dtype=np.float64)
            original_values, flipped_values = array[:, 0], array[:, 1]
            chunk_effects.append(
                {
                    "prompt_family": family,
                    "episode_id": episode_id,
                    "chunk_index": chunk_index,
                    "num_seeds": len(values),
                    "original_margin_mean": float(np.mean(original_values)),
                    "flipped_margin_mean": float(np.mean(flipped_values)),
                    "margin_delta_mean": float(np.mean(flipped_values - original_values)),
                    "margin_delta_sd": float(np.std(flipped_values - original_values, ddof=1)),
                    "same_sign_rate": float(np.mean(np.sign(flipped_values) == np.sign(original_values))),
                }
            )
    return episode_effects, chunk_effects


def _paired_effect_aggregate(episode_effects: list[dict], root: Path) -> list[dict]:
    metadata = {
        row["episode_id"]: row for row in _read_csv(root / "ten_seed" / "episode_manifest.csv")
    }
    output = []
    groupings = (
        ("overall", None),
        ("layout", "layout_id"),
        ("target_color", "target_color"),
        ("target_side", "target_side"),
        ("condition", "condition"),
    )
    for family in FAMILIES:
        family_rows = [row for row in episode_effects if row["prompt_family"] == family]
        for grouping, field in groupings:
            grouped: dict[str, list[dict]] = defaultdict(list)
            for row in family_rows:
                level = "all" if field is None else metadata[row["episode_id"]][field]
                grouped[level].append(row)
            for level, selected in sorted(grouped.items()):
                weights = np.asarray([row["num_paired_chunk_seeds"] for row in selected], dtype=np.float64)
                output.append(
                    {
                        "prompt_family": family,
                        "grouping": grouping,
                        "level": level,
                        "num_trajectories": len(selected),
                        "num_paired_chunk_seeds": int(np.sum(weights)),
                        "mean_margin_delta_flip_minus_original": float(
                            np.average(
                                [row["mean_margin_delta_flip_minus_original"] for row in selected],
                                weights=weights,
                            )
                        ),
                        "mean_absolute_margin_delta": float(
                            np.average(
                                [row["mean_absolute_margin_delta"] for row in selected], weights=weights
                            )
                        ),
                        "same_sign_rate": float(
                            np.average([row["same_sign_rate"] for row in selected], weights=weights)
                        ),
                        "opposite_sign_rate": float(
                            np.average([row["opposite_sign_rate"] for row in selected], weights=weights)
                        ),
                        "mean_episode_correlation_with_original": float(
                            np.mean([row["correlation_with_original"] for row in selected])
                        ),
                    }
                )
    return output


def _plot_aggregate(summary: list[dict], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row_index, family in enumerate(FAMILIES):
        family_rows = {row["decision"]: row for row in summary if row["prompt_family"] == family}
        prefix_x = np.asarray(PREFIXES)
        axes[row_index, 0].plot(
            prefix_x,
            [family_rows[f"{value}pct"]["original_accuracy"] for value in PREFIXES],
            marker="o",
            label="original",
        )
        axes[row_index, 0].plot(
            prefix_x,
            [family_rows[f"{value}pct"]["flipped_accuracy"] for value in PREFIXES],
            marker="o",
            label="exterior hflip",
        )
        axes[row_index, 0].set_title(f"{family}: prefix accuracy")
        axes[row_index, 0].set_xlabel("pre-grasp prefix (%)")
        axes[row_index, 0].set_ylim(0, 1)

        x = np.arange(len(METHODS))
        axes[row_index, 1].bar(
            x - 0.18,
            [family_rows[method]["original_accuracy"] for method in METHODS],
            0.36,
            label="original",
        )
        axes[row_index, 1].bar(
            x + 0.18,
            [family_rows[method]["flipped_accuracy"] for method in METHODS],
            0.36,
            label="exterior hflip",
        )
        axes[row_index, 1].set_xticks(x, METHODS)
        axes[row_index, 1].set_ylim(0, 1)
        axes[row_index, 1].set_title(f"{family}: final accuracy")

        axes[row_index, 2].bar(
            x - 0.18,
            [family_rows[method]["original_first_label_rate"] for method in METHODS],
            0.36,
            label="original",
        )
        axes[row_index, 2].bar(
            x + 0.18,
            [family_rows[method]["flipped_first_label_rate"] for method in METHODS],
            0.36,
            label="exterior hflip",
        )
        axes[row_index, 2].axhline(0.5, color="black", linestyle="--", linewidth=1)
        axes[row_index, 2].set_xticks(x, METHODS)
        axes[row_index, 2].set_ylim(0, 1)
        first_label = "Blue" if family == "color" else "Base-Left"
        axes[row_index, 2].set_title(f"{family}: predicted {first_label} rate")
        for axis in axes[row_index]:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _plot_visual_audit(root: Path, output_path: Path) -> None:
    """Show the exact exterior-image intervention for all layout/side combinations."""
    config = json.loads((root / "ten_seed" / "scoring_config.json").read_text())
    episodes = color_evaluation.discover_color_episodes(
        Path(config["data_dir"]), str(config["start_episode"])
    )
    selected = sorted(
        [
            episode
            for episode in episodes
            if episode.target_color == "blue" and episode.spec.condition == "C0"
        ],
        key=lambda episode: (episode.layout_id, episode.blue_block_side),
    )
    if len(selected) != 4:
        raise ValueError(f"Expected four layout/blue-side audit episodes, got {len(selected)}")
    figure, axes = plt.subplots(len(selected), 2, figsize=(12, 10))
    for row_index, color_episode in enumerate(selected):
        episode = color_episode.spec
        arrays = droid_dataset.load_episode_arrays(episode)
        first_pregrasp = next(
            chunk
            for chunk in droid_dataset.iter_action_chunks(
                episode,
                action_horizon=16,
                arrays=arrays,
            )
            if chunk.phase == "pregrasp"
        )
        original = first_pregrasp.exterior_image
        flipped = np.ascontiguousarray(original[:, ::-1])
        axes[row_index, 0].imshow(original)
        axes[row_index, 1].imshow(flipped)
        label = f"{color_episode.layout_id}, blue base-{color_episode.blue_block_side}"
        axes[row_index, 0].set_ylabel(label)
        for column in range(2):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
    axes[0, 0].set_title("original exterior")
    axes[0, 1].set_title("exterior horizontal flip")
    figure.suptitle("Exact visual intervention (first pre-grasp frame, Blue-target C0)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _plot_paired_trajectory_curves(root: Path, output_dir: Path) -> None:
    paths = {
        ("color", "original"): root / "color24_chunk_curves.csv",
        ("color", "exterior_hflip"): (
            root / "exterior_hflip" / "color_analysis" / "color24_chunk_curves.csv"
        ),
        ("side", "original"): root / "side_prompts" / "early_weighted_chunk_scores.csv",
        ("side", "exterior_hflip"): (
            root / "exterior_hflip" / "side_analysis" / "early_weighted_chunk_scores.csv"
        ),
    }
    rows = {key: _read_csv(path) for key, path in paths.items()}
    episode_ids = sorted({row["episode_id"] for row in rows[("color", "original")]})
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = {
        "color": (
            "local_margin_blue_minus_red_mean",
            "flat_cumulative_margin",
            "linear_cumulative_score",
            "exponential_cumulative_score",
        ),
        "side": (
            "local_margin_left_minus_right_mean",
            "flat_cumulative_margin",
            "linear_cumulative_score",
            "exponential_cumulative_score",
        ),
    }
    ylabels = ("local margin", "flat cumulative", "linear score", "exp score")
    for episode_id in episode_ids:
        figure, axes = plt.subplots(4, 2, figsize=(14, 12), sharex="col")
        for column, family in enumerate(FAMILIES):
            for visual_condition, color in (("original", "tab:blue"), ("exterior_hflip", "tab:orange")):
                selected = sorted(
                    [
                        row
                        for row in rows[(family, visual_condition)]
                        if row["episode_id"] == episode_id
                    ],
                    key=lambda row: int(row["chunk_index"]),
                )
                progress = np.asarray([float(row["progress"]) for row in selected]) * 100
                for row_index, field in enumerate(fields[family]):
                    axes[row_index, column].plot(
                        progress,
                        [float(row[field]) for row in selected],
                        marker="o",
                        color=color,
                        label=visual_condition,
                    )
            axes[0, column].set_title(f"{family} prompts")
            axes[-1, column].set_xlabel("pre-grasp progress (%)")
            for row_index, ylabel in enumerate(ylabels):
                axes[row_index, column].set_ylabel(ylabel)
                axes[row_index, column].axhline(0, color="black", linewidth=0.8)
                axes[row_index, column].grid(alpha=0.2)
                axes[row_index, column].legend(fontsize=8)
        figure.suptitle(episode_id)
        figure.tight_layout()
        figure.savefig(output_dir / f"{episode_id}.png", dpi=155)
        plt.close(figure)


def main(args: Args) -> None:
    root = args.root_dir.resolve()
    output_dir = root / "exterior_hflip"
    pairing, flip_rows = _prepare_condition_inputs(root)
    _run_condition_analyzers(root, args.exponential_alpha)

    episode_comparison = _episode_comparison(root)
    summary, stratified = _aggregate_comparison(episode_comparison)
    episode_effects, chunk_effects = _raw_paired_effects(root, flip_rows)
    paired_aggregate = _paired_effect_aggregate(episode_effects, root)
    _write_csv(output_dir / "episode_decision_comparison.csv", episode_comparison)
    _write_csv(output_dir / "decision_summary.csv", summary)
    _write_csv(output_dir / "stratified_summary.csv", stratified)
    _write_csv(output_dir / "episode_paired_effects.csv", episode_effects)
    _write_csv(output_dir / "chunk_paired_effects.csv", chunk_effects)
    _write_csv(output_dir / "paired_effect_summary.csv", paired_aggregate)
    _plot_aggregate(summary, output_dir / "exterior_hflip_aggregate.png")
    _plot_visual_audit(root, output_dir / "exterior_hflip_visual_audit.png")
    _plot_paired_trajectory_curves(root, output_dir / "paired_trajectory_curves")
    diagnostics = {
        "num_trajectories": 24,
        "visual_intervention": "horizontal flip of exterior image only",
        "unchanged_inputs": [
            "wrist image",
            "robot state",
            "ground-truth action chunk",
            "chunk boundaries",
            "seed and RNG key",
            "flow timestep and paired noise within each prompt family",
        ],
        "pairing_validation": pairing,
        "exponential_alpha": args.exponential_alpha,
        "margin_definitions": {
            "color": "energy_blue - energy_red; positive supports red",
            "side": "energy_left - energy_right; positive supports right",
        },
        "interpretation_limit": "horizontal mirror is an OOD causal probe, not a natural-camera accuracy test",
    }
    (output_dir / "analysis_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(
        json.dumps(
            {
                "decision_summary": summary,
                "paired_effect_overall": [
                    row for row in paired_aggregate if row["grouping"] == "overall"
                ],
                "diagnostics": diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
