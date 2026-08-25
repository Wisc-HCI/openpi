"""Compare color and left/right prompt scoring on the same 24 trajectories."""

from __future__ import annotations

from collections import defaultdict
import csv
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro

PREFIXES = (10, 20, 30, 40, 100)
METHODS = ("flat", "linear", "exponential")


@dataclasses.dataclass(frozen=True)
class Args:
    root_dir: Path = Path("artifacts/spatial_legibility_color24_20260818_pi05_droid_h16")


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


def _decision_fields(decision: str) -> dict[str, str]:
    if decision.endswith("pct"):
        prefix = decision.removesuffix("pct")
        return {
            "color_score": f"margin_{prefix}pct",
            "color_prediction": f"prediction_color_{prefix}pct",
            "color_seed": f"seed_correct_rate_{prefix}pct",
            "side_score": f"margin_{prefix}pct",
            "side_prediction": f"prediction_{prefix}pct",
            "side_seed": f"seed_correct_rate_{prefix}pct",
        }
    return {
        "color_score": f"{decision}_final_score",
        "color_prediction": f"{decision}_final_prediction_color",
        "color_seed": f"{decision}_seed_correct_rate",
        "side_score": f"{decision}_final_score",
        "side_prediction": f"{decision}_final_prediction",
        "side_seed": f"{decision}_seed_correct_rate",
    }


def _side_letter(side: str) -> str:
    return side[0].upper()


def _color_to_side(prediction: str, metadata: dict[str, str]) -> str:
    if prediction == "blue":
        return metadata["blue_block_side"]
    if prediction == "red":
        return metadata["red_block_side"]
    return "-"


def _summary_metrics(rows: list[dict], decision: str) -> dict[str, float]:
    return {
        "accuracy": float(np.mean([row[f"side_{decision}_correct"] for row in rows])),
        "camera_frame_accuracy": float(
            np.mean([row[f"side_{decision}_prediction"] != _side_letter(row["target_side"]) for row in rows])
        ),
        "seed_pooled_accuracy": float(np.mean([row[f"side_{decision}_seed_correct_rate"] for row in rows])),
        "predicted_left_rate": float(
            np.mean([row[f"side_{decision}_prediction"] == "L" for row in rows])
        ),
        "mean_score_left_minus_right": float(np.mean([row[f"side_{decision}_score"] for row in rows])),
    }


def _stratified_side_summary(rows: list[dict], decisions: tuple[str, ...]) -> list[dict]:
    groupings = (
        ("overall", ()),
        ("target_color", ("target_color",)),
        ("target_side", ("target_side",)),
        ("layout", ("layout_id",)),
        ("blue_side", ("blue_block_side",)),
        ("condition", ("condition",)),
        ("target_color_x_side", ("target_color", "target_side")),
        ("layout_x_side", ("layout_id", "target_side")),
        ("condition_x_side", ("condition", "target_side")),
        ("condition_x_target_color", ("condition", "target_color")),
    )
    output = []
    for decision in decisions:
        for grouping_name, fields in groupings:
            grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
            for row in rows:
                grouped[tuple(row[field] for field in fields)].append(row)
            for key, selected in sorted(grouped.items()):
                output.append(
                    {
                        "decision": decision,
                        "grouping": grouping_name,
                        "level": "all" if not key else "/".join(key),
                        "num_trajectories": len(selected),
                        **_summary_metrics(selected, decision),
                    }
                )
    return output


def _side_matched_pair_summary(rows: list[dict], decisions: tuple[str, ...]) -> list[dict]:
    output = []
    for decision in decisions:
        same_side_pairs = []
        grouped_same_side: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped_same_side[(row["layout_id"], row["target_side"], row["condition"])].append(row)
        for group_rows in grouped_same_side.values():
            by_color = {row["target_color"]: row for row in group_rows}
            if set(by_color) != {"blue", "red"}:
                raise ValueError("Incomplete same-side color-swap pair")
            first, second = by_color["blue"], by_color["red"]
            same_side_pairs.append(
                {
                    "both_correct": bool(
                        first[f"side_{decision}_correct"] and second[f"side_{decision}_correct"]
                    ),
                    "desired_relation": (
                        first[f"side_{decision}_prediction"] == second[f"side_{decision}_prediction"]
                    ),
                }
            )

        side_swap_pairs = []
        grouped_side_swap: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped_side_swap[(row["layout_id"], row["target_color"], row["condition"])].append(row)
        for group_rows in grouped_side_swap.values():
            by_side = {row["target_side"]: row for row in group_rows}
            if set(by_side) != {"left", "right"}:
                raise ValueError("Incomplete same-color side-swap pair")
            left, right = by_side["left"], by_side["right"]
            side_swap_pairs.append(
                {
                    "both_correct": bool(
                        left[f"side_{decision}_correct"] and right[f"side_{decision}_correct"]
                    ),
                    "desired_relation": (
                        left[f"side_{decision}_prediction"] != right[f"side_{decision}_prediction"]
                    ),
                }
            )

        for pair_type, pairs in (
            ("same_target_side_color_swap", same_side_pairs),
            ("same_target_color_side_swap", side_swap_pairs),
        ):
            output.append(
                {
                    "decision": decision,
                    "pair_type": pair_type,
                    "num_pairs": len(pairs),
                    "both_correct_rate": float(np.mean([pair["both_correct"] for pair in pairs])),
                    "desired_prediction_relation_rate": float(
                        np.mean([pair["desired_relation"] for pair in pairs])
                    ),
                }
            )
    return output


def _plot_comparison(summary: list[dict], output_path: Path) -> None:
    by_decision = {row["decision"]: row for row in summary}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    prefix_names = tuple(f"{value}pct" for value in PREFIXES)
    x_prefix = np.asarray(PREFIXES)
    axes[0].plot(
        x_prefix,
        [by_decision[name]["color_accuracy"] for name in prefix_names],
        marker="o",
        label="blue/red prompts",
    )
    axes[0].plot(
        x_prefix,
        [by_decision[name]["side_accuracy"] for name in prefix_names],
        marker="o",
        label="left/right prompts",
    )
    axes[0].set_title("Prefix accuracy")
    axes[0].set_xlabel("pre-grasp prefix (%)")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    x = np.arange(len(METHODS))
    axes[1].bar(
        x - 0.18,
        [by_decision[name]["color_accuracy"] for name in METHODS],
        0.36,
        label="blue/red prompts",
    )
    axes[1].bar(
        x + 0.18,
        [by_decision[name]["side_accuracy"] for name in METHODS],
        0.36,
        label="left/right prompts",
    )
    axes[1].set_xticks(x, METHODS)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Final weighted accuracy")
    axes[1].legend()

    axes[2].bar(
        x - 0.18,
        [by_decision[name]["color_implied_left_rate"] for name in METHODS],
        0.36,
        label="color-prompt implied Left",
    )
    axes[2].bar(
        x + 0.18,
        [by_decision[name]["side_predicted_left_rate"] for name in METHODS],
        0.36,
        label="side-prompt Left",
    )
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set_xticks(x, METHODS)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Base-Left prediction rate")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def main(args: Args) -> None:
    root = args.root_dir.resolve()
    color_rows = _read_csv(root / "color24_episode_summary.csv")
    side_rows = _read_csv(root / "side_prompts" / "early_weighted_episode_summary.csv")
    manifest = _read_csv(root / "ten_seed" / "episode_manifest.csv")
    color_by_id = {row["episode_id"]: row for row in color_rows}
    side_by_id = {row["episode_id"]: row for row in side_rows}
    metadata_by_id = {row["episode_id"]: row for row in manifest}
    if color_by_id.keys() != side_by_id.keys() or color_by_id.keys() != metadata_by_id.keys():
        raise ValueError("Color, side, and manifest episode sets differ")

    decisions = tuple(f"{value}pct" for value in PREFIXES) + METHODS
    combined = []
    for episode_id in sorted(color_by_id):
        color = color_by_id[episode_id]
        side = side_by_id[episode_id]
        metadata = metadata_by_id[episode_id]
        row = {
            key: metadata[key]
            for key in (
                "episode_id",
                "layout_id",
                "condition",
                "target_color",
                "target_side",
                "blue_block_side",
                "red_block_side",
            )
        }
        for decision in decisions:
            fields = _decision_fields(decision)
            color_prediction = color[fields["color_prediction"]]
            side_prediction = side[fields["side_prediction"]]
            implied_side = _color_to_side(color_prediction, metadata)
            row.update(
                {
                    f"color_{decision}_score": float(color[fields["color_score"]]),
                    f"color_{decision}_prediction": color_prediction,
                    f"color_{decision}_implied_side": _side_letter(implied_side),
                    f"color_{decision}_correct": int(color_prediction == metadata["target_color"]),
                    f"color_{decision}_seed_correct_rate": float(color[fields["color_seed"]]),
                    f"side_{decision}_score": float(side[fields["side_score"]]),
                    f"side_{decision}_prediction": side_prediction,
                    f"side_{decision}_correct": int(side_prediction == _side_letter(metadata["target_side"])),
                    f"side_{decision}_seed_correct_rate": float(side[fields["side_seed"]]),
                }
            )
        combined.append(row)

    comparison_summary = []
    for decision in decisions:
        color_correct = np.asarray([row[f"color_{decision}_correct"] for row in combined], dtype=bool)
        side_correct = np.asarray([row[f"side_{decision}_correct"] for row in combined], dtype=bool)
        comparison_summary.append(
            {
                "decision": decision,
                "color_accuracy": float(np.mean(color_correct)),
                "side_accuracy": float(np.mean(side_correct)),
                "color_seed_pooled_accuracy": float(
                    np.mean([row[f"color_{decision}_seed_correct_rate"] for row in combined])
                ),
                "side_seed_pooled_accuracy": float(
                    np.mean([row[f"side_{decision}_seed_correct_rate"] for row in combined])
                ),
                "both_correct_count": int(np.sum(color_correct & side_correct)),
                "color_only_correct_count": int(np.sum(color_correct & ~side_correct)),
                "side_only_correct_count": int(np.sum(~color_correct & side_correct)),
                "neither_correct_count": int(np.sum(~color_correct & ~side_correct)),
                "color_implied_left_rate": float(
                    np.mean([row[f"color_{decision}_implied_side"] == "L" for row in combined])
                ),
                "side_predicted_left_rate": float(
                    np.mean([row[f"side_{decision}_prediction"] == "L" for row in combined])
                ),
            }
        )

    side_stratified = _stratified_side_summary(combined, decisions)
    side_pairs = _side_matched_pair_summary(combined, decisions)
    _write_csv(root / "color_vs_side_episode_comparison.csv", combined)
    _write_csv(root / "color_vs_side_summary.csv", comparison_summary)
    _write_csv(root / "side_prompt_stratified_summary.csv", side_stratified)
    _write_csv(root / "side_prompt_matched_pair_summary.csv", side_pairs)
    _plot_comparison(comparison_summary, root / "color_vs_side_prompt_comparison.png")
    print("Wrote color-vs-side prompt comparison for 24 episodes")


if __name__ == "__main__":
    main(tyro.cli(Args))
