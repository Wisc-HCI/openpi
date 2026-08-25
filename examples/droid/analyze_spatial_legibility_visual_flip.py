"""Analyze paired horizontal image-flip residual scores."""

from __future__ import annotations

from collections import defaultdict
import csv
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro

CONDITIONS = ("original", "exterior_hflip", "wrist_hflip", "both_hflip")
METHODS = ("flat", "linear", "exponential")
PREFIXES = (0.1, 0.2, 0.3, 0.4, 1.0)


@dataclasses.dataclass(frozen=True)
class Args:
    input_dir: Path = Path(
        "artifacts/spatial_legibility_camera_adjusted_latest_pair_pi05_droid_h16/visual_flip"
    )
    exponential_alpha: float = 3.0


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


def _prediction(value: float) -> str:
    return "R" if value > 0 else "L" if value < 0 else "-"


def _correct(prediction: str, target: str) -> bool:
    return prediction == target[0].upper()


def _weights(method: str, count: int, alpha: float) -> np.ndarray:
    time = np.arange(count, dtype=np.float64) / count
    if method == "flat":
        return np.ones(count)
    if method == "linear":
        return 1.0 - time
    if method == "exponential":
        return np.exp(-alpha * time)
    raise ValueError(method)


def _prefix_value(progress: np.ndarray, margins: np.ndarray, prefix: float) -> float:
    return float(np.interp(prefix, np.concatenate([[0.0], progress]), np.concatenate([[0.0], np.cumsum(margins)])))


def main(args: Args) -> None:
    args = dataclasses.replace(args, input_dir=args.input_dir.resolve())
    rows = _read_csv(args.input_dir / "raw_scores.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["episode_id"], row["visual_condition"])].append(row)

    matrices = {}
    episode_rows = []
    chunk_rows = []
    for (episode_id, condition), selected in sorted(grouped.items()):
        target_base = selected[0]["target_side_base"]
        target_camera = selected[0]["target_side_exterior_camera"]
        seeds = sorted({int(row["seed"]) for row in selected})
        chunk_indices = sorted({int(row["chunk_index"]) for row in selected})
        progress = np.asarray(
            [float(next(row["progress_pregrasp"] for row in selected if int(row["chunk_index"]) == chunk)) for chunk in chunk_indices]
        )
        matrix = np.asarray(
            [
                [
                    float(
                        next(
                            row["margin_left_minus_right"]
                            for row in selected
                            if int(row["seed"]) == seed and int(row["chunk_index"]) == chunk
                        )
                    )
                    for chunk in chunk_indices
                ]
                for seed in seeds
            ],
            dtype=np.float64,
        )
        matrices[(episode_id, condition)] = matrix
        mean_margin = np.mean(matrix, axis=0)
        episode_row = {
            "episode_id": episode_id,
            "target_side_base": target_base,
            "target_side_exterior_camera": target_camera,
            "visual_condition": condition,
            "num_seeds": len(seeds),
            "pregrasp_chunks": len(chunk_indices),
        }
        for prefix in PREFIXES:
            mean_value = _prefix_value(progress, mean_margin, prefix)
            seed_values = np.asarray([_prefix_value(progress, seed_margin, prefix) for seed_margin in matrix])
            prediction = _prediction(mean_value)
            name = round(prefix * 100)
            episode_row[f"margin_{name}pct"] = mean_value
            episode_row[f"prediction_{name}pct"] = prediction
            episode_row[f"base_correct_{name}pct"] = int(_correct(prediction, target_base))
            episode_row[f"camera_correct_{name}pct"] = int(_correct(prediction, target_camera))
            episode_row[f"base_seed_correct_rate_{name}pct"] = float(
                np.mean([_correct(_prediction(value), target_base) for value in seed_values])
            )
            episode_row[f"camera_seed_correct_rate_{name}pct"] = float(
                np.mean([_correct(_prediction(value), target_camera) for value in seed_values])
            )
        for method in METHODS:
            weights = _weights(method, len(chunk_indices), args.exponential_alpha)
            mean_value = float(np.average(mean_margin, weights=weights))
            seed_values = np.average(matrix, axis=1, weights=weights)
            prediction = _prediction(mean_value)
            episode_row[f"{method}_score"] = mean_value
            episode_row[f"{method}_prediction"] = prediction
            episode_row[f"{method}_base_correct"] = int(_correct(prediction, target_base))
            episode_row[f"{method}_camera_correct"] = int(_correct(prediction, target_camera))
            episode_row[f"{method}_base_seed_correct_rate"] = float(
                np.mean([_correct(_prediction(value), target_base) for value in seed_values])
            )
            episode_row[f"{method}_camera_seed_correct_rate"] = float(
                np.mean([_correct(_prediction(value), target_camera) for value in seed_values])
            )
        episode_rows.append(episode_row)
        for column, (chunk_index, chunk_progress) in enumerate(zip(chunk_indices, progress, strict=True)):
            values = matrix[:, column]
            chunk_rows.append(
                {
                    "episode_id": episode_id,
                    "target_side_base": target_base,
                    "visual_condition": condition,
                    "chunk_index": chunk_index,
                    "progress_pregrasp": chunk_progress,
                    "margin_mean": float(np.mean(values)),
                    "margin_seed_sd": float(np.std(values, ddof=1)),
                    "right_support_rate": float(np.mean(values > 0)),
                    "cumulative_margin": float(np.sum(mean_margin[: column + 1])),
                }
            )

    method_rows = []
    for condition in CONDITIONS:
        selected = [row for row in episode_rows if row["visual_condition"] == condition]
        method_rows.extend(
            {
                "visual_condition": condition,
                "method": method,
                "base_frame_accuracy": float(np.mean([row[f"{method}_base_correct"] for row in selected])),
                "camera_frame_accuracy": float(np.mean([row[f"{method}_camera_correct"] for row in selected])),
                "base_frame_seed_pooled_accuracy": float(
                    np.mean([row[f"{method}_base_seed_correct_rate"] for row in selected])
                ),
                "camera_frame_seed_pooled_accuracy": float(
                    np.mean([row[f"{method}_camera_seed_correct_rate"] for row in selected])
                ),
            }
            for method in METHODS
        )

    prefix_rows = []
    for condition in CONDITIONS:
        selected = [row for row in episode_rows if row["visual_condition"] == condition]
        for prefix in PREFIXES:
            name = round(prefix * 100)
            prefix_rows.append(
                {
                    "visual_condition": condition,
                    "prefix": prefix,
                    "base_frame_accuracy": float(np.mean([row[f"base_correct_{name}pct"] for row in selected])),
                    "camera_frame_accuracy": float(
                        np.mean([row[f"camera_correct_{name}pct"] for row in selected])
                    ),
                    "base_frame_seed_pooled_accuracy": float(
                        np.mean([row[f"base_seed_correct_rate_{name}pct"] for row in selected])
                    ),
                    "camera_frame_seed_pooled_accuracy": float(
                        np.mean([row[f"camera_seed_correct_rate_{name}pct"] for row in selected])
                    ),
                }
            )

    effect_rows = []
    episode_ids = sorted({row["episode_id"] for row in episode_rows})
    for episode_id in episode_ids:
        original = matrices[(episode_id, "original")].ravel()
        for condition in CONDITIONS[1:]:
            flipped = matrices[(episode_id, condition)].ravel()
            effect_rows.append(
                {
                    "episode_id": episode_id,
                    "visual_condition": condition,
                    "mean_margin_delta_vs_original": float(np.mean(flipped - original)),
                    "mean_absolute_margin_delta_vs_original": float(np.mean(np.abs(flipped - original))),
                    "same_sign_rate": float(np.mean(np.sign(flipped) == np.sign(original))),
                    "opposite_sign_rate": float(np.mean(np.sign(flipped) == -np.sign(original))),
                    "correlation_with_original": float(np.corrcoef(flipped, original)[0, 1]),
                    "correlation_with_negated_original": float(np.corrcoef(flipped, -original)[0, 1]),
                }
            )

    _write_csv(args.input_dir / "episode_summary.csv", episode_rows)
    _write_csv(args.input_dir / "chunk_summary.csv", chunk_rows)
    _write_csv(args.input_dir / "method_summary.csv", method_rows)
    _write_csv(args.input_dir / "prefix_summary.csv", prefix_rows)
    _write_csv(args.input_dir / "paired_effect_summary.csv", effect_rows)

    for episode_id in episode_ids:
        figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
        episode_chunks = [row for row in chunk_rows if row["episode_id"] == episode_id]
        for condition in CONDITIONS:
            selected = sorted(
                [row for row in episode_chunks if row["visual_condition"] == condition],
                key=lambda row: row["chunk_index"],
            )
            progress = np.asarray([row["progress_pregrasp"] for row in selected]) * 100
            axes[0].plot(progress, [row["margin_mean"] for row in selected], marker="o", label=condition)
            axes[1].plot(progress, [row["cumulative_margin"] for row in selected], marker="o", label=condition)
        row_by_condition = {
            row["visual_condition"]: row for row in episode_rows if row["episode_id"] == episode_id
        }
        x = np.arange(len(METHODS), dtype=np.float64)
        width = 0.18
        for index, condition in enumerate(CONDITIONS):
            axes[2].bar(
                x + (index - 1.5) * width,
                [row_by_condition[condition][f"{method}_score"] for method in METHODS],
                width,
                label=condition,
            )
        axes[2].set_xticks(x, METHODS)
        axes[2].set_ylabel("final weighted score")
        axes[2].set_xlabel("method")
        axes[0].set_ylabel("local margin")
        axes[1].set_ylabel("cumulative margin")
        axes[1].set_xlabel("pre-grasp progress (%)")
        axes[0].set_title(episode_id)
        for axis in axes:
            axis.axhline(0, color="black", linewidth=0.8)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(args.input_dir / f"{episode_id}_visual_flip.png", dpi=160)
        plt.close(figure)

    print(f"Wrote visual-flip analysis for {len(episode_ids)} episodes to {args.input_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
