"""Analyze flat and early-weighted residual evidence on eight trajectories."""

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
    input_dir: Path = Path(
        "artifacts/spatial_legibility_controlled_alpha_0_2_4_comparison/early_weighted_pi05_droid_10seeds"
    )
    output_dir: Path = Path("artifacts/spatial_legibility_controlled_alpha_0_2_4_comparison")
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


def _prediction(margin: float, *, tolerance: float = 1e-12) -> str:
    if margin > tolerance:
        return "R"
    if margin < -tolerance:
        return "L"
    return "-"


def _is_correct(prediction: str, target_side: str) -> bool:
    return prediction == target_side[0].upper()


def _weights(method: str, count: int, alpha: float) -> np.ndarray:
    time = np.arange(count, dtype=np.float64) / count
    if method == "flat":
        return np.ones(count, dtype=np.float64)
    if method == "linear":
        return 1.0 - time
    if method == "exponential":
        return np.exp(-alpha * time)
    raise ValueError(f"Unknown method {method}")


def _interpolate_prefix(progress: np.ndarray, cumulative_numerator: np.ndarray, prefix: float) -> float:
    x = np.concatenate([[0.0], progress])
    y = np.concatenate([[0.0], cumulative_numerator])
    return float(np.interp(prefix, x, y))


def _aggregate_chunks(score_rows: list[dict[str, str]]) -> tuple[list[dict], dict[str, dict[int, list[dict]]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    raw_by_episode_chunk: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for row in score_rows:
        if row["phase"] == "pregrasp":
            grouped[(row["episode_id"], int(row["chunk_index"]))].append(row)

    chunks = []
    for (episode_id, chunk_index), rows in grouped.items():
        rows.sort(key=lambda row: int(row["seed"]))
        side_margins = np.asarray(
            [float(row["energy_left"]) - float(row["energy_right"]) for row in rows], dtype=np.float64
        )
        target_side = rows[0]["target_side"]
        mean_margin = float(np.mean(side_margins))
        mean_prediction = _prediction(mean_margin)
        target_support = side_margins < 0 if target_side == "left" else side_margins > 0
        chunk = {
            "episode_id": episode_id,
            "condition": rows[0]["condition"],
            "target_side": target_side,
            "chunk_index": chunk_index,
            "start": int(rows[0]["start"]),
            "stop": int(rows[0]["stop"]),
            "executed_steps": int(rows[0]["executed_steps"]),
            "progress": float(rows[0]["progress_pregrasp"]),
            "num_seeds": len(rows),
            "energy_left_mean": float(np.mean([float(row["energy_left"]) for row in rows])),
            "energy_right_mean": float(np.mean([float(row["energy_right"]) for row in rows])),
            "local_margin_left_minus_right_mean": mean_margin,
            "local_margin_seed_sd": float(np.std(side_margins, ddof=1)) if len(rows) > 1 else 0.0,
            "right_support_rate": float(np.mean(side_margins > 0)),
            "mean_sign_seed_agreement": float(np.mean([_prediction(value) == mean_prediction for value in side_margins])),
            "true_target_seed_support_rate": float(np.mean(target_support)),
        }
        chunks.append(chunk)
        raw_by_episode_chunk[episode_id][chunk_index] = [
            {
                "seed": int(row["seed"]),
                "side_margin": side_margin,
            }
            for row, side_margin in zip(rows, side_margins, strict=True)
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
        target_side = rows[0]["target_side"]
        progress = np.asarray([row["progress"] for row in rows], dtype=np.float64)
        local_margin = np.asarray([row["local_margin_left_minus_right_mean"] for row in rows], dtype=np.float64)
        count = len(rows)
        cumulative_by_method = {}
        score_by_method = {}
        for method in METHODS:
            weights = _weights(method, count, alpha)
            numerator = np.cumsum(weights * local_margin)
            denominator = np.cumsum(weights)
            cumulative_by_method[method] = numerator
            score_by_method[method] = numerator / denominator

        prefix_margins = {
            prefix: _interpolate_prefix(progress, cumulative_by_method["flat"], prefix) for prefix in PREFIXES
        }
        prefix_predictions = {prefix: _prediction(margin) for prefix, margin in prefix_margins.items()}
        early_correct = any(_is_correct(prefix_predictions[prefix], target_side) for prefix in (0.2, 0.3, 0.4))
        late_reversal = early_correct and not _is_correct(prefix_predictions[1.0], target_side)

        seeds = sorted({item["seed"] for values in raw_by_episode_chunk[episode_id].values() for item in values})
        seed_margin_rows = []
        for seed in seeds:
            seed_margins = []
            for row in rows:
                matches = [
                    item["side_margin"] for item in raw_by_episode_chunk[episode_id][row["chunk_index"]] if item["seed"] == seed
                ]
                if len(matches) != 1:
                    raise ValueError(f"Missing or duplicate seed {seed} for {episode_id} chunk {row['chunk_index']}")
                seed_margins.append(matches[0])
            seed_margin_rows.append(np.asarray(seed_margins, dtype=np.float64))
        seed_margin_matrix = np.stack(seed_margin_rows)

        prefix_seed_predictions = {}
        prefix_seed_correct_rates = {}
        prefix_seed_agreements = {}
        for prefix in PREFIXES:
            seed_values = np.asarray(
                [_interpolate_prefix(progress, np.cumsum(seed_margins), prefix) for seed_margins in seed_margin_matrix]
            )
            seed_predictions = [_prediction(value) for value in seed_values]
            prefix_seed_predictions[prefix] = seed_predictions
            prefix_seed_correct_rates[prefix] = float(
                np.mean([_is_correct(prediction, target_side) for prediction in seed_predictions])
            )
            prefix_seed_agreements[prefix] = float(
                np.mean([prediction == prefix_predictions[prefix] for prediction in seed_predictions])
            )

        method_seed_agreements = {}
        method_seed_correct_rates = {}
        for method in METHODS:
            weights = _weights(method, count, alpha)
            seed_values = np.sum(seed_margin_matrix * weights[None, :], axis=1) / np.sum(weights)
            mean_prediction = _prediction(float(score_by_method[method][-1]))
            seed_predictions = [_prediction(value) for value in seed_values]
            method_seed_agreements[method] = float(np.mean([value == mean_prediction for value in seed_predictions]))
            method_seed_correct_rates[method] = float(
                np.mean([_is_correct(prediction, target_side) for prediction in seed_predictions])
            )

        seed_final_agreement = method_seed_agreements["flat"]

        episode_row = {
            "episode_id": episode_id,
            "condition": rows[0]["condition"],
            "target_side": target_side,
            "pregrasp_chunks": count,
            "num_seeds": len(seeds),
            "prediction_10pct": prefix_predictions[0.1],
            "prediction_20pct": prefix_predictions[0.2],
            "prediction_30pct": prefix_predictions[0.3],
            "prediction_40pct": prefix_predictions[0.4],
            "prediction_100pct": prefix_predictions[1.0],
            "margin_10pct": prefix_margins[0.1],
            "margin_20pct": prefix_margins[0.2],
            "margin_30pct": prefix_margins[0.3],
            "margin_40pct": prefix_margins[0.4],
            "margin_100pct": prefix_margins[1.0],
            "seed_correct_rate_10pct": prefix_seed_correct_rates[0.1],
            "seed_correct_rate_20pct": prefix_seed_correct_rates[0.2],
            "seed_correct_rate_30pct": prefix_seed_correct_rates[0.3],
            "seed_correct_rate_40pct": prefix_seed_correct_rates[0.4],
            "seed_correct_rate_100pct": prefix_seed_correct_rates[1.0],
            "seed_prediction_agreement_10pct": prefix_seed_agreements[0.1],
            "seed_prediction_agreement_20pct": prefix_seed_agreements[0.2],
            "seed_prediction_agreement_30pct": prefix_seed_agreements[0.3],
            "seed_prediction_agreement_40pct": prefix_seed_agreements[0.4],
            "seed_prediction_agreement_100pct": prefix_seed_agreements[1.0],
            "early_correct_20_to_40pct": int(early_correct),
            "late_reversal": int(late_reversal),
            "flat_final_score": float(score_by_method["flat"][-1]),
            "flat_final_prediction": _prediction(float(score_by_method["flat"][-1])),
            "linear_final_score": float(score_by_method["linear"][-1]),
            "linear_final_prediction": _prediction(float(score_by_method["linear"][-1])),
            "exponential_final_score": float(score_by_method["exponential"][-1]),
            "exponential_final_prediction": _prediction(float(score_by_method["exponential"][-1])),
            "flat_seed_correct_rate": method_seed_correct_rates["flat"],
            "linear_seed_correct_rate": method_seed_correct_rates["linear"],
            "exponential_seed_correct_rate": method_seed_correct_rates["exponential"],
            "flat_seed_prediction_agreement": method_seed_agreements["flat"],
            "linear_seed_prediction_agreement": method_seed_agreements["linear"],
            "exponential_seed_prediction_agreement": method_seed_agreements["exponential"],
            "final_prediction_seed_agreement": seed_final_agreement,
            "final_prediction_stable_80pct": int(seed_final_agreement >= 0.8),
            "mean_chunk_sign_seed_agreement": float(np.mean([row["mean_sign_seed_agreement"] for row in rows])),
            "minimum_chunk_sign_seed_agreement": float(np.min([row["mean_sign_seed_agreement"] for row in rows])),
        }
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
            "target_side": target_side,
            "progress": progress,
            "local_margin": local_margin,
            "flat_cumulative_margin": cumulative_by_method["flat"],
            "linear_score": score_by_method["linear"],
            "exponential_score": score_by_method["exponential"],
        }
    episode_rows.sort(key=lambda row: row["episode_id"])
    curve_rows.sort(key=lambda row: (row["episode_id"], row["chunk_index"]))
    return episode_rows, curve_rows, curves


def _method_summary(episodes: list[dict]) -> list[dict]:
    output = []
    for method in METHODS:
        prediction_field = f"{method}_final_prediction"
        correct = [_is_correct(row[prediction_field], row["target_side"]) for row in episodes]
        left = [row for row in episodes if row["target_side"] == "left"]
        right = [row for row in episodes if row["target_side"] == "right"]
        output.append(
            {
                "method": method,
                "overall_accuracy": float(np.mean(correct)),
                "left_accuracy": float(np.mean([_is_correct(row[prediction_field], "left") for row in left])),
                "right_accuracy": float(np.mean([_is_correct(row[prediction_field], "right") for row in right])),
                "seed_pooled_accuracy": float(np.mean([row[f"{method}_seed_correct_rate"] for row in episodes])),
                "left_seed_pooled_accuracy": float(np.mean([row[f"{method}_seed_correct_rate"] for row in left])),
                "right_seed_pooled_accuracy": float(np.mean([row[f"{method}_seed_correct_rate"] for row in right])),
            }
        )
    return output


def _prefix_summary(episodes: list[dict]) -> list[dict]:
    output = []
    for prefix in PREFIXES:
        field = f"prediction_{round(prefix * 100)}pct"
        correct = [_is_correct(row[field], row["target_side"]) for row in episodes]
        left = [row for row in episodes if row["target_side"] == "left"]
        right = [row for row in episodes if row["target_side"] == "right"]
        output.append(
            {
                "prefix": prefix,
                "overall_accuracy": float(np.mean(correct)),
                "left_accuracy": float(np.mean([_is_correct(row[field], "left") for row in left])),
                "right_accuracy": float(np.mean([_is_correct(row[field], "right") for row in right])),
                "seed_pooled_accuracy": float(
                    np.mean([row[f"seed_correct_rate_{round(prefix * 100)}pct"] for row in episodes])
                ),
                "left_seed_pooled_accuracy": float(
                    np.mean([row[f"seed_correct_rate_{round(prefix * 100)}pct"] for row in left])
                ),
                "right_seed_pooled_accuracy": float(
                    np.mean([row[f"seed_correct_rate_{round(prefix * 100)}pct"] for row in right])
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
        axes[0].bar(progress, curve["local_margin"], width=max(2.0, 70.0 / len(progress)), alpha=0.8)
        axes[0].set_ylabel("local m")
        axes[0].set_title(f"{episode_id} (GT {curve['target_side'].upper()})")
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


def main(args: Args) -> None:
    args = dataclasses.replace(args, input_dir=args.input_dir.resolve(), output_dir=args.output_dir.resolve())
    scoring_config = json.loads((args.input_dir / "scoring_config.json").read_text())
    score_rows = _read_csv(args.input_dir / "chunk_scores.csv")
    chunks, raw_by_episode_chunk = _aggregate_chunks(score_rows)
    episodes, curve_rows, curves = _episode_analysis(chunks, raw_by_episode_chunk, args.exponential_alpha)
    method_rows = _method_summary(episodes)
    prefix_rows = _prefix_summary(episodes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "early_weighted_chunk_scores.csv", curve_rows)
    _write_csv(args.output_dir / "early_weighted_episode_summary.csv", episodes)
    _write_csv(args.output_dir / "early_weighted_method_summary.csv", method_rows)
    _write_csv(args.output_dir / "early_weighted_prefix_summary.csv", prefix_rows)
    _plot_curves(curves, args.output_dir)

    first_chunks = [row for row in curve_rows if row["chunk_index"] == 0]
    final_chunks = []
    for episode_id in {row["episode_id"] for row in curve_rows}:
        episode_chunks = [row for row in curve_rows if row["episode_id"] == episode_id]
        final_chunks.append(max(episode_chunks, key=lambda row: row["chunk_index"]))
    diagnostics = {
        "checkpoint_dir": scoring_config["checkpoint_dir"],
        "num_trajectories": len(episodes),
        "num_seeds": len(scoring_config["seeds"]),
        "seeds": scoring_config["seeds"],
        "num_samples_per_seed": scoring_config["num_samples"],
        "exponential_alpha": args.exponential_alpha,
        "margin_definition": "energy_left - energy_right; positive supports right",
        "analysis_phase": "pregrasp only",
        "prefix_interpolation": "piecewise-linear cumulative numerator between chunk-stop progress points",
        "early_correct_definition": "correct at any of 20%, 30%, or 40%",
        "late_reversal_definition": "early_correct and incorrect at 100%",
        "early_correct_count": int(sum(row["early_correct_20_to_40pct"] for row in episodes)),
        "late_reversal_count": int(sum(row["late_reversal"] for row in episodes)),
        "stable_final_prediction_count_80pct": int(sum(row["final_prediction_stable_80pct"] for row in episodes)),
        "first_chunk_mean_margin": float(np.mean([row["local_margin_left_minus_right_mean"] for row in first_chunks])),
        "first_chunk_left_support_count": int(
            sum(row["local_margin_left_minus_right_mean"] < 0 for row in first_chunks)
        ),
        "first_chunk_correct_count": int(
            sum(
                _is_correct(_prediction(row["local_margin_left_minus_right_mean"]), row["target_side"])
                for row in first_chunks
            )
        ),
        "final_chunk_left_support_count": int(
            sum(row["local_margin_left_minus_right_mean"] < 0 for row in final_chunks)
        ),
        "final_chunk_correct_count": int(
            sum(
                _is_correct(_prediction(row["local_margin_left_minus_right_mean"]), row["target_side"])
                for row in final_chunks
            )
        ),
    }
    (args.output_dir / "early_weighted_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps({"methods": method_rows, "prefixes": prefix_rows, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
