"""Compare pi0.5-base and pi0.5-DROID residual observers across fixed temperatures."""

from __future__ import annotations

from collections import defaultdict
import csv
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    pi05_base_dir: Path = Path("artifacts/spatial_legibility_eval_pi05_base_h16")
    pi05_droid_dir: Path = Path("artifacts/spatial_legibility_eval_pi05_droid_h16_matched")
    output_dir: Path = Path("artifacts/spatial_legibility_pi05_base_temperature_comparison")
    temperatures: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

    def __post_init__(self) -> None:
        if not self.temperatures or any(value <= 0 for value in self.temperatures):
            raise ValueError("temperatures must contain positive values")


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


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def _auc(progress: np.ndarray, values: np.ndarray) -> float:
    return float(np.trapz(np.concatenate([[0.5], values]), np.concatenate([[0.0], progress])))


def _episode_evidence(directory: Path) -> list[dict]:
    rows = _read_csv(directory / "chunk_scores_aggregated.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["phase"] == "pregrasp":
            grouped[row["episode_id"]].append(row)

    output = []
    for episode_id, episode_rows in grouped.items():
        episode_rows.sort(key=lambda row: int(row["chunk_index"]))
        margins = np.asarray(
            [float(row["margin_alternative_minus_true"]) for row in episode_rows], dtype=np.float64
        )
        cumulative_margin = np.cumsum(margins)
        progress = np.asarray([float(row["progress_pregrasp"]) for row in episode_rows], dtype=np.float64)
        output.append(
            {
                "episode_id": episode_id,
                "split": episode_rows[0]["split"],
                "condition": episode_rows[0]["condition"],
                "target_side": episode_rows[0]["target_side"],
                "progress": progress,
                "cumulative_margin": cumulative_margin,
                "final_correct": int(cumulative_margin[-1] > 0),
                "raw_sign_auc": _auc(progress, (cumulative_margin > 0).astype(np.float64)),
            }
        )
    return sorted(output, key=lambda row: (row["split"], row["condition"], row["episode_id"]))


def _mean(rows: list[dict], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _temperature_rows(run: str, episodes: list[dict], temperatures: tuple[float, ...]) -> list[dict]:
    output = []
    for temperature in temperatures:
        for episode in episodes:
            belief = _sigmoid(episode["cumulative_margin"] / temperature)
            output.append(
                {
                    "run": run,
                    "temperature": temperature,
                    "split": episode["split"],
                    "condition": episode["condition"],
                    "episode_id": episode["episode_id"],
                    "target_side": episode["target_side"],
                    "final_correct": episode["final_correct"],
                    "raw_sign_early_auc": episode["raw_sign_auc"],
                    "probability_early_auc": _auc(episode["progress"], belief),
                    "final_true_belief": float(belief[-1]),
                    "final_cumulative_margin": float(episode["cumulative_margin"][-1]),
                    "prefix_nll": float(np.mean(np.logaddexp(0.0, -episode["cumulative_margin"] / temperature))),
                }
            )
    return output


def _aggregate(rows: list[dict], *, by_condition: bool) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        condition = row["condition"] if by_condition and row["split"] == "test" else "all"
        if by_condition and row["split"] != "test":
            continue
        grouped[(row["run"], row["temperature"], row["split"], condition)].append(row)

    output = []
    for (run, temperature, split, condition), selected in sorted(grouped.items()):
        output.append(
            {
                "run": run,
                "temperature": temperature,
                "split": split,
                "condition": condition,
                "n_trajectories": len(selected),
                "final_accuracy": _mean(selected, "final_correct"),
                "raw_sign_early_auc": _mean(selected, "raw_sign_early_auc"),
                "probability_early_auc": _mean(selected, "probability_early_auc"),
                "mean_final_true_belief": _mean(selected, "final_true_belief"),
                "mean_prefix_nll": _mean(selected, "prefix_nll"),
            }
        )
    return output


def _markdown_table(rows: list[dict], fields: tuple[tuple[str, str], ...]) -> str:
    header = "| " + " | ".join(label for _, label in fields) + " |"
    separator = "|" + "|".join("---:" for _ in fields) + "|"
    body = []
    for row in rows:
        values = []
        for field, _ in fields:
            value = row[field]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def _plot_sensitivity(rows: list[dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, split in zip(axes, ("train", "test"), strict=True):
        for run in sorted({row["run"] for row in rows}):
            selected = sorted(
                [row for row in rows if row["run"] == run and row["split"] == split],
                key=lambda row: row["temperature"],
            )
            axis.plot(
                [row["temperature"] for row in selected],
                [row["probability_early_auc"] for row in selected],
                marker="o",
                label=run,
            )
        axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        axis.axvline(1.0, color="gray", linewidth=0.8, linestyle=":")
        axis.set_xscale("log")
        axis.set_xlabel("Fixed posterior temperature")
        axis.set_title(f"{split.capitalize()} trajectories")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean probability Early-AUC")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "temperature_sensitivity.png", dpi=180)
    plt.close(figure)


def main(args: Args) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = [
        *_temperature_rows("pi05_base_h16", _episode_evidence(args.pi05_base_dir), args.temperatures),
        *_temperature_rows("pi05_droid_h16", _episode_evidence(args.pi05_droid_dir), args.temperatures),
    ]
    aggregate_rows = _aggregate(episode_rows, by_condition=False)
    condition_rows = _aggregate(episode_rows, by_condition=True)
    _write_csv(output_dir / "episode_temperature_metrics.csv", episode_rows)
    _write_csv(output_dir / "temperature_sensitivity.csv", aggregate_rows)
    _write_csv(output_dir / "condition_temperature_sensitivity.csv", condition_rows)
    _plot_sensitivity(aggregate_rows, output_dir)

    tau_one = [row for row in aggregate_rows if row["temperature"] == 1.0]
    tau_one_conditions = [row for row in condition_rows if row["temperature"] == 1.0]
    sensitivity_test = [row for row in aggregate_rows if row["split"] == "test"]
    fields = (
        ("run", "Run"),
        ("split", "Split"),
        ("n_trajectories", "N"),
        ("final_accuracy", "Final acc."),
        ("raw_sign_early_auc", "Raw-sign AUC"),
        ("probability_early_auc", "Probability AUC"),
        ("mean_final_true_belief", "Final belief"),
        ("mean_prefix_nll", "Prefix NLL"),
    )
    condition_fields = (
        ("run", "Run"),
        ("condition", "Condition"),
        ("final_accuracy", "Final acc."),
        ("raw_sign_early_auc", "Raw-sign AUC"),
        ("probability_early_auc", "Probability AUC"),
        ("mean_final_true_belief", "Final belief"),
    )
    sensitivity_fields = (
        ("run", "Run"),
        ("temperature", "Temperature"),
        ("probability_early_auc", "Test probability AUC"),
        ("mean_final_true_belief", "Test final belief"),
        ("mean_prefix_nll", "Test prefix NLL"),
    )
    report = f"""# pi0.5-base versus pi0.5-DROID: fixed-temperature comparison

This analysis reads saved residual energies; it does not rescore trajectories or modify prior artifacts.
Temperature 1 is the pre-specified primary probability mapping. The remaining temperatures are a sensitivity
analysis, not candidates selected by held-out performance. Final accuracy and raw-sign AUC are temperature-free.

## Fixed temperature 1

{_markdown_table(tau_one, fields)}

## Held-out conditions at temperature 1

{_markdown_table(tau_one_conditions, condition_fields)}

## Held-out temperature sensitivity

{_markdown_table(sensitivity_test, sensitivity_fields)}

No temperature is declared "best" here. Selecting one requires a separate, representative calibration set
and a criterion fixed before evaluating the eight held-out trajectories.
"""
    (output_dir / "REPORT.md").write_text(report)


if __name__ == "__main__":
    main(tyro.cli(Args))
