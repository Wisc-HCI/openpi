"""Analyze the fixed-layout C0--C4 spatial-legibility experiment at a fixed temperature."""

from __future__ import annotations

from collections import defaultdict
import csv
import dataclasses
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


CONDITIONS = ("C0", "C1", "C2", "C3", "C4")


@dataclasses.dataclass(frozen=True)
class Args:
    pi05_droid_dir: Path = Path("artifacts/spatial_legibility_controlled_pi05_droid_h16")
    pi05_base_dir: Path = Path("artifacts/spatial_legibility_controlled_pi05_base_h16")
    output_dir: Path = Path("artifacts/spatial_legibility_controlled_comparison")
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


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


def _mean(rows: list[dict], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _average_ranks(values: np.ndarray) -> np.ndarray:
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


def _spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])


def _aggregate_seeds(score_rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in score_rows:
        if row["phase"] == "pregrasp":
            grouped[(row["episode_id"], int(row["chunk_index"]))].append(row)

    output = []
    for (_, _), rows in grouped.items():
        base = rows[0]
        output.append(
            {
                "episode_id": base["episode_id"],
                "condition": base["condition"],
                "target_side": base["target_side"],
                "chunk_index": int(base["chunk_index"]),
                "progress": float(base["progress_pregrasp"]),
                "energy_true": float(np.mean([float(row["energy_true"]) for row in rows])),
                "energy_alternative": float(np.mean([float(row["energy_alternative"]) for row in rows])),
                "local_margin": float(np.mean([float(row["margin_alternative_minus_true"]) for row in rows])),
                "seed_margin_sd": float(np.std([float(row["margin_alternative_minus_true"]) for row in rows], ddof=1)),
            }
        )
    return sorted(output, key=lambda row: (row["episode_id"], row["chunk_index"]))


def _episode_metrics(run: str, directory: Path, temperature: float) -> tuple[list[dict], dict[str, dict]]:
    manifest = {row["episode_id"]: row for row in _read_csv(directory / "episode_manifest.csv")}
    chunks = _aggregate_seeds(_read_csv(directory / "chunk_scores.csv"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in chunks:
        grouped[row["episode_id"]].append(row)

    episode_rows = []
    curves = {}
    for episode_id, rows in grouped.items():
        rows.sort(key=lambda row: row["chunk_index"])
        info = manifest[episode_id]
        metadata_path = Path(info["path"]).parent / "spatial_legibility_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        progress = np.asarray([row["progress"] for row in rows], dtype=np.float64)
        local_margin = np.asarray([row["local_margin"] for row in rows], dtype=np.float64)
        cumulative_margin = np.cumsum(local_margin)
        x = np.concatenate([[0.0], progress])
        margin_curve = np.concatenate([[0.0], cumulative_margin])
        belief = _sigmoid(margin_curve / temperature)
        correct_curve = np.concatenate([[0.5], (cumulative_margin > 0).astype(np.float64)])
        weights = 1.0 - x
        weighted_denominator = float(np.trapz(weights, x))
        margin_30 = float(np.interp(0.3, x, margin_curve))
        margin_50 = float(np.interp(0.5, x, margin_curve))
        sustained_correct_progress = float("nan")
        for index, value in enumerate(cumulative_margin):
            if value > 0 and np.all(cumulative_margin[index:] > 0):
                sustained_correct_progress = float(progress[index])
                break

        episode_rows.append(
            {
                "run": run,
                "condition": info["condition"],
                "pair_index": int(info["pair_index"]),
                "episode_id": episode_id,
                "target_side": info["target_side"],
                "alpha_normalized": float(metadata["alpha_normalized"]),
                "alpha_m": float(metadata["alpha_m"]),
                "planned_path_length_m": float(metadata["planned_path_length_m"]),
                "actual_path_length_m": float(info["pregrasp_path_length_m"]),
                "direct_distance_m": float(info["pregrasp_direct_distance_m"]),
                "path_ratio": float(info["pregrasp_path_ratio"]),
                "max_line_deviation_m": float(info["pregrasp_max_line_deviation_m"]),
                "pregrasp_seconds": float(info["pregrasp_seconds"]),
                "pregrasp_chunks": len(rows),
                "mean_true_energy": float(np.mean([row["energy_true"] for row in rows])),
                "mean_alternative_energy": float(np.mean([row["energy_alternative"] for row in rows])),
                "mean_local_margin": float(np.mean(local_margin)),
                "final_cumulative_margin": float(cumulative_margin[-1]),
                "final_correct": int(cumulative_margin[-1] > 0),
                "raw_sign_early_auc": float(np.trapz(correct_curve, x)),
                "probability_early_auc_tau1": float(np.trapz(belief, x)),
                "weighted_probability_auc_tau1": float(np.trapz(belief * weights, x) / weighted_denominator),
                "final_true_belief_tau1": float(belief[-1]),
                "margin_at_30pct_linear": margin_30,
                "correct_at_30pct_linear": int(margin_30 > 0),
                "belief_at_30pct_tau1_linear": float(_sigmoid(margin_30 / temperature)),
                "margin_at_50pct_linear": margin_50,
                "correct_at_50pct_linear": int(margin_50 > 0),
                "belief_at_50pct_tau1_linear": float(_sigmoid(margin_50 / temperature)),
                "sustained_correct_progress": sustained_correct_progress,
                "mean_seed_margin_sd": float(np.mean([row["seed_margin_sd"] for row in rows])),
            }
        )
        curves[f"{run}:{episode_id}"] = {
            "run": run,
            "condition": info["condition"],
            "target_side": info["target_side"],
            "progress": x,
            "cumulative_margin": margin_curve,
            "belief": belief,
        }
    return sorted(episode_rows, key=lambda row: (row["pair_index"], row["target_side"])), curves


def _condition_metrics(episode_rows: list[dict]) -> list[dict]:
    output = []
    for run in ("pi05_droid", "pi05_base"):
        for condition in CONDITIONS:
            selected = [row for row in episode_rows if row["run"] == run and row["condition"] == condition]
            if len(selected) != 2 or {row["target_side"] for row in selected} != {"left", "right"}:
                raise ValueError(f"Expected one left/right pair for {run}/{condition}, got {selected}")
            pair = {row["target_side"]: row for row in selected}
            output.append(
                {
                    "run": run,
                    "condition": condition,
                    "alpha_normalized": _mean(selected, "alpha_normalized"),
                    "alpha_m": _mean(selected, "alpha_m"),
                    "planned_path_length_m": _mean(selected, "planned_path_length_m"),
                    "actual_path_length_m": _mean(selected, "actual_path_length_m"),
                    "path_ratio": _mean(selected, "path_ratio"),
                    "max_line_deviation_m": _mean(selected, "max_line_deviation_m"),
                    "pregrasp_seconds": _mean(selected, "pregrasp_seconds"),
                    "mean_true_energy": _mean(selected, "mean_true_energy"),
                    "mean_alternative_energy": _mean(selected, "mean_alternative_energy"),
                    "mean_local_margin": _mean(selected, "mean_local_margin"),
                    "mean_final_cumulative_margin": _mean(selected, "final_cumulative_margin"),
                    "pair_discrimination": pair["left"]["final_cumulative_margin"]
                    + pair["right"]["final_cumulative_margin"],
                    "pair_left_bias": (
                        pair["left"]["final_cumulative_margin"] - pair["right"]["final_cumulative_margin"]
                    )
                    / 2,
                    "final_accuracy": _mean(selected, "final_correct"),
                    "raw_sign_early_auc": _mean(selected, "raw_sign_early_auc"),
                    "probability_early_auc_tau1": _mean(selected, "probability_early_auc_tau1"),
                    "weighted_probability_auc_tau1": _mean(selected, "weighted_probability_auc_tau1"),
                    "mean_final_true_belief_tau1": _mean(selected, "final_true_belief_tau1"),
                    "accuracy_at_30pct_linear": _mean(selected, "correct_at_30pct_linear"),
                    "mean_margin_at_30pct_linear": _mean(selected, "margin_at_30pct_linear"),
                    "accuracy_at_50pct_linear": _mean(selected, "correct_at_50pct_linear"),
                    "mean_margin_at_50pct_linear": _mean(selected, "margin_at_50pct_linear"),
                }
            )
    return output


def _trend_metrics(condition_rows: list[dict]) -> list[dict]:
    fields = (
        "mean_true_energy",
        "mean_local_margin",
        "mean_final_cumulative_margin",
        "pair_discrimination",
        "final_accuracy",
        "raw_sign_early_auc",
        "probability_early_auc_tau1",
        "weighted_probability_auc_tau1",
        "accuracy_at_30pct_linear",
        "mean_margin_at_30pct_linear",
        "accuracy_at_50pct_linear",
        "mean_margin_at_50pct_linear",
    )
    output = []
    for run in ("pi05_droid", "pi05_base"):
        selected = sorted([row for row in condition_rows if row["run"] == run], key=lambda row: row["alpha_normalized"])
        x = np.asarray([row["alpha_normalized"] for row in selected], dtype=np.float64)
        for field in fields:
            y = np.asarray([row[field] for row in selected], dtype=np.float64)
            output.append(
                {
                    "run": run,
                    "metric": field,
                    "spearman_rho": _spearman(x, y),
                    "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                    "linear_slope_per_alpha": float(np.polyfit(x, y, 1)[0]),
                    "nondecreasing_steps_of_4": int(np.sum(np.diff(y) >= 0)),
                }
            )
    return output


def _plot_condition_trends(condition_rows: list[dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for run, marker in (("pi05_droid", "o"), ("pi05_base", "s")):
        rows = sorted([row for row in condition_rows if row["run"] == run], key=lambda row: row["alpha_normalized"])
        x = [row["alpha_normalized"] for row in rows]
        axes[0, 0].plot(x, [row["mean_true_energy"] for row in rows], marker=marker, label=run)
        axes[0, 1].plot(x, [row["mean_final_cumulative_margin"] for row in rows], marker=marker, label=run)
        axes[1, 0].plot(x, [row["raw_sign_early_auc"] for row in rows], marker=marker, label=run)
        axes[1, 1].plot(x, [row["probability_early_auc_tau1"] for row in rows], marker=marker, label=run)
    titles = (
        "True-instruction residual (predictability proxy)",
        "Final cumulative residual margin",
        "Temperature-free raw-sign Early-AUC",
        "Probability Early-AUC at fixed temperature 1",
    )
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title)
        axis.set_xlabel("Designed curvature alpha")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axes[1, 1].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    figure.tight_layout()
    figure.savefig(output_dir / "condition_trends.png", dpi=180)
    plt.close(figure)


def _plot_margin_curves(curves: dict[str, dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = dict(zip(CONDITIONS, plt.cm.viridis(np.linspace(0.05, 0.95, len(CONDITIONS))), strict=True))
    for row, run in enumerate(("pi05_droid", "pi05_base")):
        for column, side in enumerate(("left", "right")):
            axis = axes[row, column]
            for curve in curves.values():
                if curve["run"] == run and curve["target_side"] == side:
                    axis.plot(
                        curve["progress"],
                        curve["cumulative_margin"],
                        marker="o",
                        color=colors[curve["condition"]],
                        label=curve["condition"],
                    )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"{run}: {side}")
            axis.set_xlabel("Pregrasp progress")
            axis.set_ylabel("Cumulative alternative-minus-true residual")
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(output_dir / "cumulative_margin_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown_table(rows: list[dict], fields: tuple[tuple[str, str], ...], digits: int = 4) -> str:
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


def main(args: Args) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    droid_rows, droid_curves = _episode_metrics("pi05_droid", args.pi05_droid_dir, args.temperature)
    base_rows, base_curves = _episode_metrics("pi05_base", args.pi05_base_dir, args.temperature)
    episode_rows = [*droid_rows, *base_rows]
    condition_rows = _condition_metrics(episode_rows)
    trend_rows = _trend_metrics(condition_rows)
    _write_csv(output_dir / "episode_metrics_tau1.csv", episode_rows)
    _write_csv(output_dir / "condition_metrics_tau1.csv", condition_rows)
    _write_csv(output_dir / "trend_metrics.csv", trend_rows)
    _plot_condition_trends(condition_rows, output_dir)
    _plot_margin_curves({**droid_curves, **base_curves}, output_dir)

    condition_fields = (
        ("condition", "Condition"),
        ("alpha_m", "Designed offset (m)"),
        ("max_line_deviation_m", "Actual deviation (m)"),
        ("mean_true_energy", "True energy"),
        ("mean_final_cumulative_margin", "Final margin"),
        ("final_accuracy", "Final acc."),
        ("raw_sign_early_auc", "Raw-sign AUC"),
        ("probability_early_auc_tau1", "P-AUC tau=1"),
        ("accuracy_at_30pct_linear", "Acc. at 30%"),
        ("accuracy_at_50pct_linear", "Acc. at 50%"),
    )
    trend_fields = (
        ("metric", "Metric"),
        ("spearman_rho", "Spearman rho"),
        ("pearson_r", "Pearson r"),
        ("nondecreasing_steps_of_4", "Nondecreasing steps / 4"),
    )
    sections = []
    for run in ("pi05_droid", "pi05_base"):
        run_conditions = [row for row in condition_rows if row["run"] == run]
        run_trends = [row for row in trend_rows if row["run"] == run]
        sections.append(
            f"## {run}\n\n{_markdown_table(run_conditions, condition_fields)}\n\n"
            f"### Curvature trends\n\n{_markdown_table(run_trends, trend_fields)}"
        )
    report = f"""# Controlled fixed-layout spatial-legibility evaluation

- Data: ten successful L0 trajectories collected on 2026-08-16, one left/right pair for each C0--C4.
- Designed lateral offsets: 0, 42.36, 84.73, 127.09, and 169.45 mm.
- Residual evidence: non-overlapping 16-action chunks, three seeds and eight common-noise samples.
- Probability summaries use a fixed, pre-specified temperature of {args.temperature:g}; no temperature is fitted
  on these ten trajectories.
- `Raw-sign AUC`, cumulative margin, and top-1 accuracies are temperature-free.
- The 30% and 50% snapshots linearly interpolate cumulative evidence between completed chunk endpoints and are
  therefore secondary to the full chunk curves.

{chr(10).join(sections)}

These are descriptive five-condition trends with one trajectory per target and condition. They do not provide
independent within-cell replication or an inferential test of monotonicity.
"""
    (output_dir / "REPORT.md").write_text(report)


if __name__ == "__main__":
    main(tyro.cli(Args))
