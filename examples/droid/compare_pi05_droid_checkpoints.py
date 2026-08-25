"""Compare base pi0.5-DROID and small-dataset-fine-tuned legibility scores."""

from __future__ import annotations

from collections import defaultdict
import csv
import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    finetuned_dir: Path = Path("artifacts/spatial_legibility_eval_39999")
    base_matched_dir: Path = Path("artifacts/spatial_legibility_eval_pi05_droid_h16_matched")
    base_native_dir: Path = Path("artifacts/spatial_legibility_eval_pi05_droid")
    output_dir: Path = Path("artifacts/spatial_legibility_pi05_droid_comparison")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _raw_sign_metrics(directory: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    rows = _read_csv(directory / "chunk_scores_aggregated.csv")
    by_episode = defaultdict(list)
    for row in rows:
        if row["phase"] == "pregrasp":
            by_episode[row["episode_id"]].append(row)

    episode = {}
    for episode_id, episode_rows in by_episode.items():
        episode_rows.sort(key=lambda row: int(row["chunk_index"]))
        cumulative_margin = 0.0
        cumulative_correct = []
        progress = []
        true_energy = 0.0
        alternative_energy = 0.0
        for row in episode_rows:
            cumulative_margin += float(row["margin_alternative_minus_true"])
            cumulative_correct.append(float(cumulative_margin > 0))
            progress.append(float(row["progress_pregrasp"]))
            true_energy += float(row["energy_true"])
            alternative_energy += float(row["energy_alternative"])
        raw_sign_auc = float(np.trapz([0.5, *cumulative_correct], [0.0, *progress]))
        normalized_contrast = cumulative_margin / (true_energy + alternative_energy)
        episode[episode_id] = {
            "split": episode_rows[0]["split"],
            "condition": episode_rows[0]["condition"],
            "target_side": episode_rows[0]["target_side"],
            "raw_sign_early_auc": raw_sign_auc,
            "cumulative_prefix_accuracy": float(np.mean(cumulative_correct)),
            "normalized_final_contrast": normalized_contrast,
        }

    split = {}
    for split_name in ("train", "test"):
        selected = [row for row in episode.values() if row["split"] == split_name]
        split[split_name] = {
            "raw_sign_early_auc": float(np.mean([row["raw_sign_early_auc"] for row in selected])),
            "cumulative_prefix_accuracy": float(np.mean([row["cumulative_prefix_accuracy"] for row in selected])),
            "normalized_final_contrast": float(np.mean([row["normalized_final_contrast"] for row in selected])),
            "absolute_normalized_final_contrast": float(
                np.mean([abs(row["normalized_final_contrast"]) for row in selected])
            ),
        }
    return episode, split


def _aggregate_row(label: str, directory: Path) -> tuple[dict, dict[str, dict]]:
    summary = json.loads((directory / "summary.json").read_text())
    scoring_config = json.loads((directory / "scoring_config.json").read_text())
    episode, raw_split = _raw_sign_metrics(directory)
    train = summary["split_summary"]["train"]
    test = summary["split_summary"]["test"]
    row = {
        "run": label,
        "config_name": summary.get("config_name", scoring_config["config_name"]),
        "train_final_accuracy": train["pregrasp_final_accuracy"],
        "test_final_accuracy": test["pregrasp_final_accuracy"],
        "train_calibrated_early_auc": train["model_early_auc_mean"],
        "test_calibrated_early_auc": test["model_early_auc_mean"],
        "train_raw_sign_early_auc": raw_split["train"]["raw_sign_early_auc"],
        "test_raw_sign_early_auc": raw_split["test"]["raw_sign_early_auc"],
        "train_true_energy": train["mean_pregrasp_true_energy"],
        "test_true_energy": test["mean_pregrasp_true_energy"],
        "test_to_train_energy_ratio": summary["heldout_to_train_true_energy_ratio"],
        "train_normalized_contrast": raw_split["train"]["normalized_final_contrast"],
        "test_normalized_contrast": raw_split["test"]["normalized_final_contrast"],
        "train_absolute_contrast": raw_split["train"]["absolute_normalized_final_contrast"],
        "test_absolute_contrast": raw_split["test"]["absolute_normalized_final_contrast"],
        "posterior_temperature": summary["posterior_temperature"],
    }
    return row, episode


def _plot_comparison(heldout_rows: list[dict], aggregate_rows: list[dict], output_dir: Path) -> None:
    labels = [f"{row['condition'].split('_')[0]}-{row['target_side'][0].upper()}" for row in heldout_rows]
    x = np.arange(len(labels))
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(
        x - width / 2,
        [row["base_normalized_contrast"] for row in heldout_rows],
        width,
        label="base pi05_droid",
    )
    axes[0].bar(
        x + width / 2,
        [row["finetuned_normalized_contrast"] for row in heldout_rows],
        width,
        label="fine-tuned 39999",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Normalized final residual contrast")
    axes[0].set_title("Positive supports the true instruction")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()

    selected = [row for row in aggregate_rows if row["run"] in {"fine_tuned_39999_h16", "base_pi05_droid_h16"}]
    series_x = np.arange(2)
    for index, row in enumerate(selected):
        offset = (index - 0.5) * width
        axes[1].bar(
            series_x + offset,
            [row["train_true_energy"], row["test_true_energy"]],
            width,
            label=row["run"],
        )
    axes[1].set_xticks(series_x, ["32 fine-tuning episodes", "8 held-out episodes"])
    axes[1].set_ylabel("Mean true-instruction residual energy")
    axes[1].set_title("Fine-tuning creates a narrow training energy well")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "checkpoint_comparison.png", dpi=180)
    plt.close(figure)


def main(args: Args) -> None:
    finetuned, finetuned_raw = _aggregate_row("fine_tuned_39999_h16", args.finetuned_dir)
    base_matched, base_matched_raw = _aggregate_row("base_pi05_droid_h16", args.base_matched_dir)
    base_native, _ = _aggregate_row("base_pi05_droid_native_h15", args.base_native_dir)
    aggregate_rows = [finetuned, base_matched, base_native]

    finetuned_metrics = {row["episode_id"]: row for row in _read_csv(args.finetuned_dir / "episode_metrics.csv")}
    base_metrics = {row["episode_id"]: row for row in _read_csv(args.base_matched_dir / "episode_metrics.csv")}
    heldout_rows = []
    for episode_id, base_raw in base_matched_raw.items():
        if base_raw["split"] != "test":
            continue
        fine_raw = finetuned_raw[episode_id]
        base = base_metrics[episode_id]
        fine = finetuned_metrics[episode_id]
        base_correct = int(base["pregrasp_final_correct"])
        fine_correct = int(fine["pregrasp_final_correct"])
        if base_correct and not fine_correct:
            outcome_change = "correct_to_wrong"
        elif not base_correct and fine_correct:
            outcome_change = "wrong_to_correct"
        elif base_correct:
            outcome_change = "stayed_correct"
        else:
            outcome_change = "stayed_wrong"
        heldout_rows.append(
            {
                "condition": base_raw["condition"],
                "episode_id": episode_id,
                "target_side": base_raw["target_side"],
                "base_correct": base_correct,
                "finetuned_correct": fine_correct,
                "base_predicted_side": base_raw["target_side"]
                if base_correct
                else ("right" if base_raw["target_side"] == "left" else "left"),
                "finetuned_predicted_side": fine_raw["target_side"]
                if fine_correct
                else ("right" if fine_raw["target_side"] == "left" else "left"),
                "outcome_change": outcome_change,
                "base_cumulative_margin": float(base["cumulative_pregrasp_margin"]),
                "finetuned_cumulative_margin": float(fine["cumulative_pregrasp_margin"]),
                "base_normalized_contrast": base_raw["normalized_final_contrast"],
                "finetuned_normalized_contrast": fine_raw["normalized_final_contrast"],
                "base_raw_sign_early_auc": base_raw["raw_sign_early_auc"],
                "finetuned_raw_sign_early_auc": fine_raw["raw_sign_early_auc"],
                "base_true_energy": float(base["mean_pregrasp_true_energy"]),
                "finetuned_true_energy": float(fine["mean_pregrasp_true_energy"]),
            }
        )
    heldout_rows.sort(key=lambda row: (row["condition"], row["episode_id"]))

    pair_rows = []
    for condition in sorted({row["condition"] for row in heldout_rows}):
        pair = {row["target_side"]: row for row in heldout_rows if row["condition"] == condition}
        if set(pair) != {"left", "right"}:
            raise ValueError(f"Expected left/right pair for {condition}, got {set(pair)}")
        base_left_logit = pair["left"]["base_cumulative_margin"]
        base_right_logit = -pair["right"]["base_cumulative_margin"]
        fine_left_logit = pair["left"]["finetuned_cumulative_margin"]
        fine_right_logit = -pair["right"]["finetuned_cumulative_margin"]
        pair_rows.append(
            {
                "condition": condition,
                "base_pair_left_bias": (base_left_logit + base_right_logit) / 2,
                "finetuned_pair_left_bias": (fine_left_logit + fine_right_logit) / 2,
                "base_left_vs_right_discrimination": base_left_logit - base_right_logit,
                "finetuned_left_vs_right_discrimination": fine_left_logit - fine_right_logit,
                "base_pair_ranking_correct": int(base_left_logit > base_right_logit),
                "finetuned_pair_ranking_correct": int(fine_left_logit > fine_right_logit),
                "base_prediction_pattern": f"{pair['left']['base_predicted_side']}/{pair['right']['base_predicted_side']}",
                "finetuned_prediction_pattern": (
                    f"{pair['left']['finetuned_predicted_side']}/{pair['right']['finetuned_predicted_side']}"
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "checkpoint_aggregate_comparison.csv", aggregate_rows)
    _write_csv(args.output_dir / "heldout_outcome_changes.csv", heldout_rows)
    _write_csv(args.output_dir / "pair_bias_discrimination.csv", pair_rows)
    _plot_comparison(heldout_rows, aggregate_rows, args.output_dir)

    aggregate_table = "\n".join(
        "| {run} | {train_final_accuracy:.3f} | {test_final_accuracy:.3f} | "
        "{train_raw_sign_early_auc:.3f} | {test_raw_sign_early_auc:.3f} | "
        "{train_calibrated_early_auc:.3f} | {test_calibrated_early_auc:.3f} | "
        "{train_true_energy:.3f} | {test_true_energy:.3f} | {test_to_train_energy_ratio:.2f}x | "
        "{test_normalized_contrast:.4f} | {posterior_temperature:.4f} |".format(**row)
        for row in aggregate_rows
    )
    heldout_table = "\n".join(
        "| {condition} | {target_side} | {base_correct} | {finetuned_correct} | {outcome_change} | "
        "{base_cumulative_margin:.4f} | {finetuned_cumulative_margin:.4f} | "
        "{base_raw_sign_early_auc:.3f} | {finetuned_raw_sign_early_auc:.3f} |".format(**row)
        for row in heldout_rows
    )
    flipped = [row for row in heldout_rows if row["outcome_change"] == "correct_to_wrong"]
    train_energy_reduction = 1 - finetuned["train_true_energy"] / base_matched["train_true_energy"]
    test_energy_reduction = 1 - finetuned["test_true_energy"] / base_matched["test_true_energy"]
    pair_table = "\n".join(
        "| {condition} | {base_prediction_pattern} | {finetuned_prediction_pattern} | "
        "{base_pair_left_bias:.4f} | {finetuned_pair_left_bias:.4f} | "
        "{base_left_vs_right_discrimination:.4f} | {finetuned_left_vs_right_discrimination:.4f} | "
        "{base_pair_ranking_correct} | {finetuned_pair_ranking_correct} |".format(**row)
        for row in pair_rows
    )
    report = f"""# Base pi0.5-DROID versus checkpoint 39999

The primary weight-isolation comparison uses the same 16-step fine-tuning configuration for both checkpoints.
The official native 15-step base configuration is included as a horizon robustness check. Existing checkpoint-
39999 artifacts are read only and are not regenerated or modified.

## Aggregate comparison

| Run | Train acc. | Test acc. | Train raw-sign AUC | Test raw-sign AUC | Train calibrated AUC | Test calibrated AUC | Train energy | Test energy | Test/train energy | Test norm. contrast | Temperature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{aggregate_table}

`Raw-sign AUC` integrates whether the cumulative residual margin has the correct sign and is independent of
posterior temperature. `Normalized contrast` is cumulative `(alternative - true)` residual divided by their
sum; its sign controls top-1 correctness and its magnitude measures relative separation.

## Held-out outcome changes (matched 16-step comparison)

| Condition | Target | Base correct | Fine-tuned correct | Change | Base margin | Fine-tuned margin | Base raw-sign AUC | Fine-tuned raw-sign AUC |
|---|---|---:|---:|---|---:|---:|---:|---:|
{heldout_table}

Fine-tuning changes **{len(flipped)}/8** held-out trajectories from correct to wrong and changes none from wrong
to correct. The flips are: **{", ".join(f"{row['condition']} {row['target_side']}" for row in flipped)}**.

## Same-layout pair decomposition

For each condition, `Prediction L/R` lists the predicted side for the true-left and true-right trajectories.
`Pair left bias` is the mean left-instruction logit across the pair; nonzero values indicate a layout-level
absolute side preference. `L-vs-R discrimination` subtracts the right-trajectory left logit from the left-
trajectory left logit; positive values mean that the motion-dependent within-layout ranking is correct.

| Condition | Base prediction L/R | Fine-tuned prediction L/R | Base pair left bias | Fine-tuned pair left bias | Base L-vs-R discrimination | Fine-tuned L-vs-R discrimination | Base rank correct | Fine-tuned rank correct |
|---|---|---|---:|---:|---:|---:|---:|---:|
{pair_table}

The fine-tuned model predicts one fixed side for both trajectories in C1, C2, and C3 (left, left, and right,
respectively), consistent with a strong layout-level prior. C1 and C2 still have the correct within-pair
ordering, but the prior is large enough to make the right trajectory cross the wrong side of zero. C0 and C3
also reverse the within-pair ranking, so absolute bias alone is not the full explanation.

## Interpretation

- The base weights have weak in-set instruction separation: only 19/32 final rankings are correct under the
  matched configuration, calibrated Early-AUC is 0.500, and the fitted temperature is 9.247. Thus its 7/8
  held-out top-1 result should not be described as confident calibrated recognition; beliefs remain near 0.5.
- Nevertheless, the base model contains useful held-out directional ranking: test raw-sign Early-AUC is 0.752,
  versus 0.515 after fine-tuning, and seven final raw margins have the correct sign.
- Fine-tuning lowers mean true residual on the exact fine-tuning episodes by **{train_energy_reduction:.1%}**
  (0.302 to 0.015), but lowers it on held-out episodes by only **{test_energy_reduction:.1%}** (0.290 to 0.254).
  This creates the 16.9x held-out/train energy ratio seen only after small-data fine-tuning.
- Fine-tuning greatly strengthens and correctly directs training-episode contrast, but on held-out data it often
  strengthens or reverses evidence in the wrong direction. This pattern is consistent with small-dataset
  specialization/negative transfer rather than a general failure of the original DROID model to score curved
  motion.
- The official native 15-step base run also obtains 7/8 with nearly identical raw-sign behavior, so the base
  result is not an artifact of forcing a 16-step horizon.

Because only eight held-out trajectories were inspected, 7/8 is an encouraging diagnostic, not a population-
level accuracy estimate. A new preregistered test set is needed after this comparison.

Artifacts: `checkpoint_aggregate_comparison.csv`, `heldout_outcome_changes.csv`,
`pair_bias_discrimination.csv`, and `checkpoint_comparison.png`.
"""
    (args.output_dir / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main(tyro.cli(Args))
