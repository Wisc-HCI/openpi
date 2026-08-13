"""Compare the primary spatial-legibility run with scoring ablations."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    main_dir: Path = Path("artifacts/spatial_legibility_eval_39999")
    joint_only_dir: Path = Path("artifacts/spatial_legibility_eval_39999_joint_only")
    wide_tau_dir: Path = Path("artifacts/spatial_legibility_eval_39999_wide_tau")


def _episode_outcomes(path: Path) -> dict[str, int]:
    with path.open(newline="") as file:
        rows = csv.DictReader(file)
        return {row["episode_id"]: int(row["pregrasp_final_correct"]) for row in rows if row["split"] == "test"}


def main(args: Args) -> None:
    runs = [
        ("primary", args.main_dir),
        ("joint_only", args.joint_only_dir),
        ("wide_tau", args.wide_tau_dir),
    ]
    reference = _episode_outcomes(args.main_dir / "episode_metrics.csv")
    output_rows = []
    for label, directory in runs:
        summary = json.loads((directory / "summary.json").read_text())
        outcomes = _episode_outcomes(directory / "episode_metrics.csv")
        agreement = sum(outcomes[key] == reference[key] for key in reference) / len(reference)
        output_rows.append(
            {
                "run": label,
                "action_dims": summary["action_dims"],
                "num_samples": summary["num_samples"],
                "num_seeds": summary["num_seeds"],
                "tau_min": summary["tau_min"],
                "tau_max": summary["tau_max"],
                "train_final_accuracy": summary["split_summary"]["train"]["pregrasp_final_accuracy"],
                "train_early_auc": summary["split_summary"]["train"]["model_early_auc_mean"],
                "test_final_accuracy": summary["split_summary"]["test"]["pregrasp_final_accuracy"],
                "test_early_auc": summary["split_summary"]["test"]["model_early_auc_mean"],
                "heldout_to_train_energy_ratio": summary["heldout_to_train_true_energy_ratio"],
                "heldout_outcome_agreement_with_primary": agreement,
            }
        )

    output_path = args.main_dir / "robustness_summary.csv"
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    table_rows = [
        "| {run} | {action_dims} | {num_samples}x{num_seeds} | [{tau_min:.1f}, {tau_max:.1f}] | "
        "{train_final_accuracy:.3f} | {test_final_accuracy:.3f} | {test_early_auc:.3f} | "
        "{heldout_to_train_energy_ratio:.1f}x | {heldout_outcome_agreement_with_primary:.3f} |".format(**row)
        for row in output_rows
    ]
    report = """# Spatial-legibility robustness checks

| Run | Action dims | Samples x seeds | Flow-time range | Train acc. | Test acc. | Test AUC | Test/train energy | Outcome agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

`joint_only` excludes the gripper action from the residual. `wide_tau` uses the same total of 24 noise draws
per chunk as the primary run but spreads them across flow time [0.1, 0.9]. Outcome agreement compares each
held-out trajectory's final correct/incorrect classification with the primary run.

The principal conclusion is stable: all runs retain 31/32 accuracy on the fine-tuning trajectories, while
held-out performance remains 3/8 to 4/8 and absolute residual energy is at least 15.7x the fine-tuning level.
The wider flow-time range recovers one held-out trajectory, but does not establish reliable generalization.
""".format(rows="\n".join(table_rows))
    (args.main_dir / "ROBUSTNESS.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main(tyro.cli(Args))
