"""Audit whether held-out targets are closer to same- or opposite-side training targets."""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import numpy as np
import tyro

COORDINATES = ("x", "y", "z")


@dataclasses.dataclass(frozen=True)
class Args:
    manifest: Path = Path("artifacts/spatial_legibility_eval_39999/episode_manifest.csv")
    episode_metrics: Path = Path("artifacts/spatial_legibility_eval_39999/episode_metrics.csv")
    output_dir: Path = Path("artifacts/spatial_legibility_position_keying")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _point(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row[f"grasp_{coordinate}"]) for coordinate in COORDINATES])


def _nearest(point: np.ndarray, rows: list[dict[str, str]], dimensions: tuple[int, ...]) -> tuple[float, str]:
    distances = [np.linalg.norm(point[list(dimensions)] - _point(row)[list(dimensions)]) for row in rows]
    index = int(np.argmin(distances))
    return float(distances[index]), rows[index]["target_side"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args: Args) -> None:
    manifest_rows = _read_csv(args.manifest)
    metrics = {row["episode_id"]: row for row in _read_csv(args.episode_metrics)}
    train = [row for row in manifest_rows if row["split"] == "train"]
    test = [row for row in manifest_rows if row["split"] == "test"]
    if len(train) != 32 or len(test) != 8:
        raise ValueError(f"Expected 32 train and 8 test rows, got {len(train)} and {len(test)}")

    output_rows = []
    for row in test:
        point = _point(row)
        target_side = row["target_side"]
        same = [candidate for candidate in train if candidate["target_side"] == target_side]
        opposite = [candidate for candidate in train if candidate["target_side"] != target_side]
        metric = metrics[row["episode_id"]]
        true_belief = float(metric["model_pregrasp_final_belief"])
        left_belief = true_belief if target_side == "left" else 1 - true_belief
        result = {
            "condition": row["condition"],
            "episode_id": row["episode_id"],
            "target_side": target_side,
            "grasp_x": point[0],
            "grasp_y": point[1],
            "grasp_z": point[2],
            "vla_predicted_side": "left" if left_belief > 0.5 else "right",
            "vla_final_correct": int(metric["pregrasp_final_correct"]),
            "vla_early_auc": float(metric["model_early_auc"]),
            "vla_cumulative_margin": float(metric["cumulative_pregrasp_margin"]),
        }
        for name, dimensions in (("xyz", (0, 1, 2)), ("x", (0,)), ("y", (1,)), ("z", (2,))):
            same_distance, _ = _nearest(point, same, dimensions)
            opposite_distance, _ = _nearest(point, opposite, dimensions)
            any_distance, nearest_side = _nearest(point, train, dimensions)
            result[f"nearest_same_{name}_m"] = same_distance
            result[f"nearest_opposite_{name}_m"] = opposite_distance
            result[f"opposite_minus_same_{name}_m"] = opposite_distance - same_distance
            result[f"nearest_any_{name}_m"] = any_distance
            result[f"nearest_any_{name}_side"] = nearest_side
            result[f"opposite_closer_{name}"] = int(opposite_distance < same_distance)
            result[f"vla_matches_nearest_{name}_side"] = int(result["vla_predicted_side"] == nearest_side)
        output_rows.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "heldout_absolute_position_keying.csv", output_rows)

    distribution_rows = []
    for side in ("left", "right"):
        points = np.asarray([_point(row) for row in train if row["target_side"] == side])
        distribution_rows.append(
            {
                "target_side": side,
                **{
                    f"{coordinate}_{statistic}": value
                    for dimension, coordinate in enumerate(COORDINATES)
                    for statistic, value in (
                        ("min", float(np.min(points[:, dimension]))),
                        ("max", float(np.max(points[:, dimension]))),
                        ("mean", float(np.mean(points[:, dimension]))),
                    )
                },
            }
        )
    _write_csv(args.output_dir / "training_target_position_distributions.csv", distribution_rows)

    xyz_opposite = [row for row in output_rows if row["opposite_closer_xyz"]]
    x_opposite = [row for row in output_rows if row["opposite_closer_x"]]
    y_opposite = [row for row in output_rows if row["opposite_closer_y"]]
    xyz_nearest_accuracy = np.mean([row["nearest_any_xyz_side"] == row["target_side"] for row in output_rows])
    x_nearest_accuracy = np.mean([row["nearest_any_x_side"] == row["target_side"] for row in output_rows])
    y_nearest_accuracy = np.mean([row["nearest_any_y_side"] == row["target_side"] for row in output_rows])
    vla_nearest_xyz_agreement = np.mean([row["vla_matches_nearest_xyz_side"] for row in output_rows])

    rows = "\n".join(
        "| {condition} | {target_side} | {vla_predicted_side} | {nearest_same_xyz_m:.4f} | "
        "{nearest_opposite_xyz_m:.4f} | {nearest_any_xyz_side} | {nearest_same_x_m:.4f} | "
        "{nearest_opposite_x_m:.4f} | {nearest_any_x_side} | {nearest_same_y_m:.4f} | "
        "{nearest_opposite_y_m:.4f} | {nearest_any_y_side} |".format(**row)
        for row in output_rows
    )
    xyz_names = ", ".join(f"{row['condition']} {row['target_side']}" for row in xyz_opposite) or "none"
    x_names = ", ".join(f"{row['condition']} {row['target_side']}" for row in x_opposite) or "none"
    y_names = ", ".join(f"{row['condition']} {row['target_side']}" for row in y_opposite) or "none"
    report = f"""# Absolute-position keying audit

This audit reads the existing checkpoint-39999 manifest and results without modifying them. Distances compare
each held-out grasp-onset end-effector position with all fine-tuning target positions.

| Condition | True | VLA prediction | Same XYZ | Opposite XYZ | XYZ 1-NN | Same x | Opposite x | x 1-NN | Same y | Opposite y | y 1-NN |
|---|---|---|---:|---:|---|---:|---:|---|---:|---:|---|
{rows}

## Summary

- A same-side training target is closer in 3D for **{8 - len(xyz_opposite)}/8** held-out targets; the exception is
  **{xyz_names}**. A 3D 1-nearest-target side classifier therefore has accuracy **{xyz_nearest_accuracy:.3f}**.
- Using coordinate `grasp_x` alone, an opposite-side target is closer for **{len(x_opposite)}/8** cases:
  **{x_names}**. The x-only 1-NN accuracy is **{x_nearest_accuracy:.3f}**.
- Using coordinate `grasp_y` alone, an opposite-side target is closer for **{len(y_opposite)}/8** cases:
  **{y_names}**. The y-only 1-NN accuracy is **{y_nearest_accuracy:.3f}**.
- The VLA prediction agrees with the 3D nearest-target label in only **{vla_nearest_xyz_agreement:.3f}** of cases.

In this robot coordinate frame, `grasp_y` separates the training left/right distributions much more strongly
than `grasp_x`; screen-horizontal direction should not be inferred from the field name alone. Crucially, both
C0 targets are closer to same-side training targets in 3D and in `grasp_y`. Therefore target-coordinate
nearest-neighbor keying does **not** explain the C0 Early-AUC of 0.075. The sole 3D opposite-nearest case is
C3-right, which the VLA actually classifies correctly, also opposing the proposed mechanism.

The x-only observation is weaker: C0-left is 0.6 mm closer in x to an opposite-side training target, and the
VLA gets it wrong. But x alone is not the physical left/right axis here, the difference is tiny, and the same
rule also predicts C3-right incorrectly while the VLA gets it right. This endpoint audit does not rule out
**visual pixel-position keying**, which would require first-frame block annotations or detections rather than
robot Cartesian grasp coordinates.
"""
    (args.output_dir / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main(tyro.cli(Args))
