"""Run left/right same-prompt and candidate-order checks on the 24 trajectories."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import evaluate_color_legibility_24 as color_evaluation
import evaluate_color_legibility_sanity_24 as color_sanity
import tyro

from openpi.instruction_likelihood import droid_dataset


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: Path = Path("/home/hci-lab/.cache/openpi/openpi-assets/checkpoints/pi05_droid")
    data_dir: Path = Path(
        "/home/hci-lab/repos/droid/data/spatial_legibility_controlled/success/2026-08-18"
    )
    start_episode: str = "2026_08_18_00_40_18_199635_L1_left_C0"
    output_dir: Path = Path(
        "artifacts/spatial_legibility_color24_20260818_pi05_droid_h16/side_prompts/ten_seed"
    )
    config_name: str = "pi05_droid_finetune"
    num_samples: int = 8
    seeds: tuple[int, ...] = (0, 1, 2)
    tau_min: float = 0.3
    tau_max: float = 0.7
    action_dims: int = 8
    overwrite: bool = False


def main(args: Args) -> None:
    color_evaluation.BLUE_INSTRUCTION = droid_dataset.LEFT_INSTRUCTION
    color_evaluation.RED_INSTRUCTION = droid_dataset.RIGHT_INSTRUCTION
    color_sanity.main(
        color_sanity.Args(
            checkpoint_dir=args.checkpoint_dir,
            data_dir=args.data_dir,
            start_episode=args.start_episode,
            output_dir=args.output_dir,
            config_name=args.config_name,
            num_samples=args.num_samples,
            seeds=args.seeds,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            action_dims=args.action_dims,
            overwrite=args.overwrite,
        )
    )

    output_dir = args.output_dir.resolve()
    raw_path = output_dir / "sanity_raw.csv"
    with raw_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    mode_mapping = {
        "blue_blue": "left_left",
        "red_red": "right_right",
        "blue_red": "left_right",
        "red_blue": "right_left",
    }
    for row in rows:
        row["mode"] = mode_mapping[row["mode"]]
    with raw_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "sanity_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["prompt_family"] = "left_right"
    for old, new in (
        ("same_blue_max_abs_delta", "same_left_max_abs_delta"),
        ("same_red_max_abs_delta", "same_right_max_abs_delta"),
        ("swap_blue_max_abs_delta", "swap_left_max_abs_delta"),
        ("swap_red_max_abs_delta", "swap_right_max_abs_delta"),
    ):
        summary[new] = summary.pop(old)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
