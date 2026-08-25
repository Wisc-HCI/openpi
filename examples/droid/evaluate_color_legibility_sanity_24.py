"""Run prompt-equality and batch-order checks for the 24 color trajectories."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
import time

import evaluate_color_legibility_24 as color_evaluation
import evaluate_spatial_legibility as evaluation
import numpy as np
import tyro

from openpi.instruction_likelihood import droid_dataset


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: Path = Path("/home/hci-lab/.cache/openpi/openpi-assets/checkpoints/pi05_droid")
    data_dir: Path = Path(
        "/home/hci-lab/repos/droid/data/spatial_legibility_controlled/success/2026-08-18"
    )
    start_episode: str = "2026_08_18_00_40_18_199635_L1_left_C0"
    output_dir: Path = Path("artifacts/spatial_legibility_color24_20260818_pi05_droid_h16/ten_seed")
    config_name: str = "pi05_droid_finetune"
    num_samples: int = 8
    seeds: tuple[int, ...] = (0, 1, 2)
    tau_min: float = 0.3
    tau_max: float = 0.7
    action_dims: int = 8
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must be non-empty")


def _representative_chunks(chunks: list[droid_dataset.DroidActionChunk]) -> list[droid_dataset.DroidActionChunk]:
    pregrasp = [chunk for chunk in chunks if chunk.phase == "pregrasp"]
    if not pregrasp:
        raise ValueError("Episode has no pre-grasp chunks")
    middle = min(pregrasp, key=lambda chunk: abs(chunk.progress_pregrasp - 0.4))
    selected = {chunk.chunk_index: chunk for chunk in (pregrasp[0], middle, pregrasp[-1])}
    return [selected[index] for index in sorted(selected)]


def main(args: Args) -> None:
    args = dataclasses.replace(
        args,
        checkpoint_dir=args.checkpoint_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    raw_path = args.output_dir / "sanity_raw.csv"
    summary_path = args.output_dir / "sanity_summary.json"
    if (raw_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Sanity output exists below {args.output_dir}; pass --overwrite to replace it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = color_evaluation.discover_color_episodes(args.data_dir, args.start_episode)
    scoring_args = evaluation.Args(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        config_name=args.config_name,
        num_samples=args.num_samples,
        seeds=args.seeds,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        action_dims=args.action_dims,
        split="test",
    )
    scorer = evaluation.CheckpointScorer(scoring_args)
    modes = (
        ("blue_blue", color_evaluation.BLUE_INSTRUCTION, color_evaluation.BLUE_INSTRUCTION),
        ("red_red", color_evaluation.RED_INSTRUCTION, color_evaluation.RED_INSTRUCTION),
        ("blue_red", color_evaluation.BLUE_INSTRUCTION, color_evaluation.RED_INSTRUCTION),
        ("red_blue", color_evaluation.RED_INSTRUCTION, color_evaluation.BLUE_INSTRUCTION),
    )
    fields = (
        "episode_id",
        "layout_id",
        "condition",
        "target_color",
        "target_side",
        "blue_block_side",
        "chunk_index",
        "progress_pregrasp",
        "seed",
        "mode",
        "first_prompt",
        "second_prompt",
        "energy_first",
        "energy_second",
    )
    rows = []
    start = time.monotonic()
    for episode_index, color_episode in enumerate(episodes):
        episode = color_episode.spec
        arrays = droid_dataset.load_episode_arrays(episode)
        chunks = list(
            droid_dataset.iter_action_chunks(episode, action_horizon=scorer.action_horizon, arrays=arrays)
        )
        selected = _representative_chunks(chunks)
        for chunk in selected:
            rng_index = episode_index * 10_000 + chunk.chunk_index
            for seed in args.seeds:
                for mode, first_prompt, second_prompt in modes:
                    energy_first, energy_second = color_evaluation.score_prompt_pair(
                        scorer,
                        chunk,
                        first_prompt,
                        second_prompt,
                        seed=seed,
                        rng_index=rng_index,
                    )
                    rows.append(
                        {
                            "episode_id": episode.episode_id,
                            "layout_id": color_episode.layout_id,
                            "condition": episode.condition,
                            "target_color": color_episode.target_color,
                            "target_side": episode.target_side,
                            "blue_block_side": color_episode.blue_block_side,
                            "chunk_index": chunk.chunk_index,
                            "progress_pregrasp": chunk.progress_pregrasp,
                            "seed": seed,
                            "mode": mode,
                            "first_prompt": first_prompt,
                            "second_prompt": second_prompt,
                            "energy_first": energy_first,
                            "energy_second": energy_second,
                        }
                    )
        print(f"[{episode_index + 1:02d}/{len(episodes):02d}] {episode.episode_id}: {len(selected)} chunks")

    with raw_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        key = (row["episode_id"], row["chunk_index"], row["seed"])
        grouped.setdefault(key, {})[row["mode"]] = row
    same_blue_delta = []
    same_red_delta = []
    swap_blue_delta = []
    swap_red_delta = []
    margin_delta = []
    prediction_agreement = []
    for modes_by_key in grouped.values():
        blue_blue = modes_by_key["blue_blue"]
        red_red = modes_by_key["red_red"]
        blue_red = modes_by_key["blue_red"]
        red_blue = modes_by_key["red_blue"]
        same_blue_delta.append(abs(blue_blue["energy_first"] - blue_blue["energy_second"]))
        same_red_delta.append(abs(red_red["energy_first"] - red_red["energy_second"]))
        swap_blue_delta.append(abs(blue_red["energy_first"] - red_blue["energy_second"]))
        swap_red_delta.append(abs(blue_red["energy_second"] - red_blue["energy_first"]))
        forward_margin = blue_red["energy_first"] - blue_red["energy_second"]
        swapped_margin = red_blue["energy_second"] - red_blue["energy_first"]
        margin_delta.append(abs(forward_margin - swapped_margin))
        prediction_agreement.append(bool((forward_margin >= 0) == (swapped_margin >= 0)))

    tolerance = 1e-6
    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "config_name": args.config_name,
        "tested_trajectories": len(episodes),
        "tested_chunks_per_trajectory": 3,
        "tested_seeds": list(args.seeds),
        "num_comparisons": len(grouped),
        "same_blue_max_abs_delta": max(same_blue_delta),
        "same_red_max_abs_delta": max(same_red_delta),
        "swap_blue_max_abs_delta": max(swap_blue_delta),
        "swap_red_max_abs_delta": max(swap_red_delta),
        "swap_margin_max_abs_delta": max(margin_delta),
        "swap_prediction_agreement": float(np.mean(prediction_agreement)),
        "tolerance": tolerance,
        "same_prompt_pass": max(same_blue_delta + same_red_delta) <= tolerance,
        "batch_order_swap_pass": max(swap_blue_delta + swap_red_delta + margin_delta) <= tolerance
        and all(prediction_agreement),
        "paired_noise_by_construction": True,
        "paired_noise_implementation": (
            "Pi0.score_actions samples one times/noises tensor and duplicates it across the two candidates"
        ),
        "elapsed_seconds": time.monotonic() - start,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
