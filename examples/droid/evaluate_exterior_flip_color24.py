"""Score exterior-camera horizontal flips on the balanced 24-trajectory experiment."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
import time

import evaluate_color_legibility_24 as color_evaluation
import evaluate_spatial_legibility as evaluation
import jax
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
    output_dir: Path = Path(
        "artifacts/spatial_legibility_color24_20260818_pi05_droid_h16/exterior_hflip/ten_seed"
    )
    config_name: str = "pi05_droid_finetune"
    num_samples: int = 8
    seeds: tuple[int, ...] = tuple(range(10))
    tau_min: float = 0.3
    tau_max: float = 0.7
    action_dims: int = 8
    overwrite: bool = False


def _flip_exterior(chunk: droid_dataset.DroidActionChunk) -> droid_dataset.DroidActionChunk:
    """Horizontally mirror only the exterior image; keep every other input unchanged."""
    return dataclasses.replace(
        chunk,
        exterior_image=np.ascontiguousarray(chunk.exterior_image[:, ::-1]),
    )


def main(args: Args) -> None:
    args = dataclasses.replace(
        args,
        checkpoint_dir=args.checkpoint_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    scores_path = args.output_dir / "chunk_scores.csv"
    config_path = args.output_dir / "scoring_config.json"
    if scores_path.exists() and not args.overwrite:
        raise FileExistsError(f"{scores_path} exists; pass --overwrite to replace it")
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
    config_path.write_text(
        json.dumps(
            {
                "checkpoint_dir": str(args.checkpoint_dir),
                "data_dir": str(args.data_dir),
                "start_episode": args.start_episode,
                "output_dir": str(args.output_dir),
                "config_name": args.config_name,
                "action_horizon": scorer.action_horizon,
                "num_samples": args.num_samples,
                "seeds": list(args.seeds),
                "tau_min": args.tau_min,
                "tau_max": args.tau_max,
                "action_dims": args.action_dims,
                "episodes": [episode.spec.episode_id for episode in episodes],
                "visual_condition": "exterior_hflip",
                "transformation": "np.ascontiguousarray(exterior_image[:, ::-1]); wrist unchanged",
                "prompt_families": {
                    "color": list(color_evaluation.COLOR_INSTRUCTIONS),
                    "side": list(droid_dataset.CANDIDATE_INSTRUCTIONS),
                },
                "margin": "energy_first - energy_second; color: blue-red, side: left-right",
                "pairing_within_prompt_family": (
                    "same observation, action, noise tensor, noisy action, flow timestep, seed, and RNG key"
                ),
                "paired_original_reference": {
                    "color": "../../ten_seed/chunk_scores.csv",
                    "side": "../../side_prompts/ten_seed/chunk_scores.csv",
                    "rng_key": "episode_index * 10000 + chunk_index, identical to original runs",
                },
                "analysis_phase": "pregrasp only; scorer records both phases",
                "jax_devices": [str(device) for device in jax.devices()],
            },
            indent=2,
        )
        + "\n"
    )

    fields = (
        "episode_id",
        "layout_id",
        "condition",
        "target_color",
        "target_side_base",
        "target_side_exterior_original",
        "blue_block_side_base",
        "red_block_side_base",
        "chunk_index",
        "phase",
        "start",
        "stop",
        "executed_steps",
        "progress_pregrasp",
        "progress_full",
        "seed",
        "visual_condition",
        "prompt_family",
        "first_prompt",
        "second_prompt",
        "energy_first",
        "energy_second",
        "margin_first_minus_second",
        "cartesian_x",
        "cartesian_y",
        "cartesian_z",
    )
    prompt_pairs = (
        ("color", color_evaluation.BLUE_INSTRUCTION, color_evaluation.RED_INSTRUCTION),
        ("side", droid_dataset.LEFT_INSTRUCTION, droid_dataset.RIGHT_INSTRUCTION),
    )

    start_time = time.monotonic()
    score_count = 0
    with scores_path.open("w", newline="") as scores_file:
        writer = csv.DictWriter(scores_file, fieldnames=fields)
        writer.writeheader()
        for episode_index, color_episode in enumerate(episodes):
            episode_start = time.monotonic()
            episode = color_episode.spec
            arrays = droid_dataset.load_episode_arrays(episode)
            chunks = list(
                droid_dataset.iter_action_chunks(
                    episode,
                    action_horizon=scorer.action_horizon,
                    arrays=arrays,
                )
            )
            for chunk in chunks:
                flipped_chunk = _flip_exterior(chunk)
                rng_index = episode_index * 10_000 + chunk.chunk_index
                for seed in args.seeds:
                    for prompt_family, first_prompt, second_prompt in prompt_pairs:
                        energy_first, energy_second = color_evaluation.score_prompt_pair(
                            scorer,
                            flipped_chunk,
                            first_prompt,
                            second_prompt,
                            seed=seed,
                            rng_index=rng_index,
                        )
                        writer.writerow(
                            {
                                "episode_id": episode.episode_id,
                                "layout_id": color_episode.layout_id,
                                "condition": episode.condition,
                                "target_color": color_episode.target_color,
                                "target_side_base": episode.target_side,
                                "target_side_exterior_original": (
                                    "right" if episode.target_side == "left" else "left"
                                ),
                                "blue_block_side_base": color_episode.blue_block_side,
                                "red_block_side_base": color_episode.red_block_side,
                                "chunk_index": chunk.chunk_index,
                                "phase": chunk.phase,
                                "start": chunk.start,
                                "stop": chunk.stop,
                                "executed_steps": chunk.executed_steps,
                                "progress_pregrasp": chunk.progress_pregrasp,
                                "progress_full": chunk.progress_full,
                                "seed": seed,
                                "visual_condition": "exterior_hflip",
                                "prompt_family": prompt_family,
                                "first_prompt": first_prompt,
                                "second_prompt": second_prompt,
                                "energy_first": energy_first,
                                "energy_second": energy_second,
                                "margin_first_minus_second": energy_first - energy_second,
                                "cartesian_x": float(chunk.cartesian_position[0]),
                                "cartesian_y": float(chunk.cartesian_position[1]),
                                "cartesian_z": float(chunk.cartesian_position[2]),
                            }
                        )
                        score_count += 1
                scores_file.flush()
            elapsed = time.monotonic() - episode_start
            print(
                f"[{episode_index + 1:02d}/{len(episodes):02d}] {color_episode.layout_id} "
                f"target-{episode.target_side}/{color_episode.target_color} {episode.condition}: "
                f"{len(chunks)} chunks, {elapsed / 60:.1f} min",
                flush=True,
            )

    elapsed = time.monotonic() - start_time
    print(f"Wrote {score_count} rows to {scores_path} in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main(tyro.cli(Args))
