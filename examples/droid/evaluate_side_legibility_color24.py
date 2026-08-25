"""Score left/right prompts on the balanced 24-trajectory color experiment."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
import time

import evaluate_color_legibility_24 as color_evaluation
import evaluate_spatial_legibility as evaluation
import jax
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
    seeds: tuple[int, ...] = tuple(range(10))
    tau_min: float = 0.3
    tau_max: float = 0.7
    action_dims: int = 8
    overwrite: bool = False


def main(args: Args) -> None:
    args = dataclasses.replace(
        args,
        checkpoint_dir=args.checkpoint_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    scores_path = args.output_dir / "chunk_scores.csv"
    manifest_path = args.output_dir / "episode_manifest.csv"
    config_path = args.output_dir / "scoring_config.json"
    if (scores_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Output exists below {args.output_dir}; pass --overwrite to replace it")
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
                "candidate_instructions": list(droid_dataset.CANDIDATE_INSTRUCTIONS),
                "margin": "energy_left - energy_right; positive supports right",
                "pairing": "same action, noise, noisy action, and flow timestep for left/right candidates",
                "analysis_phase": "pregrasp only; scorer also records postgrasp rows",
                "evidence_timestamp": "chunk stop; no future-action leakage",
                "jax_devices": [str(device) for device in jax.devices()],
            },
            indent=2,
        )
        + "\n"
    )

    score_fields = (
        "split",
        "condition",
        "pair_index",
        "episode_id",
        "target_side",
        "instruction",
        "recorded_color_instruction",
        "layout_id",
        "target_color",
        "blue_block_side",
        "red_block_side",
        "chunk_index",
        "phase",
        "start",
        "stop",
        "executed_steps",
        "progress_pregrasp",
        "progress_full",
        "seed",
        "num_samples",
        "tau_min",
        "tau_max",
        "action_dims",
        "energy_left",
        "energy_right",
        "energy_true",
        "energy_alternative",
        "margin_alternative_minus_true",
        "cartesian_x",
        "cartesian_y",
        "cartesian_z",
    )
    manifest_fields = (
        "episode_id",
        "path",
        "layout_id",
        "condition",
        "target_color",
        "target_side",
        "blue_block_side",
        "red_block_side",
        "recorded_color_instruction",
        "scored_side_instruction",
        "length",
        "motion_start",
        "grasp_start",
        "pregrasp_frames",
        "grasp_x",
        "grasp_y",
        "grasp_z",
    )

    start_time = time.monotonic()
    score_count = 0
    with scores_path.open("w", newline="") as scores_file, manifest_path.open("w", newline="") as manifest_file:
        score_writer = csv.DictWriter(scores_file, fieldnames=score_fields)
        manifest_writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
        score_writer.writeheader()
        manifest_writer.writeheader()
        for episode_index, color_episode in enumerate(episodes):
            episode_start = time.monotonic()
            episode = color_episode.spec
            side_instruction = (
                droid_dataset.LEFT_INSTRUCTION
                if episode.target_side == "left"
                else droid_dataset.RIGHT_INSTRUCTION
            )
            arrays = droid_dataset.load_episode_arrays(episode)
            manifest_writer.writerow(
                {
                    "episode_id": episode.episode_id,
                    "path": str(episode.path),
                    "layout_id": color_episode.layout_id,
                    "condition": episode.condition,
                    "target_color": color_episode.target_color,
                    "target_side": episode.target_side,
                    "blue_block_side": color_episode.blue_block_side,
                    "red_block_side": color_episode.red_block_side,
                    "recorded_color_instruction": episode.instruction,
                    "scored_side_instruction": side_instruction,
                    "length": arrays.length,
                    "motion_start": arrays.motion_start,
                    "grasp_start": arrays.grasp_start,
                    "pregrasp_frames": arrays.grasp_start - arrays.motion_start,
                    "grasp_x": float(arrays.cartesian_positions[arrays.grasp_start, 0]),
                    "grasp_y": float(arrays.cartesian_positions[arrays.grasp_start, 1]),
                    "grasp_z": float(arrays.cartesian_positions[arrays.grasp_start, 2]),
                }
            )
            manifest_file.flush()

            chunks = list(
                droid_dataset.iter_action_chunks(
                    episode,
                    action_horizon=scorer.action_horizon,
                    arrays=arrays,
                )
            )
            for chunk in chunks:
                rng_index = episode_index * 10_000 + chunk.chunk_index
                for seed in args.seeds:
                    energy_left, energy_right = scorer.score(chunk, seed=seed, rng_index=rng_index)
                    if episode.target_side == "left":
                        energy_true, energy_alternative = energy_left, energy_right
                    else:
                        energy_true, energy_alternative = energy_right, energy_left
                    score_writer.writerow(
                        {
                            "split": "test",
                            "condition": episode.condition,
                            "pair_index": episode_index,
                            "episode_id": episode.episode_id,
                            "target_side": episode.target_side,
                            "instruction": side_instruction,
                            "recorded_color_instruction": episode.instruction,
                            "layout_id": color_episode.layout_id,
                            "target_color": color_episode.target_color,
                            "blue_block_side": color_episode.blue_block_side,
                            "red_block_side": color_episode.red_block_side,
                            "chunk_index": chunk.chunk_index,
                            "phase": chunk.phase,
                            "start": chunk.start,
                            "stop": chunk.stop,
                            "executed_steps": chunk.executed_steps,
                            "progress_pregrasp": chunk.progress_pregrasp,
                            "progress_full": chunk.progress_full,
                            "seed": seed,
                            "num_samples": args.num_samples,
                            "tau_min": args.tau_min,
                            "tau_max": args.tau_max,
                            "action_dims": args.action_dims,
                            "energy_left": energy_left,
                            "energy_right": energy_right,
                            "energy_true": energy_true,
                            "energy_alternative": energy_alternative,
                            "margin_alternative_minus_true": energy_alternative - energy_true,
                            "cartesian_x": float(chunk.cartesian_position[0]),
                            "cartesian_y": float(chunk.cartesian_position[1]),
                            "cartesian_z": float(chunk.cartesian_position[2]),
                        }
                    )
                    score_count += 1
                scores_file.flush()
            elapsed = time.monotonic() - episode_start
            pregrasp_count = sum(chunk.phase == "pregrasp" for chunk in chunks)
            print(
                f"[{episode_index + 1:02d}/{len(episodes):02d}] {color_episode.layout_id} "
                f"target-{episode.target_side}/{color_episode.target_color} {episode.condition}: "
                f"{pregrasp_count}/{len(chunks)} pre/total chunks x {len(args.seeds)} seeds in {elapsed:.1f}s"
            )

    elapsed = time.monotonic() - start_time
    print(f"Wrote {score_count} score rows to {scores_path} in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main(tyro.cli(Args))
