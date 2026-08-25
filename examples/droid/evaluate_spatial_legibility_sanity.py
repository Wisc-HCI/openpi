"""Run real-checkpoint invariance checks for the two-prompt residual scorer.

The checks use representative early, middle, and final pre-grasp chunks from
each trajectory.  Every prompt pairing for a row receives the same RNG key, so
same-prompt equality and candidate-order equivariance are tested without noise
or flow-time confounds.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
import time

import evaluate_spatial_legibility as evaluation
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.instruction_likelihood import droid_dataset


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: Path = Path("/home/hci-lab/.cache/openpi/openpi-assets/checkpoints/pi05_droid")
    test_date_dir: Path = Path("/tmp/spatial_legibility_new8_20260817")
    output_dir: Path = Path(
        "artifacts/spatial_legibility_controlled_alpha_0_2_4_comparison/early_weighted_pi05_droid_10seeds"
    )
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


def _score_prompts(
    scorer: evaluation.CheckpointScorer,
    chunk: droid_dataset.DroidActionChunk,
    first_prompt: str,
    second_prompt: str,
    *,
    seed: int,
    rng_index: int,
) -> tuple[float, float]:
    first_observation, first_actions = scorer._transform(chunk, first_prompt)  # noqa: SLF001
    second_observation, second_actions = scorer._transform(chunk, second_prompt)  # noqa: SLF001
    np.testing.assert_array_equal(first_actions, second_actions)
    rng = jax.random.fold_in(jax.random.key(seed), rng_index)
    energies = scorer.score_actions(
        rng,
        first_observation,
        second_observation,
        jnp.asarray(first_actions)[None, ...],
        jnp.asarray(chunk.executed_steps),
        num_samples=scorer.args.num_samples,
        tau_min=scorer.args.tau_min,
        tau_max=scorer.args.tau_max,
        action_dims=scorer.args.action_dims,
    )
    energies = np.asarray(energies, dtype=np.float64)
    if energies.shape != (1, 2) or not np.all(np.isfinite(energies)):
        raise ValueError(f"Unexpected energy output {energies}")
    return float(energies[0, 0]), float(energies[0, 1])


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
        test_date_dir=args.test_date_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    output_path = args.output_dir / "sanity_raw.csv"
    summary_path = args.output_dir / "sanity_summary.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_args = evaluation.Args(
        checkpoint_dir=args.checkpoint_dir,
        test_date_dir=args.test_date_dir,
        output_dir=args.output_dir,
        config_name=args.config_name,
        num_samples=args.num_samples,
        seeds=args.seeds,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        action_dims=args.action_dims,
        split="test",
    )
    episodes = droid_dataset.discover_episode_pairs(args.test_date_dir, "test")
    scorer = evaluation.CheckpointScorer(evaluation_args)
    modes = (
        ("left_left", droid_dataset.LEFT_INSTRUCTION, droid_dataset.LEFT_INSTRUCTION),
        ("right_right", droid_dataset.RIGHT_INSTRUCTION, droid_dataset.RIGHT_INSTRUCTION),
        ("left_right", droid_dataset.LEFT_INSTRUCTION, droid_dataset.RIGHT_INSTRUCTION),
        ("right_left", droid_dataset.RIGHT_INSTRUCTION, droid_dataset.LEFT_INSTRUCTION),
    )
    fields = (
        "episode_id",
        "target_side",
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
    for episode_index, episode in enumerate(episodes):
        arrays = droid_dataset.load_episode_arrays(episode)
        chunks = list(droid_dataset.iter_action_chunks(episode, action_horizon=scorer.action_horizon, arrays=arrays))
        selected = _representative_chunks(chunks)
        for chunk in selected:
            rng_index = episode_index * 10_000 + chunk.chunk_index
            for seed in args.seeds:
                for mode, first_prompt, second_prompt in modes:
                    energy_first, energy_second = _score_prompts(
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
                            "target_side": episode.target_side,
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

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        key = (row["episode_id"], row["chunk_index"], row["seed"])
        grouped.setdefault(key, {})[row["mode"]] = row
    same_left_delta = []
    same_right_delta = []
    swap_left_delta = []
    swap_right_delta = []
    margin_delta = []
    prediction_agreement = []
    for modes_by_key in grouped.values():
        left_left = modes_by_key["left_left"]
        right_right = modes_by_key["right_right"]
        left_right = modes_by_key["left_right"]
        right_left = modes_by_key["right_left"]
        same_left_delta.append(abs(left_left["energy_first"] - left_left["energy_second"]))
        same_right_delta.append(abs(right_right["energy_first"] - right_right["energy_second"]))
        swap_left_delta.append(abs(left_right["energy_first"] - right_left["energy_second"]))
        swap_right_delta.append(abs(left_right["energy_second"] - right_left["energy_first"]))
        forward_margin = left_right["energy_first"] - left_right["energy_second"]
        swapped_margin = right_left["energy_second"] - right_left["energy_first"]
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
        "same_left_max_abs_delta": max(same_left_delta),
        "same_right_max_abs_delta": max(same_right_delta),
        "swap_left_max_abs_delta": max(swap_left_delta),
        "swap_right_max_abs_delta": max(swap_right_delta),
        "swap_margin_max_abs_delta": max(margin_delta),
        "swap_prediction_agreement": float(np.mean(prediction_agreement)),
        "tolerance": tolerance,
        "same_prompt_pass": max(same_left_delta + same_right_delta) <= tolerance,
        "batch_order_swap_pass": max(swap_left_delta + swap_right_delta + margin_delta) <= tolerance
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
