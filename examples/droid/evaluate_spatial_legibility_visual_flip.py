"""Evaluate horizontal image-flip ablations on the latest left/right pair."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
import time

import evaluate_spatial_legibility as evaluation
import numpy as np
import tyro

from openpi.instruction_likelihood import droid_dataset

CONDITIONS = ("original", "exterior_hflip", "wrist_hflip", "both_hflip")


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: Path = Path("/home/hci-lab/.cache/openpi/openpi-assets/checkpoints/pi05_droid")
    data_dir: Path = Path("/home/hci-lab/repos/droid/data/spatial_legibility_controlled/success/2026-08-17")
    output_dir: Path = Path(
        "artifacts/spatial_legibility_camera_adjusted_latest_pair_pi05_droid_h16/visual_flip"
    )
    config_name: str = "pi05_droid_finetune"
    num_samples: int = 8
    seeds: tuple[int, ...] = tuple(range(10))
    tau_min: float = 0.3
    tau_max: float = 0.7
    action_dims: int = 8
    overwrite: bool = False


def _discover_latest_pair(data_dir: Path) -> list[droid_dataset.EpisodeSpec]:
    paths = sorted(data_dir.glob("2026_08_17_02_*/trajectory.h5"))
    if len(paths) != 2:
        raise ValueError(f"Expected exactly two 02:xx latest-pair trajectories below {data_dir}, got {paths}")
    episodes = []
    for path in paths:
        metadata = json.loads((path.parent / "metadata_openpi.json").read_text())
        instruction = str(metadata["language_instruction"])
        target_side = str(metadata["target_side"])
        if target_side == "left" and instruction != droid_dataset.LEFT_INSTRUCTION:
            raise ValueError(f"Left metadata/prompt mismatch in {path.parent}")
        if target_side == "right" and instruction != droid_dataset.RIGHT_INSTRUCTION:
            raise ValueError(f"Right metadata/prompt mismatch in {path.parent}")
        episodes.append(
            droid_dataset.EpisodeSpec(
                path=path,
                split="test",
                episode_id=path.parent.name,
                instruction=instruction,
                target_side=target_side,
                pair_index=0,
                condition=str(metadata["condition"]),
            )
        )
    if {episode.target_side for episode in episodes} != {"left", "right"}:
        raise ValueError(f"Latest pair does not contain one left and one right trajectory: {episodes}")
    return episodes


def _condition_chunk(
    chunk: droid_dataset.DroidActionChunk,
    condition: str,
) -> droid_dataset.DroidActionChunk:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}")
    exterior = chunk.exterior_image
    wrist = chunk.wrist_image
    if condition in {"exterior_hflip", "both_hflip"}:
        exterior = np.ascontiguousarray(exterior[:, ::-1])
    if condition in {"wrist_hflip", "both_hflip"}:
        wrist = np.ascontiguousarray(wrist[:, ::-1])
    return dataclasses.replace(chunk, exterior_image=exterior, wrist_image=wrist)


def main(args: Args) -> None:
    args = dataclasses.replace(
        args,
        checkpoint_dir=args.checkpoint_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    output_path = args.output_dir / "raw_scores.csv"
    config_path = args.output_dir / "config.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_args = evaluation.Args(
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
    episodes = _discover_latest_pair(args.data_dir)
    scorer = evaluation.CheckpointScorer(evaluation_args)
    config_path.write_text(
        json.dumps(
            {
                **dataclasses.asdict(args),
                "checkpoint_dir": str(args.checkpoint_dir),
                "data_dir": str(args.data_dir),
                "output_dir": str(args.output_dir),
                "episodes": [episode.episode_id for episode in episodes],
                "conditions": list(CONDITIONS),
                "pairing": "same state, action, RNG key, flow timesteps, and noise across visual conditions",
                "margin": "energy_left - energy_right; positive supports right",
            },
            indent=2,
        )
        + "\n"
    )

    fields = (
        "episode_id",
        "target_side_base",
        "target_side_exterior_camera",
        "chunk_index",
        "progress_pregrasp",
        "seed",
        "visual_condition",
        "energy_left",
        "energy_right",
        "margin_left_minus_right",
    )
    start = time.monotonic()
    rows = []
    for episode_index, episode in enumerate(episodes):
        arrays = droid_dataset.load_episode_arrays(episode)
        chunks = [
            chunk
            for chunk in droid_dataset.iter_action_chunks(
                episode,
                action_horizon=scorer.action_horizon,
                arrays=arrays,
            )
            if chunk.phase == "pregrasp"
        ]
        for chunk in chunks:
            rng_index = episode_index * 10_000 + chunk.chunk_index
            for seed in args.seeds:
                for condition in CONDITIONS:
                    conditioned_chunk = _condition_chunk(chunk, condition)
                    energy_left, energy_right = scorer.score(
                        conditioned_chunk,
                        seed=seed,
                        rng_index=rng_index,
                    )
                    rows.append(
                        {
                            "episode_id": episode.episode_id,
                            "target_side_base": episode.target_side,
                            "target_side_exterior_camera": "right" if episode.target_side == "left" else "left",
                            "chunk_index": chunk.chunk_index,
                            "progress_pregrasp": chunk.progress_pregrasp,
                            "seed": seed,
                            "visual_condition": condition,
                            "energy_left": energy_left,
                            "energy_right": energy_right,
                            "margin_left_minus_right": energy_left - energy_right,
                        }
                    )
        print(f"[{episode_index + 1:02d}/{len(episodes):02d}] {episode.episode_id}: {len(chunks)} pre-grasp chunks")

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path} in {(time.monotonic() - start) / 60:.1f} min")


if __name__ == "__main__":
    main(tyro.cli(Args))
