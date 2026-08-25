"""Score the balanced 24-trajectory color-legibility experiment from 2026-08-18."""

from __future__ import annotations

import csv
import dataclasses
import itertools
import json
from pathlib import Path
import time

import evaluate_spatial_legibility as evaluation
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.instruction_likelihood import droid_dataset

BLUE_INSTRUCTION = "pick up the blue block"
RED_INSTRUCTION = "pick up the red block"
COLOR_INSTRUCTIONS = (BLUE_INSTRUCTION, RED_INSTRUCTION)
LAYOUTS = ("L1", "L3")
SIDES = ("left", "right")
COLORS = ("blue", "red")
CONDITIONS = ("C0", "C1", "C2")


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
    seeds: tuple[int, ...] = tuple(range(10))
    tau_min: float = 0.3
    tau_max: float = 0.7
    action_dims: int = 8
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if not 0 <= self.tau_min < self.tau_max <= 1:
            raise ValueError("Expected 0 <= tau_min < tau_max <= 1")
        if not 1 <= self.action_dims <= 8:
            raise ValueError("action_dims must be between 1 and 8")


@dataclasses.dataclass(frozen=True)
class ColorEpisode:
    spec: droid_dataset.EpisodeSpec
    layout_id: str
    target_color: str
    blue_block_side: str
    red_block_side: str
    alpha_normalized: float
    alpha_m: float
    qc_passed: bool
    metadata_success: bool
    metadata_failure: bool


def discover_color_episodes(data_dir: Path, start_episode: str) -> list[ColorEpisode]:
    paths = [path for path in sorted(data_dir.glob("*/trajectory.h5")) if path.parent.name >= start_episode]
    if len(paths) != 24:
        raise ValueError(f"Expected 24 trajectories at or after {start_episode}, got {len(paths)}")

    episodes = []
    observed_cells = set()
    for index, path in enumerate(paths):
        metadata_path = path.parent / "metadata_openpi.json"
        qc_path = path.parent / "trial_qc.json"
        metadata = json.loads(metadata_path.read_text())
        qc = json.loads(qc_path.read_text())
        layout_id = str(metadata.get("layout_id", ""))
        condition = str(metadata.get("condition", ""))
        target_side = str(metadata.get("target_side", ""))
        target_color = str(metadata.get("target_color", ""))
        blue_side = str(metadata.get("blue_block_side", ""))
        red_side = str(metadata.get("red_block_side", ""))
        instruction = str(metadata.get("language_instruction") or metadata.get("current_task") or "")

        if layout_id not in LAYOUTS:
            raise ValueError(f"Unexpected layout {layout_id!r} in {metadata_path}")
        if condition not in CONDITIONS:
            raise ValueError(f"Unexpected condition {condition!r} in {metadata_path}")
        if target_side not in SIDES or blue_side not in SIDES or red_side not in SIDES:
            raise ValueError(f"Invalid side metadata in {metadata_path}")
        if target_color not in COLORS:
            raise ValueError(f"Unexpected target color {target_color!r} in {metadata_path}")
        if red_side == blue_side:
            raise ValueError(f"Blue and red blocks occupy the same side in {metadata_path}")
        expected_side = blue_side if target_color == "blue" else red_side
        expected_instruction = f"pick up the {target_color} block"
        if target_side != expected_side:
            raise ValueError(
                f"Target {target_color} should be on {expected_side}, not {target_side}, in {metadata_path}"
            )
        if instruction != expected_instruction:
            raise ValueError(f"Expected {expected_instruction!r}, got {instruction!r} in {metadata_path}")
        expected_suffix = f"_{layout_id}_{target_side}_{condition}"
        if not path.parent.name.endswith(expected_suffix):
            raise ValueError(f"Episode name disagrees with metadata: {path.parent.name} vs {expected_suffix}")
        if not bool(qc.get("passed")):
            raise ValueError(f"Trial QC did not pass: {qc_path}")

        cell = (layout_id, blue_side, target_color, condition)
        if cell in observed_cells:
            raise ValueError(f"Duplicate factorial cell {cell}")
        observed_cells.add(cell)
        spec = droid_dataset.EpisodeSpec(
            path=path,
            split="test",
            episode_id=path.parent.name,
            instruction=instruction,
            target_side=target_side,
            pair_index=index,
            condition=condition,
        )
        episodes.append(
            ColorEpisode(
                spec=spec,
                layout_id=layout_id,
                target_color=target_color,
                blue_block_side=blue_side,
                red_block_side=red_side,
                alpha_normalized=float(metadata["alpha_normalized"]),
                alpha_m=float(metadata["alpha_m"]),
                qc_passed=True,
                metadata_success=bool(metadata.get("success")),
                metadata_failure=bool(metadata.get("failure")),
            )
        )

    expected_cells = set(itertools.product(LAYOUTS, SIDES, COLORS, CONDITIONS))
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise ValueError(f"Incomplete factorial design; missing={missing}, extra={extra}")
    return episodes


def score_prompt_pair(
    scorer: evaluation.CheckpointScorer,
    chunk: droid_dataset.DroidActionChunk,
    first_prompt: str,
    second_prompt: str,
    *,
    seed: int,
    rng_index: int,
) -> tuple[float, float]:
    """Score two prompts with identical actions, flow times, and noise."""
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


def _episode_geometry(arrays: droid_dataset.EpisodeArrays) -> dict[str, float]:
    xyz = arrays.cartesian_positions[arrays.motion_start : arrays.grasp_start + 1, :3].astype(np.float64)
    segment_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    path_length = float(np.sum(segment_lengths))
    start, end = xyz[0], xyz[-1]
    displacement = end - start
    direct_distance = float(np.linalg.norm(displacement))
    if direct_distance <= 1e-9:
        path_ratio = float("nan")
        max_deviation = 0.0
    else:
        unit = displacement / direct_distance
        relative = xyz - start
        projection = np.outer(relative @ unit, unit)
        path_ratio = path_length / direct_distance
        max_deviation = float(np.max(np.linalg.norm(relative - projection, axis=1)))
    return {
        "pregrasp_path_length_m": path_length,
        "pregrasp_direct_distance_m": direct_distance,
        "pregrasp_path_ratio": path_ratio,
        "pregrasp_max_line_deviation_m": max_deviation,
    }


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

    episodes = discover_color_episodes(args.data_dir, args.start_episode)
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
                "candidate_instructions": list(COLOR_INSTRUCTIONS),
                "margin": "energy_blue - energy_red; positive supports red",
                "pairing": "same action, noise, noisy action, and flow timestep for blue/red candidates",
                "analysis_phase": "pregrasp only; scorer also records postgrasp rows",
                "evidence_timestamp": "chunk stop; no future-action leakage",
                "inclusion_rule": "success directory, episode >= start_episode, trial_qc.passed=true",
                "metadata_flag_note": "collector metadata remains success=false/failure=true after successful archival",
                "jax_devices": [str(device) for device in jax.devices()],
            },
            indent=2,
        )
        + "\n"
    )

    score_fields = (
        "episode_id",
        "layout_id",
        "condition",
        "alpha_normalized",
        "alpha_m",
        "target_color",
        "target_side",
        "blue_block_side",
        "red_block_side",
        "instruction",
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
        "energy_blue",
        "energy_red",
        "energy_true",
        "energy_alternative",
        "margin_blue_minus_red",
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
        "alpha_normalized",
        "alpha_m",
        "target_color",
        "target_side",
        "blue_block_side",
        "red_block_side",
        "instruction",
        "qc_passed",
        "metadata_success",
        "metadata_failure",
        "length",
        "motion_start",
        "grasp_start",
        "pregrasp_frames",
        "pregrasp_seconds",
        "motion_start_x",
        "motion_start_y",
        "motion_start_z",
        "grasp_x",
        "grasp_y",
        "grasp_z",
        "pregrasp_path_length_m",
        "pregrasp_direct_distance_m",
        "pregrasp_path_ratio",
        "pregrasp_max_line_deviation_m",
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
            arrays = droid_dataset.load_episode_arrays(episode)
            geometry = _episode_geometry(arrays)
            manifest_writer.writerow(
                {
                    "episode_id": episode.episode_id,
                    "path": str(episode.path),
                    "layout_id": color_episode.layout_id,
                    "condition": episode.condition,
                    "alpha_normalized": color_episode.alpha_normalized,
                    "alpha_m": color_episode.alpha_m,
                    "target_color": color_episode.target_color,
                    "target_side": episode.target_side,
                    "blue_block_side": color_episode.blue_block_side,
                    "red_block_side": color_episode.red_block_side,
                    "instruction": episode.instruction,
                    "qc_passed": color_episode.qc_passed,
                    "metadata_success": color_episode.metadata_success,
                    "metadata_failure": color_episode.metadata_failure,
                    "length": arrays.length,
                    "motion_start": arrays.motion_start,
                    "grasp_start": arrays.grasp_start,
                    "pregrasp_frames": arrays.grasp_start - arrays.motion_start,
                    "pregrasp_seconds": (arrays.grasp_start - arrays.motion_start) / 15.0,
                    "motion_start_x": float(arrays.cartesian_positions[arrays.motion_start, 0]),
                    "motion_start_y": float(arrays.cartesian_positions[arrays.motion_start, 1]),
                    "motion_start_z": float(arrays.cartesian_positions[arrays.motion_start, 2]),
                    "grasp_x": float(arrays.cartesian_positions[arrays.grasp_start, 0]),
                    "grasp_y": float(arrays.cartesian_positions[arrays.grasp_start, 1]),
                    "grasp_z": float(arrays.cartesian_positions[arrays.grasp_start, 2]),
                    **geometry,
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
                    energy_blue, energy_red = score_prompt_pair(
                        scorer,
                        chunk,
                        BLUE_INSTRUCTION,
                        RED_INSTRUCTION,
                        seed=seed,
                        rng_index=rng_index,
                    )
                    if color_episode.target_color == "blue":
                        energy_true, energy_alternative = energy_blue, energy_red
                    else:
                        energy_true, energy_alternative = energy_red, energy_blue
                    score_writer.writerow(
                        {
                            "episode_id": episode.episode_id,
                            "layout_id": color_episode.layout_id,
                            "condition": episode.condition,
                            "alpha_normalized": color_episode.alpha_normalized,
                            "alpha_m": color_episode.alpha_m,
                            "target_color": color_episode.target_color,
                            "target_side": episode.target_side,
                            "blue_block_side": color_episode.blue_block_side,
                            "red_block_side": color_episode.red_block_side,
                            "instruction": episode.instruction,
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
                            "energy_blue": energy_blue,
                            "energy_red": energy_red,
                            "energy_true": energy_true,
                            "energy_alternative": energy_alternative,
                            "margin_blue_minus_red": energy_blue - energy_red,
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
                f"blue-{color_episode.blue_block_side} target-{color_episode.target_color} "
                f"{episode.condition}: {pregrasp_count}/{len(chunks)} pre/total chunks x "
                f"{len(args.seeds)} seeds in {elapsed:.1f}s"
            )

    elapsed = time.monotonic() - start_time
    print(f"Wrote {score_count} score rows to {scores_path} in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main(tyro.cli(Args))
