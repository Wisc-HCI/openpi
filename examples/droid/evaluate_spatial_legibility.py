"""Score spatial-intent trajectories with a fine-tuned pi0.5-DROID checkpoint.

This script evaluates the two fixed instruction hypotheses
``pick up the left block`` and ``pick up the right block`` using common-random-
number flow-matching residuals.  Raw held-out DROID trajectories are read in place;
they are not added to the LeRobot fine-tuning dataset.

Example:

    uv run examples/droid/evaluate_spatial_legibility.py \
      --checkpoint-dir /workspace/checkpoints/pi05_droid_finetune/spatial/39999 \
      --data-dir /home/hci-lab/repos/droid/data \
      --output-dir artifacts/spatial_legibility_eval_39999 \
      --overwrite
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi import transforms
from openpi.instruction_likelihood import droid_dataset
from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from openpi.training import checkpoints
from openpi.training import config as training_config


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: Path
    data_dir: Path = Path("/home/hci-lab/repos/droid/data")
    train_date_dir: Path | None = None
    test_date_dir: Path | None = None
    output_dir: Path = Path("artifacts/spatial_legibility_eval_39999")
    config_name: str = "pi05_droid_finetune"
    date: str = "2026-08-09"
    num_samples: int = 8
    seeds: tuple[int, ...] = (0, 1, 2)
    tau_min: float = 0.3
    tau_max: float = 0.7
    action_dims: int = 8
    split: str = "both"
    max_episodes: int | None = None
    controlled_test: bool = False
    overwrite: bool = False

    def __post_init__(self):
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if not 0 <= self.tau_min < self.tau_max <= 1:
            raise ValueError("Expected 0 <= tau_min < tau_max <= 1")
        if not 1 <= self.action_dims <= 8:
            raise ValueError("action_dims must be between 1 and 8")
        if self.split not in {"train", "test", "both"}:
            raise ValueError("split must be train, test, or both")


def _local_data_config(
    train_config: training_config.TrainConfig,
    checkpoint_dir: Path,
) -> training_config.DataConfig:
    local_assets = training_config.AssetsConfig(
        assets_dir=str(checkpoint_dir / "assets"),
        asset_id="droid",
    )
    data_factory = dataclasses.replace(train_config.data, assets=local_assets)
    return data_factory.create(train_config.assets_dirs, train_config.model)


class CheckpointScorer:
    def __init__(self, args: Args):
        self.args = args
        train_config = training_config.get_config(args.config_name)
        if not train_config.model.pi05:
            raise ValueError(f"{args.config_name} is not a pi0.5 config")
        if train_config.model.action_dim != 32:
            raise ValueError(f"Expected pi05-DROID model action dimension 32, got {train_config.model.action_dim}")

        params_path = args.checkpoint_dir / "params"
        if not params_path.exists():
            raise FileNotFoundError(f"Checkpoint params not found: {params_path}")
        print(f"Loading {args.config_name} from {args.checkpoint_dir}")
        params = model_lib.restore_params(params_path, dtype=jnp.bfloat16)
        self.model = train_config.model.load(params)
        self.action_horizon = train_config.model.action_horizon
        self.data_config = _local_data_config(train_config, args.checkpoint_dir)
        if self.data_config.asset_id is None:
            raise ValueError("DROID data config has no asset_id")
        norm_stats = checkpoints.load_norm_stats(args.checkpoint_dir / "assets", self.data_config.asset_id)
        self.input_transform = transforms.compose(
            [
                *self.data_config.data_transforms.inputs,
                transforms.Normalize(norm_stats, use_quantiles=self.data_config.use_quantile_norm, strict=True),
                *self.data_config.model_transforms.inputs,
            ]
        )
        self.score_actions = nnx_utils.module_jit(
            self.model.score_actions,
            static_argnames=("num_samples", "action_dims"),
        )

    def _transform(
        self,
        chunk: droid_dataset.DroidActionChunk,
        instruction: str,
    ) -> tuple[model_lib.Observation, np.ndarray]:
        raw = {
            "observation/exterior_image_1_left": chunk.exterior_image,
            "observation/wrist_image_left": chunk.wrist_image,
            "observation/joint_position": chunk.joint_position,
            "observation/gripper_position": chunk.gripper_position,
            "actions": chunk.actions,
            "prompt": instruction,
        }
        transformed = self.input_transform(raw)
        actions = np.asarray(transformed.pop("actions"), dtype=np.float32)
        if actions.shape != (self.action_horizon, self.model.action_dim):
            raise ValueError(f"Transformed actions have unexpected shape {actions.shape}")
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        return model_lib.Observation.from_dict(batched), actions

    def score(
        self,
        chunk: droid_dataset.DroidActionChunk,
        *,
        seed: int,
        rng_index: int,
    ) -> tuple[float, float]:
        left_observation, left_actions = self._transform(chunk, droid_dataset.LEFT_INSTRUCTION)
        right_observation, right_actions = self._transform(chunk, droid_dataset.RIGHT_INSTRUCTION)
        if not np.array_equal(left_actions, right_actions):
            raise AssertionError("Action normalization unexpectedly depends on the instruction")
        rng = jax.random.fold_in(jax.random.key(seed), rng_index)
        energies = self.score_actions(
            rng,
            left_observation,
            right_observation,
            jnp.asarray(left_actions)[None, ...],
            jnp.asarray(chunk.executed_steps),
            num_samples=self.args.num_samples,
            tau_min=self.args.tau_min,
            tau_max=self.args.tau_max,
            action_dims=self.args.action_dims,
        )
        energies = np.asarray(energies, dtype=np.float64)
        if energies.shape != (1, 2) or not np.all(np.isfinite(energies)):
            raise ValueError(f"Unexpected energy output {energies}")
        return float(energies[0, 0]), float(energies[0, 1])


SCORE_FIELDS = (
    "split",
    "condition",
    "pair_index",
    "episode_id",
    "target_side",
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
    "energy_left",
    "energy_right",
    "energy_true",
    "energy_alternative",
    "margin_alternative_minus_true",
    "cartesian_x",
    "cartesian_y",
    "cartesian_z",
)

EPISODE_FIELDS = (
    "split",
    "condition",
    "pair_index",
    "episode_id",
    "target_side",
    "instruction",
    "path",
    "length",
    "motion_start",
    "grasp_start",
    "pregrasp_frames",
    "pregrasp_seconds",
    "full_active_frames",
    "full_active_seconds",
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


def _episode_geometry(arrays: droid_dataset.EpisodeArrays) -> dict[str, float]:
    xyz = arrays.cartesian_positions[arrays.motion_start : arrays.grasp_start + 1, :3].astype(np.float64)
    segment_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    path_length = float(np.sum(segment_lengths))
    start, end = xyz[0], xyz[-1]
    displacement = end - start
    direct_distance = float(np.linalg.norm(displacement))
    if direct_distance <= 1e-9:
        max_deviation = 0.0
        path_ratio = float("nan")
    else:
        unit = displacement / direct_distance
        relative = xyz - start
        projection = np.outer(relative @ unit, unit)
        max_deviation = float(np.max(np.linalg.norm(relative - projection, axis=1)))
        path_ratio = path_length / direct_distance
    return {
        "pregrasp_path_length_m": path_length,
        "pregrasp_direct_distance_m": direct_distance,
        "pregrasp_path_ratio": path_ratio,
        "pregrasp_max_line_deviation_m": max_deviation,
    }


def _select_episodes(args: Args) -> tuple[list[droid_dataset.EpisodeSpec], list[droid_dataset.EpisodeSpec]]:
    train_root = args.train_date_dir or args.data_dir / "success" / args.date
    test_root = args.test_date_dir or args.data_dir / "failure" / args.date
    train = droid_dataset.discover_episode_pairs(train_root, "train")
    if args.controlled_test:
        test = droid_dataset.discover_controlled_condition_pairs(test_root)
        if len(train) != 32:
            raise ValueError(f"Expected 32 reference trajectories, got {len(train)}")
    else:
        test = droid_dataset.discover_episode_pairs(test_root, "test")
        droid_dataset.validate_expected_dataset(train, test)
    if args.split == "train":
        test = []
    elif args.split == "test":
        train = []
    episodes = train + test
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
        train = [episode for episode in episodes if episode.split == "train"]
        test = [episode for episode in episodes if episode.split == "test"]
    return train, test


def main(args: Args) -> None:
    args = dataclasses.replace(
        args,
        checkpoint_dir=args.checkpoint_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        train_date_dir=args.train_date_dir.resolve() if args.train_date_dir else None,
        test_date_dir=args.test_date_dir.resolve() if args.test_date_dir else None,
        output_dir=args.output_dir.resolve(),
    )
    scores_path = args.output_dir / "chunk_scores.csv"
    manifest_path = args.output_dir / "episode_manifest.csv"
    config_path = args.output_dir / "scoring_config.json"
    if scores_path.exists() and not args.overwrite:
        raise FileExistsError(f"{scores_path} exists; pass --overwrite to replace it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, test = _select_episodes(args)
    episodes = train + test
    print(f"Using {len(train)} training and {len(test)} held-out trajectories")
    scorer = CheckpointScorer(args)
    config_path.write_text(
        json.dumps(
            {
                **dataclasses.asdict(args),
                "checkpoint_dir": str(args.checkpoint_dir),
                "data_dir": str(args.data_dir),
                "train_date_dir": str(args.train_date_dir) if args.train_date_dir else None,
                "test_date_dir": str(args.test_date_dir) if args.test_date_dir else None,
                "output_dir": str(args.output_dir),
                "jax_devices": [str(device) for device in jax.devices()],
                "candidate_instructions": list(droid_dataset.CANDIDATE_INSTRUCTIONS),
                "evidence_timestamp": "chunk stop; no future-action leakage",
            },
            indent=2,
        )
    )

    start_time = time.monotonic()
    score_count = 0
    with scores_path.open("w", newline="") as scores_file, manifest_path.open("w", newline="") as manifest_file:
        score_writer = csv.DictWriter(scores_file, fieldnames=SCORE_FIELDS)
        manifest_writer = csv.DictWriter(manifest_file, fieldnames=EPISODE_FIELDS)
        score_writer.writeheader()
        manifest_writer.writeheader()
        for episode_index, episode in enumerate(episodes):
            episode_start = time.monotonic()
            arrays = droid_dataset.load_episode_arrays(episode)
            geometry = _episode_geometry(arrays)
            manifest_writer.writerow(
                {
                    "split": episode.split,
                    "condition": episode.condition,
                    "pair_index": episode.pair_index,
                    "episode_id": episode.episode_id,
                    "target_side": episode.target_side,
                    "instruction": episode.instruction,
                    "path": str(episode.path),
                    "length": arrays.length,
                    "motion_start": arrays.motion_start,
                    "grasp_start": arrays.grasp_start,
                    "pregrasp_frames": arrays.grasp_start - arrays.motion_start,
                    "pregrasp_seconds": (arrays.grasp_start - arrays.motion_start) / 15.0,
                    "full_active_frames": arrays.length - arrays.motion_start,
                    "full_active_seconds": (arrays.length - arrays.motion_start) / 15.0,
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
                    energy_left, energy_right = scorer.score(chunk, seed=seed, rng_index=rng_index)
                    if episode.target_side == "left":
                        energy_true, energy_alternative = energy_left, energy_right
                    else:
                        energy_true, energy_alternative = energy_right, energy_left
                    score_writer.writerow(
                        {
                            "split": episode.split,
                            "condition": episode.condition,
                            "pair_index": episode.pair_index,
                            "episode_id": episode.episode_id,
                            "target_side": episode.target_side,
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
            print(
                f"[{episode_index + 1:02d}/{len(episodes):02d}] {episode.split} {episode.condition} "
                f"{episode.target_side}: {len(chunks)} chunks x {len(args.seeds)} seeds in {elapsed:.1f}s"
            )

    total_elapsed = time.monotonic() - start_time
    print(f"Wrote {score_count} score rows to {scores_path} in {total_elapsed / 60:.1f} min")


if __name__ == "__main__":
    main(tyro.cli(Args))
