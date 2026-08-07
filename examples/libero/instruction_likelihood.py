"""Inventory LIBERO tasks and run the gated pi0.5 instruction-likelihood pilot.

This entry point never downloads data or weights and never runs gradients.  Test 1
uses one demonstration per Spatial task (at most ten independent paired samples) so
the implementation can be validated before any scale-up.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import pathlib
import sys

import numpy as np
from scipy import stats

from openpi.instruction_likelihood import bddl
from openpi.instruction_likelihood import libero_dataset
from openpi.instruction_likelihood import scorer as scorer_lib
from openpi.instruction_likelihood.statistics import paired_lower_test

DEFAULT_BDDL_ROOT = pathlib.Path("third_party/libero/libero/libero/bddl_files")


def _write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inventory(args: argparse.Namespace) -> None:
    spatial = bddl.load_task_specs(args.bddl_root, "libero_spatial")
    object_tasks = bddl.load_task_specs(args.bddl_root, "libero_object")
    candidates = bddl.build_candidate_sets(spatial)
    records = []
    for task in spatial:
        candidate_set = candidates[task.task_id]
        mismatch = bddl.choose_cross_scene_mismatch(task, object_tasks)
        target_type = task.target_type
        bowl_placements = {
            name: list(task.placement_of(name).signature) for name, kind in task.objects.items() if kind == target_type
        }
        records.append(
            {
                "task_id": task.task_id,
                "benchmark_instruction": task.language,
                "bddl_language": task.bddl_language,
                "objects": task.objects,
                "fixtures": task.fixtures,
                "initial_facts": [
                    {"predicate": fact.predicate, "arguments": list(fact.arguments)} for fact in task.initial_facts
                ],
                "goal_facts": [
                    {"predicate": fact.predicate, "arguments": list(fact.arguments)} for fact in task.goal_facts
                ],
                "target_type": target_type,
                "target_object_placements": bowl_placements,
                "executable_task_ids": list(candidate_set.executable_task_ids),
                "executable_instructions": list(candidate_set.executable_instructions),
                "rejected_task_reasons": candidate_set.rejected,
                "cross_scene_mismatch": {
                    "suite": mismatch.suite,
                    "task_id": mismatch.task_id,
                    "instruction": mismatch.language,
                    "problem": mismatch.problem,
                },
            }
        )
    payload = {
        "suite": "libero_spatial",
        "task_count": len(spatial),
        "candidate_rule": (
            "candidate target type must occur at the candidate's referring placement in the current :init, "
            "and all goal entity types must exist"
        ),
        "benchmark_vs_bddl_language_mismatch_count": sum(task.language != task.bddl_language for task in spatial),
        "tasks": records,
    }
    _write_json(args.output_dir / "libero_spatial_inventory.json", payload)
    print(
        json.dumps(
            {"task_count": len(spatial), "candidate_counts": [len(x.executable_task_ids) for x in candidates.values()]}
        )
    )


def inspect_demos(args: argparse.Namespace) -> None:
    demos = libero_dataset.discover_demos(args.demo_root)
    payload = {
        "demo_root": str(args.demo_root.resolve()),
        "demo_count": len(demos),
        "tasks": sorted({demo.task_id for demo in demos}),
        "demos": [
            {
                "path": str(demo.path.resolve()),
                "episode": demo.episode,
                "task_id": demo.task_id,
                "language": demo.language,
                "length": demo.length,
            }
            for demo in demos
        ],
    }
    _write_json(args.output, payload)
    print(json.dumps({"demo_count": len(demos), "task_count": len(payload["tasks"])}))
    if not demos:
        raise SystemExit("No standard LIBERO HDF5 demonstrations found")


def _pilot_demos(demos: list[libero_dataset.DemoRef], max_tasks: int) -> list[libero_dataset.DemoRef]:
    by_task: dict[str, list[libero_dataset.DemoRef]] = {}
    for demo in demos:
        by_task.setdefault(demo.task_id, []).append(demo)
    selected = [sorted(task_demos, key=lambda item: item.episode)[0] for _, task_demos in sorted(by_task.items())]
    return selected[:max_tasks]


def _result_name(demo: libero_dataset.DemoRef) -> str:
    return f"{demo.task_id}__{demo.episode}.npz"


def _validate_true_prompt(demo: libero_dataset.DemoRef, task: bddl.TaskSpec) -> None:
    """Prevent silently replacing a demo's recorded true instruction."""
    if demo.language is None:
        raise ValueError(f"Demo {demo.demo_id} has no HDF5 language instruction")
    if demo.language != task.language:
        raise ValueError(
            f"Demo {demo.demo_id} prompt {demo.language!r} does not match benchmark prompt {task.language!r}"
        )


def _require_test1_passed(output_dir: pathlib.Path) -> None:
    summary_path = output_dir / "test1_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Test-1 gate is closed: missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("result", {}).get("passed", False):
        raise SystemExit(f"Test-1 gate is closed: {summary_path} does not record passed=true")


def _load_result(path: pathlib.Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as result:
        arrays = {key: np.asarray(result[key]) for key in result.files if key != "metadata_json"}
        metadata = json.loads(str(result["metadata_json"]))
    return arrays, metadata


def _save_test1_demo(
    output_path: pathlib.Path,
    demo: libero_dataset.DemoRef,
    true_task: bddl.TaskSpec,
    mismatch: bddl.TaskSpec,
    chunks: list[libero_dataset.ActionChunk],
    chunk_scores: list[scorer_lib.ChunkScores],
    config: scorer_lib.ResidualConfig,
    checkpoint: pathlib.Path,
) -> None:
    residual = np.stack([score.residual for score in chunk_scores])
    normalized_actions = np.stack([score.normalized_actions for score in chunk_scores])
    noise = np.stack([score.noise for score in chunk_scores])
    chunk_energy = scorer_lib.aggregate_residuals(residual)
    normalized_scores = scorer_lib.cumulative_posteriors(chunk_energy, np.ones(2, dtype=bool), config.temperature)
    metadata = {
        "schema_version": 1,
        "sanity_test": 1,
        "demo_id": demo.demo_id,
        "hdf5_path": str(demo.path.resolve()),
        "episode": demo.episode,
        "trajectory_length": demo.length,
        "candidate_roles": ["matched_true", "cross_scene_mismatch"],
        "candidate_task_ids": [true_task.task_id, mismatch.task_id],
        "candidate_instructions": [true_task.language, mismatch.language],
        "flow_timesteps": list(config.flow_timesteps),
        "noise_samples": config.noise_samples,
        "seed": config.seed,
        "action_normalization": "pi05_libero checkpoint q01/q99 mapped to [-1,1] before 32-D padding",
        "residual_definition": "v_theta(x_t,t,o,l) - (epsilon - normalized_action)",
        "aggregation": "mean squared signed residual over flow timestep, noise, 10 action steps, and physical dims 0:7",
        "instruction_prior": "uniform over the two test-1 alternatives",
        "softmax_temperature": config.temperature,
        "checkpoint": str(checkpoint.resolve()),
        "image_rotation_180": True,
        "chunk_tail_policy": "drop incomplete tail; never pad demonstrated actions",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        residual=residual,
        normalized_actions=normalized_actions,
        noise=noise,
        flow_timesteps=np.asarray(config.flow_timesteps, dtype=np.float32),
        chunk_starts=np.asarray([chunk.start for chunk in chunks], dtype=np.int32),
        chunk_stops=np.asarray([chunk.stop for chunk in chunks], dtype=np.int32),
        chunk_energy=chunk_energy,
        cumulative_energy=np.cumsum(chunk_energy, axis=0),
        normalized_scores=normalized_scores,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _summarize_test1(result_dir: pathlib.Path) -> dict[str, object]:
    paths = sorted(result_dir.glob("*.npz"))
    matched: list[float] = []
    mismatched: list[float] = []
    demos: list[dict[str, object]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as result:
            energy = np.asarray(result["chunk_energy"])
            metadata = json.loads(str(result["metadata_json"]))
        if energy.ndim != 2 or energy.shape[1] != 2:
            raise ValueError(f"Unexpected test-1 energy shape in {path}: {energy.shape}")
        paired = float(np.mean(energy[:, 0]))
        mismatch = float(np.mean(energy[:, 1]))
        matched.append(paired)
        mismatched.append(mismatch)
        demos.append(
            {
                "demo_id": metadata["demo_id"],
                "chunk_count": int(energy.shape[0]),
                "matched_mean_residual": paired,
                "mismatch_mean_residual": mismatch,
                "mismatch_minus_matched": mismatch - paired,
            }
        )
    if len(paths) < 2:
        raise ValueError(f"Need at least two completed demo results below {result_dir}")
    test = paired_lower_test(np.asarray(matched), np.asarray(mismatched))
    return {
        "test": 1,
        "hypothesis": "matched instruction residual < cross-scene mismatch residual",
        "unit_of_analysis": "demo (mean across that demo's native stride-1 sliding chunks)",
        "alpha": 0.05,
        "result": dataclasses.asdict(test),
        "demos": demos,
    }


def score_test1(args: argparse.Namespace) -> None:
    spatial = {task.task_id: task for task in bddl.load_task_specs(args.bddl_root, "libero_spatial")}
    object_tasks = bddl.load_task_specs(args.bddl_root, "libero_object")
    all_demos = [demo for demo in libero_dataset.discover_demos(args.demo_root) if demo.task_id in spatial]
    demos = _pilot_demos(all_demos, args.max_tasks)
    if len(demos) < 6:
        raise SystemExit(
            f"Found only {len(demos)} Spatial tasks with demos; at least 6 independent pairs are needed for a "
            "two-sided-resolution-compatible one-sided Wilcoxon p<0.05 pilot"
        )
    config = scorer_lib.ResidualConfig(
        flow_timesteps=tuple(args.flow_timesteps),
        noise_samples=args.noise_samples,
        seed=args.seed,
        eval_batch_size=args.eval_batch_size,
    )
    scorer = scorer_lib.Pi05InstructionLikelihoodScorer(args.checkpoint, config=config)
    if scorer.action_horizon != 10:
        raise ValueError(f"Expected pi05_libero horizon 10, got {scorer.action_horizon}")
    result_dir = args.output_dir / "test1_raw"
    for demo_index, demo in enumerate(demos):
        output_path = result_dir / _result_name(demo)
        if output_path.exists() and not args.overwrite:
            print(f"resume: {output_path}", file=sys.stderr)
            continue
        task = spatial[demo.task_id]
        _validate_true_prompt(demo, task)
        mismatch = bddl.choose_cross_scene_mismatch(task, object_tasks)
        chunks = list(
            libero_dataset.iter_action_chunks(
                demo,
                action_horizon=scorer.action_horizon,
                stride=1,
                rotate_images_180=True,
            )
        )
        if not chunks:
            raise ValueError(f"{demo.demo_id} has no complete action chunk")
        scores = [
            scorer.score_chunk(chunk, [task.language, mismatch.language], chunk_index=demo_index * 100_000 + index)
            for index, chunk in enumerate(chunks)
        ]
        _save_test1_demo(output_path, demo, task, mismatch, chunks, scores, config, args.checkpoint)
        print(f"saved {output_path}", file=sys.stderr)
    summary = _summarize_test1(result_dir)
    _write_json(args.output_dir / "test1_summary.json", summary)
    print(json.dumps(summary["result"], sort_keys=True))


def _save_single_condition(
    output_path: pathlib.Path,
    *,
    demo: libero_dataset.DemoRef,
    condition: str,
    instruction: str,
    chunks: list[libero_dataset.ActionChunk],
    chunk_scores: list[scorer_lib.ChunkScores],
    config: scorer_lib.ResidualConfig,
    checkpoint: pathlib.Path,
    extra_arrays: dict[str, np.ndarray] | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> None:
    residual = np.stack([score.residual for score in chunk_scores])
    normalized_actions = np.stack([score.normalized_actions for score in chunk_scores])
    noise = np.stack([score.noise for score in chunk_scores])
    chunk_energy = scorer_lib.aggregate_residuals(residual)
    metadata = {
        "schema_version": 1,
        "condition": condition,
        "demo_id": demo.demo_id,
        "hdf5_path": str(demo.path.resolve()),
        "episode": demo.episode,
        "trajectory_length": demo.length,
        "instruction": instruction,
        "flow_timesteps": list(config.flow_timesteps),
        "noise_samples": config.noise_samples,
        "flow_noise_seed": config.seed,
        "action_normalization": "pi05_libero checkpoint q01/q99 mapped to [-1,1] before 32-D padding",
        "residual_definition": "v_theta(x_t,t,o,l) - (epsilon - normalized_action)",
        "checkpoint": str(checkpoint.resolve()),
        "image_rotation_180": True,
        "chunk_stride": 1,
        "chunk_tail_policy": "drop incomplete tail; never pad actions",
    }
    metadata.update(extra_metadata or {})
    arrays = {
        "residual": residual,
        "normalized_actions": normalized_actions,
        "noise": noise,
        "flow_timesteps": np.asarray(config.flow_timesteps, dtype=np.float32),
        "chunk_starts": np.asarray([chunk.start for chunk in chunks], dtype=np.int32),
        "chunk_stops": np.asarray([chunk.stop for chunk in chunks], dtype=np.int32),
        "chunk_energy": chunk_energy,
        "cumulative_energy": np.cumsum(chunk_energy, axis=0),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    arrays.update(extra_arrays or {})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def _summarize_test2(output_dir: pathlib.Path) -> dict[str, object]:
    shuffled_paths = sorted((output_dir / "test2_raw").glob("*.npz"))
    baseline: list[float] = []
    shuffled: list[float] = []
    demos: list[dict[str, object]] = []
    for shuffled_path in shuffled_paths:
        shuffled_arrays, metadata = _load_result(shuffled_path)
        baseline_path = output_dir / "test1_raw" / shuffled_path.name
        baseline_arrays, _ = _load_result(baseline_path)
        if not np.array_equal(baseline_arrays["chunk_starts"], shuffled_arrays["chunk_starts"]):
            raise ValueError(f"Baseline/shuffle chunk mismatch for {metadata['demo_id']}")
        baseline_mean = float(np.mean(baseline_arrays["chunk_energy"][:, 0]))
        shuffled_mean = float(np.mean(shuffled_arrays["chunk_energy"][:, 0]))
        baseline.append(baseline_mean)
        shuffled.append(shuffled_mean)
        demos.append(
            {
                "demo_id": metadata["demo_id"],
                "chunk_count": len(shuffled_arrays["chunk_starts"]),
                "baseline_mean_residual": baseline_mean,
                "shuffled_mean_residual": shuffled_mean,
                "shuffled_minus_baseline": shuffled_mean - baseline_mean,
            }
        )
    if len(shuffled_paths) < 2:
        raise ValueError(f"Need at least two completed test-2 results below {output_dir / 'test2_raw'}")
    test = paired_lower_test(np.asarray(baseline), np.asarray(shuffled))
    return {
        "test": 2,
        "hypothesis": "chronological action residual < globally time-shuffled action residual",
        "unit_of_analysis": "demo (mean across native stride-1 chunks)",
        "shuffle_definition": "one deterministic permutation of the complete raw action time axis per demo; observations remain chronological",
        "alpha": 0.05,
        "result": dataclasses.asdict(test),
        "demos": demos,
    }


def score_test2(args: argparse.Namespace) -> None:
    _require_test1_passed(args.output_dir)
    spatial = {task.task_id: task for task in bddl.load_task_specs(args.bddl_root, "libero_spatial")}
    all_demos = [demo for demo in libero_dataset.discover_demos(args.demo_root) if demo.task_id in spatial]
    demos = _pilot_demos(all_demos, args.max_tasks)
    config = scorer_lib.ResidualConfig(
        flow_timesteps=tuple(args.flow_timesteps),
        noise_samples=args.noise_samples,
        seed=args.seed,
        eval_batch_size=args.eval_batch_size,
    )
    scorer = scorer_lib.Pi05InstructionLikelihoodScorer(args.checkpoint, config=config)
    result_dir = args.output_dir / "test2_raw"
    for demo_index, demo in enumerate(demos):
        output_path = result_dir / _result_name(demo)
        if output_path.exists() and not args.overwrite:
            print(f"resume: {output_path}", file=sys.stderr)
            continue
        chunks = list(
            libero_dataset.iter_action_chunks(
                demo, action_horizon=scorer.action_horizon, stride=1, rotate_images_180=True
            )
        )
        actions = libero_dataset.load_demo_actions(demo)
        permutation_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 2, demo_index]))
        permutation = permutation_rng.permutation(len(actions))
        shuffled_actions = actions[permutation]
        shuffled_chunks = [
            dataclasses.replace(chunk, actions=shuffled_actions[chunk.start : chunk.stop]) for chunk in chunks
        ]
        task = spatial[demo.task_id]
        _validate_true_prompt(demo, task)
        scores = [
            scorer.score_chunk(chunk, [task.language], chunk_index=demo_index * 100_000 + index)
            for index, chunk in enumerate(shuffled_chunks)
        ]
        _save_single_condition(
            output_path,
            demo=demo,
            condition="temporal_shuffle",
            instruction=task.language,
            chunks=shuffled_chunks,
            chunk_scores=scores,
            config=config,
            checkpoint=args.checkpoint,
            extra_arrays={"source_action_indices": permutation.astype(np.int32)},
            extra_metadata={"action_permutation_seed_components": [args.seed, 2, demo_index]},
        )
        print(f"saved {output_path}", file=sys.stderr)
    summary = _summarize_test2(args.output_dir)
    _write_json(args.output_dir / "test2_summary.json", summary)
    print(json.dumps(summary["result"], sort_keys=True))


def _require_test2_passed(output_dir: pathlib.Path) -> None:
    summary_path = output_dir / "test2_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Test-2 gate is closed: missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("result", {}).get("passed", False):
        raise SystemExit(f"Test-2 gate is closed: {summary_path} does not record passed=true")


def _summarize_test3(output_dir: pathlib.Path) -> dict[str, object]:
    random_paths = sorted((output_dir / "test3_raw").glob("*.npz"))
    baseline: list[float] = []
    shuffled: list[float] = []
    random_values: list[float] = []
    demos: list[dict[str, object]] = []
    for random_path in random_paths:
        random_arrays, metadata = _load_result(random_path)
        baseline_arrays, _ = _load_result(output_dir / "test1_raw" / random_path.name)
        shuffled_arrays, _ = _load_result(output_dir / "test2_raw" / random_path.name)
        for other, label in ((baseline_arrays, "baseline"), (shuffled_arrays, "shuffle")):
            if not np.array_equal(other["chunk_starts"], random_arrays["chunk_starts"]):
                raise ValueError(f"{label}/random chunk mismatch for {metadata['demo_id']}")
        baseline_mean = float(np.mean(baseline_arrays["chunk_energy"][:, 0]))
        shuffled_mean = float(np.mean(shuffled_arrays["chunk_energy"][:, 0]))
        random_mean = float(np.mean(random_arrays["chunk_energy"][:, 0]))
        baseline.append(baseline_mean)
        shuffled.append(shuffled_mean)
        random_values.append(random_mean)
        demos.append(
            {
                "demo_id": metadata["demo_id"],
                "baseline_mean_residual": baseline_mean,
                "shuffled_mean_residual": shuffled_mean,
                "random_mean_residual": random_mean,
                "random_minus_shuffled": random_mean - shuffled_mean,
                "random_minus_baseline": random_mean - baseline_mean,
            }
        )
    if len(random_paths) < 2:
        raise ValueError(f"Need at least two completed test-3 results below {output_dir / 'test3_raw'}")
    baseline_array = np.asarray(baseline)
    shuffled_array = np.asarray(shuffled)
    random_array = np.asarray(random_values)
    random_vs_baseline = paired_lower_test(baseline_array, random_array)
    random_vs_shuffle = paired_lower_test(shuffled_array, random_array)
    passed = bool(random_vs_baseline.passed and random_vs_shuffle.passed)
    return {
        "test": 3,
        "hypothesis": "Gaussian random action residual is higher than both chronological and shuffled actions",
        "unit_of_analysis": "demo (mean across native stride-1 chunks)",
        "random_definition": "N(0,1) in checkpoint quantile-normalized physical 7-D action space, inverse-transformed to raw actions before the official input pipeline",
        "alpha": 0.05,
        "result": {
            "count": len(random_array),
            "baseline_mean": float(np.mean(baseline_array)),
            "shuffled_mean": float(np.mean(shuffled_array)),
            "random_mean": float(np.mean(random_array)),
            "random_vs_baseline": dataclasses.asdict(random_vs_baseline),
            "random_vs_shuffle": dataclasses.asdict(random_vs_shuffle),
            "passed": passed,
        },
        "demos": demos,
    }


def score_test3(args: argparse.Namespace) -> None:
    _require_test1_passed(args.output_dir)
    _require_test2_passed(args.output_dir)
    spatial = {task.task_id: task for task in bddl.load_task_specs(args.bddl_root, "libero_spatial")}
    all_demos = [demo for demo in libero_dataset.discover_demos(args.demo_root) if demo.task_id in spatial]
    demos = _pilot_demos(all_demos, args.max_tasks)
    config = scorer_lib.ResidualConfig(
        flow_timesteps=tuple(args.flow_timesteps),
        noise_samples=args.noise_samples,
        seed=args.seed,
        eval_batch_size=args.eval_batch_size,
    )
    scorer = scorer_lib.Pi05InstructionLikelihoodScorer(args.checkpoint, config=config)
    result_dir = args.output_dir / "test3_raw"
    for demo_index, demo in enumerate(demos):
        output_path = result_dir / _result_name(demo)
        if output_path.exists() and not args.overwrite:
            print(f"resume: {output_path}", file=sys.stderr)
            continue
        chunks = list(
            libero_dataset.iter_action_chunks(
                demo, action_horizon=scorer.action_horizon, stride=1, rotate_images_180=True
            )
        )
        action_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 3, demo_index]))
        normalized_random_actions = action_rng.standard_normal((demo.length, 7), dtype=np.float32)
        raw_random_actions = scorer.normalized_to_raw_physical_actions(normalized_random_actions)
        random_chunks = [
            dataclasses.replace(chunk, actions=raw_random_actions[chunk.start : chunk.stop]) for chunk in chunks
        ]
        task = spatial[demo.task_id]
        _validate_true_prompt(demo, task)
        scores = [
            scorer.score_chunk(chunk, [task.language], chunk_index=demo_index * 100_000 + index)
            for index, chunk in enumerate(random_chunks)
        ]
        _save_single_condition(
            output_path,
            demo=demo,
            condition="gaussian_random_actions",
            instruction=task.language,
            chunks=random_chunks,
            chunk_scores=scores,
            config=config,
            checkpoint=args.checkpoint,
            extra_arrays={
                "generated_normalized_action_sequence": normalized_random_actions,
                "generated_raw_action_sequence": raw_random_actions,
            },
            extra_metadata={"action_rng_seed_components": [args.seed, 3, demo_index]},
        )
        print(f"saved {output_path}", file=sys.stderr)
    summary = _summarize_test3(args.output_dir)
    _write_json(args.output_dir / "test3_summary.json", summary)
    print(json.dumps(summary["result"], sort_keys=True))


def _require_test3_passed(output_dir: pathlib.Path) -> None:
    summary_path = output_dir / "test3_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Test-3 gate is closed: missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("result", {}).get("passed", False):
        raise SystemExit(f"Test-3 gate is closed: {summary_path} does not record passed=true")


def _summarize_test4(output_dir: pathlib.Path) -> dict[str, object]:
    candidate_paths = sorted((output_dir / "candidate_raw").glob("*.npz"))
    demos: list[dict[str, object]] = []
    rhos: list[float] = []
    endpoint_gains: list[float] = []
    final_correct: list[bool] = []
    for path in candidate_paths:
        arrays, metadata = _load_result(path)
        true_probability = np.asarray(arrays["normalized_scores"][:, 0], dtype=np.float64)
        # Include the uniform prior at prefix length zero in the trend analysis.
        prefix_positions = np.concatenate([[0], arrays["chunk_stops"]])
        probability_with_prior = np.concatenate([[0.5], true_probability])
        rho = float(stats.spearmanr(prefix_positions, probability_with_prior).statistic)
        endpoint_gain = float(true_probability[-1] - 0.5)
        correct = bool(np.argmax(arrays["normalized_scores"][-1]) == 0)
        rhos.append(rho)
        endpoint_gains.append(endpoint_gain)
        final_correct.append(correct)
        demos.append(
            {
                "demo_id": metadata["demo_id"],
                "true_instruction": metadata["candidate_instructions"][0],
                "alternate_instruction": metadata["candidate_instructions"][1],
                "prefix_count": len(true_probability),
                "spearman_rho": rho,
                "prior_probability": 0.5,
                "first_prefix_probability": float(true_probability[0]),
                "final_probability": float(true_probability[-1]),
                "final_minus_prior": endpoint_gain,
                "final_top1_correct": correct,
            }
        )
    if not candidate_paths:
        raise ValueError(f"No candidate results below {output_dir / 'candidate_raw'}")
    median_rho = float(np.median(rhos))
    median_endpoint_gain = float(np.median(endpoint_gains))
    positive_rho_fraction = float(np.mean(np.asarray(rhos) > 0.0))
    positive_endpoint_fraction = float(np.mean(np.asarray(endpoint_gains) > 0.0))
    trend_observed = bool(
        median_rho > 0.0
        and median_endpoint_gain > 0.0
        and positive_rho_fraction >= 0.6
        and positive_endpoint_fraction >= 0.6
    )
    return {
        "test": 4,
        "hypothesis": "true-instruction posterior generally rises with observed prefix length",
        "candidate_prior": "uniform over the two BDDL-validated executable Spatial instructions",
        "result": {
            "count": len(candidate_paths),
            "median_spearman_rho": median_rho,
            "mean_spearman_rho": float(np.mean(rhos)),
            "positive_rho_fraction": positive_rho_fraction,
            "median_final_minus_prior": median_endpoint_gain,
            "mean_final_minus_prior": float(np.mean(endpoint_gains)),
            "positive_endpoint_fraction": positive_endpoint_fraction,
            "final_top1_accuracy": float(np.mean(final_correct)),
            "trend_observed": trend_observed,
        },
        "demos": demos,
    }


def score_test4(args: argparse.Namespace) -> None:
    _require_test1_passed(args.output_dir)
    _require_test2_passed(args.output_dir)
    _require_test3_passed(args.output_dir)
    spatial_tasks = bddl.load_task_specs(args.bddl_root, "libero_spatial")
    spatial = {task.task_id: task for task in spatial_tasks}
    candidate_sets = bddl.build_candidate_sets(spatial_tasks)
    all_demos = [demo for demo in libero_dataset.discover_demos(args.demo_root) if demo.task_id in spatial]
    demos = _pilot_demos(all_demos, args.max_tasks)
    config = scorer_lib.ResidualConfig(
        flow_timesteps=tuple(args.flow_timesteps),
        noise_samples=args.noise_samples,
        seed=args.seed,
        temperature=args.temperature,
        eval_batch_size=args.eval_batch_size,
    )
    scorer = scorer_lib.Pi05InstructionLikelihoodScorer(args.checkpoint, config=config)
    result_dir = args.output_dir / "candidate_raw"
    for demo_index, demo in enumerate(demos):
        output_path = result_dir / _result_name(demo)
        if output_path.exists() and not args.overwrite:
            print(f"resume: {output_path}", file=sys.stderr)
            continue
        task = spatial[demo.task_id]
        _validate_true_prompt(demo, task)
        candidate_set = candidate_sets[demo.task_id]
        alternate_ids = [task_id for task_id in candidate_set.executable_task_ids if task_id != demo.task_id]
        if len(alternate_ids) != 1:
            raise ValueError(f"Expected exactly one alternate for {demo.task_id}, got {alternate_ids}")
        alternate = spatial[alternate_ids[0]]
        chunks = list(
            libero_dataset.iter_action_chunks(
                demo, action_horizon=scorer.action_horizon, stride=1, rotate_images_180=True
            )
        )
        baseline_arrays, baseline_metadata = _load_result(args.output_dir / "test1_raw" / _result_name(demo))
        alternate_scores = [
            scorer.score_chunk(chunk, [alternate.language], chunk_index=demo_index * 100_000 + index)
            for index, chunk in enumerate(chunks)
        ]
        alternate_residual = np.stack([score.residual for score in alternate_scores])
        alternate_noise = np.stack([score.noise for score in alternate_scores])
        alternate_actions = np.stack([score.normalized_actions for score in alternate_scores])
        if not np.array_equal(alternate_noise, baseline_arrays["noise"]):
            raise AssertionError(f"Flow noise does not match test-1 baseline for {demo.demo_id}")
        if not np.array_equal(alternate_actions, baseline_arrays["normalized_actions"]):
            raise AssertionError(f"Normalized actions do not match test-1 baseline for {demo.demo_id}")
        residual = np.concatenate([baseline_arrays["residual"][:, :1], alternate_residual], axis=1)
        chunk_energy = scorer_lib.aggregate_residuals(residual)
        normalized_scores = scorer_lib.cumulative_posteriors(chunk_energy, np.ones(2, dtype=bool), config.temperature)
        metadata = {
            "schema_version": 1,
            "sanity_tests": [4, 5],
            "demo_id": demo.demo_id,
            "hdf5_path": str(demo.path.resolve()),
            "episode": demo.episode,
            "trajectory_length": demo.length,
            "candidate_task_ids": [task.task_id, alternate.task_id],
            "candidate_instructions": [task.language, alternate.language],
            "candidate_roles": ["true", "physical_same_scene_alternate"],
            "candidate_validation": "BDDL :init placement predicate and required goal entity types",
            "flow_timesteps": list(config.flow_timesteps),
            "noise_samples": config.noise_samples,
            "flow_noise_seed": config.seed,
            "softmax_temperature": config.temperature,
            "instruction_prior": "uniform over two physically executable candidates",
            "residual_definition": baseline_metadata["residual_definition"],
            "action_normalization": baseline_metadata["action_normalization"],
            "chunk_stride": 1,
            "checkpoint": str(args.checkpoint.resolve()),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            residual=residual,
            normalized_actions=baseline_arrays["normalized_actions"],
            noise=baseline_arrays["noise"],
            flow_timesteps=baseline_arrays["flow_timesteps"],
            chunk_starts=baseline_arrays["chunk_starts"],
            chunk_stops=baseline_arrays["chunk_stops"],
            chunk_energy=chunk_energy,
            cumulative_energy=np.cumsum(chunk_energy, axis=0),
            normalized_scores=normalized_scores,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        print(f"saved {output_path}", file=sys.stderr)
    summary = _summarize_test4(args.output_dir)
    _write_json(args.output_dir / "test4_summary.json", summary)
    print(json.dumps(summary["result"], sort_keys=True))


def _require_test4_results(output_dir: pathlib.Path) -> None:
    summary_path = output_dir / "test4_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Test-5 input is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("result", {}).get("count", 0) == 0:
        raise SystemExit(f"Test-5 input contains no demos: {summary_path}")


def _flow_subset_indices(available: np.ndarray, requested: tuple[float, ...]) -> np.ndarray:
    available = np.asarray(available, dtype=np.float64)
    indices: list[int] = []
    for timestep in requested:
        matches = np.flatnonzero(np.isclose(available, timestep, rtol=0.0, atol=1e-6))
        if len(matches) != 1:
            raise ValueError(f"Requested flow timestep {timestep} is not uniquely present in {available.tolist()}")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int32)


def _test5_scenarios(candidate_paths: list[pathlib.Path]) -> dict[str, object]:
    timestep_sets: dict[str, tuple[float, ...]] = {
        "all": (0.1, 0.3, 0.5, 0.7, 0.9),
        "low": (0.1, 0.3),
        "middle": (0.3, 0.5, 0.7),
        "high": (0.7, 0.9),
        "spread": (0.1, 0.5, 0.9),
    }
    temperatures = (0.5, 1.0, 2.0)
    per_scenario: dict[str, dict[str, object]] = {}
    per_demo: list[dict[str, object]] = []
    loaded: list[tuple[pathlib.Path, dict[str, np.ndarray], dict[str, object]]] = []
    for path in candidate_paths:
        arrays, metadata = _load_result(path)
        residual = arrays.get("residual")
        if residual is None or residual.ndim != 6 or residual.shape[1] != 2:
            raise ValueError(
                f"Expected [prefix,2,flow,noise,step,dim] residuals in {path}, got {None if residual is None else residual.shape}"
            )
        loaded.append((path, arrays, metadata))

    for set_name, requested_timesteps in timestep_sets.items():
        for temperature in temperatures:
            scenario_id = f"{set_name}__temperature_{temperature:g}"
            correct: list[bool] = []
            true_probabilities: list[float] = []
            margins: list[float] = []
            for path, arrays, metadata in loaded:
                indices = _flow_subset_indices(arrays["flow_timesteps"], requested_timesteps)
                chunk_energy = scorer_lib.aggregate_residuals(arrays["residual"][:, :, indices])
                posterior = scorer_lib.cumulative_posteriors(
                    chunk_energy, np.ones(chunk_energy.shape[1], dtype=bool), temperature
                )
                cumulative = np.sum(chunk_energy, axis=0)
                is_correct = bool(np.argmin(cumulative) == 0)
                energy_margin = float(cumulative[1] - cumulative[0])
                final_true_probability = float(posterior[-1, 0])
                correct.append(is_correct)
                margins.append(energy_margin)
                true_probabilities.append(final_true_probability)
                per_demo.append(
                    {
                        "scenario": scenario_id,
                        "demo_id": metadata["demo_id"],
                        "source_file": str(path),
                        "final_top1_correct": is_correct,
                        "final_true_probability": final_true_probability,
                        "alternate_minus_true_cumulative_energy": energy_margin,
                    }
                )
            per_scenario[scenario_id] = {
                "flow_timestep_set": list(requested_timesteps),
                "softmax_temperature": temperature,
                "count": len(correct),
                "final_top1_accuracy": float(np.mean(correct)),
                "mean_final_true_probability": float(np.mean(true_probabilities)),
                "median_final_true_probability": float(np.median(true_probabilities)),
                "mean_alternate_minus_true_cumulative_energy": float(np.mean(margins)),
                "median_alternate_minus_true_cumulative_energy": float(np.median(margins)),
            }

    baseline_id = "all__temperature_1"
    baseline_accuracy = float(per_scenario[baseline_id]["final_top1_accuracy"])
    accuracies = [float(scenario["final_top1_accuracy"]) for scenario in per_scenario.values()]
    minimum_accuracy = min(accuracies)
    accuracy_ratio = minimum_accuracy / baseline_accuracy if baseline_accuracy > 0.0 else 0.0
    # "One order of magnitude" is operationalized literally: the worst setting
    # must retain at least 10% of baseline top-1 accuracy.  Absolute accuracy and
    # drop are also reported so this permissive threshold cannot hide instability.
    no_order_magnitude_collapse = bool(baseline_accuracy > 0.0 and accuracy_ratio >= 0.1)
    return {
        "test": 5,
        "hypothesis": "final-prefix top-1 accuracy does not collapse by an order of magnitude under timestep and temperature changes",
        "source": "candidate_raw residual tensors from sanity tests 4/5; no additional model forwards",
        "baseline_scenario": baseline_id,
        "order_magnitude_retention_threshold": 0.1,
        "result": {
            "count": len(candidate_paths),
            "scenario_count": len(per_scenario),
            "baseline_top1_accuracy": baseline_accuracy,
            "minimum_top1_accuracy": minimum_accuracy,
            "maximum_absolute_accuracy_drop": baseline_accuracy - minimum_accuracy,
            "minimum_to_baseline_accuracy_ratio": accuracy_ratio,
            "passed": no_order_magnitude_collapse,
        },
        "scenarios": per_scenario,
        "demos": per_demo,
    }


def _write_sanity_summary(output_dir: pathlib.Path) -> None:
    summaries = {
        test: json.loads((output_dir / f"test{test}_summary.json").read_text(encoding="utf-8")) for test in range(1, 6)
    }
    results = {test: summary["result"] for test, summary in summaries.items()}
    rows = [
        {
            "test": 1,
            "check": "paired vs cross-scene mismatch",
            "n": results[1]["count"],
            "primary_value": f"matched={results[1]['left_mean']:.6f}; mismatch={results[1]['right_mean']:.6f}",
            "effect": f"median_delta={results[1]['median_difference']:.6f}",
            "pvalue": f"{results[1]['pvalue']:.9f}",
            "criterion": "one-sided paired Wilcoxon p<0.05 and matched residual lower",
            "passed": results[1]["passed"],
        },
        {
            "test": 2,
            "check": "temporal shuffle",
            "n": results[2]["count"],
            "primary_value": f"chronological={results[2]['left_mean']:.6f}; shuffled={results[2]['right_mean']:.6f}",
            "effect": f"median_delta={results[2]['median_difference']:.6f}",
            "pvalue": f"{results[2]['pvalue']:.9f}",
            "criterion": "one-sided paired Wilcoxon p<0.05 and shuffled residual higher",
            "passed": results[2]["passed"],
        },
        {
            "test": 3,
            "check": "Gaussian random actions",
            "n": results[3]["count"],
            "primary_value": (
                f"chronological={results[3]['baseline_mean']:.6f}; shuffled={results[3]['shuffled_mean']:.6f}; "
                f"random={results[3]['random_mean']:.6f}"
            ),
            "effect": "random higher than chronological and shuffled",
            "pvalue": (
                f"vs_chronological={results[3]['random_vs_baseline']['pvalue']:.9f}; "
                f"vs_shuffled={results[3]['random_vs_shuffle']['pvalue']:.9f}"
            ),
            "criterion": "both one-sided paired Wilcoxon tests p<0.05",
            "passed": results[3]["passed"],
        },
        {
            "test": 4,
            "check": "true-instruction posterior prefix trend",
            "n": results[4]["count"],
            "primary_value": (
                f"median_rho={results[4]['median_spearman_rho']:.6f}; "
                f"positive_rho_fraction={results[4]['positive_rho_fraction']:.3f}"
            ),
            "effect": (
                f"median_final_minus_prior={results[4]['median_final_minus_prior']:.6f}; "
                f"final_top1={results[4]['final_top1_accuracy']:.3f}"
            ),
            "pvalue": "NA (descriptive trend; significance not required)",
            "criterion": "median rho/gain positive and >=60% demos positive",
            "passed": results[4]["trend_observed"],
        },
        {
            "test": 5,
            "check": "flow timestep and softmax temperature sensitivity",
            "n": results[5]["count"],
            "primary_value": (
                f"baseline_top1={results[5]['baseline_top1_accuracy']:.3f}; "
                f"minimum_top1={results[5]['minimum_top1_accuracy']:.3f}"
            ),
            "effect": f"minimum_to_baseline_ratio={results[5]['minimum_to_baseline_accuracy_ratio']:.3f}",
            "pvalue": "NA (stability check)",
            "criterion": "worst top-1 retains >=10% of baseline (literal one-order threshold)",
            "passed": results[5]["passed"],
        },
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (output_dir / "sanity_summary.csv").write_text(stream.getvalue(), encoding="utf-8")
    _write_json(
        output_dir / "sanity_summary.json",
        {
            "all_five_passed": all(bool(row["passed"]) for row in rows),
            "tests_1_to_3_significant": all(bool(results[test]["passed"]) for test in (1, 2, 3)),
            "rows": rows,
        },
    )


def summarize_test5(args: argparse.Namespace) -> None:
    _require_test1_passed(args.output_dir)
    _require_test2_passed(args.output_dir)
    _require_test3_passed(args.output_dir)
    _require_test4_results(args.output_dir)
    candidate_paths = sorted((args.output_dir / "candidate_raw").glob("*.npz"))
    if not candidate_paths:
        raise SystemExit(f"No candidate residual files below {args.output_dir / 'candidate_raw'}")
    summary = _test5_scenarios(candidate_paths)
    _write_json(args.output_dir / "test5_summary.json", summary)
    _write_sanity_summary(args.output_dir)
    print(json.dumps(summary["result"], sort_keys=True))


def audit_results(args: argparse.Namespace) -> None:
    """Fail closed unless every saved pilot tensor satisfies the documented schema."""
    summary_path = args.output_dir / "sanity_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing combined summary: {summary_path}")
    combined = json.loads(summary_path.read_text(encoding="utf-8"))
    if not combined.get("all_five_passed", False) or not combined.get("tests_1_to_3_significant", False):
        raise ValueError("Combined summary does not record all five checks and tests 1-3 as passed")
    cache_path = args.output_dir / "cache_validation.json"
    if not cache_path.exists() or not json.loads(cache_path.read_text(encoding="utf-8")).get("passed", False):
        raise ValueError("Missing or failed native-full-versus-prefix-cache implementation validation")

    expected_candidates = {"test1_raw": 2, "test2_raw": 1, "test3_raw": 1, "candidate_raw": 2}
    expected_count = int(json.loads((args.output_dir / "test1_summary.json").read_text())["result"]["count"])
    directory_records: dict[str, dict[str, tuple[dict[str, np.ndarray], dict[str, object]]]] = {}
    tensor_count = 0
    residual_scalar_count = 0
    for directory, candidate_count in expected_candidates.items():
        paths = sorted((args.output_dir / directory).glob("*.npz"))
        if len(paths) != expected_count:
            raise ValueError(f"Expected {expected_count} files in {directory}, found {len(paths)}")
        records: dict[str, tuple[dict[str, np.ndarray], dict[str, object]]] = {}
        for path in paths:
            arrays, metadata = _load_result(path)
            trajectory_length = int(metadata["trajectory_length"])
            prefix_count = trajectory_length - 10 + 1
            flow_count = len(arrays["flow_timesteps"])
            noise_samples = int(metadata["noise_samples"])
            expected_residual_shape = (prefix_count, candidate_count, flow_count, noise_samples, 10, 7)
            if arrays["residual"].shape != expected_residual_shape:
                raise ValueError(f"{path}: residual {arrays['residual'].shape} != {expected_residual_shape}")
            if arrays["normalized_actions"].shape != (prefix_count, 10, 32):
                raise ValueError(f"{path}: unexpected normalized action shape {arrays['normalized_actions'].shape}")
            if arrays["noise"].shape != (prefix_count, flow_count, noise_samples, 10, 32):
                raise ValueError(f"{path}: unexpected noise shape {arrays['noise'].shape}")
            if not all(np.all(np.isfinite(arrays[key])) for key in ("residual", "normalized_actions", "noise")):
                raise ValueError(f"{path}: non-finite raw tensor")
            np.testing.assert_array_equal(arrays["chunk_starts"], np.arange(prefix_count))
            np.testing.assert_array_equal(arrays["chunk_stops"], np.arange(prefix_count) + 10)
            np.testing.assert_allclose(
                arrays["chunk_energy"], scorer_lib.aggregate_residuals(arrays["residual"]), rtol=1e-12, atol=1e-12
            )
            if "normalized_scores" in arrays:
                if arrays["normalized_scores"].shape != (prefix_count, candidate_count):
                    raise ValueError(f"{path}: unexpected normalized score shape")
                np.testing.assert_allclose(np.sum(arrays["normalized_scores"], axis=1), 1.0, atol=1e-12)
            records[str(metadata["demo_id"])] = (arrays, metadata)
            tensor_count += 1
            residual_scalar_count += int(arrays["residual"].size)
        directory_records[directory] = records

    demo_ids = set(directory_records["test1_raw"])
    if any(set(records) != demo_ids for records in directory_records.values()):
        raise ValueError("Raw-result directories do not contain the same demo IDs")
    for demo_id in sorted(demo_ids):
        baseline_arrays = directory_records["test1_raw"][demo_id][0]
        candidate_arrays = directory_records["candidate_raw"][demo_id][0]
        np.testing.assert_array_equal(candidate_arrays["residual"][:, 0], baseline_arrays["residual"][:, 0])
        np.testing.assert_array_equal(candidate_arrays["normalized_actions"], baseline_arrays["normalized_actions"])
        for directory in ("test2_raw", "test3_raw", "candidate_raw"):
            np.testing.assert_array_equal(directory_records[directory][demo_id][0]["noise"], baseline_arrays["noise"])

    inventory = json.loads((args.output_dir / "libero_spatial_inventory.json").read_text(encoding="utf-8"))
    if inventory["task_count"] != expected_count:
        raise ValueError("Pilot/raw count no longer covers exactly one demo from every inventoried Spatial task")
    if any(len(task["executable_task_ids"]) != 2 for task in inventory["tasks"]):
        raise ValueError("Inventory contains a layout without exactly two BDDL-grounded executable candidates")
    if any(task["cross_scene_mismatch"]["suite"] != "libero_object" for task in inventory["tasks"]):
        raise ValueError("Inventory contains a mismatch outside LIBERO-Object")

    payload = {
        "passed": True,
        "demo_count": len(demo_ids),
        "raw_npz_count": tensor_count,
        "raw_residual_scalar_count": residual_scalar_count,
        "checks": [
            "all five sanity checks passed and tests 1-3 are significant",
            "native full forward and external prefix-cache hook agree within recorded bfloat16 tolerance",
            "one demo from each of all 10 inventoried Spatial tasks",
            "all raw residual/action/noise tensors finite and schema-valid",
            "native stride-1 10-step chunk boundaries cover every complete prefix",
            "saved chunk energies exactly reproduce from raw 7-D residual tensors",
            "candidate true residual/actions and all common flow noise exactly match the test-1 baseline",
            "every layout has exactly two BDDL-grounded candidates and mismatch suite is LIBERO-Object",
        ],
    }
    _write_json(args.output_dir / "artifact_audit.json", payload)
    print(json.dumps(payload, sort_keys=True))


def validate_cache(args: argparse.Namespace) -> None:
    spatial = {task.task_id: task for task in bddl.load_task_specs(args.bddl_root, "libero_spatial")}
    demos = [demo for demo in libero_dataset.discover_demos(args.demo_root) if demo.task_id in spatial]
    scorer = scorer_lib.Pi05InstructionLikelihoodScorer(args.checkpoint)
    if demos:
        demo = sorted(demos, key=lambda item: (item.task_id, item.episode))[0]
        task = spatial[demo.task_id]
        _validate_true_prompt(demo, task)
        chunk = next(
            libero_dataset.iter_action_chunks(
                demo, action_horizon=scorer.action_horizon, stride=1, rotate_images_180=True
            )
        )
        demo_id = demo.demo_id
        instruction = task.language
        input_source = "real first chunk from the local LIBERO HDF5"
    else:
        baseline_paths = sorted((args.result_dir / "test1_raw").glob("*.npz"))
        if not baseline_paths:
            raise SystemExit("No Spatial HDF5 or saved test-1 action tensor is available for cache validation")
        arrays, metadata = _load_result(baseline_paths[0])
        raw_actions = scorer.normalized_to_raw_physical_actions(arrays["normalized_actions"][0, :, :7])
        chunk = libero_dataset.ActionChunk(
            start=0,
            stop=scorer.action_horizon,
            base_image=np.zeros((128, 128, 3), dtype=np.uint8),
            wrist_image=np.zeros((128, 128, 3), dtype=np.uint8),
            state=np.zeros(8, dtype=np.float32),
            actions=raw_actions,
        )
        demo_id = str(metadata["demo_id"])
        instruction = str(metadata["candidate_instructions"][0])
        input_source = (
            "saved real checkpoint-normalized demonstrated action chunk inverse-transformed to raw; "
            "deterministic zero observation used only for cached/full implementation equivalence"
        )
    validation = scorer.validate_prefix_cache(chunk, instruction)
    payload = {
        "passed": validation.allclose,
        "comparison": dataclasses.asdict(validation),
        "demo_id": demo_id,
        "instruction": instruction,
        "input_source": input_source,
        "chunk_start": chunk.start,
        "chunk_stop": chunk.stop,
        "checkpoint": str(args.checkpoint.resolve()),
        "model_action_horizon": scorer.action_horizon,
        "model_action_dim": scorer.model_action_dim,
        "purpose": "external prefix KV-cache hook versus unmodified native full velocity-residual forward",
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    if not validation.allclose:
        raise SystemExit("Prefix-cache validation failed")


def summarize_test1(args: argparse.Namespace) -> None:
    summary = _summarize_test1(args.result_dir)
    _write_json(args.output, summary)
    print(json.dumps(summary["result"], sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--bddl-root", type=pathlib.Path, default=DEFAULT_BDDL_ROOT)
    inventory_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    inventory_parser.set_defaults(func=inventory)

    inspect_parser = subparsers.add_parser("inspect-demos")
    inspect_parser.add_argument("--demo-root", type=pathlib.Path, required=True)
    inspect_parser.add_argument("--output", type=pathlib.Path, required=True)
    inspect_parser.set_defaults(func=inspect_demos)

    score_parser = subparsers.add_parser("score-test1")
    score_parser.add_argument("--demo-root", type=pathlib.Path, required=True)
    score_parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    score_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    score_parser.add_argument("--bddl-root", type=pathlib.Path, default=DEFAULT_BDDL_ROOT)
    score_parser.add_argument("--flow-timesteps", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    score_parser.add_argument("--noise-samples", type=int, default=1)
    score_parser.add_argument("--eval-batch-size", type=int, default=1)
    score_parser.add_argument("--seed", type=int, default=0)
    score_parser.add_argument("--max-tasks", type=int, default=10, choices=range(6, 11))
    score_parser.add_argument("--overwrite", action="store_true")
    score_parser.set_defaults(func=score_test1)

    summary_parser = subparsers.add_parser("summarize-test1")
    summary_parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    summary_parser.add_argument("--output", type=pathlib.Path, required=True)
    summary_parser.set_defaults(func=summarize_test1)

    test2_parser = subparsers.add_parser("score-test2")
    test2_parser.add_argument("--demo-root", type=pathlib.Path, required=True)
    test2_parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    test2_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    test2_parser.add_argument("--bddl-root", type=pathlib.Path, default=DEFAULT_BDDL_ROOT)
    test2_parser.add_argument("--flow-timesteps", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    test2_parser.add_argument("--noise-samples", type=int, default=1)
    test2_parser.add_argument("--eval-batch-size", type=int, default=1)
    test2_parser.add_argument("--seed", type=int, default=0)
    test2_parser.add_argument("--max-tasks", type=int, default=10, choices=range(6, 11))
    test2_parser.add_argument("--overwrite", action="store_true")
    test2_parser.set_defaults(func=score_test2)

    test3_parser = subparsers.add_parser("score-test3")
    test3_parser.add_argument("--demo-root", type=pathlib.Path, required=True)
    test3_parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    test3_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    test3_parser.add_argument("--bddl-root", type=pathlib.Path, default=DEFAULT_BDDL_ROOT)
    test3_parser.add_argument("--flow-timesteps", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    test3_parser.add_argument("--noise-samples", type=int, default=1)
    test3_parser.add_argument("--eval-batch-size", type=int, default=1)
    test3_parser.add_argument("--seed", type=int, default=0)
    test3_parser.add_argument("--max-tasks", type=int, default=10, choices=range(6, 11))
    test3_parser.add_argument("--overwrite", action="store_true")
    test3_parser.set_defaults(func=score_test3)

    test4_parser = subparsers.add_parser("score-test4")
    test4_parser.add_argument("--demo-root", type=pathlib.Path, required=True)
    test4_parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    test4_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    test4_parser.add_argument("--bddl-root", type=pathlib.Path, default=DEFAULT_BDDL_ROOT)
    test4_parser.add_argument("--flow-timesteps", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    test4_parser.add_argument("--noise-samples", type=int, default=1)
    test4_parser.add_argument("--eval-batch-size", type=int, default=1)
    test4_parser.add_argument("--temperature", type=float, default=1.0)
    test4_parser.add_argument("--seed", type=int, default=0)
    test4_parser.add_argument("--max-tasks", type=int, default=10, choices=range(6, 11))
    test4_parser.add_argument("--overwrite", action="store_true")
    test4_parser.set_defaults(func=score_test4)

    test5_parser = subparsers.add_parser("summarize-test5")
    test5_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    test5_parser.set_defaults(func=summarize_test5)

    audit_parser = subparsers.add_parser("audit-results")
    audit_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    audit_parser.set_defaults(func=audit_results)

    cache_parser = subparsers.add_parser("validate-cache")
    cache_parser.add_argument("--demo-root", type=pathlib.Path, required=True)
    cache_parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    cache_parser.add_argument("--output", type=pathlib.Path, required=True)
    cache_parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    cache_parser.add_argument("--bddl-root", type=pathlib.Path, default=DEFAULT_BDDL_ROOT)
    cache_parser.set_defaults(func=validate_cache)
    return parser


if __name__ == "__main__":
    parsed_args = _parser().parse_args()
    parsed_args.func(parsed_args)
