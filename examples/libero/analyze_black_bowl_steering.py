"""Aggregate a formal paired two-instruction LIBERO steering evaluation.

The unit of pairing is (target instruction, official init state, sampling
replicate). All methods must use the same initial observation and query-level
sampling seeds for that unit.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import typing

import numpy as np
import tyro

_CONDITIONS = ("base", "time_decay", "belief")


@dataclasses.dataclass
class Args:
    input_dir: pathlib.Path = pathlib.Path("artifacts/black_bowl_steering_formal_100")
    expected_trials_per_group: int = 100
    episode_prefix: str = "black_bowl"
    expected_targets: str = "cookie,cabinet"
    report_title: str = "Black-bowl steering formal evaluation"
    strict: bool = False


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total**2)) / denominator
    return [center - half_width, center + half_width]


def _mean_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    half_width = 1.959963984540054 * std / math.sqrt(len(array))
    return {
        "n": len(array),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _stable_seed(*labels: str) -> int:
    return int(
        np.random.SeedSequence(
            [20260824, *[ord(character) for label in labels for character in label]]
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _cluster_bootstrap_summary(
    cluster_values: list[tuple[int, float]], *, seed: int, samples: int = 20_000
) -> dict[str, float | int]:
    by_cluster: dict[int, list[float]] = {}
    for cluster, value in cluster_values:
        by_cluster.setdefault(cluster, []).append(value)
    cluster_means = np.asarray(
        [np.mean(by_cluster[cluster]) for cluster in sorted(by_cluster)], dtype=np.float64
    )
    if len(cluster_means) == 0:
        return {"clusters": 0, "mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cluster_means), size=(samples, len(cluster_means)))
    bootstrap_means = np.mean(cluster_means[indices], axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "clusters": len(cluster_means),
        "mean": float(np.mean(cluster_means)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _mcnemar_exact_p_value(method_only: int, base_only: int) -> float:
    discordant = method_only + base_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(method_only, base_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _trajectory_metrics(path: pathlib.Path) -> dict[str, typing.Any]:
    with np.load(path) as trajectory:
        states = np.asarray(trajectory["states"], dtype=np.float64)
        actions = np.asarray(trajectory["executed_actions"], dtype=np.float64)
        policy_mask = np.asarray(trajectory["policy_action_mask"], dtype=np.bool_)
        initial_simulator_state_sha256 = None
        first_policy_observation_sha256 = None
        if "simulator_states" in trajectory:
            initial_simulator_state_sha256 = hashlib.sha256(
                np.ascontiguousarray(trajectory["simulator_states"][0]).tobytes()
            ).hexdigest()
        if all(
            key in trajectory
            for key in ("query_agentview_images", "query_wrist_images", "query_robot_states")
        ):
            digest = hashlib.sha256()
            for key in ("query_agentview_images", "query_wrist_images", "query_robot_states"):
                digest.update(np.ascontiguousarray(trajectory[key][0]).tobytes())
            first_policy_observation_sha256 = digest.hexdigest()

    if len(states) != len(actions) + 1 or len(policy_mask) != len(actions):
        raise ValueError(f"Malformed trajectory arrays in {path}")
    eef_step_distances = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)
    policy_actions = actions[policy_mask]
    action_tv = (
        float(np.sum(np.linalg.norm(np.diff(policy_actions, axis=0), axis=1)))
        if len(policy_actions) > 1
        else 0.0
    )
    return {
        "eef_path_length_m": float(np.sum(eef_step_distances[policy_mask])),
        "action_total_variation_l2": action_tv,
        "initial_simulator_state_sha256": initial_simulator_state_sha256,
        "first_policy_observation_sha256": first_policy_observation_sha256,
    }


def _episode_row(metadata_path: pathlib.Path) -> dict[str, typing.Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    trajectory_path = pathlib.Path(str(metadata["trajectory"]))
    if not trajectory_path.is_file():
        candidate = metadata_path.parent / trajectory_path.name
        if not candidate.is_file():
            raise FileNotFoundError(trajectory_path)
        trajectory_path = candidate
    diagnostics = metadata["query_diagnostics"]
    scales = [float(row["effective_guidance_scale"]) for row in diagnostics]
    negative_beliefs = [
        float(row["belief_negative"])
        for row in diagnostics
        if row.get("belief_negative") is not None
    ]
    return {
        "condition": str(metadata["condition"]),
        "target": str(metadata["target"]),
        "init_state_index": int(metadata["init_state_index"]),
        "repeat_index": int(metadata["repeat_index"]),
        "success": bool(metadata["success"]),
        "policy_steps": int(metadata["policy_steps"]),
        "policy_queries": int(metadata["policy_queries"]),
        "initial_observation_sha256": str(metadata["initial_observation_sha256"]),
        "sampling_seeds": [int(item["sampling_seed"]) for item in diagnostics],
        "first_guidance_scale": scales[0],
        "mean_post_first_guidance_scale": float(np.mean(scales[1:])) if len(scales) > 1 else math.nan,
        "max_post_first_guidance_scale": float(np.max(scales[1:])) if len(scales) > 1 else math.nan,
        "belief_negative_after_first_chunk": negative_beliefs[1] if len(negative_beliefs) > 1 else math.nan,
        "metadata_path": str(metadata_path),
        **_trajectory_metrics(trajectory_path),
    }


def _load_rows(input_dir: pathlib.Path, episode_prefix: str) -> list[dict[str, typing.Any]]:
    paths = sorted(input_dir.glob(f"{episode_prefix}_*_init*_repeat*_*.json"))
    rows = [_episode_row(path) for path in paths]
    seen: dict[tuple[str, str, int, int], str] = {}
    for row in rows:
        key = (row["condition"], row["target"], row["init_state_index"], row["repeat_index"])
        if key in seen:
            raise RuntimeError(f"Duplicate episode key {key}: {seen[key]} and {row['metadata_path']}")
        seen[key] = row["metadata_path"]
    return rows


def _validate_pairing(
    rows: list[dict[str, typing.Any]], expected: int, targets: tuple[str, ...], *, strict: bool
) -> dict[str, typing.Any]:
    counts = {
        f"{condition}/{target}": sum(
            row["condition"] == condition and row["target"] == target for row in rows
        )
        for condition in _CONDITIONS
        for target in targets
    }
    complete = all(count == expected for count in counts.values())
    if strict and not complete:
        raise RuntimeError(f"Expected {expected} episodes in every group, got {counts}")

    by_unit: dict[tuple[str, int, int], list[dict[str, typing.Any]]] = {}
    for row in rows:
        key = (row["target"], row["init_state_index"], row["repeat_index"])
        by_unit.setdefault(key, []).append(row)

    raw_render_hash_mismatches = []
    simulator_state_mismatches = []
    policy_observation_mismatches = []
    seed_mismatches = []
    incomplete_units = []
    for key, unit_rows in by_unit.items():
        conditions = {row["condition"] for row in unit_rows}
        if conditions != set(_CONDITIONS):
            incomplete_units.append({"unit": key, "conditions": sorted(conditions)})
            continue
        if len({row["initial_observation_sha256"] for row in unit_rows}) != 1:
            raw_render_hash_mismatches.append(key)
        simulator_hashes = [row.get("initial_simulator_state_sha256") for row in unit_rows]
        if all(item is not None for item in simulator_hashes) and len(set(simulator_hashes)) != 1:
            simulator_state_mismatches.append(key)
        policy_hashes = [row.get("first_policy_observation_sha256") for row in unit_rows]
        if all(item is not None for item in policy_hashes) and len(set(policy_hashes)) != 1:
            policy_observation_mismatches.append(key)
        seed_lists = [row["sampling_seeds"] for row in unit_rows]
        shared_length = min(map(len, seed_lists))
        if any(seeds[:shared_length] != seed_lists[0][:shared_length] for seeds in seed_lists[1:]):
            seed_mismatches.append(key)

    # New schema-v2 trajectories preserve exact simulator state and the first
    # observation actually sent to the policy. Those are the controlled inputs.
    # A raw reset render is never consumed during LIBERO's dummy-action warmup
    # and can differ by a few GPU-rendering pixels while both controlled inputs
    # remain bitwise identical. Older artifacts fall back to the raw hash.
    has_policy_hashes = all(row.get("first_policy_observation_sha256") is not None for row in rows)
    controlled_input_mismatches = (
        simulator_state_mismatches + policy_observation_mismatches
        if has_policy_hashes
        else raw_render_hash_mismatches
    )
    if strict and (controlled_input_mismatches or seed_mismatches or incomplete_units):
        raise RuntimeError(
            "Pairing validation failed: "
            f"controlled_input_mismatches={controlled_input_mismatches}, "
            f"raw_render_hash_mismatches={raw_render_hash_mismatches}, "
            f"seed_mismatches={seed_mismatches}, "
            f"incomplete_units={incomplete_units[:5]}"
        )
    return {
        "counts": counts,
        "complete": complete,
        "initial_observation_hashes_match": not raw_render_hash_mismatches,
        "raw_reset_render_hashes_match": not raw_render_hash_mismatches,
        "initial_simulator_states_match": not simulator_state_mismatches,
        "first_policy_observation_hashes_match": not policy_observation_mismatches,
        "controlled_policy_inputs_match": not controlled_input_mismatches,
        "raw_reset_render_hash_mismatch_units": raw_render_hash_mismatches,
        "common_sampling_seed_prefixes_match": not seed_mismatches,
        "complete_paired_units": len(incomplete_units) == 0,
        "incomplete_unit_count": len(incomplete_units),
    }


def _group_summary(selected: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
    successes = sum(row["success"] for row in selected)
    total = len(selected)
    result: dict[str, typing.Any] = {
        "trials": total,
        "successes": successes,
        "success_rate": successes / total if total else math.nan,
        "success_rate_ci95_wilson": _wilson_interval(successes, total),
        "success_rate_ci95_init_state_cluster_bootstrap": _cluster_bootstrap_summary(
            [(row["init_state_index"], float(row["success"])) for row in selected],
            seed=_stable_seed(
                "group",
                str(selected[0]["condition"]) if selected else "none",
                str(selected[0]["target"]) if selected else "none",
            ),
        ),
    }
    for metric in (
        "policy_steps",
        "policy_queries",
        "eef_path_length_m",
        "action_total_variation_l2",
        "first_guidance_scale",
        "mean_post_first_guidance_scale",
        "max_post_first_guidance_scale",
        "belief_negative_after_first_chunk",
    ):
        values = [float(row[metric]) for row in selected if math.isfinite(float(row[metric]))]
        result[metric] = _mean_summary(values)
    return result


def _paired_comparison(
    rows: list[dict[str, typing.Any]], target: str, method: str, reference: str
) -> dict[str, typing.Any]:
    reference_rows = {
        (row["init_state_index"], row["repeat_index"]): row
        for row in rows
        if row["target"] == target and row["condition"] == reference
    }
    treatment = {
        (row["init_state_index"], row["repeat_index"]): row
        for row in rows
        if row["target"] == target and row["condition"] == method
    }
    keys = sorted(reference_rows.keys() & treatment.keys())
    method_only = sum(
        treatment[key]["success"] and not reference_rows[key]["success"] for key in keys
    )
    reference_only = sum(
        reference_rows[key]["success"] and not treatment[key]["success"] for key in keys
    )
    success_deltas = [
        float(treatment[key]["success"]) - float(reference_rows[key]["success"]) for key in keys
    ]
    result: dict[str, typing.Any] = {
        "method_condition": method,
        "reference_condition": reference,
        "pairs": len(keys),
        "method_success_reference_failure": method_only,
        "reference_success_method_failure": reference_only,
        "both_success": sum(
            treatment[key]["success"] and reference_rows[key]["success"] for key in keys
        ),
        "both_failure": sum(
            not treatment[key]["success"] and not reference_rows[key]["success"] for key in keys
        ),
        "success_rate_difference_method_minus_reference": (
            float(np.mean(success_deltas)) if keys else math.nan
        ),
        "success_rate_difference_init_state_cluster_bootstrap": _cluster_bootstrap_summary(
            [(key[0], delta) for key, delta in zip(keys, success_deltas, strict=True)],
            seed=_stable_seed("success", method, reference, target),
        ),
        # This exact test treats the two sampling replicates as separate paired
        # episodes. Prefer the clustered bootstrap CI for the primary inference.
        "mcnemar_exact_two_sided_p_episode_level": _mcnemar_exact_p_value(
            method_only, reference_only
        ),
    }
    for metric in ("policy_steps", "eef_path_length_m", "action_total_variation_l2"):
        deltas = [
            float(treatment[key][metric]) - float(reference_rows[key][metric]) for key in keys
        ]
        metric_summary = _mean_summary(deltas)
        metric_summary["init_state_cluster_bootstrap"] = _cluster_bootstrap_summary(
            [(key[0], delta) for key, delta in zip(keys, deltas, strict=True)],
            seed=_stable_seed(metric, method, reference, target, "all"),
        )
        result[f"paired_delta_{metric}_all_pairs"] = metric_summary

        both_success_keys = [
            key for key in keys if treatment[key]["success"] and reference_rows[key]["success"]
        ]
        successful_deltas = [
            float(treatment[key][metric]) - float(reference_rows[key][metric])
            for key in both_success_keys
        ]
        successful_summary = _mean_summary(successful_deltas)
        successful_summary["init_state_cluster_bootstrap"] = _cluster_bootstrap_summary(
            [
                (key[0], delta)
                for key, delta in zip(both_success_keys, successful_deltas, strict=True)
            ],
            seed=_stable_seed(metric, method, reference, target, "both_success"),
        )
        result[f"paired_delta_{metric}_both_success"] = successful_summary
    return result


def _write_csv(rows: list[dict[str, typing.Any]], path: pathlib.Path) -> None:
    fields = [
        "condition",
        "target",
        "init_state_index",
        "repeat_index",
        "success",
        "policy_steps",
        "policy_queries",
        "eef_path_length_m",
        "action_total_variation_l2",
        "first_guidance_scale",
        "mean_post_first_guidance_scale",
        "max_post_first_guidance_scale",
        "belief_negative_after_first_chunk",
        "initial_observation_sha256",
        "metadata_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _write_markdown(
    summary: dict[str, typing.Any], path: pathlib.Path, targets: tuple[str, ...], report_title: str
) -> None:
    lines = [
        f"# {report_title}",
        "",
        "| Method | Target | Success | 95% Wilson CI | EEF path (m) | Action TV (L2) | Policy steps |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for condition in _CONDITIONS:
        for target in targets:
            row = summary["groups"][f"{condition}/{target}"]
            low, high = row["success_rate_ci95_wilson"]
            lines.append(
                f"| {condition} | {target} | {row['successes']}/{row['trials']} "
                f"({row['success_rate']:.1%}) | [{low:.1%}, {high:.1%}] | "
                f"{row['eef_path_length_m']['mean']:.4f} | "
                f"{row['action_total_variation_l2']['mean']:.4f} | "
                f"{row['policy_steps']['mean']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Executed guidance weights",
            "",
            "| Method | Target | First segment | Mean max after first | Mean over later segments |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for condition in ("time_decay", "belief"):
        for target in targets:
            row = summary["groups"][f"{condition}/{target}"]
            lines.append(
                f"| {condition} | {target} | {row['first_guidance_scale']['mean']:.4f} | "
                f"{row['max_post_first_guidance_scale']['mean']:.4f} | "
                f"{row['mean_post_first_guidance_scale']['mean']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "The confidence interval resamples the 50 initial states as clusters; the exact McNemar p-value is episode-level.",
            "",
        ]
    )
    for method, reference in (
        ("time_decay", "base"),
        ("belief", "base"),
        ("belief", "time_decay"),
    ):
        for target in targets:
            comparison = summary["paired_comparisons"][f"{method}-vs-{reference}/{target}"]
            cluster = comparison["success_rate_difference_init_state_cluster_bootstrap"]
            lines.append(
                f"- {method} vs {reference}, {target}: success-rate difference "
                f"{comparison['success_rate_difference_method_minus_reference']:+.1%} "
                f"(state-cluster bootstrap 95% CI [{cluster['ci95_low']:+.1%}, "
                f"{cluster['ci95_high']:+.1%}]); episode-level McNemar exact "
                f"p={comparison['mcnemar_exact_two_sided_p_episode_level']:.4g}; "
                f"discordant wins/losses={comparison['method_success_reference_failure']}/"
                f"{comparison['reference_success_method_failure']}."
            )
    lines.extend(
        [
            "",
            "## Motion deltas on jointly successful pairs",
            "",
            "Negative deltas mean the first method used less motion than the reference. Confidence intervals resample init-state clusters.",
            "",
            "| Comparison | Target | Pairs | EEF path delta (m) | Action-TV delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for method, reference in (
        ("time_decay", "base"),
        ("belief", "base"),
        ("belief", "time_decay"),
    ):
        for target in targets:
            comparison = summary["paired_comparisons"][f"{method}-vs-{reference}/{target}"]
            path_delta = comparison[
                "paired_delta_eef_path_length_m_both_success"
            ]["init_state_cluster_bootstrap"]
            tv_delta = comparison[
                "paired_delta_action_total_variation_l2_both_success"
            ]["init_state_cluster_bootstrap"]
            lines.append(
                f"| {method} - {reference} | {target} | {comparison['both_success']} | "
                f"{path_delta['mean']:+.4f} [{path_delta['ci95_low']:+.4f}, "
                f"{path_delta['ci95_high']:+.4f}] | {tv_delta['mean']:+.3f} "
                f"[{tv_delta['ci95_low']:+.3f}, {tv_delta['ci95_high']:+.3f}] |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: Args) -> None:
    if args.expected_trials_per_group < 1:
        raise ValueError("expected_trials_per_group must be positive")
    targets = tuple(target.strip() for target in args.expected_targets.split(",") if target.strip())
    if len(targets) != 2 or len(set(targets)) != 2:
        raise ValueError("expected_targets must contain exactly two distinct comma-separated targets")
    if not args.episode_prefix:
        raise ValueError("episode_prefix must be nonempty")
    rows = _load_rows(args.input_dir, args.episode_prefix)
    if not rows:
        raise RuntimeError(f"No formal episode JSON files found in {args.input_dir}")
    validation = _validate_pairing(
        rows, args.expected_trials_per_group, targets, strict=args.strict
    )
    groups = {
        f"{condition}/{target}": _group_summary(
            [row for row in rows if row["condition"] == condition and row["target"] == target]
        )
        for condition in _CONDITIONS
        for target in targets
    }
    comparisons = (
        ("time_decay", "base"),
        ("belief", "base"),
        ("belief", "time_decay"),
    )
    paired = {
        f"{method}-vs-{reference}/{target}": _paired_comparison(
            rows, target, method, reference
        )
        for method, reference in comparisons
        for target in targets
    }
    summary = {
        "input_dir": str(args.input_dir),
        "episode_prefix": args.episode_prefix,
        "targets": list(targets),
        "report_title": args.report_title,
        "expected_trials_per_group": args.expected_trials_per_group,
        "validation": validation,
        "groups": groups,
        "paired_comparisons": paired,
    }
    args.input_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.input_dir / "formal_summary.json"
    csv_path = args.input_dir / "formal_episodes.csv"
    markdown_path = args.input_dir / "formal_summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(rows, csv_path)
    _write_markdown(summary, markdown_path, targets, args.report_title)
    print(json.dumps(validation, indent=2))
    print(f"Wrote {json_path}, {csv_path}, and {markdown_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
