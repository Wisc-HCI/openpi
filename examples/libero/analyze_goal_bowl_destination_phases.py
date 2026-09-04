"""Analyze phase-aligned guidance and geometric intent evidence for the bowl task."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import pathlib
import typing

import numpy as np
import tyro

import analyze_black_bowl_steering as _formal


_CONDITIONS = ("base", "time_decay", "belief")
_METRICS = (
    "pre_lift_steps",
    "pre_lift_eef_path_m",
    "pre_lift_action_tv_l2",
    "guidance_at_close",
    "guidance_at_grasp",
    "guidance_at_lift",
    "post_lift_evidence_5",
    "post_lift_evidence_10",
    "post_lift_evidence_15",
    "post_lift_evidence_20",
    "post_lift_early_auc20",
    "post_lift_wrong_fraction15",
)
_COMPARISONS = (
    ("time_decay", "base"),
    ("belief", "base"),
    ("belief", "time_decay"),
)


@dataclasses.dataclass
class Args:
    input_dir: pathlib.Path
    episode_prefix: str = "goal_bowl_destination"
    expected_trials_per_group: int = 100
    expected_targets: str = "cabinet,plate"
    output_stem: str = "phase_legibility"
    lift_threshold_m: float = 0.005


def _finite(value: typing.Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _trajectory_path(metadata_path: pathlib.Path, metadata: dict[str, typing.Any]) -> pathlib.Path:
    path = pathlib.Path(str(metadata["trajectory"]))
    if path.is_file():
        return path
    candidate = metadata_path.parent / path.name
    if not candidate.is_file():
        raise FileNotFoundError(path)
    return candidate


def _event_query(query_steps: np.ndarray, event_step: int | None) -> int | None:
    if event_step is None:
        return None
    query = int(np.searchsorted(query_steps, event_step, side="right") - 1)
    return query if query >= 0 else None


def _query_value(values: list[float | None], query: int | None, offset: int = 0) -> float:
    if query is None or query + offset < 0 or query + offset >= len(values):
        return math.nan
    value = values[query + offset]
    return float(value) if _finite(value) else math.nan


def _first_lift_state(
    height: np.ndarray, *, start: int, threshold_m: float, consecutive: int = 3
) -> int | None:
    threshold = float(height[0]) + threshold_m
    for index in range(start, len(height) - consecutive + 1):
        if np.all(height[index : index + consecutive] > threshold):
            return index
    return None


def _episode_row(metadata_path: pathlib.Path, *, lift_threshold_m: float) -> dict[str, typing.Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    diagnostics = metadata["query_diagnostics"]
    guidance = [row["effective_guidance_scale"] for row in diagnostics]
    belief_positive = [row.get("belief_positive") for row in diagnostics]
    belief_negative = [row.get("belief_negative") for row in diagnostics]

    with np.load(_trajectory_path(metadata_path, metadata)) as trajectory:
        states = np.asarray(trajectory["states"], dtype=np.float64)
        actions = np.asarray(trajectory["executed_actions"], dtype=np.float64)
        policy_mask = np.asarray(trajectory["policy_action_mask"], dtype=np.bool_)
        query_steps = np.asarray(trajectory["policy_query_steps"], dtype=np.int64)
        body_positions = np.asarray(trajectory["body_positions"], dtype=np.float64)
        site_positions = np.asarray(trajectory["site_positions"], dtype=np.float64)
        bowl_grasped = np.asarray(trajectory["bowl_grasped"], dtype=np.bool_)

    body_names = list(metadata["state_recording"]["body_names"])
    site_names = list(metadata["state_recording"]["site_names"])
    bowl_index = body_names.index("akita_black_bowl_1")
    plate_index = body_names.index("plate_1")
    cabinet_index = site_names.index("wooden_cabinet_1_top_side")

    close_actions = np.flatnonzero(policy_mask & (actions[:, 6] > 0))
    close_step = int(close_actions[0]) if len(close_actions) else None
    grasp_states = np.flatnonzero(bowl_grasped)
    grasp_state = int(grasp_states[0]) if len(grasp_states) else None
    lift_state = _first_lift_state(
        body_positions[:, bowl_index, 2],
        start=close_step if close_step is not None else int(query_steps[0]),
        threshold_m=lift_threshold_m,
    )
    close_query = _event_query(query_steps, close_step)
    grasp_query = _event_query(query_steps, grasp_state)
    lift_query = _event_query(query_steps, lift_state)

    policy_start = int(query_steps[0])
    pre_lift_steps = math.nan
    pre_lift_path = math.nan
    pre_lift_tv = math.nan
    evidence = {5: math.nan, 10: math.nan, 15: math.nan, 20: math.nan}
    early_auc20 = math.nan
    wrong_fraction15 = math.nan
    if lift_state is not None and lift_state > policy_start:
        pre_lift_steps = float(lift_state - policy_start)
        pre_lift_path = float(
            np.linalg.norm(np.diff(states[policy_start : lift_state + 1, :3], axis=0), axis=1).sum()
        )
        pre_actions = actions[policy_start:lift_state][policy_mask[policy_start:lift_state]]
        pre_lift_tv = (
            float(np.linalg.norm(np.diff(pre_actions, axis=0), axis=1).sum())
            if len(pre_actions) > 1
            else 0.0
        )

        cabinet = site_positions[:, cabinet_index]
        plate = body_positions[:, plate_index]
        true_goal, alternate_goal = (
            (cabinet, plate) if metadata["target"] == "cabinet" else (plate, cabinet)
        )
        end = min(lift_state + 21, len(states))
        indices = np.arange(lift_state, end)
        separation = float(np.linalg.norm(true_goal[lift_state] - alternate_goal[lift_state]))
        true_start = float(np.linalg.norm(states[lift_state, :3] - true_goal[lift_state]))
        alternate_start = float(np.linalg.norm(states[lift_state, :3] - alternate_goal[lift_state]))
        geometric_evidence = (
            (true_start - np.linalg.norm(states[indices, :3] - true_goal[indices], axis=1))
            - (
                alternate_start
                - np.linalg.norm(states[indices, :3] - alternate_goal[indices], axis=1)
            )
        ) / separation
        if len(geometric_evidence) < 21:
            geometric_evidence = np.pad(
                geometric_evidence,
                (0, 21 - len(geometric_evidence)),
                mode="constant",
                constant_values=float(geometric_evidence[-1]),
            )
        for offset in evidence:
            evidence[offset] = float(geometric_evidence[offset])
        weights = np.linspace(1.0, 0.05, 20)
        early_auc20 = float(np.average(geometric_evidence[1:21], weights=weights))
        wrong_fraction15 = float(np.mean(geometric_evidence[1:16] < 0))

    row: dict[str, typing.Any] = {
        "condition": str(metadata["condition"]),
        "target": str(metadata["target"]),
        "init_state_index": int(metadata["init_state_index"]),
        "repeat_index": int(metadata["repeat_index"]),
        "success": bool(metadata["success"]),
        "policy_steps": int(metadata["policy_steps"]),
        "close_step": close_step,
        "grasp_state": grasp_state,
        "lift_state": lift_state,
        "close_query": close_query,
        "grasp_query": grasp_query,
        "lift_query": lift_query,
        "pre_lift_steps": pre_lift_steps,
        "pre_lift_eef_path_m": pre_lift_path,
        "pre_lift_action_tv_l2": pre_lift_tv,
        "post_lift_evidence_5": evidence[5],
        "post_lift_evidence_10": evidence[10],
        "post_lift_evidence_15": evidence[15],
        "post_lift_evidence_20": evidence[20],
        "post_lift_early_auc20": early_auc20,
        "post_lift_wrong_fraction15": wrong_fraction15,
        "metadata_path": str(metadata_path),
    }
    for event, query in (("close", close_query), ("grasp", grasp_query), ("lift", lift_query)):
        row[f"guidance_at_{event}"] = _query_value(guidance, query)
        row[f"belief_positive_at_{event}"] = _query_value(belief_positive, query)
        row[f"belief_negative_at_{event}"] = _query_value(belief_negative, query)
        for offset in range(1, 4):
            row[f"guidance_{event}_plus_{offset}"] = _query_value(guidance, query, offset)
            row[f"belief_positive_{event}_plus_{offset}"] = _query_value(
                belief_positive, query, offset
            )
            row[f"belief_negative_{event}_plus_{offset}"] = _query_value(
                belief_negative, query, offset
            )
    return row


def _mean(values: list[float]) -> dict[str, float | int]:
    return _formal._mean_summary([value for value in values if _finite(value)])


def _group_summary(rows: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
    result: dict[str, typing.Any] = {
        "trials": len(rows),
        "successes": sum(row["success"] for row in rows),
    }
    for metric in _METRICS:
        result[metric] = _mean([row[metric] for row in rows])
    for event in ("close", "grasp", "lift"):
        result[f"belief_positive_at_{event}"] = _mean(
            [row[f"belief_positive_at_{event}"] for row in rows]
        )
        result[f"belief_negative_at_{event}"] = _mean(
            [row[f"belief_negative_at_{event}"] for row in rows]
        )
        for offset in range(1, 4):
            result[f"guidance_{event}_plus_{offset}"] = _mean(
                [row[f"guidance_{event}_plus_{offset}"] for row in rows]
            )
            result[f"belief_positive_{event}_plus_{offset}"] = _mean(
                [row[f"belief_positive_{event}_plus_{offset}"] for row in rows]
            )
    return result


def _paired_metric(
    rows: list[dict[str, typing.Any]], target: str, method: str, reference: str, metric: str, *, joint_success: bool
) -> dict[str, float | int]:
    selected = {
        (row["condition"], row["init_state_index"], row["repeat_index"]): row
        for row in rows
        if row["target"] == target
    }
    pairs: list[tuple[int, float]] = []
    units = sorted({(key[1], key[2]) for key in selected})
    for init_state, repeat in units:
        treatment = selected.get((method, init_state, repeat))
        control = selected.get((reference, init_state, repeat))
        if treatment is None or control is None:
            continue
        if joint_success and not (treatment["success"] and control["success"]):
            continue
        if not (_finite(treatment[metric]) and _finite(control[metric])):
            continue
        pairs.append((init_state, float(treatment[metric]) - float(control[metric])))
    return _formal._cluster_bootstrap_summary(
        pairs,
        seed=_formal._stable_seed("phase", target, method, reference, metric, str(joint_success)),
    ) | {"pairs": len(pairs)}


def _write_markdown(summary: dict[str, typing.Any], path: pathlib.Path) -> None:
    lines = [
        "# Bowl-destination phase-aligned legibility",
        "",
        f"Lift/transport onset is the first three consecutive states after the close command with bowl height at least {summary['lift_threshold_m'] * 1000:.1f} mm above its initial value.",
        "",
        "Evidence is measured from bowl-lift onset. Positive values indicate relative progress toward the true destination; negative values indicate motion toward the competing destination.",
        "",
        "| Method | Target | Success | W close | W grasp | W lift | Pre-lift path (m) | Pre-lift TV | AUC20 | Wrong first 15 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in _CONDITIONS:
        for target in summary["targets"]:
            group = summary["groups"][f"{condition}/{target}"]
            lines.append(
                f"| {condition} | {target} | {group['successes']}/{group['trials']} | "
                f"{group['guidance_at_close']['mean']:.3f} | {group['guidance_at_grasp']['mean']:.3f} | "
                f"{group['guidance_at_lift']['mean']:.3f} | {group['pre_lift_eef_path_m']['mean']:.4f} | "
                f"{group['pre_lift_action_tv_l2']['mean']:.3f} | {group['post_lift_early_auc20']['mean']:.4f} | "
                f"{group['post_lift_wrong_fraction15']['mean']:.3f} |"
            )
    lines.extend([
        "",
        "## Belief phase profile",
        "",
        "| Target | b+ close | b+ grasp | b+ lift | W grasp+1 | W grasp+2 | W grasp+3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for target in summary["targets"]:
        group = summary["groups"][f"belief/{target}"]
        lines.append(
            f"| {target} | {group['belief_positive_at_close']['mean']:.3f} | "
            f"{group['belief_positive_at_grasp']['mean']:.3f} | {group['belief_positive_at_lift']['mean']:.3f} | "
            f"{group['guidance_grasp_plus_1']['mean']:.3f} | {group['guidance_grasp_plus_2']['mean']:.3f} | "
            f"{group['guidance_grasp_plus_3']['mean']:.3f} |"
        )
    lines.extend([
        "",
        "## Paired deltas on jointly successful episodes",
        "",
        "Deltas are method minus reference. Intervals resample the 50 init states as clusters.",
        "",
        "| Comparison | Target | Metric | Pairs | Delta [95% CI] |",
        "|---|---|---|---:|---:|",
    ])
    for method, reference in _COMPARISONS:
        for target in summary["targets"]:
            for metric in ("pre_lift_eef_path_m", "pre_lift_action_tv_l2", "post_lift_early_auc20", "post_lift_wrong_fraction15"):
                result = summary["paired"][f"{method}-vs-{reference}/{target}/{metric}/joint_success"]
                lines.append(
                    f"| {method} - {reference} | {target} | {metric} | {result['pairs']} | "
                    f"{result['mean']:+.4f} [{result['ci95_low']:+.4f}, {result['ci95_high']:+.4f}] |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: Args) -> None:
    targets = tuple(item.strip() for item in args.expected_targets.split(",") if item.strip())
    paths = sorted(args.input_dir.glob(f"{args.episode_prefix}_*_init*_repeat*_*.json"))
    if args.lift_threshold_m <= 0:
        raise ValueError("lift_threshold_m must be positive")
    rows = [_episode_row(path, lift_threshold_m=args.lift_threshold_m) for path in paths]
    counts = {
        f"{condition}/{target}": sum(
            row["condition"] == condition and row["target"] == target for row in rows
        )
        for condition in _CONDITIONS
        for target in targets
    }
    if any(count != args.expected_trials_per_group for count in counts.values()):
        raise RuntimeError(f"Unexpected group counts: {counts}")
    groups = {
        f"{condition}/{target}": _group_summary(
            [row for row in rows if row["condition"] == condition and row["target"] == target]
        )
        for condition in _CONDITIONS
        for target in targets
    }
    paired = {
        f"{method}-vs-{reference}/{target}/{metric}/{scope}": _paired_metric(
            rows, target, method, reference, metric, joint_success=(scope == "joint_success")
        )
        for method, reference in _COMPARISONS
        for target in targets
        for metric in _METRICS
        for scope in ("all", "joint_success")
    }
    summary = {
        "input_dir": str(args.input_dir),
        "targets": targets,
        "lift_threshold_m": args.lift_threshold_m,
        "counts": counts,
        "groups": groups,
        "paired": paired,
    }
    json_path = args.input_dir / f"{args.output_stem}.json"
    csv_path = args.input_dir / f"{args.output_stem}_episodes.csv"
    markdown_path = args.input_dir / f"{args.output_stem}.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    _write_markdown(summary, markdown_path)
    print(f"Wrote {json_path}, {csv_path}, and {markdown_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
