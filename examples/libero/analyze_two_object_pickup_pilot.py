"""Analyze the paired fine-tuned two-object pickup steering pilot."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import pathlib
from typing import Any

import numpy as np
import tyro

CONDITIONS = ("base", "time_decay_g0p9", "time_decay_g0p5", "belief")
TARGETS = ("cream_cheese", "tomato_sauce")
BODY_NAME = {
    "cream_cheese": "cream_cheese_1",
    "tomato_sauce": "tomato_sauce_1",
}


@dataclasses.dataclass
class Args:
    input_dir: pathlib.Path = pathlib.Path("artifacts/two_object_pickup_finetune_pilot_10")
    expected_trials: int = 10


def _mean(values: list[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else math.nan


def _std(values: list[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0


def _at(values: list[float], index: int) -> float:
    return float(values[index]) if len(values) > index else math.nan


def _episode(metadata_path: pathlib.Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    trajectory_path = pathlib.Path(metadata["trajectory"])
    if not trajectory_path.is_file():
        trajectory_path = metadata_path.parent / trajectory_path.name
    with np.load(trajectory_path) as data:
        states = np.asarray(data["states"], dtype=np.float64)
        actions = np.asarray(data["executed_actions"], dtype=np.float64)
        policy_mask = np.asarray(data["policy_action_mask"], dtype=np.bool_)
        body_positions = np.asarray(data["body_positions"], dtype=np.float64)

    body_names = list(metadata["state_recording"]["body_names"])
    target = str(metadata["target"])
    alternate = next(name for name in TARGETS if name != target)
    target_index = body_names.index(BODY_NAME[target])
    alternate_index = body_names.index(BODY_NAME[alternate])
    first_policy_action = int(np.flatnonzero(policy_mask)[0])
    policy_state_indices = np.flatnonzero(policy_mask) + 1

    eef_before_policy = states[first_policy_action, :3]
    eef_after_policy = states[policy_state_indices, :3]
    target_initial = body_positions[first_policy_action, target_index]
    alternate_initial = body_positions[first_policy_action, alternate_index]
    separation = float(np.linalg.norm(target_initial - alternate_initial))
    initial_relative_evidence = (
        np.linalg.norm(eef_before_policy - alternate_initial)
        - np.linalg.norm(eef_before_policy - target_initial)
    ) / separation
    relative_evidence = (
        np.linalg.norm(eef_after_policy - alternate_initial, axis=1)
        - np.linalg.norm(eef_after_policy - target_initial, axis=1)
    ) / separation
    evidence_change = relative_evidence - initial_relative_evidence

    policy_step_distances = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)[policy_mask]
    policy_actions = actions[policy_mask]
    action_tv = (
        float(np.sum(np.linalg.norm(np.diff(policy_actions, axis=0), axis=1)))
        if len(policy_actions) > 1
        else 0.0
    )

    target_lift = body_positions[policy_state_indices, target_index, 2] - target_initial[2]
    alternate_lift = body_positions[policy_state_indices, alternate_index, 2] - alternate_initial[2]
    lift_indices = np.flatnonzero(target_lift >= 0.03)

    diagnostics = list(metadata["query_diagnostics"])
    weights = [float(row["effective_guidance_scale"]) for row in diagnostics]
    bpos = [
        math.nan if row.get("belief_positive") is None else float(row["belief_positive"])
        for row in diagnostics
    ]
    bneg = [
        math.nan if row.get("belief_negative") is None else float(row["belief_negative"])
        for row in diagnostics
    ]
    energies_pos = [
        math.nan if row.get("energy_positive") is None else float(row["energy_positive"])
        for row in diagnostics
    ]
    energies_neg = [
        math.nan if row.get("energy_negative") is None else float(row["energy_negative"])
        for row in diagnostics
    ]
    lift_step = int(lift_indices[0] + 1) if len(lift_indices) else None
    lift_query = (
        min((lift_step - 1) // int(metadata["replan_steps"]), len(diagnostics) - 1)
        if lift_step is not None
        else None
    )

    return {
        "condition": str(metadata["condition"]),
        "target": target,
        "init_state_index": int(metadata["init_state_index"]),
        "success": bool(metadata["success"]),
        "policy_steps": int(metadata["policy_steps"]),
        "policy_queries": int(metadata["policy_queries"]),
        "eef_path_length_m": float(np.sum(policy_step_distances)),
        "action_total_variation_l2": action_tv,
        "evidence_auc_first_10": _mean(evidence_change[:10].tolist()),
        "evidence_auc_first_20": _mean(evidence_change[:20].tolist()),
        "evidence_at_step_10": _at(evidence_change.tolist(), 9),
        "evidence_at_step_20": _at(evidence_change.tolist(), 19),
        "evidence_at_step_50": _at(evidence_change.tolist(), 49),
        "evidence_at_step_100": _at(evidence_change.tolist(), 99),
        "max_target_lift_m": float(np.max(target_lift)),
        "max_alternate_lift_m": float(np.max(alternate_lift)),
        "wrong_object_lifted": bool(np.max(alternate_lift) >= 0.03 and np.max(target_lift) < 0.03),
        "target_lift_step_3cm": lift_step,
        "w_query_1": _at(weights, 0),
        "w_query_2": _at(weights, 1),
        "w_query_5": _at(weights, 4),
        "w_mean_first_10_queries": _mean(weights[:10]),
        "w_mean_all_queries": _mean(weights),
        "belief_positive_query_2": _at(bpos, 1),
        "belief_negative_query_2": _at(bneg, 1),
        "belief_positive_mean_first_10": _mean(bpos[:10]),
        "belief_negative_mean_first_10": _mean(bneg[:10]),
        "belief_positive_final": _at(bpos, len(bpos) - 1),
        "belief_negative_final": _at(bneg, len(bneg) - 1),
        "w_at_target_lift_3cm": math.nan if lift_query is None else weights[lift_query],
        "belief_positive_at_target_lift_3cm": (
            math.nan if lift_query is None else bpos[lift_query]
        ),
        "energy_margin_pos_minus_neg_mean": _mean(
            (np.asarray(energies_pos) - np.asarray(energies_neg)).tolist()
        ),
        "initial_observation_sha256": str(metadata["initial_observation_sha256"]),
        "sampling_seeds": [int(row["sampling_seed"]) for row in diagnostics],
        "metadata": str(metadata_path),
    }


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["success"]]
    metrics = (
        "policy_steps",
        "eef_path_length_m",
        "action_total_variation_l2",
        "evidence_auc_first_10",
        "evidence_auc_first_20",
        "evidence_at_step_10",
        "evidence_at_step_20",
        "evidence_at_step_50",
        "evidence_at_step_100",
        "max_target_lift_m",
        "max_alternate_lift_m",
        "w_query_1",
        "w_query_2",
        "w_query_5",
        "w_mean_first_10_queries",
        "w_mean_all_queries",
        "belief_positive_query_2",
        "belief_negative_query_2",
        "belief_positive_mean_first_10",
        "belief_negative_mean_first_10",
        "belief_positive_final",
        "belief_negative_final",
        "w_at_target_lift_3cm",
        "belief_positive_at_target_lift_3cm",
        "energy_margin_pos_minus_neg_mean",
    )
    result: dict[str, Any] = {
        "n": len(rows),
        "successes": len(successes),
        "success_rate": len(successes) / len(rows),
        "wrong_object_lift_rate": sum(row["wrong_object_lifted"] for row in rows) / len(rows),
        "successful_policy_steps_mean": _mean([row["policy_steps"] for row in successes]),
        "successful_eef_path_length_m_mean": _mean(
            [row["eef_path_length_m"] for row in successes]
        ),
        "target_lift_step_3cm_mean": _mean(
            [
                math.nan if row["target_lift_step_3cm"] is None else row["target_lift_step_3cm"]
                for row in rows
            ]
        ),
    }
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        result[f"{metric}_mean"] = _mean(values)
        result[f"{metric}_std"] = _std(values)
    return result


def main(args: Args) -> None:
    paths = sorted(args.input_dir.glob("two_object_pickup_*_init*_repeat0_*.json"))
    rows = [_episode(path) for path in paths]
    expected_total = len(CONDITIONS) * len(TARGETS) * args.expected_trials
    if len(rows) != expected_total:
        raise RuntimeError(f"Expected {expected_total} episodes, found {len(rows)}")

    counts = {
        f"{condition}/{target}": sum(
            row["condition"] == condition and row["target"] == target for row in rows
        )
        for condition in CONDITIONS
        for target in TARGETS
    }
    if any(count != args.expected_trials for count in counts.values()):
        raise RuntimeError(f"Incomplete groups: {counts}")

    mismatched_observations = []
    mismatched_seeds = []
    for target in TARGETS:
        for init_state in range(args.expected_trials):
            unit = [
                row
                for row in rows
                if row["target"] == target and row["init_state_index"] == init_state
            ]
            if len({row["initial_observation_sha256"] for row in unit}) != 1:
                mismatched_observations.append([target, init_state])
            shared = min(len(row["sampling_seeds"]) for row in unit)
            if len({tuple(row["sampling_seeds"][:shared]) for row in unit}) != 1:
                mismatched_seeds.append([target, init_state])
    if mismatched_observations or mismatched_seeds:
        raise RuntimeError(
            f"Pairing failed: observations={mismatched_observations}, seeds={mismatched_seeds}"
        )

    groups = {
        condition: {
            target: _group(
                [row for row in rows if row["condition"] == condition and row["target"] == target]
            )
            for target in TARGETS
        }
        for condition in CONDITIONS
    }
    overall = {
        condition: _group([row for row in rows if row["condition"] == condition])
        for condition in CONDITIONS
    }

    paired_vs_base: dict[str, dict[str, dict[str, int]]] = {}
    for condition in CONDITIONS[1:]:
        paired_vs_base[condition] = {}
        for target in TARGETS:
            base = {
                row["init_state_index"]: row["success"]
                for row in rows
                if row["condition"] == "base" and row["target"] == target
            }
            method = {
                row["init_state_index"]: row["success"]
                for row in rows
                if row["condition"] == condition and row["target"] == target
            }
            paired_vs_base[condition][target] = {
                "both_success": sum(base[i] and method[i] for i in base),
                "method_only_success": sum((not base[i]) and method[i] for i in base),
                "base_only_success": sum(base[i] and (not method[i]) for i in base),
                "both_failure": sum((not base[i]) and (not method[i]) for i in base),
            }

    csv_path = args.input_dir / "pilot_episode_metrics.csv"
    csv_fields = [key for key in rows[0] if key not in {"sampling_seeds"}]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_fields} for row in rows)

    report = {
        "checkpoint": (
            "/workspace/checkpoints/checkpoints/pi05_libero_legibility_finetune/"
            "libero_legibility_v1/19999"
        ),
        "design": {
            "conditions": list(CONDITIONS),
            "targets": list(TARGETS),
            "trials_per_group": args.expected_trials,
            "total_episodes": len(rows),
            "paired_initial_observations": True,
            "common_sampling_seed_prefixes": True,
        },
        "groups": groups,
        "overall": overall,
        "paired_success_vs_base": paired_vs_base,
    }
    json_path = args.input_dir / "pilot_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Two-object pickup fine-tune steering pilot",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        f"All {len(rows)} episodes passed paired initial-observation and common sampling-seed checks.",
        "",
        "| Condition | Target | Success | Successful steps | Successful EEF path (m) | Evidence Δ@20 / @50 | W q1 / q2 / q5 / at lift |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for target in TARGETS:
            group = groups[condition][target]
            lines.append(
                f"| {condition} | {target} | {group['successes']}/{group['n']} "
                f"| {group['successful_policy_steps_mean']:.1f} "
                f"| {group['successful_eef_path_length_m_mean']:.3f} "
                f"| {group['evidence_at_step_20_mean']:.3f} / {group['evidence_at_step_50_mean']:.3f} "
                f"| {group['w_query_1_mean']:.3f} / {group['w_query_2_mean']:.3f} / {group['w_query_5_mean']:.3f} / {group['w_at_target_lift_3cm_mean']:.3f} |"
            )
    lines.extend(
        [
            "",
            "The geometric evidence is the change in normalized EEF closeness to the instructed object versus the competing object; positive values favor the instructed target.",
            "",
            f"This is a screening pilot (n={args.expected_trials}/group), so differences are descriptive rather than statistically conclusive.",
        ]
    )
    markdown_path = args.input_dir / "PILOT_REPORT.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
