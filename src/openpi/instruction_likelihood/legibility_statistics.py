"""Statistics for the pre-registered LIBERO spatial-legibility experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import math
from typing import Any

import numpy as np
from scipy import stats

PREFIX_CHUNKS = {"C20": 2, "C30": 3, "C40": 4}
LEVEL_ORDER = {"A1": -1, "L0": 0, "L1": 1, "L2": 2, "L3": 3}
MAIN_LEVELS = ("L0", "L1", "L2", "L3")


@dataclasses.dataclass(frozen=True)
class SeedSummary:
    count: int
    mean: float
    standard_deviation: float
    ci95_low: float
    ci95_high: float
    mean_sign_agreement: float
    true_support_fraction: float


def residual_energy_by_seed(residual: np.ndarray) -> np.ndarray:
    """Aggregate raw physical residuals to ``[seed, chunk, candidate]`` energy.

    The expected raw layout is ``[seed,chunk,candidate,flow_timestep,
    noise_sample,action_step,physical_action_dim]``.  Seeds deliberately remain
    independent analysis units.
    """
    residual = np.asarray(residual)
    if residual.ndim != 7 or residual.shape[-1] != 7:
        raise ValueError(f"Expected [S,P,C,F,N,H,7] residuals, got {residual.shape}")
    if not np.all(np.isfinite(residual)):
        raise ValueError("Residual tensor contains non-finite values")
    return np.mean(np.square(residual, dtype=np.float64), axis=(3, 4, 5, 6))


def canonical_and_truth_margins(
    energy: np.ndarray,
    *,
    target_candidate_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical and truth-aligned local margins for two candidates.

    Candidate order is canonical and fixed in the frozen manifest.  The
    canonical margin is ``E(candidate 1) - E(candidate 0)``; positive values
    support candidate 0.  The truth-aligned margin is positive exactly when the
    lower-residual candidate is the episode's true target.
    """
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 3 or energy.shape[2] != 2:
        raise ValueError(f"energy must have shape [seed,chunk,2], got {energy.shape}")
    if target_candidate_index not in (0, 1):
        raise ValueError("target_candidate_index must be 0 or 1")
    canonical = energy[:, :, 1] - energy[:, :, 0]
    truth = canonical if target_candidate_index == 0 else -canonical
    return canonical, truth


def prefix_scores(local_margin: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-seed C20/C30/C40 and three full-trajectory scores."""
    local_margin = np.asarray(local_margin, dtype=np.float64)
    if local_margin.ndim != 2 or local_margin.shape[1] != 10:
        raise ValueError(f"local_margin must have shape [seed,10], got {local_margin.shape}")
    result = {name: np.mean(local_margin[:, :count], axis=1) for name, count in PREFIX_CHUNKS.items()}
    progress = (np.arange(10, dtype=np.float64) + 0.5) / 10.0
    weights = {
        "flat_score": np.ones(10, dtype=np.float64),
        "linear_score": 1.0 - progress,
        "exponential_score": np.exp(-3.0 * progress),
    }
    for name, weight in weights.items():
        result[name] = np.sum(local_margin * weight[None, :], axis=1) / np.sum(weight)
    return result


def summarize_seed_values(values: np.ndarray) -> SeedSummary:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("At least one finite seed value is required")
    mean = float(np.mean(values))
    if len(values) == 1:
        standard_deviation = 0.0
        ci_low = ci_high = mean
    else:
        standard_deviation = float(np.std(values, ddof=1))
        half_width = float(stats.t.ppf(0.975, len(values) - 1) * standard_deviation / math.sqrt(len(values)))
        ci_low, ci_high = mean - half_width, mean + half_width
    mean_sign = np.sign(mean)
    # An exactly zero mean has no prediction sign; counting zero-valued seeds
    # as unanimous would incorrectly label an uninformative observer stable.
    agreement = 0.0 if mean_sign == 0.0 else float(np.mean(np.sign(values) == mean_sign))
    return SeedSummary(
        count=len(values),
        mean=mean,
        standard_deviation=standard_deviation,
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        mean_sign_agreement=agreement,
        true_support_fraction=float(np.mean(values > 0.0)),
    )


def summarize_episode_residuals(
    residual: np.ndarray,
    *,
    target_candidate_index: int,
) -> dict[str, Any]:
    """Create all local/prefix summaries from one raw residual tensor."""
    energy = residual_energy_by_seed(residual)
    canonical, truth = canonical_and_truth_margins(energy, target_candidate_index=target_candidate_index)
    scores = prefix_scores(truth)
    summary: dict[str, Any] = {
        "seed_count": int(residual.shape[0]),
        "chunk_count": int(residual.shape[1]),
        "target_candidate_index": target_candidate_index,
        "local_truth_margin": [dataclasses.asdict(summarize_seed_values(truth[:, index])) for index in range(10)],
    }
    for name, values in scores.items():
        seed_summary = summarize_seed_values(values)
        summary[name] = dataclasses.asdict(seed_summary)
        if name in PREFIX_CHUNKS:
            summary[f"{name}_correct"] = bool(seed_summary.mean > 0.0)
    return summary


def balanced_accuracy(rows: Sequence[Mapping[str, Any]], score_key: str) -> float:
    """Balanced accuracy over semantic target identities from signed predictions."""
    by_target: dict[str, list[bool]] = {}
    for row in rows:
        target = str(row["target_identity"])
        value = float(row[score_key])
        by_target.setdefault(target, []).append(value > 0.0)
    if len(by_target) != 2 or any(not values for values in by_target.values()):
        raise ValueError("Balanced accuracy requires both target identities")
    return float(np.mean([np.mean(values) for values in by_target.values()]))


def matched_monotonicity(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "C30",
    group_keys: Sequence[str] = ("geometry_id", "layout_id", "simulator_seed", "target_identity"),
) -> dict[str, Any]:
    """Summarize within-initialization L0--L3 ordering without pseudo-replication."""
    grouped: dict[tuple[object, ...], dict[str, float]] = {}
    for row in rows:
        level = str(row["legibility_level"])
        if level not in LEVEL_ORDER:
            raise ValueError(f"Unknown legibility level {level!r}")
        key = tuple(row[name] for name in group_keys)
        levels = grouped.setdefault(key, {})
        if level in levels:
            raise ValueError(f"Duplicate {level} row in matched group {key}")
        levels[level] = float(row[score_key])

    complete: list[tuple[tuple[object, ...], dict[str, float]]] = []
    for key, levels in grouped.items():
        if all(level in levels for level in MAIN_LEVELS):
            complete.append((key, levels))
    if not complete:
        raise ValueError("No complete matched L0-L3 groups")

    rhos: list[float] = []
    l3_minus_l0: list[float] = []
    adjacent: list[bool] = []
    l3_above_l0: list[bool] = []
    anti_minus_l0: list[float] = []
    group_rows: list[dict[str, Any]] = []
    x = np.arange(4, dtype=np.float64)
    for key, levels in complete:
        y = np.asarray([levels[level] for level in MAIN_LEVELS])
        if np.all(y == y[0]):
            raw_rho = float("nan")
            rho_defined = False
        else:
            raw_rho = float(stats.spearmanr(x, y).statistic)
            rho_defined = bool(np.isfinite(raw_rho))
        # A constant score sequence has undefined rank correlation.  Treat it
        # as zero evidence for the preregistered positive relationship so the
        # analysis reports a clean failure instead of emitting NaN JSON.
        rho = raw_rho if rho_defined else 0.0
        differences = np.diff(y)
        delta = float(y[-1] - y[0])
        rhos.append(rho)
        l3_minus_l0.append(delta)
        adjacent.extend(bool(value > 0.0) for value in differences)
        l3_above_l0.append(delta > 0.0)
        if "A1" in levels:
            anti_minus_l0.append(float(levels["A1"] - levels["L0"]))
        group_rows.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "spearman_rho": rho,
                "spearman_defined": rho_defined,
                "L3_minus_L0": delta,
                "adjacent_success_fraction": float(np.mean(differences > 0.0)),
                "A1_minus_L0": float(levels["A1"] - levels["L0"]) if "A1" in levels else None,
            }
        )
    return {
        "matched_group_count": len(complete),
        "mean_spearman_rho": float(np.mean(rhos)),
        "median_spearman_rho": float(np.median(rhos)),
        "undefined_spearman_group_count": sum(not row["spearman_defined"] for row in group_rows),
        "mean_L3_minus_L0": float(np.mean(l3_minus_l0)),
        "median_L3_minus_L0": float(np.median(l3_minus_l0)),
        "L3_above_L0_fraction": float(np.mean(l3_above_l0)),
        "adjacent_order_success_fraction": float(np.mean(adjacent)),
        "mean_A1_minus_L0": float(np.mean(anti_minus_l0)) if anti_minus_l0 else None,
        "groups": group_rows,
    }
