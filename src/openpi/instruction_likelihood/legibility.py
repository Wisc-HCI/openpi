"""Pure NumPy trajectory geometry for spatial-legibility experiments.

The functions in this module deliberately do not depend on LIBERO or a policy
checkpoint.  This keeps trajectory construction and the geometry-only observer
usable as a pre-registration gate before any model residuals are inspected.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import numbers
from typing import Optional, Tuple, Union

import numpy as np

# Python 3.8 is used by the LIBERO rollout container, so these annotations
# intentionally use typing.Optional/Union/Tuple instead of PEP 604/585 syntax.
# ruff: noqa: UP006, UP007, UP035


# Ratios are signed with respect to the direction pointing away from the
# distractor.  The values are multiplied by the distance between the two goals.
LEGIBILITY_LEVELS = {
    "A1": -0.10,
    "L0": 0.00,
    "L1": 0.10,
    "L2": 0.20,
    "L3": 0.30,
}


@dataclasses.dataclass(frozen=True)
class BezierTrajectory:
    """One arc-length-resampled cubic Bezier trajectory.

    ``positions`` contains ``num_segments + 1`` points, including both
    endpoints.  ``progress`` is cumulative executed-path length normalized to
    ``[0, 1]``.  All point-like fields have the same dimensionality.
    """

    level: str
    deviation_ratio: float
    target_separation: float
    control_offset: float
    start: np.ndarray
    goal: np.ndarray
    distractor: np.ndarray
    away_direction: np.ndarray
    control1: np.ndarray
    control2: np.ndarray
    positions: np.ndarray
    progress: np.ndarray

    @property
    def num_segments(self) -> int:
        return int(self.positions.shape[0] - 1)


@dataclasses.dataclass(frozen=True)
class TrajectoryMetrics:
    """Geometry measurements for one executed trajectory."""

    path_length: float
    excess_length: float
    max_lateral_deviation: float
    max_deviation_progress: float
    min_distractor_distance: float


@dataclasses.dataclass(frozen=True)
class GeometricObserverResult:
    """Two-goal observer output at every trajectory waypoint.

    Columns of ``candidate_costs`` and ``candidate_confidence`` are ordered as
    ``[true goal, distractor]``.  Costs are excess path costs: the cost of the
    observed prefix plus a straight optimal completion, minus the initially
    optimal straight-line cost for that candidate.
    """

    progress: np.ndarray
    candidate_costs: np.ndarray
    candidate_confidence: np.ndarray
    true_goal_confidence: np.ndarray


def _as_point(name: str, value: Sequence[float]) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.ndim != 1 or point.size < 2:
        raise ValueError(f"{name} must be a one-dimensional point with at least two coordinates")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must contain only finite values")
    return point


def _as_path(name: str, value: Sequence[Sequence[float]]) -> np.ndarray:
    path = np.asarray(value, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 2:
        raise ValueError(f"{name} must have shape [at least 2 points, at least 2 coordinates]")
    if not np.all(np.isfinite(path)):
        raise ValueError(f"{name} must contain only finite values")
    return path


def _require_same_dimension(**points: np.ndarray) -> None:
    dimensions = {point.shape[0] for point in points.values()}
    if len(dimensions) != 1:
        shapes = {name: point.shape for name, point in points.items()}
        raise ValueError(f"All points must have the same dimensionality; got {shapes}")


def _positive_integer(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
    return int(value)


def _path_progress(positions: np.ndarray) -> Tuple[np.ndarray, float]:
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= np.finfo(np.float64).eps:
        raise ValueError("Trajectory must have positive finite path length")
    progress = cumulative / total
    progress[0] = 0.0
    progress[-1] = 1.0
    return progress, total


def _lateral_direction(start: np.ndarray, goal: np.ndarray, away_direction: np.ndarray) -> np.ndarray:
    # Spatial legibility is manipulated only in the table plane.  In 3-D robot
    # coordinates the final approach also changes height; projecting against
    # that vertical component would incorrectly count some z motion as lateral.
    delta = (goal - start).copy()
    away_in_plane = away_direction.copy()
    delta[2:] = 0.0
    away_in_plane[2:] = 0.0
    direct_distance = float(np.linalg.norm(delta))
    away_norm = float(np.linalg.norm(away_in_plane))
    if direct_distance <= np.finfo(np.float64).eps:
        raise ValueError("start and goal must be distinct in the table plane")
    if away_norm <= np.finfo(np.float64).eps:
        raise ValueError("away_direction must be nonzero in the table plane")

    parallel = delta / direct_distance
    away_unit = away_in_plane / away_norm
    lateral = away_unit - float(np.dot(away_unit, parallel)) * parallel
    lateral_norm = float(np.linalg.norm(lateral))
    # The scale-dependent threshold catches directions which are numerically,
    # rather than just exactly, parallel to the start-goal line.
    if lateral_norm <= 1e-10:
        raise ValueError("away_direction must have a table-plane component perpendicular to the start-goal line")
    return lateral / lateral_norm


def cubic_bezier(
    start: Sequence[float],
    control1: Sequence[float],
    control2: Sequence[float],
    goal: Sequence[float],
    parameter: Union[float, Sequence[float], np.ndarray],
) -> np.ndarray:
    """Evaluate a cubic Bezier curve at one or more parameters in ``[0, 1]``."""

    start_array = _as_point("start", start)
    control1_array = _as_point("control1", control1)
    control2_array = _as_point("control2", control2)
    goal_array = _as_point("goal", goal)
    _require_same_dimension(
        start=start_array,
        control1=control1_array,
        control2=control2_array,
        goal=goal_array,
    )

    parameter_array = np.asarray(parameter, dtype=np.float64)
    if parameter_array.ndim > 1:
        raise ValueError("parameter must be a scalar or one-dimensional array")
    if not np.all(np.isfinite(parameter_array)) or np.any(parameter_array < 0.0) or np.any(parameter_array > 1.0):
        raise ValueError("Bezier parameters must be finite and within [0, 1]")

    u = parameter_array[..., np.newaxis]
    one_minus_u = 1.0 - u
    return (
        one_minus_u**3 * start_array
        + 3.0 * one_minus_u**2 * u * control1_array
        + 3.0 * one_minus_u * u**2 * control2_array
        + u**3 * goal_array
    )


def arc_length_resample(points: Sequence[Sequence[float]], num_segments: int) -> np.ndarray:
    """Linearly resample a polyline into exactly ``num_segments`` arc intervals.

    Duplicate consecutive input points are allowed and discarded from the
    interpolation abscissa.  The returned first and last points are exact copies
    of the input endpoints.
    """

    path = _as_path("points", points)
    num_segments = _positive_integer("num_segments", num_segments)

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if not np.isfinite(total_length) or total_length <= np.finfo(np.float64).eps:
        raise ValueError("Cannot arc-length-resample a zero-length path")

    # np.interp expects strictly increasing sample positions.  Keep the last
    # occurrence so an endpoint duplicated in the input is still reproduced.
    keep = np.concatenate((np.diff(cumulative) > np.finfo(np.float64).eps, np.ones(1, dtype=bool)))
    unique_lengths = cumulative[keep]
    unique_points = path[keep]
    target_lengths = np.linspace(0.0, total_length, num_segments + 1, dtype=np.float64)
    resampled = np.column_stack(
        [np.interp(target_lengths, unique_lengths, unique_points[:, dimension]) for dimension in range(path.shape[1])]
    )
    resampled[0] = path[0]
    resampled[-1] = path[-1]
    return resampled


def signed_lateral_deviation(
    positions: Sequence[Sequence[float]],
    start: Sequence[float],
    goal: Sequence[float],
    away_direction: Sequence[float],
) -> np.ndarray:
    """Return signed table-plane displacement perpendicular to the start-goal line.

    Positive values point toward the perpendicular component of
    ``away_direction``; negative values point in the opposite direction.
    """

    path = _as_path("positions", positions)
    start_array = _as_point("start", start)
    goal_array = _as_point("goal", goal)
    away_array = _as_point("away_direction", away_direction)
    _require_same_dimension(start=start_array, goal=goal_array, away_direction=away_array)
    if path.shape[1] != start_array.shape[0]:
        raise ValueError("positions and reference points must have the same dimensionality")
    lateral = _lateral_direction(start_array, goal_array, away_array)
    return np.asarray((path - start_array) @ lateral, dtype=np.float64)


def max_signed_lateral_deviation(
    positions: Sequence[Sequence[float]],
    start: Sequence[float],
    goal: Sequence[float],
    away_direction: Sequence[float],
) -> float:
    """Return the largest-magnitude signed lateral deviation of a path."""

    deviations = signed_lateral_deviation(positions, start, goal, away_direction)
    index = int(np.argmax(np.abs(deviations)))
    value = float(deviations[index])
    scale = max(1.0, float(np.linalg.norm(_as_point("goal", goal) - _as_point("start", start))))
    if abs(value) <= 32.0 * np.finfo(np.float64).eps * scale:
        return 0.0
    return value


def _control_points(
    start: np.ndarray,
    goal: np.ndarray,
    away_unit: np.ndarray,
    control_offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    delta = goal - start
    control1 = start + 0.25 * delta + control_offset * away_unit
    control2 = start + 0.65 * delta
    return control1, control2


def _positions_for_offset(
    start: np.ndarray,
    goal: np.ndarray,
    away_unit: np.ndarray,
    control_offset: float,
    num_segments: int,
    curve_samples: int,
) -> np.ndarray:
    control1, control2 = _control_points(start, goal, away_unit, control_offset)
    parameters = np.linspace(0.0, 1.0, curve_samples, dtype=np.float64)
    dense_curve = cubic_bezier(start, control1, control2, goal, parameters)
    # The experimental manipulation is table-plane geometry only.  Resample
    # the XY curve by arc length, then assign every level the identical linear
    # profile in all non-planar dimensions.  Because every XY interval and
    # every Z interval are constant, the resulting 3-D waypoint spacing is also
    # uniform while L0--L3 have exactly the same vertical timing.
    planar = arc_length_resample(dense_curve[:, :2], num_segments)
    if dense_curve.shape[1] == 2:
        return planar
    nonplanar = np.column_stack(
        [
            np.linspace(start[dimension], goal[dimension], num_segments + 1, dtype=np.float64)
            for dimension in range(2, dense_curve.shape[1])
        ]
    )
    return np.concatenate((planar, nonplanar), axis=1)


def solve_control_offset(
    start: Sequence[float],
    goal: Sequence[float],
    away_direction: Sequence[float],
    target_deviation: float,
    *,
    num_segments: int = 100,
    curve_samples: int = 4097,
    tolerance: float = 1e-9,
    max_iterations: int = 80,
) -> float:
    """Numerically find the control offset for an executed signed deviation.

    The objective is measured after dense Bezier sampling and arc-length
    resampling, so ``target_deviation`` describes the actual returned waypoints,
    not the displacement of a control point.  A positive target deviates along
    ``away_direction`` and a negative target deviates against it.
    """

    start_array = _as_point("start", start)
    goal_array = _as_point("goal", goal)
    away_array = _as_point("away_direction", away_direction)
    _require_same_dimension(start=start_array, goal=goal_array, away_direction=away_array)
    num_segments = _positive_integer("num_segments", num_segments)
    curve_samples = _positive_integer("curve_samples", curve_samples, minimum=3)
    max_iterations = _positive_integer("max_iterations", max_iterations)
    if not np.isfinite(target_deviation):
        raise ValueError("target_deviation must be finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    # Validate the geometry even for L0, so changing the level later cannot turn
    # a previously accepted, ill-defined scene into a failing one.
    _lateral_direction(start_array, goal_array, away_array)
    away_norm = float(np.linalg.norm(away_array))
    away_unit = away_array / away_norm
    target_magnitude = abs(float(target_deviation))
    if target_magnitude == 0.0:
        return 0.0
    sign = 1.0 if target_deviation > 0.0 else -1.0

    def achieved_magnitude(offset_magnitude: float) -> float:
        positions = _positions_for_offset(
            start_array,
            goal_array,
            away_unit,
            sign * offset_magnitude,
            num_segments,
            curve_samples,
        )
        deviations = signed_lateral_deviation(positions, start_array, goal_array, away_unit)
        if sign > 0.0:
            return max(0.0, float(np.max(deviations)))
        return max(0.0, float(-np.min(deviations)))

    lower = 0.0
    upper = max(target_magnitude, float(np.linalg.norm(goal_array - start_array)), 1e-6)
    upper_value = achieved_magnitude(upper)
    for _ in range(64):
        if upper_value >= target_magnitude:
            break
        upper *= 2.0
        if not np.isfinite(upper):
            raise RuntimeError("Could not bracket a finite Bezier control offset")
        upper_value = achieved_magnitude(upper)
    else:
        raise RuntimeError("Could not bracket the requested lateral deviation")

    absolute_tolerance = max(float(tolerance), target_magnitude * 1e-10)
    midpoint = upper
    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        achieved = achieved_magnitude(midpoint)
        if abs(achieved - target_magnitude) <= absolute_tolerance:
            return sign * midpoint
        if achieved < target_magnitude:
            lower = midpoint
        else:
            upper = midpoint
    return sign * midpoint


def generate_legibility_trajectory(
    start: Sequence[float],
    true_goal: Sequence[float],
    distractor: Sequence[float],
    level: Union[str, float],
    *,
    away_direction: Optional[Sequence[float]] = None,
    num_segments: int = 100,
    curve_samples: int = 4097,
    solver_tolerance: float = 1e-9,
) -> BezierTrajectory:
    """Construct one of A1/L0/L1/L2/L3 as an executed waypoint path.

    A numeric ``level`` is accepted as a custom signed deviation ratio.  When an
    away direction is omitted, the vector from the distractor to the true goal is
    used, which is the natural direction for a two-target scene.
    """

    start_array = _as_point("start", start)
    goal_array = _as_point("true_goal", true_goal)
    distractor_array = _as_point("distractor", distractor)
    _require_same_dimension(start=start_array, true_goal=goal_array, distractor=distractor_array)
    num_segments = _positive_integer("num_segments", num_segments)
    curve_samples = _positive_integer("curve_samples", curve_samples, minimum=3)

    if isinstance(level, str):
        level_name = level.upper()
        if level_name not in LEGIBILITY_LEVELS:
            raise ValueError(f"Unknown legibility level {level!r}; expected one of {sorted(LEGIBILITY_LEVELS)}")
        deviation_ratio = float(LEGIBILITY_LEVELS[level_name])
    elif isinstance(level, numbers.Real) and not isinstance(level, bool) and np.isfinite(level):
        deviation_ratio = float(level)
        level_name = f"custom:{deviation_ratio:g}"
    else:
        raise ValueError("level must be a known level name or a finite numeric ratio")

    separation = float(np.linalg.norm(goal_array - distractor_array))
    if separation <= np.finfo(np.float64).eps:
        raise ValueError("true_goal and distractor must be distinct")
    away_array = (
        _as_point("away_direction", away_direction) if away_direction is not None else goal_array - distractor_array
    )
    _require_same_dimension(start=start_array, true_goal=goal_array, away_direction=away_array)
    _lateral_direction(start_array, goal_array, away_array)
    away_unit = away_array / np.linalg.norm(away_array)
    target_deviation = deviation_ratio * separation
    offset = solve_control_offset(
        start_array,
        goal_array,
        away_unit,
        target_deviation,
        num_segments=num_segments,
        curve_samples=curve_samples,
        tolerance=solver_tolerance,
    )
    control1, control2 = _control_points(start_array, goal_array, away_unit, offset)
    positions = _positions_for_offset(
        start_array,
        goal_array,
        away_unit,
        offset,
        num_segments,
        curve_samples,
    )
    progress, _ = _path_progress(positions)
    return BezierTrajectory(
        level=level_name,
        deviation_ratio=deviation_ratio,
        target_separation=separation,
        control_offset=float(offset),
        start=start_array.copy(),
        goal=goal_array.copy(),
        distractor=distractor_array.copy(),
        away_direction=away_unit.copy(),
        control1=control1,
        control2=control2,
        positions=positions,
        progress=progress,
    )


def compute_trajectory_metrics(
    positions: Sequence[Sequence[float]],
    start: Sequence[float],
    true_goal: Sequence[float],
    distractor: Sequence[float],
    away_direction: Sequence[float],
    *,
    l0_path_length: Optional[float] = None,
) -> TrajectoryMetrics:
    """Measure efficiency, lateral shape, and distractor clearance.

    If ``l0_path_length`` is omitted, the straight start-goal distance is used.
    For the specified L0 Bezier construction this equals its path length.  Passing
    the measured L0 length explicitly also supports controller-executed paths.
    """

    path = _as_path("positions", positions)
    start_array = _as_point("start", start)
    goal_array = _as_point("true_goal", true_goal)
    distractor_array = _as_point("distractor", distractor)
    away_array = _as_point("away_direction", away_direction)
    _require_same_dimension(
        start=start_array,
        true_goal=goal_array,
        distractor=distractor_array,
        away_direction=away_array,
    )
    if path.shape[1] != start_array.shape[0]:
        raise ValueError("positions and reference points must have the same dimensionality")
    progress, path_length = _path_progress(path)

    baseline = float(np.linalg.norm(goal_array - start_array)) if l0_path_length is None else float(l0_path_length)
    if not np.isfinite(baseline) or baseline <= np.finfo(np.float64).eps:
        raise ValueError("l0_path_length must be positive and finite")

    deviations = signed_lateral_deviation(path, start_array, goal_array, away_array)
    max_index = int(np.argmax(np.abs(deviations)))
    maximum = float(deviations[max_index])
    scale = max(1.0, float(np.linalg.norm(goal_array - start_array)))
    if abs(maximum) <= 32.0 * np.finfo(np.float64).eps * scale:
        maximum = 0.0
        max_progress = 0.0
    else:
        max_progress = float(progress[max_index])

    min_distractor_distance = float(np.min(np.linalg.norm(path - distractor_array, axis=1)))
    return TrajectoryMetrics(
        path_length=path_length,
        excess_length=(path_length - baseline) / baseline,
        max_lateral_deviation=maximum,
        max_deviation_progress=max_progress,
        min_distractor_distance=min_distractor_distance,
    )


def geometric_goal_observer(
    positions: Sequence[Sequence[float]],
    true_goal: Sequence[float],
    distractor: Sequence[float],
    *,
    beta: float = 1.0,
    candidate_priors: Sequence[float] = (0.5, 0.5),
) -> GeometricObserverResult:
    """Infer a two-goal belief from every observed trajectory prefix.

    For candidate ``g`` at waypoint ``x_t``, the observer uses

    ``C(prefix) + ||x_t - g|| - ||x_0 - g||``.

    This is the extra cost of explaining the prefix followed by an optimal
    straight completion, relative to acting optimally toward that candidate from
    the start.  Beliefs are proportional to ``prior * exp(-beta * cost)``.
    ``beta`` therefore has inverse-distance units (for metre coordinates, values
    such as 20--100 are often informative and should be fixed before evaluation).
    """

    path = _as_path("positions", positions)
    true_goal_array = _as_point("true_goal", true_goal)
    distractor_array = _as_point("distractor", distractor)
    _require_same_dimension(true_goal=true_goal_array, distractor=distractor_array)
    if path.shape[1] != true_goal_array.shape[0]:
        raise ValueError("positions and goals must have the same dimensionality")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")

    priors = np.asarray(candidate_priors, dtype=np.float64)
    if priors.shape != (2,) or not np.all(np.isfinite(priors)) or np.any(priors <= 0.0):
        raise ValueError("candidate_priors must contain two positive finite values")
    priors = priors / np.sum(priors)

    progress, _ = _path_progress(path)
    prefix_lengths = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )
    goals = np.stack((true_goal_array, distractor_array), axis=0)
    optimal_from_start = np.linalg.norm(goals - path[0], axis=1)
    remaining = np.linalg.norm(path[:, np.newaxis, :] - goals[np.newaxis, :, :], axis=2)
    candidate_costs = prefix_lengths[:, np.newaxis] + remaining - optimal_from_start[np.newaxis, :]
    # Triangle inequality makes these non-negative.  Remove only floating-point
    # underflow so the exact start cost is zero without masking invalid geometry.
    candidate_costs = np.maximum(candidate_costs, 0.0)

    logits = np.log(priors)[np.newaxis, :] - float(beta) * candidate_costs
    logits -= np.max(logits, axis=1, keepdims=True)
    candidate_confidence = np.exp(logits)
    candidate_confidence /= np.sum(candidate_confidence, axis=1, keepdims=True)
    return GeometricObserverResult(
        progress=progress,
        candidate_costs=candidate_costs,
        candidate_confidence=candidate_confidence,
        true_goal_confidence=candidate_confidence[:, 0].copy(),
    )


__all__ = [
    "LEGIBILITY_LEVELS",
    "BezierTrajectory",
    "GeometricObserverResult",
    "TrajectoryMetrics",
    "arc_length_resample",
    "compute_trajectory_metrics",
    "cubic_bezier",
    "generate_legibility_trajectory",
    "geometric_goal_observer",
    "max_signed_lateral_deviation",
    "signed_lateral_deviation",
    "solve_control_offset",
]
