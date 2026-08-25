import numpy as np
import pytest

from openpi.instruction_likelihood.legibility import LEGIBILITY_LEVELS
from openpi.instruction_likelihood.legibility import arc_length_resample
from openpi.instruction_likelihood.legibility import compute_trajectory_metrics
from openpi.instruction_likelihood.legibility import cubic_bezier
from openpi.instruction_likelihood.legibility import generate_legibility_trajectory
from openpi.instruction_likelihood.legibility import geometric_goal_observer
from openpi.instruction_likelihood.legibility import max_signed_lateral_deviation
from openpi.instruction_likelihood.legibility import solve_control_offset

START = np.array([0.0, 0.0, 0.25])
TRUE_GOAL = np.array([0.40, 0.05, 0.10])
DISTRACTOR = np.array([0.40, -0.05, 0.10])
AWAY = TRUE_GOAL - DISTRACTOR


def test_legibility_levels_hit_actual_executed_deviation_and_have_exact_segment_count():
    separation = np.linalg.norm(TRUE_GOAL - DISTRACTOR)
    trajectories = {
        level: generate_legibility_trajectory(START, TRUE_GOAL, DISTRACTOR, level, num_segments=100)
        for level in LEGIBILITY_LEVELS
    }

    for level, ratio in LEGIBILITY_LEVELS.items():
        trajectory = trajectories[level]
        assert trajectory.positions.shape == (101, 3)
        np.testing.assert_array_equal(trajectory.positions[0], START)
        np.testing.assert_array_equal(trajectory.positions[-1], TRUE_GOAL)
        assert trajectory.progress[0] == 0.0
        assert trajectory.progress[-1] == 1.0
        measured = max_signed_lateral_deviation(trajectory.positions, START, TRUE_GOAL, AWAY)
        assert measured == pytest.approx(ratio * separation, abs=2e-8)

    metrics = {
        level: compute_trajectory_metrics(
            trajectory.positions,
            START,
            TRUE_GOAL,
            DISTRACTOR,
            trajectory.away_direction,
        )
        for level, trajectory in trajectories.items()
    }
    assert metrics["L0"].path_length < metrics["L1"].path_length
    assert metrics["L1"].path_length < metrics["L2"].path_length < metrics["L3"].path_length
    assert metrics["L0"].excess_length == pytest.approx(0.0, abs=1e-12)


def test_control_points_follow_registered_bezier_definition():
    trajectory = generate_legibility_trajectory(START, TRUE_GOAL, DISTRACTOR, "L2", num_segments=20)
    delta = TRUE_GOAL - START
    np.testing.assert_allclose(
        trajectory.control1,
        START + 0.25 * delta + trajectory.control_offset * trajectory.away_direction,
    )
    np.testing.assert_allclose(trajectory.control2, START + 0.65 * delta)

    sampled = cubic_bezier(START, trajectory.control1, trajectory.control2, TRUE_GOAL, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(sampled[0], START)
    np.testing.assert_array_equal(sampled[1], TRUE_GOAL)


def test_negative_deviation_solver_builds_anti_legible_path():
    offset = solve_control_offset(START, TRUE_GOAL, AWAY, -0.01, num_segments=37)
    assert offset < 0.0
    executed = generate_legibility_trajectory(START, TRUE_GOAL, DISTRACTOR, -0.10, num_segments=37)
    assert max_signed_lateral_deviation(executed.positions, START, TRUE_GOAL, AWAY) == pytest.approx(-0.01, abs=2e-8)


def test_all_levels_share_exact_vertical_profile_and_uniform_3d_spacing():
    trajectories = [
        generate_legibility_trajectory(START, TRUE_GOAL, DISTRACTOR, level, num_segments=100)
        for level in LEGIBILITY_LEVELS
    ]
    expected_z = np.linspace(START[2], TRUE_GOAL[2], 101)
    for trajectory in trajectories:
        np.testing.assert_allclose(trajectory.positions[:, 2], expected_z, atol=1e-15)
        segment_lengths = np.linalg.norm(np.diff(trajectory.positions, axis=0), axis=1)
        assert np.ptp(segment_lengths) < 2e-6


def test_arc_length_resample_returns_equal_intervals_for_line_and_handles_duplicates():
    points = np.array([[0.0, 0.0], [0.0, 0.0], [0.2, 0.0], [1.0, 0.0], [1.0, 0.0]])
    resampled = arc_length_resample(points, 10)
    assert resampled.shape == (11, 2)
    np.testing.assert_allclose(np.linalg.norm(np.diff(resampled, axis=0), axis=1), 0.1)
    np.testing.assert_array_equal(resampled[[0, -1]], points[[0, -1]])


def test_metrics_use_measured_l0_and_report_clearance_and_deviation_progress():
    l0 = generate_legibility_trajectory(START, TRUE_GOAL, DISTRACTOR, "L0", num_segments=100)
    l2 = generate_legibility_trajectory(START, TRUE_GOAL, DISTRACTOR, "L2", num_segments=100)
    l0_metrics = compute_trajectory_metrics(l0.positions, START, TRUE_GOAL, DISTRACTOR, AWAY)
    l2_metrics = compute_trajectory_metrics(
        l2.positions,
        START,
        TRUE_GOAL,
        DISTRACTOR,
        AWAY,
        l0_path_length=l0_metrics.path_length,
    )
    assert l2_metrics.excess_length > 0.0
    assert l2_metrics.max_lateral_deviation == pytest.approx(0.02, abs=2e-8)
    assert 0.20 < l2_metrics.max_deviation_progress < 0.40
    expected_clearance = np.min(np.linalg.norm(l2.positions - DISTRACTOR, axis=1))
    assert l2_metrics.min_distractor_distance == pytest.approx(expected_clearance)


def test_geometric_observer_starts_at_prior_and_legible_path_reveals_true_goal_earlier():
    straight = generate_legibility_trajectory(START, TRUE_GOAL, DISTRACTOR, "L0", num_segments=100)
    legible = generate_legibility_trajectory(START, TRUE_GOAL, DISTRACTOR, "L3", num_segments=100)
    straight_result = geometric_goal_observer(straight.positions, TRUE_GOAL, DISTRACTOR, beta=50.0)
    legible_result = geometric_goal_observer(legible.positions, TRUE_GOAL, DISTRACTOR, beta=50.0)

    assert straight_result.candidate_costs.shape == (101, 2)
    assert straight_result.candidate_confidence.shape == (101, 2)
    np.testing.assert_allclose(straight_result.candidate_confidence.sum(axis=1), 1.0)
    np.testing.assert_allclose(straight_result.candidate_costs[0], 0.0, atol=1e-15)
    assert straight_result.true_goal_confidence[0] == pytest.approx(0.5)
    assert legible_result.true_goal_confidence[20] > straight_result.true_goal_confidence[20]
    assert legible_result.true_goal_confidence[30] > straight_result.true_goal_confidence[30]
    assert straight_result.true_goal_confidence[-1] > 0.99


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: arc_length_resample(np.zeros((2, 3)), 10), "zero-length"),
        (lambda: solve_control_offset(START, START, AWAY, 0.01), "distinct"),
        (lambda: solve_control_offset(START, TRUE_GOAL, TRUE_GOAL - START, 0.01), "perpendicular"),
        (
            lambda: generate_legibility_trajectory(START, TRUE_GOAL, DISTRACTOR, "L1", away_direction=np.zeros(3)),
            "nonzero",
        ),
        (lambda: generate_legibility_trajectory(START, TRUE_GOAL, TRUE_GOAL, "L1"), "distinct"),
        (lambda: generate_legibility_trajectory(START, TRUE_GOAL, DISTRACTOR, "unknown"), "Unknown"),
        (lambda: geometric_goal_observer(np.stack((START, TRUE_GOAL)), TRUE_GOAL, DISTRACTOR, beta=0.0), "beta"),
    ],
)
def test_invalid_geometry_is_rejected(call, message):
    with pytest.raises(ValueError, match=message):
        call()
