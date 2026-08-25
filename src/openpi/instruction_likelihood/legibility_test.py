import numpy as np

from openpi.instruction_likelihood import legibility


def _scene():
    start = np.array([0.35, 0.0, 1.15])
    true_goal = np.array([0.0, 0.05, 0.98])
    distractor = np.array([0.0, -0.05, 0.98])
    return start, true_goal, distractor


def test_registered_levels_hit_executed_deviation_and_have_100_segments():
    start, true_goal, distractor = _scene()
    for level, ratio in legibility.LEGIBILITY_LEVELS.items():
        trajectory = legibility.generate_legibility_trajectory(start, true_goal, distractor, level)
        assert trajectory.positions.shape == (101, 3)
        deviation = legibility.max_signed_lateral_deviation(
            trajectory.positions, start, true_goal, trajectory.away_direction
        )
        np.testing.assert_allclose(deviation, ratio * 0.1, atol=2e-8)


def test_path_cost_and_geometric_evidence_increase_from_l0_to_l3():
    start, true_goal, distractor = _scene()
    trajectories = {
        level: legibility.generate_legibility_trajectory(start, true_goal, distractor, level)
        for level in ("L0", "L1", "L2", "L3")
    }
    lengths = [
        legibility.compute_trajectory_metrics(
            trajectories[level].positions,
            start,
            true_goal,
            distractor,
            trajectories[level].away_direction,
        ).path_length
        for level in ("L0", "L1", "L2", "L3")
    ]
    assert np.all(np.diff(lengths) > 0.0)
    confidence = {
        level: legibility.geometric_goal_observer(
            trajectory.positions, true_goal, distractor, beta=40.0
        ).true_goal_confidence[30]
        for level, trajectory in trajectories.items()
    }
    assert confidence["L3"] > confidence["L2"] > confidence["L1"] > confidence["L0"]
