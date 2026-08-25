import numpy as np

from openpi.instruction_likelihood import legibility_statistics as statistics


def test_truth_margin_and_prefix_scores_follow_candidate_order():
    energy = np.zeros((2, 10, 2), dtype=np.float64)
    energy[:, :, 0] = 1.0
    energy[:, :, 1] = np.arange(10)[None, :] + 2.0
    canonical, target_zero = statistics.canonical_and_truth_margins(energy, target_candidate_index=0)
    _, target_one = statistics.canonical_and_truth_margins(energy, target_candidate_index=1)
    np.testing.assert_array_equal(target_zero, canonical)
    np.testing.assert_array_equal(target_one, -canonical)
    scores = statistics.prefix_scores(target_zero)
    np.testing.assert_allclose(scores["C20"], 1.5)
    np.testing.assert_allclose(scores["C30"], 2.0)
    np.testing.assert_allclose(scores["C40"], 2.5)


def test_residual_energy_keeps_seed_chunk_and_candidate_axes():
    residual = np.ones((3, 10, 2, 5, 2, 10, 7), dtype=np.float32)
    residual[:, :, 1] *= 2.0
    energy = statistics.residual_energy_by_seed(residual)
    assert energy.shape == (3, 10, 2)
    np.testing.assert_allclose(energy[:, :, 0], 1.0)
    np.testing.assert_allclose(energy[:, :, 1], 4.0)


def test_seed_summary_reports_sign_stability_and_ci():
    summary = statistics.summarize_seed_values(np.array([1.0, 2.0, 3.0, -0.5]))
    assert summary.count == 4
    assert summary.mean > 0.0
    assert summary.mean_sign_agreement == 0.75
    assert summary.true_support_fraction == 0.75
    assert summary.ci95_low < summary.mean < summary.ci95_high


def test_zero_margin_is_uninformative_not_stable():
    summary = statistics.summarize_seed_values(np.zeros(10))
    assert summary.mean == 0.0
    assert summary.mean_sign_agreement == 0.0
    assert summary.true_support_fraction == 0.0


def test_matched_monotonicity_uses_complete_groups():
    rows = []
    for target in ("milk", "orange_juice"):
        for level, score in zip(statistics.MAIN_LEVELS, (0.0, 1.0, 2.0, 3.0), strict=True):
            rows.append(
                {
                    "geometry_id": "G1",
                    "layout_id": "A",
                    "simulator_seed": 0,
                    "target_identity": target,
                    "legibility_level": level,
                    "C30": score,
                }
            )
    result = statistics.matched_monotonicity(rows)
    assert result["matched_group_count"] == 2
    assert result["mean_spearman_rho"] == 1.0
    assert result["L3_above_L0_fraction"] == 1.0
    assert result["adjacent_order_success_fraction"] == 1.0


def test_constant_matched_scores_report_undefined_rho_as_failure_safe_zero():
    rows = [
        {
            "geometry_id": "G1",
            "layout_id": "A",
            "simulator_seed": 0,
            "target_identity": "milk",
            "legibility_level": level,
            "C30": 0.0,
        }
        for level in statistics.MAIN_LEVELS
    ]
    result = statistics.matched_monotonicity(rows)
    assert result["mean_spearman_rho"] == 0.0
    assert result["undefined_spearman_group_count"] == 1
    assert result["groups"][0]["spearman_defined"] is False
