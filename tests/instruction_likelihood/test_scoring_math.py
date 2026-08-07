import numpy as np

from openpi.instruction_likelihood.scorer import aggregate_residuals
from openpi.instruction_likelihood.scorer import cumulative_posteriors
from openpi.instruction_likelihood.statistics import paired_lower_test


def test_aggregate_residuals_uses_all_and_only_physical_dimensions():
    residual = np.ones((2, 3, 4, 2, 10, 7), dtype=np.float32)
    residual[:, 1] *= 2
    energy = aggregate_residuals(residual)
    np.testing.assert_allclose(energy[:, 0], 1.0)
    np.testing.assert_allclose(energy[:, 1], 4.0)


def test_cumulative_posterior_excludes_mismatch_and_accumulates_evidence():
    energies = np.array([[1.0, 2.0, 100.0], [1.0, 2.0, 100.0]])
    posterior = cumulative_posteriors(energies, np.array([True, True, False]), temperature=1.0)
    assert posterior[1, 0] > posterior[0, 0]
    np.testing.assert_array_equal(posterior[:, 2], 0.0)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0)


def test_paired_gate_requires_direction_and_significance():
    result = paired_lower_test(np.zeros(10), np.arange(1.0, 11.0))
    assert result.passed
    assert result.median_difference > 0


def test_quantile_inverse_formula_round_trips():
    q01 = np.array([-2.0, 0.0])
    q99 = np.array([2.0, 4.0])
    normalized = np.array([[-1.0, 0.0], [1.0, 1.0]])
    raw = (normalized + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    recovered = (raw - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    np.testing.assert_allclose(recovered, normalized)
