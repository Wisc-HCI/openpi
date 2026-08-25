import numpy as np

from openpi.instruction_likelihood import scorer


def test_seeded_noise_is_subset_stable_for_resumed_scoring():
    kwargs = {
        "chunk_index": 3,
        "flow_count": 2,
        "noise_samples": 4,
        "action_horizon": 10,
        "model_action_dim": 32,
    }
    together = scorer._seeded_common_noise(np.array([2, 7]), **kwargs)  # noqa: SLF001
    resumed = scorer._seeded_common_noise(np.array([7]), **kwargs)  # noqa: SLF001
    np.testing.assert_array_equal(together[1], resumed[0])
    assert not np.array_equal(together[0], together[1])


def test_aggregate_seeded_residuals_preserves_seed_axis():
    residual = np.ones((10, 3, 2, 5, 2, 10, 7), dtype=np.float32)
    residual[:, :, 1] *= 2.0
    energy = scorer.aggregate_seeded_residuals(residual)
    assert energy.shape == (10, 3, 2)
    np.testing.assert_allclose(energy[:, :, 0], 1.0)
    np.testing.assert_allclose(energy[:, :, 1], 4.0)
