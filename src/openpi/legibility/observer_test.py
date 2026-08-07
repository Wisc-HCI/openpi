import numpy as np
import pytest

from openpi.legibility import observer


def test_equal_energies_keep_uniform_belief():
    belief_filter = observer.BeliefFilter(temperature=0.1)

    belief = belief_filter.update(np.asarray([0.2, 0.2]))

    np.testing.assert_allclose(belief, [0.5, 0.5])


def test_recursive_updates_accumulate_evidence_for_positive_instruction():
    belief_filter = observer.BeliefFilter(temperature=0.1)

    first = belief_filter.update(np.asarray([0.2, 0.24]))
    second = belief_filter.update(np.asarray([0.2, 0.24]))

    assert 0.5 < first[0] < second[0] < 1.0
    assert belief_filter.num_updates == 2


def test_reset_restores_uniform_prior():
    belief_filter = observer.BeliefFilter(temperature=0.1)
    belief_filter.update(np.asarray([0.1, 0.4]))

    belief_filter.reset()

    np.testing.assert_allclose(belief_filter.belief, [0.5, 0.5])
    assert belief_filter.num_updates == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": 0.1, "num_samples": 0}, "num_samples"),
        ({"temperature": 0.1, "tau_min": 0.8, "tau_max": 0.2}, "tau_min"),
        ({"temperature": 0.1, "action_dims": 0}, "action_dims"),
        ({"temperature": 0.1, "guidance_lambda": -1.0}, "guidance_lambda"),
    ],
)
def test_observer_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        observer.ObserverConfig(**kwargs)
