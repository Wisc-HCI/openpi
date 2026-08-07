import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.legibility import observer as _observer
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class _FakeModel:
    action_horizon = 2
    action_dim = 3

    def __init__(self):
        self.call = None
        self.score_call = None

    def sample_actions(self, rng, observation, **kwargs):
        self.call = (observation, kwargs)
        return jnp.zeros((1, 2, 3), dtype=jnp.float32)

    def score_actions(self, rng, positive_observation, negative_observation, actions, executed_steps, **kwargs):
        self.score_call = (positive_observation, negative_observation, actions, executed_steps, kwargs)
        return jnp.asarray([[0.1, 0.2]], dtype=jnp.float32)


def test_fixed_negative_prompt_is_transformed_and_passed_to_sampling(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda method: method)

    def tokenize_prompt(data):
        prompt = data.pop("prompt")
        token = 1 if prompt == "positive instruction" else 2
        return {
            **data,
            "tokenized_prompt": np.asarray([token], dtype=np.int32),
            "tokenized_prompt_mask": np.asarray([True]),
        }

    policy = _policy.Policy(
        model,
        transforms=[tokenize_prompt],
        sample_kwargs={"guidance_scale": 1.5},
        negative_prompt="negative instruction",
    )
    observation = {
        "image": {"camera": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"camera": np.ones((), dtype=np.bool_)},
        "state": np.zeros(3, dtype=np.float32),
        "prompt": "positive instruction",
    }

    result = policy.infer(observation)

    assert result["actions"].shape == (2, 3)
    positive_observation, sample_kwargs = model.call
    negative_observation = sample_kwargs["negative_observation"]
    np.testing.assert_array_equal(positive_observation.tokenized_prompt, [[1]])
    np.testing.assert_array_equal(negative_observation.tokenized_prompt, [[2]])
    np.testing.assert_array_equal(negative_observation.state, positive_observation.state)
    assert sample_kwargs["guidance_scale"] == 1.5


def test_online_observer_updates_belief_and_guidance_scale(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda method, **kwargs: method)

    def tokenize_prompt(data):
        prompt = data.pop("prompt")
        token = 1 if prompt == "positive instruction" else 2
        return {
            **data,
            "tokenized_prompt": np.asarray([token], dtype=np.int32),
            "tokenized_prompt_mask": np.asarray([True]),
        }

    policy = _policy.Policy(
        model,
        transforms=[tokenize_prompt],
        negative_prompt="negative instruction",
        observer_config=_observer.ObserverConfig(
            temperature=0.1,
            num_samples=2,
            tau_min=0.3,
            tau_max=0.7,
            action_dims=3,
            belief_weighted=True,
            guidance_lambda=1.0,
        ),
    )
    observation = {
        "image": {"camera": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"camera": np.ones((), dtype=np.bool_)},
        "state": np.zeros(3, dtype=np.float32),
        "prompt": "positive instruction",
        "observer_reset": True,
    }

    first = policy.infer(observation)
    second_observation = {
        **observation,
        "observer_reset": False,
        "previous_executed_actions": np.full((1, 3), 0.25, dtype=np.float32),
    }
    second = policy.infer(second_observation)

    assert first["legibility"]["belief_negative"] == pytest.approx(0.5)
    assert first["legibility"]["effective_guidance_scale"] == pytest.approx(0.5)
    assert second["legibility"]["belief_negative"] < 0.5
    assert second["legibility"]["effective_guidance_scale"] == pytest.approx(second["legibility"]["belief_negative"])
    assert second["legibility"]["observer_updates"] == 1
    _, _, scored_actions, executed_steps, score_kwargs = model.score_call
    np.testing.assert_allclose(scored_actions[0, 0], 0.25)
    np.testing.assert_allclose(scored_actions[0, 1], 0.0)
    assert int(executed_steps) == 1
    assert score_kwargs["num_samples"] == 2


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
