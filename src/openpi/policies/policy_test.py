from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi import transforms as _transforms
from openpi.legibility import observer as _observer
from openpi.models import model as _model
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

    def sample_actions(self, rng, observation, *, negative_observation=None, **kwargs):
        if negative_observation is not None:
            kwargs["negative_observation"] = negative_observation
        self.call = (observation, kwargs)
        return jnp.zeros((1, 2, 3), dtype=jnp.float32)

    def score_actions(self, rng, positive_observation, negative_observation, actions, executed_steps, **kwargs):
        self.score_call = (positive_observation, negative_observation, actions, executed_steps, kwargs)
        return jnp.asarray([[0.1, 0.2]], dtype=jnp.float32)


def test_fixed_negative_prompt_is_transformed_and_passed_to_sampling(monkeypatch):
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
        guidance_zero_first_chunk=True,
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
    assert first["legibility"]["effective_guidance_scale"] == pytest.approx(0.0)
    assert first["guidance"]["query_index"] == 0
    assert second["legibility"]["belief_negative"] < 0.5
    assert second["legibility"]["effective_guidance_scale"] == pytest.approx(second["legibility"]["belief_negative"])
    assert second["guidance"]["query_index"] == 1
    assert second["legibility"]["observer_updates"] == 1
    _, _, scored_actions, executed_steps, score_kwargs = model.score_call
    np.testing.assert_allclose(scored_actions[0, 0], 0.25)
    np.testing.assert_allclose(scored_actions[0, 1], 0.0)
    assert int(executed_steps) == 1
    assert score_kwargs["num_samples"] == 2


def test_guidance_decay_is_per_query_and_resets(monkeypatch):
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
        sample_kwargs={"guidance_scale": 2.0},
        negative_prompt="negative instruction",
        guidance_decay=0.5,
    )
    observation = {
        "image": {"camera": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"camera": np.ones((), dtype=np.bool_)},
        "state": np.zeros(3, dtype=np.float32),
        "prompt": "positive instruction",
        "sampling_seed": 123,
    }

    first = policy.infer({**observation, "observer_reset": True})
    second = policy.infer(observation)
    reset = policy.infer({**observation, "observer_reset": True})

    assert first["guidance"]["effective_guidance_scale"] == pytest.approx(2.0)
    assert second["guidance"]["effective_guidance_scale"] == pytest.approx(1.0)
    assert reset["guidance"]["effective_guidance_scale"] == pytest.approx(2.0)
    assert first["guidance"]["sampling_seed"] == 123


def _request_observation(**overrides):
    return {
        "image": {"camera": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"camera": np.ones((), dtype=np.bool_)},
        "state": np.zeros(3, dtype=np.float32),
        "prompt": "positive instruction",
        **overrides,
    }


def _tokenize_request_prompt(data):
    # Check that request metadata is removed before transforms or JAX batching.
    assert "negative_prompt" not in data
    assert "steering" not in data
    assert "rollout_id" not in data
    token = {"positive instruction": 1, "server negative": 2, "client negative": 3}[data.pop("prompt")]
    return {
        **data,
        "tokenized_prompt": np.asarray([token], dtype=np.int32),
        "tokenized_prompt_mask": np.asarray([True]),
    }


@pytest.fixture
def request_model(monkeypatch):
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda method, **kwargs: method)
    return _FakeModel()


def test_request_negative_overrides_default_without_leaking_to_later_requests(request_model):
    policy = _policy.Policy(
        request_model,
        transforms=[_tokenize_request_prompt],
        negative_prompt="server negative",
        sample_kwargs={"guidance_scale": 2.0},
        guidance_decay=0.5,
    )
    obs = _request_observation(negative_prompt="client negative")
    first = policy.infer(obs)
    positive, kwargs = request_model.call
    np.testing.assert_array_equal(positive.tokenized_prompt, [[1]])
    np.testing.assert_array_equal(kwargs["negative_observation"].tokenized_prompt, [[3]])
    np.testing.assert_array_equal(kwargs["negative_observation"].state, positive.state)
    np.testing.assert_array_equal(kwargs["negative_observation"].images["camera"], positive.images["camera"])
    assert kwargs["guidance_scale"] == 2.0
    assert first["guidance"]["negative_prompt"] == "client negative"
    assert obs["negative_prompt"] == "client negative"
    assert policy.metadata["supports_request_negative_prompt"]

    second = policy.infer(obs)
    assert second["guidance"]["effective_guidance_scale"] == 1.0
    fallback = policy.infer(_request_observation())
    np.testing.assert_array_equal(request_model.call[1]["negative_observation"].tokenized_prompt, [[2]])
    assert fallback["guidance"]["negative_prompt"] == "server negative"
    assert fallback["guidance"]["query_index"] == 0
    assert fallback["guidance"]["effective_guidance_scale"] == 2.0


def test_request_enables_guidance_without_server_negative_and_omission_disables_it(request_model):
    policy = _policy.Policy(request_model, transforms=[_tokenize_request_prompt])
    normal = policy.infer(_request_observation())
    assert "negative_observation" not in request_model.call[1]
    assert "guidance_scale" not in request_model.call[1]
    assert normal["guidance"]["effective_guidance_scale"] == 0.0

    guided = policy.infer(_request_observation(negative_prompt="client negative"))
    assert request_model.call[1]["guidance_scale"] == 1.0
    assert guided["guidance"]["query_index"] == 0
    normal_again = policy.infer(_request_observation())
    assert "negative_observation" not in request_model.call[1]
    assert normal_again["guidance"]["negative_prompt"] is None
    assert normal_again["guidance"]["effective_guidance_scale"] == 0.0


@pytest.mark.parametrize("bad_prompt", ["", "  ", 123, ["client negative"]])
def test_request_rejects_invalid_negative_prompt(request_model, bad_prompt):
    policy = _policy.Policy(request_model, transforms=[_tokenize_request_prompt])
    with pytest.raises(ValueError, match="non-empty string"):
        policy.infer(_request_observation(negative_prompt=bad_prompt))
    assert request_model.call is None


def test_request_negative_rejects_unsupported_model(monkeypatch):
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda method: method)
    model = SimpleNamespace(sample_actions=lambda rng, observation: None)
    policy = _policy.Policy(model)
    assert not policy.metadata["supports_request_negative_prompt"]
    with pytest.raises(ValueError, match="JAX Pi0/Pi0.5"):
        policy.infer(_request_observation(negative_prompt="client negative"))


@pytest.mark.parametrize("changed_prompt", [{"negative_prompt": "server negative"}, {"prompt": "server negative"}])
def test_observer_uses_request_prompt_and_resets_when_either_instruction_changes(request_model, changed_prompt):
    policy = _policy.Policy(
        request_model,
        transforms=[_tokenize_request_prompt],
        observer_config=_observer.ObserverConfig(temperature=0.1, action_dims=3, belief_weighted=True),
        guidance_zero_first_chunk=True,
    )
    obs = _request_observation(negative_prompt="client negative")
    first = policy.infer(obs)
    assert first["guidance"]["effective_guidance_scale"] == 0.0
    actions = np.ones((1, 3), dtype=np.float32)
    second = policy.infer({**obs, "previous_executed_actions": actions})
    np.testing.assert_array_equal(request_model.score_call[1].tokenized_prompt, [[3]])
    assert second["legibility"]["observer_updates"] == 1
    assert second["legibility"]["belief_negative"] < 0.5

    reset_obs = {**obs, **changed_prompt, "previous_executed_actions": actions}
    reset = policy.infer(reset_obs)
    assert reset["legibility"]["observer_updates"] == 0
    assert reset["legibility"]["belief_negative"] == 0.5
    assert reset["guidance"]["query_index"] == 0
    assert reset["guidance"]["effective_guidance_scale"] == 0.0
    resumed = policy.infer(reset_obs)
    assert resumed["legibility"]["observer_updates"] == 1
    expected_negative_token = 2 if "negative_prompt" in changed_prompt else 3
    np.testing.assert_array_equal(request_model.score_call[1].tokenized_prompt, [[expected_negative_token]])


def test_observer_requires_request_prompt_when_no_default_is_set(request_model):
    policy = _policy.Policy(request_model, observer_config=_observer.ObserverConfig(temperature=0.1, action_dims=3))
    with pytest.raises(ValueError, match="request or as a server default"):
        policy.infer(_request_observation())


@pytest.mark.parametrize("scale", [0.0, 2.5])
@pytest.mark.parametrize("with_observer", [False, True])
def test_factory_retains_guidance_scale_without_default_negative(
    request_model, monkeypatch, tmp_path, scale, with_observer
):
    data_config = SimpleNamespace(
        data_transforms=_transforms.Group(),
        model_transforms=_transforms.Group(inputs=[_tokenize_request_prompt]),
        use_quantile_norm=False,
    )
    config = SimpleNamespace(
        model=SimpleNamespace(model_type=_model.ModelType.PI05, load=lambda _: request_model, action_dim=3),
        data=SimpleNamespace(create=lambda *_: data_config),
        assets_dirs=tmp_path,
        policy_metadata={},
    )
    monkeypatch.setattr(_policy_config.download, "maybe_download", lambda _: tmp_path)
    monkeypatch.setattr(_model, "restore_params", lambda *args, **kwargs: {})
    observer_config = _observer.ObserverConfig(temperature=0.1, action_dims=3) if with_observer else None
    policy = _policy_config.create_trained_policy(
        config, tmp_path, norm_stats={}, guidance_scale=scale, observer_config=observer_config
    )
    result = policy.infer(_request_observation(negative_prompt="client negative"))
    assert result["guidance"]["effective_guidance_scale"] == scale
    assert ("negative_observation" in request_model.call[1]) == (scale > 0)


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
