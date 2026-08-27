"""The load-time rejection matrix, and the artifact-identity facade behind it.

Everything here is refused at config load because none of it fails loudly
later: an unsatisfiable warm start is downgraded to a MISS with a log line, an
extra checkpoint is simply never consulted, and a library built by the wrong
pooling has identical dimensions to the right one.
"""

from __future__ import annotations

import pytest

from openpi.cache.cache_storage import CacheStorage
from openpi.cache.config import (
    CacheConfig,
    CheckpointConfig,
    ConfigValidationError,
    GateConfig,
    JudgeConfig,
    KeyBuilderConfig,
    WritePolicyConfig,
)
from openpi.cache.groot.load_guard import (
    validate_artifact_identity,
    validate_groot_cache_config,
)


def _config(**overrides) -> CacheConfig:
    config = CacheConfig()
    config.key_builder = KeyBuilderConfig(type="cp1_groot_spatial_pool_16")
    config.checkpoints = {
        "cp1": CheckpointConfig(
            enabled=True,
            gate=GateConfig(type="always_search"),
            judge=JudgeConfig(type="always_hit"),
        ),
        "cp3": CheckpointConfig(enabled=False),
    }
    config.write_policy = WritePolicyConfig(type="never")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _hysteresis_gate() -> GateConfig:
    """The N4 operating point this line serves (gate-threshold Pareto)."""
    return GateConfig(
        type="score_hysteresis",
        theta_low=0.9,
        theta_high=0.9,
        j=3,
        probe_interval=3,
        L=6,
    )


def test_a_valid_recipe_passes():
    validate_groot_cache_config(_config())


def test_cp3_enabled_is_refused():
    config = _config()
    config.checkpoints["cp3"] = CheckpointConfig(enabled=True)
    with pytest.raises(ConfigValidationError, match="cp3"):
        validate_groot_cache_config(config)


def test_warm_tiers_are_refused():
    config = _config()
    config.checkpoints["cp1"].judge.warm_tiers = [{"0.3": 0.9}]
    with pytest.raises(ConfigValidationError, match="warm_tiers"):
        validate_groot_cache_config(config)


def test_warm_start_judge_is_refused():
    config = _config()
    config.checkpoints["cp1"].judge = JudgeConfig(type="always_warm_start")
    with pytest.raises(ConfigValidationError, match="judge.type"):
        validate_groot_cache_config(config)


def test_non_always_search_gate_is_refused():
    config = _config()
    config.checkpoints["cp1"].gate = GateConfig(type="random", p_inference=0.5)
    with pytest.raises(ConfigValidationError, match="gate.type"):
        validate_groot_cache_config(config)


def test_hysteresis_gate_is_refused_by_default():
    # The default path is what exp/robocasa365/serve_groot_n15.py calls: that
    # module is untouched by the opt-in change and therefore cannot pass the
    # keyword at all, so "default rejects" is exactly "RoboCasa365 rejects".
    config = _config()
    config.checkpoints["cp1"].gate = _hysteresis_gate()
    with pytest.raises(ConfigValidationError, match="gate.type"):
        validate_groot_cache_config(config)


def test_hysteresis_gate_passes_only_when_opted_in():
    config = _config()
    config.checkpoints["cp1"].gate = _hysteresis_gate()
    validate_groot_cache_config(config, allow_hysteresis_gate=True)


@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize(
    "gate",
    [
        GateConfig(type="random", p_inference=0.5),
        GateConfig(type="periodic", cache_len=2, inference_len=2),
        GateConfig(type="always_skip"),
        GateConfig(type="client_controlled"),
    ],
)
def test_opt_in_admits_nothing_beyond_hysteresis(gate, allow):
    # The opt-in widens the allow-set by exactly one member. Without this the
    # flag could drift into a general "any gate" escape hatch, which is the
    # design that was rejected: the guard would stop owning the knowledge of
    # which gates can ever be served.
    config = _config()
    config.checkpoints["cp1"].gate = gate
    with pytest.raises(ConfigValidationError, match="gate.type"):
        validate_groot_cache_config(config, allow_hysteresis_gate=allow)


@pytest.mark.parametrize("allow", [False, True])
def test_the_other_four_guards_are_unaffected_by_the_opt_in(allow):
    # CP3 / warm tiers / judge type / write policy must behave identically in
    # both modes: the opt-in is about gates and nothing else.
    config = _config(write_policy=WritePolicyConfig(type="always"))
    config.checkpoints["cp3"].enabled = True
    config.checkpoints["cp1"].judge = JudgeConfig(type="warm_start")
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_groot_cache_config(config, allow_hysteresis_gate=allow)
    message = str(excinfo.value)
    assert "enabled checkpoints" in message
    assert "judge.type" in message
    assert "write_policy" in message


def test_online_write_policy_is_refused():
    config = _config(write_policy=WritePolicyConfig(type="on_any_miss"))
    with pytest.raises(ConfigValidationError, match="write_policy"):
        validate_groot_cache_config(config)


def test_every_problem_is_reported_at_once():
    config = _config(write_policy=WritePolicyConfig(type="always"))
    config.checkpoints["cp1"].gate = GateConfig(type="periodic", cache_len=2, inference_len=2)
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_groot_cache_config(config)
    message = str(excinfo.value)
    assert "gate.type" in message and "write_policy" in message


# ------------------------------------------------------------------
# Artifact identity
# ------------------------------------------------------------------


_ABSENT = object()


class _Backend:
    """Minimal backend stub. `artifact_meta` is present or absent, as in life."""

    def __init__(self, meta=_ABSENT) -> None:
        if meta is not _ABSENT:
            self.artifact_meta = meta


def _wrap(backend) -> CacheStorage:
    """A CacheStorage facade over a stub backend, without its real constructor."""
    storage = CacheStorage.__new__(CacheStorage)
    storage._backend = backend  # noqa: SLF001 - facade under test
    return storage


def test_matching_identity_passes():
    storage = _wrap(
        _Backend({"key_builder_type": "cp1_groot_spatial_pool_16", "checkpoint_id": "CP1"})
    )
    validate_artifact_identity(storage, _config())


def test_same_dimension_family_mismatch_is_caught():
    """mean-pool and max-pool artifacts share vector_dims; only the name differs."""
    config = _config()
    config.key_builder = KeyBuilderConfig(type="cp1_groot_mean_pool")
    storage = _wrap(
        _Backend({"key_builder_type": "cp1_groot_max_pool", "checkpoint_id": "CP1"})
    )
    with pytest.raises(ConfigValidationError, match="mean-pool and max-pool"):
        validate_artifact_identity(storage, config)


def test_pi05_artifact_under_a_groot_recipe_is_caught():
    storage = _wrap(
        _Backend({"key_builder_type": "cp1_spatial_pool_16", "checkpoint_id": "CP1"})
    )
    with pytest.raises(ConfigValidationError, match="cp1_spatial_pool_16"):
        validate_artifact_identity(storage, _config())


def test_wrong_checkpoint_id_is_caught():
    storage = _wrap(
        _Backend({"key_builder_type": "cp1_groot_spatial_pool_16", "checkpoint_id": "CP3"})
    )
    with pytest.raises(ConfigValidationError, match="checkpoint_id"):
        validate_artifact_identity(storage, _config())


def test_legacy_artifact_reads_as_a_dict_of_nones_and_is_refused():
    """A real legacy artifact: load_artifact ran, but the pkl had no identity."""
    storage = _wrap(_Backend({"key_builder_type": None, "checkpoint_id": None}))
    with pytest.raises(ConfigValidationError, match="predates identity recording"):
        validate_artifact_identity(storage, _config())


def test_backend_without_the_attribute_reads_as_none_and_is_refused():
    """Distinct from the legacy case: nothing was ever loaded."""
    storage = _wrap(_Backend())
    assert storage.artifact_meta is None
    with pytest.raises(ConfigValidationError, match="exposes no artifact identity"):
        validate_artifact_identity(storage, _config())


def test_facade_forwards_without_reaching_through():
    meta = {"key_builder_type": "cp1_groot_max_pool", "checkpoint_id": "CP1"}
    storage = _wrap(_Backend(meta))
    assert storage.artifact_meta == meta
    # A property on the facade, not an attribute callers dig out of the backend.
    assert isinstance(CacheStorage.artifact_meta, property)


# ------------------------------------------------------------------
# The identity path as it actually runs: a real pickle through load_artifact
# ------------------------------------------------------------------


def _write_artifact(path, *, identity: bool, dims=None):
    """A minimal but genuine artifact pickle, with or without the identity fields."""
    import pickle

    dims = dims or {"robot_state": 4}
    payload = {"vector_dims": dims, "entries": []}
    if identity:
        payload["key_builder_type"] = "cp1_groot_spatial_pool_16"
        payload["checkpoint_id"] = "CP1"
    path.write_bytes(pickle.dumps(payload))
    return dims


def _loaded_backend(path, dims):
    from openpi.cache.backends.in_memory_backend import InMemoryBackend

    backend = InMemoryBackend(vector_dims=dims)
    backend.load_artifact(str(path))
    return backend


def test_real_legacy_pickle_reads_back_as_a_dict_of_nones(tmp_path):
    """The shape that matters: load_artifact ran, but the file predates identity."""
    path = tmp_path / "legacy.pkl"
    dims = _write_artifact(path, identity=False)
    storage = _wrap(_loaded_backend(path, dims))

    # Builder identity reads back as None for a legacy pickle. The meta dict
    # also carries the content-identity fields (library_sha256, entry_count,
    # action schema, intermediates completeness) added by the dispatch-surface
    # line — those are computed from the file at load time, so subset-compare
    # the builder-identity keys instead of demanding dict equality.
    for key in ("key_builder_type", "checkpoint_id", "prompt_pool"):
        assert storage.artifact_meta[key] is None
    assert "library_sha256" in storage.artifact_meta
    with pytest.raises(ConfigValidationError, match="predates identity recording"):
        validate_artifact_identity(storage, _config())


def test_real_current_pickle_round_trips_its_identity(tmp_path):
    path = tmp_path / "current.pkl"
    dims = _write_artifact(path, identity=True)
    storage = _wrap(_loaded_backend(path, dims))

    assert storage.artifact_meta["key_builder_type"] == "cp1_groot_spatial_pool_16"
    validate_artifact_identity(storage, _config())


def test_a_never_loaded_backend_exposes_nothing(tmp_path):
    """Distinct from legacy: no artifact was read at all."""
    from openpi.cache.backends.in_memory_backend import InMemoryBackend

    storage = _wrap(InMemoryBackend(vector_dims={"robot_state": 4}))
    assert storage.artifact_meta is None
    with pytest.raises(ConfigValidationError, match="exposes no artifact identity"):
        validate_artifact_identity(storage, _config())
