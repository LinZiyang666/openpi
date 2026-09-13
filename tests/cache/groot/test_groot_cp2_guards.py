"""Load-time rejection matrix for the GR00T CP2 arm, and the CP2 artifact binding.

The GR00T guard now admits exactly one of two mutually exclusive checkpoint
sets, {cp1} or {cp2}; a CP2 recipe must name the GR00T CP2 builder, and every
warm-start rule reads the enabled checkpoint instead of a fixed ``cp1``. The
storage choke point then binds the artifact's projection *including the
layout block* to the configured builder, both for a fresh ``build_shared_storage``
and for the concurrent / hot-load assembly that delegates to it.
"""

from __future__ import annotations

import copy
import pickle

import pytest
import torch
import yaml

from exp.actioncache_baseline import libs
from exp.actioncache_baseline.export_arms import assert_arm_yaml, cp2_arm_yaml
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.cp2_vlm_key_builder import get_projection_spec
from openpi.cache.config import (
    CacheConfig,
    CheckpointConfig,
    ConfigValidationError,
    GateConfig,
    JudgeConfig,
    KeyBuilderConfig,
    WritePolicyConfig,
    build_cache_components,
    build_shared_storage,
    cp2_expected_projection_meta,
    load_cache_config,
    required_warm_timesteps,
)
from openpi.cache.groot.cp2_key_builder import GrootCP2TernaryKeyBuilder
from openpi.cache.groot.load_guard import (
    enabled_checkpoint_names,
    single_enabled_checkpoint,
    validate_artifact_identity,
    validate_groot_cache_config,
)
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

GROOT = libs.GROOT_LIBERO
PROJ = libs.ProjectionArgs(seed=7, d=8, p=0.25, input_dim=5 * 4 + 2,
                           layout={"kind": "groot_encoded_v1", "token_len": 5, "feature_dim": 4, "state_feat_dim": 2})
PI05_PROJ = libs.ProjectionArgs(seed=7, d=8, p=0.25, input_dim=64)
K8 = "groot_n15_k8_v1"


# ------------------------------------------------------------------
# validate_groot_cache_config: the {cp2} arm
# ------------------------------------------------------------------


def _cp2_config(**overrides) -> CacheConfig:
    config = CacheConfig()
    config.key_builder = KeyBuilderConfig(type="cp2_groot_ternary")
    config.checkpoints = {
        "cp2": CheckpointConfig(enabled=True, gate=GateConfig(type="always_search"),
                                judge=JudgeConfig(type="threshold", threshold=0.8)),
    }
    config.write_policy = WritePolicyConfig(type="never")
    config.denoise_schedule = K8  # both tiers name the teacher loop (G2-B3)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _cp1_config() -> CacheConfig:
    config = CacheConfig()
    config.key_builder = KeyBuilderConfig(type="cp1_groot_spatial_pool_16")
    config.checkpoints = {
        "cp1": CheckpointConfig(enabled=True, gate=GateConfig(type="always_search"),
                                judge=JudgeConfig(type="always_hit")),
        "cp3": CheckpointConfig(enabled=False),
    }
    config.write_policy = WritePolicyConfig(type="never")
    return config


def test_cp2_n0_recipe_passes():
    validate_groot_cache_config(_cp2_config())
    validate_groot_cache_config(_cp2_config(), num_inference_timesteps=8)


def test_cp2_n0_recipe_must_name_the_teacher_loop_and_match_the_live_head():
    """N_hit=0 never warm-starts, but its MISS is the k=8 teacher the library and the
    cost table describe: the schedule is mandatory and checked against the live head."""
    with pytest.raises(ConfigValidationError, match="must name denoise_schedule"):
        validate_groot_cache_config(_cp2_config(denoise_schedule=None))
    with pytest.raises(ConfigValidationError, match="runs 4 steps"):
        validate_groot_cache_config(_cp2_config(), num_inference_timesteps=4)
    with pytest.raises(ConfigValidationError, match="not a GR00T loop"):
        validate_groot_cache_config(_cp2_config(denoise_schedule="pi05_v1"))
    with pytest.raises(ConfigValidationError, match="denoise_schedule"):
        validate_groot_cache_config(_cp2_config(denoise_schedule="not_a_schedule"))


def test_cp2_n1_recipe_passes_under_the_live_k8_loop():
    config = _cp2_config()
    config.checkpoints["cp2"].judge.warm_tiers = [{"threshold": 0.8, "start_t": 0.875}]
    validate_groot_cache_config(config, num_inference_timesteps=8)
    validate_groot_cache_config(config)


def test_cp1_plus_cp2_is_refused():
    config = _cp2_config()
    config.checkpoints["cp1"] = CheckpointConfig(enabled=True, gate=GateConfig(type="always_search"),
                                                 judge=JudgeConfig(type="always_hit"))
    with pytest.raises(ConfigValidationError, match=r"exactly \{'cp1'\} or \{'cp2'\}"):
        validate_groot_cache_config(config)
    assert enabled_checkpoint_names(config) == frozenset({"cp1", "cp2"})
    assert single_enabled_checkpoint(config) is None


@pytest.mark.parametrize("builder", ["cp2_vlm_ternary", "cp1_groot_spatial_pool_16", "placeholder"])
def test_cp2_with_any_other_builder_is_refused(builder):
    config = _cp2_config(key_builder=KeyBuilderConfig(type=builder))
    with pytest.raises(ConfigValidationError, match="must use key_builder.type='cp2_groot_ternary'"):
        validate_groot_cache_config(config)


def test_cp2_warm_tier_without_a_schedule_is_refused():
    config = _cp2_config(denoise_schedule=None)
    config.checkpoints["cp2"].judge.warm_tiers = [{"threshold": 0.8, "start_t": 0.875}]
    with pytest.raises(ConfigValidationError, match="denoise_schedule"):
        validate_groot_cache_config(config)


def test_cp2_warm_tier_under_another_live_loop_is_refused():
    config = _cp2_config()
    config.checkpoints["cp2"].judge.warm_tiers = [{"threshold": 0.8, "start_t": 0.875}]
    with pytest.raises(ConfigValidationError, match="runs 4 steps"):
        validate_groot_cache_config(config, num_inference_timesteps=4)


def test_cp2_warm_tier_under_the_pi05_schedule_is_refused():
    config = _cp2_config(denoise_schedule="pi05_v1")
    config.checkpoints["cp2"].judge.warm_tiers = [{"threshold": 0.8, "start_t": 0.875}]
    with pytest.raises(ConfigValidationError, match="not a GR00T loop"):
        validate_groot_cache_config(config)


def test_hysteresis_opt_in_does_not_extend_to_cp2():
    config = _cp2_config()
    config.checkpoints["cp2"].gate = GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.9,
                                                j=3, probe_interval=3, L=6)
    with pytest.raises(ConfigValidationError, match="cp2.gate.type"):
        validate_groot_cache_config(config, allow_hysteresis_gate=True)


def test_cp2_online_write_policy_is_refused():
    config = _cp2_config(write_policy=WritePolicyConfig(type="always"))
    with pytest.raises(ConfigValidationError, match="write_policy"):
        validate_groot_cache_config(config)


def test_cp1_recipes_are_judged_exactly_as_before():
    validate_groot_cache_config(_cp1_config())
    config = _cp1_config()
    config.checkpoints["cp3"] = CheckpointConfig(enabled=True)
    with pytest.raises(ConfigValidationError, match="cp3"):
        validate_groot_cache_config(config)


# ------------------------------------------------------------------
# required_warm_timesteps: every enabled checkpoint, no fixed name
# ------------------------------------------------------------------


def _cp(judge: JudgeConfig, enabled=True) -> CheckpointConfig:
    return CheckpointConfig(enabled=enabled, gate=GateConfig(type="always_search"), judge=judge)


@pytest.mark.parametrize(
    "checkpoints, expected",
    [
        ({"cp1": _cp(JudgeConfig(type="threshold", threshold=0.8, warm_tiers=[{"threshold": 0.7, "start_t": 0.5}]))}, {0.5}),
        ({"cp1": _cp(JudgeConfig(type="threshold", threshold=0.8, warm_tiers=[{"threshold": 0.7, "start_t": 0.5}])),
          "cp3": _cp(JudgeConfig(type="always_warm_start", start_t=0.25))}, {0.5, 0.25}),
        ({"cp3": _cp(JudgeConfig(type="always_warm_start", start_t=0.25))}, {0.25}),
        ({"cp2": _cp(JudgeConfig(type="threshold", threshold=1.5, warm_tiers=[{"threshold": 0.7, "start_t": 0.875}]))}, {0.875}),
        ({"cp2": _cp(JudgeConfig(type="threshold", threshold=0.8))}, set()),
        ({"cp2": _cp(JudgeConfig(type="threshold", threshold=0.8, warm_tiers=[{"threshold": 0.7, "start_t": 0.875}]), enabled=False)}, set()),
    ],
)
def test_required_warm_timesteps_by_checkpoint_set(checkpoints, expected):
    config = CacheConfig()
    config.checkpoints = checkpoints
    assert required_warm_timesteps(config) == frozenset(expected)


# ------------------------------------------------------------------
# validate_artifact_identity: the expected checkpoint follows the recipe
# ------------------------------------------------------------------


class _Backend:
    def __init__(self, meta) -> None:
        self.artifact_meta = meta


def _wrap(meta) -> CacheStorage:
    storage = CacheStorage.__new__(CacheStorage)
    storage._backend = _Backend(meta)  # noqa: SLF001 - facade under test
    return storage


def test_identity_expects_cp2_for_a_cp2_recipe():
    good = {"key_builder_type": "cp2_groot_ternary", "checkpoint_id": "CP2", "schedule_id": K8, "denoising_num_steps": 8}
    validate_artifact_identity(_wrap(good), _cp2_config())
    with pytest.raises(ConfigValidationError, match="expected 'CP2'"):
        validate_artifact_identity(_wrap({**good, "checkpoint_id": "CP1"}), _cp2_config())
    with pytest.raises(ConfigValidationError, match="built by 'cp2_vlm_ternary'"):
        validate_artifact_identity(_wrap({**good, "key_builder_type": "cp2_vlm_ternary"}), _cp2_config())


def test_identity_binds_the_cp2_library_loop_for_a_full_hit_only_recipe():
    """A k=4 library under a k=8 n0 recipe (or the other way round) is refused even
    though no intermediates would ever be read."""
    k4 = {"key_builder_type": "cp2_groot_ternary", "checkpoint_id": "CP2", "schedule_id": "groot_n15_k4_v1",
          "denoising_num_steps": 4}
    with pytest.raises(ConfigValidationError, match="does not match the recipe's"):
        validate_artifact_identity(_wrap(k4), _cp2_config())
    unstamped = {"key_builder_type": "cp2_groot_ternary", "checkpoint_id": "CP2", "schedule_id": None}
    with pytest.raises(ConfigValidationError, match="does not match the recipe's"):
        validate_artifact_identity(_wrap(unstamped), _cp2_config())
    bad_steps = {"key_builder_type": "cp2_groot_ternary", "checkpoint_id": "CP2", "schedule_id": K8, "denoising_num_steps": 4}
    with pytest.raises(ConfigValidationError, match="denoising_num_steps=4"):
        validate_artifact_identity(_wrap(bad_steps), _cp2_config())


def test_identity_still_expects_cp1_for_a_cp1_recipe():
    with pytest.raises(ConfigValidationError, match="expected 'CP1'"):
        validate_artifact_identity(_wrap({"key_builder_type": "cp1_groot_spatial_pool_16", "checkpoint_id": "CP2"}), _cp1_config())


# ------------------------------------------------------------------
# The deployed GR00T arm yaml through load_cache_config / assert_arm_yaml
# ------------------------------------------------------------------


def _doc(tier="n0", theta=0.65, preload="/tmp/does-not-matter.pkl", projection=PROJ):
    return cp2_arm_yaml(preload_path=preload, projection=projection, tier=tier, theta_raw=theta, profile=GROOT)


def _write(tmp_path, doc, name="arm.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p


def test_groot_arm_yaml_round_trips_both_tiers(tmp_path):
    for tier in ("n0", "n1"):
        p = _write(tmp_path, _doc(tier=tier), f"{tier}.yaml")
        cfg = load_cache_config(p)
        assert sorted(cfg.checkpoints) == ["cp2"]
        assert cfg.key_builder.type == "cp2_groot_ternary"
        kb = cfg.key_builder.cp2_groot
        assert (kb.seed, kb.d, kb.p, kb.token_len, kb.feature_dim, kb.state_feat_dim) == (7, 8, 0.25, 5, 4, 2)
        assert cfg.denoise_schedule == K8
        assert cfg.keys.vlm_out.enabled and not cfg.keys.robot_state.enabled
        assert dict(cfg.backend.vector_dims) == {libs.FIELD: 8}
        validate_groot_cache_config(cfg, num_inference_timesteps=8)
        assert_arm_yaml(p, tier=tier, theta_raw=0.65, projection=PROJ, preload_path="/tmp/does-not-matter.pkl", profile=GROOT)
        assert libs.cp2_tier_of_config(cfg) == tier  # profile inferred from the builder
    n1 = load_cache_config(_write(tmp_path, _doc(tier="n1"), "n1b.yaml")).checkpoints["cp2"].judge
    assert n1.threshold == libs.N1_FULL_THRESHOLD
    assert n1.warm_tiers == [{"threshold": libs.theta_norm(0.65), "start_t": 0.875}]
    assert cp2_expected_projection_meta(load_cache_config(p)) == PROJ.expected_projection_meta(GROOT)


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda d: d["key_builder"]["cp2_groot"].update({"p": 1.5}), "cp2_groot.p"),
        (lambda d: d["key_builder"]["cp2_groot"].update({"token_len": 0}), "cp2_groot.token_len"),
        (lambda d: d["key_builder"]["cp2_groot"].update({"d": 0}), "cp2_groot.d"),
        (lambda d: d["checkpoints"].update({"cp1": copy.deepcopy(d["checkpoints"]["cp2"])}), "mutually exclusive"),
        (lambda d: d["backend"].update({"vector_dims": {"vlm_out": 9}}), "vector_dims must be exactly"),
        (lambda d: d["checkpoints"]["cp2"]["judge"].update({"type": "always_hit"}), "judge.type must be 'threshold'"),
        (lambda d: d.update({"write_policy": {"type": "always"}}), "write_policy.type='never'"),
    ],
)
def test_groot_arm_rules_reject(tmp_path, mutate, fragment):
    doc = _doc()
    mutate(doc)
    with pytest.raises(ConfigValidationError) as exc:
        load_cache_config(_write(tmp_path, doc))
    assert fragment in str(exc.value)


def test_a_pi05_shaped_cp2_yaml_loads_but_the_groot_guard_refuses_it(tmp_path):
    """The config layer accepts any well-formed CP2 arm; the island's serving guard
    is what refuses the Pi0.5 builder on GR00T."""
    doc = _doc()
    doc["key_builder"] = {"type": "cp2_vlm_ternary", "cp2_vlm": {"seed": 7, "d": 8, "p": 0.25, "input_dim": 22}}
    doc.pop("denoise_schedule")
    cfg = load_cache_config(_write(tmp_path, doc))
    with pytest.raises(ConfigValidationError, match="must use key_builder.type='cp2_groot_ternary'"):
        validate_groot_cache_config(cfg)


def test_groot_contract_requires_the_schedule():
    config = _cp2_config(denoise_schedule=None)
    problems = libs.cp2_contract_problems(config)
    assert any("denoise_schedule" in p for p in problems)


# ------------------------------------------------------------------
# Artifact binding at the storage choke point (both assembly entries)
# ------------------------------------------------------------------


def _groot_builder(projection=PROJ):
    lay = projection.layout
    return GrootCP2TernaryKeyBuilder(seed=projection.seed, d=projection.d, p=projection.p, token_len=lay["token_len"],
                                     feature_dim=lay["feature_dim"], state_feat_dim=lay["state_feat_dim"])


def _write_groot_artifact(path, *, projection=None, id_policy=libs.ID_POLICY, builder_type="cp2_groot_ternary",
                          schedule_id=K8, num_steps=8, d=8, warm_t=0.875):
    entries = []
    for i in range(3):
        entries.append(CacheEntry(
            id=f"e{i}", checkpoint_id=CheckpointID.CP2, query_keys={libs.FIELD: torch.randn(d)},
            payload=CachePayload(action_chunk=torch.zeros(16, 32), intermediates={warm_t: torch.zeros(16, 32)},
                                 denoising_num_steps=num_steps, task_key="t", schedule_id=schedule_id),
            step_idx=i, trajectory_id="traj", prev_ids=[], next_ids=[],
        ))
    art = {"key_builder_type": builder_type, "checkpoint_id": "CP2", "vector_dims": {libs.FIELD: d},
           "entries": entries, "projection": _groot_builder().projection_meta() if projection is None else projection,
           "id_policy": id_policy, "stage1_path": "groot_reconstructed_template", "schedule_id": schedule_id,
           "teacher": GROOT.name}
    with open(path, "wb") as f:
        pickle.dump(art, f)
    return path


def test_matching_groot_artifact_loads_through_both_assembly_entries(tmp_path):
    ok = _write_groot_artifact(tmp_path / "ok.pkl")
    for tier in ("n0", "n1"):
        cfg = load_cache_config(_write(tmp_path, _doc(tier=tier, preload=str(ok)), f"{tier}.yaml"))
        storage = build_shared_storage(cfg)
        assert storage.artifact_meta["projection"]["layout"]["kind"] == "groot_encoded_v1"
        assert storage.artifact_meta["schedule_id"] == K8
        validate_artifact_identity(storage, cfg)
        components = build_cache_components(cfg)  # the single-connection entry delegates to the same choke point
        assert components["storage"].artifact_meta["projection"] == storage.artifact_meta["projection"]


@pytest.mark.parametrize(
    "artifact_kwargs, yaml_mutate, fragment",
    [
        ({"projection": _groot_builder(libs.ProjectionArgs(seed=8, d=8, p=0.25, input_dim=22, layout=PROJ.layout)).projection_meta()},
         None, "projection.seed"),
        ({"projection": _groot_builder(libs.ProjectionArgs(seed=7, d=8, p=0.25, input_dim=26,
                                                           layout={**PROJ.layout, "token_len": 6})).projection_meta()},
         None, "projection.D=26"),
        # Same D, different split of the token axis: only the layout block tells them apart.
        ({"projection": _groot_builder(libs.ProjectionArgs(seed=7, d=8, p=0.25, input_dim=22,
                                                           layout={**PROJ.layout, "token_len": 4, "feature_dim": 5})).projection_meta()},
         None, "projection.layout"),
        ({"projection": get_projection_spec(7, 8, 0.25, 22).meta()}, None, "metadata keys"),  # Pi0.5-shaped meta: no layout
        ({"projection": {"seed": 7}}, None, "metadata keys"),
        ({"id_policy": "rehashed"}, None, "id_policy"),
        ({"builder_type": "cp2_vlm_ternary"}, None, "does not match configured"),
        ({}, lambda d: d["key_builder"]["cp2_groot"].update({"seed": 8}), "projection.seed"),
        ({}, lambda d: d["key_builder"]["cp2_groot"].update({"state_feat_dim": 3}), "projection.D=22"),
    ],
)
def test_groot_binding_negatives(tmp_path, artifact_kwargs, yaml_mutate, fragment):
    pkl = _write_groot_artifact(tmp_path / "art.pkl", **artifact_kwargs)
    doc = _doc(preload=str(pkl))
    if yaml_mutate is not None:
        yaml_mutate(doc)
    cfg = load_cache_config(_write(tmp_path, doc))
    with pytest.raises(ConfigValidationError, match=fragment):
        build_shared_storage(cfg)


def test_groot_n1_refuses_a_library_under_another_schedule_or_missing_the_tier(tmp_path):
    k4 = _write_groot_artifact(tmp_path / "k4.pkl", schedule_id="groot_n15_k4_v1", num_steps=4, warm_t=0.75)
    cfg = load_cache_config(_write(tmp_path, _doc(tier="n1", preload=str(k4)), "k4.yaml"))
    with pytest.raises(ConfigValidationError, match="does not match the config's"):
        build_shared_storage(cfg)
    no_tier = _write_groot_artifact(tmp_path / "t5.pkl", warm_t=0.5)
    cfg = load_cache_config(_write(tmp_path, _doc(tier="n1", preload=str(no_tier)), "t5.yaml"))
    with pytest.raises(ConfigValidationError, match="0.8750"):
        build_shared_storage(cfg)
    # n0 never touches intermediates: the same library loads.
    build_shared_storage(load_cache_config(_write(tmp_path, _doc(tier="n0", preload=str(no_tier)), "t5n0.yaml")))


def test_pi05_cp2_artifacts_still_bind_under_the_generalised_check(tmp_path):
    spec = get_projection_spec(PI05_PROJ.seed, PI05_PROJ.d, PI05_PROJ.p, PI05_PROJ.input_dim)
    entries = [CacheEntry(id=f"e{i}", checkpoint_id=CheckpointID.CP2, query_keys={libs.FIELD: torch.randn(8)},
                          payload=CachePayload(action_chunk=torch.zeros(10, 32), intermediates={0.1: torch.zeros(10, 32)},
                                               denoising_num_steps=10, task_key="t"),
                          step_idx=i, trajectory_id="traj", prev_ids=[], next_ids=[]) for i in range(2)]
    pkl = tmp_path / "pi05.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"key_builder_type": "cp2_vlm_ternary", "checkpoint_id": "CP2", "vector_dims": {libs.FIELD: 8},
                     "entries": entries, "projection": spec.meta(), "id_policy": libs.ID_POLICY, "stage1_path": "online"}, f)
    doc = cp2_arm_yaml(preload_path=str(pkl), projection=PI05_PROJ, tier="n1", theta_raw=0.85, profile=libs.PI05)
    cfg = load_cache_config(_write(tmp_path, doc, "pi05.yaml"))
    assert cfg.key_builder.type == "cp2_vlm_ternary" and cfg.denoise_schedule is None
    build_shared_storage(cfg)
    # ...and a GR00T library under the Pi0.5 config is refused by builder type.
    groot = _write_groot_artifact(tmp_path / "g.pkl")
    doc = cp2_arm_yaml(preload_path=str(groot), projection=PI05_PROJ, tier="n0", theta_raw=0.85, profile=libs.PI05)
    with pytest.raises(ConfigValidationError, match="does not match configured"):
        build_shared_storage(load_cache_config(_write(tmp_path, doc, "cross.yaml")))
