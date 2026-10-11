"""REVIEW_B2 N6 and B11: lossy role gates and real paired qualification rows."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from exp.r17_integration.common.contracts import CaptureSpec
from exp.r17_integration.common.config import RecipeConfig, build_shared_storage, config_from_dict, validate_cache_config
from exp.r17_integration.common.provider import binding_for, validate_binding, validate_config
from exp.r17_integration.common.serving import make_bundle_validator, make_connection_factory as shared_factory
from exp.r17_integration.models.dp.fixtures import (LIBERO_DATA, libero_observation,
    read_libero_qualification_fixture, write_libero_qualification_fixture)
from exp.r17_integration.models.dp.manifest import file_digest
from exp.r17_integration.models.dp.provider import register_provider
from exp.r17_integration.models.dp.request_noise import RequestNoise
from exp.r17_integration.models.dp.sampler import SamplerProfile
from exp.r17_integration.models.dp.serve import plain_config
from exp.r17_integration.models.dp.wire_policy import WirePolicy
from exp.r17_integration.tests.models.dp.conftest import observation, episode_metadata


@pytest.fixture(params=["libero", "metaworld"])
def dp_binding(request, make_adapter, monkeypatch):
    from exp.r17_integration.common import provider as registry
    monkeypatch.setattr(registry, "_providers", {})
    monkeypatch.setattr(registry, "_traces", {})
    adapter = make_adapter(request.param)
    return adapter, register_provider(adapter)


def study_config(provider):
    return plain_config(provider, num_steps=1, allow_lossy_sampler=True,
                        sampler_study_label="review_B2_K1_plain_step_reduction")


def test_N6_shared_binding_rejects_default_K1_and_accepts_K2_4_10_100(dp_binding):
    adapter, provider = dp_binding
    protocol = adapter.metadata().allowed_benchmark_protocols[0]
    with pytest.raises(ValueError, match="K1.*lossy"):
        binding_for(provider, protocol, num_steps=1)
    # Reviewer repro: constructing a K1 YAML used to succeed without opt-in.
    old = asdict(plain_config(provider, num_steps=2))
    old["adapter_binding"].update(num_steps=1, sampler_identity=provider.sampler_for_steps(1).identity)
    with pytest.raises(ValueError, match="K1.*lossy"):
        config_from_dict(old)
    for k in (2, 4, 10, 100):
        config = plain_config(provider, num_steps=k)
        validate_config(config)
        p, sampler = validate_binding(config.adapter_binding)
        assert p is provider and sampler.num_steps == k
        assert "allow_lossy_sampler" not in config.adapter_binding
        assert "sampler_study_label" not in config.adapter_binding


@pytest.mark.parametrize("allow,label", [(True, None), (True, ""), (True, "   "),
                                         (False, "study"), ("true", "study"), (True, 1)])
def test_N6_K1_requires_strict_flag_and_label(dp_binding, allow, label):
    _, provider = dp_binding
    with pytest.raises(ValueError, match="lossy|study_label"):
        plain_config(provider, num_steps=1, allow_lossy_sampler=allow, sampler_study_label=label)


@pytest.mark.parametrize("role", ["cache_miss", "recipe", "teacher", "library", "extended_commit"])
def test_N6_study_flag_cannot_enable_cache_MISS_teacher_or_recipe(dp_binding, role):
    _, provider = dp_binding
    cfg = study_config(provider)
    if role == "cache_miss":
        cfg.checkpoints = plain_config(provider, num_steps=2, retrieval=True).checkpoints
        cfg.checkpoints["cp1"].judge.type = "always_miss"
    elif role == "recipe":
        cfg.recipe = RecipeConfig(enabled=True, store_path="unused", store_digest="unused",
            teacher_sampler_identity=provider.sampler_for_steps(100).identity, ordered_tasks=["reach"])
    elif role == "teacher":
        cfg.recipe.teacher_sampler_identity = provider.sampler_for_steps(1).identity
    elif role == "library":
        cfg.adapter_binding["library_sampler_identity"] = provider.sampler_for_steps(100).identity
    else:
        cfg.adapter_binding["commit_horizon"] = 10
    with pytest.raises(ValueError, match="plain|teacher|replan"):
        # Must fail before publication, even through P1 rather than P4 factory.
        make_bundle_validator(provider)(cfg)


def test_N6_K1_teacher_rejected_with_K2_MISS_and_no_study_flag(dp_binding):
    _, provider = dp_binding
    cfg = plain_config(provider, num_steps=2)
    cfg.recipe.teacher_sampler_identity = provider.sampler_for_steps(1).identity
    with pytest.raises(ValueError, match="K1.*teacher"):
        validate_config(cfg)
    cfg.recipe.teacher_sampler_identity = None
    cfg.adapter_binding["library_sampler_identity"] = provider.sampler_for_steps(1).identity
    with pytest.raises(ValueError, match="K1.*teacher"):
        validate_binding(cfg.adapter_binding)


def test_N6_sealed_other_eta_K1_teacher_cannot_bypass_hash_gate(dp_binding, tmp_path):
    adapter, provider = dp_binding
    cfg = plain_config(provider, num_steps=2)
    teacher = SamplerProfile("ddim", 1, eta=.7).to_spec(adapter.model.cfg)
    path = tmp_path / "teacher1"
    path.mkdir()
    from exp.r17_integration.common.contracts import digest
    manifest = {"schema": "r17.store.v1", "status": "READY", "rows": [{"split":"library", "init_pool":"B"}],
        "binding": {"sampler": asdict(teacher), "model": asdict(adapter.metadata())}}
    manifest['content_digest'] = digest(manifest)
    (path / "READY.json").write_text(json.dumps(manifest))
    cfg.recipe = RecipeConfig(enabled=True, store_path=str(path), store_digest=manifest['content_digest'],
        teacher_sampler_identity=teacher.identity, ordered_tasks=["reach"])
    with pytest.raises(ValueError, match="K1.*teacher"):
        validate_config(cfg)


def test_N6_labelled_plain_K1_study_still_runs_one_NFE_with_capture_parity(dp_binding):
    adapter, provider = dp_binding
    cfg = config_from_dict(yaml.safe_load(yaml.safe_dump(asdict(study_config(provider)))))
    assert cfg.adapter_binding["sampler_study_label"] == "review_B2_K1_plain_step_reduction"
    storage = build_shared_storage(cfg)
    records = []
    plain = shared_factory(provider, cfg, storage)()
    debug = shared_factory(provider, cfg, storage, capture_sink=records.append)()
    for p in (plain, debug):
        p.on_episode_start(extra_metadata=episode_metadata())
    obs = observation(adapter.domain)
    a, b = plain.infer(obs), debug.infer(obs)
    assert a["actions"].tobytes() == b["actions"].tobytes()
    assert a["actions"].shape == (5, adapter.metadata().layout.valid_action_dim)
    assert a["__hit_meta__"]["stage_calls"] == (1, 1, 1)
    assert a["__hit_meta__"]["sampler_identity"] == provider.sampler_for_steps(1).identity
    assert a["__hit_meta__"]["sampler_study_label"] == cfg.adapter_binding["sampler_study_label"]
    assert a["__hit_meta__"]["sampler_quality_status"] == "lossy_study"
    assert len(records) == 1 and records[0].actual_nfe == 1
    assert not records[0].native_snapshots
    assert records[0].diagnostics["sampler_study_label"] == cfg.adapter_binding["sampler_study_label"]
    for p in (plain, debug):
        p.on_episode_end(execution={"decision_seq": 0, "executed_count": 3})
    metadata = WirePolicy(adapter).get_metadata()["r17_sampler_admission"]
    assert metadata["descent_steps"] == [2, 4, 10, 100]
    assert metadata["K1"]["allowed_role"] == "plain_step_reduction_study"


def test_N6_default_DDPM100_and_native_K1_math_unchanged(dp_binding, req_context):
    adapter, provider = dp_binding
    cfg = plain_config(provider)
    assert cfg.adapter_binding["num_steps"] == 100
    assert cfg.adapter_binding["commit_horizon"] == 5 and not cfg.checkpoints
    assert provider.sampler_for_steps(100).family == "ddpm"
    s2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(observation(adapter.domain), req_context)))
    before = torch.random.get_rng_state().clone()
    # Binding permission never removes the offline native arithmetic API.
    low = adapter.sample(s2, SamplerProfile(num_steps=1), RequestNoise(18), CaptureSpec())
    assert low.actual_nfe == 1 and torch.equal(before, torch.random.get_rng_state())


def test_N6_bound_K1_study_markers_rejected_by_P2_dev_descent(make_adapter, monkeypatch):
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    from exp.r17_integration.conductor.manifest import EndpointSpec, episode_tasks
    from exp.r17_integration.conductor.strategy import MetaWorldDescentStrategy, PerTaskDescent
    from openpi.conductor.task import ServerEndpoint
    monkeypatch.setattr(registry, "_providers", {})
    adapter = make_adapter()
    provider = register_provider(adapter)
    cfg = study_config(provider)
    spec = adapter.metadata()
    endpoint = EndpointSpec(ServerEndpoint("local", 1), spec.checkpoint_identity,
        spec.norm_identity, spec.layout.identity, FORMAL_SPEC.identity, allowed_k=(1, 2))
    episodes = episode_tasks(campaign="P4_K1_study", arm="plain_step_reduction", bundle_id="K1_study",
        endpoint=endpoint.endpoint, model_identity=spec.checkpoint_identity, norm_identity=spec.norm_identity,
        layout_identity=spec.layout.identity, num_steps=1, partition="dev", names=["reach-v3"], indices=[30])
    controller = PerTaskDescent(["reach-v3"], ["k1"], {"reach-v3": {30: True}}, n_dev=1)
    candidate = replace(episodes[0], extra={**episodes[0].extra, "descent_level": "k1",
        "allow_lossy_sampler": cfg.adapter_binding["allow_lossy_sampler"],
        "sampler_study_label": cfg.adapter_binding["sampler_study_label"],
        "sampler_quality_status": "lossy_study"})
    with pytest.raises(ValueError, match="lossy sampler study"):
        MetaWorldDescentStrategy([candidate], endpoints=[endpoint], controller=controller)
    k2 = replace(episodes[0], extra={**episodes[0].extra, "num_steps": 2, "descent_level": "k2"})
    healthy = PerTaskDescent(["reach-v3"], ["k2"], {"reach-v3": {30: True}}, n_dev=1)
    MetaWorldDescentStrategy([k2], endpoints=[endpoint], controller=healthy)


def test_B11_qualification_fixture_uses_recorded_pair_verbatim(make_adapter, req_context, tmp_path):
    if not Path(LIBERO_DATA).is_dir():
        pytest.skip("read-only real paired libero_96 sample unavailable")
    path = tmp_path / "wire_observations.npz"
    receipt = write_libero_qualification_fixture(path, frames=(0, 1, 2))
    rows, sealed = read_libero_qualification_fixture(path)
    from exp.r17_integration.qualification.model_runtime import wire_observations
    qualified = wire_observations({"model": "dp", "domain": "libero", "assets": str(tmp_path)})
    assert receipt == sealed and not sealed["synthetic"] and sealed["wire_size"] == 96
    assert len(rows) == 3 and not sealed["quality_sr_claim"]
    adapter = make_adapter("libero")
    for frame, row in enumerate(rows):
        real, provenance = libero_observation(frame=frame)
        assert sealed["frames"][frame] == provenance
        for key in ("observation/image", "observation/wrist_image", "observation/state"):
            assert row[key].tobytes() == real[key].tobytes()
            assert qualified[frame][key].tobytes() == real[key].tobytes()
        assert not np.array_equal(row["observation/wrist_image"], row["observation/image"][::-1])
        assert provenance["camera_png_sha256"][0] != provenance["camera_png_sha256"][1]
        native = adapter.prepare(row, req_context)
        stage1 = adapter.run_stage1(native)
        assert torch.equal(adapter.run_stage2(stage1).data["conditioning"], adapter.model.condition(native.data["inputs"]))
    with np.load(path, allow_pickle=False) as data:
        assert data["prompt"].dtype.kind == "U"
    with pytest.raises(FileExistsError):
        write_libero_qualification_fixture(path)


def test_B11_missing_recorded_wrist_fails_instead_of_synthesizing(tmp_path):
    path = tmp_path / "wire_observations.npz"
    np.savez_compressed(path, **{"observation/image": np.zeros((3,96,96,3), np.uint8),
        "observation/state": np.zeros((3,8),np.float32), "prompt": np.asarray(["reach"]*3)})
    path.with_suffix(".json").write_text(json.dumps({"schema": "r17.dp.libero_qualification_fixture.v1",
        "synthetic": False, "real_paired_views": True, "npz_sha256": file_digest(path)}))
    with pytest.raises(ValueError, match="both recorded camera keys"):
        read_libero_qualification_fixture(path)
    from exp.r17_integration.qualification.model_runtime import wire_observations
    with pytest.raises(ValueError, match="recorded paired.*wrist"):
        wire_observations({"model": "dp", "domain": "libero", "assets": str(tmp_path)})
