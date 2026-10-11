"""Regression coverage for Review B B5/B7/B8/B9/B11."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

from exp.r17_integration.common.contracts import CaptureSpec
from exp.r17_integration.models.dp.fixtures import LIBERO_DATA, libero_observation
from exp.r17_integration.models.dp.request_noise import RequestNoise
from exp.r17_integration.models.dp.sampler import SamplerProfile
from exp.r17_integration.tests.models.dp.conftest import observation, episode_metadata, seal_test_library, canonical_recipe_options


def test_snapshot_condition_digest_once_per_request_not_once_per_step(make_adapter, req_context, monkeypatch):
    from exp.r17_integration.models.dp import staged
    adapter = make_adapter()
    s2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(observation(), req_context)))
    condition = s2.data["conditioning"]
    calls = []
    real = staged.tensor_digest
    def counted(tensor):
        if tensor is condition:
            calls.append(1)
        return real(tensor)
    monkeypatch.setattr(staged, "tensor_digest", counted)
    plain = adapter.sample(s2, SamplerProfile(), RequestNoise(19), CaptureSpec())
    assert not calls
    captured = adapter.sample(s2, SamplerProfile(), RequestNoise(19), CaptureSpec(True, True))
    assert len(captured.snapshots) == 99 and len(calls) == 1
    assert len({s.conditioning_scope for s in captured.snapshots}) == 1
    assert torch.equal(plain.normalized_chunk, captured.normalized_chunk)
    assert plain.rng_receipt == captured.rng_receipt
    result = adapter.resume(s2, captured.snapshots[-1], captured.sampler, RequestNoise(19), CaptureSpec())
    assert len(calls) == 2 and torch.equal(result.normalized_chunk, plain.normalized_chunk)


@pytest.mark.parametrize("domain", ["libero", "metaworld"])
def test_capabilities_do_not_claim_generic_warm_library_recipe_qualification(make_adapter, domain):
    caps = make_adapter(domain).metadata().capabilities
    for name in ("native_warm_resume", "native_warm_reset", "library", "recipe"):
        assert caps.features[name].status == "implemented_unverified"
        with pytest.raises(ValueError, match="unavailable"):
            caps.require(name)
    assert "K1 lossy" in caps.features["k_step"].reason
    assert caps.features["cp2"].status == "unsupported"
    if domain == "libero":
        # These need P2's global task identity and acknowledgment-capable runner.
        for name in ("conductor", "worker"):
            assert caps.features[name].status == "implemented_unverified"


@pytest.mark.parametrize("side", [224, 99])
def test_libero_rejects_resampled_wire_and_binds_expected_size(make_adapter, req_context, side):
    from exp.r17_integration.models.dp.manifest import feature_spec
    adapter = make_adapter("libero")
    obs = observation("libero")
    obs["observation/image"] = np.zeros((side,side,3), np.uint8)
    with pytest.raises(ValueError, match="96.*256"):
        adapter.prepare(obs, req_context)
    assert feature_spec("libero", "wire")["wire_image_sizes"] == [96,256]


def test_libero_true_paired_recorded_views_use_both_adapter_branches(make_adapter, req_context):
    if not Path(LIBERO_DATA).is_dir():
        pytest.skip("read-only real paired libero_96 sample unavailable")
    obs, provenance = libero_observation(frame=0)
    assert obs["observation/image"].shape == obs["observation/wrist_image"].shape == (96,96,3)
    assert not np.array_equal(obs["observation/wrist_image"], obs["observation/image"][::-1])
    assert provenance["camera_png_sha256"][0] != provenance["camera_png_sha256"][1]
    adapter = make_adapter("libero")
    prepared = adapter.prepare(obs, req_context)
    stages = adapter.run_stage1(prepared)
    with torch.inference_mode():
        assert torch.equal(adapter.run_stage2(stages).data["conditioning"], adapter.model.condition(prepared.data["inputs"]))
    assert np.array_equal(stages.fields["vision_1"], stages.data["features"]["left_wrist_0_rgb"][0].numpy())
    changed = dict(obs, **{"observation/wrist_image": np.zeros((96,96,3), np.uint8)})
    other = adapter.run_stage1(adapter.prepare(changed, req_context))
    assert np.array_equal(stages.fields["vision_0"], other.fields["vision_0"])
    assert not np.array_equal(stages.fields["vision_1"], other.fields["vision_1"])


def test_LIBERO_wire_size_is_bound_to_P2_benchmark_identity(make_adapter, req_context):
    from dataclasses import replace
    from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec, SUITES
    from exp.r17_integration.models.dp.staged import DPAdapter
    # Synthetic task labels exercise the real P2 contract, not official SR.
    tasks = tuple((f"{suite}/task{i}", f"pick object{i}", suite) for suite in SUITES for i in range(10))
    a = LiberoBenchmarkSpec(tasks, wire_size=96)
    b = LiberoBenchmarkSpec(tasks, wire_size=256)
    base = make_adapter("libero")
    adapter = DPAdapter(base.model, base.normalization.stats, base.language,
        domain="libero", weight_identity=base.weight_identity, libero_benchmark_specs=[asdict(a),asdict(b)])
    assert adapter.metadata().allowed_benchmark_protocols == (a.identity,b.identity)
    assert a.identity != b.identity
    adapter.prepare(observation("libero"), replace(req_context, benchmark_protocol=a.identity))
    with pytest.raises(ValueError, match="bound benchmark protocol"):
        adapter.prepare(observation("libero"), replace(req_context, benchmark_protocol=b.identity))
    image = np.zeros((256,256,3), np.uint8)
    adapter.prepare(dict(observation("libero"), **{"observation/image": image, "observation/wrist_image": image}),
                    replace(req_context, benchmark_protocol=b.identity))


@pytest.mark.parametrize("k", [100, 2])
@pytest.mark.parametrize("capture_mode", ["sink", "directory"])
def test_serve_CLI_deploys_shared_five_layers_and_capture_even_at_full_K(make_adapter, monkeypatch, tmp_path, k, capture_mode):
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.common.layers import LayoutRecipe, LibraryData
    from exp.r17_integration.models.dp import loader, serve
    from exp.r17_integration.models.dp.provider import register_provider
    from exp.r17_integration.common import websocket_policy_server as ws
    monkeypatch.setattr(registry, "_providers", {})
    adapter = make_adapter()
    provider = register_provider(adapter)
    config = serve.plain_config(provider, num_steps=k)
    from exp.r17_integration.common.config import RecipeConfig
    config_path = tmp_path / "config.yaml"
    # Only the entrypoint registers the resident provider in the tested process.
    registry._providers.clear()
    load_path = tmp_path / "load.json"
    load_path.write_text(json.dumps({"diagnostic": True}))
    rng = np.random.default_rng(21)
    library = LibraryData(features={"vision_0": rng.normal(size=(24,64)).astype(np.float32)},
                          state=rng.normal(size=(24,4)).astype(np.float32),
                          action=rng.normal(size=(24,16,4)).astype(np.float32),
                          task_id=np.full(24, episode_metadata("reach-v3")["task_id"], np.int64), episode=np.repeat(np.arange(6),4),
                          step=np.tile(np.arange(4),6), source_digest="synthetic_library_only")
    recipe = LayoutRecipe(adapter.metadata().layout, 50, periodic_gap=0).fit(library)
    store = seal_test_library(tmp_path / "cli_teacher", adapter, library, task_name="reach-v3")
    config.recipe = RecipeConfig(enabled=True, store_path=str(store.directory), store_digest=store.manifest["content_digest"],
          teacher_sampler_identity=adapter.metadata().default_sampler.identity, periodic_gap=0, **canonical_recipe_options(adapter))
    config_path.write_text(yaml.safe_dump(asdict(config)))
    configurations, captures = [], []
    def recipe_hook(cfg, storage):
        configurations.append(cfg.adapter_binding["num_steps"])
        return recipe.fork()
    monkeypatch.setattr(sys.modules[__name__], "recipe_hook", recipe_hook, raising=False)
    monkeypatch.setattr(sys.modules[__name__], "capture_hook", captures.append, raising=False)
    monkeypatch.setattr(loader, "load_adapter", lambda spec: adapter)
    class Server:
        def __init__(self, policy, **kwargs):
            self.factory = kwargs["connection_policy_factory"]
            assert "r17_model_spec" in kwargs["metadata"]
        def serve_forever(self):
            a, b = self.factory(), self.factory()
            assert a.recipe is not b.recipe and a.recipe.guard is not b.recipe.guard
            assert a.recipe.use_corrector and a.recipe.use_guard and a.recipe.periodic_gap == 0
            a.on_episode_start(task_id=0, init_state_id=0, episode_id="cli",
                extra_metadata={**episode_metadata("reach-v3"), "attempt": 1, "task_uid": "p4_cli",
                                "task_data_hash": "fixture", "split": "library"})
            result = a.infer(dict(observation(), __request__={"task_name": "reach-v3", "original_init_idx": 0,
                                  "explicit_seed": 18, "decision_seq": 0}))
            assert result["__hit_meta__"]["stage_calls"] == (1,1,k)
            assert result["__hit_meta__"]["capture_enabled"] and not result["__hit_meta__"]["pricing_eligible"]
            if capture_mode == "sink":
                assert len(captures) == 1 and captures[0].actual_nfe == k
                assert np.asarray(captures[0].model_capture["native_grid"]).dtype == np.int64
            a.on_episode_end(execution={"decision_seq": 0, "executed_count": 3})
    monkeypatch.setattr(ws, "WebsocketPolicyServer", Server)
    capture_args = ["--capture-sink", __name__+":capture_hook"] if capture_mode == "sink" else ["--debug-directory", str(tmp_path / "debug")]
    monkeypatch.setattr(sys, "argv", ["dp.serve", "--load-spec", str(load_path), "--cache-config", str(config_path),
         "--port", "8765", "--recipe-factory", __name__+":recipe_hook", *capture_args])
    serve.main()
    assert configurations == [k,k]
    if capture_mode == "directory":
        from exp.r17_integration.debug.reader_profiles import CaptureReader
        paths = list((tmp_path / "debug").glob("binding_*"))
        assert len(paths) == 1
        rows = list(CaptureReader(paths[0]).decisions())
        assert len(rows) == 1 and rows[0][0]["actual_nfe"] == k
        assert rows[0][1]["native_grid"].tolist() == list(SamplerProfile(num_steps=k).to_spec(adapter.model.cfg).native_timesteps)
        acknowledged = [json.loads(line) for line in (paths[0]/"execution.jsonl").read_text().splitlines()]
        assert acknowledged[0]["executed_count"] == 3


def test_default_factory_consumes_bundle_recipe_and_K100_teacher_for_K2_MISS(make_adapter, monkeypatch, tmp_path):
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.common.config import RecipeConfig, build_shared_storage
    from exp.r17_integration.common.recipe import recipe_identity
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.models.dp.provider import register_provider, make_connection_factory
    from exp.r17_integration.models.dp.serve import plain_config
    from exp.r17_integration.search.store import write_store
    from exp.r17_integration.tests.debug.fixtures import record
    monkeypatch.setattr(registry, "_providers", {})
    adapter = make_adapter()
    provider = register_provider(adapter)
    binding = model_binding(adapter.metadata(), adapter.metadata().allowed_benchmark_protocols[0])
    rows = [record(init=10+episode, seq=step, n_decisions=4) for episode in range(6) for step in range(4)]
    rng = np.random.default_rng(21)
    store = write_store(tmp_path / "teacher100", binding, rows,
        {"key_vision_0": list(rng.normal(size=(24,64)).astype(np.float32)),
         "state": list(rng.normal(size=(24,4)).astype(np.float32)),
         "normalized_actions": list(rng.normal(size=(24,16,4)).astype(np.float32))})
    plain = plain_config(provider, num_steps=2)
    plain_policy = make_connection_factory(adapter, provider=provider, default_config=plain,
                                          default_storage=build_shared_storage(plain))()
    assert plain_policy.recipe is None
    config = plain_config(provider, num_steps=2)
    config.recipe = RecipeConfig(enabled=True, store_path=str(store.directory),
        store_digest=store.manifest["content_digest"], teacher_sampler_identity=adapter.metadata().default_sampler.identity,
        corrector=True, guard=True, periodic_gap=3, k=4, kref=3, **canonical_recipe_options(adapter))
    factory = make_connection_factory(adapter, provider=provider, default_config=config,
                                      default_storage=build_shared_storage(config))
    a, b = factory(), factory()
    assert a.sampler.family == "ddim" and a.sampler.num_steps == 2
    assert a.recipe is not b.recipe and a.recipe.action is b.recipe.action
    assert a.recipe.periodic_gap == 3 and a.recipe.use_guard and a.recipe.use_corrector
    assert a.recipe.bundle_recipe_identity == recipe_identity(config)
    a.on_episode_start(extra_metadata=episode_metadata("reach-v3"))
    result = a.infer(observation())
    assert result["actions"].shape == (5,4)


def test_formal_LIBERO_extended_commit_rejects_old_client_identity_and_ack(make_adapter, monkeypatch):
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.common.config import build_shared_storage
    from exp.r17_integration.models.dp.provider import register_provider, make_connection_factory
    from exp.r17_integration.models.dp.serve import plain_config
    monkeypatch.setattr(registry, "_providers", {})
    adapter = make_adapter("libero")
    provider = register_provider(adapter)
    config = plain_config(provider, num_steps=2, commit_horizon=10)
    factory = make_connection_factory(adapter, provider=provider, default_config=config,
                                      default_storage=build_shared_storage(config))
    old = factory()
    with pytest.raises(ValueError, match="ack_protocol"):
        old.on_episode_start(task_id=3, extra_metadata={"orig_init_state_idx": 7})
    incomplete = factory()
    incomplete.on_episode_start(task_id=3, extra_metadata={"ack_protocol": "r17.execution.v1"})
    with pytest.raises(ValueError, match="context"):
        incomplete.infer(observation("libero"))
    new = factory()
    new.on_episode_start(task_id=3, extra_metadata={**episode_metadata("libero_spatial/task3"),
                                                   "task_id": 3, "original_init_idx": 7, "orig_init_state_idx": 7})
    result = new.infer(observation("libero"))
    assert new._episode["original_init_idx"] == 7
    with pytest.raises(ValueError, match="acknowledgment"):
        new.infer(observation("libero"))
    tail = new.infer(dict(observation("libero"), __execution__={"decision_seq": 0, "executed_count": 5}))
    assert tail["__hit_meta__"]["stage_calls"] == (0,0,0)
    assert result["actions"][5:].tobytes() == tail["actions"].tobytes()
