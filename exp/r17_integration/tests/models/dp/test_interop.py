from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import CaptureSpec, DecisionCapture
from exp.r17_integration.models.dp.config import LoadSpec
from exp.r17_integration.models.dp.loader import load_adapter
from exp.r17_integration.models.dp.request_noise import RequestNoise
from exp.r17_integration.models.dp.sampler import SamplerProfile
from exp.r17_integration.tests.models.dp.conftest import observation, episode_metadata, seal_test_library, canonical_recipe_options


@pytest.mark.parametrize("domain", ["libero", "metaworld"])
def test_DP_geometry_shared_five_layers_connection_recipe(make_adapter, monkeypatch, tmp_path, domain):
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.common.layers import LayoutRecipe, LibraryData
    from exp.r17_integration.models.dp.provider import make_connection_factory, register_provider
    from exp.r17_integration.models.dp.serve import plain_config
    from exp.r17_integration.common.config import build_shared_storage, RecipeConfig
    monkeypatch.setattr(registry, "_providers", {})
    monkeypatch.setattr(registry, "_traces", {})
    docs = None
    if domain == "libero":
        from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec, SUITES
        benchmark = LiberoBenchmarkSpec(tuple((f"{suite}/task{i}", f"pick object{i}", suite)
            for suite in SUITES for i in range(10)), wire_size=96)
        docs = [asdict(benchmark)]
    adapter = make_adapter(domain, libero_benchmark_specs=docs)
    spec = adapter.metadata()
    options = canonical_recipe_options(adapter)
    task_name = options["ordered_tasks"][0]
    rng = np.random.default_rng(21)
    n = 24
    library = LibraryData(features={k: rng.normal(size=(n, dim)).astype(np.float32)
                                   for k, dim in spec.feature_spec.fields.items() if k != "robot_state"},
                          state=rng.normal(size=(n, len(spec.layout.valid_state_indices))).astype(np.float32),
                          action=rng.normal(size=(n, 16, spec.layout.action_width)).astype(np.float32),
                          task_id=np.zeros(n, np.int64), episode=np.repeat(np.arange(6), 4),
                          step=np.tile(np.arange(4), 6), source_digest="synthetic_library_only")
    recipe = LayoutRecipe(spec.layout, len(options["ordered_tasks"]),
                           corrector=True, periodic_gap=3, early=True).fit(library)
    for fold in recipe.fit_evidence["loeo"]:
        assert fold["query_episode"] not in fold["donor_episodes"]
    provider = register_provider(adapter)
    cfg = plain_config(provider, num_steps=2, commit_horizon=10)
    store = seal_test_library(tmp_path / "teacher", adapter, library, task_name=task_name)
    cfg.recipe = RecipeConfig(enabled=True, store_path=str(store.directory), store_digest=store.manifest["content_digest"],
          teacher_sampler_identity=spec.default_sampler.identity, periodic_gap=3, **options)
    storage = build_shared_storage(cfg)
    factory = make_connection_factory(adapter, provider=provider, default_config=cfg, default_storage=storage,
                                      recipe_factory=lambda cfg, storage: recipe.fork())
    a, b = factory(), factory()
    a.on_episode_start(task_id=0, extra_metadata=episode_metadata(task_name))
    b.on_episode_start(task_id=0, extra_metadata=episode_metadata(task_name))
    obs = observation(domain)
    first = a.infer(obs)
    second = b.infer(obs)
    assert first["actions"].shape == (10, spec.layout.valid_action_dim)
    assert first["actions"].tobytes() == second["actions"].tobytes()
    assert a.recipe is not b.recipe and a.recipe.action is b.recipe.action
    assert np.isfinite(first["actions"]).all()
    # Shared periodic look policy is evaluated through the ordinary P1 executor.
    looks = []
    for seq in range(1, 8):
        answer = a.infer(dict(obs, __execution__={"decision_seq": seq-1, "executed_count": 5}))
        looks.append(answer["__hit_meta__"]["stage_calls"][0])
    assert 1 in looks


def test_model_native_capture_P6_closed_NPZ_H5_roundtrip(make_adapter, req_context, tmp_path):
    from exp.r17_integration.debug.capture_bridge import CaptureBridge, export_h5
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    from exp.r17_integration.debug.validate import model_binding
    adapter = make_adapter().with_sampler(SamplerProfile(num_steps=2))
    req_context = replace(req_context, attempt=1)
    p = adapter.prepare(observation(), req_context)
    s1 = adapter.run_stage1(p)
    s2 = adapter.run_stage2(s1)
    result = adapter.sample(s2, adapter.metadata().default_sampler, RequestNoise(9), CaptureSpec(True, True))
    wire = adapter.to_wire(result.normalized_chunk, p)
    decision = DecisionCapture(req_context, "policy", 0, 0, (1, 1, 2), 2, result.normalized_chunk, wire.actions,
                                fields=adapter.retrieval_fields(s1),
                                sampler_receipt={"native_timesteps": result.sampler.native_timesteps,
                                                 "identity": result.sampler.identity},
                                native_snapshots=result.snapshots, model_capture=result.capture)
    trace = adapter.make_trace_adapter().capture(decision)
    record = {"campaign": req_context.campaign, "arm": "full", "bundle_generation": "g1",
              "benchmark_protocol": req_context.benchmark_protocol, "task_name": req_context.task_name,
              "orig_init_idx": 0, "task_data_hash": "fixture", "task_uid": "p4/episode", "attempt": 1,
              "decision_seq": 0, "actual_nfe": 2, "source": "policy", "split": "library"}
    binding = model_binding(adapter.metadata(), req_context.benchmark_protocol)
    bridge = CaptureBridge(tmp_path / "capture", enabled=True, binding=binding)
    bridge.capture(record, trace["arrays"], rawkeys=adapter.retrieval_fields(s1), prompt=p.data["prompt"])
    stats = bridge.close()
    assert stats["drained"] and not stats["errors"]
    reader = CaptureReader(tmp_path / "capture")
    loaded = list(reader.decisions())
    assert len(loaded) == 1
    _, arrays, keys = loaded[0]
    for name, value in trace["arrays"].items():
        assert np.array_equal(arrays[name], value) and arrays[name].dtype == value.dtype
    assert arrays["snapshot_rng_state"].dtype == np.uint8
    assert arrays["snapshot_native_timestep"].dtype == np.int64
    path = export_h5(tmp_path / "capture", tmp_path / "capture.h5")
    import h5py
    with h5py.File(path, "r") as h5:
        assert h5.attrs["r17_closed_ok"]
        assert np.array_equal(h5["step_000000/arrays/snapshot_native_timestep"], arrays["snapshot_native_timestep"])


@pytest.mark.parametrize("mode", ["latest", "weight", "formal", "language"])
def test_strict_loader_fails_fast_on_mutability_digest_and_diagnostic_gate(tmp_path, mode):
    root = tmp_path / ("latest" if mode == "latest" else "immutable")
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"untrusted dummy, not loaded")
    (root / "metadata.json").write_text((Path(__file__).parents[3] / "models/dp/samples/audited_libero_metadata.json").read_text())
    spec = LoadSpec(str(root), "metaworld")
    if mode == "weight":
        spec = replace(spec, expected_weight_sha256="wrong")
    elif mode == "formal":
        spec = replace(spec, diagnostic=False)
    elif mode == "language":
        meta = json.loads((root / "metadata.json").read_text())
        meta["wrapper"] = {"language_kind": "pooled"}
        (root / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_adapter(spec)


def test_explicit_DDIM_provider_family_is_not_silently_DDPM(make_adapter, monkeypatch):
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.models.dp.provider import register_provider
    from exp.r17_integration.models.dp.serve import plain_config
    from exp.r17_integration.common.provider import validate_binding
    monkeypatch.setattr(registry, "_providers", {})
    monkeypatch.setattr(registry, "_traces", {})
    ddpm_adapter = make_adapter()
    ddpm_provider = register_provider(ddpm_adapter)
    adapter = ddpm_adapter.with_sampler(SamplerProfile("ddim", 100, .5))
    provider = register_provider(adapter)
    assert provider.provider_id != ddpm_provider.provider_id
    config = plain_config(provider, num_steps=2)
    _, sampler = validate_binding(config.adapter_binding)
    assert sampler.family == "ddim" and sampler.config["eta"] == .5
    assert "ddim_baseline" in adapter.metadata().model_id


def test_P2_episode_runner_canonical_wire_protocol_and_actual_execution(make_adapter, monkeypatch):
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, R17MetaWorldEpisodeRunner
    from exp.r17_integration.conductor.manifest import episode_tasks
    from exp.r17_integration.models.dp.provider import make_connection_factory, register_provider
    from exp.r17_integration.models.dp.serve import plain_config
    from exp.r17_integration.tests.benchmarks.test_metaworld import Env
    from exp.r17_integration.common.config import build_shared_storage
    from openpi.conductor.task import ServerEndpoint
    monkeypatch.setattr(registry, "_providers", {})
    monkeypatch.setattr(registry, "_traces", {})
    adapter = make_adapter()
    provider = register_provider(adapter)
    config = plain_config(provider, num_steps=2, commit_horizon=10)
    captures = []
    factory = make_connection_factory(adapter, provider=provider, default_config=config,
                                     default_storage=build_shared_storage(config), capture_sink=captures.append)
    class Client:
        def select_bundle(self, bid):
            assert bid == "default"
            self.policy = factory()
            self.policy.on_task_begin()
        def episode_start(self, **kwargs):
            self.policy.on_episode_start(**kwargs)
        def infer(self, obs):
            return self.policy.infer(obs)
        def episode_end(self, **kwargs):
            self.policy.on_episode_end(**kwargs)
        def close(self):
            self.policy.on_task_end()
    client = Client()
    env = Env(success=7)
    acknowledgments = []
    runner = R17MetaWorldEpisodeRunner(env_factory=lambda *args: env, client_factory=lambda _: client,
                                       execution_ack=acknowledgments.append, framework_context=True)
    spec = adapter.metadata()
    task = episode_tasks(campaign="p4_fake", arm="full", bundle_id="default",
                          endpoint=ServerEndpoint("127.0.0.1", 1), model_identity=spec.checkpoint_identity,
                          norm_identity=spec.norm_identity, layout_identity=spec.layout.identity,
                          num_steps=2, partition="smoke", names=["reach-v3"], indices=[0])[0]
    from exp.r17_integration.common.recipe import bundle_identity
    task = replace(task, extra={**task.extra, "bundle_identity": bundle_identity(config)})
    result = runner.run(replace(task, attempt=1), lambda *args: None)
    assert result.success and result.n_steps == 7 and env.closed
    assert spec.allowed_benchmark_protocols == (FORMAL_SPEC.identity,)
    assert [r["executed_count"] for r in acknowledgments] == [5, 2]
    assert captures[0].stage_calls == (1, 1, 2) and captures[1].stage_calls == (0, 0, 0)
    assert captures[1].cursor == 5 and client.policy._chunk is None
    assert captures[0].request.task_name == "reach-v3"
