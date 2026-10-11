"""Review A regressions: exercise worker wire and production integration seams."""
from dataclasses import asdict, replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from exp.r17_integration.models.groot.manifest import library_metadata, validate_library
from exp.r17_integration.models.groot.serve import default_config, make_connection_factory
from exp.r17_integration.common.config import build_shared_storage
from .test_shared_factory import published
from .conftest import begin_episode


def teacher_store(directory, manifest, *, episodes=3, seed=517):
    """A sealed diagnostic teacher store with three complete ordered episodes."""
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.search.store import write_store
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        for step in range(8):
            rows.append({"campaign": "P5-review-teacher", "arm": "full-k4", "bundle_generation": "teacher4",
                "benchmark_protocol": manifest.benchmark_protocol, "task_name": "reach-v3", "orig_init_idx": 10+ep,
                "task_data_hash": f"cpu-task-{ep}", "task_uid": f"P5-library-{ep}", "attempt": 1,
                "decision_seq": step, "split": "library", "success": True})
    n = len(rows)
    arrays = {"key_vision_0": rng.normal(size=(n, 16*manifest.hidden_width)).astype(np.float32),
              "key_robot_state": rng.normal(size=(n, 4)).astype(np.float32),
              "state": rng.normal(size=(n, 4)).astype(np.float32),
              "normalized_actions": rng.normal(size=(n, 16, 32)).astype(np.float32)}
    return write_store(directory, model_binding(manifest.model_spec(), manifest.benchmark_protocol), rows, arrays,
                       source_hashes={"fixture": "CPU diagnostic; no checkpoint data"},
                       metadata={"key_fields": ["vision_0", "robot_state"], "diagnostic": True})


# Diagnostic normalized owner-IR coefficients for CPU fixtures only (not measured
# GR00T-MW prices). target_ir bundles must carry a bound r17.ir_costs.v1 profile.
FIXTURE_IR_COSTS = dict(vision=.148, miss=.7)


def recipe_config(config, store, *, corrector=False, guard=True, periodic_gap=None, target_ir=None):
    from exp.r17_integration.common.benchmarks import get_benchmark, task_table_identity
    from exp.r17_integration.common.config import RecipeConfig
    from exp.r17_integration.common.contracts import digest
    from exp.r17_integration.common.ir import bound_costs
    from exp.r17_integration.common.recipe import fit_identity
    protocol = config.adapter_binding["benchmark_protocol"]
    teacher_identity = digest(store.manifest["binding"]["sampler"])
    config.recipe = RecipeConfig(enabled=True, store_path=str(store.directory),
        store_digest=store.manifest["content_digest"], teacher_sampler_identity=teacher_identity,
        corrector=corrector, guard=guard, periodic_gap=periodic_gap, target_ir=target_ir,
        k=4, kref=2, early=False, ordered_tasks=list(get_benchmark(protocol).ordered_tasks),
        task_table_identity=task_table_identity(protocol))
    if target_ir is not None:
        config.recipe.ir_costs = bound_costs(config, source_identity="cpu-fixture-diagnostic-not-measured",
                                             **FIXTURE_IR_COSTS)
    config.recipe.fit_identity = fit_identity(config)
    return config


class EpisodeEnv:
    """CPU environment for the actual P2 runner/kernel, without simulator imports."""
    def __init__(self):
        self.steps = 0
        self.actions = []
        self.closed = False
        # Native-shaped numeric fields let the actual P2 physical adapter run on CPU.
        self._r17_task_data_hash = "P5-worker-cpu-task"
        self._target_pos = np.array([.1, .2, .3])
        self.model = NS(nbody=1, ngeom=0, njnt=0, nsite=0, nu=4,
                        geom_bodyid=np.array([], np.int64), body_parentid=np.array([0]),
                        jnt_bodyid=np.array([], np.int64), jnt_type=np.array([], np.int64),
                        jnt_qposadr=np.array([], np.int64), jnt_dofadr=np.array([], np.int64))
        for kind in ("body", "geom", "joint", "site", "actuator"):
            setattr(self.model, kind, lambda i, k=kind: NS(name=f"{k}{i}"))
        self.data = NS(qpos=np.zeros(4, np.float64), qvel=np.zeros(4, np.float64),
                       ctrl=np.zeros(4), act=np.empty(0), xpos=np.zeros((1, 3)), xquat=np.zeros((1, 4)),
                       cvel=np.zeros((1, 6)), site_xpos=np.empty((0, 3)), contact=[], ncon=0, time=0.)

    def reset(self):
        self.steps = 0
        return np.arange(39, dtype=np.float32) / 100, {}

    def render(self):
        frame = np.full((480, 480, 3), 2, np.uint8)
        frame[0, 0] = [3, 4, 5]
        return frame

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        self.steps += 1
        self.data.qpos[0] = self.steps
        self.data.ctrl = np.asarray(action).copy()
        self.data.time = self.steps * .02
        return np.arange(39, dtype=np.float32) / 100, 0., False, False, {"success": self.steps == 22}

    def close(self):
        self.closed = True


class PolicyClient:
    """Pass the P2 client's actual messages unchanged to the shared policy."""
    def __init__(self, policy):
        self.policy = policy
        self.requests = []

    def select_bundle(self, bundle):
        assert bundle == "default"
        self.policy.on_task_begin()

    def episode_start(self, **kwargs):
        self.policy.on_episode_start(**kwargs)

    def episode_end(self, success=False, *, execution=None):
        self.policy.on_episode_end(success=success, execution=execution)

    def acknowledge_execution(self, execution):
        return self.policy.acknowledge_execution(execution)

    def infer(self, observation):
        self.requests.append(observation)
        return self.policy.infer(observation)

    def close(self):
        self.policy.on_task_end()


def test_review_a1_real_p2_runner_worker_wire_reaches_groot(published):
    from exp.r17_integration.benchmarks.metaworld import MetaWorldBenchmarkAdapter
    from exp.r17_integration.conductor.manifest import episode_tasks
    from openpi.conductor.task import ServerEndpoint
    from exp.r17_integration.common.recipe import bundle_identity
    cfg = default_config(published, commit_horizon=10)
    captures = []
    policy = make_connection_factory(published, cfg, build_shared_storage(cfg), capture_sink=captures.append)()
    client = PolicyClient(policy)
    environment = EpisodeEnv()
    runner = MetaWorldBenchmarkAdapter(env_factory=lambda *a: environment).make_episode_runner(
        client_factory=lambda endpoint: client, framework_context=True)
    task = episode_tasks(campaign="P5-review", arm="full", bundle_id="default",
        endpoint=ServerEndpoint("cpu", 1), model_identity=published.model_spec.checkpoint_identity,
        norm_identity=published.model_spec.norm_identity, layout_identity=published.model_spec.layout.identity,
        num_steps=4, partition="smoke", names=["reach-v3"], indices=[7],
        yaml_identity=bundle_identity(cfg), bundle_identity=bundle_identity(cfg))[0]
    try:
        result = runner.run(task, lambda *a: None)
    finally:
        runner.close()
    assert result.success and result.n_steps == 7 and environment.closed
    assert len(client.requests) == len(captures) == 2
    assert set(client.requests[0]) == {"observation/image", "observation/state", "prompt", "__request__"}
    assert client.requests[0]["__request__"]["original_init_idx"] == 7
    assert client.requests[0]["__request__"]["task_name"] == "reach-v3"
    assert client.requests[1]["__execution__"]["executed_count"] == 5
    assert [c.stage_calls for c in captures] == [(1, 1, 4), (0, 0, 0)]
    assert policy.adapter.backend.apply_count == policy.adapter.backend.inverse_count == 1
    # The genuine kernel rotates the render before making the worker message.
    np.testing.assert_array_equal(client.requests[0]["observation/image"][-1, -1], [3, 4, 5])


def test_review_a1_both_wire_key_sets_have_exact_actions(published, obs):
    cfg = default_config(published)
    factory = make_connection_factory(published, cfg, build_shared_storage(cfg))
    canonical = factory().infer(obs)
    worker = factory().infer({"observation/image": obs["image"], "observation/state": obs["state"],
                             "prompt": obs["prompt"], "__request__": obs["__request__"]})
    assert canonical["actions"].tobytes() == worker["actions"].tobytes()


@pytest.mark.parametrize("miss_k", [2, 1])
def test_review_a4_k4_teacher_full_actions_compatible_with_reduced_miss(manifest, miss_k):
    teacher = library_metadata(manifest)
    endpoint = replace(manifest, denoising_steps=miss_k)
    assert manifest.model_spec().default_sampler.identity != endpoint.model_spec().default_sampler.identity
    validate_library(teacher, endpoint)


def test_review_a5_metadata_is_real_p2_worker_model_binding(manifest):
    from exp.r17_integration.debug.validate import model_binding
    metadata = manifest.to_dict()
    actual = model_binding(metadata["r17_model_spec"], manifest.benchmark_protocol)
    assert actual == model_binding(manifest.model_spec(), manifest.benchmark_protocol)


def test_review_a12_direct_hooks_not_advertised_as_deployed(manifest):
    features = manifest.model_spec().capabilities.features
    for hook in ("trace", "batch_stage3", "native_resume", "warm_reset_hook", "library"):
        assert features[hook].status == "implemented_unverified", hook
        with pytest.raises(ValueError, match="unavailable"):
            manifest.model_spec().capabilities.require(hook)


@pytest.mark.parametrize("commit_horizon", [7, 12])
def test_review_a9_gr_entry_rejects_non_replan_aligned_commit(published, commit_horizon):
    # Actual P2 formal decisions advance by five controls, not arbitrary prefixes.
    with pytest.raises(ValueError, match="multiple"):
        default_config(published, commit_horizon=commit_horizon)


@pytest.mark.parametrize("miss_k", [4, 2, 1])
def test_review_a4_stock_teacher_artifact_and_recipe_miss_grids_independent(
        guarded_stub, manifest, obs, tmp_path, monkeypatch, miss_k):
    from exp.r17_integration.models.groot.provider import make_provider, register_provider
    from exp.r17_integration.models.groot.serve import make_bundle_validator
    from exp.r17_integration.search.build_library import save_stock_artifact
    import exp.r17_integration.common.provider as registry
    monkeypatch.setattr(registry, "_providers", {})
    monkeypatch.setattr(registry, "_traces", {})
    teacher = teacher_store(tmp_path / "teacher4", manifest)
    artifact = tmp_path / "teacher4.pkl"
    save_stock_artifact(teacher, artifact)
    provider = register_provider(make_provider(guarded_stub(miss_k), replace(manifest, denoising_steps=miss_k)))
    cfg = recipe_config(default_config(provider, commit_horizon=10), teacher, periodic_gap=1)
    cfg.backend.in_memory.preload_path = str(artifact)
    storage = build_shared_storage(cfg)  # real stock loader + common/P5 validators
    make_bundle_validator(provider)(cfg, storage)
    core = make_connection_factory(provider, cfg, storage)()
    begin_episode(core)
    first = core.infer(obs)
    assert first["__hit_meta__"]["source"] == "library"
    assert first["__hit_meta__"]["stage_calls"] == (1, 0, 0)
    second = core.infer(dict(obs, __execution__={"decision_seq": 0, "executed_count": 5}))
    assert second["__hit_meta__"]["source"] == "policy"
    assert second["__hit_meta__"]["actual_nfe"] == miss_k
    assert cfg.recipe.teacher_sampler_identity == manifest.model_spec().default_sampler.identity
    assert core.sampler.num_steps == miss_k
    assert core.adapter.backend.model.action_head.num_inference_timesteps == miss_k


def test_review_a4_native_snapshot_still_rejects_cross_k(guarded_stub, manifest, obs, context):
    from exp.r17_integration.common.contracts import CaptureSpec
    from exp.r17_integration.common.request_rng import RequestRNG
    from exp.r17_integration.common.serving import full_call
    from exp.r17_integration.models.groot.provider import GrootMetaWorldAdapter
    teacher = GrootMetaWorldAdapter(guarded_stub(4), manifest)
    result, _ = full_call(teacher, obs, context, capture=CaptureSpec(True, True))
    endpoint = GrootMetaWorldAdapter(guarded_stub(2), replace(manifest, denoising_steps=2))
    prepared = endpoint.prepare(obs, context)
    s2 = endpoint.run_stage2(endpoint.run_stage1(prepared))
    with pytest.raises(ValueError, match="identity"):
        endpoint.resume(s2, result.snapshots[0], endpoint.metadata().default_sampler, RequestRNG(context))


@pytest.mark.parametrize("miss_k", [4, 2, 1])
def test_review_a5_real_serve_cli_and_worker_debug_over_localhost(
        guarded_stub, manifest, tmp_path, monkeypatch, miss_k):
    """Real CLI assembly, WS transport, worker main and P2 episode; CPU model/env only."""
    import asyncio
    import socket
    import exp.r17_integration.common.provider as registry
    import exp.r17_integration.common.websocket_policy_server as r17ws
    import exp.r17_integration.models.groot.loader as loader
    import exp.r17_integration.models.groot.serve as entry
    import exp.r17_integration.conductor.worker_entry as worker
    import openpi.conductor.worker as worker_transport
    from openpi.serving import websocket_policy_server as stock_registry
    from exp.mw_finetune import rendering
    from exp.r17_integration.conductor.manifest import episode_tasks
    from exp.r17_integration.common.recipe import bundle_identity
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    from exp.r17_integration.debug.validate import model_binding
    from openpi.conductor.task import ServerEndpoint
    from exp.r17_integration.models.groot.provider import make_provider
    monkeypatch.setattr(registry, "_providers", {})
    monkeypatch.setattr(registry, "_traces", {})
    monkeypatch.setattr(stock_registry, "_bundles", {})
    monkeypatch.setattr(stock_registry, "_current_bundle", None)
    monkeypatch.setattr(stock_registry, "_bundle_version", 0)
    endpoint_manifest = replace(manifest, denoising_steps=miss_k)
    backend = guarded_stub(miss_k)
    monkeypatch.setattr(loader, "load_checkpoint", lambda *a, **kw: (backend, endpoint_manifest))
    monkeypatch.setattr(rendering, "configure", lambda *a: None)
    monkeypatch.setattr(rendering, "make_env", lambda *a: EpisodeEnv())
    monkeypatch.setattr(worker, "attest_worker", lambda *a: {"cpu_diagnostic": True})
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    cfg = default_config(make_provider(backend, endpoint_manifest), commit_horizon=10)
    task = episode_tasks(campaign="P5-deployed-debug", arm=f"full-k{miss_k}", bundle_id="default",
        endpoint=ServerEndpoint("127.0.0.1", port), model_identity=manifest.checkpoint_identity,
        norm_identity=manifest.norm_identity, layout_identity=manifest.model_spec().layout.identity,
        num_steps=miss_k, partition="smoke", names=["reach-v3"], indices=[7],
        yaml_identity=bundle_identity(cfg), bundle_identity=bundle_identity(cfg),
        task_data_hashes={("reach-v3", 7): "P5-worker-cpu-task"})[0]
    results = []

    class OneTaskWorker:
        def __init__(self, worker_id, server_key, runner, **kwargs):
            self.runner = runner

        def run_forever(self):
            results.append(self.runner.run(task, lambda *a: None))

    monkeypatch.setattr(worker_transport, "WorkerLoop", OneTaskWorker)
    real_server = r17ws.WebsocketPolicyServer

    class DrivingServer(real_server):
        def __init__(self, *args, **kwargs):
            self.ready = asyncio.Event()
            kwargs["ready_callback"] = self.ready.set
            assert kwargs["metadata"]["r17_model_spec"] == asdict(endpoint_manifest.model_spec())
            super().__init__(*args, **kwargs)

        def serve_forever(self, **kwargs):
            async def drive():
                serving = asyncio.create_task(self.run(stop_on_request=True))
                try:
                    await asyncio.wait_for(self.ready.wait(), 5)
                    await asyncio.wait_for(asyncio.to_thread(worker.main, [
                        "--worker-id", "P5-cpu", "--server-key", f"127.0.0.1:{port}",
                        "--driver-host", "unused", "--driver-port", "1", "--debug-directory", str(tmp_path / "physical")]), 10)
                finally:
                    self.request_stop()
                    await asyncio.wait_for(serving, 5)
            asyncio.run(drive())

    monkeypatch.setattr(r17ws, "WebsocketPolicyServer", DrivingServer)
    entry.main(["--checkpoint", "cpu-stub", "--port", str(port), "--host", "127.0.0.1",
                "--denoising-steps", str(miss_k), "--device", "cpu", "--commit-horizon", "10",
                "--tokenizer-file", "cpu-tokenizer", "--code-file", "cpu-source",
                "--debug-directory", str(tmp_path / "model")])
    assert len(results) == 1 and results[0].success and results[0].n_steps == 7
    binding = model_binding(endpoint_manifest.model_spec(), manifest.benchmark_protocol)
    model_paths = list((tmp_path / "model").glob("binding_*"))
    physical_paths = list((tmp_path / "physical").glob("binding_*"))
    assert len(model_paths) == len(physical_paths) == 1
    model_reader = CaptureReader(model_paths[0], expected_binding=binding)
    physical_reader = CaptureReader(physical_paths[0], expected_binding=binding)
    decisions = list(model_reader.decisions())
    physical = list(physical_reader.physical())
    assert len(decisions) == 2 and physical
    assert [row[0]["executed_count"] for row in decisions] == [5, 2]
    assert [row[0]["cursor_end"] for row in decisions] == [5, 7]
    assert decisions[-1][0]["terminal"] is True
    assert [row[0]["stage_calls"] for row in decisions] == [[1, 1, miss_k], [0, 0, 0]]
    assert backend.model.action_head.nfe == miss_k
    assert len(decisions[0][0]["native_snapshots"]) == miss_k - 1
    assert decisions[0][1]["normalized_actions"].shape == (16, 32)
    assert set(decisions[0][2]) == {"vision_0", "robot_state"}
    controls = [row for row in physical if row[0]["event"] == "control"]
    assert len(controls) == 7 and controls[0][1]["qpos"].dtype == np.float64
    assert all(row[0]["task_uid"] == task.task_uid for row in physical)
    assert all(row[0]["orig_init_idx"] == 7 for row in physical)
    assert decisions[0][0]["task_uid"] == task.task_uid
