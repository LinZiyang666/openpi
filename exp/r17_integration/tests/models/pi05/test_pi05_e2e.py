"""CPU first-target loopback: original WS, stock cache, π native sampler hooks."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import socket
import threading
import numpy as np
import pytest
import yaml

from exp.r17_integration.tests.models.pi05.test_pi05 import fixture_adapter, observation, request
from exp.r17_integration.tests.benchmarks.test_metaworld import Env
from exp.r17_integration.models.pi05.provider import register_provider, config_document, make_connection_factory, make_capture_sink_factory
from exp.r17_integration.common.provider import binding_for
from exp.r17_integration.common.serving import full_call, make_bundle_validator
from exp.r17_integration.common.layers import LibraryData, fit_library
from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, MetaWorldBenchmarkAdapter
from exp.r17_integration.conductor.manifest import EndpointSpec, episode_tasks
from exp.r17_integration.conductor.strategy import MetaWorldStrategy
from exp.r17_integration.conductor.driver import MetaWorldConductorDriver
from openpi.conductor.task import ServerEndpoint
from openpi.conductor.protocol import task_from_wire, result_to_wire


def tiny_library_data(adapter):
    arrays, keys, states = [], [], []
    for i in range(9):
        obs = observation()
        obs["observation/image"][:] = i*20
        obs["observation/state"] += i*.1
        prepared = adapter.prepare(obs, replace(request(), decision_seq=i))
        fields = adapter.retrieval_fields(adapter.run_stage1(prepared))
        keys.append(fields["vision_0"].numpy())
        states.append(fields["robot_state"].numpy())
        normalized = np.full((5, 32), i*.03, dtype=np.float32)
        normalized[:, 3] = -.4
        arrays.append(normalized)
    data = LibraryData({"vision_0": np.array(keys)}, np.array(states), np.array(arrays),
                       np.full(9, 43), np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3),
                       source_digest="synthetic CPU velocity library, not a quality experiment")
    return data


def tiny_library(adapter):
    data = tiny_library_data(adapter)
    return fit_library(data, layout=adapter.metadata().layout, task_count=50, commit_horizon=5,
                       k=2, kref=2, corrector=True, periodic_gap=2, early=True)



def tiny_store(adapter, directory):
    from exp.r17_integration.search.store import write_store
    from exp.r17_integration.debug.validate import model_binding
    data = tiny_library_data(adapter)
    rows = [{"campaign": "cpu_teacher", "arm": "teacher", "bundle_generation": "teacher_gen1",
             "benchmark_protocol": FORMAL_SPEC.identity, "task_name": "reach-v3", "orig_init_idx": 10 + i // 3,
             "task_data_hash": "fixture_teacher_" + str(i // 3), "task_uid": "teacher_" + str(i // 3),
             "attempt": 1, "decision_seq": i % 3, "split": "library"} for i in range(9)]
    store = write_store(directory, model_binding(adapter.metadata(), FORMAL_SPEC.identity), rows,
                        {"key_vision_0": data.features["vision_0"], "state": data.state, "normalized_actions": data.action})
    recipe = {"schema": "r17.recipe.v1", "enabled": True, "store_path": str(directory),
              "store_digest": store.manifest["content_digest"], "teacher_sampler_identity": adapter.metadata().default_sampler.identity,
              "corrector": True, "guard": False, "periodic_gap": 2, "k": 2, "kref": 2,
              "task_table_identity": FORMAL_SPEC.task_table_identity,
              "ordered_tasks": [r[0] for r in FORMAL_SPEC.tasks]}
    return store, recipe

def test_first_pi_mw_real_ws_two_clients_two_yamls_conductor_and_capture(tmp_path):
    from exp.r17_integration.common.config import load_cache_config, build_shared_storage
    from openpi.serving import websocket_policy_server as W
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from exp.r17_integration.debug.capture_bridge import CaptureBridge, PhysicalSink, DecisionSink
    from exp.r17_integration.debug.validate import model_binding

    adapter, model = fixture_adapter()
    provider = register_provider(adapter, provider_id="p2_cpu_ws_" + tmp_path.name)
    full, retrieval = tmp_path / "full.yaml", tmp_path / "retrieval.yaml"
    full.write_text(yaml.safe_dump(config_document(provider, evidence_dir=tmp_path / "miss_evidence")))
    teacher, recipe = tiny_store(adapter, tmp_path / "teacher")
    retrieval.write_text(yaml.safe_dump(config_document(provider, k=2, evidence_dir=tmp_path / "miss_evidence", recipe=recipe)))
    config = load_cache_config(full)
    storage = build_shared_storage(config)
    fitted = tiny_library(adapter)
    assert fitted.use_corrector and not fitted.use_guard and fitted.periodic_gap == 2
    captures, wrappers = [], []
    model_capture = make_capture_sink_factory(tmp_path / "model_capture")
    def sink_factory(core):
        sink = model_capture(core)
        def capture(record):
            captures.append(record)
            sink(record)
        capture.on_execution = sink.on_execution
        return capture
    factory = make_connection_factory(provider, config, storage, capture_sink_factory=sink_factory,
                                      recipe_factory=__import__("exp.r17_integration.common.recipe", fromlist=["make_recipe_factory"]).make_recipe_factory(provider))
    def tracked_factory(base, bundle):
        p = factory(base, bundle)
        wrappers.append(p)
        return p
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    ready = threading.Event()
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    server = WebsocketPolicyServer(factory(None, "default"), host="127.0.0.1", port=port,
             concurrent=True, connection_policy_factory=tracked_factory, ready_callback=ready.set,
             bundle_validator=make_bundle_validator(provider))
    thread = threading.Thread(target=lambda: server.serve_forever(stop_on_request=True), daemon=True)
    saved = (dict(W._bundles), W._current_bundle, W._bundle_version)
    thread.start()
    assert ready.wait(10), "local server did not become ready"
    clients = []
    from exp.r17_integration.debug.capture_bridge import PhysicalCaptureFactory
    from exp.r17_integration.models.pi05.adapter import sampler_for_steps
    physical = PhysicalCaptureFactory(tmp_path / "physical_capture", {"r17_model_spec": asdict(adapter.metadata())},
                                      sampler_resolver=sampler_for_steps)
    try:
        endpoint = ServerEndpoint("127.0.0.1", port)
        spec = adapter.metadata()
        endpoints = [EndpointSpec(endpoint, spec.checkpoint_identity, spec.norm_identity, spec.layout.identity, FORMAL_SPEC.identity)]
        tasks = []
        for arm, bundle, k in (("full", "full_gen1", 10), ("retrieval", "retrieval_gen1", 2)):
            tasks.extend(episode_tasks(campaign="cpu", arm=arm, bundle_id=bundle, endpoint=endpoint,
                        model_identity=spec.checkpoint_identity, norm_identity=spec.norm_identity,
                        layout_identity=spec.layout.identity, partition="smoke", num_steps=k, names=["reach-v3"],
                        bundle_generation=bundle, task_data_hashes={("reach-v3", i): "fixture_hash" for i in (0, 1)}))
        strategy = MetaWorldStrategy(tasks, endpoints=endpoints, bundles={"full_gen1": full, "retrieval_gen1": retrieval})
        def client_factory(s):
            c = WebsocketClientPolicy(s.host, s.port)
            clients.append(c)
            return c
        driver = MetaWorldConductorDriver(strategy, dispatch_path=tmp_path / "dispatch.jsonl",
                   yaml_weights={t.yaml_id: 2 for t in tasks}, servers=[endpoint], journal_path=str(tmp_path / "journal.jsonl"),
                   ctl_factory=client_factory)
        driver.drive_stages_once()
        b = MetaWorldBenchmarkAdapter(env_factory=lambda *a: Env(success=12))
        runners = [b.make_episode_runner(client_factory, capture_sink=physical, framework_context=True) for _ in range(2)]
        def work(r):
            while True:
                _, payload = driver.handle_pull(endpoint.key)
                if "task" not in payload:
                    return
                task = task_from_wire(payload["task"])
                result = r.run(task, lambda *a: None)
                driver.handle_result(result_to_wire(result))
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(work, runners))
        driver.drive_stages_once()
        rows = [json.loads(r) for r in (tmp_path / "journal.jsonl").read_text().splitlines()]
        assert len(rows) == 4 and all(r["accepted"] is True and r.get("error") is None for r in rows)
        assert all(r["attempt"] == 1 for r in rows)
        assert {r["task_uid"] for r in rows} == {t.task_uid for t in tasks}
        assert any(c.source == "library" and c.actual_nfe == 0 for c in captures)
        assert any(c.source == "policy" and c.actual_nfe == 10 for c in captures)
        assert any(c.source == "policy" and c.actual_nfe == 2 for c in captures)
        assert all(c.normalized_chunk.shape == (5, 32) and c.wire_actions.shape[1] == 4 for c in captures)
        recipes = [p.recipe for p in wrappers if p.recipe is not None]
        assert len(recipes) >= 2 and len({id(p) for p in recipes}) == len(recipes)
        assert len({id(p.orchestrator) for p in wrappers}) == len(wrappers)
        for c in captures:
            assert c.request.task_name == "reach-v3" and c.request.original_init_idx in (0, 1)
        # Reject unknown/rebound identity without replacing the current stack.
        c = WebsocketClientPolicy("127.0.0.1", port)
        clients.append(c)
        c.select_bundle("full_gen1")
        with pytest.raises(RuntimeError):
            c.select_bundle("missing")
        bad = config_document(provider, evidence_dir=tmp_path / "miss_evidence")
        bad["adapter_binding"]["checkpoint_identity"] = "wrong_checkpoint"
        with pytest.raises(RuntimeError):
            c.load_cache_config(yaml_content=yaml.safe_dump(bad), yaml_id="bad", bundle_id="bad")
        c.episode_start("metaworld_mt50", "reach-v3", 0, extra_metadata={"task_uid": "cpu:eval:43:8",
                         "task_data_hash": "fixture_hash", "original_init_idx": 8,
                         "campaign": "cpu", "arm": "full", "bundle_generation": "full_gen1"})
        obs = observation()
        from exp.r17_integration.benchmarks.metaworld import decision_seed
        probe_request = replace(request(), benchmark_protocol=FORMAL_SPEC.identity, attempt=1, task_id=43,
                                arm="full", bundle_generation="full_gen1", explicit_seed=decision_seed("reach-v3", 8, 0))
        obs["__request__"] = asdict(probe_request)
        response = c.infer(obs)
        _, expected = full_call(adapter, observation(), probe_request)
        assert response["actions"].dtype == expected.actions.dtype
        assert response["actions"].tobytes() == expected.actions.tobytes()
        from exp.r17_integration.benchmarks.execution import end_episode
        end_episode(c, {"success": False}, [{"decision_seq": 0, "executed_count": 5, "terminal": True}])
        for r in runners:
            r.close()
        assert all(s["drained"] for s in physical.close())
        assert all(s["drained"] for s in model_capture.close())
        from exp.r17_integration.debug.reader_profiles import CaptureReader
        readers = [CaptureReader(bridge.directory) for bridge in model_capture.bridges.values()]
        assert {r.manifest["binding"]["sampler"]["num_steps"] for r in readers} == {10, 2}
        written = [row for reader in readers for row in reader.decisions()]
        for reader in readers:
            k = reader.manifest["binding"]["sampler"]["num_steps"]
            assert all(row[0]["actual_nfe"] in (0, k) for row in reader.decisions())
        assert len(written) == len(captures)
        assert all(r[0]["task_data_hash"] == "fixture_hash" for r in written)
        assert {r[0]["task_uid"] for r in written if r[0]["task_uid"] != "cpu:eval:43:8"} == {t.task_uid for t in tasks}
        physical_readers = [CaptureReader(b.directory) for b in physical.bridges.values()]
        assert {r.manifest["binding"]["sampler"]["num_steps"] for r in physical_readers} == {10, 2}
        assert sum(len(list(r.physical())) for r in physical_readers) > 48
    finally:
        for c in clients:
            c.close()
        physical.close()
        model_capture.close()
        server.request_stop()
        thread.join(timeout=10)
        W._bundles.clear()
        W._bundles.update(saved[0])
        W._current_bundle, W._bundle_version = saved[1:]
    assert not thread.is_alive()


def test_p6_sealed_store_to_p1_recipe_bridge_and_identity_rejection(tmp_path):
    from exp.r17_integration.search.store import write_store
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.models.pi05.provider import recipe_from_store
    a, _ = fixture_adapter()
    fitted = tiny_library(a)
    rows = []
    for i in range(9):
        rows.append({"campaign": "cpu", "arm": "teacher", "bundle_generation": "teacher",
                     "benchmark_protocol": FORMAL_SPEC.identity, "task_name": "reach-v3", "task_id": 43,
                     "orig_init_idx": 10+i//3, "task_data_hash": "fixture", "task_uid": f"teacher:eval:43:{10+i//3}",
                     "attempt": 1, "decision_seq": i%3, "split": "library"})
    # P6 writes/validates bytes; P2 consumes only named library arrays; P1 fits.
    features = np.array([np.arange(32768, dtype=np.float32)*(.1+i*.01) for i in range(9)])
    directory = tmp_path / "store"
    write_store(directory, model_binding(a.metadata(), FORMAL_SPEC.identity), rows,
                {"key_vision_0": features, "state": fitted.state, "normalized_actions": fitted.action})
    recipe = recipe_from_store(a, directory)
    assert recipe.layout.identity == a.spec.layout.identity
    assert recipe.use_corrector and not recipe.use_guard
    assert recipe.fit_evidence["partition"] == "library"
    assert all(r["query_episode"] not in r["donor_episodes"] for r in recipe.fit_evidence["loeo"])
    original = a.spec
    a.spec = replace(original, norm_identity="foreign_norm")
    with pytest.raises(ValueError, match="binding"):
        recipe_from_store(a, directory)


def test_emit_configs_and_plan_only_are_model_free_and_reject_changed_resume(tmp_path):
    from exp.r17_integration.models.pi05.serve import main as serve
    from exp.r17_integration.conductor.driver import main as conductor
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"content-only dummy, never load")
    norm = checkpoint / "assets/metaworld_mt50/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    configs = tmp_path / "configs"
    serve(["--checkpoint", str(checkpoint), "--emit-configs", str(configs)])
    assert {p.name for p in configs.iterdir()} == {"k10.yaml", "k2.yaml", "k1.yaml", "model_manifest.json"}
    m = json.loads((configs / "model_manifest.json").read_text())
    assert m["spec"]["checkpoint_identity"] == m["checkpoint_identity"]
    assert m["spec"]["layout"]["horizon"] == 5
    assert [s["num_steps"] for s in m["available_samplers"]] == [10, 2, 1]
    argv = ["--manifest", str(configs / "model_manifest.json"), "--endpoint", "never_connect:9",
            "--full-yaml", str(configs / "k10.yaml"), "--k2-yaml", str(configs / "k2.yaml"),
            "--output", str(tmp_path / "plan"), "--smoke", "--plan-only"]
    conductor(argv)
    plan = json.loads((tmp_path / "plan/episode_manifest.json").read_text())
    assert plan["expected_episodes"] == 2 and plan["quality_sr_claim"] is False
    assert [t["orig_init_state_idx"] for t in plan["tasks"]] == [0, 1]
    assert all(t["extra"]["task_name"] == "reach-v3" for t in plan["tasks"])
    assert plan["benchmark"]["expected_formal_episodes"] == 500
    conductor(argv)  # same plan is idempotent
    with pytest.raises(SystemExit) as error:
        conductor([a for a in argv if a not in {"--smoke", "--plan-only"}])
    assert error.value.code == 2
    path = configs / "k2.yaml"
    path.write_text(path.read_text() + "\n# changed content must invalidate resume\n")
    with pytest.raises(ValueError, match="resume"):
        conductor(argv)
