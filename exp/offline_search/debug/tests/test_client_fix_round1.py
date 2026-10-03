"""Fix round 1 regressions: real binding shapes, diagnostics, oracle and ops."""
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from exp.offline_search.debug.client.adapter import MujocoAdapter
from exp.offline_search.debug.client import parity
from exp.offline_search.debug.tests.test_client_capture import FakeEnv, make_episode
from exp.offline_search.debug.tests.test_transport_debug import run_root
from exp.offline_search.debug.transport.collect import certify_tree, verify_arm
from exp.offline_search.debug.transport.validation import validate_episode


def test_nested_render_wrappers_keep_goal_state_and_predicates():
    inner = FakeEnv()
    inner.parsed_problem = {"goal_state": inner.goal_state}
    inner.goal_state = None
    inner.satisfied = True
    outer = NS(sim=inner.sim, env=NS(sim=inner.sim, env=inner))
    adapter = MujocoAdapter(outer)
    assert adapter.inner is inner and adapter.goals == [["on", "mug", "target"], ["open", "drawer"]]
    assert adapter.catalog["object_body_ids"]["mug"] == 2
    assert adapter.capabilities["predicates"]["status"] == "available"
    np.testing.assert_array_equal(adapter.predicates(), [1., 0.])


def test_native_contact_force_receives_raw_structs(monkeypatch):
    env = FakeEnv()
    env.sim.contact_force = None  # exercise the actual mujoco import path
    raw_model, raw_data = object(), object()
    env.sim.model._model, env.sim.data._data = raw_model, raw_data
    calls = []
    module = ModuleType("mujoco")
    def force(model, data, index, out):
        if model is not raw_model or data is not raw_data:
            raise TypeError("native API rejects Python binding wrappers")
        assert out.dtype == np.float64
        calls.append(index)
        out[:] = [1., 2., 3., 4., 5., 6.]
    module.mj_contactForce = force
    monkeypatch.setitem(sys.modules, "mujoco", module)
    adapter = MujocoAdapter(env)
    physical = adapter.physical(env.observe())
    assert calls == [0] and adapter.capabilities["contacts"]["status"] == "available"
    np.testing.assert_array_equal(physical["contact_force"], [[1., 2., 3., 4., 5., 6.]])


def stacked_adapter(anchored):
    env = FakeEnv()
    model = env.sim.model
    old_body, old_geom, old_joint = model.body_id2name, model.geom_id2name, model.joint_id2name
    model.nbody, model.ngeom, model.njnt = 8, 9, 5
    model.body_parentid = np.r_[model.body_parentid, 0, 0, 6]
    model.geom_bodyid = np.r_[model.geom_bodyid, 5, 6, 7]
    model.jnt_bodyid = np.r_[model.jnt_bodyid, 5, 6]
    model.jnt_type = np.r_[model.jnt_type, 0, 0]
    model.body_id2name = lambda i: old_body(i) if i < 5 else ["free_middle", "free_top", "top_child"][i - 5]
    model.geom_id2name = lambda i: old_geom(i) if i < 6 else ["middle_collision", "top_collision", "child_collision"][i - 6]
    model.joint_id2name = lambda i: old_joint(i) if i < 3 else "free_" + str(i)
    # Descendant geom -> free middle -> mug -> articulated fixture. Reversed
    # contact order forces propagation, rather than a one-pass approximation.
    pairs = [(8, 6), (6, 3)] + ([(3, 5)] if anchored else [])
    env.sim.data.contact = [NS(geom1=a, geom2=b) for a, b in pairs]
    env.sim.data.ncon = len(pairs)
    return MujocoAdapter(env)


@pytest.mark.parametrize("anchored", [True, False])
def test_reset_support_chains_include_fixtures_and_reject_unanchored_cycles(anchored):
    adapter = stacked_adapter(anchored)
    check = adapter.resting_selfcheck()
    assert check["status"] == "available" and check["diagnostic_only"]
    assert check["passed"] is anchored
    assert check["supported_body_ids"] == ([2, 5, 6] if anchored else [])
    assert check["missing_support_contacts"] == ([] if anchored else [2, 5, 6])
    assert check["missing_table_contacts"] == [2, 5, 6]  # informative, never admission


@pytest.mark.parametrize("raised", [False, True])
def test_reset_selfcheck_never_rejects_episode_or_receipt(tmp_path, monkeypatch, raised):
    def diagnostic(self):
        if raised:
            raise RuntimeError("selfcheck unavailable")
        return dict(status="error", reason="legacy direct-table diagnostic", missing_table_contacts=[2])
    monkeypatch.setattr(MujocoAdapter, "resting_selfcheck", diagnostic)
    root = run_root(tmp_path)
    client, _, _, _ = make_episode(root / "runs/A/debug/client", n=5)
    directory = root / "runs/A/debug/client" / client.identity["episode_key"]
    assert not client.errors and validate_episode(directory, "A:eval:0:0", 1)["status"] == "complete"
    certify_tree(root, "A")
    receipt = json.loads((root / "runs/A/debug/receipts" / directory.name / "complete.json").read_text())
    assert receipt["status"] == "complete" and verify_arm(root, "A", 1, False)["verified"]


def test_backfill_admission_ignores_support_diagnostic(tmp_path, monkeypatch):
    from exp.offline_search.debug.backfill.replay import replay_episode, robot_state
    monkeypatch.setattr(MujocoAdapter, "resting_selfcheck", lambda self: dict(status="error", reason="diagnostic only"))
    env = FakeEnv()
    source = dict(initial_state=np.zeros(5), action=np.array([[.1, -.2]]), decision_seq=np.array([0], np.int32))
    meta = dict(env_seed=7, suite="fake", episode=0, sha256="0" * 64, provenance="exact fixture controls")
    ep = dict(task_id=0, task="goal", orig_init_state_idx=0, success=False)
    outcome = replay_episode(env, source, meta, ep, [0], robot_state(env.observe())[None], tmp_path)
    assert outcome["status"] == "PASS", outcome


def test_old_capture_source_export_reports_unterminated_final_line(tmp_path):
    from exp.offline_search.debug.backfill.prepare_source import export
    directory = tmp_path / "p3"
    directory.mkdir()
    rows = [dict(ev="attempt_start", environment_seed=7, suite="fake"),
            dict(ev="reset", initial=dict(sim_state=[0., 0.])),
            dict(ev="control", control=0, action_issued=[.1, -.2], decision_step=0), dict(ev="rollout_end")]
    (directory / "controls.jsonl").write_bytes(("".join(json.dumps(row) + "\n" for row in rows)).encode() + b'{"partial":\xff')
    record = export(directory, tmp_path / "source.npz", 0, "exact old capture provenance", "wire")["0"]
    assert len(record["skipped_jsonl_lines"]) == 1 and record["skipped_jsonl_lines"][0]["line"] == 5


def test_partial_oracle_keeps_resolved_goal_and_reports_every_unresolved_subject():
    env = FakeEnv()
    env.goal_state = [["right", "mug", "plate"], ["on", "missing", "target"], ["in", "world", "region"]]
    env.obj_body_id["world"] = 0
    env._eval_predicate = lambda predicate: False
    adapter = MujocoAdapter(env)
    adapter.set_reference()
    payload = adapter.oracle(env.observe())
    assert payload["status"] == "partial" and payload["in_window"] and payload["object_id"] == "mug"
    objects = {row["object_id"]: row for row in payload["objects"]}
    assert objects["mug"]["status"] == "available" and objects["mug"]["predicate_known"]
    assert {r["object_id"] for r in payload["unresolved_objects"]} == {"missing", "world"}
    assert "mapping" in objects["missing"]["reason"] and "movable" in objects["world"]["reason"]
    assert all(r["status"] == "unsupported" and not r["in_window"] for r in payload["unresolved_objects"])
    env.goal_state = [["on", "missing", "target"]]
    empty = MujocoAdapter(env)
    empty.set_reference()
    assert empty.oracle(env.observe())["status"] == "unsupported"


def test_manifest_hashes_capture_paths_and_ignores_analysis(tmp_path, monkeypatch):
    from exp.offline_search.debug.ops import config
    repo = tmp_path / "repo"
    files = ["debug/schema.py", "debug/server/observer.py", "debug/client/capture.py", "debug/transport/sink.py",
             "debug/ops/chain_debug.sh", "closed_loop/plugin.py", "closed_loop/blind.py", "closed_loop/stage_overrides.py",
             "closed_loop/serve_pi05.py", "closed_loop/serve_groot.py", "debug/reader.py", "debug/aug/job.py",
             "debug/tools/decision/common.py", "debug/backfill/replay.py", "debug/tests/test_client_unused.py"]
    for rel in files:
        path = repo / "exp/offline_search" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original\n")
    monkeypatch.setattr(config, "REPO_ROOT", repo)
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "arms.json").write_text('[{"arm":"A"}]')
    original = config.manifest(root, "A")
    assert len(original["code_files"]) == 10
    assert "exp/offline_search/debug/ops/chain_debug.sh" in original["code_files"]
    for rel in files[10:]:
        (repo / "exp/offline_search" / rel).write_text("# analysis changed\n")
    assert config.manifest(root, "A") == original
    (repo / "exp/offline_search/closed_loop/plugin.py").write_text("# capture changed\n")
    with pytest.raises(ValueError, match="differs"):
        config.manifest(root, "A")


def test_receiver_default_and_bounded_remote_probe_without_network(tmp_path, monkeypatch):
    from exp.offline_search.debug.ops import receiver
    assert inspect.signature(receiver.ensure).parameters["port"].default == 23199
    (tmp_path / "state").mkdir()
    (tmp_path / "state/osdebug_receiver.json").write_text('{"port":23199}')
    calls = []
    monkeypatch.delenv("OSDEBUG_RECEIVER_HOST", raising=False)
    monkeypatch.delenv("OSDEBUG_RECEIVER_PORT", raising=False)
    monkeypatch.setattr(receiver.subprocess, "run", lambda *args, **kw: calls.append((args, kw)))
    assert receiver.probe_remote(tmp_path) == dict(reachable=True, from_host="timan107", host="ziyanglin.com", port=23199)
    argv, kwargs = calls[0][0][0], calls[0][1]
    assert argv[:4] == ["tether", "exec", "timan107", "--"] and argv[-2:] == ["ziyanglin.com", "23199"]
    assert "timeout=3" in argv[-3] and kwargs["timeout"] == 15 and kwargs["check"]
    captured = []
    monkeypatch.setattr(receiver, "ensure", lambda root, port: captured.append(port))
    monkeypatch.setattr(sys, "argv", ["receiver", "--run-root", str(tmp_path), "--ensure"])
    receiver.main()
    assert captured == [23199]
    def unavailable(*args, **kw):
        raise subprocess.TimeoutExpired(args[0], 15)
    monkeypatch.setattr(receiver.subprocess, "run", unavailable)
    monkeypatch.setattr(sys, "argv", ["receiver", "--run-root", str(tmp_path), "--probe"])
    with pytest.raises(SystemExit) as exc:
        receiver.main()
    assert exc.value.code == 1
    chain = Path("exp/offline_search/debug/ops/chain_debug.sh").read_text()
    assert chain.index('--run-root "$RUN" --ensure') < chain.index('--run-root "$RUN" --probe') < chain.index('servers_up "$arm" || return 1')
    assert "RECEIVER_UNREACHABLE" in chain and "OSDEBUG_APOOL_RECORD" in chain
    assert "OSDEBUG_APOOL_DIR" in Path("exp/offline_search/debug/ops/remote/run_arm_debug.sh").read_text()


@pytest.mark.parametrize("spec", ["/bare/path", "libero_10=/one", "libero_10=/one,libero_10=/two", "other=/one,libero_spatial=/two"])
def test_parity_rejects_incomplete_or_ambiguous_nondefault_pools(tmp_path, monkeypatch, spec):
    monkeypatch.setattr(sys, "argv", ["parity", "--out", str(tmp_path), "--init-states-dir", spec])
    with pytest.raises(SystemExit) as exc:
        parity.main()  # argument rejection precedes all simulator imports
    assert exc.value.code == 2


@pytest.mark.parametrize("custom", [False, True])
def test_parity_cli_uses_safe_default_inits_and_each_suites_explicit_pool(tmp_path, monkeypatch, custom):
    import examples.libero
    module = ModuleType("examples.libero.main")
    module.LIBERO_DUMMY_ACTION, module.LIBERO_ENV_RESOLUTION = [0., 0., 0., 0., 0., 0., -1.], 8
    loaded = []
    def load(task, suite, task_id, directory):
        loaded.append((suite.name, directory))
        return np.zeros((3, 5))
    module._load_init_states = load
    module._get_libero_env = lambda *args: (FakeEnv(), "goal")
    monkeypatch.setattr(FakeEnv, "close", lambda self: None, raising=False)
    monkeypatch.setitem(sys.modules, "examples.libero.main", module)
    monkeypatch.setattr(examples.libero, "main", module, raising=False)
    benchmark = NS(get_benchmark_dict=lambda: {name: (lambda name=name: NS(name=name, get_task=lambda i: NS(language="goal")))
                                              for name in ("libero_10", "libero_spatial")})
    parent, child = ModuleType("libero"), ModuleType("libero.libero")
    child.benchmark, parent.libero = benchmark, child
    monkeypatch.setitem(sys.modules, "libero", parent)
    monkeypatch.setitem(sys.modules, "libero.libero", child)
    argv = ["parity", "--out", str(tmp_path), "--controls", "12", "--settle-controls", "2"]
    if custom:
        argv += ["--init-states-dir", "libero_10=/l10,libero_spatial=/spatial"]
    monkeypatch.setattr(sys, "argv", argv)
    parity.main()
    assert loaded == [("libero_10", "/l10" if custom else ""), ("libero_spatial", "/spatial" if custom else "")]
    rows = json.loads((tmp_path / "parity.json").read_text())
    assert len(rows) == 6 and all(row["PASS"] for row in rows)
    assert [row["init"] for row in rows] == [0, 1, 2, 0, 1, 2]


def test_float64_wire_response_roundtrips_observer_blocks_and_issued_action_validation(tmp_path):
    from exp.offline_search.debug import fixtures, reader, schema, validate
    from exp.offline_search.debug.server.observer import ServerObserver
    root = tmp_path / "fixture"
    arm = fixtures.make_synthetic_arm(root, n_episodes=1)
    server = arm / "debug/server_fixture"
    rows = [json.loads(line) for line in (server / "decisions.jsonl").read_text().splitlines()]
    block_path = server / "blocks/d_1_000000.npz"
    block = schema.read_npz_block(block_path)
    wire = block["served_wire"][0].astype(np.float64) + np.float64(2. ** -40)
    assert not np.array_equal(wire, wire.astype(np.float32).astype(np.float64))
    expected = wire.copy()
    # Construct only the passive observer, without its startup/threads/provenance
    # or a policy/server. Exercise the actual response-copy implementation.
    observer = ServerObserver.__new__(ServerObserver)
    meta = json.loads((server / "meta.json").read_text())
    observer.manifest = dict(meta, block_controls=5, normalized_state_dim=8,
                             cost_weights=dict(full=.152, policy=.848))
    observer.runtime = NS(r4=False, debug_library_shas={"current": "fixture"})
    observer.config, observer.campaign, observer._anchors = {}, root.name, {}
    session = NS(conn=0, step=1, last_vision_step=0, stage1_calls=1, method=None,
                 _dec=dict(vision=True, hit=True, served=block["served_chunk"][0], lib="current", extras={}))
    arrays = {}
    capture = NS(metric={}, errors=[], before_calls={id(session): 0}, dispatch={}, stage_ms={})
    observer._decision(capture, session, {"actions": wire}, dict(rows[0]), arrays, {})
    assert arrays["served_wire"].dtype == np.float64 and arrays["served_wire"].tobytes() == expected.tobytes()
    wire.fill(999.)
    block["served_wire"] = block["served_wire"].astype(np.float64)
    block["served_wire"][0] = arrays["served_wire"]
    schema.write_npz_block(block_path, block)
    client = arm / "debug/client" / rows[0]["episode_key"]
    controls_path = client / "controls_0000.npz"
    controls = schema.read_npz_block(controls_path)
    mask = controls["decision_seq"] == 0
    controls["action"][mask] = expected
    schema.write_npz_block(controls_path, controls)
    receipt_path = arm / "debug/receipts" / client.name / "complete.json"
    receipt = json.loads(receipt_path.read_text())
    from exp.offline_search.debug.transport.receiver import digest
    for info in receipt["files"]:
        if info["name"] == controls_path.name:
            info.update(bytes=controls_path.stat().st_size, sha256=digest(controls_path))
    receipt_path.write_text(json.dumps(receipt))
    audit = validate.validate_arm(reader.open_arm(root, "synthetic"), 1, False)
    assert audit["status"] == "PASS", audit
    # Reintroducing the reviewed float32 conversion must reject those controls.
    block["served_wire"] = block["served_wire"].astype(np.float32)
    schema.write_npz_block(block_path, block)
    audit = validate.validate_arm(reader.open_arm(root, "synthetic"), 1, False)
    assert any(row["code"] == "issued_action" for row in audit["capture_missing"])
