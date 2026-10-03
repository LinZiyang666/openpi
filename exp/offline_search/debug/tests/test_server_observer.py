import json
import hashlib
import os
import signal
import subprocess
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from exp.offline_search.closed_loop import plugin
from exp.offline_search.debug import schema
from exp.offline_search.debug.server import ServerObserver
from exp.offline_search.debug.server.observer import parse_config, UnavailableObserver
from exp.offline_search.debug.server.diagnostics import metric_tap
from exp.offline_search.debug.server.dispatch import install_dispatch_tap
from exp.offline_search.debug.tests.test_server_support import no_git, server_json, server_rows


def setup(directory, method=None, oracle=False):
    rt = types.SimpleNamespace(model="generic", suite="fake", H=6, method=method, oracle=oracle,
        tag="observer", dims=types.SimpleNamespace(RAW_STATE_DIM=3, ACT_FULL_DIMS=4),
        tables={"current": np.zeros((3, 6, 4), np.float32)}, method_name="Fixture",
        r4=False, gpu=None, policy_tail=False, opts=types.SimpleNamespace(os_method="fake:Fixture", kwargs={}, os_fit_artifact=""))
    rt.debug = ServerObserver(rt, directory, dict(campaign="contract", model_manifest=dict(
        H=6, action_dim=4, valid_action_dims=[0, 1], normalized_state_dim=3, state_dim=3, block_controls=2),
        cost_weights=dict(full=.2, policy=.8)))
    action = np.arange(24, dtype=np.float32).reshape(6, 4)
    session = types.SimpleNamespace(rt=rt, conn=0, method=method, step=1, last_vision_step=0, stage1_calls=0,
        ep_meta=dict(uid="arm:eval:19:20", attempt=1, task_id=19, init=20), t_obs=1,
        _dec=dict(step=0, served=action, policy=action + 100, topk=np.array([0, 1]), scores=np.array([1., .5]),
                  lib="current", conf=1., hit=False, extras={"d1": .3, "w_eff": 2., **{"extra_"+str(i):i for i in range(100)}},
                  q_us=10., t_s0=1000000, t_s1=1000000),
        b_rs=types.SimpleNamespace(n=1, a=np.array([[1., 2., 3.]], np.float32)),
        b_v0=types.SimpleNamespace(n=1, a=np.array([[4., 5.]], np.float32)),
        b_v1=types.SimpleNamespace(n=1, a=np.array([[6., 7.]], np.float32)))
    session.set_obs = lambda obs: None
    session.after_infer = lambda *args: setattr(session, "_dec", None)
    epkey = schema.episode_key(session.ep_meta["uid"], 1)
    envelope = dict(v=1, task_uid=session.ep_meta["uid"], attempt=1, dispatch_gen=8, episode_key=epkey,
                    decision_seq=0, decision_id=schema.decision_id(epkey, 8, 0), t_client_send=1.)
    obs = {"__debug__": envelope, "observation/state": np.array([1 / 3, 2., 3.], np.float64),
           "observation/image": np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
           "observation/wrist_image": np.arange(60, dtype=np.uint8).reshape(4, 5, 3), "prompt": "exact ✓"}
    return rt, session, obs


def test_wire_copy_before_transform_and_before_cleanup(tmp_path):
    method = types.SimpleNamespace(debug_record=lambda: dict(array=np.array([2 ** 63 + 2], np.uint64),
                                                             scalars={"full_"+str(i): i for i in range(90)}))
    rt, session, obs = setup(tmp_path, method)
    wire_image, wire_state = obs["observation/image"].copy(), obs["observation/state"].copy()
    returned = np.ones((6, 2), np.float32)
    def infer(received):
        assert "__debug__" not in received
        received["observation/image"].fill(0)
        received["observation/state"].fill(9)
        return {"actions": returned}
    conn = plugin._ConnPolicy(types.SimpleNamespace(infer=infer), [session], "fixture")
    result = conn.infer(obs)
    assert result["__debug__"]["status"] == "available"
    returned.fill(123)
    rt.debug.close()
    record, = server_rows(tmp_path)
    data = schema.read_npz_block(tmp_path / "blocks" / record["blk"])
    np.testing.assert_array_equal(data["img_third"][0], wire_image)
    assert data["state_wire"][0].tobytes() == wire_state.tobytes()
    np.testing.assert_array_equal(data["served_wire"][0], 1.)
    np.testing.assert_array_equal(data["served_chunk"][0], np.arange(24, dtype=np.float32).reshape(6, 4) + 100)
    assert session._dec is None
    assert record["extras"]["extra_99"] == 99 and record["diag"]["scalars"]["full_89"] == 89
    assert data[record["diag"]["array"]["array"]][0, 0] == 2 ** 63 + 2
    assert data["prompts"][0] == "exact ✓"


def test_reserved_envelopes_always_removed_oracle_explicit(tmp_path, caplog):
    for allowed in (False, True):
        rt, session, obs = setup(tmp_path / str(allowed), oracle=allowed)
        received = []
        session.method = types.SimpleNamespace(debug_record=lambda: {})
        obs["__oracle__"] = {"inside_window": True, "distance": .02}
        conn = plugin._ConnPolicy(types.SimpleNamespace(infer=lambda o: (received.append(o), {"actions": np.ones((6, 2), np.float32)})[1]), [session], "fixture")
        result = conn.infer(obs)
        view = plugin.OnlineQueryView(session, 0, 19, None)
        assert view.oracle == (obs["__oracle__"] if allowed else None)
        assert "__oracle__" not in received[0] and "__debug__" not in received[0]
        assert "__oracle__" in obs  # reserved-field pop does not mutate caller
        assert result["__debug__"]["status"] == ("available" if allowed else "error")
        rt.debug.close()
        record, = server_rows(tmp_path / str(allowed))
        assert record["oracle_status"]["status"] == ("available" if allowed else "error")
    assert "--os-oracle is absent" in caplog.text
    rt, session, obs = setup(tmp_path / "off")
    rt.debug.close(); rt.debug = None
    captured = []
    conn = plugin._ConnPolicy(types.SimpleNamespace(infer=lambda o: (captured.append(o), {"actions": []})[1]), [session], "fixture")
    assert "__debug__" not in conn.infer(obs)
    assert "__debug__" not in captured[0]


def test_diagnostic_error_never_changes_response(tmp_path):
    def fail():
        raise ValueError("intentional diagnostic error")
    rt, session, obs = setup(tmp_path, types.SimpleNamespace(debug_record=fail))
    action = np.arange(12, dtype=np.float32).reshape(6, 2)
    conn = plugin._ConnPolicy(types.SimpleNamespace(infer=lambda obs: {"actions": action}), [session], "fixture")
    result = conn.infer(obs)
    assert result["actions"] is action and result["__debug__"]["status"] == "error"
    rt.debug.close()
    row, = server_rows(tmp_path)
    assert row["diag_status"]["status"] == "error"
    assert "intentional diagnostic error" in row["diag_status"]["reason"]


def test_unavailable_startup_and_closed_queue_only_change_telemetry(tmp_path):
    rt, session, obs = setup(tmp_path)
    rt.debug.close()
    actions = np.ones((6, 2), np.float32)
    conn = plugin._ConnPolicy(types.SimpleNamespace(infer=lambda obs: {"actions": actions}), [session], "fixture")
    result = conn.infer(obs)
    assert result["actions"] is actions and result["__debug__"]["status"] == "error"
    rt.debug = UnavailableObserver(rt, OSError("startup disk unavailable"))
    result = conn.infer(obs)
    assert result["actions"] is actions and result["__debug__"]["status"] == "error"


def test_serving_exception_captured_without_cleanup_loss(tmp_path):
    rt, session, obs = setup(tmp_path)
    def fail(obs):
        raise RuntimeError("serving failure")
    conn = plugin._ConnPolicy(types.SimpleNamespace(infer=fail), [session], "fixture")
    with pytest.raises(RuntimeError, match="serving failure"):
        conn.infer(obs)
    rt.debug.close()
    row, = server_rows(tmp_path)
    assert not row["ok"] and "serving failure" in row["serving_error"]
    assert row["served_wire_status"]["status"] == "error" and session._dec is None


def test_config_cli_and_profile_restored(tmp_path):
    args = ["--os-method", "fake:Class", "--os-cell", "pi05_spatial_cache", "--os-log-dir", str(tmp_path)]
    opts, rest = plugin.parse_cli(args + ["--os-debug-dir", str(tmp_path / "debug"), "--os-debug-config", '{"campaign":"c"}', "--os-oracle", "--port", "23311"])
    assert rest == ["--port", "23311"] and opts.debug_config == {"campaign": "c"} and opts.os_oracle
    for value in ('[]', '{"queue_bytes":0}', '{"rawkeys_rate":0.5}'):
        with pytest.raises(ValueError):
            parse_config(value)
    previous = sys.getprofile()
    with pytest.raises(RuntimeError):
        with metric_tap(types.SimpleNamespace(method=None), {}):
            raise RuntimeError("failure")
    assert sys.getprofile() is previous


def test_startup_provenance_uses_repository_directory(tmp_path, monkeypatch):
    commands = []
    expected_repo = Path(plugin.__file__).resolve().parents[3]
    def check(command, cwd, stderr):
        commands.append(command)
        assert Path(cwd) == expected_repo
        return b"a" * 40 if command[1] == "rev-parse" else b"dirty diff\n"
    monkeypatch.setattr(subprocess, "check_output", check)
    rt, _, _ = setup(tmp_path)
    rt.debug.close()
    meta = server_json(tmp_path, "meta")
    assert meta["git_head"] == "a" * 40
    assert meta["dirty_diff_sha256"] == hashlib.sha256(b"dirty diff\n").hexdigest()
    assert commands[1][-1] == ":(exclude)tests/review_tests"


@pytest.mark.parametrize("model", ["pi05", "groot"])
def test_actual_dispatch_counts_do_not_add_stage_or_rng_calls(tmp_path, model):
    calls = []
    rng = np.random.default_rng(90)
    def stage(name):
        def run(value):
            calls.append(name)
            return value + np.float32(rng.random())
        return run
    owner = types.SimpleNamespace(_orchestrator=object())
    if model == "pi05":
        owner._stage1_fn, owner._stage2_fn, owner._stage3_fn = stage("s1"), stage("s2"), stage("s3")
        forward = lambda v: owner._stage3_fn(owner._stage2_fn(owner._stage1_fn(v)))
    else:
        owner._runner = types.SimpleNamespace(run_stage1=stage("s1"), run_stage2_llm=stage("s2"), run_stage3=stage("s3"))
        owner._runner.run_stage2 = lambda v: owner._runner.run_stage3(owner._runner.run_stage2_llm(v))
        forward = lambda v: owner._runner.run_stage2(owner._runner.run_stage1(v))
    before = forward(np.zeros(4, np.float32))
    rng = np.random.default_rng(90)
    calls.clear()
    cap = types.SimpleNamespace(dispatch={}, stage_ms={})
    session = types.SimpleNamespace(rt=types.SimpleNamespace(model=model), _debug_capture=cap)
    result = install_dispatch_tap(owner, session)
    after = forward(np.zeros(4, np.float32))
    assert before.tobytes() == after.tobytes() and calls == ["s1", "s2", "s3"]
    assert result["status"] == "available"
    assert cap.dispatch == (dict(stage1=1, stage2=1, stage3=1) if model == "pi05" else
                            dict(stage1=1, stage23=1, stage2=1, stage3=1))


def test_sigterm_drains_background_writer(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", '''
import signal, sys
from pathlib import Path
import numpy as np
from types import SimpleNamespace as NS
from exp.offline_search.debug.server import ServerObserver
import exp.offline_search.debug.server.manifest as manifest
manifest.git_provenance = lambda: {}
r = NS(model="generic", suite="fake", H=2, method=None, tables={}, opts=NS(os_method="fake", kwargs={}, os_fit_artifact=""))
o = ServerObserver(r, sys.argv[1], {"campaign":"signal"})
for i in range(31):
    o.writer.put({"decision_id":"signal:0:"+str(i)}, {"state_wire":np.array([i], np.float64)})
print("ready", flush=True)
signal.pause()
''', str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        child.send_signal(signal.SIGTERM)
        _, error = child.communicate(timeout=10)
        assert child.returncode == 143, error
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
    stats = server_json(tmp_path, "writer_stats")
    assert stats["written"] == stats["accepted"] == 31 and stats["drained"]
