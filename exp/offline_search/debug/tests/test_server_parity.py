"""Recorded keys + actual serving methods; no model, socket or simulator.

The optional artifact tests exercise the deployed frozen R7 fits when present.
All writes belong to the test scratch directory, never to the fit/store tree.
"""
import hashlib
import itertools
import json
import pickle
import random
import threading
import time
import types
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from exp.offline_search.closed_loop import plugin
from exp.offline_search.debug import schema
from exp.offline_search.debug.server import ServerObserver
from exp.offline_search.debug.server.diagnostics import method_chain
from exp.offline_search.harness import api, dims, store
from exp.offline_search.debug.tests.test_server_support import no_git, server_json, server_rows

ROOT = Path("/home/weiland/trace_runs/offline_search_store")
FITS = Path("/home/weiland/trace_runs/os_closed_loop/r07_main/fits")


def fitted(family, model="pi05"):
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import sources, load_base, FitUnpickler
    if family == "A":
        return load_base(sources()[model + "_spatial_50"])
    if family == "Policy":
        from exp.offline_search.rounds.r04.k4_eval.seeded_inference import SeededInference
        method = SeededInference()
        method.prof = api.NULL_PROFILER
        lib = store.LibraryView(ROOT, model + "_spatial", "current")
        method.fit(lib, api.Context(root=ROOT, cell=model + "_spatial_cache", seed=0, scratch=Path("/tmp/r8_S1")))
        return method, dict(spec="exp.offline_search.rounds.r04.k4_eval.seeded_inference:SeededInference", kwargs={})
    names = dict(CU="r7_"+model+"_spatial_50_CU30.pkl", CT="r7_"+model+"_spatial_50_CT30.pkl",
                 SF="r7_"+model+"_spatial_50_SF1.pkl", SW="r7_sw_pi05_sp_50.pkl")
    with (FITS / names[family]).open("rb") as stream:
        blob = FitUnpickler(stream).load()
    return blob["method"], blob


class FakeOrchestrator:
    def __init__(self, session):
        self.session = session
        self._twins = None
        self.on_episode_start()

    def on_episode_start(self, **kwargs):
        self._real = types.SimpleNamespace(state=types.SimpleNamespace(step_counter=0, state_history=[]))

    def _feed_verdict_to_gate(self, *args, **kwargs):
        pass

    def broadcast_action(self, chunk):
        self.session.on_executed(chunk)

    def clear(self):
        pass


class FakePolicy:
    def __init__(self, session, queries):
        self.session, self.queries = session, queries
        self.orch = FakeOrchestrator(session)
        self.rng = np.random.default_rng(581)
        self.encodes = self.policy_calls = 0

    def stage1(self, obs):
        self.rng.random()  # fake encoder consumes a live RNG draw
        self.encodes += 1
        if self.session.rt.request_cameras:
            self.session._camera_work["looks"] = 1

    def _osp_prepare_blind(self, obs):
        state = np.array(self.queries.rs[obs["_row"]], copy=True)
        return state, state

    def _osp_blind_output(self, action, state):
        return {"actions": np.asarray(action[:, :7] + np.float32(state[0] * .1), np.float32)}

    def on_episode_start(self, **kwargs):
        self.orch.on_episode_start()

    def on_episode_end(self, **kwargs):
        pass

    def infer(self, obs):
        assert "__debug__" not in obs and "__oracle__" not in obs
        self.stage1(obs)
        row, session = obs["_row"], self.session
        state = np.array(self.queries.rs[row], copy=True)
        k0, k1 = np.array(self.queries.key_v0[row]), np.array(self.queries.key_v1[row])
        if getattr(session, "_camera_mode", "full") == "wrist_only":
            k0.fill(0)
        ctx = types.SimpleNamespace(task_key=obs["prompt"], current_step=self.orch._real.state.step_counter,
                                    query_keys=dict(vision_0=k0, vision_1=k1, robot_state=state), checkpoint_id="CP1")
        session.on_search(ctx)
        hit = session._dec.get("hit", True)
        if hit:
            action = session._dec["served"]
        else:
            self.policy_calls += 1
            action = self.rng.standard_normal((session.rt.H, 32)).astype(np.float32)
            if session.rt.request_cameras and session._camera_mode == "wrist_only":
                session._camera_work["completions"] = 1
        self.orch._real.state.step_counter += 1
        self.orch._real.state.state_history.append(state)
        self.orch.broadcast_action(action)
        return {**self._osp_blind_output(action, state),
                "diag": session.wire_diag(), "hit_type": "FULL_HIT" if hit else "MISS"}


def runtime(method, blob, directory, family, model="pi05"):
    rt = object.__new__(plugin.PluginRuntime)
    rt.api, rt.dims, rt.store = api, dims, store
    rt.method, rt.method_name = method, method.name
    rt.model, rt.suite, rt.cell = model, "spatial", model + "_spatial_cache"
    rt.H = dims.HORIZON[model]
    rt.lib = store.LibraryView(ROOT, model + "_spatial", "current")
    rt.lib_key = model + "_spatial"
    rt.task_map = {str(k): int(v) for k, v in rt.lib.meta["task_map"].items()}
    rt.cur_action = np.asarray(rt.lib.action)
    rt.tables = {"current": rt.cur_action}
    rt.lib_sizes = {"current": rt.lib.L}
    rt.ids = list(rt.lib.ids)
    rt.id_to_row = {key: i for i, key in enumerate(rt.ids)}
    rt.ep_index = {}
    rt.gpu, rt.ctrl, rt.debug = None, None, None
    rt.blind = family != "Policy"
    rt.policy_tail = family in ("CU", "CT")
    rt.policy_tail_blocks = 1
    rt.r4 = True
    rt.request_cameras = family == "SW"
    rt.oracle = rt.randomized = rt.shadow_native = rt.native_mode = False
    rt.judge = plugin.JudgeSpec.parse("periodic:1" if family == "Policy" else "guard_only") if family in ("CU", "CT", "Policy") else None
    rt.decision_count = 0
    rt.decision_lock = threading.RLock()
    rt._conn_ids = itertools.count()
    rt.sessions = weakref.WeakSet()
    rt.opts = types.SimpleNamespace(os_seed=0, os_tokens="off", os_log_inputs=False, os_fit_artifact="",
                                    os_method=blob.get("spec", "unknown"), kwargs=blob.get("kwargs", {}))
    rt.tag = "parity"
    rt._wlock = threading.Lock()
    rt.log_dir = directory
    directory.mkdir(parents=True, exist_ok=True)
    rt.dec_path = directory / "legacy.jsonl"
    rt.inputs_dir = directory / "inputs"
    return rt


def digest(session, inner):
    dynamic = {}
    for i, method in enumerate(method_chain(session.method)):
        for name in ("_anchor", "_policy_gate_anchor", "_follow_plan", "last_blind_extras", "_last_log", "_tilt_log",
                     "_deviation_latched", "_look_due_step", "_anchor_index", "_last_cooldown_anchor", "_last_extra_control",
                     "parameter", "next_camera_mode", "_camera_mode", "_episode_identity", "_grip_memo"):
            if hasattr(method, name):
                dynamic[str(i) + name] = getattr(method, name)
        tracker = getattr(method, "tracker", None)
        if tracker is not None:
            dynamic[str(i) + "tracker"] = vars(tracker)
    dynamic.update(hits=session.hits, vision=session.has_vision, step=session.step, blind_age=session.blind_age,
                   policy_tail=session._policy_tail, rng=inner.rng.bit_generator.state,
                   encodes=inner.encodes, policy_calls=inner.policy_calls)
    if session.ep is not None:
        for name in ("b_v0", "b_v1", "b_rs", "b_raw", "b_aex"):
            buf = getattr(session, name)
            dynamic[name] = buf.a[:buf.n]
    return hashlib.sha256(pickle.dumps(dynamic, protocol=4)).hexdigest()


def replay(family, directory, debug, model="pi05"):
    import torch
    from exp.offline_search.closed_loop.selftest import pick_episodes
    from exp.offline_search.rounds.r04.k4_eval import seeded_inference
    seeded_inference._seeded = None
    random.seed(97); np.random.seed(97); torch.manual_seed(97)
    method, blob = fitted(family) if model == "pi05" else fitted(family, model)
    rt = runtime(method, blob, directory, family) if model == "pi05" else runtime(method, blob, directory, family, model)
    qc = store.QueryCell(ROOT, model + "_spatial_cache")
    image_shape = rt.lib.meta.get("img_shape") or ([256, 256, 3] if model == "groot" else [224, 224, 3])
    session = plugin.PluginSession(rt, None, None, None, "fake")
    if debug:
        rt.debug = ServerObserver(rt, directory / "debug", {"campaign": "tape", "writer_queue_bytes": 8 * 1024 ** 2})
    inner = FakePolicy(session, qc)
    conn = plugin._ConnPolicy(inner, [session], "default")
    responses, states, rngs, times = [], [], [], []
    wall_clock = time.perf_counter
    try:
        # Wire responses contain live timing fields. A frozen clock makes their
        # bytes comparable, including those fields, without deleting diagnostics.
        with patch.object(plugin.time, "perf_counter_ns", return_value=1000000), patch.object(plugin.time, "perf_counter", return_value=1.), patch.object(plugin.time, "time", return_value=1.):
            for ei in pick_episodes(qc, 20):
                ep = qc.episodes[ei]
                uid = "tape:eval:{}:{}".format(ep["task_id"], ep["init"])
                key = schema.episode_key(uid, 1)
                conn.on_episode_start(task=ep["task"], extra_metadata=dict(task_uid=uid, attempt=1,
                                     task_id=ep["task_id"], orig_init_state_idx=ep["init"]))
                for seq, row in enumerate(range(ep["start"], ep["end"])):
                    envelope = dict(v=1, task_uid=uid, attempt=1, episode_key=key, dispatch_gen=4,
                                    decision_seq=seq, decision_id=schema.decision_id(key, 4, seq), t_client_send=1.)
                    obs = {"prompt": ep["task"], "observation/state": np.asarray(qc.raw_state[row], np.float64),
                           "observation/image": np.full(tuple(image_shape), seq % 256, np.uint8),
                           "observation/wrist_image": np.full(tuple(image_shape), (seq + 1) % 256, np.uint8),
                           "__extra__": dict(decision_id=seq, executed_steps=5), "__debug__": envelope, "_row": row}
                    started = wall_clock()
                    out = conn.infer(obs)
                    times.append((wall_clock() - started) * 1000)
                    if debug:
                        assert out.pop("__debug__")["status"] == "available"
                    responses.append(pickle.dumps(out, protocol=4))
                    states.append(digest(session, inner))
                    rngs.append(hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy()), protocol=4)).hexdigest())
                    if len(responses) >= 220:
                        break
                conn.on_episode_end(success=False)
                if len(responses) >= 220:
                    break
    finally:
        if rt.debug:
            rt.debug.close()
    rows = [json.loads(line) for line in rt.dec_path.read_text().splitlines() if json.loads(line).get("ev") == "dec"]
    return responses, states, rngs, rows, times


@pytest.mark.parametrize("family", ["A", "CU", "CT", "SF", "SW", "Policy"])
def test_real_method_request_tape_parity(tmp_path, family):
    if not ROOT.is_dir() or not FITS.is_dir():
        pytest.skip("deployed R7 fits/store unavailable; portable contract tests still run")
    off = replay(family, tmp_path / "off", False)
    on = replay(family, tmp_path / "on", True)
    assert len(off[0]) >= 200
    assert off[:4] == on[:4]  # response bytes, method/history digests, RNG state, legacy log bytes
    stats = server_json(tmp_path / "on/debug", "writer_stats")
    assert stats["written"] == len(on[0]) and stats["drained"] and not stats["errors"]
    records = server_rows(tmp_path / "on/debug")
    assert sum(r["policy_calls"] for r in records) == sum(not r["hit"] and r["vision"] for r in records)
    if family != "Policy":
        assert any(not row["vision"] for row in records)
        assert all(row["blind_age_controls"] >= 5 for row in records if not row["vision"])
        assert any(r["keys_status"]["wrist"]["status"] == "available" for r in records if r["vision"])
    if family == "SW":
        assert any(r["camera_mode"] == "wrist_only" for r in records)
    if family in ("CU", "CT"):
        assert any(0 < row.get("p_effective", 0) < 1 for row in records)
        assert any(row["policy_calls"] for row in records)
        assert any(row["src"] == "policy_tail" for row in records)
    report = dict(family=family, requests=len(on[0]), response_bytes_identical=True, method_digests_identical=True,
                  rng_states_identical=True, legacy_logs_identical=True,
                  off_median_ms=float(np.median(off[4])), on_median_ms=float(np.median(on[4])),
                  median_overhead_ms=float(np.median(on[4]) - np.median(off[4])))
    (tmp_path / "parity_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


@pytest.mark.parametrize("family", ["A", "CU", "CT", "SF", "Policy"])
def test_groot_method_request_tape_parity(tmp_path, family):
    if not (ROOT / "library/groot_spatial/current").is_dir() or not FITS.is_dir():
        pytest.skip("deployed GR00T fits/store unavailable")
    off = replay(family, tmp_path / "off", False, model="groot")
    on = replay(family, tmp_path / "on", True, model="groot")
    assert len(off[0]) >= 200 and off[:4] == on[:4]
    stats = server_json(tmp_path / "on/debug", "writer_stats")
    assert stats["written"] == len(on[0]) and stats["drained"] and not stats["errors"]
    report = dict(model="groot", family=family, requests=len(on[0]), response_bytes_identical=True,
                  method_digests_identical=True, rng_states_identical=True, legacy_logs_identical=True,
                  off_median_ms=float(np.median(off[4])), on_median_ms=float(np.median(on[4])),
                  median_overhead_ms=float(np.median(on[4]) - np.median(off[4])))
    (tmp_path / "parity_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
