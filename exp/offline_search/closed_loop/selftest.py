"""CPU integration test of the plugin on recorded store episodes (no GPU, no simulator, no websocket).

Drives the REAL per-connection stack -- patched build_per_connection_components, CacheOrchestrator, PluginStrategy /
PluginJudge / PluginStorage, native shadow search on the real in-memory backend, _ConnPolicy wrapper -- with a key
builder that returns the store's recorded keys for each decision, then checks:

  1. plugin top-1 == recorded online top-1 (rec_top1) and native-shadow top-1 == rec_top1   (B0 / ProbeB0 only)
  2. online outputs (topk, scores, confidence, synthesized action, extras incl. ProbeB0 / ProbeHist QueryView digests)
     == the offline harness (run_jobs_inprocess) on the same store episodes, bit for bit, on every episode whose
     executed chunks equal the recorded ones (all of them when (1) holds; history-driven methods compare on the
     episodes' first decision otherwise)
  3. served chunk == executed chunk (exec_ok) on every decision; with a library pick, == the recorded a_exec

    taskset -c 34-37,78-81 .venv/bin/python -m exp.offline_search.closed_loop.selftest --cell pi05_spatial_cache \
        --yaml <arm yaml> --method exp.offline_search.closed_loop.probe:ProbeB0 --episodes 20 --out /tmp/osplug_selftest

Mixed HIT/MISS mode (``--judge <spec>`` = the server's --os-judge, plus --judge-cap / --judge-step0 / --judge-burst):
the fake policy answers a MISS with the replay cell's recorded full-inference chunk a_inf[row] (what the policy produced
at that very state in the trace run), so the executed history after a MISS is a real policy chunk. Extra checks:

  4. the client-side verdict (CheckResult.hit_type) == the logged hit flag on every decision; exec_ok is True on HITs
     and None on MISSes; a_exec == a_inf[row] on MISS rows and == the served payload on HIT rows
  5. the verdict rule (verify_logs.check_verdicts) re-derived from the logged conf / tau / run / step / os_force_miss
     equals every logged verdict; run == trailing HITs
  6. offline replay of the logged inputs through the harness mini store (verify_logs machinery; MISS rows become
     policy rows) == online bit for bit -> the QueryView after a MISS (prev_hit False, prev_a_exec = policy chunk,
     hist_hit) is what the offline harness constructs
  7. with ``--replay-cell <m>_<s>_inf --judge periodic:1`` (every decision a MISS, executed = the inf cell's own
     a_exec) the online outputs are ALSO compared with the offline harness run on the real inf cell: the QueryView
     after a MISS equals the offline inf-cell view

    ... --judge threshold:0.985 --judge-cap 4 --method exp.offline_search.closed_loop.probe:ProbeForce
    ... --judge periodic:1 --replay-cell pi05_spatial_inf --method exp.offline_search.closed_loop.probe:ProbeB0
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402


class FakeKB:
    """Key builder returning the store's recorded query keys for the current row."""

    def __init__(self, qc):
        self.qc = qc
        self.row = None
        self._cache = {}

    def collect(self, checkpoint_id, **kw):
        pass

    def build(self, checkpoint_id):
        import torch

        r = self.row
        return {"vision_0": torch.from_numpy(np.array(self.qc.key_v0[r])),
                "vision_1": torch.from_numpy(np.array(self.qc.key_v1[r])),
                "robot_state": torch.from_numpy(np.array(self.qc.rs[r]))}

    def _slice(self):
        import torch

        j = int(self.qc.tok_index[self.row])
        if j < 0:
            raise LookupError("row not in the tok subsample")
        return {"vision_0": torch.from_numpy(np.array(self.qc.tok("v0")[j])),
                "vision_1": torch.from_numpy(np.array(self.qc.tok("v1")[j]))}

    @property
    def cached_data(self):
        return self._cache

    def clear(self):
        pass


class FakePolicy:
    """Stand-in for the interceptor: CP1 check -> FULL_HIT payload -> broadcast_action, like the real FULL_HIT path.
    Mixed mode (policy is not None): a MISS executes the fake policy chunk policy[row] (stage 2/3 stand-in) and
    broadcasts it, like the real MISS path (Interceptor -> broadcast_action -> PluginStrategy.record_action)."""

    def __init__(self, orch, kb, policy=None):
        self.orch, self.kb, self.policy = orch, kb, policy

    def infer(self, obs):
        import torch

        from openpi.cache.components.judge import HitType
        from openpi.cache.types import CheckpointID

        r = int(obs["_row"])
        self.kb.row = r
        res = self.orch.check(CheckpointID.CP1, stage1=None)
        if res.hit_type == HitType.FULL_HIT:
            chunk = res.payload.action_chunk
        elif res.hit_type == HitType.MISS and self.policy is not None:
            chunk = torch.from_numpy(np.array(self.policy[r], dtype=np.float32))
        else:
            raise RuntimeError(f"expected FULL_HIT, got {res.hit_type}")
        self.orch.broadcast_action(chunk)
        self.orch.clear()
        return {"winner": res.entry_id, "hit_type": res.hit_type.name, "actions": np.asarray(chunk),
                "diag": (res.factor_outputs or {}).get("osplug")}

    def on_task_begin(self):
        self.orch.on_task_begin()

    def on_episode_start(self, experiment="", task="", episode_id=-1, episode_name="", extra_metadata=None):
        self.orch.on_episode_start(task_key=task, episode_id=str(episode_id), extra_metadata=extra_metadata)

    def on_episode_end(self, success):
        self.orch.on_episode_end()

    def on_task_end(self):
        self.orch.on_task_end()


def pick_episodes(qc, n):
    """Spread over tasks; tok-subsample inits first."""
    eps = qc.episodes
    by_task = {}
    for i, e in enumerate(eps):
        by_task.setdefault(e["task_id"], []).append(i)
    order = []
    j = 0
    while len(order) < min(n, len(eps)):
        for t in sorted(by_task):
            lst = sorted(by_task[t], key=lambda i: (eps[i]["init"] not in (0, 10, 20, 30, 40), eps[i]["init"]))
            if j < len(lst) and len(order) < n:
                order.append(lst[j])
        j += 1
    return sorted(order)


def compare_offline(off, sel, qc, online, ep_ok):
    """Decision-wise equality of the online npz vs an offline run over jobs (sel, rows) on qc; ep_ok[uid] says whether
    the episode's online history equals the one offline saw (else only step 0 is compared)."""
    i = 0
    eq = {"topk": 0, "scores": 0, "conf": 0, "synth": 0, "extras": 0}
    compared = 0
    first_diff = None
    for ei in sel:
        e = qc.episodes[ei]
        z = online[e["uid"]]
        hist_ok = ep_ok[e["uid"]]
        for s in range(e["end"] - e["start"]):
            j = i + s
            if not hist_ok and s > 0:
                continue        # the online history diverged from the recorded one: offline sees other chunks
            compared += 1
            k = off["topk"][j].size
            same_topk = np.array_equal(z["topk"][s, :k], off["topk"][j]) and z["lib"][s] == off["lib"][j]
            same_sc = np.array_equal(z["scores"][s, :k], off["scores"][j])
            same_conf = z["conf"][s] == off["conf"][j]
            osyn = off["synth"][j]
            same_syn = (osyn is None and not z["used_synth"][s]) or (
                osyn is not None and np.array_equal(z["synth"][s, :5, :7], osyn))
            exo = off["extras"][j] or {}
            same_ex = all(f"x_{kk}" in z.files and np.float64(z[f"x_{kk}"][s]) == np.float64(v) for kk, v in exo.items()
                          if np.asarray(v).size == 1)
            for key, v in (("topk", same_topk), ("scores", same_sc), ("conf", same_conf), ("synth", same_syn),
                           ("extras", same_ex)):
                eq[key] += int(bool(v))
            if first_diff is None and not (same_topk and same_sc and same_conf and same_syn and same_ex):
                bad = {kk: (float(z[f"x_{kk}"][s]) if f"x_{kk}" in z.files else None, float(v)) for kk, v in exo.items()
                       if np.asarray(v).size == 1 and not (f"x_{kk}" in z.files and
                                                            np.float64(z[f"x_{kk}"][s]) == np.float64(v))}
                first_diff = {"uid": e["uid"], "step": s, "topk": bool(same_topk), "scores": bool(same_sc),
                              "conf": bool(same_conf), "synth": bool(same_syn), "extras_diff": bad}
        i += e["end"] - e["start"]
    return compared, {k: v / max(compared, 1) for k, v in eq.items()}, first_diff


def blind_main(a):
    """Two real orchestrators, two reset episodes each, poison stage 1 on blind rows."""
    from . import plugin, verify_logs
    from exp.offline_search.harness import store
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args = ["--os-method", a.method, "--os-kwargs", a.kwargs, "--os-cell", a.cell, "--os-root", a.root,
            "--os-log-dir", str(out), "--os-tag", "blindtest", "--os-log-inputs", "--os-blind"]
    if a.judge:
        args += ["--os-judge", a.judge, "--os-judge-cap", str(a.judge_cap),
                 "--os-judge-step0", a.judge_step0, "--os-judge-burst", str(a.judge_burst)]
    if a.fit_artifact:
        args += ["--os-fit-artifact", a.fit_artifact]
    opts, _ = plugin.parse_cli(args)
    rt = plugin.install(opts, store.parse_cell(a.cell)[0])
    cfg = cc.load_cache_config(a.yaml)
    shared = cc.build_shared_storage(cfg)
    qc = store.QueryCell(a.root, a.replay_cell or a.cell)

    class BlindFake(FakePolicy):
        def __init__(self, *args):
            super().__init__(*args)
            self.calls = self.broadcasts = 0
            broadcast = self.orch.broadcast_action
            def counted(chunk):
                self.broadcasts += 1
                return broadcast(chunk)
            self.orch.broadcast_action = counted

        def stage1(self, obs):
            if obs.get("_expect_blind"):
                raise AssertionError("stage 1 called on an expected blind decision")
            self.calls += 1

        def infer(self, obs):
            self.stage1(obs)
            return super().infer(obs)

        def _osp_prepare_blind(self, obs):
            return np.array(qc.rs[int(obs["_row"])], copy=True), None

        def _osp_blind_output(self, action, state):
            return {"actions": action.copy()}

    def factory(_base, bundle_id="default"):
        comps = cc.build_per_connection_components(cfg, shared, quiet=True)
        kb = FakeKB(qc)
        for s in plugin._TLS.new_sessions:
            s.kb = kb
        orch = CacheOrchestrator(storage=comps["storage"], key_builder=kb, gates=comps["gates"],
                                 judges=comps["judges"], search_strategies=comps["search_strategies"],
                                 timer=comps["timer"], write_policy=comps.get("write_policy"),
                                 offline_writers=comps.get("offline_writers", ()), library_stats=comps.get("library_stats"))
        return BlindFake(orch, kb, np.asarray(qc.a_inf))

    conns = [plugin._wrap_factory(factory)(None, str(i)) for i in range(2)]
    eps = pick_episodes(qc, 2)
    served, expected, snapshots = [], [], []
    rejected_preflight = output_fallbacks = partial_looks = 0
    is_probe = a.method.endswith(":ProbeBlind")
    for restart in range(2):
        for i, conn in enumerate(conns):
            e = qc.episodes[eps[i]]
            conn.on_episode_start(task=e["task"], episode_id=e["init"],
                                  extra_metadata={"task_uid": f"blind-c{i}-e{restart}", "task_id": e["task_id"],
                                                  "orig_init_state_idx": e["init"]})
        for step in range(12):
            for i, conn in enumerate(conns):
                e = qc.episodes[eps[i]]
                s = conn._osp_sessions[0]
                due = rt.judge and rt.judge.mode == "periodic" and rt.decision_count % rt.judge.k == rt.judge.k - 1
                blind = bool(is_probe and step % 6 in (1, 2) and s.hits and s.hits[-1] and not due
                             and s.blind_age < s.method.budget
                             and not (rt.judge and (s.burst_left > 0 or
                                 (rt.judge.cap > 0 and rt.judge.mode not in ("always", "periodic")
                                  and plugin._trailing_hits(s.hits) >= rt.judge.cap))))
                obs = {"observation/state": np.asarray(qc.raw_state[e["start"] + step], np.float64),
                       "prompt": e["task"], "_row": e["start"] + step, "_expect_blind": blind,
                       "observation/image": np.zeros((2, 2, 3), np.uint8),
                       "observation/wrist_image": np.zeros((2, 2, 3), np.uint8),
                       "__extra__": {"decision_id": step, "executed_steps": 5}}
                if is_probe and step == 1 and blind:
                    from exp.offline_search.closed_loop.blind import BlindResult
                    adapter = conn._osp_adapter
                    before = (s.step, s.b_aex.n, len(s.has_vision), conn.orch._step_counter, rt.decision_count)
                    original_step = s.method.blind_step
                    s.method.blind_step = lambda bq: BlindResult(np.zeros((1, 32), np.float32),
                        np.array([0], np.int64), np.ones(1, np.float32), "current", {})
                    s.set_obs(obs)
                    s._decision_index = rt.decision_count
                    assert plugin._try_blind(s, adapter, obs) is None and s._look_reason == 8
                    rejected_preflight += 1
                    del s.method.blind_step  # restore class dispatch
                    assert before == (s.step, s.b_aex.n, len(s.has_vision), conn.orch._step_counter, rt.decision_count)
                    saved_method, _ = plugin.clone_method(s.method)
                    output = adapter.output
                    def fail_output(*args):
                        raise ValueError("intentional output-transform failure before commit")
                    adapter.output = fail_output
                    s.set_obs(obs)
                    assert plugin._try_blind(s, adapter, obs) is None and s._look_reason == 8
                    output_fallbacks += 1
                    adapter.output = output
                    s.method = saved_method
                    assert before == (s.step, s.b_aex.n, len(s.has_vision), conn.orch._step_counter, rt.decision_count)
                    partial = {**obs, "__extra__": {"decision_id": step, "executed_steps": 4}}
                    s.set_obs(partial)
                    assert plugin._try_blind(s, adapter, partial) is None and s._look_reason == 6
                    partial_looks += 1
                    assert before == (s.step, s.b_aex.n, len(s.has_vision), conn.orch._step_counter, rt.decision_count)
                result = conn.infer(obs)
                served.append(np.asarray(result["actions"]))
                expected.append(blind)
                assert s.step == step + 1 == s.b_aex.n == len(s.hits) == len(s.has_vision)
                assert conn.orch._step_counter == step + 1
                assert len(conn.orch._state_history) == step + 1 == len(conn.orch._action_history)
                assert np.array_equal(s.b_aex.a[step], result["actions"])
                assert np.array_equal(s.b_rs.a[step], qc.rs[e["start"] + step])
                if not s.has_vision[-1]:
                    assert np.isnan(s.b_v0.a[step]).all() and np.isnan(s.b_v1.a[step]).all()
                    assert result["__hit_meta__"]["searched"] is False
                    assert s.hits[-1] == 1
                    q = plugin.OnlineQueryView(s, step, s.ep.task_id, s.ep)
                    assert not q.has_tok and not q.has_vision
                    for field in ("key_v0", "key_v1", "tok_v0", "tok_v1", "img0", "img1"):
                        try:
                            getattr(q, field)
                        except rt.api.TokensUnavailable:
                            pass
                        else:
                            raise AssertionError(f"blind query exposed {field}")
                if step == 2:
                    before = (s.step, s.b_aex.n, conn.orch._step_counter, rt.decision_count)
                    try:
                        conn.infer(obs)
                    except ValueError as exc:
                        assert "duplicate decision_id" in str(exc)
                    else:
                        raise AssertionError("duplicate decision was not rejected")
                    assert before == (s.step, s.b_aex.n, conn.orch._step_counter, rt.decision_count)
        for conn in conns:
            conn.on_episode_end(success=False)
    decs = [json.loads(line) for line in rt.dec_path.read_text().splitlines() if '"ev": "dec"' in line]
    visions = sum(d["vision"] for d in decs)
    assert len(decs) == 48
    assert sum(c.calls for c in conns) == visions
    assert sum(c.broadcasts for c in conns) == 48
    assert all(d["s1_ms"] is None and d["s23_ms"] is None for d in decs if not d["vision"])
    assert all(d["served_head"] == action[:5, :7].tolist() for d, action in zip(decs, served))
    if is_probe:
        assert [not d["vision"] for d in decs] == expected
        if a.judge == "guard_only" and json.loads(a.kwargs).get("budget", 2) == 2 and not a.judge_cap and a.judge_burst == 1:
            seq = [d["src"] for d in decs if d["conn"] == 0][:6]
            assert seq == ["cache", "cache_blind", "cache_blind", "cache", "policy", "cache"], seq
    rc = verify_logs.main(["--log-dir", str(out), "--tag", "blindtest", "--work", str(out / "verify")])
    rep = dict(PASS=rc == 0, decisions=48, vision=visions, blind=48-visions,
               miss=sum(not d["hit"] for d in decs), stage1_calls=sum(c.calls for c in conns),
               broadcasts=sum(c.broadcasts for c in conns), connections=2, episodes=4, duplicate_rejections=4,
               rejected_preflight=rejected_preflight, output_fallbacks=output_fallbacks, partial_looks=partial_looks)
    (out / "selftest_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="library cell the method is fitted on (<m>_<s>_cache)")
    ap.add_argument("--yaml", required=True, help="served arm yaml (library + key builder)")
    ap.add_argument("--method", required=True)
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--root", default="/dev/shm/offline_search_store")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-shadow", action="store_true")
    ap.add_argument("--fit-artifact", default="", help="load the method from this prefit pickle (plugin --os-fit-artifact)")
    ap.add_argument("--judge", default=None, help="mixed mode: the server's --os-judge spec (absent = pure cache)")
    ap.add_argument("--judge-cap", type=int, default=0)
    ap.add_argument("--judge-step0", default="judge", choices=("judge", "miss", "hit"))
    ap.add_argument("--judge-burst", type=int, default=1)
    ap.add_argument("--replay-cell", default="", help="query cell whose recorded episodes are replayed (default --cell); "
                    "an _inf cell with --judge periodic:1 checks the after-MISS QueryView against the inf-cell view")
    ap.add_argument("--blind", action="store_true", help="run the interleaved R4 blind serving test")
    a = ap.parse_args(argv)
    if a.blind:
        return blind_main(a)

    from . import plugin
    from . import verify_logs as vl
    from exp.offline_search.harness import api, run, store

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(out / "inputs" / "*.npz")) + glob.glob(str(out / "decisions_*.jsonl")):
        os.remove(f)
    model = store.parse_cell(a.cell)[0]
    replay_cell = a.replay_cell or a.cell
    mixed = a.judge is not None
    pargs = ["--os-method", a.method, "--os-kwargs", a.kwargs, "--os-cell", a.cell, "--os-root", a.root,
             "--os-log-dir", str(out), "--os-tag", "selftest", "--os-log-inputs"]
    if a.no_shadow:
        pargs.append("--os-no-shadow-native")
    if a.fit_artifact:
        pargs += ["--os-fit-artifact", a.fit_artifact]
    if mixed:
        pargs += ["--os-judge", a.judge, "--os-judge-cap", str(a.judge_cap), "--os-judge-step0", a.judge_step0,
                  "--os-judge-burst", str(a.judge_burst)]
    opts, rest = plugin.parse_cli(pargs)
    assert not rest, rest
    rt = plugin.install(opts, model=model)

    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator

    cfg = cc.load_cache_config(a.yaml)
    shared = cc.build_shared_storage(cfg)
    qc = store.QueryCell(a.root, replay_cell)
    if qc.lib_key != store.lib_key(a.cell):
        raise SystemExit(f"replay cell {replay_cell} is not a {store.lib_key(a.cell)} cell")
    kb = FakeKB(qc)
    policy = np.asarray(qc.a_inf) if mixed else None      # the recorded full inference at every replayed state

    def factory(_base, bundle_id="default"):
        comps = cc.build_per_connection_components(cfg, shared, quiet=True)
        comps["key_builder"] = kb
        for s in plugin._TLS.new_sessions:
            s.kb = kb
        orch = CacheOrchestrator(storage=comps["storage"], key_builder=kb, gates=comps["gates"],
                                 judges=comps["judges"], search_strategies=comps["search_strategies"],
                                 timer=comps["timer"], write_policy=comps.get("write_policy"),
                                 offline_writers=comps.get("offline_writers", ()),
                                 library_stats=comps.get("library_stats"))
        return FakePolicy(orch, kb, policy)

    conn = plugin._wrap_factory(factory)(None, "default")
    conn.on_task_begin()
    sel = pick_episodes(qc, a.episodes)
    served_winners = []
    client_hits = []
    for ei in sel:
        e = qc.episodes[ei]
        conn.on_episode_start(experiment="selftest", task=e["task"], episode_id=e["init"], episode_name="",
                              extra_metadata={"task_id": e["task_id"], "orig_init_state_idx": e["init"],
                                              "task_uid": e["uid"], "attempt": 1})
        for r in range(e["start"], e["end"]):
            outp = conn.infer({"observation/state": np.asarray(qc.raw_state[r], np.float64), "_row": r,
                               "observation/image": np.zeros((2, 2, 3), np.uint8),
                               "observation/wrist_image": np.zeros((2, 2, 3), np.uint8)})
            served_winners.append(outp["winner"])
            client_hits.append(1 if outp["hit_type"] == "FULL_HIT" else 0)
        conn.on_episode_end(success=bool(e["success"]))
    conn.on_task_end()

    # ---- online logs
    online = {}
    for f in sorted(glob.glob(str(out / "inputs" / "*.npz"))):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        online[m["uid"]] = z
    decs = [json.loads(line) for line in open(rt.dec_path) if '"ev": "dec"' in line]
    rep = {"cell": a.cell, "replay_cell": replay_cell, "method": rt.method_name, "episodes": len(sel),
           "decisions": len(decs), "judge": rt.judge.as_dict() if rt.judge is not None else None}
    rep["exec_ok"] = int(sum(1 for d in decs if d.get("exec_ok")))
    rep["winner_matches_log"] = bool([d["winner"] for d in decs] == served_winners)

    # ---- (1) agreement with the recorded online top-1
    rec = np.asarray(qc.rec_top1)
    ag_plugin = ag_native = n = 0
    ep_all_agree = {}
    ep_hist_equal = {}          # online executed chunks == the replay cell's recorded a_exec (offline sees the same)
    for ei in sel:
        e = qc.episodes[ei]
        z = online[e["uid"]]
        rows = np.arange(e["start"], e["end"])
        p = z["top1"]
        libs = z["lib"]
        ok = (p == rec[rows]) & (libs == "current")
        ag_plugin += int(ok.sum())
        if "native_top1" in z.files:
            ag_native += int((z["native_top1"] == rec[rows]).sum())
        n += rows.size
        ep_all_agree[e["uid"]] = bool(ok.all()) and not bool(z["used_synth"].any())   # executed == recorded
        ep_hist_equal[e["uid"]] = bool(z["a_exec"].shape[0] == rows.size and
                                       np.array_equal(z["a_exec"], np.asarray(qc.a_exec[rows], np.float32)))
        # (3) executed chunk vs recorded a_exec when the plugin served a library pick
        if ok.all() and not z["used_synth"].any() and not mixed:
            if not np.array_equal(z["a_exec"], np.asarray(qc.a_exec[rows], np.float32)):
                rep.setdefault("a_exec_mismatch_eps", []).append(e["uid"])
    rep.update({"agree_rec_top1": ag_plugin / n, "native_agree_rec_top1": ag_native / n if not a.no_shadow else None,
                "episodes_all_agree": int(sum(ep_all_agree.values()))})

    ok_all = rep["winner_matches_log"] and not rep.get("a_exec_mismatch_eps")
    if not mixed:
        ok_all = ok_all and rep["exec_ok"] == rep["decisions"]

    if not mixed:
        # ---- (2) offline harness on the same episodes
        cls, _src = run.load_method_class(a.method)
        F = run._fit_cell(cls, json.loads(a.kwargs), a.cell, root=a.root, out_dir=out / "offline", seed=0, profile=False)
        jobs = [(ei, np.arange(qc.episodes[ei]["start"], qc.episodes[ei]["end"], dtype=np.int64)) for ei in sel]
        off = run.run_jobs_inprocess(F["method"], qc, jobs, lib_sizes=F["lib_sizes"], seed=0, cell=a.cell)
        compared, eqf, first_diff = compare_offline(off, sel, qc, online, ep_all_agree)
        rep["offline_compared"] = compared
        rep["offline_equal"] = eqf
        rep["first_diff"] = first_diff
        ok_all = ok_all and all(v == 1.0 for v in eqf.values())
    else:
        # ---- (4) verdict-aware bookkeeping
        hits_log = np.concatenate([online[qc.episodes[ei]["uid"]]["hit"] for ei in sel]).astype(int)
        mx = {"n": int(hits_log.size), "n_hit": int(hits_log.sum()), "n_miss": int((hits_log == 0).sum()),
              "client_verdict_equals_log": bool(np.array_equal(hits_log, np.asarray(client_hits))),
              "exec_ok_hits": int(sum(1 for d in decs if d.get("hit") and d.get("exec_ok") is True)),
              "exec_ok_none_on_miss": int(sum(1 for d in decs if not d.get("hit") and d.get("exec_ok") is None)),
              "src_ok": all((d["src"] == "cache") == bool(d["hit"]) for d in decs),
              "miss_exec_is_policy": 0, "hit_exec_is_served": 0, "judge_mix": {}, "seq": {}}
        for d in decs:
            mx["judge_mix"][d["judge"]] = mx["judge_mix"].get(d["judge"], 0) + 1
        lib_tables = rt.tables
        for ei in sel:
            e = qc.episodes[ei]
            z = online[e["uid"]]
            rows = np.arange(e["start"], e["end"])
            h = z["hit"]
            for s, r in enumerate(rows):
                if h[s] == 0:
                    mx["miss_exec_is_policy"] += int(np.array_equal(z["a_exec"][s], policy[r].astype(np.float32)))
                else:
                    srv = z["synth"][s] if z["used_synth"][s] else np.asarray(lib_tables[str(z["lib"][s])][int(z["top1"][s])], np.float32)
                    mx["hit_exec_is_served"] += int(np.array_equal(z["a_exec"][s], srv))
            # sequence patterns exercised (HIT->MISS->HIT, consecutive MISS)
            hs = "".join("H" if v else "M" for v in h)
            mx["seq"]["HMH"] = mx["seq"].get("HMH", 0) + hs.count("HMH")
            mx["seq"]["MM"] = mx["seq"].get("MM", 0) + sum(1 for i in range(len(hs) - 1) if hs[i:i + 2] == "MM")
            mx["seq"]["MH"] = mx["seq"].get("MH", 0) + hs.count("MH")
        mx["bookkeeping_ok"] = (mx["client_verdict_equals_log"] and mx["src_ok"] and
                                mx["exec_ok_hits"] == mx["n_hit"] and mx["exec_ok_none_on_miss"] == mx["n_miss"] and
                                mx["miss_exec_is_policy"] == mx["n_miss"] and mx["hit_exec_is_served"] == mx["n_hit"])
        # ---- (5) verdict rule
        J = rt.judge.as_dict()
        bad, run_bad = [], 0
        for ei in sel:
            z = online[qc.episodes[ei]["uid"]]
            c = vl.check_verdicts(J, {k: z[k] for k in z.files if k != "meta"})
            bad += c["bad"]
            run_bad += c["run_bad"]
        mx["verdict_violations"], mx["run_violations"], mx["first_violations"] = len(bad), run_bad, bad[:3]
        # ---- (6) offline replay of the logged inputs (mini store with MISS rows as policy rows)
        eps = vl.load_logs(out, "selftest")
        work = out / "verify_work"
        if work.exists():
            shutil.rmtree(work)
        episodes = vl.build_ministore(work, pathlib.Path(a.root), a.cell, eps, rt.H)
        off, F = vl.run_offline(a.method, json.loads(a.kwargs), a.cell, work, 0, episodes)
        eq = dict(topk=0, scores=0, conf=0, lib=0, synth=0, extras=0)
        first = None
        j = 0
        for (_, m, z), e in zip(eps, episodes):
            for s in range(e["num_steps"]):
                k = off["topk"][j].size
                c = {"topk": np.array_equal(z["topk"][s, :k], off["topk"][j]),
                     "scores": np.array_equal(z["scores"][s, :k], off["scores"][j]),
                     "conf": bool(z["conf"][s] == off["conf"][j]), "lib": str(z["lib"][s]) == off["lib"][j]}
                osyn = off["synth"][j]
                c["synth"] = (osyn is None and not z["used_synth"][s]) or (
                    osyn is not None and np.array_equal(np.nan_to_num(z["synth"][s, :5, :7]), np.nan_to_num(osyn)))
                exo = off["extras"][j] or {}
                dbad = {kk: (float(z[f"x_{kk}"][s]) if f"x_{kk}" in z else None, float(np.asarray(v).reshape(-1)[0]))
                        for kk, v in exo.items() if np.asarray(v).size == 1 and
                        not (f"x_{kk}" in z and np.float64(z[f"x_{kk}"][s]) == np.float64(np.asarray(v).reshape(-1)[0]))}
                c["extras"] = not dbad
                for kk, v in c.items():
                    eq[kk] += int(bool(v))
                if first is None and not all(c.values()):
                    first = {"uid": m["uid"], "step": s, **{kk: bool(v) for kk, v in c.items()}, "extras_diff": dbad}
                j += 1
        mx["offline_ministore_equal"] = {kk: v / max(j, 1) for kk, v in eq.items()}
        mx["offline_ministore_first_diff"] = first
        # ---- (7) direct comparison with the replay cell where the executed history equals the recorded one
        n_hist = int(sum(ep_hist_equal.values()))
        mx["episodes_history_equals_replay_cell"] = n_hist
        if n_hist:
            cls, _src = run.load_method_class(a.method)
            F2 = run._fit_cell(cls, json.loads(a.kwargs), a.cell, root=a.root, out_dir=out / "offline", seed=0,
                               profile=False)
            jobs = [(ei, np.arange(qc.episodes[ei]["start"], qc.episodes[ei]["end"], dtype=np.int64)) for ei in sel]
            off2 = run.run_jobs_inprocess(F2["method"], qc, jobs, lib_sizes=F2["lib_sizes"], seed=0, cell=a.cell)
            compared, eqf, first_diff = compare_offline(off2, sel, qc, online, ep_hist_equal)
            mx["replay_cell_compared"] = compared
            mx["replay_cell_equal"] = eqf
            mx["replay_cell_first_diff"] = first_diff
            ok_all = ok_all and all(v == 1.0 for v in eqf.values())
        rep["mixed"] = mx
        ok_all = (ok_all and mx["bookkeeping_ok"] and mx["verdict_violations"] == 0 and mx["run_violations"] == 0
                  and all(v == 1.0 for v in mx["offline_ministore_equal"].values()))
    rep["q_us_median"] = float(np.median([d["q_us"] for d in decs]))
    rep["native_us_median"] = float(np.median([d["native_us"] for d in decs]))
    rep["PASS"] = bool(ok_all)
    print(json.dumps(rep, indent=1))
    (out / "selftest_report.json").write_text(json.dumps(rep, indent=1))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
