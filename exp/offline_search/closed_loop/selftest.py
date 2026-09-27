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
    """Stand-in for the interceptor: CP1 check -> FULL_HIT payload -> broadcast_action, like the real FULL_HIT path."""

    def __init__(self, orch, kb):
        self.orch, self.kb = orch, kb

    def infer(self, obs):
        from openpi.cache.components.judge import HitType
        from openpi.cache.types import CheckpointID

        self.kb.row = int(obs["_row"])
        res = self.orch.check(CheckpointID.CP1, stage1=None)
        if res.hit_type != HitType.FULL_HIT:
            raise RuntimeError(f"expected FULL_HIT, got {res.hit_type}")
        chunk = res.payload.action_chunk
        self.orch.broadcast_action(chunk)
        self.orch.clear()
        return {"winner": res.entry_id, "actions": np.asarray(chunk)}

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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--yaml", required=True, help="served arm yaml (library + key builder)")
    ap.add_argument("--method", required=True)
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--root", default="/dev/shm/offline_search_store")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-shadow", action="store_true")
    ap.add_argument("--fit-artifact", default="", help="load the method from this prefit pickle (plugin --os-fit-artifact)")
    a = ap.parse_args(argv)

    from exp.offline_search.closed_loop import plugin
    from exp.offline_search.harness import api, run, store

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(out / "inputs" / "*.npz")) + glob.glob(str(out / "decisions_*.jsonl")):
        os.remove(f)
    model = store.parse_cell(a.cell)[0]
    pargs = ["--os-method", a.method, "--os-kwargs", a.kwargs, "--os-cell", a.cell, "--os-root", a.root,
             "--os-log-dir", str(out), "--os-tag", "selftest", "--os-log-inputs"]
    if a.no_shadow:
        pargs.append("--os-no-shadow-native")
    if a.fit_artifact:
        pargs += ["--os-fit-artifact", a.fit_artifact]
    opts, rest = plugin.parse_cli(pargs)
    assert not rest, rest
    rt = plugin.install(opts, model=model)

    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator

    cfg = cc.load_cache_config(a.yaml)
    shared = cc.build_shared_storage(cfg)
    qc = store.QueryCell(a.root, a.cell)
    kb = FakeKB(qc)

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
        return FakePolicy(orch, kb)

    conn = plugin._wrap_factory(factory)(None, "default")
    conn.on_task_begin()
    sel = pick_episodes(qc, a.episodes)
    served_winners = []
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
        conn.on_episode_end(success=bool(e["success"]))
    conn.on_task_end()

    # ---- online logs
    online = {}
    for f in sorted(glob.glob(str(out / "inputs" / "*.npz"))):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        online[m["uid"]] = z
    decs = [json.loads(line) for line in open(rt.dec_path) if '"ev": "dec"' in line]
    rep = {"cell": a.cell, "method": rt.method_name, "episodes": len(sel), "decisions": len(decs)}
    rep["exec_ok"] = int(sum(1 for d in decs if d.get("exec_ok")))
    rep["winner_matches_log"] = bool([d["winner"] for d in decs] == served_winners)

    # ---- (1) agreement with the recorded online top-1
    rec = np.asarray(qc.rec_top1)
    ag_plugin = ag_native = n = 0
    ep_all_agree = {}
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
        # (3) executed chunk vs recorded a_exec when the plugin served a library pick
        if ok.all() and not z["used_synth"].any():
            if not np.array_equal(z["a_exec"], np.asarray(qc.a_exec[rows], np.float32)):
                rep.setdefault("a_exec_mismatch_eps", []).append(e["uid"])
    rep.update({"agree_rec_top1": ag_plugin / n, "native_agree_rec_top1": ag_native / n if not a.no_shadow else None,
                "episodes_all_agree": int(sum(ep_all_agree.values()))})

    # ---- (2) offline harness on the same episodes
    cls, _src = run.load_method_class(a.method)
    F = run._fit_cell(cls, json.loads(a.kwargs), a.cell, root=a.root, out_dir=out / "offline", seed=0, profile=False)
    jobs = [(ei, np.arange(qc.episodes[ei]["start"], qc.episodes[ei]["end"], dtype=np.int64)) for ei in sel]
    off = run.run_jobs_inprocess(F["method"], qc, jobs, lib_sizes=F["lib_sizes"], seed=0, cell=a.cell)
    i = 0
    eq = {"topk": 0, "scores": 0, "conf": 0, "synth": 0, "extras": 0}
    compared = 0
    first_diff = None
    for ei in sel:
        e = qc.episodes[ei]
        z = online[e["uid"]]
        hist_ok = ep_all_agree[e["uid"]]
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
            same_ex = all(np.float64(z[f"x_{kk}"][s]) == np.float64(v) for kk, v in exo.items()
                          if np.asarray(v).size == 1)
            for key, v in (("topk", same_topk), ("scores", same_sc), ("conf", same_conf), ("synth", same_syn),
                           ("extras", same_ex)):
                eq[key] += int(bool(v))
            if first_diff is None and not (same_topk and same_sc and same_conf and same_syn and same_ex):
                bad = {kk: (float(z[f"x_{kk}"][s]), float(v)) for kk, v in exo.items()
                       if np.asarray(v).size == 1 and np.float64(z[f"x_{kk}"][s]) != np.float64(v)}
                first_diff = {"uid": e["uid"], "step": s, "topk": bool(same_topk), "scores": bool(same_sc),
                              "conf": bool(same_conf), "synth": bool(same_syn), "extras_diff": bad}
        i += e["end"] - e["start"]
    rep["offline_compared"] = compared
    rep["offline_equal"] = {k: v / max(compared, 1) for k, v in eq.items()}
    rep["first_diff"] = first_diff
    rep["q_us_median"] = float(np.median([d["q_us"] for d in decs]))
    rep["native_us_median"] = float(np.median([d["native_us"] for d in decs]))
    print(json.dumps(rep, indent=1))
    (out / "selftest_report.json").write_text(json.dumps(rep, indent=1))
    ok = (rep["exec_ok"] == rep["decisions"] and rep["winner_matches_log"] and all(v == 1.0 for v in rep["offline_equal"].values())
          and not rep.get("a_exec_mismatch_eps"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
