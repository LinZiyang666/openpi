"""Fixed-observation replay of recorded store episodes through the REAL installed plugin (CPU, no GPU/simulator).

Per decision the connection receives the replay cell's recorded observation (keys via selftest.FakeKB, rs,
raw_state, prompt, decision_id, executed_steps=5). The fake interceptor answers a MISS with the recorded full
inference a_inf[row] (the policy's chunk at that very state), a HIT with the served payload, and the blind path's
output transform is the identity, so every returned action is the normalized H x 32 chunk that was served.
Observations do not respond to served actions: this measures decisions on a recorded observation stream, it is
not a rollout (no SR, no counterfactual state evolution).

Writes <out>/decisions_<tag>.jsonl (plugin log), <out>/served.npz (row, ep, step, served chunk, vision, hit,
source per decision, in replay order) and <out>/report.json (counts). Optional --os-log-inputs npz.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[5]
ROOT = "/home/weiland/trace_runs/offline_search_store"


def select_episodes(qc, spec):
    n = len(qc.episodes)
    if spec in ("all", "0"):
        return list(range(n))
    if "," in spec or spec.startswith("="):
        return [int(x) for x in spec.lstrip("=").split(",") if x]
    k = int(spec)
    return sorted(set(np.linspace(0, n - 1, k, dtype=int).tolist()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--kwargs", required=True)
    ap.add_argument("--cell", required=True, help="fit cell <m>_<s>_cache")
    ap.add_argument("--replay-cell", required=True, help="recorded query cell <m>_<s>_{cache,inf}")
    ap.add_argument("--fit-artifact", default="")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--policy-tail", action="store_true")
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--episodes", default="all", help="'all' | N evenly spaced | '=i,j,k'")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--tag", default="replay")
    ap.add_argument("--log-inputs", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from exp.offline_search.closed_loop import plugin, selftest
    from exp.offline_search.harness import store
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    model, suite, _ = store.parse_cell(a.cell)
    if store.lib_key(a.replay_cell) != store.lib_key(a.cell):
        raise SystemExit("replay cell must belong to the fit cell's model x suite")
    args = ["--os-method", a.method, "--os-kwargs", a.kwargs, "--os-cell", a.cell, "--os-root", a.root,
            "--os-log-dir", str(out), "--os-tag", a.tag, "--os-blind", "--os-no-shadow-native"]
    if a.policy_tail:
        args.append("--os-policy-tail")
    if a.blocks is not None:
        args += ["--os-policy-tail-blocks", str(a.blocks)]
    if a.judge:
        args += ["--os-judge", a.judge]
    if a.fit_artifact:
        args += ["--os-fit-artifact", a.fit_artifact]
    if a.log_inputs:
        args.append("--os-log-inputs")
    opts, rest = plugin.parse_cli(args)
    assert not rest, rest
    t0 = time.time()
    rt = plugin.install(opts, model)
    yaml = REPO / f"exp/trace_dual/config/tr_{model}_{'sp' if suite == 'spatial' else 'l10'}_cache.yaml"
    cfg = cc.load_cache_config(str(yaml))
    shared = cc.build_shared_storage(cfg)
    qc = store.QueryCell(a.root, a.replay_cell)
    a_inf = np.asarray(qc.a_inf, np.float32)

    class Fake(selftest.FakePolicy):
        calls = 0

        def stage1(self, obs):                   # instrumented by the plugin (stage-1 timing / call count)
            Fake.calls += 1

        def infer(self, obs):                    # only vision decisions reach the interceptor
            self.stage1(obs)
            return super().infer(obs)

        def _osp_prepare_blind(self, obs):
            return np.array(qc.rs[int(obs["_row"])], copy=True), None

        def _osp_blind_output(self, action, state):
            return {"actions": action.copy()}

    def factory(_base, bundle_id="default"):
        comps = cc.build_per_connection_components(cfg, shared, quiet=True)
        kb = selftest.FakeKB(qc)
        for s in plugin._TLS.new_sessions:
            s.kb = kb
        orch = CacheOrchestrator(storage=comps["storage"], key_builder=kb, gates=comps["gates"],
                                 judges=comps["judges"], search_strategies=comps["search_strategies"],
                                 timer=comps["timer"], write_policy=comps.get("write_policy"),
                                 offline_writers=comps.get("offline_writers", ()),
                                 library_stats=comps.get("library_stats"))
        return Fake(orch, kb, a_inf)

    conn = plugin._wrap_factory(factory)(None, "0")
    s = conn._osp_sessions[0]
    eps = select_episodes(qc, a.episodes)
    rows, epi, steps, served, vision, hit = [], [], [], [], [], []
    img = np.zeros((2, 2, 3), np.uint8)
    t1 = time.time()
    for n_done, ei in enumerate(eps):
        e = qc.episodes[ei]
        conn.on_episode_start(task=e["task"], episode_id=e["init"],
                              extra_metadata={"task_uid": e["uid"], "task_id": e["task_id"],
                                              "orig_init_state_idx": e["init"], "attempt": 1})
        for step, r in enumerate(range(e["start"], e["end"])):
            obs = {"observation/state": np.asarray(qc.raw_state[r], np.float64), "prompt": e["task"], "_row": r,
                   "observation/image": img, "observation/wrist_image": img,
                   "__extra__": {"decision_id": step, "executed_steps": 5}}
            res = conn.infer(obs)
            act = np.asarray(res["actions"], np.float32)
            assert act.shape == (rt.H, 32) and s.step == step + 1 == s.b_aex.n
            assert np.array_equal(s.b_aex.a[step], act), "executed history != served chunk"
            rows.append(r); epi.append(ei); steps.append(step); served.append(act)
            vision.append(bool(s.has_vision[-1])); hit.append(bool(s.hits[-1]))
        conn.on_episode_end(success=bool(e["success"]))
        if (n_done + 1) % 50 == 0:
            print(f"{a.tag}: {n_done + 1}/{len(eps)} episodes, {len(rows)} decisions, "
                  f"{(time.time() - t1) / len(rows) * 1e3:.2f} ms/decision", flush=True)
    decs = [json.loads(line) for line in rt.dec_path.read_text().splitlines() if '"ev": "dec"' in line]
    assert len(decs) == len(rows), (len(decs), len(rows))
    src = [d.get("src", "cache") for d in decs]
    served = np.stack(served)
    np.savez(out / "served.npz", row=np.asarray(rows, np.int64), ep=np.asarray(epi, np.int32),
             step=np.asarray(steps, np.int32), served=served, vision=np.asarray(vision),
             hit=np.asarray(hit), src=np.asarray(src))
    blind_runs, run = collections.Counter(), 0
    for v, st in zip(vision, steps):
        if st == 0:
            run = 0
        run = 0 if v else run + 1
        if not v:
            blind_runs[run] += 1
    report = dict(method=rt.method_name, spec=a.method, kwargs=json.loads(a.kwargs), cell=a.cell,
                  replay_cell=a.replay_cell, fit_artifact=a.fit_artifact or None, judge=a.judge,
                  policy_tail=a.policy_tail, blocks=a.blocks, episodes=len(eps), decisions=len(rows),
                  vision=int(sum(vision)), misses=int(sum(not h for h in hit)),
                  src=dict(collections.Counter(src)), stage1_calls=Fake.calls,
                  max_blind_run=max(blind_runs) if blind_runs else 0,
                  served_sha256=hashlib.sha256(served.tobytes()).hexdigest(),
                  vision_sha256=hashlib.sha256(np.asarray(vision).tobytes()).hexdigest(),
                  wall_s=round(time.time() - t0, 1), ms_per_decision=round((time.time() - t1) / len(rows) * 1e3, 3))
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
