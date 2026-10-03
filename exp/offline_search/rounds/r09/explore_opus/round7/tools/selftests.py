"""CPU plugin selftests for round 7 (budget-only arms) (real judge chain on recorded store keys; no GPU, no simulator, nothing launched).

Per model:
  stack       fable's frozen r3c artifact (the control arm), production kwargs
  off         the gated class with both gates disabled -> must reproduce ``stack`` decision by decision
  C20 / C15   the two production budget arms (their published artifacts)
  forceC      call_budget=1 + forced verdicts, gate P off -> at most one guard call per episode
Debug artifacts are written to ``round7/artifacts/selftest/`` (never used by an arm).
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from exp.offline_search.harness import api
from exp.offline_search.rounds.r09.explore_opus.round2.methods import publish
from exp.offline_search.rounds.r09.explore_opus.round5 import methods as m5
from exp.offline_search.rounds.r09.explore_opus.round7.tools.prepare_arms import (
    HERE, M5, R3C, RUN, STACKS, fit_arg, spec, variants)

REPO = Path("/home/weiland/projects/openpi")
STORE = "/home/weiland/trace_runs/offline_search_store"
CPUS = "2-9,46-53"
FORCE = [2, 4, 6, 8, 10]


def _debug_artifact(model, tag, gate):
    src_arm, cls_name = STACKS[model]
    src = spec(R3C, src_arm)
    kw = dict(copy.deepcopy(src["kwargs"]), **gate)
    m = getattr(m5, cls_name)(**kw)
    m.prof = api.NULL_PROFILER
    m.fit(None, SimpleNamespace(cell=src["cell"]))
    out = HERE / "artifacts" / "selftest" / f"{model}_{tag}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    publish(out, m, f"{M5}:{cls_name}", kw, src["cell"], dict(debug=True, purpose=f"selftest {tag}"))
    return f"{M5}:{cls_name}", kw, str(out)


def jobs():
    arms = {a["arm"]: a for a in json.loads((RUN / "arms.json").read_text())}
    out = []
    for model, (src_arm, _) in STACKS.items():
        src = spec(R3C, src_arm)
        yaml = arms[f"r9o7_{model}_l10_50_stack"]["yaml"]
        flags = ["--blind", "--policy-tail", "--judge", "guard_only"] + (["--policy-tail-blocks", "1"] if model == "groot" else [])
        out.append((f"{model}_stack", src["cell"], yaml, src["method"], src["kwargs"], fit_arg(src), flags))
        for tag, gate in [("off", dict(pace_lag_max=None, call_budget=None, force_noprog_at=[])),
                          ("forceC", dict(pace_lag_max=None, call_budget=1, force_noprog_at=FORCE))]:
            method, kw, art = _debug_artifact(model, tag, gate)
            out.append((f"{model}_{tag}", src["cell"], yaml, method, kw, art, flags))
        for tag, _ in variants():
            a = arms[f"r9o7_{model}_l10_50_{tag}"]
            out.append((f"{model}_{tag}", src["cell"], a["yaml"], a["method"], a["kwargs"], fit_arg(a), flags))
    return out


def run(job, episodes):
    name, cell, yaml, method, kw, art, flags = job
    outdir = RUN / "selftest" / name
    if outdir.exists():
        raise SystemExit(f"{outdir} exists; selftests need a fresh directory")
    log = RUN / "selftest" / f"{name}.log"
    cmd = ["taskset", "-c", CPUS, str(REPO / ".venv/bin/python"), "-m", "exp.offline_search.closed_loop.selftest",
           "--cell", cell, "--yaml", yaml, "--method", method, "--kwargs", json.dumps(kw), "--fit-artifact", art,
           "--root", STORE, *flags, "--episodes", str(episodes), "--no-shadow", "--out", str(outdir)]
    env = dict(PYTHONPATH=".:src", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PATH="/usr/bin:/bin", HOME="/home/weiland")
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO), env=env).returncode
    return name, rc


def decisions(name):
    """Selftest decision logs: synthetic selftest episodes ("blind-c<conn>-e<ep>", replayed recorded store keys);
    they carry no evaluation init index, so the RULE-1 init filter does not apply (nothing from runs is read)."""
    rows = []
    for p in sorted((RUN / "selftest" / name).glob("decisions*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("ev") == "dec":
                if not str(r.get("uid", "")).startswith("blind-"):
                    raise RuntimeError(f"unexpected selftest uid {r.get('uid')!r}")
                rows.append(r)
    return rows


def summarize(name):
    rows = decisions(name)
    out = dict(decisions=len(rows), misses=sum(1 for r in rows if r.get("hit") is False), gated_pace=0, gated_budget=0,
               blind_after_gated=0, look_after_gated=0, np_misses=0, max_calls_per_episode=0)
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["uid"], []).append(r)
    for uid, e in by_ep.items():
        e.sort(key=lambda r: r["step"])
        out["max_calls_per_episode"] = max(out["max_calls_per_episode"], sum(1 for r in e if r.get("hit") is False))
        for a, b in zip(e, e[1:] + [None]):
            ex = a.get("extras") or {}
            if a.get("hit") is False and ex.get("os_reason") == 4.0:
                out["np_misses"] += 1
            g = ex.get("r9o5_gate", 0.0)
            if g == 1.0:
                out["gated_pace"] += 1
            if g == 2.0:
                out["gated_budget"] += 1
            if g and b is not None and b["step"] == a["step"] + 1:
                out["blind_after_gated" if not b.get("vision") else "look_after_gated"] += 1
    return out


def compare(a, b):
    ra, rb = decisions(a), decisions(b)
    key = lambda r: (r["uid"], r["step"])     # noqa: E731
    da, db = {key(r): r for r in ra}, {key(r): r for r in rb}
    diff = [k for k in da if k not in db or any(da[k].get(f) != db[k].get(f) for f in ("hit", "vision", "top1", "served_head", "src"))]
    return dict(n=len(da), n_other=len(db), differing=len(diff), first=[list(k) for k in diff[:5]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    (RUN / "selftest").mkdir(parents=True, exist_ok=True)
    js = jobs()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda j: run(j, a.episodes), js))
    report = {}
    for name, rc in res:
        rep = {}
        try:
            rep = json.loads((RUN / "selftest" / name / "selftest_report.json").read_text())
        except Exception:
            pass
        report[name] = dict(rc=rc, PASS=rep.get("PASS"), **summarize(name))
    for model in STACKS:
        report[f"{model}_off_vs_stack"] = compare(f"{model}_off", f"{model}_stack")
    (RUN / "selftest" / "summary.json").write_text(json.dumps(report, indent=1))
    (HERE / "out" / "selftest_summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
