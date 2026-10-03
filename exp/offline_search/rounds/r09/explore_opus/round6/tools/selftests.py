"""CPU plugin selftests for round 6 (real judge chain on recorded store keys; nothing launched).

Per model: the control stack artifact; the takeover class with a dwell that can never be met (``never``) -> must
reproduce the control decision by decision; a forced takeover at decision 2 (``forced``) -> every later fresh decision
of the 24-decision window is a policy call (reason 95) followed by its policy tail; and every production arm.
Debug artifacts go to ``round6/artifacts/selftest/`` (never used by an arm).
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from exp.offline_search.rounds.r09.explore_opus.round2.methods import publish
from exp.offline_search.rounds.r09.explore_opus.round6.tools.prepare_arms import (
    CONTROL, HERE, M6, PLAN, R3C, RUN, build_method, fit_arg, spec, takeover_kwargs)

REPO = Path("/home/weiland/projects/openpi")
STORE = "/home/weiland/trace_runs/offline_search_store"
CPUS = "2-9,46-53"


def _debug(model, tag, src_arm, cls_name, drop, **over):
    src = spec(R3C, src_arm)
    kw = dict(takeover_kwargs(src["kwargs"], drop), **over)
    m = build_method(cls_name, kw, src["cell"])
    out = HERE / "artifacts" / "selftest" / f"{model}_{tag}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    publish(out, m, f"{M6}:{cls_name}", kw, src["cell"], dict(debug=True, purpose=f"selftest {tag}"))
    return f"{M6}:{cls_name}", kw, str(out)


def jobs():
    arms = {a["arm"]: a for a in json.loads((RUN / "arms.json").read_text())}
    out = []
    for model in ("pi05", "groot"):
        csrc = spec(R3C, CONTROL[model])
        yaml = arms[f"r9o6_{model}_l10_50_stack"]["yaml"]
        flags = ["--blind", "--policy-tail", "--judge", "guard_only"] + (["--policy-tail-blocks", "1"] if model == "groot" else [])
        out.append((f"{model}_stack", csrc["cell"], yaml, csrc["method"], csrc["kwargs"], fit_arg(csrc), flags))
        for suffix, src_arm, cls_name, drop in PLAN[model]:
            out.append((f"{model}_{suffix}_never", csrc["cell"], yaml,
                        *_debug(model, f"{suffix}_never", src_arm, cls_name, drop, dwell=1000), flags))
            out.append((f"{model}_{suffix}_forced", csrc["cell"], yaml,
                        *_debug(model, f"{suffix}_forced", src_arm, cls_name, drop, force_exhaust_at=[2]), flags))
            a = arms[f"r9o6_{model}_l10_50_{suffix}"]
            out.append((f"{model}_{suffix}", csrc["cell"], a["yaml"], a["method"], a["kwargs"], fit_arg(a), flags))
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
    """Selftest logs: synthetic episodes ("blind-c<conn>-e<ep>") with no evaluation init index."""
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
    out = dict(decisions=len(rows), misses=sum(1 for r in rows if r.get("hit") is False), takeover_decisions=0,
               takeover_misses=0, policy_tail_after_takeover_miss=0, fresh_in_window_not_called=0)
    by = {}
    for r in rows:
        by.setdefault(r["uid"], []).append(r)
    for e in by.values():
        e.sort(key=lambda r: r["step"])
        for a, b in zip(e, e[1:] + [None]):
            ex = a.get("extras") or {}
            if ex.get("r9o6_takeover") == 1.0:
                out["takeover_decisions"] += 1
                if a.get("hit") is False:
                    out["takeover_misses"] += int(ex.get("os_reason") == 95.0)
                    if b is not None and b.get("src") == "policy_tail":
                        out["policy_tail_after_takeover_miss"] += 1
                elif a.get("vision"):
                    out["fresh_in_window_not_called"] += 1
    return out


def compare(a, b):
    key = lambda r: (r["uid"], r["step"])     # noqa: E731
    da, db = {key(r): r for r in decisions(a)}, {key(r): r for r in decisions(b)}
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
        try:
            rep = json.loads((RUN / "selftest" / name / "selftest_report.json").read_text())
        except Exception:
            rep = {}
        report[name] = dict(rc=rc, PASS=rep.get("PASS"), **summarize(name))
    report["pi05_stack_X_never_vs_stack"] = compare("pi05_stack_X_never", "pi05_stack")
    report["groot_stack_X_never_vs_stack"] = compare("groot_stack_X_never", "groot_stack")
    (RUN / "selftest" / "summary.json").write_text(json.dumps(report, indent=1))
    (HERE / "out" / "selftest_summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
