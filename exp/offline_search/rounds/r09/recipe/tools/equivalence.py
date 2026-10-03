"""CPU equivalence proof: R9Recipe vs the reference artifacts, decision by decision (plugin selftest replays of
recorded store episodes through the REAL per-connection plugin stack; nothing launched, no GPU).

Pairs (each side run with identical selftest flags derived from the reference arm's plugin args):
  prod_<cell>      every emitted cell (8 defaults + 2 optional look-saving settings), budget off: the reference arm's
                   artifact vs the emitted R9Recipe arm artifact (pace_wrist cells also compare the camera plan extras)
  esc1_pi05_l10_50 escalation exercised: fable NpGraspEsc3 with lag_threshold 1 vs R9Recipe(escalation, lag 1)
  bud_pi05_l10_50  budget exercised: round-5 GatedNpGraspEsc3 (lag 1, budget 2, gate P off) vs R9Recipe(lag 1, budget 2)
  bud_groot_l10_50 budget exercised: round-5 GatedNpGraspStackGroot3 (budget 0) vs R9Recipe(budget 0)
  bud0_pi05_l10_50 budget exercised on escalation verdicts: GatedNpGraspEsc3 (lag 1, budget 0) vs R9Recipe(lag 1, budget 0)
The debug references import the exploratory classes (build/test time only). Compared per decision: hit, vision, top1,
served_head, src, judge verdict, os_force_miss, os_reason.  Output: ``out/equivalence.json``.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from exp.offline_search.harness import api
from exp.offline_search.rounds.r09.recipe.tools.build_eq import (
    HERE, REFS, RUN, SPEC, build, fit_arg, recipe_kwargs, ref_row)

EQ_DIR = "equivalence_v2"      # v1 (before the 2026-10-02 cell update) is kept in RUN / "equivalence"

REPO = Path("/home/weiland/projects/openpi")
STORE = "/home/weiland/trace_runs/offline_search_store"
CPUS = "2-9,46-53"
FIELDS = ("hit", "vision", "top1", "served_head", "src", "judge")
EX_FIELDS = ("os_force_miss", "os_reason", "os_sw_next_camera", "os_sw_reason", "os_pw_lag", "os_pw_pace", "os_sf_granted")


def selftest_flags(ref):
    a = ref["plugin_args"]
    flags = []
    if "--os-blind" in a:
        flags.append("--blind")
    if "--os-policy-tail" in a:
        flags.append("--policy-tail")
    if "--os-policy-tail-blocks" in a:
        flags += ["--policy-tail-blocks", a[a.index("--os-policy-tail-blocks") + 1]]
    if "--os-judge" in a:
        flags += ["--judge", a[a.index("--os-judge") + 1]]
    return flags


def ref_yaml(cell):
    root, arm = REFS[cell]
    return ref_row(cell)["yaml"]


def _publish(path, method, spec, kw, cell):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(method=method, registered={}, spec=spec, kwargs=kw, cell=cell, fit_s=0.0), f, protocol=4)
    return str(path)


def debug_reference(kind, cell):
    """Exploratory reference classes with the debug thresholds (build/test time only)."""
    ref = ref_row(cell)
    kw = copy.deepcopy(ref["kwargs"])
    if kind == "esc1":
        from exp.offline_search.rounds.r09.explore_fable.round3.tools.methods import NpGraspEsc3 as C
        spec = "exp.offline_search.rounds.r09.explore_fable.round3.tools.methods:NpGraspEsc3"
        kw["lag_threshold"] = 1
    elif cell.startswith("pi05"):
        from exp.offline_search.rounds.r09.explore_opus.round5.methods import GatedNpGraspEsc3 as C
        spec = "exp.offline_search.rounds.r09.explore_opus.round5.methods:GatedNpGraspEsc3"
        kw.update(lag_threshold=1, pace_lag_max=None, call_budget=0 if kind == "bud0" else 2, force_noprog_at=[])
    else:
        from exp.offline_search.rounds.r09.explore_opus.round5.methods import GatedNpGraspStackGroot3 as C
        spec = "exp.offline_search.rounds.r09.explore_opus.round5.methods:GatedNpGraspStackGroot3"
        kw.update(pace_lag_max=None, call_budget=0, force_noprog_at=[])
    m = C(**kw)
    m.prof = api.NULL_PROFILER
    m.fit(None, SimpleNamespace(cell=ref["cell"]))
    return spec, kw, _publish(HERE / "artifacts" / "equivalence" / f"ref_{kind}_{cell}.pkl", m, spec, kw, ref["cell"])


def jobs():
    arms = {a["arm"]: a for a in json.loads((RUN / "arms.json").read_text())}
    out = []
    for cell in REFS:
        ref = ref_row(cell)
        flags = selftest_flags(ref)
        out.append((f"prod_{cell}", "ref", ref["cell"], ref_yaml(cell), ref["method"], ref["kwargs"], fit_arg(ref), flags))
        a = arms[f"r9eq_{cell}"]
        out.append((f"prod_{cell}", "recipe", ref["cell"], ref_yaml(cell), a["method"], a["kwargs"], fit_arg(a), flags))
    for kind, cell, over in (("esc1", "pi05_l10_50", dict(lag_threshold=1)),
                             ("bud", "pi05_l10_50", dict(lag_threshold=1, call_budget=2)),
                             ("bud", "groot_l10_50", dict(call_budget=0)),
                             ("bud0", "pi05_l10_50", dict(lag_threshold=1, call_budget=0))):
        ref = ref_row(cell)
        flags = selftest_flags(ref)
        spec, kw, art = debug_reference(kind, cell)
        out.append((f"{kind}_{cell}", "ref", ref["cell"], ref_yaml(cell), spec, kw, art, flags))
        rkw = recipe_kwargs(cell, **over)
        rart = build(cell, rkw, HERE / "artifacts" / "equivalence" / f"recipe_{kind}_{cell}.pkl")
        out.append((f"{kind}_{cell}", "recipe", ref["cell"], ref_yaml(cell), SPEC, rkw, rart, flags))
    return out


def run(job, episodes):
    pair, side, cell, yaml, method, kw, art, flags = job
    outdir = RUN / EQ_DIR / pair / side
    if outdir.exists():
        raise SystemExit(f"{outdir} exists; selftests need a fresh directory")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    log = RUN / EQ_DIR / pair / f"{side}.log"
    cmd = ["taskset", "-c", CPUS, str(REPO / ".venv/bin/python"), "-m", "exp.offline_search.closed_loop.selftest",
           "--cell", cell, "--yaml", yaml, "--method", method, "--kwargs", json.dumps(kw), "--fit-artifact", art,
           "--root", STORE, *flags, "--episodes", str(episodes), "--no-shadow", "--out", str(outdir)]
    env = dict(PYTHONPATH=".:src", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PATH="/usr/bin:/bin", HOME="/home/weiland")
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO), env=env).returncode
    return pair, side, rc


def decisions(pair, side):
    """Selftest logs: synthetic episodes ("blind-c<conn>-e<ep>") with no evaluation init index."""
    rows = []
    for p in sorted((RUN / EQ_DIR / pair / side).glob("decisions*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("ev") == "dec":
                if not str(r.get("uid", "")).startswith("blind-"):
                    raise RuntimeError(f"unexpected selftest uid {r.get('uid')!r}")
                rows.append(r)
    return rows


def compare(pair):
    key = lambda r: (r["uid"], r["step"])     # noqa: E731
    a, b = {key(r): r for r in decisions(pair, "ref")}, {key(r): r for r in decisions(pair, "recipe")}
    diff = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            diff.append(list(k))
            continue
        ra, rb = a[k], b[k]
        if any(ra.get(f) != rb.get(f) for f in FIELDS) or any(
                (ra.get("extras") or {}).get(f) != (rb.get("extras") or {}).get(f) for f in EX_FIELDS):
            diff.append(list(k))
    ra = list(a.values())
    return dict(decisions=len(a), recipe_decisions=len(b), differing=len(diff), first=diff[:5],
                misses=sum(1 for r in ra if r.get("hit") is False), blind=sum(1 for r in ra if not r.get("vision")),
                escalation_misses=sum(1 for r in ra if (r.get("extras") or {}).get("os_reason") == 91.0 and r.get("hit") is False),
                budget_gated=sum(1 for r in b.values() if (r.get("extras") or {}).get("r9o5_gate") == 2.0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="", help="comma-separated pair names to (re)run; existing pairs are re-compared")
    ap.add_argument("--dir", default=EQ_DIR, help="selftest directory under the run root (fresh per suite version)")
    a = ap.parse_args(argv)
    globals()["EQ_DIR"] = a.dir
    js = jobs()
    if a.only:
        keep = set(a.only.split(","))
        done = [p.name for p in (RUN / EQ_DIR).iterdir() if p.is_dir()] if (RUN / EQ_DIR).exists() else []
        js = [j for j in js if j[0] in keep]
    else:
        done = []
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda j: run(j, a.episodes), js))
    report = {}
    pairs = list(dict.fromkeys([*done, *(p for p, _, _ in res)]))
    for pair in pairs:
        rcs = {side: rc for p, side, rc in res if p == pair} or "earlier run"
        passes = {}
        for side in ("ref", "recipe"):
            try:
                passes[side] = json.loads((RUN / EQ_DIR / pair / side / "selftest_report.json").read_text()).get("PASS")
            except Exception:
                passes[side] = None
        report[pair] = dict(rc=rcs, PASS=passes, **compare(pair))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / f"{EQ_DIR}.json").write_text(json.dumps(report, indent=1))
    (RUN / EQ_DIR / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
