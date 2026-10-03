"""Optional GR00T LIBERO-10-50 escalation settings of R9Recipe -> run root r09_recipe_gesc (full 500 manifest).

Arms (nothing launched here; no existing default changes):
  r9eq_groot_l10_50_esc     stack + pace-lag escalation exactly as on pi0.5 LIBERO-10-50 (lag 12, deadline 80, persistent)
  r9eq_groot_l10_50_esc12   same trigger, at most 12 escalation calls per episode, then back to the stack (guard active)
Manifest: tasks 0-9 x inits 0-49, copied verbatim from r09_recipe_full_g/eval500.json (thresholds are the frozen
round-2 constants; nothing is fitted or read from any init here).

CPU selftests (``selftest`` sub-command), real plugin stack on recorded store episodes:
  ref_prod   RefGrootEsc (fable's r3c stack composition + my round-2 GR00T escalation class, lag 12) vs the esc arm
  ref_lag1   the same reference vs R9Recipe with lag_threshold 1 (forced trigger: escalation fires in the replays)
  forced     R9Recipe with the trigger forced at decision 2 (debug kwarg): persistent escalation from there on
  forced_cap2 the same + escalation_max_calls 2: exactly 2 escalation calls per episode, afterwards the stack
  arm_esc12  the production esc12 artifact (PASS)
  compat_*   artifacts pickled with the previous recipe.py (r09_recipe_eq) under the new recipe.py vs their references
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from exp.offline_search.harness import api
from exp.offline_search.rounds.r09.recipe.tools.build_eq import (
    HERE, REFS, SPEC, build, fit_arg, recipe_kwargs, ref_row, sha)

RUNS = Path("/home/weiland/trace_runs/os_closed_loop")
RUN = RUNS / "r09_recipe_gesc"
EQ_RUN = RUNS / "r09_recipe_eq"
MANIFEST_SRC = RUNS / "r09_recipe_full_g" / "eval500.json"
REPO = Path("/home/weiland/projects/openpi")
STORE = "/home/weiland/trace_runs/offline_search_store"
CPUS = "2-9,46-53"
CELL = "groot_l10_50"
ARMS = {"r9eq_groot_l10_50_esc": dict(escalation=True, lag_threshold=12, deadline=80),
        "r9eq_groot_l10_50_esc12": dict(escalation=True, lag_threshold=12, deadline=80, escalation_max_calls=12)}
REF_MODULE = "exp.offline_search.rounds.r09.recipe.tools.build_gesc"
REF_SPEC = f"{REF_MODULE}:RefGrootEsc"


def _ref_class():
    """Independent reference: fable's r3c composition (_GraspStack3) over my round-2 GR00T escalation judge."""
    from exp.offline_search.rounds.r08.abl.judge import TriggerGrootCommitJudge
    from exp.offline_search.rounds.r09.explore_fable.round3.tools.methods import _GraspStack3
    from exp.offline_search.rounds.r09.explore_opus.round2.methods import EscalateOnlyNPGroot

    class _Ref(_GraspStack3, EscalateOnlyNPGroot):
        family = "r9_recipe_ref_groot_esc"
        _BASE_CLASS = TriggerGrootCommitJudge

        def __init__(self, onlynp_fit="", corrected_fit="", lag_threshold=12, deadline=80, empty_aperture=0.0009,
                     closed_sign=-1.0, hold_decisions=2, burst=2, max_calls=0, force_trigger_at=()):
            EscalateOnlyNPGroot.__init__(self, onlynp_fit=onlynp_fit, lag_threshold=lag_threshold, deadline=deadline)
            self._gm_init(onlynp_fit, corrected_fit, empty_aperture, closed_sign, hold_decisions, burst, max_calls,
                          force_trigger_at, f"REF_groot_np_corr_esc_L{lag_threshold:g}_D{deadline}")

        def fit(self, lib, ctx):
            self._gm_fit(ctx)

        def reset(self, episode):
            super().reset(episode)
            self._gm_reset(episode)

        def query(self, q):
            return self._gm_query(super().query(q), q)
    return _Ref


def __getattr__(name):            # module-level lazy class so pickles can resolve build_gesc:RefGrootEsc
    if name == "RefGrootEsc":
        cls = _ref_class()
        cls.__name__ = cls.__qualname__ = "RefGrootEsc"
        cls.__module__ = REF_MODULE                 # also when this file runs as __main__
        globals()["RefGrootEsc"] = cls
        return cls
    raise AttributeError(name)


def _publish(path, method, spec, kw, cell):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(method=method, registered={}, spec=spec, kwargs=kw, cell=cell, fit_s=0.0), f, protocol=4)
    return str(path)


def build_ref(lag):
    ref = ref_row(CELL)
    kw = copy.deepcopy(ref["kwargs"])
    kw.update(lag_threshold=lag, deadline=80)
    import importlib
    m = getattr(importlib.import_module(REF_MODULE), "RefGrootEsc")(**kw)
    m.prof = api.NULL_PROFILER
    m.fit(None, SimpleNamespace(cell=ref["cell"]))
    return REF_SPEC, kw, _publish(HERE / "artifacts" / "gesc" / f"ref_groot_esc_L{lag}.pkl", m, REF_SPEC, kw, ref["cell"])


def emit_arms():
    (RUN / "manifests").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MANIFEST_SRC, RUN / "manifests" / "eval500.json")
    pairs = json.loads((RUN / "manifests" / "eval500.json").read_text())
    assert len(pairs) == 500 and {(t, i) for t, i in pairs} == {(t, i) for t in range(10) for i in range(50)}
    ref = ref_row(CELL)
    rows, checks = [], []
    for name, over in ARMS.items():
        kw = recipe_kwargs(CELL, **over)
        art = build(CELL, kw, HERE / "artifacts" / f"{name}.pkl")
        a = [x for x in ref["plugin_args"] if x != "--os-debug"]
        a[a.index("--os-fit-artifact") + 1] = art
        rows.append(dict(name=name, model=ref["model"], suite=ref["suite_short"], mode="plugin", method=SPEC, kwargs=kw,
                         cost_ledger=True, client_overrides=copy.deepcopy(ref.get("client_overrides")), full_model=True,
                         manifest=str(RUN / "manifests" / "eval500.json"), plugin_args=a))
        checks.append(dict(arm=name, artifact=art, artifact_sha256=sha(art), kwargs=kw))
    (HERE / "arms_gesc.json").write_text(json.dumps(rows, indent=1))
    (HERE / "out" / "gesc_checks.json").write_text(json.dumps(checks, indent=1))
    emit(["--run-root", str(RUN), "--spec", str(HERE / "arms_gesc.json")])
    print(json.dumps([r["name"] for r in rows]))


def _flags(ref):
    a = ref["plugin_args"]
    return ["--blind", "--policy-tail", "--policy-tail-blocks", a[a.index("--os-policy-tail-blocks") + 1],
            "--judge", a[a.index("--os-judge") + 1]]


def _run(job, episodes=12):
    name, cell, yaml, method, kw, art, flags = job
    outdir = RUN / "selftest" / name
    if outdir.exists():
        raise SystemExit(f"{outdir} exists; selftests need a fresh directory")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["taskset", "-c", CPUS, str(REPO / ".venv/bin/python"), "-m", "exp.offline_search.closed_loop.selftest",
           "--cell", cell, "--yaml", yaml, "--method", method, "--kwargs", json.dumps(kw), "--fit-artifact", art,
           "--root", STORE, *flags, "--episodes", str(episodes), "--no-shadow", "--out", str(outdir)]
    env = dict(PYTHONPATH=".:src", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PATH="/usr/bin:/bin", HOME="/home/weiland")
    with open(RUN / "selftest" / f"{name}.log", "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO), env=env).returncode
    return name, rc


def _decisions(name):
    rows = []
    for p in sorted((RUN / "selftest" / name).glob("decisions*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("ev") == "dec":
                if not str(r.get("uid", "")).startswith("blind-"):
                    raise RuntimeError(f"unexpected selftest uid {r.get('uid')!r}")
                rows.append(r)
    return rows


def _compare(a, b):
    key = lambda r: (r["uid"], r["step"])     # noqa: E731
    da, db = {key(r): r for r in _decisions(a)}, {key(r): r for r in _decisions(b)}
    diff = [k for k in sorted(set(da) | set(db)) if k not in da or k not in db or any(
        da[k].get(f) != db[k].get(f) for f in ("hit", "vision", "top1", "served_head", "src", "judge")) or any(
        (da[k].get("extras") or {}).get(f) != (db[k].get("extras") or {}).get(f) for f in ("os_force_miss", "os_reason"))]
    return dict(n=len(da), n_other=len(db), differing=len(diff), first=[list(k) for k in diff[:5]])


def _episodes(name):
    return sorted({r["uid"] for r in _decisions(name)})


def _summary(name):
    rows = _decisions(name)
    per_ep = {}
    for r in rows:
        if r.get("hit") is False and (r.get("extras") or {}).get("os_reason") == 91.0:
            per_ep[r["uid"]] = per_ep.get(r["uid"], 0) + 1
    rep = {}
    try:
        rep = json.loads((RUN / "selftest" / name / "selftest_report.json").read_text())
    except Exception:
        pass
    return dict(PASS=rep.get("PASS"), decisions=len(rows), misses=sum(1 for r in rows if r.get("hit") is False),
                escalation_calls_per_episode=per_ep, guard_calls=sum(1 for r in rows if r.get("hit") is False and
                                                                      (r.get("extras") or {}).get("os_reason") == 4.0))


def selftest(workers=6):
    ref = ref_row(CELL)
    flags, yaml = _flags(ref), ref["yaml"]
    arms = {a["arm"]: a for a in json.loads((RUN / "arms.json").read_text())}
    s12, k12, a12 = build_ref(12)
    s1, k1, a1 = build_ref(1)
    kw_lag1 = recipe_kwargs(CELL, escalation=True, lag_threshold=1, deadline=80)
    art_lag1 = build(CELL, kw_lag1, HERE / "artifacts" / "gesc" / "recipe_groot_esc_L1.pkl")
    kw_cap = recipe_kwargs(CELL, escalation=True, lag_threshold=12, deadline=80, escalation_max_calls=2,
                           force_escalation_at=[2])
    art_cap = build(CELL, kw_cap, HERE / "artifacts" / "gesc" / "recipe_groot_esc_forced2_cap2.pkl")
    kw_force = recipe_kwargs(CELL, escalation=True, lag_threshold=12, deadline=80, force_escalation_at=[2])
    art_force = build(CELL, kw_force, HERE / "artifacts" / "gesc" / "recipe_groot_esc_forced2.pkl")
    esc = arms["r9eq_groot_l10_50_esc"]
    jobs = [("ref_prod__ref", ref["cell"], yaml, s12, k12, a12, flags),
            ("ref_prod__recipe", ref["cell"], yaml, SPEC, esc["kwargs"], fit_arg(esc), flags),
            ("ref_lag1__ref", ref["cell"], yaml, s1, k1, a1, flags),
            ("ref_lag1__recipe", ref["cell"], yaml, SPEC, kw_lag1, art_lag1, flags),
            ("forced__recipe", ref["cell"], yaml, SPEC, kw_force, art_force, flags),
            ("forced_cap2__recipe", ref["cell"], yaml, SPEC, kw_cap, art_cap, flags),
            ("arm_esc12", ref["cell"], yaml, SPEC, arms["r9eq_groot_l10_50_esc12"]["kwargs"], fit_arg(arms["r9eq_groot_l10_50_esc12"]), flags)]
    eq_arms = {a["arm"]: a for a in json.loads((EQ_RUN / "arms.json").read_text())}
    for cell in ("pi05_l10_50", "groot_l10_50"):               # backward compatibility of older recipe pickles
        r = ref_row(cell)
        e = eq_arms[f"r9eq_{cell}"]
        jobs += [(f"compat_{cell}__ref", r["cell"], r["yaml"], r["method"], r["kwargs"], fit_arg(r), _flags(r)),
                 (f"compat_{cell}__recipe", r["cell"], r["yaml"], SPEC, e["kwargs"], fit_arg(e), _flags(r))]
    with ThreadPoolExecutor(workers) as ex:
        rcs = dict(ex.map(_run, jobs))
    report = dict(rc=rcs, ref_prod=_compare("ref_prod__ref", "ref_prod__recipe"),
                  ref_lag1=_compare("ref_lag1__ref", "ref_lag1__recipe"),
                  compat_pi05_l10_50=_compare("compat_pi05_l10_50__ref", "compat_pi05_l10_50__recipe"),
                  compat_groot_l10_50=_compare("compat_groot_l10_50__ref", "compat_groot_l10_50__recipe"),
                  summaries={n: _summary(n) for n in rcs})
    forced = report["summaries"]["forced__recipe"]["escalation_calls_per_episode"]
    cap = report["summaries"]["forced_cap2__recipe"]["escalation_calls_per_episode"]
    report["cap_check"] = dict(forced_escalation_calls=forced, forced_cap2_escalation_calls=cap,
                               ok=bool(cap) and all(v == 2 for v in cap.values()) and all(v > 2 for v in forced.values())
                               and len(forced) == len(_episodes("forced__recipe")))
    (HERE / "out" / "gesc_selftest.json").write_text(json.dumps(report, indent=1))
    (RUN / "selftest" / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["emit", "selftest"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    emit_arms() if a.action == "emit" else selftest(a.workers)


if __name__ == "__main__":
    main()
