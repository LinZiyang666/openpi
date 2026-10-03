"""Round 4 (fable): build the six-cell generalization batch ``r09_fable_r4`` (21 arms, NOT launched here).

Per cell (pi0.5 and GR00T x {LIBERO-10 500, Spatial 50, Spatial 500}):
  * ``<cell>_np_corr05``      NpGraspStack3 / NpGraspStackGroot3, max_calls 0: frozen only-no-progress judge of that cell
                              + CorrectedCacheJ (half-strength per-task heads fitted on inits 0-19 of that cell's library)
  * ``<cell>_np_corr05_esc``  (pi0.5 only) the same plus opus's EscalateOnlyNP (lag 12 / deadline 80, imported unchanged)
  * ``<cell>_A``              cache control: copy of the R8 ``r8_<model>_<suite>_<lib>_A`` row (frozen A artifact)
  * ``<cell>_onlynp``         only-no-progress control: copy of the ``r8abl_onlynp_*`` row (frozen judge artifact)
Manifest tasks 0-9 x inits 20-29 (evaluation split; heads are fitted on inits 0-19 only).  Nothing here is task-indexed.

Sub-commands: ``spec`` (arms_in.json + kwargs registry + manifest), ``prefit`` (corrected bases, then stacks),
``check`` (artifact-level assertions: judge.burst unchanged, single judge family, CorrectedCacheJ blend .5, same library),
``selftest`` (CPU plugin selftests, production kwargs + forced empty-grasp triggers, every stack arm).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = next(q for q in HERE.parents if (q / "exp" / "offline_search").is_dir() and (q / "src").is_dir())
R4 = Path(os.environ.get("R9F_R4_ROOT", "/home/weiland/trace_runs/os_closed_loop/r09_fable_r4"))
STORE = "/home/weiland/trace_runs/offline_search_store"
R8_MAIN = Path("/home/weiland/trace_runs/os_closed_loop/r08_main/arms.json")
R8_ABL = Path("/home/weiland/trace_runs/os_closed_loop/r08_abl/arms.json")
MANIFEST_SRC = Path("/home/weiland/trace_runs/os_closed_loop/r09_fable_r2/manifests/eval100_inits20_29.json")
HEADS = HERE.parents[2] / "round2" / "out" / "corrector"
M3 = "exp.offline_search.rounds.r09.explore_fable.round3.tools.methods"
M2 = "exp.offline_search.rounds.r09.explore_fable.round2.tools.methods"
CELLS = [("pi05", "l10", 500), ("pi05", "spatial", 50), ("pi05", "spatial", 500),
         ("groot", "l10", 500), ("groot", "spatial", 50), ("groot", "spatial", 500)]
ABBR = {"pi05": "p", "groot": "g"}
SUITE_ABBR = {"l10": "l10", "spatial": "sp"}
APERTURE = {"pi05": 0.001, "groot": 0.0009}       # empty-grasp aperture thresholds (round 2, robot-state only)
CLOSED_SIGN = {"pi05": 1.0, "groot": -1.0}         # normalized gripper command that means "close"
STACK_CLASS = {"pi05": "NpGraspStack3", "groot": "NpGraspStackGroot3"}
CPUS = os.environ.get("R9F_CPUS", "22-37,66-81")
TRAIN_INITS, EVAL_INITS = range(0, 20), range(20, 30)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path):
    return {a["arm"]: a for a in json.loads(Path(path).read_text())}


def _fit_artifact(row):
    args = row["plugin_args"]
    return args[args.index("--os-fit-artifact") + 1]


def cell_name(model, suite, lib):
    return f"{model}_{suite}_{lib}"


def cell_inputs(model, suite, lib, r8_main=None, r8_abl=None):
    """Frozen inputs of one cell: A row (cache control), only-no-progress row, corrector head."""
    r8_main = r8_main if r8_main is not None else _rows(R8_MAIN)
    r8_abl = r8_abl if r8_abl is not None else _rows(R8_ABL)
    a_row = r8_main[f"r8_{model}_{suite}_{lib}_A"]
    np_row = r8_abl[f"r8abl_onlynp_{ABBR[model]}_{SUITE_ABBR[suite]}_{lib}"]
    base_kwargs = dict(a_row["kwargs"])
    if base_kwargs != np_row["kwargs"]["base_kwargs"]:
        raise ValueError(f"{cell_name(model, suite, lib)}: A kwargs differ from the only-no-progress base kwargs")
    expected_lib = "big" if lib == 500 else "current"
    if base_kwargs["lib"] != expected_lib or base_kwargs["kref"] != (8 if lib == 500 else 5):
        raise ValueError(f"{cell_name(model, suite, lib)}: unexpected library/kref {base_kwargs}")
    head = HEADS / f"head_{model}_{suite}_{lib}_motion_pertask.npz"
    return dict(cell=f"{model}_{suite}_cache", base_kwargs=base_kwargs, base_fit=_fit_artifact(a_row), a_row=a_row,
                onlynp_fit=_fit_artifact(np_row), np_row=np_row, head=str(head))


def build(run_root=R4):
    """Return (arms_in rows, prefit registry, provenance)."""
    fits = Path(run_root) / "fits"
    r8_main, r8_abl = _rows(R8_MAIN), _rows(R8_ABL)
    arms, prefit, prov = [], {}, {}
    judge_args = ["--os-root", STORE, "--os-no-shadow-native", "--os-blind", "--os-policy-tail", "--os-policy-tail-blocks", "1",
                  "--os-judge", "guard_only"]
    for model, suite, lib in CELLS:
        ci = cell_inputs(model, suite, lib, r8_main, r8_abl)
        cn = cell_name(model, suite, lib)
        pre = f"r9f4_{cn}"
        overrides_judge = dict(ci["np_row"]["client_overrides"])
        # 1) corrected base (half strength, per-task heads fitted on inits 0-19)
        corr_name = f"{pre}_corr05pt"
        corr_kwargs = dict(ci["base_kwargs"], head_path=ci["head"], blend=0.5, base_fit=ci["base_fit"], correct_gripper=False)
        prefit[corr_name] = dict(stage=1, method=f"{M2}:CorrectedCache", kwargs=corr_kwargs, cell=ci["cell"], model=model)
        corr_fit = str(fits / f"{corr_name}.pkl")
        # 2) stack arms
        stack_kwargs = dict(onlynp_fit=ci["onlynp_fit"], corrected_fit=corr_fit, empty_aperture=APERTURE[model],
                            closed_sign=CLOSED_SIGN[model], hold_decisions=2, burst=2, max_calls=0, force_trigger_at=[])
        stacks = [(f"{pre}_np_corr05", f"{M3}:{STACK_CLASS[model]}", stack_kwargs)]
        if model == "pi05":
            stacks.append((f"{pre}_np_corr05_esc", f"{M3}:NpGraspEsc3",
                           dict(onlynp_fit=ci["onlynp_fit"], corrected_fit=corr_fit, lag_threshold=12, deadline=80,
                                empty_aperture=APERTURE[model], closed_sign=CLOSED_SIGN[model], hold_decisions=2, burst=2,
                                max_calls=0, force_trigger_at=[])))
        for name, method, kw in stacks:
            prefit[name] = dict(stage=2, method=method, kwargs=kw, cell=ci["cell"], model=model)
            arms.append(dict(name=name, model=model, suite=suite, mode="plugin", full_model=True, method=method, kwargs=kw,
                             cost_ledger=True, client_overrides=overrides_judge,
                             plugin_args=judge_args + ["--os-fit-artifact", str(fits / f"{name}.pkl")]))
        # 3) controls: copies of the frozen R8 rows (same method, kwargs, artifact, plugin args, client overrides)
        a, n = ci["a_row"], ci["np_row"]
        arms.append(dict(name=f"{pre}_A", model=model, suite=suite, mode="plugin", method=a["method"], kwargs=dict(a["kwargs"]),
                         cost_ledger=True, client_overrides=dict(a["client_overrides"]), plugin_args=list(a["plugin_args"])))
        arms.append(dict(name=f"{pre}_onlynp", model=model, suite=suite, mode="plugin", full_model=True, method=n["method"],
                         kwargs=json.loads(json.dumps(n["kwargs"])), cost_ledger=True, client_overrides=overrides_judge,
                         plugin_args=list(n["plugin_args"])))
        prov[cn] = dict(cache_control_source=a["arm"], cache_artifact=ci["base_fit"], cache_artifact_sha256=sha(ci["base_fit"]),
                        onlynp_control_source=n["arm"], onlynp_artifact=ci["onlynp_fit"], onlynp_artifact_sha256=sha(ci["onlynp_fit"]),
                        head=ci["head"], head_sha256=sha(ci["head"]), head_train_inits=[TRAIN_INITS.start, TRAIN_INITS.stop - 1],
                        eval_inits=[EVAL_INITS.start, EVAL_INITS.stop - 1])
    names = [a["name"] for a in arms]
    if len(set(names)) != len(names):
        raise ValueError("duplicate arm names")
    return arms, prefit, prov


def write_spec(run_root=R4):
    run_root = Path(run_root)
    arms, prefit, prov = build(run_root)
    for d in ("fits", "prefit_logs", "manifests", "selftest"):
        (run_root / d).mkdir(parents=True, exist_ok=True)
    (run_root / "arms_in.json").write_text(json.dumps(arms, indent=1))
    (run_root / "r4_kwargs.json").write_text(json.dumps(prefit, indent=1))
    (run_root / "provenance.json").write_text(json.dumps(prov, indent=1))
    manifest = json.loads(MANIFEST_SRC.read_text())
    pairs = {(int(t), int(i)) for t, i in manifest}
    if pairs != {(t, i) for t in range(10) for i in EVAL_INITS}:
        raise ValueError("manifest is not tasks 0-9 x inits 20-29")
    shutil.copyfile(MANIFEST_SRC, run_root / "manifests" / "eval100_inits20_29.json")
    print(f"{len(arms)} arms, {len(prefit)} prefits -> {run_root}")
    return arms, prefit


def _env(threads="2"):
    return dict(os.environ, PYTHONPATH=".:src", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                MKL_NUM_THREADS=threads)


def _run(cmd, log, threads="2"):
    cmd = ["taskset", "-c", CPUS] + cmd
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=_env(threads), cwd=str(REPO))
    return p.returncode


def prefit(run_root=R4, workers=6):
    run_root = Path(run_root)
    reg = json.loads((run_root / "r4_kwargs.json").read_text())
    for stage in (1, 2):
        jobs = {k: v for k, v in reg.items() if v["stage"] == stage}

        def one(item):
            name, spec = item
            art = run_root / "fits" / f"{name}.pkl"
            if art.exists():
                return name, "exists"
            cmd = [str(REPO / ".venv/bin/python"), "-m", "exp.offline_search.closed_loop.plugin", "--os-method", spec["method"],
                   "--os-kwargs", json.dumps(spec["kwargs"]), "--os-cell", spec["cell"], "--os-root", STORE,
                   "--os-log-dir", str(run_root / "prefit_logs"), "--os-fit-artifact", str(art)]
            rc = _run(cmd, run_root / "prefit_logs" / f"{name}.log")
            return name, f"rc={rc}"

        with ThreadPoolExecutor(min(workers, len(jobs))) as ex:
            for name, status in ex.map(one, jobs.items()):
                print(f"stage {stage} {name}: {status}")
                if status not in ("exists", "rc=0"):
                    raise SystemExit(f"prefit failed: {name} ({status})")


def check(run_root=R4):
    """Artifact-level assertions for every stack arm (the r3c fixes), printed as a table."""
    run_root = Path(run_root)
    sys.path.insert(0, str(REPO))
    from exp.offline_search.rounds.r09.explore_fable.round3.tools import methods as m3
    from exp.offline_search.rounds.r08.abl.judge import TriggerCommitJudge, TriggerGrootCommitJudge
    reg = json.loads((run_root / "r4_kwargs.json").read_text())
    rows = []
    for name, spec in reg.items():
        if spec["stage"] != 2:
            continue
        with open(run_root / "fits" / f"{name}.pkl", "rb") as f:
            blob = pickle.load(f)
        obj = blob["method"] if isinstance(blob, dict) else blob
        with open(spec["kwargs"]["onlynp_fit"], "rb") as f:
            src_blob = pickle.load(f)
        src = src_blob["method"] if isinstance(src_blob, dict) else src_blob
        fam = TriggerGrootCommitJudge if spec["model"] == "groot" else TriggerCommitJudge
        checks = dict(judge_burst_unchanged=int(obj.burst) == int(src.burst) == 0, gm_burst=int(obj.gm_burst) == 2,
                      max_calls=int(obj.gm_max_calls) == 0, single_family=isinstance(obj, fam) and isinstance(src, fam),
                      base_is_corrected_judge=type(obj.base).__name__ == "CorrectedCacheJ", blend_half=float(obj.base.blend) == 0.5,
                      same_library=bool(__import__("numpy").array_equal(obj.base.act, src.base.act)),
                      disabled_guards=sorted(getattr(src, "disabled_guards", []) or []) == ["overtime", "stuck", "terminal"],
                      cell=(blob.get("cell") if isinstance(blob, dict) else None) == spec["cell"],
                      sign=float(obj.gm_sign) == CLOSED_SIGN[spec["model"]],
                      escalation=(hasattr(obj, "lag_threshold") and int(obj.lag_threshold) == 12 and int(obj.deadline) == 80)
                      if "esc" in name else not hasattr(obj, "lag_threshold"))
        rows.append((name, type(obj).__name__, checks))
    bad = [(n, [k for k, v in c.items() if not v]) for n, _, c in rows if not all(c.values())]
    for n, cls, c in rows:
        print(f"{n:42s} {cls:20s} judge.burst=0 gm_burst=2 blend=.5 {'OK' if all(c.values()) else 'FAIL ' + str([k for k, v in c.items() if not v])}")
    if bad:
        raise SystemExit(f"artifact checks failed: {bad}")
    print(f"{len(rows)} stack artifacts pass all checks")


def selftest(run_root=R4, workers=9, episodes=4):
    """CPU plugin selftests: production kwargs (fit artifact) and forced empty-grasp triggers (max_calls 2, force at 2 and 6)."""
    run_root = Path(run_root)
    reg = json.loads((run_root / "r4_kwargs.json").read_text())
    jobs = []
    for name, spec in reg.items():
        if spec["stage"] != 2:
            continue
        yaml = run_root / "config" / f"{name}.yaml"
        if not yaml.exists():
            raise SystemExit(f"emit the arms first: {yaml} missing")
        base = [str(REPO / ".venv/bin/python"), "-m", "exp.offline_search.closed_loop.selftest", "--cell", spec["cell"], "--yaml", str(yaml),
                "--method", spec["method"], "--root", STORE, "--blind", "--policy-tail", "--policy-tail-blocks", "1", "--judge", "guard_only",
                "--episodes", str(episodes), "--no-shadow"]
        jobs.append((f"{name}_prod", base + ["--kwargs", json.dumps(spec["kwargs"]), "--fit-artifact", str(run_root / "fits" / f"{name}.pkl"),
                                              "--out", str(run_root / "selftest" / f"{name}_prod")]))
        forced = dict(spec["kwargs"], max_calls=2, force_trigger_at=[2, 6])
        jobs.append((f"{name}_forced", base + ["--kwargs", json.dumps(forced), "--out", str(run_root / "selftest" / f"{name}_forced")]))

    def one(job):
        tag, cmd = job
        rc = _run(cmd, run_root / "selftest" / f"{tag}.log", threads="1")
        out = (run_root / "selftest" / f"{tag}.log").read_text()
        passed = '"PASS": true' in out and "P2 does not support" not in out and rc == 0
        miss = out.split('"miss": ')[1].split(",")[0] if '"miss": ' in out else "?"
        return tag, passed, miss

    with ThreadPoolExecutor(min(workers, len(jobs))) as ex:
        results = list(ex.map(one, jobs))
    for tag, passed, miss in results:
        print(f"{tag:50s} {'PASS' if passed else 'FAIL'} miss={miss}")
    if not all(p for _, p, _ in results):
        raise SystemExit("selftest failures")
    print(f"{len(results)} selftests pass")


PRICE = {"pi05": dict(look=0.152, call=0.848), "groot": dict(look=0.148, call=0.852)}   # owner IR prices (R8 report)


def arm_outcomes(run_root, arm, inits=EVAL_INITS):
    """SR and owner IR of one finished arm, from the client journal and the servers' decision logs.

    Records whose init is outside ``inits`` (default: the evaluation split 20-29) are dropped at parse time and never
    aggregated; the batch manifest only contains inits 20-29 anyway."""
    run = Path(run_root) / "runs" / arm
    model = "groot" if "groot" in arm else "pi05"
    ok = {}
    journal = run / "client" / "journal.jsonl"
    if not journal.exists():
        return None
    for line in journal.open():
        r = json.loads(line)
        if not r.get("accepted") or r.get("status") not in ("done", "failed") or r.get("error"):
            continue
        t, i = int(r["task_uid"].split(":")[-2]), int(r["task_uid"].split(":")[-1])
        if i in inits:
            ok[(t, i)] = bool(r.get("success"))
    vis, call = [], []
    for f in sorted(run.glob("server_*/decisions_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            if r.get("ev") != "dec":
                continue
            if int(r["uid"].split(":")[-1]) not in inits:
                continue
            vis.append(bool(r.get("vision")))
            call.append(not bool(r.get("hit", True)))
    if not ok:
        return None
    v = sum(vis) / len(vis) if vis else float("nan")
    c = sum(call) / len(call) if call else float("nan")
    return dict(arm=arm, n=len(ok), sr=sum(ok.values()) / len(ok), n_dec=len(vis), look=v, call=c,
                ir=PRICE[model]["look"] * v + PRICE[model]["call"] * c)


def score(run_root=R4, inits=EVAL_INITS):
    """Pre-registered scoring (PREDICTION_R4.md): H1 generalization, H2 escalation increment, H3 no harm."""
    run_root = Path(run_root)
    table, d1, d2, h3 = {}, [], [], []
    for model, suite, lib in CELLS:
        cn = cell_name(model, suite, lib)
        kinds = ["A", "onlynp", "np_corr05"] + (["np_corr05_esc"] if model == "pi05" else [])
        table[cn] = {k: arm_outcomes(run_root, f"r9f4_{cn}_{k}", inits) for k in kinds}
        t = table[cn]
        if t["np_corr05"] and t["onlynp"]:
            d1.append((t["np_corr05"]["sr"] - t["onlynp"]["sr"], t["np_corr05"]["ir"] - t["onlynp"]["ir"]))
        if model == "pi05" and t["np_corr05_esc"] and t["np_corr05"]:
            d2.append((t["np_corr05_esc"]["sr"] - t["np_corr05"]["sr"], t["np_corr05_esc"]["ir"] - t["np_corr05"]["ir"]))
        if t["np_corr05"] and t["A"]:
            h3.append(t["np_corr05"]["sr"] >= t["A"]["sr"] - 0.03)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    verdict = dict(
        H1=dict(complete=len(d1) == 6, d_sr=mean([x for x, _ in d1]), d_ir=mean([y for _, y in d1]),
                passed=len(d1) == 6 and mean([x for x, _ in d1]) >= 0.03 and mean([y for _, y in d1]) <= 0.005),
        H2=dict(complete=len(d2) == 3, d_sr=mean([x for x, _ in d2]), d_ir=mean([y for _, y in d2]),
                passed=len(d2) == 3 and mean([x for x, _ in d2]) >= 0.02 and mean([y for _, y in d2]) <= 0.012),
        H3=dict(complete=len(h3) == 6, passed=len(h3) == 6 and all(h3), cells_ok=sum(h3)))
    for cn, t in table.items():
        print(cn)
        for k, v in t.items():
            print(f"   {k:14s} " + ("(missing)" if v is None else f"n={v['n']:3d} SR {v['sr']:.3f} IR {v['ir']:.3f} (look {v['look']:.3f} call {v['call']:.3f})"))
    print(json.dumps(verdict, indent=1))
    (run_root / "score.json").write_text(json.dumps(dict(table=table, verdict=verdict, inits=[inits.start, inits.stop - 1]), indent=1))
    return table, verdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["spec", "prefit", "check", "selftest", "score"])
    ap.add_argument("--run-root", default=str(R4))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=4)
    a = ap.parse_args(argv)
    if a.action == "spec":
        write_spec(a.run_root)
    elif a.action == "prefit":
        prefit(a.run_root, a.workers)
    elif a.action == "check":
        check(a.run_root)
    elif a.action == "score":
        score(a.run_root)
    else:
        selftest(a.run_root, a.workers, a.episodes)


if __name__ == "__main__":
    main()
