"""Build the R9Recipe artifacts for the 8 LIBERO cells and emit the equivalence arms (nothing launched).

Run root: /home/weiland/trace_runs/os_closed_loop/r09_recipe_eq, manifest tasks 0-9 x inits 20-29.
Each arm ``r9eq_<model>_<suite>_<lib>`` serves R9Recipe (budget off) with the plugin args, yaml source and client
overrides of the reference arm it must reproduce (cell defaults, update of 2026-10-02):
  50-demo cells          -> fable's leading stacks (r3c for LIBERO-10, r4 for Spatial)
  LIBERO-10-500 (both)   -> fable's r4 stacks r9f4_<m>_l10_500_np_corr05 (only-no-progress + corrector, no escalation)
  Spatial-500 (both)     -> the plain cache A rows
plus two optional (non-default) look-saving arms: ``r9eq_pi05_l10_500_look`` (r9f8_pi05_l10_500_wpace_fg) and
``r9eq_pi05_spatial_500_look`` (r9f8_pi05_spatial_500_wpace_fg).  Look saving is default OFF everywhere (update of
2026-10-02: the full-500 closed loop showed it costs success).
The recipe kwargs are read from the reference artifacts themselves (judge = the stack's only-no-progress artifact;
base / head / blend = the stack's corrected-cache artifact kwargs; escalation iff the reference is the escalation
stack), so the recipe sees exactly the same frozen inputs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r09.recipe.recipe import R9Recipe

HERE = Path(__file__).resolve().parents[1]
RUNS = Path("/home/weiland/trace_runs/os_closed_loop")
RUN = RUNS / "r09_recipe_eq"
SPEC = "exp.offline_search.rounds.r09.recipe.recipe:R9Recipe"
# cell -> (reference run root, reference arm)
REFS = {
    "pi05_l10_50": ("r09_fable_r3c", "r9f3c_pi05_l10_50_np_corr05_esc"),
    "groot_l10_50": ("r09_fable_r3c", "r9f3c_groot_l10_50_np_corr05"),
    "pi05_spatial_50": ("r09_fable_r4", "r9f4_pi05_spatial_50_np_corr05"),
    "groot_spatial_50": ("r09_fable_r4_g", "r9f4_groot_spatial_50_np_corr05"),
    "pi05_l10_500": ("r09_fable_r4", "r9f4_pi05_l10_500_np_corr05"),
    "pi05_spatial_500": ("r09_fable_r4", "r9f4_pi05_spatial_500_A"),
    "groot_l10_500": ("r09_fable_r4_g", "r9f4_groot_l10_500_np_corr05"),
    "groot_spatial_500": ("r09_fable_r4_g", "r9f4_groot_spatial_500_A"),
    # optional (non-default) settings
    "pi05_l10_500_look": ("r09_fable_r8", "r9f8_pi05_l10_500_wpace_fg"),
    "pi05_spatial_500_look": ("r09_fable_r8", "r9f8_pi05_spatial_500_wpace_fg"),
}
DEFAULT_CELLS = ("pi05_l10_50", "groot_l10_50", "pi05_spatial_50", "groot_spatial_50", "pi05_l10_500", "pi05_spatial_500",
                 "groot_l10_500", "groot_spatial_500")
ESC_CLASS = "NpGraspEsc3"
STACK_CLASSES = ("NpGraspEsc3", "NpGraspStack3", "NpGraspStackGroot3")
FOLLOW_BASE = "exp.offline_search.rounds.r07.c1_follow.methods:StageFollow"
STORE = "/home/weiland/trace_runs/offline_search_store"
REPO = Path("/home/weiland/projects/openpi")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref_row(cell):
    root, arm = REFS[cell]
    for s in json.loads((RUNS / root / "arms.json").read_text()):
        if s["arm"] == arm:
            return s
    raise KeyError(arm)


def fit_arg(row):
    a = row["plugin_args"]
    return a[a.index("--os-fit-artifact") + 1]


def recipe_kwargs(cell, **override):
    """Recipe kwargs from the reference arm's own frozen inputs."""
    ref = ref_row(cell)
    cls = ref["method"].split(":")[-1]
    if cls == "BlindAWM":
        kw = dict(mode="cache", base_fit=fit_arg(ref))
    elif cls == "PaceWrist":
        rk = ref["kwargs"]
        if rk.get("base_spec") != FOLLOW_BASE or not rk.get("enabled", True):
            raise ValueError(f"{cell}: unexpected PaceWrist reference")
        kw = dict(mode="pace_wrist", base_fit=rk["base_fit"], follow_kwargs=rk["base_kwargs"], wrist_fit=rk["wrist_fit"],
                  stage_fit=rk["stage_fit"], pace_lag=rk["pace_lag"])
    else:
        if cls not in STACK_CLASSES:
            raise ValueError(f"{cell}: unexpected reference class {cls}")
        rk = ref["kwargs"]
        if rk.get("max_calls", 0) != 0:
            raise ValueError(f"{cell}: reference stack has empty-grasp calls on")
        with open(rk["corrected_fit"], "rb") as f:
            cblob = FitUnpickler(f).load()       # (reads fable's corrected artifact; build time only)
        ck = cblob["kwargs"]
        if ck.get("correct_gripper"):
            raise ValueError("gripper correction is not part of the recipe")
        esc = ref["method"].split(":")[-1] == ESC_CLASS
        kw = dict(mode="stack", judge_fit=rk["onlynp_fit"], base_fit=ck["base_fit"], head_path=ck["head_path"],
                  blend=float(ck["blend"]), escalation=esc, lag_threshold=rk.get("lag_threshold", 12),
                  deadline=rk.get("deadline", 80), call_budget=None)
    kw.update(override)
    return kw


def prefit(cell, kw, out):
    """The plugin's own prefit path (real library + context); used for pace_wrist (R7 StageWrist.fit needs them)."""
    import subprocess
    ref = ref_row(cell)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logdir = HERE / "out" / "prefit_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    cmd = ["taskset", "-c", "2-9,46-53", str(REPO / ".venv/bin/python"), "-m", "exp.offline_search.closed_loop.plugin",
           "--os-method", SPEC, "--os-kwargs", json.dumps(kw), "--os-cell", ref["cell"], "--os-root", STORE,
           "--os-log-dir", str(logdir), "--os-fit-artifact", str(out)]
    env = dict(PYTHONPATH=".:src", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2", PATH="/usr/bin:/bin", HOME="/home/weiland")
    with open(logdir / f"{out.stem}.log", "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO), env=env).returncode
    if rc != 0:
        raise SystemExit(f"prefit failed for {cell} (rc={rc}); see {logdir / (out.stem + '.log')}")
    return str(out)


def build(cell, kw, out):
    if kw["mode"] == "pace_wrist":
        return prefit(cell, kw, out)
    ref = ref_row(cell)
    m = R9Recipe(**kw)
    m.prof = api.NULL_PROFILER
    m.fit(None, SimpleNamespace(cell=ref["cell"]))
    blob = dict(method=m, registered={}, spec=SPEC, kwargs=kw, cell=ref["cell"], fit_s=0.0,
                provenance=dict(reference=f"{REFS[cell][0]}/{REFS[cell][1]}", reference_artifact=fit_arg(ref),
                                reference_artifact_sha256=sha(fit_arg(ref)), recipe_sha256=sha(HERE / "recipe.py")))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(blob, f, protocol=4)
    return str(out)


def arm_row(cell, name, kw, artifact):
    ref = ref_row(cell)
    a = [x for x in ref["plugin_args"] if x != "--os-debug"]
    a[a.index("--os-fit-artifact") + 1] = artifact
    row = dict(name=name, model=ref["model"], suite=ref["suite_short"], mode="plugin", method=SPEC, kwargs=kw,
               cost_ledger=True, client_overrides=copy.deepcopy(ref.get("client_overrides")),
               manifest=str(RUN / "manifests" / "eval_inits20_29.json"), plugin_args=a)
    if ref.get("full_model"):
        row["full_model"] = True
    return row


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated cells to (re)build; the others must already exist unchanged")
    a = ap.parse_args(argv)
    only = set(a.only.split(",")) if a.only else set(REFS)
    (RUN / "manifests").mkdir(parents=True, exist_ok=True)
    (RUN / "manifests" / "eval_inits20_29.json").write_text(json.dumps(
        dict(role="DISCOVERY_EVAL_INITS_20_29", note="R9Recipe closed-loop identity check against the r3c / r4 references",
             selected=[dict(task=t, init=i) for t in range(10) for i in range(20, 30)]), indent=1))
    rows, checks = [], []
    for cell in REFS:
        kw = recipe_kwargs(cell)
        name = f"r9eq_{cell}"
        path = HERE / "artifacts" / f"{name}.pkl"
        if cell in only:
            if path.exists():
                path.unlink()                       # rebuilt from the frozen inputs below
            art = build(cell, kw, path)
        else:
            with open(path, "rb") as f:
                blob = pickle.load(f)
            if blob["kwargs"] != kw or blob["spec"] != SPEC:
                raise SystemExit(f"{name}: existing artifact does not match the current recipe kwargs; rebuild it")
            art = str(path)
        rows.append(arm_row(cell, name, kw, art))
        checks.append(dict(arm=name, cell=cell, reference=f"{REFS[cell][0]}/{REFS[cell][1]}", artifact=art,
                           artifact_sha256=sha(art), reference_artifact=fit_arg(ref_row(cell)),
                           reference_artifact_sha256=sha(fit_arg(ref_row(cell)))))
    (HERE / "arms_in.json").write_text(json.dumps(rows, indent=1))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "build_checks.json").write_text(json.dumps(checks, indent=1))
    emit(["--run-root", str(RUN), "--spec", str(HERE / "arms_in.json")])
    print(json.dumps([r["name"] for r in rows]))


if __name__ == "__main__":
    main()
