"""Round 7 (fable): ``r09_fable_r7`` -- GR00T delay-free gated follow + pi0.5 Spatial-50 combined look arm, each with its
plain stack as same-batch control (NOT launched here).  Reuses the round-5 builder (prefit/check/selftest/score)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from exp.offline_search.rounds.r09.explore_fable.round5.tools import build_r5 as b5

R7 = Path(os.environ.get("R9F_R7_ROOT", "/home/weiland/trace_runs/os_closed_loop/r09_fable_r7"))
R4 = Path("/home/weiland/trace_runs/os_closed_loop/r09_fable_r4")
REG = "r7_kwargs.json"
M7 = "exp.offline_search.rounds.r09.explore_fable.round7.tools.methods"
GROOT_VARIANT = dict(wrist_gate="off", pace_lag=1, lag_jump=2.0, follow_blocks=1, follow_stage_gate=True)
PI05_SP_VARIANT = dict(wrist_gate="pace", pace_lag=1, follow_blocks=1, follow_stage_gate=True)


def _row(root, arm):
    return {a["arm"]: a for a in json.loads((Path(root) / "arms.json").read_text())}[arm]


def _control(src, name):
    kwargs = json.loads(json.dumps(src["kwargs"]))
    if kwargs.get("max_calls") != 0 or kwargs.get("force_trigger_at") != []:
        raise ValueError("control stack must be a production stack (max_calls 0, no forced triggers)")
    base_args = [a for a in src["plugin_args"] if a != "--os-fit-artifact" and not a.endswith(".pkl")]
    row = dict(name=name, model=src["model"], suite=src["suite_short"], mode="plugin", full_model=True, method=src["method"], kwargs=kwargs,
               cost_ledger=True, client_overrides=dict(src["client_overrides"]), plugin_args=base_args + ["--os-fit-artifact", src["plugin_args"][-1]])
    return row, kwargs, base_args


def build(run_root=R7):
    fits = Path(run_root) / "fits"
    arms, prefit, prov = [], {}, {}
    # GR00T L10-50: r3c stack control + delay-free follow
    g = _row(b5.R3C, "r9f3c_groot_l10_50_np_corr05")
    ctrl, kw, args = _control(g, "r9f7_groot_l10_50_np")
    arms.append(ctrl)
    name = "r9f7_groot_l10_50_np_fgp"
    gkw = {**kw, "stage_fit": "", "wrist_fit": "", **GROOT_VARIANT}
    prefit[name] = dict(method=f"{M7}:LookCostGrootPace", kwargs=gkw, cell="groot_l10_cache", model="groot", wrist=False)
    arms.append(dict(name=name, model="groot", suite="l10", mode="plugin", full_model=True, method=f"{M7}:LookCostGrootPace", kwargs=gkw,
                     cost_ledger=True, client_overrides=dict(g["client_overrides"]), plugin_args=args + ["--os-fit-artifact", str(fits / f"{name}.pkl")]))
    prov[ctrl["name"]] = dict(control_source=g["arm"], artifact=g["plugin_args"][-1], artifact_sha256=b5.sha(g["plugin_args"][-1]))
    # pi0.5 Spatial-50: r4 stack control + pace-wrist + gated follow
    p = _row(R4, "r9f4_pi05_spatial_50_np_corr05_esc")
    ctrl, kw, args = _control(p, "r9f7_pi05_sp_50_esc")
    arms.append(ctrl)
    name = "r9f7_pi05_sp_50_esc_wpace_fg"
    pkw = {**kw, "stage_fit": str(b5.R07_FITS / "stages_pi05_sp_50.pkl"), "wrist_fit": str(b5.R07_FITS / "wrist_pi05_sp_50.pkl"), **PI05_SP_VARIANT}
    prefit[name] = dict(method=f"{b5.M5}:LookCostEsc", kwargs=pkw, cell="pi05_spatial_cache", model="pi05", wrist=True)
    arms.append(dict(name=name, model="pi05", suite="spatial", mode="plugin", full_model=True, method=f"{b5.M5}:LookCostEsc", kwargs=pkw,
                     cost_ledger=True, client_overrides=dict(p["client_overrides"]), plugin_args=args + b5.CAMERA_ARGS + ["--os-fit-artifact", str(fits / f"{name}.pkl")]))
    prov[ctrl["name"]] = dict(control_source=p["arm"], artifact=p["plugin_args"][-1], artifact_sha256=b5.sha(p["plugin_args"][-1]))
    if len({a["name"] for a in arms}) != len(arms):
        raise ValueError("duplicate arm names")
    return arms, prefit, prov


def write_spec(run_root=R7):
    run_root = Path(run_root)
    arms, prefit, prov = build(run_root)
    for d in ("fits", "prefit_logs", "manifests", "selftest"):
        (run_root / d).mkdir(parents=True, exist_ok=True)
    (run_root / "arms_in.json").write_text(json.dumps(arms, indent=1))
    (run_root / REG).write_text(json.dumps(prefit, indent=1))
    (run_root / "provenance.json").write_text(json.dumps(prov, indent=1))
    manifest = json.loads(b5.MANIFEST_SRC.read_text())
    if {(int(t), int(i)) for t, i in manifest} != {(t, i) for t in range(10) for i in b5.EVAL_INITS}:
        raise ValueError("manifest is not tasks 0-9 x inits 20-29")
    shutil.copyfile(b5.MANIFEST_SRC, run_root / "manifests" / "eval100_inits20_29.json")
    print(f"{len(arms)} arms, {len(prefit)} prefits -> {run_root}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["spec", "prefit", "check", "selftest", "score"])
    ap.add_argument("--run-root", default=str(R7))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=4)
    a = ap.parse_args(argv)
    if a.action == "spec":
        write_spec(a.run_root)
    elif a.action == "prefit":
        b5.prefit(a.run_root, a.workers, reg_name=REG)
    elif a.action == "check":
        b5.check(a.run_root, reg_name=REG)
    elif a.action == "selftest":
        b5.selftest(a.run_root, a.workers, a.episodes, reg_name=REG)
    else:
        b5.score(a.run_root)


if __name__ == "__main__":
    main()
