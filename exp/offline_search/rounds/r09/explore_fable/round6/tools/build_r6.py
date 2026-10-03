"""Round 6 (fable): one combined pi0.5 look arm (pace-gated wrist looks + gated follow) on top of the r3c stack, plus the
plain stack as same-batch control, in ``r09_fable_r6`` (NOT launched here).  Reuses round-5's builder; no new classes."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from exp.offline_search.rounds.r09.explore_fable.round5.tools import build_r5 as b5

R6 = Path(os.environ.get("R9F_R6_ROOT", "/home/weiland/trace_runs/os_closed_loop/r09_fable_r6"))
REG = "r6_kwargs.json"
VARIANT = dict(wrist_gate="pace", pace_lag=1, follow_blocks=1, follow_stage_gate=True)


def build(run_root=R6):
    fits = Path(run_root) / "fits"
    src = b5._control_row("pi05")
    stack_kwargs = json.loads(json.dumps(src["kwargs"]))
    if stack_kwargs.get("max_calls") != 0 or stack_kwargs.get("force_trigger_at") != []:
        raise ValueError("control stack must be the production r3c stack")
    base_args = [a for a in src["plugin_args"] if a != "--os-fit-artifact" and not a.endswith(".pkl")]
    ctrl = dict(name="r9f6_pi05_l10_50_esc", model="pi05", suite="l10", mode="plugin", full_model=True, method=src["method"],
                kwargs=stack_kwargs, cost_ledger=True, client_overrides=dict(src["client_overrides"]),
                plugin_args=base_args + ["--os-fit-artifact", src["plugin_args"][-1]])
    name = "r9f6_pi05_l10_50_esc_wpace_fg"
    kw = {**stack_kwargs, "wrist_gate": "off", "pace_lag": 1, "follow_blocks": 0, "follow_stage_gate": True,
          "stage_fit": str(b5.R07_FITS / "stages_pi05_l10_50.pkl"), "wrist_fit": str(b5.R07_FITS / "wrist_pi05_l10_50.pkl"), **VARIANT}
    method = f"{b5.M5}:LookCostEsc"
    arm = dict(name=name, model="pi05", suite="l10", mode="plugin", full_model=True, method=method, kwargs=kw, cost_ledger=True,
               client_overrides=dict(src["client_overrides"]), plugin_args=base_args + b5.CAMERA_ARGS + ["--os-fit-artifact", str(fits / f"{name}.pkl")])
    prefit = {name: dict(method=method, kwargs=kw, cell="pi05_l10_cache", model="pi05", wrist=True)}
    prov = {ctrl["name"]: dict(control_source=src["arm"], artifact=src["plugin_args"][-1], artifact_sha256=b5.sha(src["plugin_args"][-1]))}
    return [ctrl, arm], prefit, prov


def write_spec(run_root=R6):
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
    ap.add_argument("--run-root", default=str(R6))
    ap.add_argument("--workers", type=int, default=2)
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
