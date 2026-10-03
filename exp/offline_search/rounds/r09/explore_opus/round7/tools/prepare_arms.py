"""Freeze round 7 in a new run root (nothing launched): /home/weiland/trace_runs/os_closed_loop/r09_opus_r7.

Per LIBERO-10 50-demo cell (pi0.5, GR00T), 100 pairs (tasks 0-9 x inits 20-29):
  r9o7_<m>_l10_50_stack   fable's leading r3c stack, exact spec + fitted artifact (same-batch control)
  r9o7_<m>_l10_50_C20     + ONLY the per-episode guard-call budget C = 20 (round-5 gate C, fitted on inits 0-19)
  r9o7_<m>_l10_50_C15     + budget C = 15 (second dose point)
Classes: round-5 GatedNpGraspEsc3 / GatedNpGraspStackGroot3 with pace_lag_max=None (gate P off).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from exp.offline_search.harness import api
from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from exp.offline_search.rounds.r09.explore_opus.round2.methods import publish
from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import RUNS, dump
from exp.offline_search.rounds.r09.explore_opus.round5 import methods as m5
from exp.offline_search.rounds.r09.explore_opus.round5.tools import OUT as OUT5

HERE = Path(__file__).resolve().parents[1]
RUN = RUNS / "r09_opus_r7"
M5_FILE = Path(__file__).resolve().parents[2] / "round5" / "methods.py"
M5_SHA = "f6499e51b7652ee9cf5042b968f24fa7cf60c4ba2a2e6b80ed3edd3389ed5de4"
M5 = "exp.offline_search.rounds.r09.explore_opus.round5.methods"
R3C = RUNS / "r09_fable_r3c"
FABLE_METHODS = Path("/home/weiland/projects/openpi/exp/offline_search/rounds/r09/explore_fable/round3/tools/methods.py")
FABLE_METHODS_SHA = "fa881a577f75430b57db38cccdec345ad4b0256fc9b8c002fcb2ecbdf06c57ec"
BUDGET_C = 20          # gate C threshold fitted on inits 0-19 (round 5)
BUDGET_C2 = 15         # second dose point
STACKS = {"pi05": ("r9f3c_pi05_l10_50_np_corr05_esc", "GatedNpGraspEsc3"),
          "groot": ("r9f3c_groot_l10_50_np_corr05", "GatedNpGraspStackGroot3")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec(root, arm):
    for s in json.loads((root / "arms.json").read_text()):
        if s["arm"] == arm:
            return s
    raise KeyError(arm)


def fit_arg(s):
    a = s["plugin_args"]
    return a[a.index("--os-fit-artifact") + 1]


def row(src, name, method=None, kwargs=None, artifact=None):
    out = dict(name=name, model=src["model"], suite=src["suite_short"], mode="plugin", method=method or src["method"],
               kwargs=copy.deepcopy(kwargs if kwargs is not None else src["kwargs"]), cost_ledger=True,
               client_overrides=copy.deepcopy(src.get("client_overrides")),
               manifest=str(RUN / "manifests" / "eval_inits20_29.json"), full_model=True)
    a = [x for x in src["plugin_args"] if x != "--os-debug"]
    if artifact:
        a[a.index("--os-fit-artifact") + 1] = artifact
    out["plugin_args"] = a
    return out


def variants():
    return [(f"C{BUDGET_C}", dict(pace_lag_max=None, call_budget=BUDGET_C)),
            (f"C{BUDGET_C2}", dict(pace_lag_max=None, call_budget=BUDGET_C2))]


def main():
    fitted = json.loads((OUT5 / "thresholds.json").read_text())
    if fitted["C"] != BUDGET_C:
        raise SystemExit("round-5 fitted budget differs from the frozen constant")
    if sha(M5_FILE) != M5_SHA:
        raise SystemExit("round-5 methods.py changed since round 5")
    if sha(FABLE_METHODS) != FABLE_METHODS_SHA:
        raise SystemExit("fable round-3 methods.py changed; the stacks are not the frozen r3c stacks")
    (RUN / "manifests").mkdir(parents=True, exist_ok=True)
    art = HERE / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    dump(RUN / "manifests" / "eval_inits20_29.json",
         dict(role="DISCOVERY_EVAL_INITS_20_29", note="round-7 guard-call budget only on the leading r3c stacks; budget fitted on inits 0-19",
              selected=[dict(task=t, init=i) for t in range(10) for i in range(20, 30)]))
    rows, checks = [], []
    for model, (src_arm, cls_name) in STACKS.items():
        src = spec(R3C, src_arm)
        stack_fit = fit_arg(src)
        rows.append(row(src, f"r9o7_{model}_l10_50_stack"))                 # exact r3c spec + fitted artifact
        checks.append(dict(arm=f"r9o7_{model}_l10_50_stack", artifact=stack_fit, artifact_sha256=sha(stack_fit),
                           source=f"r09_fable_r3c/{src_arm}"))
        cls = getattr(m5, cls_name)
        for tag, gate in variants():
            kw = dict(copy.deepcopy(src["kwargs"]), **gate, force_noprog_at=[])
            m = cls(**kw)
            m.prof = api.NULL_PROFILER
            m.fit(None, SimpleNamespace(cell=src["cell"]))
            name = f"r9o7_{model}_l10_50_{tag}"
            path = publish(art / f"{name}.pkl", m, f"{M5}:{cls_name}", kw, src["cell"],
                           dict(round5_methods_sha256=sha(M5_FILE), fable_round3_methods_sha256=FABLE_METHODS_SHA,
                                stack_source=f"r09_fable_r3c/{src_arm}", stack_artifact_sha256=sha(stack_fit),
                                onlynp_fit_sha256=sha(kw["onlynp_fit"]), corrected_fit_sha256=sha(kw["corrected_fit"]),
                                design_inits="0-19", eval_inits="20-29"))
            rows.append(row(src, name, f"{M5}:{cls_name}", kw, path))
            checks.append(dict(arm=name, artifact=path, artifact_sha256=sha(path), onlynp_fit=kw["onlynp_fit"],
                               onlynp_fit_sha256=sha(kw["onlynp_fit"]), corrected_fit=kw["corrected_fit"],
                               corrected_fit_sha256=sha(kw["corrected_fit"]), pace_lag_max=gate["pace_lag_max"],
                               call_budget=gate["call_budget"]))
    dump(HERE / "arms_in.json", rows)
    dump(HERE / "out" / "prepare_checks.json", checks)
    emit(["--run-root", str(RUN), "--spec", str(HERE / "arms_in.json")])
    print(json.dumps([r["name"] for r in rows]))


if __name__ == "__main__":
    main()
