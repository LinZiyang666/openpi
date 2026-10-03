"""Freeze round 6 in a new run root (nothing launched): /home/weiland/trace_runs/os_closed_loop/r09_opus_r6.

LIBERO-10, 50-demo library, 100 pairs (tasks 0-9 x inits 20-29):
  r9o6_pi05_l10_50_stack       fable's leading r3c pi0.5 stack (only-no-progress + corrector + pace escalation), exact
                               spec + fitted artifact (same-batch control)
  r9o6_pi05_l10_50_stack_X     + exhaustion takeover, alongside the pace escalation   (ExhaustNpGraspEsc3)
  r9o6_pi05_l10_50_npcorr_X    only-no-progress + corrector, escalation REPLACED by the takeover (ExhaustNpGraspStack3)
  r9o6_groot_l10_50_stack      fable's leading r3c GR00T stack (same-batch control)
  r9o6_groot_l10_50_stack_X    + exhaustion takeover (ExhaustNpGraspStackGroot3)
Trigger thresholds from ``out/exhaust.json`` (inits 0-19); window 24 decisions (round-2 constant).
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
from exp.offline_search.rounds.r09.explore_opus.round6 import methods as m6

HERE = Path(__file__).resolve().parents[1]
RUN = RUNS / "r09_opus_r6"
M6 = "exp.offline_search.rounds.r09.explore_opus.round6.methods"
R3C = RUNS / "r09_fable_r3c"
FABLE_METHODS = Path("/home/weiland/projects/openpi/exp/offline_search/rounds/r09/explore_fable/round3/tools/methods.py")
FABLE_METHODS_SHA = "fa881a577f75430b57db38cccdec345ad4b0256fc9b8c002fcb2ecbdf06c57ec"
END_ROWS, DWELL, WINDOW = 0, 2, 24


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


def takeover_kwargs(src_kwargs, drop_escalation=False):
    kw = copy.deepcopy(src_kwargs)
    if drop_escalation:
        kw.pop("lag_threshold")
        kw.pop("deadline")
    kw.update(end_rows=END_ROWS, dwell=DWELL, window=WINDOW, force_exhaust_at=[])
    return kw


# (arm suffix, r3c source arm, class name, drop escalation)
PLAN = {"pi05": [("stack_X", "r9f3c_pi05_l10_50_np_corr05_esc", "ExhaustNpGraspEsc3", False),
                 ("npcorr_X", "r9f3c_pi05_l10_50_np_corr05_esc", "ExhaustNpGraspStack3", True)],
        "groot": [("stack_X", "r9f3c_groot_l10_50_np_corr05", "ExhaustNpGraspStackGroot3", False)]}
CONTROL = {"pi05": "r9f3c_pi05_l10_50_np_corr05_esc", "groot": "r9f3c_groot_l10_50_np_corr05"}


def build_method(cls_name, kw, cell):
    m = getattr(m6, cls_name)(**kw)
    m.prof = api.NULL_PROFILER
    m.fit(None, SimpleNamespace(cell=cell))
    return m


def main():
    fitted = json.loads((HERE / "out" / "exhaust.json").read_text())
    if (fitted["end_rows"], fitted["dwell"]) != (END_ROWS, DWELL):
        raise SystemExit("exhaust.json thresholds differ from the frozen constants")
    if sha(FABLE_METHODS) != FABLE_METHODS_SHA:
        raise SystemExit("fable round-3 methods.py changed; the stacks are not the frozen r3c stacks")
    (RUN / "manifests").mkdir(parents=True, exist_ok=True)
    art = HERE / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    dump(RUN / "manifests" / "eval_inits20_29.json",
         dict(role="DISCOVERY_EVAL_INITS_20_29", note="round-6 exhaustion takeover on the leading r3c stacks; thresholds fitted on inits 0-19",
              selected=[dict(task=t, init=i) for t in range(10) for i in range(20, 30)]))
    rows, checks = [], []
    for model in ("pi05", "groot"):
        csrc = spec(R3C, CONTROL[model])
        rows.append(row(csrc, f"r9o6_{model}_l10_50_stack"))
        checks.append(dict(arm=f"r9o6_{model}_l10_50_stack", artifact=fit_arg(csrc), artifact_sha256=sha(fit_arg(csrc)),
                           source=f"r09_fable_r3c/{CONTROL[model]}"))
        for suffix, src_arm, cls_name, drop in PLAN[model]:
            src = spec(R3C, src_arm)
            kw = takeover_kwargs(src["kwargs"], drop)
            m = build_method(cls_name, kw, src["cell"])
            name = f"r9o6_{model}_l10_50_{suffix}"
            path = publish(art / f"{name}.pkl", m, f"{M6}:{cls_name}", kw, src["cell"],
                           dict(round6_methods_sha256=sha(HERE / "methods.py"), fable_round3_methods_sha256=FABLE_METHODS_SHA,
                                stack_source=f"r09_fable_r3c/{src_arm}", onlynp_fit_sha256=sha(kw["onlynp_fit"]),
                                corrected_fit_sha256=sha(kw["corrected_fit"]), design_inits="0-19", eval_inits="20-29"))
            rows.append(row(src, name, f"{M6}:{cls_name}", kw, path))
            checks.append(dict(arm=name, artifact=path, artifact_sha256=sha(path), cls=cls_name, escalation=not drop,
                               end_rows=END_ROWS, dwell=DWELL, window=WINDOW))
    dump(HERE / "arms_in.json", rows)
    dump(HERE / "out" / "prepare_checks.json", checks)
    emit(["--run-root", str(RUN), "--spec", str(HERE / "arms_in.json")])
    print(json.dumps([r["name"] for r in rows]))


if __name__ == "__main__":
    main()
