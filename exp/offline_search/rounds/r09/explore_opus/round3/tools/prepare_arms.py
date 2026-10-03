"""Freeze the round-3 screen in a new run root (nothing is launched).

Run root: /home/weiland/trace_runs/os_closed_loop/r09_opus_r3, manifest tasks 0-9 x inits 20-29 (100 pairs).
Per LIBERO-10 50-demo cell (pi0.5, GR00T):
  _cache       same-run pure cache control (R8 A arm spec)
  _esc_w24     round-2 escalation with a 24-decision policy window (the exact round-2 artifact class/kwargs)
  _home_w24    NEW: trigger -> scripted homing to the episode's start pose (gripper open) -> 24-decision policy window
  _home_cache  NEW ablation: trigger -> homing -> cache resumes (no policy call; IR ~ cache)
plus optional same-run pure-policy references.  Constants: lag 12, deadline 80 (chosen on inits 0-19 in round 2);
homing lift 5 cm, tolerance 3 cm, at most 8 fresh homing decisions, command cap .8 (fixed a priori, not tuned);
action maps fitted on inits 0-19 (tools/action_map.py).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from exp.offline_search.harness import api
from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from exp.offline_search.rounds.r09.explore_opus.round2.methods import EscalateCalls, publish
from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import RUNS, dump
from exp.offline_search.rounds.r09.explore_opus.round3.methods import HomingEscalation

HERE = Path(__file__).resolve().parents[1]
RUN = RUNS / "r09_opus_r3"
M2 = "exp.offline_search.rounds.r09.explore_opus.round2.methods"
M3 = "exp.offline_search.rounds.r09.explore_opus.round3.methods"
LAG, DEADLINE, WINDOW = 12, 80, 24


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def r8_spec(arm):
    for s in json.loads((RUNS / "r08_main" / "arms.json").read_text()):
        if s["arm"] == arm:
            return s
    raise KeyError(arm)


def fit_arg(spec):
    a = spec["plugin_args"]
    return a[a.index("--os-fit-artifact") + 1]


def row(src, name, method=None, kwargs=None, artifact=None, args=None, full_model=None):
    out = dict(name=name, model=src["model"], suite=src["suite_short"], mode="plugin", method=method or src["method"],
               kwargs=copy.deepcopy(kwargs if kwargs is not None else src["kwargs"]), cost_ledger=True,
               client_overrides=src.get("client_overrides"), manifest=str(RUN / "manifests" / "eval_inits20_29.json"))
    a = list(args if args is not None else [x for x in src["plugin_args"] if x != "--os-debug"])
    if artifact:
        a[a.index("--os-fit-artifact") + 1] = artifact
    out["plugin_args"] = a
    if (src.get("full_model", False) if full_model is None else full_model):
        out["full_model"] = True
    return out


def main():
    (RUN / "manifests").mkdir(parents=True, exist_ok=True)
    art = HERE / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    dump(RUN / "manifests" / "eval_inits20_29.json",
         dict(role="DISCOVERY_EVAL_INITS_20_29", note="round-3 screen; constants and action maps from inits 0-19",
              selected=[dict(task=t, init=i) for t in range(10) for i in range(20, 30)]))
    prov = dict(design_inits="0-19", eval_inits="20-29", round3_methods_sha256=sha(HERE / "methods.py"),
                round2_methods_sha256=sha(HERE.parent / "round2" / "methods.py"))
    rows, checks = [], []
    for model in ("pi05", "groot"):
        a_src, p_src = r8_spec(f"r8_{model}_l10_50_A"), r8_spec(f"r8_{model}_l10_P10")
        cell, base_fit = a_src["cell"], fit_arg(a_src)
        p_args = [x for x in p_src["plugin_args"] if x != "--os-debug"]
        rows.append(row(a_src, f"r9o3_{model}_l10_50_cache"))
        ekw = dict(lag_threshold=LAG, deadline=DEADLINE, base_kwargs=a_src["kwargs"], base_fit=base_fit,
                   random_seed=26100201, coin_domain=f"R9O/escalate/{model}_l10_50", window=WINDOW)
        em = EscalateCalls(**ekw)
        em.prof = api.NULL_PROFILER
        em.fit(None, SimpleNamespace(cell=cell))
        name = f"r9o3_{model}_l10_50_esc_w24"
        path = publish(art / f"{name}.pkl", em, M2 + ":EscalateCalls", ekw, cell, prov)
        rows.append(row(a_src, name, M2 + ":EscalateCalls", ekw, path, p_args, True))
        checks.append(dict(arm=name, artifact=path, base_sha256=sha(base_fit)))
        amap = str(HERE / "out" / f"action_map_{model}.json")
        for w, suffix in ((WINDOW, "home_w24"), (0, "home_cache")):
            hkw = dict(lag_threshold=LAG, deadline=DEADLINE, window=w, home_max=8, home_tol=0.03, lift=0.05, umax=0.8,
                       action_map=amap, base_kwargs=a_src["kwargs"], base_fit=base_fit, random_seed=26100301,
                       coin_domain=f"R9O3/homing/{model}_l10_50")
            hm = HomingEscalation(**hkw)
            hm.prof = api.NULL_PROFILER
            hm.fit(None, SimpleNamespace(cell=cell))
            name = f"r9o3_{model}_l10_50_{suffix}"
            path = publish(art / f"{name}.pkl", hm, M3 + ":HomingEscalation", hkw, cell,
                           dict(prov, action_map_sha256=sha(amap)))
            rows.append(row(a_src, name, M3 + ":HomingEscalation", hkw, path, p_args, True))
            checks.append(dict(arm=name, artifact=path, base_sha256=sha(base_fit), action_map_sha256=sha(amap)))
        rows.append(row(p_src, f"r9o3_{model}_l10_P10"))
    dump(HERE / "arms_in.json", rows)
    dump(HERE / "out" / "prepare_checks.json", checks)
    emit(["--run-root", str(RUN), "--spec", str(HERE / "arms_in.json")])
    print(json.dumps([r["name"] for r in rows]))


if __name__ == "__main__":
    main()
