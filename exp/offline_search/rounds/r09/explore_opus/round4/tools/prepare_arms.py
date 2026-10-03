"""Freeze round 4 in a new run root (nothing launched): /home/weiland/trace_runs/os_closed_loop/r09_opus_r4.

Per LIBERO-10 50-demo cell (pi0.5, GR00T), 100 pairs (tasks 0-9 x inits 20-29):
  r9o4_<m>_l10_50_cache          R8 pure-cache spec (frozen BlindAWM prefit)
  r9o4_<m>_l10_50_corr           fable's standalone corrector, exactly the r9f2_<m>_l10_50_corr05pt spec and fitted
                                 artifact (CorrectedCache, per-task heads fitted on inits 0-19, blend .5)
  r9o4_<m>_l10_50_corr_esc_w24   CorrectedEscalateCalls: that same fitted corrector as the cache + escalation window
                                 (lag 12, deadline 80, window 24; round-2 constants, not retuned)
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
from exp.offline_search.rounds.r09.explore_opus.round4.methods import CorrectedEscalateCalls

HERE = Path(__file__).resolve().parents[1]
RUN = RUNS / "r09_opus_r4"
M4 = "exp.offline_search.rounds.r09.explore_opus.round4.methods"
LAG, DEADLINE, WINDOW = 12, 80, 24


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec(root, arm):
    for s in json.loads((RUNS / root / "arms.json").read_text()):
        if s["arm"] == arm:
            return s
    raise KeyError(arm)


def fit_arg(s):
    a = s["plugin_args"]
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
         dict(role="DISCOVERY_EVAL_INITS_20_29", note="round-4 stack screen; corrector heads fitted on inits 0-19",
              selected=[dict(task=t, init=i) for t in range(10) for i in range(20, 30)]))
    rows, checks = [], []
    for model in ("pi05", "groot"):
        a_src = spec("r08_main", f"r8_{model}_l10_50_A")
        p_src = spec("r08_main", f"r8_{model}_l10_P10")
        c_src = spec("r09_fable_r2", f"r9f2_{model}_l10_50_corr05pt")
        cell = a_src["cell"]
        corr_fit = fit_arg(c_src)
        rows.append(row(a_src, f"r9o4_{model}_l10_50_cache"))
        rows.append(row(c_src, f"r9o4_{model}_l10_50_corr"))           # exact fable spec + fitted artifact
        kw = dict(corrected_fit=corr_fit, corrected_kwargs=c_src["kwargs"], lag_threshold=LAG, deadline=DEADLINE,
                  window=WINDOW, random_seed=26100401, coin_domain=f"R9O4/corr_escalate/{model}_l10_50")
        m = CorrectedEscalateCalls(**kw)
        m.prof = api.NULL_PROFILER
        m.fit(None, SimpleNamespace(cell=cell))
        name = f"r9o4_{model}_l10_50_corr_esc_w24"
        path = publish(art / f"{name}.pkl", m, M4 + ":CorrectedEscalateCalls", kw, cell,
                       dict(round4_methods_sha256=sha(HERE / "methods.py"), corrected_fit_sha256=sha(corr_fit),
                            design_inits="0-19", eval_inits="20-29"))
        p_args = [x for x in p_src["plugin_args"] if x != "--os-debug"]
        rows.append(row(a_src, name, M4 + ":CorrectedEscalateCalls", kw, path, p_args, True))
        checks.append(dict(arm=name, artifact=path, artifact_sha256=sha(path), corrected_fit=corr_fit,
                           corrected_fit_sha256=sha(corr_fit), corrector_head=c_src["kwargs"]["head_path"],
                           corrector_head_sha256=sha(c_src["kwargs"]["head_path"])))
    dump(HERE / "arms_in.json", rows)
    dump(HERE / "out" / "prepare_checks.json", checks)
    emit(["--run-root", str(RUN), "--spec", str(HERE / "arms_in.json")])
    print(json.dumps([r["name"] for r in rows]))


if __name__ == "__main__":
    main()
