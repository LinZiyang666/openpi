"""Freeze the escalation screen: artifacts, arm specs, manifest (inits 20-29 only) and emitted run root.

Nothing is launched.  Run root: /home/weiland/trace_runs/os_closed_loop/r09_opus_escalation.
Design constants (chosen on discovery inits 0-19 only, see tools/triggers.py): lag threshold 12 decisions,
deadline decision 80 (400 controls) on LIBERO-10.  Evaluation manifest = tasks 0-9 x inits 20-29 (100 pairs),
disjoint from the inits used to choose the constants.  Base controllers (pure cache prefits, B-only-no-progress
prefits) were fitted on the libraries only (no test inits).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from exp.offline_search.harness import api
from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from ..methods import EscalateCalls, EscalateOnlyNP, EscalateOnlyNPGroot, publish
from .common import HERE, RUNS, dump

RUN = RUNS / "r09_opus_escalation"
MODULE = "exp.offline_search.rounds.r09.explore_opus.round2.methods"
LAG, DEADLINE, WINDOW = 12, 80, 24
CELLS = [("pi05", 50), ("groot", 50), ("pi05", 500), ("groot", 500)]
ONLYNP = {"pi05": ("r08_abl_t107", "r8abl_onlynp_p_l10_50"), "groot": ("r08_abl_t107", "r8abl_onlynp_g_l10_50")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def r8_spec(root, arm):
    for s in json.loads((RUNS / root / "arms.json").read_text()):
        if s["arm"] == arm:
            return s
    raise KeyError(arm)


def fit_arg(spec):
    a = spec["plugin_args"]
    return a[a.index("--os-fit-artifact") + 1]


def row(src, name, method=None, kwargs=None, artifact=None, extra_args=None, full_model=None):
    out = dict(name=name, model=src["model"], suite=src["suite_short"], mode="plugin",
               method=method or src["method"], kwargs=copy.deepcopy(kwargs if kwargs is not None else src["kwargs"]),
               cost_ledger=True, client_overrides=src.get("client_overrides"),
               manifest=str(RUN / "manifests" / "eval_inits20_29.json"))
    args = [a for a in src["plugin_args"] if a != "--os-debug"]
    if extra_args is not None:
        args = extra_args
    if artifact:
        args = list(args)
        args[args.index("--os-fit-artifact") + 1] = artifact
    out["plugin_args"] = args
    fm = src.get("full_model", False) if full_model is None else full_model
    if fm:
        out["full_model"] = True
    return out


def main(argv=None):
    (RUN / "manifests").mkdir(parents=True, exist_ok=True)
    art_dir = HERE / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    dump(RUN / "manifests" / "eval_inits20_29.json",
         dict(role="DISCOVERY_EVAL_INITS_20_29",
              note="Escalation constants (lag 12, deadline 80) were chosen on inits 0-19 only; evaluate on 20-29.",
              selected=[dict(task=t, init=i) for t in range(10) for i in range(20, 30)]))
    provenance = dict(design_inits="0-19", eval_inits="20-29", methods_sha256=sha(HERE / "methods.py"),
                      lag_threshold=LAG, deadline=DEADLINE, window_variant=WINDOW)
    rows, checks = [], []
    for model, lib in CELLS:
        a_src = r8_spec("r08_main", f"r8_{model}_l10_{lib}_A")
        p_src = r8_spec("r08_main", f"r8_{model}_l10_P10")
        cell = a_src["cell"]
        base_fit = fit_arg(a_src)
        # 1) same-topology pure-cache control (unchanged R8 A arm)
        rows.append(row(a_src, f"r9o_{model}_l10_{lib}_cache"))
        # 2) pure cache + persistent escalation
        kw = dict(lag_threshold=LAG, deadline=DEADLINE, base_kwargs=a_src["kwargs"], base_fit=base_fit,
                  random_seed=26100201, coin_domain=f"R9O/escalate/{model}_l10_{lib}")
        m = EscalateCalls(**kw)
        m.prof = api.NULL_PROFILER
        m.fit(None, SimpleNamespace(cell=cell))
        name = f"r9o_{model}_l10_{lib}_esc"
        path = publish(art_dir / f"{name}.pkl", m, MODULE + ":EscalateCalls", kw, cell, provenance)
        p_args = [x for x in p_src["plugin_args"] if x != "--os-debug"]
        rows.append(row(a_src, name, MODULE + ":EscalateCalls", kw, path, extra_args=p_args, full_model=True))
        checks.append(dict(arm=name, base_fit=base_fit, base_sha256=sha(base_fit), artifact=path))
        if lib == 50:
            # bounded takeover: calls only during WINDOW decisions after the trigger, then the cache resumes
            wkw = dict(kw, window=WINDOW)
            wm = EscalateCalls(**wkw)
            wm.prof = api.NULL_PROFILER
            wm.fit(None, SimpleNamespace(cell=cell))
            name = f"r9o_{model}_l10_{lib}_esc_w{WINDOW}"
            path = publish(art_dir / f"{name}.pkl", wm, MODULE + ":EscalateCalls", wkw, cell, provenance)
            rows.append(row(a_src, name, MODULE + ":EscalateCalls", wkw, path, extra_args=p_args, full_model=True))
            checks.append(dict(arm=name, base_fit=base_fit, base_sha256=sha(base_fit), artifact=path))
            root, arm = ONLYNP[model]
            b_src = r8_spec(root, arm)
            rows.append(row(b_src, f"r9o_{model}_l10_{lib}_onlynp"))
            cls = EscalateOnlyNP if model == "pi05" else EscalateOnlyNPGroot
            bkw = dict(onlynp_fit=fit_arg(b_src), lag_threshold=LAG, deadline=DEADLINE)
            bm = cls(**bkw)
            bm.prof = api.NULL_PROFILER
            bm.fit(None, SimpleNamespace(cell=cell))
            name = f"r9o_{model}_l10_{lib}_onlynp_esc"
            path = publish(art_dir / f"{name}.pkl", bm, f"{MODULE}:{cls.__name__}", bkw, cell, provenance)
            rows.append(row(b_src, name, f"{MODULE}:{cls.__name__}", bkw, path))
            checks.append(dict(arm=name, base_fit=fit_arg(b_src), base_sha256=sha(fit_arg(b_src)), artifact=path))
    # optional same-topology pure-policy references (P10), one per model
    for model in ("pi05", "groot"):
        p_src = r8_spec("r08_main", f"r8_{model}_l10_P10")
        rows.append(row(p_src, f"r9o_{model}_l10_P10"))
    spec_path = HERE / "arms_in.json"
    dump(spec_path, rows)
    dump(HERE / "out" / "prepare_checks.json", checks)
    emit(["--run-root", str(RUN), "--spec", str(spec_path)])
    print(json.dumps([r["name"] for r in rows], indent=1))


if __name__ == "__main__":
    main()
