"""Emit arms_mx.json (emit_arms.py spec rows for the mixed HIT/MISS closed-loop arms, pi0.5 sp + l10, 50-episode
library) with tau0 per cell read from tables/tables_ac.json (the h = .5 / .7 mixture fixed points of the matching
offline variant, in confidence units). Names r3mx_p_<sp|l10>_<short> (<= 40 chars).

    taskset -c <cpus> .venv/bin/python exp/offline_search/rounds/r03/h3_judge/tools/make_arms.py
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parents[1]
TAB = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r03/h3_judge/tables/tables_ac.json")
JUDGE = "exp/offline_search/rounds/r03/h3_judge/judge.py:MixedJudge"
AWM = "exp/offline_search/rounds/r02/g1_awm/awm.py:AWM"
BASE = {"base": AWM, "base_kwargs": {"lib": "current", "kref": 5}}
V_G = "MXJ_g1_ev0__AWM_joint_cur_fcur_kr5"
V_EV = "MXJ_g1_evDG_b2_rm0p1__AWM_joint_cur_fcur_kr5"
KW_G = {**BASE, "guards": True, "events": "none"}
KW_EV = {**BASE, "guards": True, "events": "all", "burst": 2, "ret_margin": 0.1}
WINDOW = 1000


def main():
    tab = json.load(open(TAB))
    tau = {(t["variant"], t["ms"], t["h"]): t["tau0"] for t in tab["tau"]}
    rows = []
    for suite, ms in (("spatial", "pi05_spatial"), ("l10", "pi05_l10")):
        sh = "sp" if suite == "spatial" else "l10"
        specs = [("awm_h70", V_G, KW_G, 0.7), ("ev_h70", V_EV, KW_EV, 0.7), ("g", V_G, KW_G, None),
                 ("awm_h50", V_G, KW_G, 0.5), ("ev_h50", V_EV, KW_EV, 0.5)]
        for short, var, kw, h in specs:
            name = f"r3mx_p_{sh}_{short}"
            assert len(name) <= 40, name
            capped = h is not None and abs(tau[(var, ms, h)]) > 1e8      # guards alone exceed the 1-h MISS budget
            t0 = None if h is None else (tau[("MXJ_g0_ev0__AWM_joint_cur_fcur_kr5", ms, h)] if capped else tau[(var, ms, h)])
            judge = "guard_only" if h is None else f"quantile:{h}:{WINDOW}:{t0:.6f}"
            if h is None:
                note = "guard-only: the plugin forces MISS on extras os_force_miss (guards 1-4), hit rate emerges"
            else:
                note = (f"quantile controller target h={h}, window {WINDOW}, tau0 = offline F4 mixture fixed point of {var} "
                        f"on {ms} (confidence units = -pred_err); forced decisions count as -inf. ")
                if capped:
                    note += ("CAPPED: the forced share alone exceeds 1-h offline, the fixed point is -inf; tau0 = the V7-only "
                             "fixed point of this cell/h as a start, the controller will drift down. ")
                note += "Do NOT add --os-judge-burst: the wrapper implements the burst itself (reason 7)."
            rows.append({"name": name, "model": "pi05", "suite": suite, "mode": "plugin", "full_model": True,
                         "method": JUDGE, "kwargs": kw,
                         "plugin_args": ["--os-fit-artifact", f"<RUN>/fits/{name}.pkl", "--os-judge", judge],
                         "_offline_variant": var, "_target_h": h, "_tau0_capped": capped,
                         "_note": note})
    (HERE / "arms_mx.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps([{k: r[k] for k in ("name", "plugin_args")} for r in rows], indent=1))


if __name__ == "__main__":
    main()
