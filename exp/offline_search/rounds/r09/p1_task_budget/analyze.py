"""Paired P1 contrasts vs CU and A using standard journals/server logs.

Default reads collected r09_p1 files and refuses incomplete/duplicate pairs.
--simulate-r8 exercises the same analysis on fable's prior dose simulator;
its output is explicitly labelled simulated, never a new experimental result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import uid_pair
from exp.offline_search.rounds.r09.explore_fable.tools import common as fable_common, paired

from .common import CELLS, DERIVED, HERE, RUN, VARIANTS, r8_name, write_json

PRICES = {"pi05": (.152, .848), "groot": (.148, .852)}
PAIRS = {(t, i) for t in range(10) for i in range(30)}


def jsonl(path):
    with Path(path).open() as f:
        for number, line in enumerate(f, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed collected JSONL {path}:{number}") from exc


def accepted_journals(rows, expected=PAIRS):
    by_uid = {}
    for r in rows:
        if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
            if uid_pair(r.get("task_uid", "")) in expected:
                by_uid[r["task_uid"]] = r
    pairs = [uid_pair(uid) for uid in by_uid]
    if len(pairs) != len(set(pairs)):
        raise ValueError("multiple accepted UIDs for one official pair")
    if set(pairs) != expected:
        raise ValueError(f"incomplete manifest: complete={len(pairs)} expected={len(expected)}")
    return by_uid


def standard_episode_table(run=RUN):
    rows = {r["arm"]: r for r in json.loads((Path(run) / "arms.json").read_text())}
    records = []
    for cell in CELLS:
        for variant in VARIANTS:
            arm = f"r9p1_{cell}_{variant}"
            root = Path(run) / "runs" / arm
            complete = accepted_journals(jsonl(root / "client/journal.jsonl"))
            paths = sorted(root.glob("server_*/decisions_*.jsonl"))
            if not paths:
                raise FileNotFoundError("no collected server decisions: " + arm)
            decisions = {}
            for path in paths:
                for d in jsonl(path):
                    uid = d.get("uid")
                    if d.get("ev") != "dec" or uid not in complete:
                        continue
                    if int(d.get("attempt", 1) or 1) != int(complete[uid].get("attempt", 1) or 1):
                        continue
                    key = (uid, int(d["step"]))
                    # Duplicate collection of one decision is harmless; conflicting
                    # accepted decisions would bias the cost and must fail closed.
                    if key in decisions and decisions[key] != d:
                        raise ValueError("conflicting accepted server decision: " + str(key))
                    decisions[key] = d
            look_price, call_price = PRICES[rows[arm]["model"]]
            for uid, journal in complete.items():
                seq = sorted((step, d) for (u, step), d in decisions.items() if u == uid)
                if not seq or [s for s, _ in seq] != list(range(len(seq))):
                    raise ValueError("missing/discontinuous server decisions: " + uid)
                n_look, n_call = 0, 0
                for _, d in seq:
                    if type(d.get("vision")) is not bool or type(d.get("hit")) is not bool:
                        raise ValueError("standard decisions require explicit vision/hit booleans")
                    n_look += int(d["vision"])
                    n_call += int(not d["hit"])
                    if not d["hit"] and not d["vision"]:
                        raise ValueError("policy call without a look")
                task, init = uid_pair(uid)
                records.append(dict(cell=cell, variant=variant, task_id=task, init=init,
                    success=int(bool(journal.get("success"))), n_dec=len(seq),
                    cost=look_price*n_look + call_price*n_call, n_look=n_look, n_call=n_call))
    return pd.DataFrame(records)


def simulated_episode_table():
    table = json.loads((HERE / "task_tables.json").read_text())
    frames = []
    for cell in CELLS:
        source = {}
        for v in ("A", "CU", "P10"):
            df = pd.read_parquet(DERIVED / "episodes" / (r8_name(cell, v) + ".parquet"),
                                filters=[("init", ">=", 0), ("init", "<", 30)])
            source[v] = df.set_index(["task_id", "init"]).sort_index()
        for variant in VARIANTS:
            records = []
            for task, init in sorted(PAIRS):
                chosen = variant if variant in ("A", "CU") else (
                    "CU" if variant == "topk_cu" else "P10") if task in table["cells"][cell]["hard_tasks"] else "A"
                r = source[chosen].loc[(task, init)]
                records.append(dict(cell=cell, variant=variant, task_id=task, init=init,
                                    success=int(r.success), cost=float(r.cost), n_dec=int(r.n_dec)))
            frames.append(pd.DataFrame(records))
    return pd.concat(frames, ignore_index=True)


def validate_episode_table(df):
    required = {"cell", "variant", "task_id", "init", "success", "cost", "n_dec"}
    if not required <= set(df):
        raise ValueError("missing episode columns")
    if set(df.cell) != set(CELLS) or set(df.variant) != set(VARIANTS):
        raise ValueError("expected exactly the 12 P1 arms")
    if not np.isfinite(df[["success", "cost", "n_dec"]].to_numpy()).all():
        raise ValueError("nonfinite episode values")
    if not df.success.isin([0, 1]).all() or (df.cost < 0).any() or (df.n_dec <= 0).any():
        raise ValueError("invalid episode outcomes/cost/decision counts")
    if df.duplicated(["cell", "variant", "task_id", "init"]).any():
        raise ValueError("duplicate arm/pair")
    for key, group in df.groupby(["cell", "variant"]):
        if len(group) != 300 or set(zip(group.task_id, group.init)) != PAIRS:
            raise ValueError("incomplete discovery300 arm: " + str(key))


def contrast(m, treatment, reference, cell, n_boot=2000):
    s, c, n = paired.mixture(m, pd.Series(treatment, index=m.index))
    rs, rc, rn = paired.mixture(m, pd.Series(reference, index=m.index))
    task = m.index.get_level_values("task_id").to_numpy()
    boot = fable_common.task_strat_bootstrap(np.column_stack([s, c, n, rs, rc, rn]), task,
                                           n_boot=n_boot, seed=26093005, stat=np.sum)
    ir, rir = float(c.sum()/n.sum()), float(rc.sum()/rn.sum())
    wins, losses = int(((s == 1) & (rs == 0)).sum()), int(((s == 0) & (rs == 1)).sum())
    ci = lambda values: [float(x) for x in np.percentile(values, [2.5, 97.5])]
    ds = float((s-rs).mean())
    threshold = ds >= (0 if "spatial" in cell else .04) and ir <= (.6 if "spatial" in cell else .5)*rir
    return dict(treatment=treatment, reference=reference, n=len(s), sr=float(s.mean()), ir=ir,
                reference_sr=float(rs.mean()), reference_ir=rir, delta_sr=ds,
                delta_sr_ci=ci((boot[:, 0]-boot[:, 3])/len(s)),
                delta_ir=ir-rir, delta_ir_ci=ci(boot[:, 1]/boot[:, 2]-boot[:, 4]/boot[:, 5]),
                ir_ratio=ir/rir, ir_ratio_ci=ci((boot[:, 1]/boot[:, 2])/(boot[:, 4]/boot[:, 5])),
                wins=wins, losses=losses, mcnemar_exact_p=float(binomtest(wins, wins+losses).pvalue) if wins+losses else 1.,
                proposal_point_bar_vs_CU=threshold if reference == "CU" else None,
                bar_note="point-estimate feasibility only; McNemar/CIs reported separately")


def holm(pvalues):
    order = np.argsort(pvalues, kind="stable")
    out = np.empty(len(order))
    running = 0.
    for rank, idx in enumerate(order):
        running = max(running, (len(order)-rank)*pvalues[idx])
        out[idx] = min(1., running)
    return out.tolist()


def analyze(df, n_boot=2000, kind="measured"):
    validate_episode_table(df)
    if n_boot < 100:
        raise ValueError("at least 100 paired bootstrap draws")
    report = dict(kind=kind, manifest="discovery300: 10 tasks x inits 0-29", n_boot=n_boot,
                  owner_prices=PRICES, multiplicity="Holm over all 12 two-sided McNemar comparisons",
                  cells={})
    contrasts = []
    for cell in CELLS:
        group = df[df.cell == cell]
        matrix = group.pivot(index=["task_id", "init"], columns="variant", values=["success", "cost", "n_dec"])
        matrix.columns = [f"{v}__{k}" for k, v in matrix.columns]
        matrix = matrix.sort_index()
        comp = [contrast(matrix, t, r, cell, n_boot=n_boot) for t in ("topk_cu", "topk_p10") for r in ("CU", "A")]
        arms = {v: paired.summarize(matrix, pd.Series(v, index=matrix.index), n_boot=0) for v in VARIANTS}
        report["cells"][cell] = dict(arms=arms, comparisons=comp,
            per_task=paired.per_task_table(matrix, VARIANTS).reset_index().to_dict("records"))
        contrasts.extend(comp)
    for c, p in zip(contrasts, holm([r["mcnemar_exact_p"] for r in contrasts])):
        c["mcnemar_holm_p"] = p
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=RUN)
    ap.add_argument("--simulate-r8", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    df = simulated_episode_table() if args.simulate_r8 else standard_episode_table(args.run_root)
    kind = "prior R8 fixed-table dose simulation (discovery reused for ranking)" if args.simulate_r8 else "measured R9 P1"
    report = analyze(df, args.n_boot, kind)
    out = args.out or (HERE / "evidence/paired_simulator.json" if args.simulate_r8 else RUN / "paired_analysis.json")
    write_json(out, report)
    for cell, result in report["cells"].items():
        for c in result["comparisons"]:
            print(f"{cell} {c['treatment']} vs {c['reference']}: delta_SR={c['delta_sr']:+.6f} "
                  f"CI={c['delta_sr_ci']} IR_ratio={c['ir_ratio']:.6f} "
                  f"wins/losses={c['wins']}/{c['losses']} McNemar_p={c['mcnemar_exact_p']:.6g} "
                  f"Holm_p={c['mcnemar_holm_p']:.6g}")
    print(f"P1_PAIRED_OK kind={kind} arms=12 pairs_per_arm=300 report={out}")


if __name__ == "__main__":
    main()
