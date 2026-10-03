"""Failure anatomy of the cache on long tasks (discovery inits 0-29 only), consolidated and reproducible.

Sections (each written to OUT/anatomy/<name>.json):
  rescue_break   per controller: rescue rate on pairs the cache ALWAYS fails (pooled replicates) and break rate on
                 pairs the cache ALWAYS solves -- for the corrector, guards, calls, pure policy and the R8
                 perturbation arms (look every 5, follow lottery, look-less, wrist variants, placebo shift)
  fragility      corrector / policy / guard break rate on always-solved pairs, stratified by how many R8
                 perturbation arms break the same pair
  failure_labels R8 forensic labels and sub-goals done of pure-cache failures, by pooled cache success bin
  script         when pure-cache episodes reach the end of the retrieved demonstrations (top-1 progress >= .9)
  backjumps      retrieval phase jumps / regressions by outcome (pi0.5; standard ledgers with top-1 rows)
  recovery       success after the lag>=12 trigger vs the share of policy calls after it (call arms, FIT inits)
  onset          forensic failure onset vs the lag>=12 trigger time (R8 pure-cache arms, FIT inits)
RULE 1: every input was filtered to inits 0-29 at parse time by catalog/episodes/r8ledger/forensics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import DERIVED, OUT, FIT_INITS, dump
from .replicates import load as rload, matrix
from .forensics import episodes as forensic_episodes
from .triggers import catalog, signals, first_trigger, episodes_from_standard, episodes_from_r8

CELLS = [(m, s, n) for m in ("pi05", "groot") for s in ("l10", "spatial") for n in (50, 500)]
PERT = ["A5", "FL", "SF1", "W10", "W5", "SW", "SHIFT"]
BINS = [-.01, 0.001, .5, .999, 1.01]
BIN_NAMES = ["p0", "low", "high", "p1"]


def pooled_cache_p(outc, cell):
    m, s, n = cell
    d = outc[(outc.model == m) & (outc.suite == s) & (outc.lib == n) & (outc.family == "cache")]
    return matrix(d).mean(1)


def rescue_break(arms, outc, catalog_outc):
    rows = []
    for cell in CELLS:
        p = pooled_cache_p(outc, cell)
        m, s, n = cell
        d = outc[(outc.model == m) & (outc.suite == s) & (outc.lib == n)]
        fams = {f: matrix(g).mean(1) for f, g in d.groupby("family") if f != "cache"}
        pre = f"r8_{m}_{s}_{n}_"
        for a in PERT:
            g = catalog_outc[(catalog_outc.root == "r08_main") & (catalog_outc.arm == pre + a)]
            if len(g):
                fams["perturb_" + a] = g.set_index(["task", "init"]).success.astype(float)
        for f, y in fams.items():
            y = y.reindex(p.index)
            p0, p1 = p == 0, p == 1
            rows.append(dict(cell=f"{m}_{s}_{n}", controller=f, pairs_p0=int(p0.sum()), pairs_p1=int(p1.sum()),
                             rescue_p0=float(y[p0].mean()) if p0.any() else None,
                             break_p1=float(1 - y[p1].mean()) if p1.any() else None,
                             sr=float(y.mean()), cache_sr=float(p.mean())))
    return rows


def fragility(outc, catalog_outc):
    out = []
    for cell in CELLS:
        m, s, n = cell
        p = pooled_cache_p(outc, cell)
        P1 = p[p == 1].index
        pre = f"r8_{m}_{s}_{n}_"
        fails, k = pd.Series(0., index=P1), 0
        for a in PERT:
            g = catalog_outc[(catalog_outc.root == "r08_main") & (catalog_outc.arm == pre + a)]
            if len(g):
                k += 1
                fails += (1 - g.set_index(["task", "init"]).success.reindex(P1).astype(float)).fillna(0)
        if not k:
            continue
        frag = fails / k
        bins = pd.cut(frag, [-.01, 0, .2, .5, 1.01], labels=["robust", "low", "mid", "fragile"])
        d = outc[(outc.model == m) & (outc.suite == s) & (outc.lib == n)]
        rec = dict(cell=f"{m}_{s}_{n}", perturbation_arms=k, p1_pairs=len(P1),
                   share_fragile_any=float((frag > 0).mean()))
        for f in ("corrector", "policy10", "guards_B", "only_no_progress"):
            if (d.family == f).any():
                y = matrix(d[d.family == f]).mean(1).reindex(P1)
                rec[f] = {str(b): dict(n=int((bins == b).sum()), break_rate=float((1 - y[bins == b]).mean()) if (bins == b).any() else None)
                          for b in bins.cat.categories}
        out.append(rec)
    return out


def failure_labels(outc):
    out = []
    for cell in CELLS:
        m, s, n = cell
        try:
            E = forensic_episodes(f"r8_{m}_{s}_{n}_A")
        except FileNotFoundError:
            continue
        p = pooled_cache_p(outc, cell)
        E["p"] = p.reindex(pd.MultiIndex.from_arrays([E.task_id.astype(int), E.init.astype(int)])).values
        E["bin"] = pd.cut(E.p, BINS, labels=BIN_NAMES)
        F = E[~E.success.astype(bool)]
        out.append(dict(cell=f"{m}_{s}_{n}", failures=len(F),
                        labels=pd.crosstab(F.label, F.bin).to_dict(),
                        subgoals_done=pd.crosstab(F.subgoals_done, F.bin).to_dict(),
                        onset_median_by_label=F.groupby("label").onset_control.median().dropna().to_dict()))
    return out


def standard_eps(arms, cell, family):
    """All replicate episodes of a family with retrieval rows (standard client ledgers for pi0.5, server-log
    ledgers for GR00T, R8 debug ledger as fallback) -- see triggers.load_cell."""
    from .triggers import load_cell
    return [(family, e) for e in load_cell(cell, family)]


def script_and_backjumps(arms, outc):
    out = []
    for cell in [c for c in CELLS if c[0] == "pi05"]:
        cat = catalog(cell)
        p = pooled_cache_p(outc, cell)
        prog, step = cat.progress.values, cat.step.values
        rec = []
        for _, e in standard_eps(arms, cell, "cache"):
            top = e.rows[:, 0]
            pr = prog[top]
            hi = np.flatnonzero(pr >= .9)
            ds = np.diff(step[top])
            runmax = np.maximum.accumulate(pr)
            rec.append(dict(task=e.task, init=e.init, success=e.success, reach90=len(hi) > 0,
                            t90=float(e.seq[hi[0]] * 5) if len(hi) else np.nan, frac_hi=float((pr >= .9).mean()),
                            back=int((ds <= -8).sum()), regress=bool(((runmax >= .55) & (pr <= .35)).any())))
        R = pd.DataFrame(rec)
        R["bin"] = pd.cut(p.reindex(pd.MultiIndex.from_frame(R[["task", "init"]])).values, BINS, labels=BIN_NAMES)
        g = R.groupby(["bin", "success"], observed=True).agg(n=("task", "size"), reach90=("reach90", "mean"),
                                                              t90_median=("t90", "median"), frac_looks_at_end=("frac_hi", "mean"),
                                                              backjumps=("back", "mean"), regressions=("regress", "mean"))
        out.append(dict(cell="_".join(map(str, cell)), table={f"{b}|{s}": v for (b, s), v in g.round(4).to_dict("index").items()}))
    return out


def recovery(arms, catalog_outc):
    out = []
    for cell, std_fams, r8 in [(("pi05", "l10", 50), ["guards_B", "only_no_progress", "uniform_calls", "stage_tilted_calls"],
                                ["r8_pi05_l10_50_IP", "r8_pi05_l10_50_O5a"]),
                               (("groot", "l10", 50), ["guards_B", "only_no_progress"],
                                ["r8_groot_l10_50_CU", "r8_groot_l10_50_IP", "r8_groot_l10_50_CT", "r8_groot_l10_50_O5a"]),
                               (("pi05", "l10", 500), ["guards_B", "only_no_progress", "uniform_calls", "stage_tilted_calls"], [])]:
        cat = catalog(cell)
        eps = []
        for f in std_fams:
            eps += standard_eps(arms, cell, f)
        for arm in r8:
            o = catalog_outc[(catalog_outc.root == "r08_main") & (catalog_outc.arm == arm)]
            eps += [(arm.split("_")[-1], e) for e in episodes_from_r8(DERIVED / "r8ledger" / f"{arm}.parquet",
                    {(int(x.task), int(x.init)): bool(x.success) for x in o.itertuples()})]
        rec = []
        for fam, e in eps:
            if e.init not in FIT_INITS:
                continue
            k = first_trigger(e, signals(e, cat), "lag", 12)
            if k is None:
                continue
            post = e.cost[k:k + 20]
            rec.append(dict(fam=fam, k=k * 5, success=e.success, share=float((post >= .9).mean()) if len(post) else np.nan,
                            to_end=e.n_dec - k))
        R = pd.DataFrame(rec)
        R["kbin"] = pd.cut(R.k, [0, 200, 250, 300, 350, 400, 530]).astype(str)
        out.append(dict(cell="_".join(map(str, cell)), n=len(R),
                        by_arm=R.groupby("fam").agg(n=("success", "size"), recovered=("success", "mean"),
                                                    policy_share=("share", "mean")).round(3).to_dict("index"),
                        by_trigger_time=R.groupby("kbin").success.agg(["size", "mean"]).round(3).to_dict("index"),
                        decisions_trigger_to_success=dict(median=float(R[R.success].to_end.median()),
                                                          q75=float(R[R.success].to_end.quantile(.75)))))
    return out


def onset(catalog_outc):
    out = []
    for cell in [("pi05", "l10", 50), ("groot", "l10", 50), ("pi05", "l10", 500), ("groot", "l10", 500)]:
        arm = f"r8_{cell[0]}_l10_{cell[2]}_A"
        cat = catalog(cell)
        o = catalog_outc[(catalog_outc.root == "r08_main") & (catalog_outc.arm == arm)]
        eps = episodes_from_r8(DERIVED / "r8ledger" / f"{arm}.parquet",
                               {(int(x.task), int(x.init)): bool(x.success) for x in o.itertuples()})
        F = forensic_episodes(arm).set_index(["task_id", "init"])
        rec = []
        for e in eps:
            if e.success or e.init not in FIT_INITS:
                continue
            k = first_trigger(e, signals(e, cat), "lag", 12)
            f = F.loc[(e.task, e.init)]
            rec.append(dict(label=f.label, onset=f.onset_control, trig=k * 5 if k is not None else np.nan))
        R = pd.DataFrame(rec)
        R["delay"] = R.trig - R.onset
        out.append(dict(cell="_".join(map(str, cell)), failures=len(R), trigger_rate=float(R.trig.notna().mean()),
                        onset_median=float(R.onset.median()), trigger_median=float(R.trig.median()),
                        delay_median=float(R.delay.median())))
    return out


def main():
    arms, outc = rload()
    cat_outc = pd.read_parquet(DERIVED / "catalog" / "outcomes.parquet")
    assert cat_outc["init"].max() < 30
    for name, fn in [("rescue_break", lambda: rescue_break(arms, outc, cat_outc)),
                     ("fragility", lambda: fragility(outc, cat_outc)),
                     ("failure_labels", lambda: failure_labels(outc)),
                     ("script_backjumps", lambda: script_and_backjumps(arms, outc)),
                     ("recovery", lambda: recovery(arms, cat_outc)),
                     ("onset", lambda: onset(cat_outc))]:
        res = fn()
        dump(OUT / "anatomy" / f"{name}.json", res)
        print("wrote", name, len(res))


if __name__ == "__main__":
    main()
