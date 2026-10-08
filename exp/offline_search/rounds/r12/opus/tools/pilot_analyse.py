"""R12 opus pilot analysis on the compact tables of tools/pilot_extract.py (B-dev only, no astra arm).

Per cell: method arms (M1 / H2 / M2) vs pooled deployed replicates of the same B-dev selection.
  - SR, IR (owner compute share), calls per episode / per look, one-camera share, completion reasons
  - stop-rule quantities: first call of ANY kind and first guard call in failing episodes, calls per failing and per
    winning episode, failing episodes with no call, share of calls in failing episodes
  - paired-by-init comparisons against the per-init deployed success rate / first-call step
  - H2/M2: the deployed no-progress rule evaluated on the method's own trajectory (recorded _noprog_span >= 2, which
    reproduces every deployed guard call exactly on the deployed arms) -> where the old guard would have called
    first and H2 had not yet called
Output: out/pilot_analysis.json and a text report on stdout.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "out"
PRICE = {"pi05": (.152, .848, .076), "groot": (.148, .852, .074)}
CELLS = ("pi05_l10_50_periodic_ir32", "groot_l10_50_periodic_ir32", "pi05_l10_50", "pi05_spatial_50",
         "pi05_spatial_200", "groot_spatial_50", "groot_l10_50")


def cell_of(arm):
    a = arm.replace("r11_devknob_", "").replace("r11_dev_", "").replace("r12_lh_", "").replace("r12_guard_", "")
    for key in CELLS:
        if ("_" + key + "_") in "_" + a + "_":
            return key
    return None


def load_all():
    arms = []
    for p in sorted(glob.glob(str(OUT / "pilot" / "*.npz"))):
        if p.endswith(".tmp.npz"):
            continue
        z = np.load(p, allow_pickle=False)
        m = json.loads(str(z["meta_json"]))
        d = {k: z[k] for k in z.files if k != "meta_json"}
        cell = cell_of(m["arm"])
        if cell is None:
            continue
        old_invalid = m["src"].startswith("r12_lookhalf_") and "_fix1_" not in m["src"]
        deployed = m["arm"].endswith("_off") or m["arm"].startswith("r11_dev") or old_invalid
        kind = "deployed" if deployed else ("M1" if "opus_m1" in m["arm"] else
                                            "H2" if m["arm"].endswith("_h2") else "M2" if m["arm"].endswith("_m2") else "?")
        arms.append(dict(meta=m, d=d, cell=cell, kind=kind, name=f"{m['src']}/{m['arm']}"))
    return arms


def episodes(arm):
    d, m = arm["d"], arm["meta"]
    by = defaultdict(list)
    for i, u in enumerate(d["uid"]):
        by[str(u)].append(i)
    eps = {}
    for u, ok in m["uids"].items():
        idx = sorted(by.get(u, []), key=lambda i: d["step"][i])
        task, init = (int(x) for x in u.split(":")[-2:])
        steps = d["step"][idx]
        called = ~d["hit"][idx]
        reason = d["reason"][idx]
        span = d["noprog_span"][idx]
        calls = steps[called]
        guard = steps[called & (reason == 4)]
        old = steps[span >= 2]                       # deployed rule on this trajectory
        eps[(task, init)] = dict(ok=bool(ok), n_looks=len(idx), calls=int(called.sum()),
                                 first_any=int(calls[0]) if len(calls) else None,
                                 first_guard=int(guard[0]) if len(guard) else None,
                                 first_old=int(old[0]) if len(old) else None,
                                 last_step=int(steps[-1]) if len(steps) else 0,
                                 one=int((d["ncam"][idx] == 1).sum()),
                                 h2_stall=int((d["h2_stall"][idx] & called).sum()),
                                 h2_riskb=int((d["h2_riskb"][idx] & called).sum()),
                                 old_flags=int((span >= 2).sum()),
                                 old_not_called=int(((span >= 2) & ~called).sum()),
                                 called_not_old=int((called & (span < 2) & (reason == 4)).sum()))
    return eps


def ir_of(arm):
    m, d = arm["meta"], arm["d"]
    v, c, half = PRICE[m["model"] or ("groot" if "groot" in arm["cell"] else "pi05")]
    if m.get("decisions") is None:
        return None
    if m.get("work") is not None:
        return m["work"] / m["decisions"]
    return (v * m["vision"] + c * m["misses"]) / m["decisions"]


def summarise(arm):
    eps = episodes(arm)
    d = arm["d"]
    n = len(eps)
    fail = [e for e in eps.values() if not e["ok"]]
    win = [e for e in eps.values() if e["ok"]]
    calls = sum(e["calls"] for e in eps.values())
    looks = len(d["step"])
    nonzero = looks - n                                         # non-step-0 looks
    creasons = defaultdict(int)
    for cr, nc, nm in zip(d["creason"], d["ncomp"], d["ncam"]):
        if nc:
            creasons[str(cr)] += 1
    first_any = [e["first_any"] for e in fail if e["first_any"] is not None]
    first_guard = [e["first_guard"] for e in fail if e["first_guard"] is not None]
    r = dict(name=arm["name"], kind=arm["kind"], cell=arm["cell"], n=n, sr=np.mean([e["ok"] for e in eps.values()]),
             ir=ir_of(arm), decisions=arm["meta"].get("decisions"), looks=looks,
             calls_ep=calls / n, calls_look=calls / looks,
             calls_dec=(calls / arm["meta"]["decisions"]) if arm["meta"].get("decisions") else None,
             one_share=float((d["ncam"] == 1).sum() / looks), one_share_nonzero=float((d["ncam"] == 1).sum() / nonzero),
             completions=dict(creasons),
             n_fail=len(fail), calls_fail_ep=np.mean([e["calls"] for e in fail]) if fail else None,
             calls_win_ep=np.mean([e["calls"] for e in win]) if win else None,
             share_calls_fail=(sum(e["calls"] for e in fail) / calls) if calls else None,
             fail_no_call=sum(e["first_any"] is None for e in fail),
             first_any_fail_mean=float(np.mean(first_any)) if first_any else None,
             first_any_fail_median=float(np.median(first_any)) if first_any else None,
             first_guard_fail_mean=float(np.mean(first_guard)) if first_guard else None,
             fail_no_guard=sum(e["first_guard"] is None for e in fail),
             first_any_all_mean=float(np.mean([e["first_any"] for e in eps.values() if e["first_any"] is not None])),
             eps=eps)
    if arm["kind"] in ("H2", "M2"):
        rows = []
        for key, e in eps.items():
            h, o = e["first_guard"], e["first_old"]
            if o is None:
                rel = "old_never"
            elif h is None:
                rel = "h2_never"
            elif h < o:
                rel = "h2_earlier"
            elif h == o:
                rel = "same"
            else:
                rel = "h2_later"
            rows.append((key, e["ok"], rel, (h - o) if (h is not None and o is not None) else None))
        r["old_vs_h2"] = rows
        r["h2_call_branch"] = dict(stall=int(sum(e["h2_stall"] for e in eps.values())),
                                   risk=int(sum(e["h2_riskb"] for e in eps.values())))
        r["old_flags_look"] = float(sum(e["old_flags"] for e in eps.values()) / looks)
    return r


def fmt(x, f="{:.3f}"):
    return "—" if x is None else f.format(x)


def main(argv=None):
    arms = load_all()
    res = defaultdict(dict)
    report = {}
    for cell in CELLS:
        cs = [a for a in arms if a["cell"] == cell]
        dep = [summarise(a) for a in cs if a["kind"] == "deployed"]
        met = [summarise(a) for a in cs if a["kind"] != "deployed"]
        if not dep:
            continue
        print(f"\n######## {cell}  deployed reps {len(dep)}")
        # pooled deployed per-init outcome / first call
        dep_init = defaultdict(list)
        dep_first = defaultdict(list)
        for r in dep:
            for k, e in r["eps"].items():
                dep_init[k].append(e["ok"])
                dep_first[k].append(e["first_any"] if e["first_any"] is not None else e["last_step"] + 2)
        hdr = (f"{'arm':58s} {'n':>3s} {'SR':>5s} {'IR':>6s} {'calls/ep':>8s} {'c/look':>6s} {'1cam':>5s} "
               f"{'nfail':>5s} {'c/fail':>6s} {'c/win':>6s} {'%cF':>5s} {'F0call':>6s} {'1stAnyF':>7s} {'med':>5s} {'1stGrdF':>7s}")
        print(hdr)
        for r in sorted(dep, key=lambda r: r["name"]) + sorted(met, key=lambda r: r["name"]):
            print(f"{(r['kind'][:3] + ' ' + r['name'])[:58]:58s} {r['n']:3d} {r['sr']:5.2f} {fmt(r['ir'], '{:.4f}'):>6s} "
                  f"{r['calls_ep']:8.2f} {r['calls_look']:6.3f} {r['one_share']:5.2f} {r['n_fail']:5d} "
                  f"{fmt(r['calls_fail_ep'], '{:.1f}'):>6s} {fmt(r['calls_win_ep'], '{:.2f}'):>6s} "
                  f"{fmt(r['share_calls_fail'], '{:.2f}'):>5s} {r['fail_no_call']:6d} {fmt(r['first_any_fail_mean'], '{:.1f}'):>7s} "
                  f"{fmt(r['first_any_fail_median'], '{:.0f}'):>5s} {fmt(r['first_guard_fail_mean'], '{:.1f}'):>7s}")
        # pooled deployed aggregate
        agg = {}
        for key in ("sr", "ir", "calls_ep", "calls_look", "calls_fail_ep", "calls_win_ep", "share_calls_fail",
                    "first_any_fail_mean", "first_guard_fail_mean"):
            vals = [(r[key], r["n"] if key in ("sr", "ir", "calls_ep", "calls_look") else r["n_fail"]) for r in dep
                    if r[key] is not None]
            agg[key] = float(sum(v * w for v, w in vals) / sum(w for _, w in vals)) if vals else None
        agg["fail_no_call_rate"] = float(sum(r["fail_no_call"] for r in dep) / max(sum(r["n_fail"] for r in dep), 1))
        agg["n"] = sum(r["n"] for r in dep)
        agg["n_fail"] = sum(r["n_fail"] for r in dep)
        print(f"{'DEPLOYED pooled':58s} {agg['n']:3d} {agg['sr']:5.3f} {fmt(agg['ir'], '{:.4f}'):>6s} {agg['calls_ep']:8.2f} "
              f"{agg['calls_look']:6.3f} {'':5s} {agg['n_fail']:5d} {fmt(agg['calls_fail_ep'], '{:.1f}'):>6s} "
              f"{fmt(agg['calls_win_ep'], '{:.2f}'):>6s} {fmt(agg['share_calls_fail'], '{:.2f}'):>5s} "
              f"{agg['fail_no_call_rate']:6.2f} {fmt(agg['first_any_fail_mean'], '{:.1f}'):>7s} {'':5s} "
              f"{fmt(agg['first_guard_fail_mean'], '{:.1f}'):>7s}")
        report[cell] = dict(deployed=agg, deployed_arms=[{k: v for k, v in r.items() if k not in ("eps", "old_vs_h2")}
                                                         for r in dep], methods={})
        for r in sorted(met, key=lambda r: r["name"]):
            # paired by init vs pooled deployed per-init rate
            diffs, dfirst, dfirst_fail = [], [], []
            for k, e in r["eps"].items():
                if k in dep_init:
                    diffs.append(e["ok"] - np.mean(dep_init[k]))
                    fa = e["first_any"] if e["first_any"] is not None else e["last_step"] + 2
                    dfirst.append(fa - np.mean(dep_first[k]))
                    if not e["ok"]:
                        dfirst_fail.append(fa - np.mean(dep_first[k]))
            diffs = np.asarray(diffs)
            pd_ = dict(n_paired=len(diffs), dSR=float(diffs.mean()), se=float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
                       dfirst_all=float(np.mean(dfirst)), dfirst_fail=float(np.mean(dfirst_fail)) if dfirst_fail else None,
                       n_fail_paired=len(dfirst_fail))
            d_unp = dict(dSR=r["sr"] - agg["sr"], dIR=(r["ir"] - agg["ir"]) if r["ir"] is not None else None,
                         dcalls_ep_rel=r["calls_ep"] / agg["calls_ep"] - 1, dcalls_look=r["calls_look"] - agg["calls_look"],
                         dfirst_any_fail=(r["first_any_fail_mean"] - agg["first_any_fail_mean"])
                         if r["first_any_fail_mean"] is not None else None,
                         dfirst_guard_fail=(r["first_guard_fail_mean"] - agg["first_guard_fail_mean"])
                         if r["first_guard_fail_mean"] is not None else None)
            print(f"   {r['kind']} {r['name'][-40:]:40s} paired dSR {pd_['dSR']:+.3f}±{1.96 * pd_['se']:.3f} (n {pd_['n_paired']}) "
                  f"| unpaired dSR {d_unp['dSR']:+.3f} dIR {fmt(d_unp['dIR'], '{:+.4f}')} calls/ep {d_unp['dcalls_ep_rel']:+.1%} "
                  f"c/look {d_unp['dcalls_look']:+.3f} | 1st call fail (any) {fmt(d_unp['dfirst_any_fail'], '{:+.1f}')} "
                  f"(guard {fmt(d_unp['dfirst_guard_fail'], '{:+.1f}')}) | paired-by-init 1st call: all {pd_['dfirst_all']:+.1f}, "
                  f"failing {fmt(pd_['dfirst_fail'], '{:+.1f}')} (n {pd_['n_fail_paired']})")
            if r["completions"]:
                print(f"      completions {r['completions']}  one-cam share {r['one_share']:.3f} (non-step-0 {r['one_share_nonzero']:.3f})")
            entry = {k: v for k, v in r.items() if k not in ("eps",)}
            entry.update(paired=pd_, unpaired=d_unp)
            if r["kind"] in ("H2", "M2"):
                rel = defaultdict(lambda: [0, 0])
                leads_fail = []
                for key, ok, rr, lead in r["old_vs_h2"]:
                    rel[rr][int(ok)] += 1
                    if not ok and rr in ("h2_later", "h2_never"):
                        e = r["eps"][key]
                        leads_fail.append((key, rr, lead, e["first_old"], e["first_guard"], e["last_step"],
                                           float(np.mean(dep_init[key])) if key in dep_init else None,
                                           len(dep_init.get(key, []))))
                print(f"      old rule vs H2 (first flag), [fail, win] episodes: {dict(rel)}; H2 call branches {r['h2_call_branch']}; "
                      f"old-rule flags/look on own trajectory {r['old_flags_look']:.3f}")
                for key, rr, lead, fo, fh, ls, pdep, nd in sorted(leads_fail, key=lambda x: x[0]):
                    print(f"         FAIL {key}: {rr:9s} old first {fo:3d}  H2 first {fmt(fh, '{:.0f}'):>3s}  last look {ls:3d}  "
                          f"deployed SR on this init {fmt(pdep, '{:.2f}')} ({nd} reps)")
                entry["old_vs_h2_counts"] = {k: v for k, v in rel.items()}
                entry["late_failures"] = leads_fail
            report[cell]["methods"][r["name"]] = entry
    (OUT / "pilot_analysis.json").write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
