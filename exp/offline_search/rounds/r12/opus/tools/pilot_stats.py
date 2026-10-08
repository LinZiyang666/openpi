"""R12 opus pilot statistics on top of pilot_analyse (B-dev only, no astra arm).

1. First-call timing with uncertainty: per cell, method minus pooled deployed, (a) mean first call of any kind in
   failing episodes (unpaired, bootstrap over episodes), (b) paired by init over ALL episodes (first call, or the last
   look + 2 when an episode never calls), (c) the same restricted to the method's failing episodes.
2. Early-stall conditional comparison. T_old = first look at which the deployed no-progress rule fires on the arm's
   own trajectory (deployed arms: the first guard call; H2/M2: recorded _noprog_span >= 2; M1: its guard). Until T_old
   both systems run the same cache-only loop, so P(fail | T_old in bucket) compares "call at T_old" (deployed) with
   the method's own call policy on comparable episodes. Buckets: T_old <= 10, 12-30, > 30, never.
3. Pooled paired dSR per method family across knob-off cells (equal cell weights), with SE.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from exp.offline_search.rounds.r12.opus.tools.pilot_analyse import CELLS, OUT, load_all, summarise

RNG = np.random.default_rng(0)
BUCKETS = (("<=10", lambda t: t is not None and t <= 10), ("12-30", lambda t: t is not None and 10 < t <= 30),
           (">30", lambda t: t is not None and t > 30), ("never", lambda t: t is None))


def boot_diff(a, b, n=4000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    d = [RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for _ in range(n)]
    return float(np.std(d))


def t_old(e, kind):
    return e["first_old"] if kind in ("H2", "M2") else e["first_guard"]


def main():
    arms = load_all()
    out = {}
    fam = defaultdict(list)
    for cell in CELLS:
        cs = [a for a in arms if a["cell"] == cell]
        dep = [summarise(a) for a in cs if a["kind"] == "deployed"]
        met = [summarise(a) for a in cs if a["kind"] != "deployed"]
        if not dep or not met:
            continue
        print(f"\n######## {cell}")
        dep_fail_first = [e["first_any"] for r in dep for e in r["eps"].values() if not e["ok"] and e["first_any"] is not None]
        dep_first = defaultdict(list)
        dep_ok = defaultdict(list)
        dep_calls_fail = [e["calls"] for r in dep for e in r["eps"].values() if not e["ok"]]
        dep_calls_win = [e["calls"] for r in dep for e in r["eps"].values() if e["ok"]]
        for r in dep:
            for k, e in r["eps"].items():
                dep_first[k].append(e["first_any"] if e["first_any"] is not None else e["last_step"] + 2)
                dep_ok[k].append(e["ok"])
        # bucket table for deployed
        def buckets(rs, kind_of):
            tab = {}
            for name, f in BUCKETS:
                ep = [e for r in rs for e in r["eps"].values() if f(t_old(e, kind_of(r)))]
                # excess failure vs the init's pooled deployed failure rate
                ex = [(not e2["ok"]) - (1 - np.mean(dep_ok[k])) for r in rs for k, e2 in r["eps"].items()
                      if f(t_old(e2, kind_of(r))) and k in dep_ok]
                tab[name] = (len(ep), float(np.mean([not e["ok"] for e in ep])) if ep else float("nan"),
                             float(np.mean(ex)) if ex else float("nan"))
            return tab
        bd = buckets(dep, lambda r: "deployed")
        line = "  ".join(f"{k}: n {v[0]:3d} fail {v[1]:.2f}" for k, v in bd.items())
        print(f"  deployed pooled   T_old buckets  {line}")
        out[cell] = dict(deployed_buckets=bd, methods={})
        for r in sorted(met, key=lambda r: r["name"]):
            fail_first = [e["first_any"] for e in r["eps"].values() if not e["ok"] and e["first_any"] is not None]
            d_unp = np.mean(fail_first) - np.mean(dep_fail_first) if fail_first else float("nan")
            se_unp = boot_diff(fail_first, dep_fail_first)
            pa, pf = [], []
            for k, e in r["eps"].items():
                if k not in dep_first:
                    continue
                fa = e["first_any"] if e["first_any"] is not None else e["last_step"] + 2
                pa.append(fa - np.mean(dep_first[k]))
                if not e["ok"]:
                    pf.append(fa - np.mean(dep_first[k]))
            pa, pf = np.asarray(pa), np.asarray(pf)
            cf = [e["calls"] for e in r["eps"].values() if not e["ok"]]
            cw = [e["calls"] for e in r["eps"].values() if e["ok"]]
            bm = buckets([r], lambda rr: rr["kind"])
            paired_sr = np.asarray([e["ok"] - np.mean(dep_ok[k]) for k, e in r["eps"].items() if k in dep_ok], float)
            res = dict(first_any_fail_unpaired=(float(d_unp), se_unp),
                       first_paired_all=(float(pa.mean()), float(pa.std(ddof=1) / np.sqrt(len(pa)))),
                       first_paired_fail=(float(pf.mean()), float(pf.std(ddof=1) / np.sqrt(len(pf)))) if len(pf) > 1 else None,
                       calls_fail_ep=(float(np.mean(cf)) if cf else None, float(np.mean(dep_calls_fail))),
                       calls_win_ep=(float(np.mean(cw)), float(np.mean(dep_calls_win))),
                       buckets=bm, paired_dSR=(float(paired_sr.mean()), float(paired_sr.std(ddof=1) / np.sqrt(len(paired_sr)))),
                       n_fail=len(cf))
            out[cell]["methods"][r["name"]] = res
            if "periodic" not in cell:
                fam[r["kind"]].append((cell, res["paired_dSR"]))
            print(f"  {r['kind']} {r['name'].split('/')[-1]:44s} 1st call in failing eps (any): {d_unp:+5.1f} ± {se_unp:4.1f} | "
                  f"paired all eps {pa.mean():+5.1f} ± {res['first_paired_all'][1]:3.1f} | paired failing "
                  f"{(pf.mean() if len(pf) else float('nan')):+5.1f} ± {(res['first_paired_fail'][1] if res['first_paired_fail'] else float('nan')):3.1f} (n {len(pf)})")
            print(f"     calls/failing ep {res['calls_fail_ep'][0] if cf else float('nan'):5.1f} vs {res['calls_fail_ep'][1]:5.1f} | "
                  f"calls/winning ep {res['calls_win_ep'][0]:5.2f} vs {res['calls_win_ep'][1]:5.2f} | paired dSR "
                  f"{res['paired_dSR'][0]:+.3f} ± {res['paired_dSR'][1]:.3f}")
            line = "  ".join(f"{k}: n {v[0]:3d} fail {v[1]:.2f} (excess {v[2]:+.2f})" for k, v in bm.items())
            print(f"     T_old buckets  {line}")
            line = "  ".join(f"{k}: excess {v[2]:+.2f}" for k, v in bd.items())
            print(f"     (deployed       {line})")
    print("\n######## pooled paired dSR, knob-off cells, equal cell weights")
    pooled = {}
    for kind, rows in fam.items():
        m = np.mean([v[0] for _, v in rows])
        se = np.sqrt(np.sum([v[1] ** 2 for _, v in rows])) / len(rows)
        pooled[kind] = dict(cells=[c for c, _ in rows], dSR=float(m), se=float(se))
        print(f"  {kind}: {len(rows)} cells {[c for c, _ in rows]} dSR {m:+.4f} ± {se:.4f} (SE)")
    out["pooled"] = pooled
    (OUT / "pilot_stats.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
