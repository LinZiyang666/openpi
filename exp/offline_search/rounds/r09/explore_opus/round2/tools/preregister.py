"""Pre-registered predictions for the escalation screen (written before any screen result exists).

For each screen cell the exact-prefix simulator (tools/triggers.py) is applied to the real replicate episodes of the
base controller with the frozen rule (lag >= 12, deadline decision 80).  Two populations are reported separately:
FIT (inits 0-19, the inits the constants were chosen on) and EVAL (inits 20-29, the screen's inits; descriptive,
nothing is tuned on it).  The single unknown, the rescue probability r of a persistent policy takeover, is varied
over a prior range [.45, .65] taken from the cross-arm recovery regression on FIT inits (see DATA_ANALYSIS.md).
"""
import json
import numpy as np
from .common import OUT, FIT_INITS, EVAL_INITS, dump
from .triggers import load_cell, catalog, signals, first_trigger, simulate

RULE = ("lag", 12, 0)
DEADLINE = 80
RS = (0.45, 0.55, 0.65)
WINDOW = 24


def one(cell, family):
    eps = load_cell(cell, family)
    if not eps:
        return None
    cat = catalog(cell)
    sig = [signals(e, cat) for e in eps]
    trig = [first_trigger(e, s, RULE[0], RULE[1], RULE[2], DEADLINE) for e, s in zip(eps, sig)]
    out = dict(cell="_".join(map(str, cell)), family=family, n_episodes=len(eps))
    for name, inits in (("fit_0_19", FIT_INITS), ("eval_20_29", EVAL_INITS)):
        idx = [j for j, e in enumerate(eps) if e.init in inits]
        E = [eps[j] for j in idx]
        T = [trig[j] for j in idx]
        base = simulate(E, [None] * len(E), 0, cell[0], cell[1])
        res = dict(base_sr=base["sr"], base_ir=base["ir"], n=len(E),
                   trigger_rate=float(np.mean([t is not None for t in T])),
                   false_alarm_rate=float(np.mean([t is not None and e.success for e, t in zip(E, T)])))
        for r in RS:
            s = simulate(E, T, r, cell[0], cell[1])
            res[f"r{r}"] = dict(sr=s["sr"], ir=s["ir"])
            # bounded 24-decision takeover: success probability assumed r x .7 (share of partial-policy
            # recoveries completing within 24 decisions on FIT inits, see anatomy/recovery.json)
            w = simulate(E, T, 0.7 * r, cell[0], cell[1], window=WINDOW)
            res[f"w{WINDOW}_r{r}"] = dict(sr=w["sr"], ir=w["ir"])
        out[name] = res
    return out


def main():
    rows = []
    for cell in [("pi05", "l10", 50), ("groot", "l10", 50), ("pi05", "l10", 500), ("groot", "l10", 500)]:
        for fam in ("cache", "only_no_progress", "guards_B", "corrector"):
            r = one(cell, fam)
            if r:
                rows.append(r)
                f, e = r["fit_0_19"], r["eval_20_29"]
                print(f"{r['cell']:16s} {fam:17s} n={r['n_episodes']:5d} | FIT base {f['base_sr']:.3f}@{f['base_ir']:.3f} -> "
                      + " ".join(f"r{x}:{f[f'r{x}']['sr']:.3f}@{f[f'r{x}']['ir']:.3f}" for x in RS)
                      + f" | EVAL base {e['base_sr']:.3f}@{e['base_ir']:.3f} -> "
                      + " ".join(f"r{x}:{e[f'r{x}']['sr']:.3f}@{e[f'r{x}']['ir']:.3f}" for x in RS)
                      + f" trig {e['trigger_rate']:.2f} FA {e['false_alarm_rate']:.3f}"
                      + " | W24 " + " ".join(f"r{x}:{e[f'w{WINDOW}_r{x}']['sr']:.3f}@{e[f'w{WINDOW}_r{x}']['ir']:.3f}" for x in RS))
    dump(OUT / "preregistration.json", dict(rule=dict(signal=RULE[0], threshold=RULE[1], deadline=DEADLINE),
                                            rescue_prior=list(RS), cells=rows))


if __name__ == "__main__":
    main()
