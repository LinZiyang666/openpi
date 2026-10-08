"""R12 opus pilot: serving-host split (local 4090 = server_2310x, H100 = server_2323x) of every B-dev arm.

GR00T numerics differ between hosts, so a method arm is compared with deployed replicates served on the SAME host
(when any exist) as well as with the all-host pool. Paired-by-init dSR, dIR, calls/episode, calls in failing / winning
episodes, first call (any kind) paired by init over all episodes.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

from exp.offline_search.rounds.r12.opus.tools.pilot_analyse import CELLS, OUT, load_all, summarise, ir_of


def host(path):
    s = " ".join(os.path.basename(x) for x in glob.glob(path + "/server_*"))
    return "h100" if ("server_2323" in s or "server_2324" in s) else "local"


def pool(rs):
    n = sum(r["n"] for r in rs)
    ok, first, fail_calls, win_calls = defaultdict(list), defaultdict(list), [], []
    for r in rs:
        for k, e in r["eps"].items():
            ok[k].append(e["ok"])
            first[k].append(e["first_any"] if e["first_any"] is not None else e["last_step"] + 2)
            (win_calls if e["ok"] else fail_calls).append(e["calls"])
    return dict(n=n, reps=len(rs), sr=sum(r["sr"] * r["n"] for r in rs) / n, ir=sum(r["ir"] * r["n"] for r in rs) / n,
                calls_ep=sum(r["calls_ep"] * r["n"] for r in rs) / n, ok=ok, first=first,
                c_fail=float(np.mean(fail_calls)) if fail_calls else float("nan"), c_win=float(np.mean(win_calls)))


def compare(r, P):
    d = np.asarray([e["ok"] - np.mean(P["ok"][k]) for k, e in r["eps"].items() if k in P["ok"]], float)
    f = np.asarray([(e["first_any"] if e["first_any"] is not None else e["last_step"] + 2) - np.mean(P["first"][k])
                    for k, e in r["eps"].items() if k in P["first"]], float)
    cf = [e["calls"] for e in r["eps"].values() if not e["ok"]]
    cw = [e["calls"] for e in r["eps"].values() if e["ok"]]
    return dict(dSR=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), dIR=r["ir"] - P["ir"],
                dcalls=r["calls_ep"] / P["calls_ep"] - 1, c_fail=(float(np.mean(cf)) if cf else float("nan"), P["c_fail"]),
                c_win=(float(np.mean(cw)), P["c_win"]), dfirst=(float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))))


def main():
    arms = load_all()
    out = {}
    for cell in CELLS:
        cs = [a for a in arms if a["cell"] == cell]
        if not any(a["kind"] != "deployed" for a in cs):
            continue
        print(f"\n######## {cell}")
        S = []
        for a in cs:
            r = summarise(a)
            r["host"] = host(a["meta"]["path"])
            S.append(r)
            print(f"   {r['kind']:8s} {r['host']:5s} {a['name'][-52:]:52s} SR {r['sr']:.2f} IR {r['ir']:.4f} calls/ep {r['calls_ep']:5.2f}")
        dep = [r for r in S if r["kind"] == "deployed"]
        pools = {"all": pool(dep)}
        for h in ("local", "h100"):
            if any(r["host"] == h for r in dep):
                pools[h] = pool([r for r in dep if r["host"] == h])
        for k, P in pools.items():
            print(f"   pool {k:5s}: {P['reps']} reps {P['n']} eps SR {P['sr']:.3f} IR {P['ir']:.4f} calls/ep {P['calls_ep']:.2f} "
                  f"calls/fail ep {P['c_fail']:.1f} calls/win ep {P['c_win']:.2f}")
        out[cell] = {}
        for r in [r for r in S if r["kind"] != "deployed"]:
            res = {}
            for k in ("all", r["host"]):
                if k not in pools:
                    continue
                c = compare(r, pools[k])
                res[k] = c
                print(f"   {r['kind']} ({r['host']}) vs {k:5s}: dSR {c['dSR']:+.3f} ± {c['se']:.3f}  dIR {c['dIR']:+.4f}  calls/ep "
                      f"{c['dcalls']:+.1%}  calls/fail ep {c['c_fail'][0]:.1f} vs {c['c_fail'][1]:.1f}  calls/win ep "
                      f"{c['c_win'][0]:.2f} vs {c['c_win'][1]:.2f}  first call (paired, all eps) {c['dfirst'][0]:+.1f} ± {c['dfirst'][1]:.1f}")
            out[cell][r["name"]] = dict(host=r["host"], **res)
    (OUT / "pilot_host.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
