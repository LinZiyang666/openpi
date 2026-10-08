"""Ledger stats (v, call rate, one-camera share, non-call looks per decision) for given summary roots."""
import glob, json, os, sys
PRICE = {"pi05": (.152, .848), "groot": (.148, .852)}
pats = sys.argv[1:]
for pat in pats:
    for p in sorted(glob.glob(pat)):
        s = json.load(open(p))
        c = s["cost_ledger"]; model = c["model"]; v0, m0 = PRICE[model]
        lh = s.get("lookhalf_cost_ledger") or (dict(s["cache_ablate_cost_ledger"], misses=0) if s.get("cache_ablate_cost_ledger") else None)
        dec = c["decisions"]; vis = c["vision_decisions"]; mis = c["misses"]; n = s["complete"]
        if lh:
            work = lh["work"]; one = lh["one_camera_looks"]
        else:
            work = v0 * vis + m0 * mis; one = 0
        print(f"{s['arm'][:58]:58s} n={n:4d} SR {s['sr']:.3f} IR {work/dec:.4f} dec/ep {dec/n:6.1f} v {vis/dec:.3f} m {mis/dec:.4f} "
              f"noncall-look/dec {(vis-mis)/dec:.3f} one-share {one/max(vis,1):.2f} calls/ep {mis/n:.2f}")
