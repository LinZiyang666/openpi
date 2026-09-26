"""Drift quantification in the stale regime (state-window distance vs err), error-share decomposition of the best
methods (transitions / failed episodes / late third), and the optimistic mix bound (stale restricted to successful
cache episodes)."""
import sys, json
sys.path.insert(0, "/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r01")
import numpy as np, pandas as pd
from h import *

print("## 1. state-window distance of the query to its nearest library window (M1 x_w0 / s_w; s_w = library median 1-NN)")
print("   stale cells: share of decisions with w0/s_w > 3 (far from any library state) and their err; fresh cells use x_d0/s_d")
for c in CELLS:
    d = load("M1_big_a0p25_kmean5_stWin_b0", c)
    j = cell_json("M1_big_a0p25_kmean5_stWin_b0", c)
    s1 = d["step"] >= 1
    if c.endswith("cache"):
        # conf = -w0/s_w - disp/s_a  ->  w0/s_w = -(conf + disp/s_a); recover s_w from a decision: not stored; use ratio via scales file
        pass
    # scales from scratch dir
    import glob, os
    sc = glob.glob(f"{rundir('M1_big_a0p25_kmean5_stWin_b0')}/scratch/{c}/*_scales.json")
    S = json.load(open(sc[0])) if sc else None
    if S is None:
        print("  no scales for", c); continue
    x = (d["x_w0"] / S["s_w"]) if c.endswith("cache") else (d["x_d0"] / S["s_d"])
    x = x[s1]; e = d["err"][s1]; g = d["grip_mis"][s1]
    eps = episodes(c); succ = np.array([ep["success"] for ep in eps])[d["ep"][s1]]
    far = x > 3
    print(f"  {c:20s} n={s1.sum():6d} p50 {np.median(x):.2f} p90 {np.percentile(x, 90):.2f}  far(>3): {far.mean():.2f} err_far {e[far].mean():.3f} err_near {e[~far].mean():.3f}"
          f"  | far share among failed-ep decisions {far[~succ].mean():.2f} vs success-ep {far[succ].mean():.2f}")

print("\n## 2. error-share decomposition (fraction of total summed err, step>=1) for the best method per regime")
for m, arm in (("M1_big_a0p25_kmean5_stWin_b0", "inf"), ("M8_b0big_k5_mean", "cache"), ("M2sw_big_m3_k8_mean", "cache")):
    print(f"### {m} ({arm})")
    for c in [x for x in CELLS if x.endswith(arm)]:
        d = load(m, c); s1 = d["step"] >= 1
        e = d["err"][s1]; tot = e.sum()
        eps = episodes(c); succ = np.array([ep["success"] for ep in eps])[d["ep"][s1]]
        # transition = teacher gripper sign changes within the executed 5 steps or vs previous executed step (breakdown's def is
        # metric-side; approximate here with grip_mis>0 OR a sign change in a_inf[:5,6]) -> use query store a_inf
        Q = f"{ROOT}/queries/{c}"
        ai = np.load(f"{Q}/a_inf.npy", mmap_mode="r")
        rows = d["row"][s1]
        g = np.sign(np.asarray(ai[rows, :5, 6]))
        prev_rows = rows - 1
        gp = np.sign(np.asarray(ai[np.maximum(prev_rows, 0), 4, 6]))
        trans = (g.min(1) != g.max(1)) | (g[:, 0] != gp)
        late = d["bin"][s1] == 2
        print(f"  {c:20s} err {e.mean():.3f} | share of total err: transitions {e[trans].sum()/tot:.2f} (n {trans.mean():.2f}, err {e[trans].mean():.3f}) "
              f"| failed-ep {e[~succ].sum()/tot:.2f} (n {(~succ).mean():.2f}, err {e[~succ].mean():.3f}) | late third {e[late].sum()/tot:.2f} (n {late.mean():.2f}) "
              f"| top-10% worst decisions {np.sort(e)[-len(e)//10:].sum()/tot:.2f}")

print("\n## 3. optimistic mix bound: stale err restricted to SUCCESSFUL cache-run episodes (states a better cache would keep visiting)")
FRESH = "M1_big_a0p25_kmean5_stWin_b0"
for ms in MS:
    ci, cc = f"{ms}_inf", f"{ms}_cache"
    F = load(FRESH, ci); ef = F["err"][F["step"] >= 1].mean()
    line = f"  {ms:13s} fresh {ef:.3f} |"
    for stale in ("M8_b0big_k5_mean", "M2sw_big_m3_k8_mean", "M7_cascade_pca32_both_med3", "M1_big_a0p25_kmean5_stWin_b0", "B0_current"):
        S = load(stale, cc); s1 = S["step"] >= 1
        eps = episodes(cc); succ = np.array([ep["success"] for ep in eps])[S["ep"]]
        es_all = S["err"][s1].mean(); es_ok = S["err"][s1 & succ].mean()
        line += f" {stale.split('_')[0]}: stale all {es_all:.3f} / succ-only {es_ok:.3f} -> mix@.5 {0.5*ef+0.5*es_all:.3f} / {0.5*ef+0.5*es_ok:.3f} |"
    print(line)

print("\n## 4. fresh regime: does the state term or continuity carry it? M1 alpha sweep err (step>=1, inf) and M2 (state only)")
for ms in MS:
    ci = f"{ms}_inf"
    vals = {}
    for m in ("M1_big_a0_med3_stWin_b0", "M1_big_a0p25_med3_stWin_b0", "M1_big_a1_med3_stWin_b0", "M2sw_big_m3_k5_med", "M2sw_big_m3_k5_mean"):
        d = load(m, ci); vals[m] = d["err"][d["step"] >= 1].mean()
    print(f"  {ci:18s} cont-only(a0,med3) {vals['M1_big_a0_med3_stWin_b0']:.3f}  a0.25 {vals['M1_big_a0p25_med3_stWin_b0']:.3f}  a1 {vals['M1_big_a1_med3_stWin_b0']:.3f}  state-only(m3,k5 med) {vals['M2sw_big_m3_k5_med']:.3f} (k5 mean {vals['M2sw_big_m3_k5_mean']:.3f})")

print("\n## 5. Rtail in cache cells = 'keep executing the previous library chunk' vs B0 re-retrieval (step>=1)")
for ms in MS:
    cc = f"{ms}_cache"
    r = load("Rtail_passthrough", cc); b = load("B0_current", cc); m8 = load("M8_b0big_k5_mean", cc)
    s1 = r["step"] >= 1
    d = r["err"][s1] - b["err"][s1]
    mu, lo, hi = ep_boot(d, r["ep"][s1], reps=600)
    print(f"  {cc:20s} Rtail {r['err'][s1].mean():.3f}  B0 {b['err'][s1].mean():.3f}  d {mu:+.3f} [{lo:+.3f},{hi:+.3f}]  (M8 {m8['err'][s1].mean():.3f}); Rtail grip_mis {r['grip_mis'][s1].mean():.3f} vs B0 {b['grip_mis'][s1].mean():.3f}")
