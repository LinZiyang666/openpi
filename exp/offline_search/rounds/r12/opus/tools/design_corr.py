"""Does the library-only per-task signal predict the B-screen per-task pure-cache camera effect? (B only)"""
import json
import numpy as np
D = "/home/weiland/projects/openpi/exp/offline_search/rounds/r12/opus/out/design"
bs = json.load(open(f"{D}/bscreen_pertask.json"))
rows = []
for cell in ["pi05_l10_50", "pi05_spatial_50", "pi05_spatial_200", "groot_spatial_50", "groot_l10_50"]:
    z = np.load(f"{D}/libsignal_{cell}.npz")
    for t in range(10):
        r = bs[cell][str(t)]
        d = np.mean(np.array(r["wrist_sub"]) - np.array(r["both"]))
        de = np.mean(z[f"t{t}_e_w"] - z[f"t{t}_e_f"]); dg = np.mean(z[f"t{t}_g_w"] - z[f"t{t}_g_f"])
        rel = de / np.mean(z[f"t{t}_e_f"]); dF = np.mean(z[f"t{t}_dF"]); same = np.mean(z[f"t{t}_top1_same"])
        rows.append((cell, t, d, de, dg, rel, dF, same, np.mean(r["both"])))
cells = sorted({r[0] for r in rows}, key=lambda c: [r[0] for r in rows].index(c))
def corr(x, y):
    x, y = np.asarray(x), np.asarray(y); return np.corrcoef(x, y)[0, 1]
names = ["de", "dg", "rel_de", "dF", "top1same"]
print("per cell corr(B-screen d, signal):")
for c in cells:
    R = [r for r in rows if r[0] == c]
    print(f"  {c:18s} " + " ".join(f"{n} {corr([r[2] for r in R], [r[3+i] for r in R]):+.2f}" for i, n in enumerate(names)))
# pooled within-cell centred
print("pooled, within-cell centred (50 tasks):")
for i, n in enumerate(names):
    xs, ys = [], []
    for c in cells:
        R = [r for r in rows if r[0] == c]
        x = np.array([r[3+i] for r in R]); y = np.array([r[2] for r in R])
        xs += list(x - x.mean()); ys += list(y - y.mean())
    rr = corr(xs, ys)
    # permutation p within cells
    rng = np.random.default_rng(0); cnt = 0
    for _ in range(5000):
        ysp = []
        for c in cells:
            R = [r for r in rows if r[0] == c]
            y = np.array([r[2] for r in R]); ysp += list(rng.permutation(y - y.mean()))
        cnt += corr(xs, ysp) <= rr
    print(f"  {n:8s} r={rr:+.3f}  one-sided perm p(r<=obs)={cnt/5000:.3f}")
# cell means
print("cell means: B-screen d vs library de")
for c in cells:
    R = [r for r in rows if r[0] == c]
    print(f"  {c:18s} d {np.mean([r[2] for r in R]):+.3f}  de {np.mean([r[3] for r in R]):+.4f}  rel {np.mean([r[5] for r in R]):+.3f}")
json.dump([dict(cell=r[0], task=r[1], d_bscreen=r[2], de=r[3], dg=r[4], rel_de=r[5], dF=r[6], top1same=r[7], sr_both=r[8]) for r in rows],
          open(f"{D}/pertask_signals.json", "w"), indent=1)
