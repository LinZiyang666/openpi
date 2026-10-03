"""Aggregate the k-fold outputs (out/kfold/*.npz) into the tables of DATA_ANALYSIS.md (printed as markdown) + JSON."""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "out"
MAIN = ("loeo", "xfit", "loeo_aug", "xfit_aug", "pair_r50_c16", "pair_r95_c64", "pair_all_c64", "pair_r95_c64_raw")
BINS = (0, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, np.inf)


def load(pattern):
    groups = defaultdict(list)
    for f in sorted(glob.glob(str(OUT / "kfold" / pattern))):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        meta = json.loads(str(z["meta_json"]))
        groups[(meta["model"], meta["suite"], meta["size"], meta["pca"], meta.get("mode", "main"))].append(({k: z[k] for k in z.files if k != "meta_json"}, meta))
    out = {}
    for key, parts in groups.items():
        arr = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
        out[key] = (arr, [p[1] for p in parts])
    return out


def boot_ci(diff, ep, n=2000, seed=0):
    """Mean row difference with a 95 % CI from an episode-cluster bootstrap."""
    ue, inv = np.unique(ep, return_inverse=True)
    s = np.bincount(inv, weights=diff)
    c = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ue), (n, len(ue)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def schemes_of(arr):
    have = {k.split("|")[1] for k in arr if k.count("|") == 2}
    extra = sorted(have - set(MAIN) - {"none"})
    return [s for s in MAIN if s in have] + extra


def summarize(arr, metas, cond="full"):
    ep = arr["ep"]
    base = arr[f"{cond}|none|e10"]
    res = dict(n_rows=int(len(ep)), n_ep=int(len(np.unique(ep))), none_e10=float(base.mean()),
               none_e5=float(arr[f"{cond}|none|e5"].mean()))
    for s in schemes_of(arr):
        e = arr[f"{cond}|{s}|e10_b0.5"]
        m, lo, hi = boot_ci(e - base, ep)
        mL, loL, hiL = boot_ci(e - arr[f"{cond}|loeo|e10_b0.5"], ep)
        r = dict(e10_b05=float(e.mean()), d_vs_none=m, ci_none=(lo, hi), rel_vs_none=m / float(base.mean()),
                 d_vs_loeo=mL, ci_loeo=(loL, hiL), e10_b1=float(arr[f"{cond}|{s}|e10_b1"].mean()),
                 e5_b05=float(arr[f"{cond}|{s}|e5_b0.5"].mean()),
                 b_opt=float(arr[f"{cond}|{s}|dot"].sum() / arr[f"{cond}|{s}|cc"].sum()))
        if cond == "full" and f"full|{s}|slope_num" in arr:
            r["slope"] = float(arr[f"full|{s}|slope_num"].sum() / arr[f"full|{s}|slope_den"].sum())
        res[s] = r
    return res


def bins_table(arr, cond="full", key="d1norm_in", schemes=("loeo", "xfit", "xfit_aug", "pair_r50_c16", "pair_r95_c64")):
    d = arr[key if cond == "full" else key.replace("d1norm_in", "d1norm_half")]
    if cond == "half":
        d = arr["d1_half"] / (arr["d1"] / arr["d1norm_in"])        # same per-task in-sample scale as "full"
    base = arr[f"{cond}|none|e10"]
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 30:
            continue
        r = dict(bin=f"[{lo:g},{hi:g})", n=int(m.sum()), frac=float(m.mean()), none=float(base[m].mean()))
        for s in schemes:
            r[s] = float(arr[f"{cond}|{s}|e10_b0.5"][m].mean() / base[m].mean() - 1)
        rows.append(r)
    return rows


def diag_table(metas):
    D = [d for m in metas for d in m["diag"]]
    g = lambda k: float(np.mean([d[k] for d in D]))
    pairs = defaultdict(list)
    for d in D:
        for name, p in d["pairs"].items():
            pairs[name].append(p)
    return dict(loeo_train_err10=g("loeo_err10"), xfit_train_err10=g("xfit_err10"),
                pairs={k: dict(n_pairs_per_task_fold=float(np.mean([p["n_pairs"] for p in v])),
                               frac_rows_no_pair=float(np.mean([p["frac_rows_no_pair"] for p in v])))
                       for k, v in pairs.items()},
                n_train_per_task_fold=float(np.mean([d["n_train"] for d in D])))


def fmt_ci(m, ci):
    return f"{m:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*.npz")
    a = ap.parse_args(argv)
    data = load(a.pattern)
    js = {}
    for key in sorted(data, key=lambda k: (k[4], k[1], k[0], k[2], k[3])):
        arr, metas = data[key]
        if len(metas) != 5:
            print(f"(skip {key}: {len(metas)} folds)")
            continue
        name = f"{key[0]}_{key[1]}_{key[2]}" + ("" if key[4] == "main" else f"_{key[4]}")
        full, half = summarize(arr, metas, "full"), summarize(arr, metas, "half")
        heldout = float(arr["full|none|e10"].mean())
        dg = diag_table(metas)
        bs = tuple(x for x in ("loeo", "xfit", "xfit_aug", "pair_r50_c16", "pair_r95_c64") if f"full|{x}|e10_b0.5" in arr)
        js[name] = dict(full=full, half=half, bins_full=bins_table(arr, "full", schemes=bs), bins_half=bins_table(arr, "half", schemes=bs),
                        diag=dg, d1norm_in_median=float(np.median(arr["d1norm_in"])),
                        d1norm_in_half_median=float(np.median(arr["d1_half"] / (arr["d1"] / arr["d1norm_in"]))),
                        heldout_none_e10=heldout, pca=key[3])
        print(f"\n### {name}  (rows {full['n_rows']}, episodes {full['n_ep']}, PCA {key[3]})")
        print(f"held-out served error (no corrector) e10 = {full['none_e10']:.4f} (executed 5 steps {full['none_e5']:.4f}); "
              f"half-library {half['none_e10']:.4f}; LOEO in-sample train e10 {dg['loeo_train_err10']:.4f}, "
              f"cross-fitted {dg['xfit_train_err10']:.4f}; held-out d1 / in-sample LOEO d1 median "
              f"{js[name]['d1norm_in_median']:.2f} (half {js[name]['d1norm_in_half_median']:.2f})")
        print("| scheme | full: e10 @.5 | Δ vs none [95% CI] | Δ vs loeo [95% CI] | e10 @1.0 | best blend | slope | half: Δ vs none [95% CI] | half best blend |")
        print("|---|---|---|---|---|---|---|---|---|")
        for s in schemes_of(arr):
            f, h = full[s], half[s]
            print(f"| {s} | {f['e10_b05']:.4f} | {fmt_ci(f['d_vs_none'], f['ci_none'])} ({100 * f['rel_vs_none']:+.1f}%) | "
                  f"{fmt_ci(f['d_vs_loeo'], f['ci_loeo'])} | {f['e10_b1']:.4f} | {f['b_opt']:.2f} | {f.get('slope', float('nan')):+.2f} | "
                  f"{fmt_ci(h['d_vs_none'], h['ci_none'])} ({100 * h['rel_vs_none']:+.1f}%) | {h['b_opt']:.2f} |")
        print("\nrelative change of e10 at blend .5 vs no corrector, by held-out d1 / in-sample LOEO d1 (full library):")
        bt = js[name]["bins_full"]
        cols = [c for c in bt[0] if c not in ("bin", "n", "frac", "none")]
        print("| d1 bin | rows | share | none e10 | " + " | ".join(cols) + " |")
        print("|---|---|---|---|" + "---|" * len(cols))
        for r in bt:
            print(f"| {r['bin']} | {r['n']} | {r['frac']:.2f} | {r['none']:.3f} | " + " | ".join(f"{100 * r[c]:+.1f}%" for c in cols) + " |")
        print("\npairs per task-fold: " + ", ".join(f"{k} {v['n_pairs_per_task_fold']:.0f} (rows w/o pair {100 * v['frac_rows_no_pair']:.1f}%)"
                                               for k, v in dg["pairs"].items()) + f"; training rows per task-fold {dg['n_train_per_task_fold']:.0f}")
    (OUT / "summary.json").write_text(json.dumps(js, indent=1) + "\n")


if __name__ == "__main__":
    main()


def breakdown(pattern="*_auto.npz", schemes=("loeo", "pair_r50_c16", "pair_r95_c64")):
    """Relative e10 change at blend .5 vs no corrector, split by episode success and by episode phase (step / ep_len)."""
    data = load(pattern)
    rows = []
    for key in sorted(data, key=lambda k: (k[1], k[0], k[2])):
        if key[4] != "main":
            continue
        arr, metas = data[key]
        if len(metas) != 5:
            continue
        base = arr["full|none|e10"]
        ph = arr["step"] / np.maximum(arr["ep_len"] - 1, 1)
        masks = {"success ep": arr["success"], "failed ep": ~arr["success"], "phase<1/3": ph < 1 / 3,
                 "phase 1/3-2/3": (ph >= 1 / 3) & (ph < 2 / 3), "phase>=2/3": ph >= 2 / 3, "step 0": arr["step"] == 0}
        for mname, m in masks.items():
            if m.sum() < 30:
                continue
            r = dict(cell=f"{key[0]}_{key[1]}_{key[2]}", subset=mname, n=int(m.sum()), none=float(base[m].mean()))
            for s in schemes:
                r[s] = float(arr[f"full|{s}|e10_b0.5"][m].mean() / base[m].mean() - 1)
            rows.append(r)
    print("| cell | rows | n | none e10 | " + " | ".join(schemes) + " |")
    print("|---|---|---|---|" + "---|" * len(schemes))
    for r in rows:
        print(f"| {r['cell']} | {r['subset']} | {r['n']} | {r['none']:.3f} | " + " | ".join(f"{100 * r[s]:+.1f}%" for s in schemes) + " |")
    return rows


def consolidated(cond="full", schemes=("loeo", "loeo_aug", "xfit", "pair_r50_c16", "pair_r95_c64", "pair_all_c64",
                                       "pair_r95_c64_raw")):
    """One row per cell x size: no-corrector error and relative change (95 % episode-bootstrap CI) per scheme."""
    data = load("*_auto.npz")
    keys = sorted([k for k in data if k[4] == "main" and len(data[k][1]) == 5], key=lambda k: (k[1], k[0], k[2]))
    print(f"| cell | size | new-ep d1 ratio | none e10 | " + " | ".join(schemes) + " |")
    print("|---|---|---|---|" + "---|" * len(schemes))
    out = {}
    for k in keys:
        arr, metas = data[k]
        res = summarize(arr, metas, cond)
        cells = []
        for s in schemes:
            r = res[s]
            b = res["none_e10"]
            star = "*" if (r["ci_none"][1] < 0 or r["ci_none"][0] > 0) else ""
            cells.append(f"{100 * r['d_vs_none'] / b:+.1f}{star} [{100 * r['ci_none'][0] / b:+.1f},{100 * r['ci_none'][1] / b:+.1f}]")
        dr = float(np.median(arr["d1norm_in"])) if cond == "full" else float(np.median(arr["d1_half"] / (arr["d1"] / arr["d1norm_in"])))
        print(f"| {k[0]} {k[1]} | {k[2]} | {dr:.2f} | {res['none_e10']:.3f} | " + " | ".join(cells) + " |")
        out[f"{k[0]}_{k[1]}_{k[2]}"] = res
    return out


def blend_slope_table(schemes=("loeo", "loeo_aug", "xfit", "pair_r50_c16", "pair_r95_c64", "pair_all_c64", "pair_r95_c64_raw")):
    data = load("*_auto.npz")
    keys = sorted([k for k in data if k[4] == "main" and len(data[k][1]) == 5], key=lambda k: (k[1], k[0], k[2]))
    print("| cell | size | " + " | ".join(f"{s} b*/slope" for s in schemes) + " | loeo e10 @1.0 vs @.5 |")
    print("|---|---|" + "---|" * len(schemes) + "---|")
    for k in keys:
        arr, metas = data[k]
        res = summarize(arr, metas, "full")
        cells = [f"{res[s]['b_opt']:.2f} / {res[s]['slope']:+.2f}" for s in schemes]
        print(f"| {k[0]} {k[1]} | {k[2]} | " + " | ".join(cells) + f" | {res['loeo']['e10_b1']:.3f} vs {res['loeo']['e10_b05']:.3f} |")


def pooled_bins(sizes=(50,), schemes=("loeo", "loeo_aug", "pair_r50_c16", "pair_r95_c64"), cond="full"):
    """Relative e10 change vs no corrector by d1 bin, pooled over all four cells of the given sizes (each cell's rows
    weighted equally within the pool: per-cell bin means averaged)."""
    data = load("*_auto.npz")
    keys = [k for k in data if k[4] == "main" and k[2] in sizes and len(data[k][1]) == 5]
    agg = defaultdict(lambda: defaultdict(list))
    for k in keys:
        arr, _ = data[k]
        d = arr["d1norm_in"] if cond == "full" else arr["d1_half"] / (arr["d1"] / arr["d1norm_in"])
        base = arr[f"{cond}|none|e10"]
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (d >= lo) & (d < hi)
            if m.sum() < 30:
                continue
            agg[(lo, hi)]["n"].append(int(m.sum()))
            agg[(lo, hi)]["none"].append(float(base[m].mean()))
            for s in schemes:
                agg[(lo, hi)][s].append(float(arr[f"{cond}|{s}|e10_b0.5"][m].mean() / base[m].mean() - 1))
    print(f"sizes {sizes}, {cond}: cells {len(keys)}")
    print("| d1 bin | cells | rows | none e10 | " + " | ".join(schemes) + " |")
    print("|---|---|---|---|" + "---|" * len(schemes))
    for (lo, hi), v in sorted(agg.items()):
        print(f"| [{lo:g},{hi:g}) | {len(v['n'])} | {sum(v['n'])} | {np.mean(v['none']):.3f} | "
              + " | ".join(f"{100 * np.mean(v[s]):+.1f}%" for s in schemes) + " |")
