import json, pathlib, sys
import numpy as np
OUT = pathlib.Path("/home/weiland/.claude/jobs/a607dd74/tmp/r02_ideation_B")
KEYS = ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]
R = {}
for p in sorted(OUT.glob("diag_*.json")):
    R[p.stem[5:]] = json.load(open(p))


def cell(key, arm):
    return R.get(f"{key}_{arm}")


def fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def row(name, get):
    vals = []
    for k in KEYS:
        try:
            vals.append(get(k))
        except Exception:
            vals.append(None)
    return f"| {name} | " + " | ".join(fmt(v) for v in vals) + " |"


print("\n## A. err by regime (cells: " + " / ".join(KEYS) + ")")
print("### stale (cache arms, step>=1)")
tags = sorted({t for c in R.values() for t in c["A"]})
for t in tags:
    print(row(t, lambda k, t=t: cell(k, "cache")["A"][t]["stale"]["err"]))
print("### stale, success-only episodes / failed episodes (pca32 st1 mean5, per library)")
for t in ["current_bcur_pca32_st1_mean5", "current_bbig_pca32_st1_mean5", "big_bbig_pca32_st1_mean5", "big_bbig_pca64_st1_mean8", "current_stateonly_mean5", "big_stateonly_mean5"]:
    print(row(t + " succ", lambda k, t=t: cell(k, "cache")["A"][t]["stale"]["err_succ"]))
    print(row(t + " fail", lambda k, t=t: cell(k, "cache")["A"][t]["stale"]["err_fail"]))
print("### fresh (inf arms, step>=1)")
for t in tags:
    print(row(t, lambda k, t=t: cell(k, "inf")["A"][t]["fresh"]["err"]))
print("### step 0 (inf arms)")
for t in tags:
    print(row(t, lambda k, t=t: cell(k, "inf")["A"][t]["step0"]["err"]))
print("### step 0 (cache arms)")
for t in [x for x in tags if "align" in x or ("pca32_st1_mean5" in x and "align" not in x)]:
    print(row(t, lambda k, t=t: cell(k, "cache")["A"][t]["step0"]["err"]))

print("\n## B. stale AURC (cache arms), pca32 st1 mean5; rows = signal; per library")
for lib in ("big", "current"):
    print(f"### library {lib}")
    sigs = sorted({s for c in R.values() if c["B"] for s in c["B"][lib].get("stale", {})})
    for s in sigs:
        print(row(s, lambda k, s=s: cell(k, "cache")["B"][lib]["stale"][s]))
print("\n## B2. fresh AURC (inf arms), same features")
for lib in ("big",):
    sigs = sorted({s for c in R.values() if c["B"] for s in c["B"][lib].get("fresh", {})})
    for s in sigs:
        if isinstance(cell("pi05_spatial", "inf")["B"][lib]["fresh"].get(s), (int, float)):
            print(row(s, lambda k, s=s: cell(k, "inf")["B"][lib]["fresh"][s]))

print("\n## C. synthesis variants, stale (cache arms): err / gmis / err_trans / gmis_trans / frac_trans")
for lk in ("big_k5", "big_k8", "current_k5"):
    print(f"### {lk}")
    for v in ["mean", "mean_snap", "mean_gmaj", "cluster_end", "cluster_pattern", "cluster_end_w", "cluster_end_prior60"]:
        for met in ["err", "gmis", "err_trans", "gmis_trans", "err_steady"]:
            print(row(f"{v} {met}", lambda k, v=v, met=met: cell(k, "cache")["C"][lk][v]["stale"][met]))
    print(row("frac_trans", lambda k: cell(k, "cache")["C"][lk]["mean"]["stale"]["frac_trans"]))
print("### fresh (inf arms) big_k5 mean vs mean_snap vs cluster_end")
for v in ["mean", "mean_snap", "cluster_end", "cluster_end_w"]:
    for met in ["err", "gmis", "err_trans", "gmis_trans"]:
        print(row(f"{v} {met}", lambda k, v=v, met=met: cell(k, "inf")["C"]["big_k5"][v]["fresh"][met]))

print("\n## D. fresh regime: -c/s_c + lam*vis/2 + alpha*z(-d), all candidates, mean5 (inf arms)")
for lib in ("big", "current"):
    print(f"### library {lib}")
    keys = sorted({s for c in R.values() if c.get("D") and lib in c["D"] for s in c["D"][lib] if s.startswith("lam")})
    for s in keys:
        for met in ["err_fresh", "aurc_cont", "aurc_cont_disp", "aurc_cont_disp_vis", "err_trans"]:
            print(row(f"{s} {met}", lambda k, s=s, met=met: cell(k, "inf")["D"][lib][s][met]))
    print(row("opt", lambda k: cell(k, "inf")["D"][lib]["lam0_a0"]["opt"]))

print("\n## E. timing (us, single thread)")
for lib in ("current", "big"):
    for met in ["proj_us_D32", "proj_us_32D", "score_us_p50", "score_us_p95", "cands_mean"]:
        print(row(f"{lib} {met}", lambda k, lib=lib, met=met: cell(k, "cache")["E"][lib][met]))
print("\ncells loaded:", sorted(R))
