"""Regime split (step 0 / fresh / stale), synthesis effect, step-third profile, and the regime-mixed estimate
(hit-rate mixes) with pooled-confidence AURC (scale compatibility) and cost."""
import sys
sys.path.insert(0, "/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r01")
import numpy as np, pandas as pd
from h import *
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)

METHODS = ["B0_current", "B3_oracle", "Rtail_passthrough",
           "M1_big_a0p25_top1_stWin_b0", "M1_big_a0p25_med3_stWin_b0", "M1_big_a0p25_kmean5_stWin_b0", "M1_big_a0_med3_stWin_b0",
           "M1_big_a1_med3_stWin_b0", "M1_big_a0p25_med3_stCont_b0", "M1_current_a0p25_med3_stWin_b0", "M1_big_a0p25_med3_stWin_b1",
           "M2sw_big_m1_k5_mean", "M2sw_big_m3_k3_mean", "M2sw_big_m3_k5_mean", "M2sw_big_m3_k8_mean", "M2sw_big_m3_k5_med", "M2sw_current_m3_k5_mean",
           "M3hmm_big_a60b15g10_em1_cont_lev", "M3hmm_big_a60b15g10_em1_nocont_mass", "M3hmm_big_a60b15g10_em3_cont_mass",
           "M4_b0cons_k5_mean", "M4_b0cons_k5_kernel", "M4_b0cons_k8_mean", "M4_b0cons_k5_med", "M4_b0cons_k3_med",
           "M5_b0sl_cont_linf_k3", "M5_b0sl_cont_l2_k3", "M5_b0sl_cont_l1_k5", "M6_ccf_b0_eq",
           "M7_cascade_none_none_med3", "M7_cascade_raw_v1only_med3", "M7_cascade_pca32_v1only_med3", "M7_cascade_pca64_v1only_med3",
           "M7_cascade_pca128_v1only_med3", "M7_cascade_raw_both_med3", "M7_cascade_pca32_both_med3", "M7_cascade_raw_b0fused_med3",
           "M7_cascade_pca32_both_med3_M64_a0p25_lc0p5",
           "M8_b0big_top1", "M8_b0big_k5_mean",
           "M9c_vzsum_big_pca128_tm_both_top1", "M9c_vzsum_big_pca128_tm_both_st1_top1", "M9c_vzsum_big_pca128_tm_both_st1_med3",
           "M9c_vzsum_current_raw_tm_both_st1_top1"]

rows = []
for m in METHODS:
    for c in CELLS:
        d = load(m, c)
        if d is None: continue
        st = d["step"]; e = d["err"]; g = d["grip_mis"]; b = d["bin"]
        s0 = st == 0; s1 = ~s0
        rows.append(dict(method=m, cell=c, n=len(e), err_all=e.mean(), err_s0=e[s0].mean(), err_s1=e[s1].mean(),
                         err_s1_early=e[s1 & (b == 0)].mean(), err_s1_mid=e[s1 & (b == 1)].mean(), err_s1_late=e[s1 & (b == 2)].mean(),
                         grip_s1=g[s1].mean(), aurc_s1=aurc(e[s1], d["confidence"][s1]), aurc_s0=aurc(e[s0], d["confidence"][s0]),
                         risk30_s1=risk_at(e[s1], d["confidence"][s1], .3)))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/regime_split.csv", index=False)

def piv(col, arm, methods=None):
    sub = df[df.cell.str.endswith(arm)]
    if methods: sub = sub[sub.method.isin(methods)]
    p = sub.pivot(index="method", columns="cell", values=col)[[c for c in CELLS if c.endswith(arm)]]
    p["mean"] = p.mean(axis=1)
    return p.sort_values("mean").round(3)

print("## A. step-0 err (state/vision only decisions; inf and cache step-0 queries are the same inits, different noise)")
print("### inf cells"); print(piv("err_s0", "inf").to_string())
print("### cache cells"); print(piv("err_s0", "cache").to_string())
print("\n## A2. step-0 AURC"); print(piv("aurc_s0", "inf").to_string())

print("\n## B. step>=1 err: FRESH regime (inf cells)"); print(piv("err_s1", "inf").to_string())
print("\n## B2. step>=1 err: STALE regime (cache cells)"); print(piv("err_s1", "cache").to_string())
print("\n## B3. step>=1 AURC fresh"); print(piv("aurc_s1", "inf").to_string())
print("\n## B4. step>=1 AURC stale"); print(piv("aurc_s1", "cache").to_string())
print("\n## B5. step>=1 grip_mis fresh"); print(piv("grip_s1", "inf").to_string())
print("\n## B6. step>=1 grip_mis stale"); print(piv("grip_s1", "cache").to_string())

print("\n## C. by step-third (step>=1), top methods")
top = ["B0_current", "B3_oracle", "Rtail_passthrough", "M1_big_a0p25_kmean5_stWin_b0", "M8_b0big_k5_mean", "M2sw_big_m3_k8_mean",
       "M7_cascade_pca32_both_med3", "M7_cascade_none_none_med3", "M3hmm_big_a60b15g10_em1_cont_lev"]
for arm in ("inf", "cache"):
    sub = df[df.cell.str.endswith(arm) & df.method.isin(top)]
    for col in ("err_s1_early", "err_s1_mid", "err_s1_late"):
        print(f"### {arm} {col}"); print(piv(col, arm, top).to_string())

# ---------------------------------------------------------------- regime-mixed estimate
print("\n## D. regime-mixed estimate per model x suite")
print("fresh rows = inf cell step>=1 (after a MISS), stale rows = cache cell step>=1 (after a HIT; states produced by B0's cache),")
print("step-0 rows = inf cell step 0. mix(p) = w0*e0 + (1-w0)*(p*e_stale + (1-p)*e_fresh), w0 = share of step-0 decisions.")
FRESH = "M1_big_a0p25_kmean5_stWin_b0"
STALES = ["M8_b0big_k5_mean", "M2sw_big_m3_k8_mean", "M7_cascade_pca32_both_med3", "M7_cascade_raw_both_med3", "M1_big_a0p25_kmean5_stWin_b0",
          "B0_current"]
STEP0 = ["M8_b0big_k5_mean", "M9c_vzsum_big_pca128_tm_both_st1_med3", "M7_cascade_pca32_both_med3", "M2sw_big_m3_k8_mean",
         "M1_big_a0p25_kmean5_stWin_b0", "B0_current"]
sb = pd.read_csv(f"{OUT}/master.csv")
def ms_of(m, c, col):
    r = sb[(sb.method == m) & (sb.cell == c)]
    return float(r[col].iloc[0]) if len(r) else float("nan")
BYTES = {"M1_big_a0p25_kmean5_stWin_b0": 268, "M8_b0big_k5_mean": 262224, "M2sw_big_m3_k8_mean": 96, "M7_cascade_pca32_both_med3": 428,
         "M7_cascade_raw_both_med3": 262316, "B0_current": 262224, "M9c_vzsum_big_pca128_tm_both_st1_med3": 1056}
mixrows = []
for ms in MS:
    ci, cc = f"{ms}_inf", f"{ms}_cache"
    F = load(FRESH, ci); s1f = F["step"] >= 1
    e_fresh = F["err"][s1f]; c_fresh = F["confidence"][s1f]; ep_f = F["ep"][s1f]
    n0 = int((F["step"] == 0).sum()); w0 = n0 / len(F["err"])
    # step-0: best over STEP0 candidates on the inf cell (same inits as cache step 0)
    s0tab = {m: load(m, ci)["err"][load(m, ci)["step"] == 0].mean() for m in STEP0}
    best0 = min(s0tab, key=s0tab.get)
    e0 = s0tab[best0]
    print(f"\n### {ms}: n0={n0} w0={w0:.3f}; step-0 err: " + ", ".join(f"{k.split('_')[0]}={v:.3f}" for k, v in s0tab.items()) + f" -> best {best0}")
    print(f"    fresh({FRESH}) step>=1 err {e_fresh.mean():.3f} AURC {aurc(e_fresh, c_fresh):.3f}; B0 fresh {load('B0_current', ci)['err'][s1f].mean():.3f}")
    for stale in STALES:
        S = load(stale, cc); s1s = S["step"] >= 1
        e_stale = S["err"][s1s]; c_stale = S["confidence"][s1s]
        line = f"    stale={stale:30s} e_stale {e_stale.mean():.3f} (B0 {load('B0_current', cc)['err'][s1s].mean():.3f}) | mix err:"
        for p in (0.3, 0.5, 0.7):
            mix = w0 * e0 + (1 - w0) * (p * e_stale.mean() + (1 - p) * e_fresh.mean())
            b0mix = w0 * load('B0_current', ci)['err'][F['step']==0].mean() + (1 - w0) * (p * load('B0_current', cc)['err'][s1s].mean() + (1 - p) * load('B0_current', ci)['err'][s1f].mean())
            line += f" p{p:.1f}: {mix:.3f} (B0 {b0mix:.3f})"
        # pooled AURC with the methods' own confidences (one threshold for both regimes) vs regime-rank-normalized
        for p in (0.3, 0.5, 0.7):
            wf = (1 - p) / len(e_fresh) * np.ones(len(e_fresh)); ws = p / len(e_stale) * np.ones(len(e_stale))
            e = np.concatenate([e_fresh, e_stale]); w = np.concatenate([wf, ws])
            own = np.concatenate([c_fresh, c_stale])
            rf = (np.argsort(np.argsort(-c_fresh)) + 0.5) / len(c_fresh); rs_ = (np.argsort(np.argsort(-c_stale)) + 0.5) / len(c_stale)
            rank = -np.concatenate([rf, rs_])                       # same acceptance fraction in both regimes
            a_own = aurc(e, own, w); a_rank = aurc(e, rank, w); a_opt = aurc(e, -e, w)
            # regime-optimal: sort by err within regime is not deployable; instead report the accepted share from stale at 30% coverage under own conf
            o = np.lexsort((np.arange(len(own)), -own)); cw = np.cumsum(w[o]); k = int(np.searchsorted(cw, 0.3)) + 1
            share_stale = float(np.sum(o[:k] >= len(e_fresh)) * (p / len(e_stale)) / 0.3)
            r30_own = risk_at(e, own, .3, w); r30_rank = risk_at(e, rank, .3, w)
            line += f"\n        p{p:.1f} pooled AURC own {a_own:.3f} / rank-norm {a_rank:.3f} / opt {a_opt:.3f}; risk@30 own {r30_own:.3f} rank {r30_rank:.3f}; stale share of accepted@30 under own conf {share_stale:.2f} (its weight {p:.1f})"
        msq = {p: (1 - w0) * ((1 - p) * ms_of(FRESH, ci, "ms_per_query") + p * ms_of(stale, cc, "ms_per_query")) + w0 * ms_of(best0, ci, "ms_per_query") for p in (0.3, 0.5, 0.7)}
        by = max(BYTES[FRESH], BYTES.get(stale, 0)) + (BYTES[FRESH] if BYTES.get(stale, 0) > 1000 else 0)
        line += f"\n        cost: ms/query@conc4 p.3/.5/.7 = {msq[0.3]:.2f}/{msq[0.5]:.2f}/{msq[0.7]:.2f}; bytes/entry union ~ {BYTES[FRESH] + BYTES.get(stale, 0)} (fresh {BYTES[FRESH]} + stale {BYTES.get(stale, 0)})"
        print(line)
        for p in (0.3, 0.5, 0.7):
            mix = w0 * e0 + (1 - w0) * (p * e_stale.mean() + (1 - p) * e_fresh.mean())
            mixrows.append(dict(ms=ms, stale=stale, p=p, mix_err=mix, e0=e0, e_fresh=e_fresh.mean(), e_stale=e_stale.mean(), w0=w0))
pd.DataFrame(mixrows).to_csv(f"{OUT}/mix.csv", index=False)

print("\n## E. confidence scale across regimes (own confidence quantiles p10/p50/p90; step>=1)")
for m in ["M1_big_a0p25_kmean5_stWin_b0", "M8_b0big_k5_mean", "M2sw_big_m3_k8_mean", "M7_cascade_pca32_both_med3", "M6_ccf_b0_eq", "B0_current",
          "M3hmm_big_a60b15g10_em1_cont_lev"]:
    for ms in MS:
        out = []
        for arm in ("inf", "cache"):
            d = load(m, f"{ms}_{arm}"); s1 = d["step"] >= 1
            q = np.percentile(d["confidence"][s1], [10, 50, 90])
            out.append(f"{arm}: {q[0]:+.3f}/{q[1]:+.3f}/{q[2]:+.3f}")
        print(f"  {m:32s} {ms:13s} " + " | ".join(out))
