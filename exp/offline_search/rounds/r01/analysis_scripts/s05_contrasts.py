"""Paired contrasts by regime part (step 0 / step>=1) with episode-bootstrap CIs: vision value, synthesis, cascade
width, state term; gripper-transition slice table; timeline summary table."""
import sys, glob, os
sys.path.insert(0, "/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r01")
import numpy as np, pandas as pd
from h import *
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)

PAIRS = [
    ("vision(B0 formula, raw keys)+mean5 vs state-only m1 mean5, big lib", "M8_b0big_k5_mean", "M2sw_big_m1_k5_mean"),
    ("vision(B0 formula)+mean5 vs state window m3 mean8", "M8_b0big_k5_mean", "M2sw_big_m3_k8_mean"),
    ("M7 vision re-rank (pca32 both) vs none, M=16", "M7_cascade_pca32_both_med3", "M7_cascade_none_none_med3"),
    ("M7 vision re-rank (raw both) vs none, M=16", "M7_cascade_raw_both_med3", "M7_cascade_none_none_med3"),
    ("M7 vision re-rank (pca32 v1only) vs none", "M7_cascade_pca32_v1only_med3", "M7_cascade_none_none_med3"),
    ("M7 raw both vs pca32 both (key size)", "M7_cascade_raw_both_med3", "M7_cascade_pca32_both_med3"),
    ("M7 pca128 v1 vs pca32 v1 (key size)", "M7_cascade_pca128_v1only_med3", "M7_cascade_pca32_v1only_med3"),
    ("M7 M=64 vs M=16 (pca32 both)", "M7_cascade_pca32_both_med3_M64_a0p25_lc0p5", "M7_cascade_pca32_both_med3"),
    ("M7 b0fused vs both (fusion formula)", "M7_cascade_raw_b0fused_med3", "M7_cascade_raw_both_med3"),
    ("M9c pca128 z-sum st1 top1 vs M8 top1 (repr: pca128 z-sum vs raw B0)", "M9c_vzsum_big_pca128_tm_both_st1_top1", "M8_b0big_top1"),
    ("M9c st1 vs st0 (state term in vision z-sum)", "M9c_vzsum_big_pca128_tm_both_st1_top1", "M9c_vzsum_big_pca128_tm_both_top1"),
    ("M9c med3 vs top1", "M9c_vzsum_big_pca128_tm_both_st1_med3", "M9c_vzsum_big_pca128_tm_both_st1_top1"),
    ("M1 kmean5 vs med3 (synthesis)", "M1_big_a0p25_kmean5_stWin_b0", "M1_big_a0p25_med3_stWin_b0"),
    ("M1 med3 vs top1 (synthesis)", "M1_big_a0p25_med3_stWin_b0", "M1_big_a0p25_top1_stWin_b0"),
    ("M2 k8 mean vs k5 mean", "M2sw_big_m3_k8_mean", "M2sw_big_m3_k5_mean"),
    ("M2 k5 med vs k5 mean", "M2sw_big_m3_k5_med", "M2sw_big_m3_k5_mean"),
    ("M2 m3 vs m1 (window)", "M2sw_big_m3_k5_mean", "M2sw_big_m1_k5_mean"),
    ("M1 alpha 0.25 vs 0 (state term in fresh)", "M1_big_a0p25_med3_stWin_b0", "M1_big_a0_med3_stWin_b0"),
    ("M1 alpha 1 vs 0.25", "M1_big_a1_med3_stWin_b0", "M1_big_a0p25_med3_stWin_b0"),
    ("M1 beta 1 vs 0 (GR00T 2-back tail)", "M1_big_a0p25_med3_stWin_b1", "M1_big_a0p25_med3_stWin_b0"),
    ("M1 stale=cont vs window (stale branch)", "M1_big_a0p25_med3_stCont_b0", "M1_big_a0p25_med3_stWin_b0"),
    ("M1 big vs current (library)", "M1_big_a0p25_med3_stWin_b0", "M1_current_a0p25_med3_stWin_b0"),
    ("M3 HMM lev vs M2 m1 k5 (transition prior)", "M3hmm_big_a60b15g10_em1_cont_lev", "M2sw_big_m1_k5_mean"),
    ("M3 HMM a80 vs a60", "M3hmm_big_a80b10g05_em1_cont_mass", "M3hmm_big_a60b15g10_em1_cont_mass"),
    ("M4 kernel5 vs mean5 (B0 synthesis)", "M4_b0cons_k5_kernel", "M4_b0cons_k5_mean"),
    ("M4 med5 vs mean5", "M4_b0cons_k5_med", "M4_b0cons_k5_mean"),
    ("M5 linf k3 vs l2 k3 (rerank strength)", "M5_b0sl_cont_linf_k3", "M5_b0sl_cont_l2_k3"),
    ("Rtail vs M1 kmean5", "Rtail_passthrough", "M1_big_a0p25_kmean5_stWin_b0"),
]
print("## paired err differences A-B by regime part (mean [95% episode-bootstrap CI]); negative = A better")
print("columns: 4 model_suite; parts: s0 = step 0, s1 = step>=1 (inf: fresh, cache: stale)")
for title, a, b in PAIRS:
    print(f"\n### {title}: A={a} B={b}")
    for arm in ("inf", "cache"):
        for part in ("s0", "s1"):
            row = []
            for ms in MS:
                c = f"{ms}_{arm}"
                A, B = load(a, c), load(b, c)
                if A is None or B is None:
                    row.append("      -       "); continue
                m = (A["step"] == 0) if part == "s0" else (A["step"] >= 1)
                d = A["err"][m] - B["err"][m]
                mu, lo, hi = ep_boot(d, A["ep"][m], reps=600)
                sig = "*" if (lo > 0 or hi < 0) else " "
                row.append(f"{mu:+.3f}[{lo:+.3f},{hi:+.3f}]{sig}")
            print(f"  {arm:5s} {part}: " + "  ".join(row))

print("\n\n## gripper-transition slice (from breakdown csv): share / err(trans) / grip_mis(trans) / err(steady) / AURC(trans)")
rows = []
for p in glob.glob(f"{OUT}/breakdown/*/breakdown.csv"):
    m = os.path.basename(os.path.dirname(p))
    df = pd.read_csv(p)
    for c in CELLS:
        t = df[(df.cell == c) & (df.slice == "grip") & (df.value == "transition")]
        s = df[(df.cell == c) & (df.slice == "grip") & (df.value == "steady")]
        if len(t) and len(s):
            rows.append(dict(method=m, cell=c, share=float(t.share.iloc[0]), err_t=float(t.err.iloc[0]), grip_t=float(t.grip.iloc[0]),
                             err_s=float(s.err.iloc[0]), aurc_t=float(t.aurc.iloc[0]), conf_t=float(t.conf.iloc[0]), conf_s=float(s.conf.iloc[0])))
g = pd.DataFrame(rows)
for col in ("err_t", "grip_t", "err_s"):
    print(f"\n### {col}")
    print(g.pivot(index="method", columns="cell", values=col)[CELLS].round(3).to_string())
print("\n### transition share per cell:", g[g.method == "B0_current"].set_index("cell").share.round(3).to_dict())

print("\n\n## timeline summary (track / stay / switch rates and err after each move; step>=1)")
for p in sorted(glob.glob(f"{OUT}/timeline/*/timeline_summary.csv")):
    m = os.path.basename(os.path.dirname(p))
    df = pd.read_csv(p).set_index("cell")
    print(f"\n### {m}")
    print(df[["track", "err|track", "stay", "err|stay", "switch", "err|switch", "runlen"]].round(3).to_string())

print("\n\n## outcome slice (cache arms): err on failed vs successful cache-run episodes, share of failed")
rows = []
for p in glob.glob(f"{OUT}/breakdown/*/breakdown.csv"):
    m = os.path.basename(os.path.dirname(p))
    df = pd.read_csv(p)
    for c in CELLS:
        f = df[(df.cell == c) & (df.slice == "outcome") & (df.value == "failure")]
        s = df[(df.cell == c) & (df.slice == "outcome") & (df.value == "success")]
        if len(f) and len(s):
            rows.append(dict(method=m, cell=c, fail_share=float(f.share.iloc[0]), err_fail=float(f.err.iloc[0]), err_succ=float(s.err.iloc[0])))
o = pd.DataFrame(rows)
print(o.pivot(index="method", columns="cell", values="err_fail")[CELLS].round(3).to_string())
print(o.pivot(index="method", columns="cell", values="err_succ")[CELLS].round(3).to_string())
print("fail share:", o[o.method == "B0_current"].set_index("cell").fail_share.round(2).to_dict())
