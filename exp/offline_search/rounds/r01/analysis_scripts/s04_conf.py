"""Confidence signals: AURC of the method's own confidence vs alternative per-decision signals from extras, per regime
(step>=1 only; step 0 separately), plus simple z-sum combos (z within cell: diagnostic, not deployable as-is)."""
import sys
sys.path.insert(0, "/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r01")
import numpy as np, pandas as pd
from h import *
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)

def z(x):
    x = np.asarray(x, np.float64); s = np.nanstd(x)
    return (x - np.nanmean(x)) / (s if s > 1e-12 else 1.0)

SIG = {
    "M1_big_a0p25_kmean5_stWin_b0": {"own": None, "-c0": ("x_c0", -1), "-d0": ("x_d0", -1), "-w0": ("x_w0", -1), "-disp": ("x_disp", -1), "-f0": ("x_f0", -1)},
    "M1_big_a0p25_med3_stWin_b0": {"own": None, "-c0": ("x_c0", -1), "-d0": ("x_d0", -1), "-w0": ("x_w0", -1), "-disp": ("x_disp", -1)},
    "M8_b0big_k5_mean": {"own": None, "b0_pick": ("x_b0_pick", 1), "-disp5": ("x_disp5_b0", -1), "margin": ("x_margin", 1), "-margin": ("x_margin", -1),
                         "-cont_act": ("x_cont_act", -1), "-cont_b0": ("x_cont_b0", -1), "-dist_rs": ("x_dist_rs", -1), "cos_v0": ("x_cos_v0", 1), "cos_v1": ("x_cos_v1", 1), "-flip10": ("x_flip10", -1)},
    "M8_b0big_top1": {"own": None, "-disp5": ("x_disp5_b0", -1), "-cont_act": ("x_cont_act", -1), "-dist_rs": ("x_dist_rs", -1), "-flip10": ("x_flip10", -1)},
    "M2sw_big_m3_k8_mean": {"own": None, "cw": ("x_cw", 1), "ca": ("x_ca", 1), "-gdis": ("x_gdis", -1), "-dk": ("x_dk", -1)},
    "M7_cascade_pca32_both_med3": {"own": None, "-c0": ("x_c0", -1), "-d0": ("x_d0", -1), "vis0": ("x_vis0", 1), "g": ("x_g", 1), "-disp": ("x_disp", -1), "cos1": ("x_cos1_0", 1), "cos0": ("x_cos0_0", 1)},
    "M7_cascade_raw_both_med3": {"own": None, "-c0": ("x_c0", -1), "-d0": ("x_d0", -1), "vis0": ("x_vis0", 1), "-disp": ("x_disp", -1)},
    "M3hmm_big_a60b15g10_em1_cont_lev": {"own": None, "lev": ("x_lev", 1), "mass5": ("x_mass5", 1), "-ent": ("x_ent", -1), "neff": ("x_neff", 1), "-neff": ("x_neff", -1), "-dw1": ("x_dw1", -1), "-dwmin": ("x_dwmin", -1), "-c1": ("x_c1", -1), "-cmin": ("x_cmin", -1)},
    "M6_ccf_b0_eq": {"own": None, "t_g": ("x_t_g", 1), "-t_cont": ("x_t_cont", -1), "-t_disp": ("x_t_disp", -1), "-cont_b0": ("x_cont_b0", -1), "-disp5": ("x_disp5_b0", -1), "-flip10": ("x_flip10", -1), "margin": ("x_margin", 1), "-dist_rs": ("x_dist_rs", -1)},
    "M9c_vzsum_big_pca128_tm_both_st1_med3": {"own": None, "s0": ("x_s0", 1), "-disp": ("x_disp", -1)},
    "B0_current": {"own": None},
}
COMBOS = {
    "M1_big_a0p25_kmean5_stWin_b0": {"z(-c0)+z(-disp)": ["-c0", "-disp"], "z(-w0)+z(-disp)": ["-w0", "-disp"], "z(-d0)+z(-disp)": ["-d0", "-disp"], "z(-c0)+z(-d0)+z(-disp)": ["-c0", "-d0", "-disp"], "z(-w0)+z(-d0)+z(-disp)": ["-w0", "-d0", "-disp"]},
    "M8_b0big_k5_mean": {"z(b0)+z(-disp5)": ["b0_pick", "-disp5"], "z(b0)+z(-disp5)+z(-cont)": ["b0_pick", "-disp5", "-cont_act"], "z(-disp5)+z(-dist_rs)": ["-disp5", "-dist_rs"], "z(b0)+z(-disp5)+z(-flip10)": ["b0_pick", "-disp5", "-flip10"]},
    "M2sw_big_m3_k8_mean": {"z(cw)+z(ca)": ["cw", "ca"]},
    "M7_cascade_pca32_both_med3": {"z(-d0)+z(-disp)+z(vis0)": ["-d0", "-disp", "vis0"], "z(-c0)+z(-disp)": ["-c0", "-disp"], "z(-d0)+z(-disp)": ["-d0", "-disp"], "z(-c0)+z(-disp)+z(vis0)": ["-c0", "-disp", "vis0"]},
    "M3hmm_big_a60b15g10_em1_cont_lev": {"z(lev)+z(-dw1)": ["lev", "-dw1"]},
}
rows = []
for m, sigs in SIG.items():
    for c in CELLS:
        d = load(m, c)
        if d is None: continue
        for part, mask in (("s1", d["step"] >= 1), ("s0", d["step"] == 0)):
            e = d["err"][mask]
            rec = dict(method=m, cell=c, part=part, n=int(mask.sum()), opt=aurc(e, -e))
            vals = {}
            for name, spec in sigs.items():
                v = d["confidence"][mask] if spec is None else spec[1] * d[spec[0]][mask].astype(np.float64)
                v = np.where(np.isfinite(v), v, np.nanmin(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0)
                vals[name] = v
                rec[name] = aurc(e, v)
            for cname, parts in COMBOS.get(m, {}).items():
                v = sum(z(vals[p]) for p in parts)
                rec[cname] = aurc(e, v)
            rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/conf_signals.csv", index=False)
for m in SIG:
    sub = df[df.method == m]
    if not len(sub): continue
    cols = [c for c in sub.columns if c not in ("method", "cell", "part", "n")]
    for part in ("s1", "s0"):
        s = sub[sub.part == part].set_index("cell")[cols]
        s = s.reindex(CELLS)
        print(f"\n## {m} — AURC by signal ({'step>=1' if part == 's1' else 'step 0'})")
        s2 = s.T.copy()
        s2["inf_mean"] = s2[[c for c in CELLS if c.endswith('inf')]].mean(axis=1)
        s2["cache_mean"] = s2[[c for c in CELLS if c.endswith('cache')]].mean(axis=1)
        print(s2.round(3).to_string())

print("\n## AUROC of own confidence for 'bad' (err > floor p90) vs not, step>=1, per cell")
def roc_auc_score(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, np.float64)
    r = pd.Series(s).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
for m in ["B0_current", "M1_big_a0p25_kmean5_stWin_b0", "M8_b0big_k5_mean", "M2sw_big_m3_k8_mean", "M7_cascade_pca32_both_med3", "M6_ccf_b0_eq", "M3hmm_big_a60b15g10_em1_cont_lev"]:
    row = []
    for c in CELLS:
        d = load(m, c); j = cell_json(m, c); s1 = d["step"] >= 1
        fl = j["floor"]; p90 = fl.get("p90")
        if not isinstance(p90, dict):
            row.append("-"); continue
        p90 = {str(k): v for k, v in p90.items()}
        thr = np.array([p90.get(str(int(b)), p90.get("all", np.nan)) for b in d["bin"]])
        bad = (d["err"] > thr)[s1]
        if bad.sum() == 0 or bad.sum() == len(bad):
            row.append("-"); continue
        row.append(f"{roc_auc_score(~bad, d['confidence'][s1]):.3f}({bad.mean():.2f})")
    print(f"  {m:32s} " + " ".join(row))
