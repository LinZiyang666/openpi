"""Frontier per cell (owner / owner+search / eager), chord vs pure cache & pure inference, pooled g500 test, policy calls/ep."""
from __future__ import annotations
import json, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import an

S = an.S
D = json.load(open(f"{S}/an.json"))
rows = D["arms"]

CELLS = {
 "p_l10_500": ["cl2_l10_500","g500","g500_repa","g500_repb","per8_500","per12_500","g500_np4","awm500_h70","k1_b0g","k1_ph1g","k1_ph2g","k1_tail1ug","k1_clk1g","k1_ph2k8","ph2c_l10_500","k7_b0g","k7_ph2g","k7_ph1g","k7_tail1ug","wrist_l10_500","cslG_500","cslGS_500","k5_g500_r1","k5_g500_r2","inf_l10_trace","inf_l10_s1001","inf_l10_s2001","inf_l10_L10","inf_l10_L10b"],
 "p_l10_50":  ["cl2_l10_50","g50","perk5_50","perk3_50","per6_50","g50_np4","awm_h70_50","awm_h50_50","k1_ph2k5_50","ph2c_l10_50","k7_ph2g_50","k7_tail1ug_50","wrist_l10_50","cslG_50","cslGS_50","k5_g50_r1","k5_g50_r2","inf_l10_trace","inf_l10_s1001","inf_l10_s2001","inf_l10_L10","inf_l10_L10b"],
 "p_sp_500":  ["cl2_sp_500","sp_g500","sp_per12_500","sp_awm500_h70","ph2c_sp_500","tail1uc_sp_500","k7_sp_ph2g","k7_sp_tail1ug","wrist_sp_500","inf_sp_trace","inf_sp_s1001","inf_sp_s2001","inf_sp_L10"],
 "p_sp_50":   ["cl2_sp_50","sp_g50","sp_perk3_50","sp_awm_h70_50","sp_awm_h50_50","k1_sp50_ph2g","wrist_sp_50","inf_sp_trace","inf_sp_s1001","inf_sp_s2001","inf_sp_L10"],
 "g_l10_500": ["cl2_g_l10_500","gb_l10_500_ph2","gb_l10_500_tail2u","inf_g_l10_trace"],
 "g_l10_50":  ["cl2_g_l10_50","gb_l10_50_ph2","gb_l10_50_tail2u","inf_g_l10_trace"],
 "g_sp_500":  ["cl2_g_sp_500","gb_sp_500_ph2","gb_sp_500_tail2u","inf_g_sp_trace"],
 "g_sp_50":   ["cl2_g_sp_50","gb_sp_50_ph2","gb_sp_50_tail2u","inf_g_sp_trace"],
}
K2_APPENDIX = {"p_l10_500": ["g500_k2","perk5_500_k2","inf_l10_k2_s1101"], "p_l10_50": ["g50_k2","perk5_50_k2"], "p_sp_500": ["sp_g500_k2","inf_sp_k2_s1101"], "p_sp_50": ["sp_g50_k2"]}
PURE_CACHE = {"p_l10_500":"cl2_l10_500","p_l10_50":"cl2_l10_50","p_sp_500":"cl2_sp_500","p_sp_50":"cl2_sp_50","g_l10_500":"cl2_g_l10_500","g_l10_50":"cl2_g_l10_50","g_sp_500":"cl2_g_sp_500","g_sp_50":"cl2_g_sp_50"}
PURE_INF = {"p_l10_500":["inf_l10_trace","inf_l10_s1001","inf_l10_s2001"],"p_l10_50":["inf_l10_trace","inf_l10_s1001","inf_l10_s2001"],"p_sp_500":["inf_sp_trace","inf_sp_s1001"],"p_sp_50":["inf_sp_trace","inf_sp_s1001"],"g_l10_500":["inf_g_l10_trace"],"g_l10_50":["inf_g_l10_trace"],"g_sp_500":["inf_g_sp_trace"],"g_sp_50":["inf_g_sp_trace"]}

def pareto(points):
    """points: list of (label, ir, sr). dominated if exists other with ir<=ir and sr>=sr and (ir<ir or sr>sr)."""
    out = {}
    for l, ir, sr in points:
        dom = [o for o, oir, osr in points if o != l and oir <= ir + 1e-12 and osr >= sr - 1e-12 and (oir < ir - 1e-12 or osr > sr + 1e-12)]
        out[l] = dom
    return out

def chord(ir, ir0, sr0, ir1, sr1):
    if ir1 == ir0: return float("nan")
    return sr0 + (sr1 - sr0) * (ir - ir0) / (ir1 - ir0)

lines = []
res = {}
for cell, labs in CELLS.items():
    labs = [l for l in labs if "sr" in rows[l]]
    pc = PURE_CACHE[cell]
    infs = [l for l in PURE_INF[cell] if "sr" in rows[l]]
    sr_inf = float(np.mean([rows[l]["sr"] for l in infs]))
    sr_pc = rows[pc]["sr"]; ir_pc = rows[pc]["ir_owner"]
    lines.append(f"\n### cell {cell}  (pure cache {pc} {sr_pc:.3f} @ {ir_pc:.3f}; pure inference mean of {infs} = {sr_inf:.3f} @ 1.0)\n")
    lines.append("| arm | SR | IR owner | chord(CL2->inf) at IR | above chord (pp) | dominated by (owner) | IR owner+search | dominated (owner+search) | IR eager | dominated (eager) | MISS/ep | dec/ep | 10-step-chunk share |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for basis, key in [("owner","ir_owner"),("search","ir_owner_plus_search"),("eager","ir_eager")]:
        pts = [(l, rows[l][key], rows[l]["sr"]) for l in labs if rows[l].get(key) is not None]
        res.setdefault(cell, {})[basis] = pareto(pts)
    for l in sorted(labs, key=lambda x: rows[x]["ir_owner"]):
        r = rows[l]
        ch = chord(r["ir_owner"], ir_pc, sr_pc, 1.0, sr_inf) if r["family"] not in ("pure_inf","pure_inf_L10") else float("nan")
        above = (r["sr"] - ch) * 100 if not math.isnan(ch) else float("nan")
        dom_o = res[cell]["owner"].get(l, []); dom_s = res[cell]["search"].get(l, []); dom_e = res[cell]["eager"].get(l, [])
        # execution-length share: tails (blind after anchor_tail) -> 10-step chunks
        share10 = None
        if r["family"] in ("pure_inf_L10",): share10 = 1.0
        elif "tail" in l:
            share10 = 2 * (1 - r["v"])  # each blind tail belongs with its anchor
        elif r["family"] in ("pure_cache","mixed","pure_inf","wrist","k5","mixed_k2"): share10 = 0.0
        elif r["family"].startswith("blind") or r["family"] == "pure_cache_blind": share10 = 0.0  # phase/clock: new 5-step plan each decision
        g = lambda x, nd=3: "-" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
        lines.append(f"| {l} | {r['sr']:.3f} | {r['ir_owner']:.3f} | {g(ch)} | {g(above,1)} | {','.join(dom_o) if dom_o else 'FRONTIER'} | {g(r['ir_owner_plus_search'])} | {','.join(dom_s) if dom_s else ('FRONTIER' if r.get('ir_owner_plus_search') is not None else '-')} | {g(r['ir_eager'])} | {','.join(dom_e) if dom_e else 'FRONTIER'} | {g(r.get('miss_per_ep'),2)} | {g(r.get('dec_per_ep'),1)} | {g(share10,2)} |\n")
    if cell in K2_APPENDIX:
        lines.append("\nK2 appendix points (not on the frontier by owner ruling):\n")
        for l in K2_APPENDIX[cell]:
            r = rows[l]
            if "sr" in r: lines.append(f"- {l}: {r['sr']:.3f} @ owner {r['ir_owner']:.3f} (eager {r['ir_eager']:.3f}); K=2 priced as s2 + s3*2/10\n")

# pooled g500 reference test for l10-500 arms
def load(l):
    run, arm = an.ARMS[l][0], an.ARMS[l][1]
    return an.load_journal(run, arm)
refs = [load(l) for l in ["g500","g500_repa","g500_repb"]]
common = sorted(set(refs[0]) & set(refs[1]) & set(refs[2]))
refmean = {k: np.mean([r[k] for r in refs]) for k in common}
rng = np.random.default_rng(1)
lines.append("\n### l10 500-library arms vs the pooled stock g500 reference (per-init mean of the three stock runs .864/.850/.832; mean SR %.3f)\n" % np.mean(list(refmean.values())))
lines.append("| arm | SR | d vs pooled ref | init-bootstrap 95% | vs .864 | vs .850 | vs .832 | min p (3 McNemar) | max p |\n|---|---|---|---|---|---|---|---|---|\n")
pooled = {}
for l in ["k7_tail1ug","k7_ph2g","k7_ph1g","k7_b0g","k1_tail1ug","k1_clk1g","k1_ph2g","k1_ph1g","k1_b0g","k1_ph2k8","per8_500","per12_500","g500_np4","awm500_h70","wrist_l10_500","g500_k2","perk5_500_k2","inf_l10_s1001","inf_l10_s2001","inf_l10_trace","inf_l10_L10","inf_l10_L10b","cl2_l10_500","ph2c_l10_500"]:
    J = load(l)
    d = np.array([J[k] - refmean[k] for k in common])
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ps = [D["pairs"][f"{l}__vs__{b}"]["p_mcnemar"] for b in ["g500","g500_repa","g500_repb"] if f"{l}__vs__{b}" in D["pairs"] and "p_mcnemar" in D["pairs"][f"{l}__vs__{b}"]]
    ds = [D["pairs"][f"{l}__vs__{b}"]["delta"] for b in ["g500","g500_repa","g500_repb"] if f"{l}__vs__{b}" in D["pairs"] and "delta" in D["pairs"][f"{l}__vs__{b}"]]
    pooled[l] = {"d": float(d.mean()), "ci": [float(lo), float(hi)], "p_each": ps, "d_each": ds}
    lines.append(f"| {l} | {rows[l]['sr']:.3f} | {d.mean():+.3f} | [{lo:+.3f},{hi:+.3f}] | " + " | ".join(f"{x:+.3f}" for x in ds) + (" | - | - | - " if not ds else "") + f" | {min(ps) if ps else float('nan'):.3f} | {max(ps) if ps else float('nan'):.3f} |\n")

# noise-floor summary
lines.append("\n### noise floor\n")
for a, b in [("g500_repa","g500"),("g500_repb","g500"),("g500_repb","g500_repa"),("inf_l10_s1001","inf_l10_trace"),("inf_l10_s2001","inf_l10_trace"),("inf_l10_s2001","inf_l10_s1001"),("inf_l10_L10b","inf_l10_L10"),("inf_sp_s1001","inf_sp_trace"),("k5_g500_r2","k5_g500_r1"),("k5_g50_r2","k5_g50_r1")]:
    p = D["pairs"][f"{a}__vs__{b}"]
    if "delta" in p:
        lines.append(f"- {a} vs {b}: dSR {p['delta']:+.3f}, discordant {p['discordant']}/{p['n']} ({p['discordant']/p['n']*100:.1f}%), SE(d) {p['se_delta']:.4f}, p {p['p_mcnemar']:.3f}\n")
srs = [rows[l]["sr"] for l in ["g500","g500_repa","g500_repb"]]
lines.append(f"- stock g500 three runs: {srs}, mean {np.mean(srs):.4f}, sd {np.std(srs, ddof=1):.4f}, SE of mean {np.std(srs, ddof=1)/math.sqrt(3):.4f}\n")
srs = [rows[l]["sr"] for l in ["inf_l10_trace","inf_l10_s1001","inf_l10_s2001"]]
lines.append(f"- l10 pure inference L=5 three runs: {srs}, mean {np.mean(srs):.4f}, sd {np.std(srs, ddof=1):.4f}\n")

open(f"{S}/frontier.md", "w").write("".join(lines))
json.dump({"pareto": res, "pooled_g500": pooled}, open(f"{S}/frontier.json", "w"), indent=1)
print("".join(lines))
