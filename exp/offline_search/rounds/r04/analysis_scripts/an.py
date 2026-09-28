"""R4 analysis: journals -> paired tests, owner/eager/search IR, frontier tables.

Read-only on /home/weiland/trace_runs; writes only into this scratch directory.
"""
from __future__ import annotations
import json, math, os, sys, itertools
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
CL = "/home/weiland/trace_runs/os_closed_loop"
DUAL = "/home/weiland/trace_runs/dual_20260923"
OWNER_TABLE = json.load(open("/home/weiland/projects/openpi/exp/offline_search/rounds/r04/cost_table_owner.json"))
EAGER_TABLE = json.load(open("/home/weiland/projects/openpi/exp/offline_search/closed_loop/ops/cost_table.json"))

# ------------------------------------------------------------------ arms
# label: (run, arm, cell, lib, family, method_for_search_cost)
ARMS = {
 # --- pure inference references
 "inf_l10_trace":   ("DUAL", "tr_pi05_l10_inf", "p_l10", "-", "pure_inf", "none"),
 "inf_sp_trace":    ("DUAL", "tr_pi05_sp_inf", "p_sp", "-", "pure_inf", "none"),
 "inf_g_l10_trace": ("DUAL", "tr_groot_l10_inf", "g_l10", "-", "pure_inf", "none"),
 "inf_g_sp_trace":  ("DUAL", "tr_groot_sp_inf", "g_sp", "-", "pure_inf", "none"),
 "inf_l10_s1001":   ("r04_cost", "r4f_p_l10_inf_s1001", "p_l10", "-", "pure_inf", "none"),
 "inf_l10_s2001":   ("r04_cost", "r4f_p_l10_inf_s2001", "p_l10", "-", "pure_inf", "none"),
 "inf_sp_s1001":    ("r04_cost", "r4f_p_sp_inf_s1001", "p_sp", "-", "pure_inf", "none"),
 "inf_sp_s2001":    ("r04_cost", "r4f_p_sp_inf_s2001", "p_sp", "-", "pure_inf", "none"),
 "inf_l10_k2_s1101":("r04_cost", "r4b2_p_l10_inf_k2_s1101", "p_l10", "-", "pure_inf_k2", "none"),
 "inf_sp_k2_s1101": ("r04_cost", "r4b2_p_sp_inf_k2_s1101", "p_sp", "-", "pure_inf_k2", "none"),
 "inf_l10_L10":     ("r04_cost", "r4f_p_l10_inf_k10_L10", "p_l10", "-", "pure_inf_L10", "none"),
 "inf_l10_L10b":    ("r04_blind", "r4b3_p_l10_50_inferL10", "p_l10", "-", "pure_inf_L10", "none"),
 "inf_sp_L10":      ("r04_cost", "r4f_p_sp_inf_k10_L10", "p_sp", "-", "pure_inf_L10", "none"),
 # --- pure cache references (R2 CL2)
 "cl2_l10_50":  ("r02_g50", "oscl50_p_l10_cl2", "p_l10", "50", "pure_cache", "AWM"),
 "cl2_l10_500": ("r02_g500", "oscl500_p_l10_cl2", "p_l10", "500", "pure_cache", "AWM"),
 "cl2_sp_50":   ("r02_g50", "oscl50_p_sp_cl2", "p_sp", "50", "pure_cache", "AWM"),
 "cl2_sp_500":  ("r02_g500", "oscl500_p_sp_cl2", "p_sp", "500", "pure_cache", "AWM"),
 "cl2_g_l10_50":  ("r02_g50", "oscl50_g_l10_cl2", "g_l10", "50", "pure_cache", "AWM"),
 "cl2_g_l10_500": ("r02_g500", "oscl500_g_l10_cl2", "g_l10", "500", "pure_cache", "AWM"),
 "cl2_g_sp_50":   ("r02_g50", "oscl50_g_sp_cl2", "g_sp", "50", "pure_cache", "AWM"),
 "cl2_g_sp_500":  ("r02_g500", "oscl500_g_sp_cl2", "g_sp", "500", "pure_cache", "AWM"),
 # --- R3 mixed
 "g50":        ("r03_mx", "r3mx_p_l10_g", "p_l10", "50", "mixed", "MixedJudge"),
 "g500":       ("r03_mx", "r3mx_p_l10_g500", "p_l10", "500", "mixed", "MixedJudge"),
 "g500_repa":  ("r04_rep", "r4rep_p_l10_g500_a", "p_l10", "500", "mixed", "MixedJudge"),
 "g500_repb":  ("r04_rep", "r4rep_p_l10_g500_b", "p_l10", "500", "mixed", "MixedJudge"),
 "perk5_50":   ("r03_mx", "r3mx_p_l10_perk5", "p_l10", "50", "mixed", "AWM"),
 "perk3_50":   ("r03_mx", "r3mx_p_l10_perk3", "p_l10", "50", "mixed", "AWM"),
 "awm_h70_50": ("r03_mx", "r3mx_p_l10_awm_h70", "p_l10", "50", "mixed", "MixedJudge"),
 "awm_h50_50": ("r03_mx", "r3mx_p_l10_awm_h50", "p_l10", "50", "mixed", "MixedJudge"),
 "awm500_h70": ("r03_mx", "r3mx_p_l10_awm500_h70", "p_l10", "500", "mixed", "MixedJudge"),
 "sp_g50":     ("r03_mx", "r3mx_p_sp_g", "p_sp", "50", "mixed", "MixedJudge"),
 "sp_perk3_50":("r03_mx", "r3mx_p_sp_perk3", "p_sp", "50", "mixed", "AWM"),
 "sp_awm_h70_50": ("r03_mx", "r3mx_p_sp_awm_h70", "p_sp", "50", "mixed", "MixedJudge"),
 "sp_awm_h50_50": ("r03_mx", "r3mx_p_sp_awm_h50", "p_sp", "50", "mixed", "MixedJudge"),
 "sp_awm500_h70": ("r03_mx", "r3mx_p_sp_awm500_h70", "p_sp", "500", "mixed", "MixedJudge"),
 # --- R4 batch 1
 "g500_np4":   ("r04_frontier", "r4_p_l10_g500_np4", "p_l10", "500", "mixed", "MixedJudge"),
 "per8_500":   ("r04_frontier", "r4_p_l10_per8_500", "p_l10", "500", "mixed", "AWM"),
 "per12_500":  ("r04_frontier", "r4_p_l10_per12_500", "p_l10", "500", "mixed", "AWM"),
 "g50_np4":    ("r04_frontier", "r4_p_l10_g50_np4", "p_l10", "50", "mixed", "MixedJudge"),
 "per6_50":    ("r04_frontier", "r4_p_l10_per6_50", "p_l10", "50", "mixed", "AWM"),
 "sp_g500":    ("r04_frontier", "r4_p_sp_g500", "p_sp", "500", "mixed", "MixedJudge"),
 "sp_per12_500": ("r04_frontier", "r4_p_sp_per12_500", "p_sp", "500", "mixed", "AWM"),
 # --- K2 appendix
 "g500_k2":     ("r04_cost", "r4b2_p_l10_g500_k2", "p_l10", "500", "mixed_k2", "MixedJudge"),
 "g50_k2":      ("r04_cost", "r4b2_p_l10_g_k2", "p_l10", "50", "mixed_k2", "MixedJudge"),
 "perk5_500_k2":("r04_cost", "r4b2_p_l10_perk5_500_k2", "p_l10", "500", "mixed_k2", "AWM"),
 "perk5_50_k2": ("r04_cost", "r4b2_p_l10_perk5_k2", "p_l10", "50", "mixed_k2", "AWM"),
 "sp_g500_k2":  ("r04_cost", "r4b2_p_sp_g500_k2", "p_sp", "500", "mixed_k2", "MixedJudge"),
 "sp_g50_k2":   ("r04_cost", "r4b2_p_sp_g_k2", "p_sp", "50", "mixed_k2", "MixedJudge"),
 # --- K1 blind (dense guard)
 "k1_b0g":     ("r04_blind", "r4b3_p_l10_500_b0g", "p_l10", "500", "blind_k1", "K1"),
 "k1_ph1g":    ("r04_blind", "r4b3_p_l10_500_ph1g", "p_l10", "500", "blind_k1", "K1"),
 "k1_ph2g":    ("r04_blind", "r4b3_p_l10_500_ph2g", "p_l10", "500", "blind_k1", "K1"),
 "k1_tail1ug": ("r04_blind", "r4b3_p_l10_500_tail1ug", "p_l10", "500", "blind_k1", "K1tail"),
 "k1_clk1g":   ("r04_blind", "r4b3_p_l10_500_clk1g", "p_l10", "500", "blind_k1", "K1"),
 "k1_ph2k8":   ("r04_blind", "r4b3_p_l10_500_ph2k8", "p_l10", "500", "blind_k1_periodic", "BlindAWM"),
 "k1_ph2k5_50":("r04_blind", "r4b3_p_l10_50_ph2k5", "p_l10", "50", "blind_k1_periodic", "BlindAWM"),
 "ph2c_l10_500": ("r04_blind", "r4b3_p_l10_500_ph2c", "p_l10", "500", "pure_cache_blind", "BlindAWM"),
 "ph2c_l10_50":  ("r04_blind", "r4b3_p_l10_50_ph2c", "p_l10", "50", "pure_cache_blind", "BlindAWM"),
 "ph2c_sp_500":  ("r04_blind", "r4b3_p_sp_500_ph2c", "p_sp", "500", "pure_cache_blind", "BlindAWM"),
 "tail1uc_sp_500": ("r04_blind", "r4b3_p_sp_500_tail1uc", "p_sp", "500", "pure_cache_blind", "BlindAWMtail"),
 "k1_sp50_ph2g": ("r04_blind", "r4b3_p_sp_50_ph2g", "p_sp", "50", "blind_k1", "K1"),
 # --- K7 blind (vision-confirmed guard)
 "k7_b0g":     ("r04_k7", "r4k7_p_l10_500_b0g", "p_l10", "500", "blind_k7", "K7b0"),
 "k7_ph2g":    ("r04_k7", "r4k7_p_l10_500_ph2g", "p_l10", "500", "blind_k7", "K7ph2"),
 "k7_ph1g":    ("r04_k7", "r4k7_p_l10_500_ph1g", "p_l10", "500", "blind_k7", "K7ph1"),
 "k7_tail1ug": ("r04_k7", "r4k7_p_l10_500_tail1ug", "p_l10", "500", "blind_k7", "K7tail"),
 "k7_ph2g_50": ("r04_k7", "r4k7_p_l10_50_ph2g", "p_l10", "50", "blind_k7", "K7ph2"),
 "k7_tail1ug_50": ("r04_k7", "r4k7_p_l10_50_tail1ug", "p_l10", "50", "blind_k7", "K7tail"),
 "k7_sp_ph2g": ("r04_k7", "r4k7_p_sp_500_ph2g", "p_sp", "500", "blind_k7", "K7ph2"),
 "k7_sp_tail1ug": ("r04_k7", "r4k7_p_sp_500_tail1ug", "p_sp", "500", "blind_k7", "K7tail"),
 # --- wrist
 "wrist_l10_500": ("r04_b4w", "r4b4_p_l10_g500_wrist", "p_l10", "500", "wrist", "WristMixedJudge"),
 "wrist_l10_50":  ("r04_b4w", "r4b4_p_l10_g50_wrist", "p_l10", "50", "wrist", "WristMixedJudge"),
 "wrist_sp_500":  ("r04_b4w", "r4b4_p_sp_g500_wrist", "p_sp", "500", "wrist", "WristMixedJudge"),
 "wrist_sp_50":   ("r04_b4w", "r4b4_p_sp_g50_wrist", "p_sp", "50", "wrist", "WristMixedJudge"),
 # --- csl
 "cslG_500":  ("r04_csl", "r4b4_p_l10_500_cslG", "p_l10", "500", "pure_cache", "ControlG"),
 "cslGS_500": ("r04_csl", "r4b4_p_l10_500_cslGS", "p_l10", "500", "pure_cache", "ControlGS"),
 "cslG_50":   ("r04_csl", "r4b4_p_l10_50_cslG", "p_l10", "50", "pure_cache", "ControlG"),
 "cslGS_50":  ("r04_csl", "r4b4_p_l10_50_cslGS", "p_l10", "50", "pure_cache", "ControlGS"),
 # --- K5 randomized (half CALL / half CACHE overlay; NOT stock replicates)
 "k5_g500_r1": ("r04_k5", "r4k5_p_l10_g500_r1", "p_l10", "500", "k5", "MixedJudge"),
 "k5_g500_r2": ("r04_k5", "r4k5_p_l10_g500_r2", "p_l10", "500", "k5", "MixedJudge"),
 "k5_g50_r1":  ("r04_k5", "r4k5_p_l10_g50_r1", "p_l10", "50", "k5", "MixedJudge"),
 "k5_g50_r2":  ("r04_k5", "r4k5_p_l10_g50_r2", "p_l10", "50", "k5", "MixedJudge"),
 # --- GR00T blind (pure cache), TODO until DONE
 "gb_l10_500_ph2":   ("r04_gblind", "r4b3_g_l10_500_ph2", "g_l10", "500", "pure_cache_blind", "BlindAWM"),
 "gb_l10_500_tail2u":("r04_gblind", "r4b3_g_l10_500_tail2u", "g_l10", "500", "pure_cache_blind", "BlindAWMtail"),
 "gb_l10_50_ph2":    ("r04_gblind", "r4b3_g_l10_50_ph2", "g_l10", "50", "pure_cache_blind", "BlindAWM"),
 "gb_l10_50_tail2u": ("r04_gblind", "r4b3_g_l10_50_tail2u", "g_l10", "50", "pure_cache_blind", "BlindAWMtail"),
 "gb_sp_500_ph2":    ("r04_gblind", "r4b3_g_sp_500_ph2", "g_sp", "500", "pure_cache_blind", "BlindAWM"),
 "gb_sp_500_tail2u": ("r04_gblind", "r4b3_g_sp_500_tail2u", "g_sp", "500", "pure_cache_blind", "BlindAWMtail"),
 "gb_sp_50_ph2":     ("r04_gblind", "r4b3_g_sp_50_ph2", "g_sp", "50", "pure_cache_blind", "BlindAWM"),
 "gb_sp_50_tail2u":  ("r04_gblind", "r4b3_g_sp_50_tail2u", "g_sp", "50", "pure_cache_blind", "BlindAWMtail"),
}

# K8 controlled single-thread method p50 (ms) per vision decision / per blind decision, by (method_key, cell, lib)
K8 = {
 ("AWM","p_sp","50"):(1.154,None), ("AWM","p_sp","500"):(1.369,None), ("AWM","p_l10","50"):(1.167,None), ("AWM","p_l10","500"):(1.405,None),
 ("AWM","g_sp","50"):(1.138,None), ("AWM","g_sp","500"):(1.253,None), ("AWM","g_l10","50"):(1.188,None), ("AWM","g_l10","500"):(1.383,None),
 ("MixedJudge","p_sp","50"):(1.530,None), ("MixedJudge","p_sp","500"):(2.113,None), ("MixedJudge","p_l10","50"):(1.646,None), ("MixedJudge","p_l10","500"):(2.096,None),
 ("K1","p_sp","50"):(1.796,0.475), ("K1","p_sp","500"):(2.184,0.478), ("K1","p_l10","50"):(1.786,0.468), ("K1","p_l10","500"):(2.128,0.467),
 ("K1tail","p_l10","500"):(2.128,0.223),
 ("BlindAWM","p_sp","50"):(1.308,0.459), ("BlindAWM","p_sp","500"):(1.335,0.491), ("BlindAWM","p_l10","50"):(1.272,0.482), ("BlindAWM","p_l10","500"):(1.426,0.478),
 ("BlindAWM","g_sp","50"):(1.201,0.461), ("BlindAWM","g_sp","500"):(1.300,0.478), ("BlindAWM","g_l10","50"):(1.232,0.465), ("BlindAWM","g_l10","500"):(1.432,0.469),
 ("BlindAWMtail","p_sp","500"):(1.335,0.223),
 ("BlindAWMtail","g_sp","50"):(1.201,0.223), ("BlindAWMtail","g_sp","500"):(1.300,0.223), ("BlindAWMtail","g_l10","50"):(1.232,0.223), ("BlindAWMtail","g_l10","500"):(1.432,0.223),
 ("WristMixedJudge","p_sp","50"):(1.265,None), ("WristMixedJudge","p_sp","500"):(1.393,None), ("WristMixedJudge","p_l10","50"):(1.269,None), ("WristMixedJudge","p_l10","500"):(1.584,None),
 ("ControlG","p_l10","50"):(1.261,None), ("ControlG","p_l10","500"):(1.658,None), ("ControlGS","p_l10","50"):(1.310,None), ("ControlGS","p_l10","500"):(1.771,None),
 ("K7b0","p_l10","500"):(2.091,None), ("K7ph1","p_l10","500"):(2.045,0.469), ("K7ph2","p_l10","500"):(2.096,0.483), ("K7tail","p_l10","500"):(2.097,0.223),
 ("K7ph2","p_l10","50"):(1.686,0.493), ("K7tail","p_l10","50"):(1.657,0.238), ("K7ph2","p_sp","500"):(1.798,0.478), ("K7tail","p_sp","500"):(1.804,0.223),
 ("none","p_sp","-"):(0.0,None), ("none","p_l10","-"):(0.0,None), ("none","g_sp","-"):(0.0,None), ("none","g_l10","-"):(0.0,None),
}
OWNER_FULL_MS = 67.5  # owner rounded denominator (K8 convention)

# ------------------------------------------------------------------ journals
def load_journal(run, arm):
    if run == "DUAL":
        p = f"{DUAL}/runs/{arm}/client/journal.jsonl"
    else:
        p = f"{CL}/{run}/runs/{arm}/client/journal.jsonl"
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not (r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error")):
            continue
        uid = r["task_uid"]
        parts = uid.split(":")
        task, init = int(parts[-2]), int(parts[-1])
        out[(task, init)] = bool(r.get("success"))
    return out

def is_done(run, arm):
    if run == "DUAL":
        return True
    return os.path.exists(f"{CL}/{run}/state/{arm}.DONE") and not os.path.exists(f"{CL}/{run}/state/{arm}.SKIPPED")

def summary(run, arm):
    if run == "DUAL":
        return None
    p = f"{CL}/{run}/summary.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get(arm)

# ------------------------------------------------------------------ stats
def wilson(k, n, z=1.959963984540054):
    if n <= 0: return (float("nan"), float("nan"))
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0, c-h), min(1, c+h))

def mcnemar_exact(b, c):
    m = b + c
    if m == 0: return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k+1))
    return float(min(1.0, 2.0*tail/(2**m)))

def paired(A, B, boot=10000, seed=0):
    """A minus B on common inits. returns dict."""
    common = sorted(set(A) & set(B))
    n = len(common)
    a = b = c = d = 0
    per_task = {}
    for k in common:
        sa, sb = A[k], B[k]
        t = k[0]
        pt = per_task.setdefault(t, {"n":0, "A":0, "B":0, "AwinBlose":0, "AloseBwin":0})
        pt["n"] += 1; pt["A"] += sa; pt["B"] += sb
        if sa and sb: a += 1
        elif sa and not sb: c += 1; pt["AwinBlose"] += 1
        elif (not sa) and sb: b += 1; pt["AloseBwin"] += 1
        else: d += 1
    dsr = (c - b) / n if n else float("nan")
    p = mcnemar_exact(b, c)
    # multinomial bootstrap of delta
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, [a/n, b/n, c/n, d/n], size=boot)
    deltas = (counts[:, 2] - counts[:, 1]) / n
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"n": n, "srA": sum(A[k] for k in common)/n, "srB": sum(B[k] for k in common)/n,
            "delta": dsr, "A_S_B_F": c, "A_F_B_S": b, "both_S": a, "both_F": d,
            "discordant": b + c, "p_mcnemar": p, "ci95": [float(lo), float(hi)],
            "se_delta": float(math.sqrt((b + c) - (c - b)**2 / n) / n) if n else float("nan"),
            "per_task": per_task}

# ------------------------------------------------------------------ IR
def owner_ir(cell, v, m, L, mode, k=10.0):
    model = "groot" if cell.startswith("g_") else "pi05"
    if model == "pi05":
        md = OWNER_TABLE["models"]["pi05"]["modes"][mode]
        s1, s2, s3 = md["s1"], md["s2"], md["s3"]
        extra = md.get("miss_s1_extra", 0.0)
        ir = v*s1 + m*(extra + s2 + s3*k/10.0)
    else:
        s1, s2, s3 = 0.148, 0.174, 0.678
        ir = v*s1 + m*(s2 + s3*k/8.0)
    return ir * 5.0 / L

def eager_ir(cell, v, m, L, mode, k=10.0):
    model = "groot" if cell.startswith("g_") else "pi05"
    md = EAGER_TABLE["models"][model]["modes"][mode] if mode in EAGER_TABLE["models"][model]["modes"] else EAGER_TABLE["models"][model]["full"]
    full = EAGER_TABLE["models"][model]["full_cost_ms"]
    kref = EAGER_TABLE["models"][model]["full"]["k"]
    extra = md.get("miss_s1_extra", 0.0)
    ir = (v*md["s1"] + m*(extra + md["s2"] + md["s3"]*k/kref)) / full
    return ir * 5.0 / L

def search_ms(method_key, cell, lib, v):
    q = K8.get((method_key, cell, lib))
    if q is None:
        return None
    qv, qb = q
    if qb is None: qb = 0.0
    return v*qv + (1-v)*qb

def arm_row(label):
    run, arm, cell, lib, fam, mk = ARMS[label]
    done = is_done(run, arm)
    J = load_journal(run, arm) if done else None
    sm = summary(run, arm)
    row = {"label": label, "run": run, "arm": arm, "cell": cell, "lib": lib, "family": fam, "done": done}
    if J is None or (sm is None and run != "DUAL"):
        row["status"] = "TODO (not DONE / no summary)"
        return row, None
    n = len(J); k = sum(J.values())
    row.update({"n": n, "success": k, "sr": k/n, "wilson": wilson(k, n)})
    row["per_task_sr"] = {t: sum(s for (tt, i), s in J.items() if tt == t)/sum(1 for (tt, i) in J if tt == t) for t in range(10)}
    # cost inputs
    v = m = L = K = None; mode = "full"
    if run == "DUAL":
        v, m, L, K = 1.0, 1.0, 5.0, 10.0 if cell.startswith("p_") else 8.0
        dec = None
    else:
        cl = sm.get("cost_ledger") or {}
        mx = sm.get("mixed") or {}
        dec = sm.get("decisions_per_episode")
        if cl:
            v = cl["v"]; m = cl["m"]; L = cl["l_per_request"]["mean"]
            K = (cl.get("k_per_miss") or {}).get("mean") or (10.0 if cell.startswith("p_") else 8.0)
            modes = cl.get("stage1_modes") or {"full": 1}
            mode = list(modes.keys())[0]
            if len(modes) > 1: row["warn_modes"] = modes
        elif mx and mx.get("h") is not None:
            v, m, L, K = 1.0, 1.0 - mx["h"], 5.0, 10.0
        else:  # pure cache legacy
            v, m, L, K = 1.0, 0.0, 5.0, 10.0
        if fam == "pure_inf":
            v, m = 1.0, 1.0
        row["dec_per_ep"] = dec
        row["judge_mix"] = mx.get("judge_mix")
        row["miss_per_ep"] = mx.get("miss_per_episode")
        row["eager_ledger_ir"] = (cl or {}).get("ir_per_five_controls")
        row["summary_sr"] = sm.get("sr")
    row.update({"v": v, "m": m, "L": L, "K": K, "stage1_mode": mode})
    row["ir_owner"] = owner_ir(cell, v, m, L, mode, K)
    row["ir_owner_fullS1"] = owner_ir(cell, v, m, L, "full", K)  # for wrist: price stage1 at full
    row["ir_eager"] = eager_ir(cell, v, m, L, mode, K)
    sms = search_ms(mk, cell, lib, v)
    row["search_ms_per_dec"] = sms
    row["ir_owner_plus_search"] = None if sms is None else row["ir_owner"] + (sms/OWNER_FULL_MS)*(5.0/L)
    # K9: GPU-resident retrieval in the stage-1 graph; vision-side increment .36-.63 ms; blind step stays CPU (not implemented on GPU)
    if sms is not None and mk != "none":
        gpu_vis = 0.40 if mk in ("AWM", "BlindAWM", "BlindAWMtail", "ControlG", "ControlGS") else 0.60
        qb = K8[(mk, cell, lib)][1] or 0.0
        row["ir_owner_plus_search_k9"] = row["ir_owner"] + ((v*gpu_vis + (1-v)*qb)/OWNER_FULL_MS)*(5.0/L)
    else:
        row["ir_owner_plus_search_k9"] = row["ir_owner"] if sms is not None else None
    return row, J

# pairs: (A, B) meaning A minus B
PAIRS = [
 # noise floor
 ("g500_repa","g500"), ("g500_repb","g500"), ("g500_repb","g500_repa"),
 ("inf_l10_s1001","inf_l10_trace"), ("inf_l10_s2001","inf_l10_trace"), ("inf_l10_s2001","inf_l10_s1001"),
 ("inf_l10_k2_s1101","inf_l10_s1001"), ("inf_l10_k2_s1101","inf_l10_trace"),
 ("inf_sp_s1001","inf_sp_trace"), ("inf_sp_s2001","inf_sp_trace"), ("inf_sp_s2001","inf_sp_s1001"), ("inf_sp_k2_s1101","inf_sp_s1001"),
 ("inf_l10_L10b","inf_l10_L10"),
 ("k5_g500_r2","k5_g500_r1"), ("k5_g50_r2","k5_g50_r1"),
 # execution length
 ("inf_l10_L10","inf_l10_s1001"), ("inf_l10_L10","inf_l10_s2001"), ("inf_l10_L10","inf_l10_trace"), ("inf_l10_L10","inf_l10_k2_s1101"),
 ("inf_l10_L10b","inf_l10_s1001"), ("inf_l10_L10b","inf_l10_s2001"), ("inf_l10_L10b","inf_l10_trace"),
 ("inf_sp_L10","inf_sp_s1001"), ("inf_sp_L10","inf_sp_trace"), ("inf_sp_L10","inf_sp_s2001"),
 # g500 vs inference
 ("g500","inf_l10_trace"), ("g500_repa","inf_l10_s1001"), ("g500_repb","inf_l10_s2001"), ("g500","inf_l10_L10"),
 # l10 500 blind arms vs stock (three runs) and controls
 *[(a, b) for a in ["k7_ph2g","k7_tail1ug","k7_ph1g","k7_b0g","k1_ph2g","k1_tail1ug","k1_ph1g","k1_b0g","k1_clk1g","k1_ph2k8","per8_500","per12_500","g500_np4","wrist_l10_500","awm500_h70","g500_k2","perk5_500_k2"] for b in ["g500","g500_repa","g500_repb"]],
 ("k7_ph2g","k7_b0g"), ("k7_tail1ug","k7_b0g"), ("k7_ph1g","k7_b0g"), ("k7_tail1ug","k7_ph2g"), ("k7_ph1g","k7_ph2g"),
 ("k1_ph2g","k1_b0g"), ("k1_tail1ug","k1_b0g"), ("k1_ph1g","k1_b0g"), ("k1_clk1g","k1_b0g"), ("k1_tail1ug","k1_clk1g"), ("k1_clk1g","k1_ph1g"), ("k1_tail1ug","k1_ph1g"),
 ("k7_ph2g","k1_ph2g"), ("k7_tail1ug","k1_tail1ug"), ("k7_b0g","k1_b0g"), ("k7_ph1g","k1_ph1g"),
 ("k1_ph2k8","per8_500"), ("k1_ph2k8","k1_ph2g"), ("k1_ph2k8","k7_ph2g"),
 ("k7_tail1ug","inf_l10_L10"), ("k7_tail1ug","inf_l10_L10b"), ("k7_tail1ug","inf_l10_s1001"), ("k7_tail1ug","inf_l10_s2001"), ("k7_tail1ug","inf_l10_trace"),
 ("k7_ph2g","inf_l10_s1001"), ("k7_ph2g","inf_l10_trace"), ("k7_ph2g","inf_l10_L10"),
 ("k1_tail1ug","inf_l10_L10"), ("k1_tail1ug","inf_l10_s1001"),
 ("k7_tail1ug","awm500_h70"), ("k7_tail1ug","per12_500"), ("k7_ph2g","per12_500"),
 ("k7_tail1ug","cl2_l10_500"), ("k7_ph2g","cl2_l10_500"),
 # pure-cache blind vs CL2
 ("ph2c_l10_500","cl2_l10_500"), ("ph2c_l10_50","cl2_l10_50"), ("ph2c_sp_500","cl2_sp_500"),
 ("tail1uc_sp_500","cl2_sp_500"), ("tail1uc_sp_500","ph2c_sp_500"), ("tail1uc_sp_500","inf_sp_s1001"), ("tail1uc_sp_500","inf_sp_trace"), ("tail1uc_sp_500","inf_sp_s2001"), ("tail1uc_sp_500","inf_sp_L10"), ("tail1uc_sp_500","inf_sp_k2_s1101"),
 ("tail1uc_sp_500","k7_sp_tail1ug"), ("tail1uc_sp_500","sp_g500"), ("tail1uc_sp_500","k7_sp_ph2g"),
 ("ph2c_l10_500","g500"), ("ph2c_l10_500","k7_ph2g"),
 # l10 50 cell
 *[(a, b) for a in ["k7_tail1ug_50","k7_ph2g_50","k1_ph2k5_50","per6_50","g50_np4","wrist_l10_50","g50_k2","perk5_50_k2","perk5_50","perk3_50","awm_h70_50"] for b in ["g50"]],
 ("k7_tail1ug_50","perk5_50"), ("k7_tail1ug_50","per6_50"), ("k7_tail1ug_50","perk3_50"), ("k7_tail1ug_50","k7_ph2g_50"), ("k7_tail1ug_50","cl2_l10_50"), ("k7_ph2g_50","cl2_l10_50"),
 ("k1_ph2k5_50","perk5_50"), ("k1_ph2k5_50","k7_ph2g_50"), ("k7_tail1ug_50","inf_l10_s1001"), ("k7_tail1ug_50","awm_h70_50"), ("k7_tail1ug_50","k7_tail1ug"),
 ("k7_ph2g_50","k7_ph2g"), ("k5_g50_r1","g50"), ("k5_g50_r2","g50"), ("k5_g500_r1","g500"), ("k5_g500_r2","g500"),
 # spatial 500
 *[(a, b) for a in ["k7_sp_tail1ug","k7_sp_ph2g","sp_per12_500","wrist_sp_500","sp_g500_k2","sp_awm500_h70","ph2c_sp_500"] for b in ["sp_g500"]],
 ("sp_g500","cl2_sp_500"), ("k7_sp_tail1ug","cl2_sp_500"), ("k7_sp_ph2g","cl2_sp_500"), ("k7_sp_tail1ug","k7_sp_ph2g"),
 ("k7_sp_tail1ug","inf_sp_s1001"), ("k7_sp_tail1ug","inf_sp_trace"), ("k7_sp_tail1ug","inf_sp_L10"), ("k7_sp_ph2g","inf_sp_s1001"), ("sp_g500","inf_sp_s1001"), ("sp_g500","inf_sp_trace"),
 ("k7_sp_tail1ug","sp_awm500_h70"), ("k7_sp_tail1ug","sp_awm_h50_50"),
 # spatial 50
 ("wrist_sp_50","sp_g50"), ("k1_sp50_ph2g","sp_g50"), ("sp_g50_k2","sp_g50"), ("sp_perk3_50","sp_g50"), ("sp_awm_h70_50","sp_g50"), ("k1_sp50_ph2g","cl2_sp_50"), ("wrist_sp_50","cl2_sp_50"), ("k1_sp50_ph2g","wrist_sp_50"),
 ("wrist_sp_50","cl2_sp_500"),
 # wrist
 ("wrist_l10_500","k7_b0g"), ("wrist_l10_50","g50"), ("wrist_sp_500","sp_g500"), ("wrist_sp_50","inf_sp_s1001"), ("wrist_l10_500","wrist_l10_50"),
 # csl
 ("cslG_500","cl2_l10_500"), ("cslGS_500","cl2_l10_500"), ("cslG_50","cl2_l10_50"), ("cslGS_50","cl2_l10_50"), ("cslGS_500","cslG_500"),
 # K2 appendix
 ("g500_k2","g500"), ("g50_k2","g50"), ("perk5_500_k2","per8_500"), ("perk5_50_k2","perk5_50"), ("sp_g500_k2","sp_g500"), ("sp_g50_k2","sp_g50"), ("inf_l10_k2_s1101","inf_l10_s1001"),
 # groot blind (TODO until done)
 ("gb_l10_500_ph2","cl2_g_l10_500"), ("gb_l10_500_tail2u","cl2_g_l10_500"), ("gb_l10_500_tail2u","gb_l10_500_ph2"), ("gb_l10_500_tail2u","inf_g_l10_trace"), ("gb_l10_500_ph2","inf_g_l10_trace"),
 ("gb_l10_50_ph2","cl2_g_l10_50"), ("gb_l10_50_tail2u","cl2_g_l10_50"), ("gb_l10_50_tail2u","gb_l10_50_ph2"),
 ("gb_sp_500_ph2","cl2_g_sp_500"), ("gb_sp_500_tail2u","cl2_g_sp_500"), ("gb_sp_500_tail2u","gb_sp_500_ph2"), ("gb_sp_500_tail2u","inf_g_sp_trace"), ("gb_sp_500_ph2","inf_g_sp_trace"),
 ("gb_sp_50_ph2","cl2_g_sp_50"), ("gb_sp_50_tail2u","cl2_g_sp_50"), ("gb_sp_50_tail2u","gb_sp_50_ph2"),
 ("cl2_g_l10_500","inf_g_l10_trace"), ("cl2_g_sp_500","inf_g_sp_trace"),
]

def main():
    rows = {}; J = {}
    for lab in ARMS:
        r, j = arm_row(lab)
        rows[lab] = r
        if j is not None: J[lab] = j
    pairs = {}
    for a, b in PAIRS:
        if a in J and b in J:
            pairs[f"{a}__vs__{b}"] = paired(J[a], J[b])
        else:
            pairs[f"{a}__vs__{b}"] = {"status": "TODO", "missing": [x for x in (a, b) if x not in J]}
    json.dump({"arms": rows, "pairs": pairs}, open(f"{S}/an.json", "w"), indent=1, default=float)
    # markdown
    with open(f"{S}/arms.md", "w") as f:
        f.write("| label | run:arm | cell | lib | fam | n | SR | Wilson | v | m | L | K | s1mode | IR owner | IR owner (S1 full) | IR eager | search ms/dec | IR owner+search | IR owner+search(K9) | dec/ep | MISS/ep |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for lab, r in rows.items():
            if "sr" not in r:
                f.write(f"| {lab} | {r['run']}:{r['arm']} | {r['cell']} | {r['lib']} | {r['family']} | TODO |||||||||||||||||\n"); continue
            g = lambda x, nd=3: "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
            f.write(f"| {lab} | {r['run']}:{r['arm']} | {r['cell']} | {r['lib']} | {r['family']} | {r['n']} | {g(r['sr'])} | [{g(r['wilson'][0])},{g(r['wilson'][1])}] | {g(r['v'])} | {g(r['m'])} | {g(r['L'],0)} | {g(r['K'],0)} | {r['stage1_mode']} | {g(r['ir_owner'])} | {g(r['ir_owner_fullS1'])} | {g(r['ir_eager'])} | {g(r['search_ms_per_dec'],2)} | {g(r['ir_owner_plus_search'])} | {g(r['ir_owner_plus_search_k9'])} | {g(r.get('dec_per_ep'),1)} | {g(r.get('miss_per_ep'),2)} |\n")
    with open(f"{S}/pairs.md", "w") as f:
        f.write("| A vs B | n | SR A | SR B | dSR (A-B) | A S/B F | A F/B S | discordant | McNemar p | boot 95% | per-task (A-B net) |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for k, p in pairs.items():
            if "status" in p:
                f.write(f"| {k} | TODO ({','.join(p['missing'])}) ||||||||||\n"); continue
            pt = " ".join(f"t{t}:{v['AwinBlose']-v['AloseBwin']:+d}" for t, v in sorted(p["per_task"].items()))
            f.write(f"| {k} | {p['n']} | {p['srA']:.3f} | {p['srB']:.3f} | {p['delta']:+.3f} | {p['A_S_B_F']} | {p['A_F_B_S']} | {p['discordant']} | {p['p_mcnemar']:.4f} | [{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}] | {pt} |\n")
    # per-task SR table
    with open(f"{S}/per_task.md", "w") as f:
        f.write("| label | " + " | ".join(f"t{t}" for t in range(10)) + " | SR |\n|---|" + "---|"*11 + "\n")
        for lab, r in rows.items():
            if "sr" not in r: continue
            f.write(f"| {lab} | " + " | ".join(f"{r['per_task_sr'][t]:.2f}" for t in range(10)) + f" | {r['sr']:.3f} |\n")
    print("arms with data:", len(J), "TODO:", [l for l in ARMS if l not in J])

if __name__ == "__main__":
    main()
