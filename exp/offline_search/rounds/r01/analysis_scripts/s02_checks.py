"""Artefact checks: duplicate variants (per-decision identity), regime identity, library/query overlap (leakage),
timing provenance, floor context."""
import sys, os, json, hashlib
sys.path.insert(0, "/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r01")
import numpy as np
from h import *

print("## 1. per-decision identity of variants (max |err_A - err_B|, fraction identical top1)")
pairs = [
    ("inf", "M1_big_a0p25_med3_stWin_b0", "M7_cascade_none_none_med3"),
    ("inf", "M1_big_a0p25_med3_stWin_b0", "M1_big_a0p25_med3_stCont_b0"),
    ("cache", "M1_big_a0p25_med3_stWin_b0", "M1_big_a0_med3_stWin_b0"),
    ("cache", "M1_big_a0p25_med3_stWin_b0", "M1_big_a1_med3_stWin_b0"),
    ("cache", "M3hmm_big_a60b15g10_em1_cont_mass", "M3hmm_big_a60b15g10_em1_cont_lev"),
    ("cache", "M3hmm_big_a60b15g10_em1_cont_mass", "M3hmm_big_a60b15g10_em1_nocont_mass"),
    ("inf", "M3hmm_big_a60b15g10_em1_cont_mass", "M3hmm_big_a60b15g10_em1_nocont_mass"),
    ("cache", "M4_b0cons_k3_med", "M5_b0sl_cont_linf_k3"),
    ("cache", "M4_b0cons_k5_med", "M5_b0sl_cont_l2_k5"),
    ("inf", "M6_ccf_b0_eq", "B0_current"),
    ("cache", "M6_ccf_b0_eq", "M6_ccf_b0_split"),
    ("cache", "M7_cascade_none_none_med3", "M1_big_a0p25_med3_stWin_b0"),
    ("cache", "Rtail_passthrough", "M1_big_a0p25_med3_stWin_b0"),
    ("inf", "M9c_vzsum_big_pca128_tm_both_st1_top1", "M9c_vzsum_big_pca128_tm_both_top1"),
]
for arm, a, b in pairs:
    row = []
    for c in [x for x in CELLS if x.endswith(arm)]:
        A, B = load(a, c), load(b, c)
        if A is None or B is None:
            row.append("-"); continue
        d = np.abs(A["err"] - B["err"]).max()
        same = float(np.mean(A["top1"] == B["top1"]))
        row.append(f"{d:.1e}/{same:.2f}")
    print(f"  [{arm}] {a} vs {b}: " + "  ".join(row))

print("\n## 2. regime flag distribution (x_regime) per cell")
for m in ("M1_big_a0p25_kmean5_stWin_b0", "M8_b0big_k5_mean", "M2sw_big_m3_k8_mean", "M3hmm_big_a60b15g10_em1_cont_lev"):
    print(" ", m)
    for c in CELLS:
        d = load(m, c)
        r = d["x_regime"]
        u, n = np.unique(r, return_counts=True)
        print(f"    {c:20s} n={len(r):6d} regime counts {dict(zip(u.tolist(), n.tolist()))}  step0={int((d['step']==0).sum())}")

print("\n## 3. leakage: query a_inf[:5,:7] bit-identical to a library chunk of the same task (fraction of query rows)")
for ms in MS:
    m, s = ms.split("_")
    libs = ["current", "bpool_cs" if m == "pi05" else "bpool_all"]
    for lib in libs:
        L = f"{ROOT}/library/{ms}/{lib}"
        act = np.load(f"{L}/action.npy", mmap_mode="r")
        tid = np.load(f"{L}/task_id.npy")
        hs = {}
        a7 = np.ascontiguousarray(np.asarray(act[:, :5, :7], np.float32))
        for i in range(len(a7)):
            hs.setdefault(int(tid[i]), set()).add(a7[i].tobytes())
        for arm in ("inf", "cache"):
            c = f"{ms}_{arm}"
            Q = f"{ROOT}/queries/{c}"
            ai = np.load(f"{Q}/a_inf.npy", mmap_mode="r")
            ae = np.load(f"{Q}/a_exec.npy", mmap_mode="r")
            eps = episodes(c)
            ep = np.load(f"{Q}/ep.npy")
            qt = np.array([e["task_id"] for e in eps])[ep]
            qi = np.ascontiguousarray(np.asarray(ai[:, :5, :7], np.float32))
            qe = np.ascontiguousarray(np.asarray(ae[:, :5, :7], np.float32))
            hit_inf = np.mean([qi[i].tobytes() in hs.get(int(qt[i]), ()) for i in range(len(qi))])
            hit_exec = np.mean([qe[i].tobytes() in hs.get(int(qt[i]), ()) for i in range(len(qe))])
            print(f"  {c:20s} lib={lib:9s} a_inf-in-lib {hit_inf:.4f}   a_exec-in-lib {hit_exec:.4f}")

print("\n## 4. query inits vs library inits (A pool vs B pool) and step-0 state distance")
for ms in MS:
    m, s = ms.split("_")
    lib = "bpool_cs" if m == "pi05" else "bpool_all"
    le = json.load(open(f"{ROOT}/library/{ms}/{lib}/episodes.json"))
    qe = episodes(f"{ms}_inf")
    linit = {(e["task_id"], e.get("orig_init_state_idx")) for e in le}
    qinit = {(e["task_id"], e["init"]) for e in qe}
    print(f"  {ms}: lib inits known={sum(1 for e in le if e.get('orig_init_state_idx') is not None)}/{len(le)}  "
          f"query (task,init) pairs {len(qinit)}, lib pairs {len(linit)} (pools differ by construction: A-pool queries vs B-pool library)")

print("\n## 5. timing provenance")
import glob
tc = {}
for p in glob.glob(f"{RES}/r01/*/*.json"):
    if os.path.basename(p) in ("summary.json", "run_meta.json"): continue
    j = json.load(open(p))
    t = j.get("timing", {})
    tc.setdefault((t.get("timing_source"), t.get("timing_concurrency")), 0)
    tc[(t.get("timing_source"), t.get("timing_concurrency"))] += 1
print("  r01 (source, concurrency) counts:", tc)
tc = {}
for p in glob.glob(f"{RES}/r00/*/*.json"):
    if os.path.basename(p) in ("summary.json", "run_meta.json"): continue
    j = json.load(open(p))
    t = j.get("timing", {})
    tc.setdefault((t.get("timing_source"), t.get("timing_concurrency")), 0)
    tc[(t.get("timing_source"), t.get("timing_concurrency"))] += 1
print("  r00 (source, concurrency) counts:", tc, " -> r00 ms/query were measured in-run (not at concurrency 4)")

print("\n## 6. worker-side mean t_query_us (all decisions; workers ran concurrently -> relative only)")
for m in ("B0_current", "M1_big_a0p25_kmean5_stWin_b0", "M2sw_big_m3_k8_mean", "M7_cascade_pca32_both_med3", "M7_cascade_raw_both_med3",
          "M8_b0big_k5_mean", "M9c_vzsum_big_pca128_tm_both_st1_med3", "M3hmm_big_a60b15g10_em1_cont_lev", "M4_b0cons_k5_kernel"):
    row = []
    for c in CELLS:
        d = load(m, c)
        row.append(f"{np.mean(d['t_query_us'])/1000:.2f}")
    print(f"  {m:40s} ms: " + " ".join(row))

print("\n## 7. floor context: err_over_floor_median / indist (scoreboard) for the top methods")
import pandas as pd
df = pd.read_csv(f"{OUT}/master.csv")
for m in ("B0_current", "M1_big_a0p25_kmean5_stWin_b0", "M8_b0big_k5_mean", "M2sw_big_m3_k8_mean", "Rtail_passthrough", "B3_oracle"):
    sub = df[df.method == m].set_index("cell").loc[CELLS]
    print(f"  {m:32s} err/floor p50: " + " ".join(f"{v:.2f}" for v in sub.err_over_floor_median) +
          " | indist: " + " ".join(f"{v:.2f}" for v in sub.indist) + " | bad@100: " + " ".join(f"{v:.2f}" for v in sub.bad_c100))
