"""Comprehensive gate-structure analysis over collected always-search verdicts.

This is the data engine behind the gate exploration roadmap (see
``logs/gate_exploration_roadmap.log.md``): it turns the 7-config gate_research
collection into the decisive numbers for each brainstormed gate scheme —
temporal transition structure (N1 hysteresis / B3 reuse-debt), run lengths and
recovery (probe design), FH-streak hazard (B3's drift premise), winner
persistence (follow-the-winner), the G0b cheap-signal AUC leaderboard
(A1 step / A2 task prior / B4 stillness / last-verdict family), sticky-rule
offline fronts with exact inference-ratio deltas and latency-currency net
values, and the success entanglement safety check.

"step" throughout = one CP1 decision step (one action chunk ≈ 10 env steps).

Usage:
    python gate_structure_analysis.py <gate_rows.jsonl> <suite_name>
"""
import sys, json, collections
import numpy as np

PATH, SUITE = sys.argv[1], sys.argv[2]

# Latency ledger (exp/cache_latency_bench/analysis/latency_breakdown.md):
# gate-skippable cost = search+judge+fetch. Three deployment tiers (ms/step).
SEARCH_MS = {"opt_2.6k": 4.0, "stock_2.6k": 34.0, "opt_50k": 70.0}
INFER_MS = 300.0  # pure-GPU S2+S3 order of magnitude (closed-loop ~1.7 s)


def warm_cost(start_t):
    t = 0.5 if start_t is None else float(start_t)
    return 1.0 - 0.5 * (1.0 - t)


def auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    n1 = labels.sum(); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rows = [json.loads(l) for l in open(PATH)]
ps = [r for r in rows if r.get("_kind") != "episode_summary"]
by_cfg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in ps:
    by_cfg[r["yaml_id"]][r["task_uid"]].append(r)
for cfg in by_cfg.values():
    for uid in cfg:
        cfg[uid].sort(key=lambda r: r["step_idx"])

print(f"===== SUITE {SUITE}: {len(ps)} decision steps =====")

for y_id in sorted(by_cfg):
    eps = by_cfg[y_id]
    short = y_id.split("__d1__")[-1].replace("_quantile", "")
    all_steps = [r for steps in eps.values() for r in steps]
    N = len(all_steps)
    ht = lambda r: r["hit_type"]

    print(f"\n######## {short}  (N={N}, episodes={len(eps)}) ########")

    # ---- 1. composition + EXACT inference ratio (warm cost formula) ----
    comp = collections.Counter(ht(r) for r in all_steps)
    cost = sum({"FULL_HIT": 0.0}.get(ht(r), warm_cost(r.get("start_t")) if ht(r) == "WARM_START" else 1.0)
               for r in all_steps)
    print(f"[1] FH={comp['FULL_HIT']} WS={comp['WARM_START']} MISS={comp['MISS']} "
          f"| exact inf_ratio={cost/N:.3f} | MISS%={100*comp['MISS']/N:.1f}")

    # ---- 2. 3-state transition matrix ----
    trans = collections.defaultdict(collections.Counter)
    for steps in eps.values():
        for a, b in zip(steps, steps[1:]):
            trans[ht(a)][ht(b)] += 1
    print("[2] transition P(next|cur):")
    for a in ("FULL_HIT", "WARM_START", "MISS"):
        tot = sum(trans[a].values())
        if tot:
            print("    %-10s -> " % a[:9] + "  ".join(
                f"{b[:4]}={trans[a][b]/tot:.2f}" for b in ("FULL_HIT", "WARM_START", "MISS")))

    # ---- 3. runs, recovery, episode taxonomy, changepoint position ----
    miss_runs, hit_runs = [], []
    recovered = censored = 0
    post_recovery_hits = []
    taxo = collections.Counter()
    cp_pos = []
    for uid, steps in eps.items():
        seq = [1 if ht(r) == "MISS" else 0 for r in steps]
        # runs
        runs = []
        cur, ln = seq[0], 1
        for v in seq[1:]:
            if v == cur:
                ln += 1
            else:
                runs.append((cur, ln)); cur, ln = v, 1
        runs.append((cur, ln))
        miss_blocks = [ln for v, ln in runs if v == 1]
        hit_blocks = [ln for v, ln in runs if v == 0]
        miss_runs += miss_blocks; hit_runs += hit_blocks
        # recovery: a MISS block followed by a hit block (vs ends episode)
        for i, (v, ln) in enumerate(runs):
            if v == 1:
                if i + 1 < len(runs):
                    recovered += 1
                    post_recovery_hits.append(runs[i + 1][1])
                else:
                    censored += 1
        # taxonomy
        mfrac = sum(seq) / len(seq)
        nblocks = len(miss_blocks)
        if mfrac == 0:
            taxo["all_hit"] += 1
        elif mfrac >= 0.95:
            taxo["all_miss"] += 1
        elif nblocks == 1 and runs[-1][0] == 1:
            taxo["one_tail_block"] += 1
        elif nblocks <= 2:
            taxo["few_blocks(<=2)"] += 1
        else:
            taxo["oscillating(>2)"] += 1
        # changepoint = start of first MISS run with len>=3
        pos = 0; found = None
        for v, ln in runs:
            if v == 1 and ln >= 3:
                found = pos; break
            pos += ln
        if found is not None:
            cp_pos.append(found / len(seq))
    mr = np.array(miss_runs); hr = np.array(hit_runs)
    print(f"[3] MISS-runs: n={len(mr)} mean={mr.mean():.1f} med={np.median(mr):.0f} p90={np.percentile(mr,90):.0f} "
          f"| HIT-runs: mean={hr.mean():.1f} med={np.median(hr):.0f} p90={np.percentile(hr,90):.0f}")
    print(f"    recovery: {recovered}/{recovered+censored} MISS-blocks recover to hits "
          f"({100*recovered/max(recovered+censored,1):.0f}%), post-recovery hit-block mean="
          f"{np.mean(post_recovery_hits) if post_recovery_hits else 0:.1f}")
    tt = sum(taxo.values())
    print("    taxonomy: " + "  ".join(f"{k}={100*v/tt:.0f}%" for k, v in taxo.most_common()))
    if cp_pos:
        cp = np.array(cp_pos)
        print(f"    changepoint (first MISS-run>=3) phase: med={np.median(cp):.2f} "
              f"IQR=[{np.percentile(cp,25):.2f},{np.percentile(cp,75):.2f}] (fraction of episode)")

    # ---- 4. B3 hazard: P(next=MISS | current notMISS-streak length L) ----
    hz_n = collections.Counter(); hz_m = collections.Counter()
    for steps in eps.items():
        pass
    for steps in eps.values():
        streak = 0
        for a, b in zip(steps, steps[1:]):
            if ht(a) != "MISS":
                streak += 1
                L = min(streak, 30)
                hz_n[L] += 1
                hz_m[L] += 1 if ht(b) == "MISS" else 0
            else:
                streak = 0
    buckets = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 31)]
    hzs = []
    for lo, hi in buckets:
        n = sum(hz_n[L] for L in range(lo, hi + 1)); m = sum(hz_m[L] for L in range(lo, hi + 1))
        if n > 50:
            hzs.append(f"L{lo}-{hi}:{m/n:.3f}(n={n})")
    print("[4] B3 hazard P(MISS|hit-streak L): " + "  ".join(hzs))

    # ---- 5. winner persistence (follow-the-winner) ----
    same = diff = 0
    dstep = collections.Counter()
    for steps in eps.values():
        for a, b in zip(steps, steps[1:]):
            if ht(a) == "FULL_HIT" and ht(b) == "FULL_HIT" and a.get("winner_id") and b.get("winner_id"):
                wa, sa = a["winner_id"].rsplit(":", 1)
                wb, sb = b["winner_id"].rsplit(":", 1)
                if wa == wb:
                    same += 1
                    d = int(sb) - int(sa)
                    dstep[min(max(d, -1), 3)] += 1
                else:
                    diff += 1
    if same + diff:
        tot = same + diff
        print(f"[5] winner persistence (FH->FH): same-episode={100*same/tot:.0f}% (n={tot})")
        if same:
            print("    Dwinner_step | " + "  ".join(f"{k}:{100*v/same:.0f}%" for k, v in sorted(dstep.items())))

    # ---- 6. G0b cheap-signal AUC leaderboard (target: current step is MISS) ----
    # build aligned arrays over steps with a previous step (t>=1)
    sig = collections.defaultdict(list); lab = []
    task_missrate = collections.defaultdict(lambda: [0, 0])
    for steps in eps.values():
        for r in steps:
            task_missrate[r["task_id"]][0] += 1 if ht(r) == "MISS" else 0
            task_missrate[r["task_id"]][1] += 1
    for uid, steps in eps.items():
        scores = [r.get("cp1_score") for r in steps]
        first3 = np.mean([s for s in scores[:3] if s is not None] or [np.nan])
        streak = 0
        for i in range(1, len(steps)):
            r = steps[i]; p = steps[i - 1]
            lab.append(1 if ht(r) == "MISS" else 0)
            sig["prev_is_MISS"].append(1.0 if ht(p) == "MISS" else 0.0)
            sig["prev_score(neg)"].append(-(p.get("cp1_score") or 0.0))
            last3 = [steps[j].get("cp1_score") or 0.0 for j in range(max(0, i - 3), i)]
            sig["last3_mean_score(neg)"].append(-float(np.mean(last3)))
            streak = streak + 1 if ht(p) != "MISS" else 0
            sig["hit_streak_len(B3)"].append(float(streak))
            sig["step_idx(A1)"].append(float(r["step_idx"]))
            sig["task_prior(A2)"].append(task_missrate[r["task_id"]][0] / task_missrate[r["task_id"]][1])
            sig["first3_score(neg,N3)"].append(-first3 if not np.isnan(first3) else 0.0)
            ds = float(np.linalg.norm(
                np.array(r["collect"]["robot_state"]) - np.array(p["collect"]["robot_state"])))
            sig["neg_motion(B4)"].append(-ds)
    lab = np.array(lab)
    print(f"[6] signal AUC -> MISS (n={len(lab)}, base={lab.mean():.2f}):")
    for k in ("prev_is_MISS", "prev_score(neg)", "last3_mean_score(neg)", "hit_streak_len(B3)",
              "step_idx(A1)", "task_prior(A2)", "first3_score(neg,N3)", "neg_motion(B4)"):
        a = auc(sig[k], lab)
        print(f"    {k:26s} AUC={a:.3f}")
    tmr = sorted(v[0] / v[1] for v in task_missrate.values())
    print(f"    per-task MISS% range: {100*tmr[0]:.0f}%..{100*tmr[-1]:.0f}% (A2 spread)")

    # ---- 7. sticky stop-after-K (+ probe every M) offline front ----
    # trajectory-invariance caveat: skipping true-MISS steps does not change
    # actions; wrongly skipped hits DO (offline numbers are approximations there).
    print("[7] sticky fronts  (skip%=searches skipped; dInf=exact inf_ratio increase; "
          "net ms/step @search:4/34/70ms vs infer 300ms)")
    for K in (2, 3, 4):
        for M in (None, 10, 5):
            skip = 0; d_inf = 0.0; lost_hits_fail_ep = 0; lost_hits = 0
            for uid, steps in eps.items():
                run = 0; stopped = False; since_probe = 0
                ep_fail = not steps[0].get("success", False)
                for r in steps:
                    if stopped:
                        since_probe += 1
                        if M is not None and since_probe >= M:
                            since_probe = 0  # probe step: search happens
                            if ht(r) != "MISS":
                                stopped = False; run = 0
                            continue
                        skip += 1
                        if ht(r) == "FULL_HIT":
                            d_inf += 1.0; lost_hits += 1; lost_hits_fail_ep += ep_fail
                        elif ht(r) == "WARM_START":
                            d_inf += 1.0 - warm_cost(r.get("start_t")); lost_hits += 1
                            lost_hits_fail_ep += ep_fail
                        continue
                    if ht(r) == "MISS":
                        run += 1
                        if run >= K:
                            stopped = True; since_probe = 0
                    else:
                        run = 0
            nets = " ".join(f"{tag}:{(skip/N)*ms - (d_inf/N)*INFER_MS:+.1f}"
                            for tag, ms in SEARCH_MS.items())
            lf = 100 * lost_hits_fail_ep / max(lost_hits, 1)
            print(f"    K={K} probe={str(M):>4s}: skip={100*skip/N:5.1f}% dInf=+{d_inf/N:.3f} "
                  f"lost-hits-in-fail-ep={lf:3.0f}% | net {nets}")

    # ---- 8. success entanglement ----
    ep_miss_s = [np.mean([1 if ht(r) == "MISS" else 0 for r in st])
                 for st in eps.values() if st[0].get("success")]
    ep_miss_f = [np.mean([1 if ht(r) == "MISS" else 0 for r in st])
                 for st in eps.values() if not st[0].get("success")]
    print(f"[8] episode MISS%: success-ep mean={100*np.mean(ep_miss_s):.0f}% (n={len(ep_miss_s)}) "
          f"vs fail-ep mean={100*np.mean(ep_miss_f):.0f}% (n={len(ep_miss_f)})")
