"""Decision-log statistics for the R4 blind / wrist / pure-cache-blind arms (read-only).

Per arm: src mix, vision share, look_reason mix (on vision decisions), blind_age distribution, judge mix (force reasons),
per-task v and m, policy usage check (src=='policy' rows, miss rows), episodes with zero MISS, MISS-after-MISS,
blind decisions per episode, stage timings (s1_ms medians), decisions per episode success/fail.
"""
from __future__ import annotations
import json, os, sys, glob, collections
from multiprocessing import Pool

S = os.path.dirname(os.path.abspath(__file__))
CL = "/home/weiland/trace_runs/os_closed_loop"

ARMS = [
 ("r04_blind", "r4b3_p_l10_500_ph2g"), ("r04_blind", "r4b3_p_l10_500_b0g"), ("r04_blind", "r4b3_p_l10_500_ph1g"),
 ("r04_blind", "r4b3_p_l10_500_tail1ug"), ("r04_blind", "r4b3_p_l10_500_clk1g"), ("r04_blind", "r4b3_p_l10_500_ph2k8"),
 ("r04_blind", "r4b3_p_l10_500_ph2c"), ("r04_blind", "r4b3_p_l10_50_ph2k5"), ("r04_blind", "r4b3_p_l10_50_ph2c"),
 ("r04_blind", "r4b3_p_sp_500_ph2c"), ("r04_blind", "r4b3_p_sp_500_tail1uc"), ("r04_blind", "r4b3_p_sp_50_ph2g"),
 ("r04_k7", "r4k7_p_l10_500_b0g"), ("r04_k7", "r4k7_p_l10_500_ph2g"), ("r04_k7", "r4k7_p_l10_500_ph1g"), ("r04_k7", "r4k7_p_l10_500_tail1ug"),
 ("r04_k7", "r4k7_p_l10_50_ph2g"), ("r04_k7", "r4k7_p_l10_50_tail1ug"), ("r04_k7", "r4k7_p_sp_500_ph2g"), ("r04_k7", "r4k7_p_sp_500_tail1ug"),
 ("r04_b4w", "r4b4_p_l10_g500_wrist"), ("r04_b4w", "r4b4_p_l10_g50_wrist"), ("r04_b4w", "r4b4_p_sp_g500_wrist"), ("r04_b4w", "r4b4_p_sp_g50_wrist"),
 ("r04_rep", "r4rep_p_l10_g500_a"), ("r04_rep", "r4rep_p_l10_g500_b"), ("r03_mx", "r3mx_p_l10_g500"), ("r03_mx", "r3mx_p_l10_g"),
 ("r04_frontier", "r4_p_sp_g500"), ("r03_mx", "r3mx_p_sp_g"),
 ("r04_csl", "r4b4_p_l10_500_cslG"), ("r04_csl", "r4b4_p_l10_500_cslGS"),
 ("r04_gblind", "r4b3_g_l10_500_ph2"), ("r04_gblind", "r4b3_g_l10_500_tail2u"), ("r04_gblind", "r4b3_g_l10_50_ph2"), ("r04_gblind", "r4b3_g_l10_50_tail2u"),
 ("r04_gblind", "r4b3_g_sp_500_ph2"), ("r04_gblind", "r4b3_g_sp_500_tail2u"), ("r04_gblind", "r4b3_g_sp_50_ph2"), ("r04_gblind", "r4b3_g_sp_50_tail2u"),
]

def accepted_uids(run, arm):
    p = f"{CL}/{run}/runs/{arm}/client/journal.jsonl"
    out = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
            out[r["task_uid"]] = (bool(r["success"]), r.get("attempt"))
    return out

def one(args):
    run, arm = args
    if not os.path.exists(f"{CL}/{run}/state/{arm}.DONE") or os.path.exists(f"{CL}/{run}/state/{arm}.SKIPPED"):
        return arm, {"status": "TODO"}
    acc = accepted_uids(run, arm)
    files = sorted(glob.glob(f"{CL}/{run}/runs/{arm}/server_*/decisions_*.jsonl"))
    src = collections.Counter(); look = collections.Counter(); judge = collections.Counter(); bage = collections.Counter()
    per_task = collections.defaultdict(lambda: {"dec":0, "vis":0, "miss":0, "blind":0})
    per_ep = collections.defaultdict(lambda: {"dec":0, "vis":0, "miss":0, "blind":0, "policy":0, "prev_miss":False, "miss_after_miss":0, "look_after_blind":collections.Counter()})
    look_after_blind = collections.Counter()   # look_reason on vision decisions that follow >=1 blind decision
    s1 = []; s23 = []; qus_v = []; qus_b = []
    miss_k = collections.Counter(); stage_modes = collections.Counter()
    src_by_hit = collections.Counter()
    n_rows = 0; n_ev = collections.Counter()
    for fp in files:
        for line in open(fp):
            try: d = json.loads(line)
            except Exception: continue
            ev = d.get("ev"); n_ev[ev] += 1
            if ev == "startup":
                stage_modes[str(d.get("stage1_mode"))] += 1
                continue
            if ev != "dec": continue
            uid = d.get("uid"); a = acc.get(uid)
            if a is None or (a[1] is not None and d.get("attempt") != a[1]): continue
            n_rows += 1
            t = d.get("task_id"); e = per_ep[uid]; pt = per_task[t]
            vision = d.get("vision", True); hit = d.get("hit", True); s = d.get("src") or ("cache" if hit else "policy")
            src[s] += 1; src_by_hit[(s, bool(hit))] += 1
            e["dec"] += 1; pt["dec"] += 1
            if vision: e["vis"] += 1; pt["vis"] += 1
            else: e["blind"] += 1; pt["blind"] += 1
            if not hit:
                e["miss"] += 1; pt["miss"] += 1
                if e["prev_miss"]: e["miss_after_miss"] += 1
                miss_k[str(d.get("miss_k"))] += 1
            if s == "policy": e["policy"] += 1
            e["prev_miss"] = (not hit)
            lr = d.get("look_reason")
            if vision:
                look[str(lr)] += 1
                if (d.get("blind_age") or 0) > 0: look_after_blind[str(lr)] += 1
            bage[int(d.get("blind_age") or 0)] += 1
            judge[str(d.get("judge"))] += 1
            if d.get("s1_ms") is not None: s1.append(d["s1_ms"])
            if d.get("s23_ms") is not None: s23.append(d["s23_ms"])
            if d.get("q_us") is not None:
                (qus_v if vision else qus_b).append(d["q_us"])
    import numpy as np
    def med(x): return float(np.median(x)) if x else None
    eps = list(per_ep.values())
    succ = {uid: acc[uid][0] for uid in per_ep}
    dec_S = [per_ep[u]["dec"] for u in per_ep if succ[u]]; dec_F = [per_ep[u]["dec"] for u in per_ep if not succ[u]]
    miss_S = [per_ep[u]["miss"] for u in per_ep if succ[u]]; miss_F = [per_ep[u]["miss"] for u in per_ep if not succ[u]]
    blind_S = [per_ep[u]["blind"] for u in per_ep if succ[u]]; blind_F = [per_ep[u]["blind"] for u in per_ep if not succ[u]]
    tot_miss = sum(e["miss"] for e in eps); tot_mam = sum(e["miss_after_miss"] for e in eps)
    out = {
        "status": "ok", "n_dec": n_rows, "episodes": len(per_ep), "journal_accepted": len(acc), "ev_counts": dict(n_ev),
        "stage1_modes_startup": dict(stage_modes),
        "src": dict(src), "src_by_hit": {f"{k[0]}|hit={k[1]}": v for k, v in src_by_hit.items()},
        "v": (sum(e["vis"] for e in eps)/n_rows) if n_rows else None,
        "m": (tot_miss/n_rows) if n_rows else None,
        "look_reason_on_vision": dict(look), "look_reason_after_blind": dict(look_after_blind),
        "blind_age_hist": dict(sorted(bage.items())), "judge_mix": dict(judge), "miss_k": dict(miss_k),
        "policy_rows": src.get("policy", 0), "episodes_with_policy": sum(1 for e in eps if e["policy"] > 0),
        "episodes_with_miss": sum(1 for e in eps if e["miss"] > 0), "episodes_zero_miss": sum(1 for e in eps if e["miss"] == 0),
        "miss_after_miss_share": (tot_mam/tot_miss) if tot_miss else None,
        "dec_per_ep_S": med(dec_S), "dec_per_ep_F": med(dec_F), "mean_dec_S": (sum(dec_S)/len(dec_S)) if dec_S else None, "mean_dec_F": (sum(dec_F)/len(dec_F)) if dec_F else None,
        "miss_per_ep_S": (sum(miss_S)/len(miss_S)) if miss_S else None, "miss_per_ep_F": (sum(miss_F)/len(miss_F)) if miss_F else None,
        "blind_per_ep_S": (sum(blind_S)/len(blind_S)) if blind_S else None, "blind_per_ep_F": (sum(blind_F)/len(blind_F)) if blind_F else None,
        "miss_share_in_F_eps": (sum(miss_F)/tot_miss) if tot_miss else None,
        "s1_ms_p50": med(s1), "s23_ms_p50": med(s23), "q_us_vision_p50": med(qus_v), "q_us_blind_p50": med(qus_b),
        "per_task": {str(t): {"v": p["vis"]/p["dec"], "m": p["miss"]/p["dec"], "dec": p["dec"]} for t, p in sorted(per_task.items())},
    }
    return arm, out

if __name__ == "__main__":
    with Pool(8) as pool:
        res = dict(pool.map(one, ARMS))
    json.dump(res, open(f"{S}/blind_stats.json", "w"), indent=1)
    for arm, r in res.items():
        if r.get("status") != "ok": print(arm, "TODO"); continue
        print(f"{arm:28s} dec={r['n_dec']:6d} v={r['v']:.3f} m={r['m']:.3f} src={r['src']} policy_rows={r['policy_rows']} eps_zero_miss={r['episodes_zero_miss']} look={r['look_reason_on_vision']} judge={r['judge_mix']}")
