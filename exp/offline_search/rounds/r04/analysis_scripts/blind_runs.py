"""Blind-run structure per arm (read-only): blind rows by blind_age, anchors followed by a tail/blind, run lengths,
gate-cut runs, per-episode blind share; plus exact dummy_cached repricing of the owner IR."""
from __future__ import annotations
import json, os, glob, collections
from multiprocessing import Pool
S = os.path.dirname(os.path.abspath(__file__))
CL = "/home/weiland/trace_runs/os_closed_loop"
ARMS = [("r04_k7","r4k7_p_l10_500_ph2g"),("r04_k7","r4k7_p_l10_500_ph1g"),("r04_k7","r4k7_p_l10_500_tail1ug"),("r04_k7","r4k7_p_l10_50_ph2g"),("r04_k7","r4k7_p_l10_50_tail1ug"),("r04_k7","r4k7_p_sp_500_ph2g"),("r04_k7","r4k7_p_sp_500_tail1ug"),
        ("r04_blind","r4b3_p_l10_500_ph2g"),("r04_blind","r4b3_p_l10_500_tail1ug"),("r04_blind","r4b3_p_l10_500_clk1g"),("r04_blind","r4b3_p_l10_500_ph1g"),("r04_blind","r4b3_p_l10_500_ph2k8"),("r04_blind","r4b3_p_l10_50_ph2k5"),
        ("r04_blind","r4b3_p_l10_500_ph2c"),("r04_blind","r4b3_p_l10_50_ph2c"),("r04_blind","r4b3_p_sp_500_ph2c"),("r04_blind","r4b3_p_sp_500_tail1uc"),("r04_blind","r4b3_p_sp_50_ph2g"),
        ("r04_gblind","r4b3_g_l10_500_ph2"),("r04_gblind","r4b3_g_l10_500_tail2u"),("r04_gblind","r4b3_g_l10_50_ph2"),("r04_gblind","r4b3_g_l10_50_tail2u"),("r04_gblind","r4b3_g_sp_500_ph2"),("r04_gblind","r4b3_g_sp_500_tail2u"),("r04_gblind","r4b3_g_sp_50_ph2"),("r04_gblind","r4b3_g_sp_50_tail2u")]

def acc(run, arm):
    out = {}
    for line in open(f"{CL}/{run}/runs/{arm}/client/journal.jsonl"):
        r = json.loads(line)
        if r.get("accepted") and r.get("status") in ("done","failed") and not r.get("error"):
            out[r["task_uid"]] = (bool(r["success"]), r.get("attempt"))
    return out

def one(args):
    run, arm = args
    A = acc(run, arm)
    rows = collections.defaultdict(list)
    for fp in sorted(glob.glob(f"{CL}/{run}/runs/{arm}/server_*/decisions_*.jsonl")):
        for line in open(fp):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("ev") != "dec": continue
            a = A.get(d.get("uid"))
            if a is None or d.get("attempt") != a[1]: continue
            rows[d["uid"]].append((d.get("step"), bool(d.get("vision", True)), bool(d.get("hit", True)), d.get("look_reason"), d.get("src")))
    blind_age_blind = collections.Counter(); run_len = collections.Counter(); anchors = 0; anchors_with_blind = 0
    anchors_hit = 0; anchors_hit_with_blind = 0; cut_reason = collections.Counter(); eps_blind_share = []
    vision_after_miss = 0; miss_then_blind = 0
    for uid, rs in rows.items():
        rs.sort(key=lambda x: x[0])
        n = len(rs); nb = sum(1 for r in rs if not r[1]); eps_blind_share.append(nb / n)
        i = 0
        while i < n:
            step, vis, hit, lr, src = rs[i]
            if vis:
                anchors += 1
                if hit: anchors_hit += 1
                j = i + 1; L = 0
                while j < n and not rs[j][1]:
                    L += 1; j += 1
                if L > 0:
                    anchors_with_blind += 1
                    if hit: anchors_hit_with_blind += 1
                    run_len[L] += 1
                    if j < n: cut_reason[str(rs[j][3])] += 1
                    else: cut_reason["episode_end"] += 1
                i = j
            else:
                i += 1
    return arm, {"anchors": anchors, "anchors_hit": anchors_hit, "anchors_with_blind": anchors_with_blind, "anchors_hit_with_blind": anchors_hit_with_blind,
                 "share_hit_anchors_followed_by_blind": anchors_hit_with_blind / anchors_hit if anchors_hit else None,
                 "run_len_hist": dict(sorted(run_len.items())), "look_reason_ending_run": dict(cut_reason),
                 "mean_run_len": (sum(k*v for k, v in run_len.items()) / sum(run_len.values())) if run_len else None,
                 "episode_blind_share_p10_p50_p90": [float(x) for x in __import__("numpy").percentile(eps_blind_share, [10, 50, 90])]}

if __name__ == "__main__":
    with Pool(8) as p:
        res = dict(p.map(one, ARMS))
    json.dump(res, open(f"{S}/blind_runs.json", "w"), indent=1)
    for k, v in res.items():
        print(k, json.dumps(v))
    # dummy_cached exact repricing (owner basis, ratio-transfer assumption for s1 = .1039): IR_dc = .1039 v + .848 m
    D = json.load(open(f"{S}/an.json"))["arms"]
    print("\nlabel | v | m | owner IR (full s1) | owner IR dummy_cached s1=.1039")
    for l, r in D.items():
        if "sr" not in r or not r["cell"].startswith("p_") or r["stage1_mode"] != "full": continue
        k = r["K"] or 10.0
        ir_dc = (0.10390199743108731 * r["v"] + r["m"] * (0.41 + 0.438 * k / 10.0)) * 5.0 / r["L"]
        print(f"{l} | {r['v']:.3f} | {r['m']:.3f} | {r['ir_owner']:.3f} | {ir_dc:.3f}")
