"""Screen readout for r09_opus_r5 (run after the chain finishes; inits 20-29 only).

Per arm: success, owner IR, calls per episode by reason, gate counts (``extras.r9o5_gate``: 1 = on-pace silence,
2 = budget), no-progress calls left at lag <= 2 (should be 0 in gated arms); paired discordance against the same-batch
``stack`` control.  RULE 1: journals and server logs via ``iter_jsonl_discovery`` (init >= 30 dropped before decoding).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import RUNS, PRICE, iter_jsonl_discovery, parse_uid, check_root
from exp.offline_search.rounds.r09.explore_opus.round3.tools.dissect import journal

RUN = RUNS / "r09_opus_r5"


def arm_stats(arm, model):
    J = journal(RUN, arm)
    v, m = PRICE[model]
    dec = looks = 0
    calls, gates, np_on_pace = {}, {1.0: 0, 2.0: 0}, 0
    for path in sorted((check_root(RUN) / "runs" / arm).glob("server_*/decisions*.jsonl")):
        for r in iter_jsonl_discovery(path, uid_key="uid"):
            if r.get("ev") != "dec":
                continue
            t, i = parse_uid(r["uid"])
            if (t, i) not in J or int(r.get("attempt", 1)) != J[(t, i)][1]:
                continue
            ex = r.get("extras") or {}
            dec += 1
            looks += bool(r.get("vision"))
            if r.get("vision") and r.get("hit") is False:
                reason = float(ex.get("os_reason", -1))
                calls[reason] = calls.get(reason, 0) + 1
                if reason == 4.0 and ex.get("r9o5_lag", 99) <= 2:
                    np_on_pace += 1
            g = float(ex.get("r9o5_gate", 0) or 0)
            if g in gates:
                gates[g] += 1
    n = len(J)
    ncalls = sum(calls.values())
    return dict(episodes=n, sr=float(np.mean([s for s, _ in J.values()])) if n else None,
                ir=(v * looks + m * ncalls) / dec if dec else None, calls_per_ep={str(k): c / n for k, c in calls.items()} if n else {},
                gated_pace_per_ep=gates[1.0] / n if n else None, gated_budget_per_ep=gates[2.0] / n if n else None,
                np_calls_on_pace=np_on_pace), J


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    out = {}
    for model in ("pi05", "groot"):
        base = f"r9o5_{model}_l10_50_stack"
        stats = {}
        for tag in ("", "_P2", "_P2C20"):
            arm = base + tag
            if not (RUN / "runs" / arm).exists():
                continue
            s, J = arm_stats(arm, model)
            stats[arm] = (s, J)
        if base not in stats:
            continue
        _, J0 = stats[base]
        for arm, (s, J) in stats.items():
            if arm != base:
                both = set(J) & set(J0)
                s["vs_stack"] = dict(pairs=len(both), gated_only=sum(J[k][0] and not J0[k][0] for k in both),
                                     stack_only=sum(J0[k][0] and not J[k][0] for k in both))
            out[arm] = s
            print(arm, json.dumps(s))
    if out:
        (RUN / "readout_r9o5.json").write_text(json.dumps(out, indent=1))
    else:
        print("no r9o5 runs yet")


if __name__ == "__main__":
    main()
