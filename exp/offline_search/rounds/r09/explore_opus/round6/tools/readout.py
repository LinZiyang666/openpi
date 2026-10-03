"""Screen readout for r09_opus_r6 (run after the chain finishes; inits 20-29 only).

Per arm: success, owner IR, calls per episode by reason, takeover counts (``extras.r9o6_*``: triggers, decisions
inside a takeover window), reason-95 calls; paired discordance against the same-batch ``stack`` control.  RULE 1: journals and server logs via ``iter_jsonl_discovery`` (init >= 30 dropped before decoding).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import RUNS, PRICE, iter_jsonl_discovery, parse_uid, check_root
from exp.offline_search.rounds.r09.explore_opus.round3.tools.dissect import journal

RUN = RUNS / "r09_opus_r6"


def arm_stats(arm, model):
    J = journal(RUN, arm)
    v, m = PRICE[model]
    dec = looks = 0
    calls, triggers, in_window = {}, 0, 0
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
            triggers += int(ex.get("r9o6_trigger", 0) == 1.0)
            in_window += int(ex.get("r9o6_takeover", 0) == 1.0)
    n = len(J)
    ncalls = sum(calls.values())
    return dict(episodes=n, sr=float(np.mean([s for s, _ in J.values()])) if n else None,
                ir=(v * looks + m * ncalls) / dec if dec else None, calls_per_ep={str(k): c / n for k, c in calls.items()} if n else {},
                takeover_triggers=triggers, takeover_window_decisions=in_window), J


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    out = {}
    for model in ("pi05", "groot"):
        base = f"r9o6_{model}_l10_50_stack"
        stats = {}
        for arm in (base, base + "_X", f"r9o6_{model}_l10_50_npcorr_X"):
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
                s["vs_stack"] = dict(pairs=len(both), arm_only=sum(J[k][0] and not J0[k][0] for k in both),
                                     stack_only=sum(J0[k][0] and not J[k][0] for k in both))
            out[arm] = s
            print(arm, json.dumps(s))
    if out:
        (RUN / "readout_r9o6.json").write_text(json.dumps(out, indent=1))
    else:
        print("no r9o6 runs yet")


if __name__ == "__main__":
    main()
