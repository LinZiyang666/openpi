"""Round-6 ledgers: the round-5 set plus the round-5 screen and every takeover (escalation) run on LIBERO-10 / 50.

Uses ``round5.tools.ledger5.extract`` unchanged (RULE 1: ``iter_jsonl_discovery`` drops init >= 30 before decoding;
holdout roots refused; output asserted init < 30).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

from exp.offline_search.rounds.r09.explore_opus.round5.tools.ledger5 import ARMS as ARMS5, extract

EXTRA = [
    # round-5 screen (inits 20-29): stack same-batch control and the two gated arms
    ("r09_opus_r5", "r9o5_pi05_l10_50_stack", "pi05"), ("r09_opus_r5", "r9o5_pi05_l10_50_stack_P2", "pi05"),
    ("r09_opus_r5", "r9o5_pi05_l10_50_stack_P2C20", "pi05"), ("r09_opus_r5", "r9o5_groot_l10_50_stack", "groot"),
    ("r09_opus_r5", "r9o5_groot_l10_50_stack_P2", "groot"), ("r09_opus_r5", "r9o5_groot_l10_50_stack_P2C20", "groot"),
    # takeover runs (inits 20-29): pure cache + persistent / windowed escalation
    ("r09_opus_escalation", "r9o_pi05_l10_50_esc", "pi05"), ("r09_opus_escalation", "r9o_groot_l10_50_esc", "groot"),
    ("r09_opus_escalation", "r9o_pi05_l10_50_esc_w24", "pi05"), ("r09_opus_escalation", "r9o_groot_l10_50_esc_w24", "groot"),
    ("r09_opus_r3", "r9o3_pi05_l10_50_esc_w24", "pi05"), ("r09_opus_r3", "r9o3_groot_l10_50_esc_w24", "groot"),
]
ARMS = ARMS5 + EXTRA


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(extract, r, arm, m) for r, arm, m in ARMS]
        for (r, arm, m), f in zip(ARMS, futs):
            print(r, arm, f.result())


if __name__ == "__main__":
    main()
