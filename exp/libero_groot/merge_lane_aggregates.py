"""Merge one suite's per-arm aggregates from the lanes that ran it.

After the handover both worker boxes ran libero_10, each on a disjoint set of
arms, so the suite's results live in two places and have to be joined before
anything is read off them.

The join is not a plain dict update, because one family is *not* disjoint. The
two sweep groups are: every hg arm was run by exactly one lane, so taking
whichever lane has it is unambiguous. The anchors are different -- one lane
carries a stale partial run of them from an earlier, interrupted attempt, and
the other carries the complete 500 episodes. Merging those by summing would
double-count the overlapping episodes' per-step verdicts and move the
inference ratio; merging by "whichever came last" would depend on file order.
So an arm is taken, whole, from the lane with the most episodes for it, and the
choice is recorded per arm.

Usage:
  uv run python -m exp.libero_groot.merge_lane_aggregates \
      --input laneA=<aggA.json> --input laneB=<aggB.json> \
      --out <merged.json> [--expect-episodes 500]
"""

from __future__ import annotations

import argparse
import json
import pathlib


def merge(sources: dict[str, dict]) -> tuple[dict, dict[str, str]]:
    """Per arm, the record from the lane that ran the most of it."""
    merged: dict[str, dict] = {}
    chosen: dict[str, str] = {}
    for lane, agg in sources.items():
        for arm, rec in agg.items():
            best = merged.get(arm)
            if best is None or rec.get("n_ep", 0) > best.get("n_ep", 0):
                merged[arm] = rec
                chosen[arm] = lane
    return merged, chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", action="append", required=True,
                    help="lane=path, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-episodes", type=int, default=500)
    args = ap.parse_args()

    sources = {}
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"--input must be lane=path, got {spec!r}")
        lane, path = spec.split("=", 1)
        sources[lane] = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

    merged, chosen = merge(sources)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    short = {a: r["n_ep"] for a, r in merged.items() if r["n_ep"] < args.expect_episodes}
    per_lane: dict[str, int] = {}
    for lane in chosen.values():
        per_lane[lane] = per_lane.get(lane, 0) + 1
    print(f"wrote {out}  arms={len(merged)}  from {per_lane}")
    # A contested arm is one both lanes have episodes for: expected for the
    # anchors, and a bug anywhere else, so it is named rather than counted.
    contested = [a for a in merged
                 if sum(1 for s in sources.values() if a in s and s[a]["n_ep"]) > 1]
    if contested:
        print(f"  arms present on more than one lane (taken from the fuller one): "
              f"{sorted(contested)}")
    print(f"  arms short of {args.expect_episodes} episodes: {short or 'none'}")


if __name__ == "__main__":
    main()
