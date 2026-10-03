"""T8 CLI: arm-level static HTML/PNG, failure modes and stage/source ledger."""

from collections import defaultdict
import csv
from pathlib import Path
from .cards import rollup
from .common import load_config, load_episodes, parser, report, cluster_interval


def main(argv=None):
    ap = parser("arm_rollup")
    ap.add_argument(
        "--pairs-file",
        type=Path,
        help="optional paired_diverge pairs.csv for transition tables",
    )
    args = ap.parse_args(argv)
    groups = defaultdict(list)
    all_episodes = load_episodes(args.run_root, args.arms, args.p3v2, args.limit)
    config = load_config(args, all_episodes)
    for ep in all_episodes:
        groups[ep.meta["arm"]].append(ep)
    rows, ledger, summary = [], [], []
    for episodes in groups.values():
        r, l, s = rollup(episodes, args.out, config)
        s["success_interval"] = cluster_interval(r, "success")
        rows.extend(r)
        ledger.extend(l)
        summary.append(s)
    pairs = []
    if args.pairs_file:
        with args.pairs_file.open() as handle:
            pairs = list(csv.DictReader(handle))
    report(
        args.out,
        "arm_rollup",
        {"episodes": rows, "stage_source_ledger": ledger, "paired_transitions": pairs},
        {"arms": summary},
        episodes=all_episodes,
    )


if __name__ == "__main__":
    main()
