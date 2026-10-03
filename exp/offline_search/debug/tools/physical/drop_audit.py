"""T4 CLI: recurrent premature opening versus closed-command carry loss."""

from collections import Counter
from .audits import drops
from .common import load_config, load_episodes, parser, parallel_map, report, safe_rows


def main(argv=None):
    args = parser("drop_audit").parse_args(argv)
    episodes = load_episodes(args.run_root, args.arms, args.p3v2, args.limit)
    config = load_config(args, episodes)
    rows = [
        r
        for group in parallel_map(
            lambda ep: safe_rows(drops, ep, config), episodes, args.procs
        )
        for r in group
    ]
    report(
        args.out,
        "drop_audit",
        {"carry_losses": rows},
        {
            "accepted_episodes": len(episodes),
            "mechanisms": dict(
                Counter(r.get("mechanism", "unavailable") for r in rows)
            ),
        },
        episodes=episodes,
    )


if __name__ == "__main__":
    main()
