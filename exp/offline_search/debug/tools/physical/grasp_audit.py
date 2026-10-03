"""T3 CLI: object-frame pre-grasp poses, lift windows and policy references."""

from .audits import grasps, grasp_references
from .common import (
    load_config,
    load_episodes,
    parser,
    parallel_map,
    report,
    safe_rows,
    cluster_interval,
)


def main(argv=None):
    ap = parser("grasp_audit")
    ap.add_argument(
        "--reference-arms",
        nargs="+",
        default=[],
        help="successful pure-policy reference arms in --arms",
    )
    args = ap.parse_args(argv)
    episodes = load_episodes(args.run_root, args.arms, args.p3v2, args.limit)
    config = load_config(args, episodes)
    rows = [
        r
        for group in parallel_map(
            lambda ep: safe_rows(grasps, ep, config), episodes, args.procs
        )
        for r in group
    ]
    atlas, summary = grasp_references(rows, args.reference_arms)
    summary["accepted_episodes"] = len(episodes)
    summary["lift_rate_interval"] = cluster_interval(rows, "lift_within_window")
    report(
        args.out,
        "grasp_audit",
        {"grasp_attempts": rows, "grasp_pose_clusters": atlas},
        summary,
        episodes=episodes,
    )


if __name__ == "__main__":
    main()
