"""T5 CLI: same-seed pairs with physical and event-aligned divergence."""

import json
from pathlib import Path
from .common import load_config, load_episodes, parser, report
from .pairs import build, library_scales


def main(argv=None, twin=False):
    name = "twin_divergence" if twin else "paired_diverge"
    ap = parser(name)
    ap.add_argument(
        "--reference-arm", required=True, help="one arm in --arms used as reference"
    )
    ap.add_argument(
        "--envelope",
        type=Path,
        help="frozen calibration JSON: object_gap_threshold, eef_gap_threshold and provenance",
    )
    ap.add_argument(
        "--library-root",
        type=Path,
        help="frozen action.npy/rs.npy for E4 sigma diagnostics",
    )
    args = ap.parse_args(argv)
    if args.reference_arm not in args.arms:
        ap.error("--reference-arm must be in --arms")
    envelope = json.loads(args.envelope.read_text()) if args.envelope else None
    episodes = load_episodes(args.run_root, args.arms, args.p3v2, args.limit)
    config = load_config(args, episodes)
    scales = library_scales(args.library_root) if args.library_root else None
    tables, summary = build(
        episodes, args.reference_arm, config, envelope, twin, scales
    )
    report(args.out, name, tables, summary, episodes=episodes)


if __name__ == "__main__":
    main()
