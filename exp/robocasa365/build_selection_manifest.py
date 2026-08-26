"""Frozen, reproducible cell selection for the ws2 search round (plan §3-W8).

Generates ``selection_manifest.json`` — the single audited source of the
ws2c control-arm and ws2e densify-arm cell sets. The driver refuses those
phases without it and pins its sha on first launch, so a resume can never
re-select.

Statistical protocol is byte-for-byte the round-1 analyzer's: complete cells
via ``load_journals``, macro_sr ranking, paired sign-flip test vs the leader
(``signflip_p``) with ``random.Random(seed)`` consumed in ranked order, and
the analyzer's own default seed 12345 — so the tied set reproduces the
published round-1 readout exactly.

Segment rules (all frozen; ties broken by cid ascending everywhere):

- ``ws2c``: the 4 iso cids (from the index) ∪ the first 8 NON-iso cells of
  the leader-tied set ordered by (macro_sr desc, cid asc). If fewer than 8
  non-iso cells tie, the shortfall is padded from the remaining non-iso cells
  in the same order and ``padding_used`` records it. Union is exactly 12.
- ``ws2e``: leader-tied top 8 by the same key (leader included; same padding
  rule), plus the 2 significantly-worse (p < alpha) cells with the LOWEST
  macro_sr as negative controls. Fewer than 2 significant cells is a
  fail-fast — an owner decision, never a silent downgrade.

The manifest is append-only: adding a segment must leave every existing
segment byte-identical, and re-generating an existing segment is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random

from exp.robocasa365.analyze_ws_search_stats import load_journals, macro_sr, signflip_p

ALGORITHM = "ws2_selection_v1"


def paired_stats(
    cells: dict, keys: list, *, seed: int, resamples: int
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Ranked cids + macro/p maps, consuming the rng exactly like the analyzer.

    Ranking is ``sorted(cells, key=-macro)`` — Python's stable sort over the
    cid-ascending insertion order, i.e. (macro desc, cid asc). The rng is
    advanced in ranked order, one ``signflip_p`` per non-leader cell, which is
    what makes the p-values bit-reproducible against the round-1 readout.
    """
    tasks = sorted({t for t, _ in keys})
    idxs = sorted({i for _, i in keys})
    macros = {cid: macro_sr(cells[cid], tasks, idxs) for cid in cells}
    ranked = sorted(cells, key=lambda c: -macros[c])
    leader = ranked[0]
    rng = random.Random(seed)
    weight = 1.0 / (len(tasks) * len(idxs))
    pvals: dict[str, float] = {leader: 1.0}
    for cid in ranked:
        if cid == leader:
            continue
        diffs = [int(cells[leader][k]) - int(cells[cid][k]) for k in keys]
        _, p = signflip_p(diffs, weight, rng, resamples)
        pvals[cid] = p
    return ranked, macros, pvals


def pick_top(ranked: list[str], macros: dict, pvals: dict, *, alpha: float,
             count: int, exclude: set[str]) -> tuple[list[str], bool]:
    """First ``count`` tied cells by (macro desc, cid asc), padded if short."""
    order_key = lambda c: (-macros[c], c)  # noqa: E731 - frozen sort key
    tied = sorted((c for c in ranked if pvals[c] >= alpha and c not in exclude), key=order_key)
    picked = tied[:count]
    padding_used = len(picked) < count
    if padding_used:
        rest = sorted((c for c in ranked if c not in exclude and c not in picked), key=order_key)
        picked += rest[: count - len(picked)]
    return picked, padding_used


def build_segment(
    journal_dir: pathlib.Path,
    index: dict,
    *,
    segment: str,
    run_prefix: str,
    episodes: int,
    resamples: int,
    alpha: float,
    seed: int,
) -> dict:
    cells, keys = load_journals(journal_dir, episodes, run_prefix)
    missing = sorted(set(index) - set(cells))
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(index)} index cells are not complete in {journal_dir} "
            f"(first: {missing[:4]}). Selection requires the full complete matrix."
        )
    # Restrict to the audited matrix: a stray complete journal from a probe or
    # a superseded cell would otherwise enter the ranking and could be selected,
    # only failing much later at the emitter.
    extra = sorted(set(cells) - set(index))
    if extra:
        print(f"[manifest] ignoring {len(extra)} journal(s) outside the index: {extra[:4]}",
              flush=True)
        cells = {cid: outcomes for cid, outcomes in cells.items() if cid in index}
    ranked, macros, pvals = paired_stats(cells, keys, seed=seed, resamples=resamples)
    leader = ranked[0]
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(journal_dir.glob(f"journal_{run_prefix}-*.jsonl"))
    }
    params = {"episodes": episodes, "resamples": resamples, "alpha": alpha, "seed": seed,
              "run_prefix": run_prefix}

    if segment == "ws2c":
        iso_cids = sorted(c for c in index if c.split("_", 1)[0] == "iso")
        if len(iso_cids) != 4:
            raise SystemExit(f"expected exactly 4 iso cells in the index, got {iso_cids}")
        top8, padding_used = pick_top(ranked, macros, pvals, alpha=alpha, count=8, exclude=set(iso_cids))
        cells_out = sorted(set(iso_cids) | set(top8))
        if len(cells_out) != 12:
            raise SystemExit(f"ws2c union must be exactly 12 cells, got {len(cells_out)}")
        return {
            "params": params, "source_journals": sources, "leader": leader,
            "iso_cids": iso_cids, "top8_cids": top8, "cells": cells_out,
            "padding_used": padding_used,
        }

    if segment == "ws2e":
        top8, padding_used = pick_top(ranked, macros, pvals, alpha=alpha, count=8, exclude=set())
        significant = [c for c in ranked if pvals[c] < alpha]
        if len(significant) < 2:
            raise SystemExit(
                f"only {len(significant)} cells are significantly worse than the leader "
                f"(p < {alpha}); the 2 negative controls cannot be chosen. Frozen rule: "
                "fail fast and escalate to the owner — no silent downgrade."
            )
        negatives = sorted(significant, key=lambda c: (macros[c], c))[:2]
        cells_out = sorted(set(top8) | set(negatives))
        if len(cells_out) != 10:
            # Padding draws from the same ranked pool as the negatives, so a
            # degenerate matrix could make the two sets overlap and silently
            # shrink the arm.
            raise SystemExit(
                f"ws2e must be exactly 10 cells (8 tied + 2 negative controls); got "
                f"{len(cells_out)} — top8={top8} negatives={negatives}"
            )
        return {
            "params": params, "source_journals": sources, "leader": leader,
            "top8_cids": top8, "negative_cids": negatives, "cells": cells_out,
            "padding_used": padding_used,
        }

    raise SystemExit(f"unknown segment {segment!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", required=True, choices=("ws2c", "ws2e"))
    ap.add_argument("--journal-dir", required=True,
                    help="per-cell journals the selection reads (round-1 dir for ws2c, "
                    "ws2 finalized dir for ws2e)")
    ap.add_argument("--index", required=True, help="index.json naming the full cell matrix")
    ap.add_argument("--manifest", required=True, help="selection_manifest.json to create/append")
    ap.add_argument("--run-prefix", default="", help="journal round tag (default: ws1 for ws2c, ws2 for ws2e)")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=0.05)
    # The round-1 analyzer's own default; reproduces its published tied set.
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    run_prefix = args.run_prefix or ("ws1" if args.segment == "ws2c" else "ws2")
    index = json.loads(pathlib.Path(args.index).read_text())
    segment = build_segment(
        pathlib.Path(args.journal_dir), index,
        segment=args.segment, run_prefix=run_prefix, episodes=args.episodes,
        resamples=args.resamples, alpha=args.alpha, seed=args.seed,
    )

    manifest_path = pathlib.Path(args.manifest)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if args.segment in manifest.get("segments", {}):
            raise SystemExit(
                f"segment {args.segment!r} already exists in {manifest_path}; the manifest "
                "is append-only — refusing to regenerate an audited selection."
            )
        before = {name: json.dumps(seg, sort_keys=True) for name, seg in manifest["segments"].items()}
    else:
        manifest = {"version": 1, "algorithm": ALGORITHM, "segments": {}}
        before = {}

    manifest["segments"][args.segment] = segment
    for name, frozen in before.items():
        if json.dumps(manifest["segments"][name], sort_keys=True) != frozen:
            raise SystemExit(f"append-only violation: segment {name!r} would change; aborting")
    # Atomic replace: the driver sha-pins this file, so a crash mid-write must
    # not leave a truncated audit record behind (write_run_plan's pattern).
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    tmp_path.replace(manifest_path)
    print(
        f"[manifest] {args.segment}: cells={segment['cells']} "
        f"padding_used={segment['padding_used']} -> {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
