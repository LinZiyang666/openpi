"""Rank the search cells and read the shape of the weight space.

Three views, in increasing order of how much they should be trusted:

*   **Ranking** -- macro success per cell. Point estimates; with 500 episodes a
    cell's own standard error is ~1.5 points, but the gap between two cells is
    what matters and that is what the paired test below measures.
*   **Weight marginals** -- mean success over every cell in which a field
    carries a given weight. Averaged over a balanced grid this is far more
    stable than any single cell, and it is what says whether a field helps at
    all.
*   **Paired sign-flip test vs the leader** -- every cell runs the same (task,
    init) pairs, so the comparison can be made episode by episode. Unpaired
    comparison throws that away and is much less sensitive; the tie set it
    reports would be far larger.

The tie set is the honest output: the leader is only "the leader" if the cells
statistically indistinguishable from it are few.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import re

_CID = re.compile(r"v0@(\d+)_v1@(\d+)_rs@(\d+)")
FIELDS = ("vision_0", "vision_1", "robot_state")


def parse_cid(cid: str) -> tuple[int, int, int]:
    m = _CID.fullmatch(cid)
    if m is None:
        raise ValueError(f"unparseable cell id: {cid!r}")
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


def load(results_dir: pathlib.Path) -> dict[str, dict[tuple[int, int], bool]]:
    """cell -> {(task_id, init_idx): success}."""
    cells: dict[str, dict[tuple[int, int], bool]] = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        rows = json.loads(path.read_text())
        cells[path.stem] = {
            (int(r["task_id"]), int(r["init_state_idx"])): bool(r["success"]) for r in rows
        }
    return cells


def sign_flip_p(a: dict, b: dict, *, iters: int = 20000, seed: int = 0) -> tuple[float, int, int]:
    """Two-sided paired randomization test. Returns (p, n_a_only, n_b_only)."""
    shared = a.keys() & b.keys()
    diffs = [int(a[k]) - int(b[k]) for k in shared]
    nz = [d for d in diffs if d != 0]
    a_only = sum(1 for d in nz if d > 0)
    b_only = len(nz) - a_only
    if not nz:
        return 1.0, 0, 0
    observed = abs(sum(nz))
    rng = random.Random(seed)
    hits = sum(
        1 for _ in range(iters)
        if abs(sum(d if rng.random() < 0.5 else -d for d in nz)) >= observed
    )
    return (hits + 1) / (iters + 1), a_only, b_only


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=pathlib.Path)
    ap.add_argument("--steps", type=int, default=6, help="Simplex resolution the ids use")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json-out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    cells = load(args.results_dir)
    if not cells:
        raise SystemExit(f"no complete result files in {args.results_dir}")
    sr = {c: sum(v.values()) / len(v) for c, v in cells.items()}
    order = sorted(sr, key=lambda c: -sr[c])
    n_ep = len(next(iter(cells.values())))
    print(f"{len(cells)} cells, {n_ep} episodes each\n")

    leader = order[0]
    print(f"{'rank':>4}  {'cell':<22}{'SR':>7}{'n':>6}   vs leader (p, cell-only/leader-only)")
    ties = []
    for i, c in enumerate(order[: args.top], start=1):
        if c == leader:
            print(f"{i:>4}  {c:<22}{sr[c]:>7.3f}{len(cells[c]):>6}   (leader)")
            continue
        p, ao, bo = sign_flip_p(cells[c], cells[leader])
        tie = p >= 0.05
        ties.append(c) if tie else None
        print(f"{i:>4}  {c:<22}{sr[c]:>7.3f}{len(cells[c]):>6}   "
              f"p={p:.3f} {ao}/{bo}{'  TIE' if tie else ''}")

    all_ties = [c for c in order[1:] if sign_flip_p(cells[c], cells[leader])[0] >= 0.05]
    print(f"\nstatistically tied with the leader: {len(all_ties)}/{len(cells) - 1}")
    if len(all_ties) > len(cells) / 3:
        print("  -> the resolution is a region, not a winner; treat the leader as one of many")

    print("\nweight marginals (mean SR over cells where the field carries that weight)")
    print(f"{'w':>6}" + "".join(f"{f:>14}" for f in FIELDS))
    marg: dict[str, dict[int, list[float]]] = {f: collections.defaultdict(list) for f in FIELDS}
    for c in cells:
        for f, u in zip(FIELDS, parse_cid(c), strict=True):
            marg[f][u].append(sr[c])
    for u in range(args.steps + 1):
        row = f"{u / args.steps:>6.3f}"
        for f in FIELDS:
            vals = marg[f].get(u, [])
            row += f"{(sum(vals) / len(vals) if vals else float('nan')):>14.3f}"
        print(row)

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"sr": sr, "leader": leader, "tied_with_leader": all_ties,
             "episodes_per_cell": n_ep}, indent=2, sort_keys=True))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
