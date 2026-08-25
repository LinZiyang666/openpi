"""Turn a coarse round's weight marginals into the fine round's search box.

The fine round must be aimed by data, but unattended it also has to be aimed
*deterministically* -- "look at the table and pick a region" is not something a
script can do at 3am. So the rule is fixed here and only its inputs come from
the coarse results:

  * per field, take the coarse weight whose marginal success is highest (the
    marginals are averages over a balanced grid, far steadier than any single
    cell, and every field's curve so far has been an interior-peaked inverted U);
  * re-express that peak in the fine grid's units and take a symmetric box
    around it;
  * drop cells where any field is zero -- the coarse rounds put the corners and
    edges decisively at the bottom, so spending fine cells there buys nothing.

The box half-width adapts so the cell count lands in a usable band: too few and
the round cannot separate anything, too many and it overruns the episode budget.
Ranking the *leader's* neighbourhood instead would be the obvious alternative
and the wrong one -- when the tie set is wide, the leader's position is mostly
noise, while the marginal peak is an average over many cells.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

_CID = re.compile(r"v0@(\d+)_v1@(\d+)_rs@(\d+)")
FIELDS = ("vision_0", "vision_1", "robot_state")
TARGET_LO, TARGET_HI = 18, 32


def marginal_peaks(sr: dict[str, float], coarse_steps: int) -> tuple[int, ...]:
    """Per-field coarse unit weight with the highest mean success."""
    acc: list[dict[int, list[float]]] = [collections.defaultdict(list) for _ in FIELDS]
    for cid, value in sr.items():
        m = _CID.fullmatch(cid)
        if m is None:
            continue
        for i, unit in enumerate(int(g) for g in m.groups()):
            acc[i][unit].append(value)
    peaks = []
    for i, _ in enumerate(FIELDS):
        means = {u: sum(v) / len(v) for u, v in acc[i].items()}
        # ties broken toward the lower weight: a field that does as well with
        # less weight is leaving room for the others.
        peaks.append(min(means, key=lambda u: (-means[u], u)))
    del coarse_steps
    return tuple(peaks)


def count_cells(fine_steps: int, lo: tuple[int, ...], hi: tuple[int, ...]) -> int:
    n = 0
    for a in range(fine_steps + 1):
        for b in range(fine_steps + 1 - a):
            units = (a, b, fine_steps - a - b)
            if all(low <= u <= high for u, low, high in zip(units, lo, hi, strict=True)):
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", type=pathlib.Path, help="Coarse round's *_summary.json")
    ap.add_argument("--coarse-steps", type=int, default=6)
    ap.add_argument("--fine-steps", type=int, default=12)
    args = ap.parse_args()

    sr = json.loads(args.summary.read_text())["sr"]
    peaks = marginal_peaks(sr, args.coarse_steps)
    scale = args.fine_steps // args.coarse_steps
    centres = tuple(p * scale for p in peaks)

    chosen = None
    for half in (3, 2, 4, 1, 5):
        lo = tuple(max(1, c - half) for c in centres)
        hi = tuple(min(args.fine_steps - 2, c + half) for c in centres)
        n = count_cells(args.fine_steps, lo, hi)
        if TARGET_LO <= n <= TARGET_HI:
            chosen = (lo, hi, n, half)
            break
    if chosen is None:  # nothing landed in band: take the widest tried
        lo = tuple(max(1, c - 5) for c in centres)
        hi = tuple(min(args.fine_steps - 2, c + 5) for c in centres)
        chosen = (lo, hi, count_cells(args.fine_steps, lo, hi), 5)

    lo, hi, n, half = chosen
    print(f"peaks(coarse units)={peaks} centres(fine)={centres} half={half} cells={n}")
    print(f"FIELD_MIN={','.join(str(x) for x in lo)}")
    print(f"FIELD_MAX={','.join(str(x) for x in hi)}")


if __name__ == "__main__":
    main()
