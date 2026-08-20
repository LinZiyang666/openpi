"""Solve the judge threshold grid and the gate theta from a fresh warmup dump.

Both numbers are quantile cuts of the same distribution -- the per-step
``cp1_score`` collected along the true policy trajectory under a force-MISS
warmup -- so they are re-derived per library rather than carried over. Carrying
a threshold across libraries is exactly the mistake the ratio-based design
exists to prevent: two libraries with different score spreads reach the same
operating point at different numeric cuts.

Cut convention is imported, not re-implemented: ``derive_thresholds`` from the
verdict phase-3 solver is what produced every historical threshold in this line,
and a second implementation that rounded differently would silently make this
run incomparable to them.

*   **judge**: for each target FULL_HIT fraction ``f_FH``, ``T_fh`` is the cut
    admitting the top ``f_FH`` of the distribution. No warm tier is solved --
    the warm-start route is disabled for this experiment, so the verdict is
    binary and ``f_FH`` is the single degree of freedom.
*   **gate**: ``theta_low = theta_high = `` the cut admitting the top 0.85,
    which reproduces the historical N4 operating point (libero_spatial anchored
    at 0.968914 = that base's fh75+ws10 boundary; the shipped gate used
    0.968929). It is held FIXED across the 16 cells so the sweep isolates the
    judge threshold; a theta that tracked ``T_fh`` would move the gate and the
    verdict together and no cell difference could be attributed.

Public interface: ``FH_GRID``, ``THETA_TOP_FRACTION``, ``load_scores_by_arm``,
``solve_arm``, ``solve_all``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from exp.verdict_factor_judge.phase3.threshold_solver import derive_thresholds

#: 16 target FULL_HIT fractions -- the historical f_FH axis, unchanged. The old
#: experiment crossed this with an f_WS axis; with warm start disabled the grid
#: collapses to this one axis, 16 cells per library instead of 83.
FH_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 17))

#: Top fraction admitted by the gate's hysteresis threshold (see module docstring).
THETA_TOP_FRACTION = 0.85

#: Below this many usable scores a quantile grid this fine is not supportable:
#: the sparsest cell (f_FH = 0.05) would rest on fewer than 25 samples.
MIN_SCORES = 500


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------


def load_scores_by_arm(per_step_jsonl: str | Path) -> dict[str, list[float]]:
    """Group finite per-step ``cp1_score`` values by ``yaml_id``.

    Rows without a score are dropped rather than counted as zero: under the
    force-MISS warmup a null score means the step never reached search, and
    folding those in as a value would drag every quantile downward.
    """
    out: dict[str, list[float]] = defaultdict(list)
    with Path(per_step_jsonl).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            score = rec.get("cp1_score")
            if score is None:
                continue
            score = float(score)
            if math.isnan(score):
                continue
            arm = rec.get("yaml_id") or rec.get("task_uid", "").split(":", 1)[0]
            if arm:
                out[arm].append(score)
    return dict(out)


# ------------------------------------------------------------------
# Solving
# ------------------------------------------------------------------


def solve_arm(scores: list[float], *, arm: str = "") -> dict:
    """Return the gate theta plus one ``T_fh`` per grid cell for one library."""
    usable = [s for s in scores if s is not None and not math.isnan(s)]
    if len(usable) < MIN_SCORES:
        raise SystemExit(
            f"arm {arm!r}: only {len(usable)} usable warmup scores (need >= {MIN_SCORES}). "
            "Solving a 16-cell quantile grid on this few would put the sparsest cell on "
            "a handful of samples; extend the warmup instead of lowering the bar."
        )
    theta = derive_thresholds(usable, THETA_TOP_FRACTION, 0.0)[0]
    cells = []
    for f_fh in FH_GRID:
        t_fh = derive_thresholds(usable, f_fh, 0.0)[0]
        cells.append({"f_fh": f_fh, "t_fh": t_fh})

    spread = {
        "n": len(usable),
        "min": min(usable),
        "max": max(usable),
        "distinct": len(set(usable)),
    }
    # A distribution with fewer distinct values than cells cannot separate them;
    # the sweep would emit duplicate operating points that look like independent
    # evidence. Surface it here, where it is still cheap to fix.
    if spread["distinct"] < len(FH_GRID):
        raise SystemExit(
            f"arm {arm!r}: warmup scores take only {spread['distinct']} distinct values, "
            f"fewer than the {len(FH_GRID)} grid cells; the cells cannot be separated."
        )
    return {"theta": theta, "cells": cells, "spread": spread}


def solve_all(per_step_jsonl: str | Path) -> dict:
    """Solve every arm present in a warmup dump."""
    by_arm = load_scores_by_arm(per_step_jsonl)
    if not by_arm:
        raise SystemExit(f"no cp1_score rows in {per_step_jsonl}")
    return {
        "source": str(per_step_jsonl),
        "fh_grid": list(FH_GRID),
        "theta_top_fraction": THETA_TOP_FRACTION,
        "arms": {
            arm: solve_arm(scores, arm=arm) for arm, scores in sorted(by_arm.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Solve thresholds from a warmup dump")
    ap.add_argument("--warmup-per-step", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    solved = solve_all(args.warmup_per_step)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(solved, indent=2) + "\n", encoding="utf-8")
    for arm, data in solved["arms"].items():
        sp = data["spread"]
        print(
            f"{arm:16s} n={sp['n']:6d} distinct={sp['distinct']:5d} "
            f"range=[{sp['min']:.6f}, {sp['max']:.6f}] theta={data['theta']:.6f}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
