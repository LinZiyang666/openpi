"""Descriptive characterisation of a `stop_loss_zero_hitshare` fit (post-hoc).

This answers the first question a reader of the negative result will ask: was
acceptance killed by the OOF safety offset, or by the deviation floor itself?
The two have very different implications -- the first is a property of the
pre-registered selection rule, the second is a property of the policy.

It re-uses `fit_surface`'s own functions read-only and CHANGES NOTHING: no
artifact is written, no threshold is re-derived, and the frozen delta rule is
not re-run. The numbers here are descriptive characterisation of an already
recorded stop-loss (plan section 7 item 7), never an alternative fit.

Usage:
  uv run python -m exp.dispatch_surface.analysis.characterize_stop_loss \
      --table exp/dispatch_surface/data/dispatch_table_fresh.jsonl \
      --fit-record exp/dispatch_surface/data/surface_fit/fit_record.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from exp.dispatch_surface.fit_surface import (
    GRID_LADDER_SV,
    assign_folds,
    evaluate_candidate_deployed,
    fit_fold_models,
    load_table,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True)
    ap.add_argument("--fit-record", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    record = json.loads(pathlib.Path(args.fit_record).read_text())
    if record.get("stop_loss") != "stop_loss_zero_hitshare":
        raise SystemExit(
            f"this tool only characterises stop_loss_zero_hitshare; record says "
            f"{record.get('stop_loss')!r}"
        )
    offset = float(record["oof_safety_offset"])
    grid = np.asarray(record["delta_grid"], dtype=float)

    table = load_table(args.table)
    fit_mask = table.split == "fit"
    models = fit_fold_models(
        table, fit_mask, assign_folds(table, fit_mask), GRID_LADDER_SV, args.alpha,
    )
    if models is None:
        raise SystemExit("fold models unavailable; the recorded stop was earlier than delta selection")

    y10 = table.y10[fit_mask]
    out = {
        "oof_safety_offset": offset,
        "delta_grid": grid.tolist(),
        "y10_floor": float(y10.min()),
        "y10_spread_p10_p90": float(np.percentile(y10, 90) - np.percentile(y10, 10)),
        "counterfactual_offsets": {},
    }
    # The counterfactual sweep is over the OFFSET ONLY. delta stays on its frozen
    # grid, so this reports "how much of the zero hitshare is the offset" and
    # nothing else. offset=0 is the no-safety-margin limit; it is NOT a proposal.
    for off in (0.0, offset / 4, offset / 2, offset):
        shares = {}
        for d in grid:
            share, acc = evaluate_candidate_deployed(
                float(d), models, table, fit_mask, off, uses_disagreement=True,
            )
            shares[f"{d:.4f}"] = {"hitshare": share, "accepted_step_accuracy": acc}
        best = max(v["hitshare"] for v in shares.values())
        out["counterfactual_offsets"][f"{off:.4f}"] = {
            "max_hitshare_over_grid": best,
            "per_delta": shares,
        }

    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
