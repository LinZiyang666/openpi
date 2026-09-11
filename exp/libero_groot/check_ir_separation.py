"""Did the addressed budgets actually come out apart?

The failure this exists to catch has happened on this project before: a grid of
arms addressed at distinct inference ratios that all land on top of each other
once deployed, so the frontier has one working point drawn sixteen times. It is
invisible in every per-arm number -- each arm runs, succeeds at some rate, and
reports a plausible ratio -- and only shows up when the ratios are sorted and
their gaps are read.

Two questions, both answered here:

*   **Separation.** Sorted realized IR, adjacent gaps, and how many pairs sit
    closer together than the grid's own step. A collapsed grid is a design
    failure to act on, not a number to publish.
*   **Drift.** Realized minus predicted, per arm. The prediction is made on the
    calibration scores under no gate; deployment adds the hysteresis gate's
    forced teacher calls and whatever covariate shift the closed loop
    introduces. A large, *uniform* drift is the gate and is expected; a drift
    that grows with the budget is the shift, and it is what silently re-orders
    a frontier.

Both are computed from the verdict counts, so only the cost constants come from
the ledger. Without ``--cost`` the check still runs on the cache-use fraction,
which is IR's only free variable once the constants are fixed.

Usage:
  uv run python -m exp.libero_groot.check_ir_separation \
      --aggregate <dir>/aggregate.json --arm-record <suite>/arm_record.json \
      [--cost <json>] [--min-gap 1.0]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from openpi.cache.types import groot_n15_schedule

from exp.robocasa365 import rit_cost_rc as rc


def load_cost(path: pathlib.Path) -> rc.StageCost:
    d = json.loads(path.read_text(encoding="utf-8"))
    return rc.StageCost(
        teacher=d.get("teacher", "groot_libero"),
        schedule=groot_n15_schedule(int(d.get("num_steps", 8))),
        stage1_ms=float(d["stage1_ms"]), stage2_ms=float(d["stage2_ms"]),
        stage3_head_ms=float(d["stage3_head_ms"]), stage3_step_ms=float(d["stage3_step_ms"]),
        provenance=str(d["provenance"]),
    )


def realized_ir(counts: dict, cost: rc.StageCost) -> float | None:
    """Priced verdict mix, as a percent of the all-MISS cost.

    Warm counts are read from the per-tier keys (``WARM_START@0.75``), never
    from the pooled ``WARM_START`` total: the two rungs differ by about 13 IR
    points at full share, so pooling them would erase most of what the ladder
    is measuring.
    """
    n = sum(v for k, v in counts.items() if k in ("FULL_HIT", "MISS") or k.startswith("WARM_START@"))
    if not n:
        return None
    spend = counts.get("FULL_HIT", 0) * rc.tier_cost(cost, "FULL_HIT", None)
    spend += counts.get("MISS", 0) * rc.miss_cost(cost)
    for key, value in counts.items():
        if key.startswith("WARM_START@"):
            spend += value * rc.tier_cost(cost, "WARM_START", float(key.split("@")[1]))
    return 100.0 * spend / (n * rc.miss_cost(cost))


def cache_use(counts: dict) -> float | None:
    """Fraction of decisions the cache answered, at any rung. Cost-free proxy."""
    n = sum(v for k, v in counts.items() if k in ("FULL_HIT", "MISS") or k.startswith("WARM_START@"))
    if not n:
        return None
    return (counts.get("FULL_HIT", 0) + counts.get("WARM_START", 0)) / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aggregate", required=True)
    ap.add_argument("--arm-record", default="")
    ap.add_argument("--cost", default="")
    ap.add_argument("--min-gap", type=float, default=1.0,
                    help="adjacent pairs closer than this are reported as collapsed")
    ap.add_argument("--min-episodes", type=int, default=500,
                    help="only arms with at least this many episodes are judged")
    args = ap.parse_args()

    agg = json.loads(pathlib.Path(args.aggregate).read_text(encoding="utf-8"))
    cost = load_cost(pathlib.Path(args.cost)) if args.cost else None
    record = (json.loads(pathlib.Path(args.arm_record).read_text(encoding="utf-8"))
              if args.arm_record else {"arms": {}})

    rows = []
    for arm, r in sorted(agg.items()):
        if r["n_ep"] < args.min_episodes:
            continue
        value = realized_ir(r["counts"], cost) if cost else cache_use(r["counts"])
        if value is None:
            continue
        meta = record.get("arms", {}).get(arm, {})
        rows.append({
            "arm": arm, "rule": meta.get("rule", "?"), "k": meta.get("k"),
            "target_ir": meta.get("target_ir"), "value": value,
            "sr": r["success_rate"], "n_ep": r["n_ep"],
        })
    if not rows:
        raise SystemExit(f"no arm in {args.aggregate} has {args.min_episodes} episodes yet")

    unit = "IR%" if cost else "cache-use"
    rows.sort(key=lambda x: x["value"])
    print(f"{len(rows)} complete arms, sorted by {unit}"
          + ("" if cost else "  (no cost ledger: cache-use fraction stands in for IR)"))
    prev = None
    collapsed = []
    for i, r in enumerate(rows):
        gap = None if prev is None else r["value"] - prev["value"]
        drift = (r["value"] - r["target_ir"]) if (cost and r["target_ir"] is not None) else None
        flag = ""
        if gap is not None and gap < args.min_gap:
            flag = "  <== collapsed onto the one above"
            collapsed.append((prev["arm"], r["arm"], gap))
        print(f"  {r['arm']:28s} {unit}={r['value']:7.3f}"
              + (f" gap={gap:6.3f}" if gap is not None else "  gap=     -")
              + f" sr={r['sr']:.3f} n={r['n_ep']}"
              + (f" drift={drift:+.2f}" if drift is not None else "") + flag)
        prev = r

    gaps = [rows[i + 1]["value"] - rows[i]["value"] for i in range(len(rows) - 1)]
    if gaps:
        print(f"\nadjacent gaps: min {min(gaps):.3f}  median {sorted(gaps)[len(gaps) // 2]:.3f}  "
              f"max {max(gaps):.3f}  span {rows[-1]['value'] - rows[0]['value']:.3f}")
    print(f"collapsed pairs (< {args.min_gap}): {len(collapsed)}/{max(len(rows) - 1, 1)}")
    for a, b, g in collapsed:
        print(f"  {a} ~ {b}  ({g:.3f})")

    if cost:
        drifts = [r["value"] - r["target_ir"] for r in rows if r["target_ir"] is not None]
        if drifts:
            print(f"\nrealized - addressed: mean {sum(drifts)/len(drifts):+.2f}  "
                  f"min {min(drifts):+.2f}  max {max(drifts):+.2f}  "
                  f"spread {max(drifts)-min(drifts):.2f}")
            print("  a uniform offset is the gate's forced teacher calls; a spread that "
                  "grows with the budget is closed-loop covariate shift")


if __name__ == "__main__":
    main()
