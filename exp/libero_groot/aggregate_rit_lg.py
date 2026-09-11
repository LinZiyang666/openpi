"""Aggregate LIBERO x GR00T rollouts into per-arm success rate and verdict mix.

The sibling of ``exp.rit_pareto.aggregate_rit``. Episode acceptance and
per-step matching are deliberately the same rules -- only terminal, accepted
journal rows count, and a per-step row belongs to the accepted attempt of its
episode -- because a difference there would move every number without failing
anything. The two differ in exactly one place: pricing.

``aggregate_rit`` prices through ``exp.dispatch_surface.analysis.analytic_cost``,
whose constants are Pi0.5's and whose ``unit_cost`` refuses any warm tier other
than the pinned 0.3. This line's rungs are GR00T's 0.75 and 0.5 on an
**ascending** schedule, where ``start_t`` is elapsed flow rather than remaining,
so the Pi0.5 formula would not merely be the wrong constants -- it would price
the ladder backwards.

The cost model is therefore optional. Without ``--cost`` this writes success
rate and the verdict counts, with ``ir_percent: null``: those are the numbers
the ladder's *accuracy* side is read from, and they do not wait on a latency
measurement. With ``--cost`` the same rows are priced through
``rit_cost_rc.tier_cost`` and the inference ratio is filled in.

Usage:
  uv run python -m exp.libero_groot.aggregate_rit_lg \
      --data-dir exp/libero_groot/data/rit/eval/libero_spatial_hg \
      --out exp/libero_groot/data/rit/eval/libero_spatial_hg/aggregate.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

from openpi.cache.types import groot_n15_schedule

from exp.robocasa365 import rit_cost_rc as rc

VERDICTS = ("FULL_HIT", "WARM_START", "MISS")
_TARGET = re.compile(r"_ir(\d+)$")
_GST_CELL = re.compile(r"_gst_f(\d+)w(\d+)v(\d+)$")
_K = re.compile(r"_rit_k(\d+)_")


def target_of(arm: str) -> float | None:
    m = _TARGET.search(arm)
    return float(m.group(1)) if m else None


def gst_cell_of(arm: str) -> tuple[int, int, int] | None:
    m = _GST_CELL.search(arm)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def ladder_k_of(arm: str) -> int | None:
    m = _K.search(arm)
    return int(m.group(1)) if m else None


def load_cost(path: pathlib.Path) -> rc.StageCost:
    d = json.loads(path.read_text(encoding="utf-8"))
    return rc.StageCost(
        teacher=d.get("teacher", "groot_libero"),
        schedule=groot_n15_schedule(int(d.get("num_steps", 8))),
        stage1_ms=float(d["stage1_ms"]),
        stage2_ms=float(d["stage2_ms"]),
        stage3_head_ms=float(d["stage3_head_ms"]),
        stage3_step_ms=float(d["stage3_step_ms"]),
        provenance=str(d["provenance"]),
        linear_stage3=bool(d.get("linear_stage3", False)),
    )


def accepted_episodes(journal: pathlib.Path) -> tuple[dict[str, int], dict[str, dict[str, bool]]]:
    """``task_uid -> accepted attempt`` and ``arm -> {task_uid: solved}``.

    A retried episode appears more than once; only the attempt the driver
    accepted is the one whose per-step rows describe what ran.
    """
    attempt: dict[str, int] = {}
    episodes: dict[str, dict[str, bool]] = collections.defaultdict(dict)
    with journal.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("status") not in ("done", "failed") or not row.get("accepted"):
                continue
            attempt[row["task_uid"]] = row["attempt"]
            episodes[row["yaml_id"]][row["task_uid"]] = row["status"] == "done"
    return attempt, episodes


def aggregate(data_dir: pathlib.Path, cost: rc.StageCost | None = None) -> dict:
    attempt, episodes = accepted_episodes(data_dir / "journal.jsonl")

    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    spend: dict[str, float] = collections.defaultdict(float)
    per_step = data_dir / "per_step.jsonl"
    if per_step.exists():
        with per_step.open(encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                hit_type = row.get("hit_type")
                if hit_type is None:
                    continue
                if attempt.get(row["task_uid"]) != row.get("attempt"):
                    continue
                if hit_type not in VERDICTS:
                    raise SystemExit(f"unpriceable hit_type {hit_type!r} in {data_dir}")
                start_t = row.get("start_t")
                key = hit_type if hit_type != "WARM_START" else f"WARM_START@{float(start_t):g}"
                counts[row["yaml_id"]][key] += 1
                if cost is not None:
                    spend[row["yaml_id"]] += rc.tier_cost(cost, hit_type, start_t)

    miss_ms = rc.miss_cost(cost) if cost is not None else None
    out = {}
    for yaml_id in sorted(episodes):
        eps = episodes[yaml_id]
        c = counts[yaml_id]
        n_dec = sum(c.values())
        warm_keys = sorted(k for k in c if k.startswith("WARM_START@"))
        out[yaml_id] = {
            "n_ep": len(eps),
            "success_rate": sum(eps.values()) / len(eps) if eps else None,
            "decisions": n_dec,
            "counts": {
                "FULL_HIT": c.get("FULL_HIT", 0),
                "WARM_START": sum(c[k] for k in warm_keys),
                "MISS": c.get("MISS", 0),
                **{k: c[k] for k in warm_keys},
            },
            "ir_percent": (
                100.0 * spend[yaml_id] / (n_dec * miss_ms)
                if cost is not None and n_dec
                else None
            ),
            "target_ir": target_of(yaml_id),
            "gst_cell": gst_cell_of(yaml_id),
            "ladder_k": ladder_k_of(yaml_id),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cost", default="", help="stage-cost json; omit to leave ir_percent null")
    args = ap.parse_args()

    cost = load_cost(pathlib.Path(args.cost)) if args.cost else None
    result = aggregate(pathlib.Path(args.data_dir), cost)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    done = [a for a, r in result.items() if r["n_ep"] >= 500]
    print(f"wrote {out}  arms={len(result)} complete={len(done)}"
          + ("" if cost else "  (ir_percent null: no cost ledger)"))
    for arm in sorted(result):
        r = result[arm]
        ir = "    -" if r["ir_percent"] is None else f"{r['ir_percent']:5.1f}"
        print(f"  {arm:28s} n={r['n_ep']:4d} sr={r['success_rate']:.3f} ir={ir}"
              f"  FH={r['counts']['FULL_HIT']:6d} WS={r['counts']['WARM_START']:6d}"
              f" MISS={r['counts']['MISS']:6d}")


if __name__ == "__main__":
    main()
