"""Aggregate RIT-Pareto rollouts into per-arm success rate and inference ratio.

``aggregate``: from one run directory (``journal.jsonl`` + ``per_step.jsonl``
written by ``run_gtp``) to per-arm success rate and the three-tier inference
ratio. Episode acceptance follows ``analyze_gtp``: only terminal, accepted
journal rows count, and per-step rows are matched to the accepted attempt.
The cost of a decision is ``unit_cost(hit_type, start_t)`` from the single
cost authority (``exp.dispatch_surface.analysis.analytic_cost``), so

    IR% = 100 * sum(cost) / (n_decisions * unit_cost("MISS"))

is the same estimand ``rit_pl.predicted_ir`` addressed offline; the GST
inference_ratio of the GTP plots is its two-tier special case.

Plotting lives downstream in the three-stage figure chain (``build_figure`` ->
``render_figure`` -> ``edit_figure``), which consumes the aggregate written here
and never re-reads a run directory.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

from exp.dispatch_surface.analysis.analytic_cost import VERDICTS, unit_cost
from exp.rit_pareto.rit_k import tier_cost

_MISS_MS = unit_cost("MISS", None)
_TARGET = re.compile(r"_ir(\d+)(?:p(\d+))?$")
_GST_CELL = re.compile(r"_gsth?_f(\d+)w(\d+)v(\d+)$")


def target_of(arm: str) -> float | None:
    m = _TARGET.search(arm)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}") if m.group(2) else float(m.group(1))


def gst_cell_of(arm: str) -> tuple[int, int, int] | None:
    m = _GST_CELL.search(arm)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def arm_label(arm: str) -> str | None:
    """Short annotation: the addressed IR of a RIT arm or the fh/w3/w5 shares of a GST cell."""
    t = target_of(arm)
    if t is not None:
        return f"IR={t:g}"
    c = gst_cell_of(arm)
    return f"{c[0]}/{c[1]}/{c[2]}" if c else None


def decision_cost(hit_type: str, start_t) -> float:
    """Per-decision cost; WARM_START at any canonical tier (K>2 ladders) via ``tier_cost``."""
    if hit_type == "WARM_START":
        return tier_cost(hit_type, start_t)
    return unit_cost(hit_type, start_t)


def aggregate(data_dir: pathlib.Path) -> dict:
    """Per-arm ``{n_ep, success_rate, decisions, counts, ir_percent, target_ir}``."""
    accepted_attempt: dict[str, int] = {}
    episodes: dict[str, dict[str, bool]] = collections.defaultdict(dict)
    with (data_dir / "journal.jsonl").open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("status") not in ("done", "failed") or not row.get("accepted"):
                continue
            accepted_attempt[row["task_uid"]] = row["attempt"]
            episodes[row["yaml_id"]][row["task_uid"]] = row["status"] == "done"

    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    cost: dict[str, float] = collections.defaultdict(float)
    with (data_dir / "per_step.jsonl").open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            hit_type = row.get("hit_type")
            if hit_type is None:
                continue
            if accepted_attempt.get(row["task_uid"]) != row.get("attempt"):
                continue
            if hit_type not in VERDICTS:
                raise SystemExit(f"unpriceable hit_type {hit_type!r} in {data_dir}")
            key = hit_type if hit_type != "WARM_START" else f"WARM_START@{float(row.get('start_t')):g}"
            counts[row["yaml_id"]][key] += 1
            cost[row["yaml_id"]] += decision_cost(hit_type, row.get("start_t"))

    out = {}
    for yaml_id in sorted(episodes):
        eps = episodes[yaml_id]
        n_dec = sum(counts[yaml_id].values())
        c = counts[yaml_id]
        warm_keys = sorted(k for k in c if k.startswith("WARM_START@"))
        out[yaml_id] = {
            "n_ep": len(eps),
            "success_rate": sum(eps.values()) / len(eps),
            "decisions": n_dec,
            "counts": {"FULL_HIT": c.get("FULL_HIT", 0),
                       "WARM_START": sum(c[k] for k in warm_keys),
                       "MISS": c.get("MISS", 0),
                       **{k: c[k] for k in warm_keys}},
            "ir_percent": 100.0 * cost[yaml_id] / (n_dec * _MISS_MS) if n_dec else None,
            "target_ir": target_of(yaml_id),
            "gst_cell": gst_cell_of(yaml_id),
            "label": arm_label(yaml_id),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--data-dir", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    res = aggregate(pathlib.Path(args.data_dir))
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    for arm, r in res.items():
        print(f"{arm}: n={r['n_ep']} SR={r['success_rate']:.3f} IR={r['ir_percent']:.2f}% "
              f"counts={r['counts']}")


if __name__ == "__main__":
    main()
