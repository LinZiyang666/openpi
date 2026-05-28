"""Aggregate the eval per-step JSONL into per-yaml inference_ratio.

inference_ratio per the verdict convention:
    FULL_HIT -> 0 ; WARM_START@t -> 1 - 0.5*(1-t) ; MISS -> 1
    inf_ratio = (0*n_FH + cost_WS*n_WS + 1*n_MISS) / N

The per-step rows come from ``run_phase2 --per-step-out`` (driver.per_step_rows),
each carrying ``yaml_id``, ``hit_type`` (FULL_HIT / WARM_START / MISS), and
``start_t`` (for the warm cost). Pairs with ``summarize.py`` (per-yaml SR) to
place each yaml on the (inference_ratio, success_rate) Pareto plane.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# Normalize the HitType enum repr / wire string to a canonical tag.
_FH = {"FULL_HIT", "HitType.FULL_HIT", "full_hit"}
_WS = {"WARM_START", "HitType.WARM_START", "warm_start"}
_MISS = {"MISS", "HitType.MISS", "miss"}


def _warm_cost(start_t: float | None) -> float:
    """WARM_START cost = 1 - 0.5*(1-t); default t=0.5 -> 0.75."""
    t = 0.5 if start_t is None else float(start_t)
    return 1.0 - 0.5 * (1.0 - t)


def summarize_inf_ratio(per_step_jsonl: str | Path) -> dict[str, dict]:
    """Collapse per-step hit_type rows into per-yaml inference_ratio."""
    agg: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {"n": 0, "n_fh": 0, "n_ws": 0, "n_miss": 0, "cost": 0.0}
    )
    for raw in Path(per_step_jsonl).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rec = json.loads(line)
        yid = rec.get("yaml_id")
        ht = str(rec.get("hit_type"))
        a = agg[yid]
        a["n"] += 1
        if ht in _FH:
            a["n_fh"] += 1
        elif ht in _WS:
            a["n_ws"] += 1
            a["cost"] += _warm_cost(rec.get("start_t"))
        elif ht in _MISS:
            a["n_miss"] += 1
            a["cost"] += 1.0
        else:  # unknown -> treat as MISS (conservative, full inference)
            a["n_miss"] += 1
            a["cost"] += 1.0
    out: dict[str, dict] = {}
    for yid, a in agg.items():
        n = a["n"]
        out[yid] = {
            "inference_ratio": (a["cost"] / n) if n else float("nan"),
            "n_steps": n,
            "n_full_hit": a["n_fh"],
            "n_warm_start": a["n_ws"],
            "n_miss": a["n_miss"],
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Summarize eval per-step JSONL -> per-yaml inference_ratio")
    ap.add_argument("--per-step", required=True, help="per-step JSONL from run_phase2 --per-step-out")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = summarize_inf_ratio(args.per_step)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote inference_ratio for {len(res)} yamls to {out}")


if __name__ == "__main__":
    main()
