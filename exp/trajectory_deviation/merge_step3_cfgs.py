"""Aggregate Step 3 per-cycle policy JSONL outputs into a cross-cfg CSV.

Input: one ``results.jsonl`` per cache config (emitted by
``run_step3_per_cycle_policy.py``). Each line is a dict with at least
``{cfg, ep, tau, n, success, total_cycles, num_inference_cycles,
inference_ratio, T_gt_cycles, burst_count}``.

Output: a single CSV grouped by ``(cfg, tau, n)`` with one row per group:
``cfg, tau, n, episodes, success_rate, mean_inference_ratio,
std_inference_ratio``. ``episodes`` is the count of unique ``ep`` values
in the group (so resume-triggered duplicate rows collapse cleanly).

Reference CLI:

    uv run exp/trajectory_deviation/merge_step3_cfgs.py \\
        --jsonl data/deviation_experiment/step3/clip_w7_d4/results.jsonl \\
        --jsonl data/deviation_experiment/step3/spatial16_w8_d4/results.jsonl \\
        --jsonl data/deviation_experiment/step3/max_pool_w3_d5/results.jsonl \\
        --out data/deviation_experiment/step3/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


_GroupKey = Tuple[str, int, int]  # (cfg, tau, n)


def _load_jsonl_records(paths: Sequence[Path]) -> List[dict]:
    """Read all JSONL files, deduplicating by ``(cfg, ep, tau, n)``.

    Duplicate keys can arise from BaseRunState retry passes that reran a
    failing unit and re-appended the later result. The last occurrence wins.
    """
    latest: Dict[Tuple[str, str, int, int], dict] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"JSONL not found: {path}")
        with path.open() as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"{path}:{line_no} malformed JSON: {e}"
                    ) from e
                try:
                    key = (
                        str(rec["cfg"]),
                        str(rec["ep"]),
                        int(rec["tau"]),
                        int(rec["n"]),
                    )
                except KeyError as e:
                    raise ValueError(
                        f"{path}:{line_no} missing required key {e}"
                    ) from e
                latest[key] = rec
    return list(latest.values())


def aggregate(records: Sequence[dict]) -> List[dict]:
    """Group ``records`` by ``(cfg, tau, n)`` and return summary rows."""
    groups: Dict[_GroupKey, List[dict]] = defaultdict(list)
    for rec in records:
        key: _GroupKey = (str(rec["cfg"]), int(rec["tau"]), int(rec["n"]))
        groups[key].append(rec)

    rows: List[dict] = []
    for (cfg, tau, n), group_recs in sorted(groups.items()):
        successes = sum(1 for r in group_recs if bool(r.get("success", False)))
        ratios = np.asarray(
            [float(r.get("inference_ratio", 0.0)) for r in group_recs],
            dtype=np.float64,
        )
        rows.append(
            {
                "cfg": cfg,
                "tau": tau,
                "n": n,
                "episodes": len(group_recs),
                "success_rate": float(successes) / float(len(group_recs)),
                "mean_inference_ratio": float(ratios.mean()) if ratios.size else 0.0,
                "std_inference_ratio": float(ratios.std(ddof=0)) if ratios.size else 0.0,
            }
        )
    return rows


def write_csv(rows: Sequence[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cfg", "tau", "n", "episodes",
        "success_rate", "mean_inference_ratio", "std_inference_ratio",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Step 3 JSONL into CSV.")
    parser.add_argument(
        "--jsonl", action="append", type=Path, required=True,
        help="Path to a results.jsonl (repeat for each cfg).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    records = _load_jsonl_records(args.jsonl)
    if not records:
        raise SystemExit("No records found across input JSONL files.")
    rows = aggregate(records)
    write_csv(rows, args.out)
    logger.info(
        "Wrote %d (cfg, tau, n) rows to %s from %d unique units.",
        len(rows), args.out, len(records),
    )


if __name__ == "__main__":
    main()
