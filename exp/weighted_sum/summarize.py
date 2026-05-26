"""Aggregate the conductor journal into per-yaml success rates for Phase-2 plots.

``ConductorDriver`` records one terminal line per episode to its journal
(``{task_uid, yaml_id, phase, status, success, ts}``). This collapses the eval
journal into ``{yaml_id: {success_rate, n, n_success}}`` — the JSON consumed by
``analysis/plot_phase2_results.py``.

Usage:
    uv run exp/weighted_sum/summarize.py \
        --journal exp/weighted_sum/data/phase2/journal.jsonl \
        --out exp/weighted_sum/data/phase2/results.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize_journal(journal_path: str | Path) -> dict[str, dict]:
    """Collapse eval/done journal records into per-yaml success rates."""
    done: dict[str, list[bool]] = defaultdict(list)
    for raw in Path(journal_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("phase") != "eval" or rec.get("status") != "done":
            continue
        done[rec["yaml_id"]].append(bool(rec.get("success")))
    out: dict[str, dict] = {}
    for yaml_id, successes in done.items():
        n = len(successes)
        n_success = sum(successes)
        out[yaml_id] = {
            "success_rate": (n_success / n) if n else 0.0,
            "n": n,
            "n_success": n_success,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Summarize Phase-2 journal -> per-yaml SR")
    ap.add_argument("--journal", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = summarize_journal(args.journal)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {len(res)} yaml summaries to {out}")


if __name__ == "__main__":
    main()
