"""Turn a RIT run's per-cell artifacts into the frontier's two measured axes.

The x axis is the **realized** inference ratio, not the one the arm was
addressed by. The ladder's cuts were fitted on a teacher-driven cohort; once
an arm is deployed the executed action changes, the trajectory diverges and
the score distribution moves with it, so the predicted ratio is an addressing
label and the measured verdict mix is the number to plot. Both are reported so
the gap between them is visible rather than assumed away.

Reads, per cell, the driver's ``summary_<run_id>.json`` (per-task success) and
``per_step_<run_id>.jsonl`` (one ``__hit_meta__`` row per inference, carrying
``hit_type`` and, for a warm verdict, the ``start_t`` that says which rung
fired). Verdicts are priced with the same ``StageCost`` the arms were emitted
against.

Usage:
  uv run python -m exp.robocasa365.summarize_rit_rc \\
      --data-dir exp/robocasa365/data/rit/groot_tp --run-prefix rit \\
      --arms-index .../index.json --cost .../rc_cost_groot_tp.json \\
      --teacher groot_tp --out .../frontier_groot_tp.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict

from exp.robocasa365 import rit_cost_rc as rc
from exp.robocasa365.emit_rit_rc import WARM_TS, schedule_of


def _cost(path: str, teacher: str) -> rc.StageCost:
    d = json.loads(pathlib.Path(path).read_text())
    return rc.StageCost(
        teacher=teacher, schedule=schedule_of(teacher), stage1_ms=d["stage1_ms"],
        stage2_ms=d["stage2_ms"], stage3_head_ms=d["stage3_head_ms"],
        stage3_step_ms=d["stage3_step_ms"], provenance=d["provenance"],
        linear_stage3=d.get("linear_stage3", False),
    )


def verdict_counts(per_step: pathlib.Path, warm_ts: list[float]) -> dict:
    """Tally verdicts by rung, keyed the way ``realized_ir`` prices them.

    A warm row is attributed by its ``start_t``: two rungs of the same ladder
    are different costs, so folding them into one WARM bucket would price the
    arm at neither. A row whose ``start_t`` is not a rung of this ladder is
    counted separately and fails the caller rather than being absorbed.
    """
    by_name: Counter = Counter()
    stray: Counter = Counter()
    n_rows = 0
    with per_step.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            hit = (row.get("hit_type") or "").upper()
            if not hit:
                continue  # header / provenance rows carry no verdict
            n_rows += 1
            if hit == "FULL_HIT":
                by_name["full"] += 1
            elif hit == "MISS":
                by_name["miss"] += 1
            elif hit == "WARM_START":
                t = row.get("start_t")
                t = None if t is None else round(float(t), 4)
                if t in warm_ts:
                    by_name[f"warm{int(round(t * 100)):02d}"] += 1
                else:
                    stray[str(t)] += 1
            else:
                stray[hit] += 1
    return {"counts": dict(by_name), "stray": dict(stray), "n_rows": n_rows}


def task_success(summary: pathlib.Path) -> dict:
    d = json.loads(summary.read_text())
    return {
        "macro_sr": d["macro_sr"],
        "complete": d.get("complete"),
        "n_err": d.get("n_err"),
        "n_missing": d.get("n_missing"),
        "tasks": {t: [v["succ"], v["n_scored"]] for t, v in d["tasks"].items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run-prefix", default="rit")
    ap.add_argument("--arms-index", required=True)
    ap.add_argument("--cost", required=True)
    ap.add_argument("--teacher", required=True, choices=("groot_tp", "pi05"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = pathlib.Path(args.data_dir)
    cost = _cost(args.cost, args.teacher)
    warm_ts = [round(t, 4) for t in WARM_TS[args.teacher]]
    index = json.loads(pathlib.Path(args.arms_index).read_text())

    points = {}
    for cid, meta in sorted(index.items()):
        if "file" not in meta:
            continue
        summaries = sorted(data.glob(f"summary_{args.run_prefix}-{cid}__*.json"))
        steps = sorted(data.glob(f"per_step_{args.run_prefix}-{cid}__*.jsonl"))
        if not summaries:
            points[cid] = {"missing": "summary"}
            continue
        merged_tasks: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        n_err = n_missing = 0
        for path in summaries:
            blob = task_success(path)
            n_err += blob["n_err"] or 0
            n_missing += blob["n_missing"] or 0
            for task, (succ, n) in blob["tasks"].items():
                merged_tasks[task][0] += succ
                merged_tasks[task][1] += n
        # Macro over tasks, not pooled: the tasks carry very different episode
        # difficulty and a pooled rate would be dominated by whichever ones the
        # cache happens to answer.
        srs = [s / n for s, n in merged_tasks.values() if n]
        macro = sum(srs) / len(srs) if srs else float("nan")

        tally = {"counts": Counter(), "stray": Counter(), "n_rows": 0}
        for path in steps:
            blob = verdict_counts(path, warm_ts)
            tally["counts"].update(blob["counts"])
            tally["stray"].update(blob["stray"])
            tally["n_rows"] += blob["n_rows"]
        counts = dict(tally["counts"])
        ladder_names = {"full", "miss"} | {
            f"warm{int(round(t * 100)):02d}" for t in warm_ts
        }
        unknown = sorted(set(counts) - ladder_names)
        if unknown or tally["stray"]:
            raise SystemExit(
                f"{cid}: verdicts outside the ladder: {unknown} stray={dict(tally['stray'])}"
            )
        tiers = rc.ladder(cost, warm_ts[: max(0, int(meta.get("k", 1)) - 1)])
        ir = rc.realized_ir(counts, tiers, cost) if counts else float("nan")
        points[cid] = {
            "k": meta.get("k"),
            "target_ir": meta.get("target_ir"),
            "predicted_ir": meta.get("predicted_ir"),
            "realized_ir": ir,
            "macro_sr": macro,
            "tasks": {t: v for t, v in sorted(merged_tasks.items())},
            "verdicts": counts,
            "n_verdict_rows": tally["n_rows"],
            "n_err": n_err,
            "n_missing": n_missing,
        }
    out = {
        "teacher": args.teacher,
        "cost": {"miss_ms": rc.miss_cost(cost), "provenance": cost.provenance},
        "warm_ts": warm_ts,
        "points": points,
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    for cid, pt in sorted(points.items()):
        if "missing" in pt:
            print(f"{cid:22s} MISSING {pt['missing']}")
            continue
        gap = (pt["realized_ir"] - (pt["predicted_ir"] or float("nan")))
        print(
            f"{cid:22s} k={pt['k']} target={pt['target_ir'] or float('nan'):6.2f} "
            f"predicted={pt['predicted_ir']:6.2f} realized={pt['realized_ir']:6.2f} "
            f"({gap:+5.2f}) macroSR={pt['macro_sr']:.4f} "
            f"err={pt['n_err']} missing={pt['n_missing']}"
        )


if __name__ == "__main__":
    main()
