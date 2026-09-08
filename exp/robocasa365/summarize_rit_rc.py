"""Turn a RIT run's per-cell artifacts into the frontier's two measured axes.

The x axis is the **realized** inference ratio, not the one the arm was
addressed by. The ladder's cuts were fitted on a teacher-driven cohort; once an
arm is deployed the executed action changes, the trajectory diverges and the
score distribution moves with it, so the predicted ratio is an addressing label
and the measured verdict mix is what gets plotted. Both are reported so the gap
between them stays visible.

Reads, per cell, the driver's ``summary_<run_id>.json`` (per-task success) and
``per_step_<run_id>.jsonl`` (one row per inference). Verdicts are priced with
the same ``StageCost`` the arms were emitted against.

Which warm rung fired
---------------------
A row carries ``start_t`` when the runner recorded it. Older rows do not, and
for those the rung is *reconstructed* rather than guessed: the deployed rule is
a deterministic walk down the ladder's cuts, so a WARM_START row's rung is the
first warm cut its ``cp1_score`` clears. That is exact, not an approximation --
the same comparison the judge made.

Usage:
  uv run python -m exp.robocasa365.summarize_rit_rc \\
      --data-dir exp/robocasa365/data/rit/groot_tp --run-prefix rit \\
      --record <rit_record.json> --cost <rc_cost.json> --teacher groot_tp \\
      --out <frontier.json>
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict

from exp.robocasa365 import rit_cost_rc as rc
from exp.robocasa365.emit_rit_rc import WARM_TS, schedule_of

HEADER_STEP_IDX = -1


def _cost(path: str, teacher: str) -> rc.StageCost:
    d = json.loads(pathlib.Path(path).read_text())
    return rc.StageCost(
        teacher=teacher, schedule=schedule_of(teacher), stage1_ms=d["stage1_ms"],
        stage2_ms=d["stage2_ms"], stage3_head_ms=d["stage3_head_ms"],
        stage3_step_ms=d["stage3_step_ms"], provenance=d["provenance"],
        linear_stage3=d.get("linear_stage3", False),
    )


def task_of(task_uid: str) -> str:
    """``<run_id>__<Task>:eval:<ordinal>:<episode>`` -> ``<Task>``.

    Split on the LAST separator: the run id itself carries ``__`` (the arm's
    cid and the library tag are joined with it), so cutting at the first one
    returns a fragment of the run id rather than the task.
    """
    return task_uid.rsplit("__", 1)[1].split(":", 1)[0]


def rung_name(start_t: float) -> str:
    return f"warm{int(round(float(start_t) * 100)):02d}"


def classify(row: dict, warm_cuts: list[tuple[float, float]]) -> str | None:
    """Ladder rung of one per-step row, or ``None`` for a non-verdict row.

    ``warm_cuts`` is ``(threshold, start_t)`` in the judge's own order, so the
    reconstruction walks it exactly as the judge did.
    """
    hit = (row.get("hit_type") or "").upper()
    if not hit or row.get("step_idx") == HEADER_STEP_IDX:
        return None
    if hit == "FULL_HIT":
        return "full"
    if hit == "MISS":
        return "miss"
    if hit != "WARM_START":
        raise ValueError(f"unknown hit_type {hit!r}")
    t = row.get("start_t")
    if t is not None:
        return rung_name(t)
    score = row.get("cp1_score")
    if score is None:
        raise ValueError("a warm verdict with neither start_t nor cp1_score cannot be priced")
    for threshold, start_t in warm_cuts:
        if float(score) >= threshold:
            return rung_name(start_t)
    raise ValueError(f"warm verdict at score {score} clears no warm cut of this arm")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run-prefix", default="rit")
    ap.add_argument("--record", required=True, help="the emitter's rit_record json")
    ap.add_argument("--config-dir", default="", help="arm yamls (default: alongside the record)")
    ap.add_argument("--cost", required=True)
    ap.add_argument("--teacher", required=True, choices=("groot_tp", "pi05"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = pathlib.Path(args.data_dir)
    cost = _cost(args.cost, args.teacher)
    warm_ts = [round(t, 4) for t in WARM_TS[args.teacher]]
    record = json.loads(pathlib.Path(args.record).read_text())
    index = record["arms"]

    points = {}
    for cid, meta in sorted(index.items()):
        if "file" not in meta:
            continue
        k = int(meta.get("k", 1))
        tiers = rc.ladder(cost, warm_ts[: max(0, k - 1)])
        # The judge's own walk order: cheapest rung first, which is also the
        # highest cut. Rungs the emitter dropped are absent from the arm and
        # must not be offered to the reconstruction.
        dropped = set(meta.get("dropped_rungs") or [])
        warm_cuts = [
            (float(meta["thetas"][t.name]), float(t.start_t))
            for t in tiers
            if t.hit_type == "WARM_START" and t.name not in dropped
        ]
        warm_cuts.sort(key=lambda x: -x[0])

        summaries = sorted(data.glob(f"summary_{args.run_prefix}-{cid}__*.json"))
        steps = sorted(data.glob(f"per_step_{args.run_prefix}-{cid}__*.jsonl"))
        if not summaries:
            points[cid] = {"missing": "summary"}
            continue
        tasks: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        n_err = n_missing = 0
        for path in summaries:
            blob = json.loads(path.read_text())
            n_err += blob.get("n_err") or 0
            n_missing += blob.get("n_missing") or 0
            for task, v in blob["tasks"].items():
                tasks[task][0] += v["succ"]
                tasks[task][1] += v["n_scored"]

        overall: Counter = Counter()
        by_task: dict[str, Counter] = defaultdict(Counter)
        for path in steps:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rung = classify(row, warm_cuts)
                    if rung is None:
                        continue
                    overall[rung] += 1
                    by_task[task_of(row["task_uid"])][rung] += 1

        srs = [s / n for s, n in tasks.values() if n]
        points[cid] = {
            "k": k,
            "target_ir": meta.get("target_ir"),
            "predicted_ir": meta.get("predicted_ir"),
            "realized_ir": rc.realized_ir(dict(overall), tiers, cost) if overall else None,
            "macro_sr": sum(srs) / len(srs) if srs else None,
            "verdicts": dict(overall),
            "n_verdict_rows": sum(overall.values()),
            "n_err": n_err,
            "n_missing": n_missing,
            "dropped_rungs": sorted(dropped),
            "per_task": {
                task: {
                    "succ": tasks[task][0], "n": tasks[task][1],
                    "sr": tasks[task][0] / tasks[task][1] if tasks[task][1] else None,
                    "realized_ir": (rc.realized_ir(dict(by_task[task]), tiers, cost)
                                    if by_task.get(task) else None),
                    "verdicts": dict(by_task.get(task, {})),
                }
                for task in sorted(tasks)
            },
        }

    out = {
        "teacher": args.teacher,
        "cost": {"miss_ms": rc.miss_cost(cost), "provenance": cost.provenance,
                 "tier_ir_share": {t.name: 100 * t.cost_ms / rc.miss_cost(cost)
                                   for t in rc.ladder(cost, warm_ts)}},
        "warm_ts": warm_ts,
        "gate_theta": record["gate"]["theta"],
        "points": points,
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print("%-16s %2s %7s %7s %8s %9s %6s %6s" %
          ("arm", "k", "target", "pred", "realized", "macroSR", "err", "miss"))
    for cid, pt in sorted(points.items()):
        if "missing" in pt:
            print("%-16s MISSING %s" % (cid, pt["missing"]))
            continue
        print("%-16s %2d %7.2f %7.2f %8.2f %9.4f %6d %6d"
              % (cid, pt["k"], pt["target_ir"] or float("nan"), pt["predicted_ir"],
                 pt["realized_ir"] or float("nan"), pt["macro_sr"] or float("nan"),
                 pt["n_err"], pt["n_missing"]))


if __name__ == "__main__":
    main()
