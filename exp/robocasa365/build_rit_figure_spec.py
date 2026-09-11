"""Turn a measured RIT frontier into an editable figure spec.

The spec is the figure's source of truth: the editor reads it, writes it back,
and any renderer draws from it. Only the thirteen per-task coordinates are
stored. The overall panel is *derived* from them and never stored, so a spec
cannot hold an overall point that disagrees with the tasks under it:

    y_overall = mean over tasks of y_task                (the macro definition)
    x_overall = sum(w_task * x_task) / sum(w_task)       (w = verdict rows)

The weighted form is exact rather than a convenience. A per-task inference
ratio is ``100 * sum(count*cost) / (rows * MISS)``, so pooling the numerators
and denominators over tasks is the same as averaging the per-task ratios
weighted by their row counts.

Usage:
  uv run python -m exp.robocasa365.build_rit_figure_spec \\
      --frontier <frontier_all13.json> --teacher <teacher_ref.json> \\
      --figure-id rit_groot_all13 --title "..." --out-dir exp/robocasa365/analysis/figures
"""

from __future__ import annotations

import argparse
import json
import pathlib

SCHEMA = "robocasa365.rit_figure/v1"

#: Two ladder depths -> two categorical hues, assigned in fixed order.
SERIES_STYLE = {
    2: {"color": "#2E6FD9", "label": "k=2  FULL + WARM@1 step"},
    3: {"color": "#C8641E", "label": "k=3  FULL + WARM@1 + WARM@2 steps"},
}
ANCHOR_CID = "always_hit"


def per_task_block(pt: dict, tasks: list[str]) -> dict:
    """Per-task coordinate plus the row count the overall panel pools by."""
    out = {}
    for task in tasks:
        sub = pt["per_task"].get(task)
        if not sub or sub.get("sr") is None or sub.get("realized_ir") is None:
            continue
        out[task] = {
            "x": round(float(sub["realized_ir"]), 4),
            "y": round(float(sub["sr"]), 6),
            "w": int(sum(sub.get("verdicts", {}).values())),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frontier", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--figure-id", required=True)
    ap.add_argument("--teacher-name", required=True, choices=("groot_tp", "pi05"),
                    help="which teacher this spec belongs to")
    ap.add_argument("--title", required=True)
    ap.add_argument("--episodes-per-task", type=int, default=50)
    ap.add_argument("--out-dir", default="exp/robocasa365/analysis/figures")
    args = ap.parse_args()

    fr = json.loads(pathlib.Path(args.frontier).read_text())
    tref = json.loads(pathlib.Path(args.teacher).read_text())
    points = fr["points"]
    tasks = sorted(tref["tasks"])

    series = []
    for k, style in SERIES_STYLE.items():
        pts = []
        for cid, pt in points.items():
            if "missing" in pt or pt.get("k") != k:
                continue
            target = pt.get("target_ir")
            pts.append({
                "id": cid,
                "label": f"IR={target:.0f}" if target is not None else cid,
                "per_task": per_task_block(pt, tasks),
            })
        pts.sort(key=lambda p: p["id"])
        series.append({"key": style["label"], "color": style["color"], "points": pts})

    anchor = points.get(ANCHOR_CID)
    anchors = []
    if anchor and "missing" not in anchor:
        anchors.append({
            "id": ANCHOR_CID,
            "label": "all-FULL_HIT",
            "per_task": per_task_block(anchor, tasks),
        })

    spec = {
        "schema": SCHEMA,
        "figure_id": args.figure_id,
        "teacher": args.teacher_name,
        "title": args.title,
        "x_label": "realized inference ratio  (% of all-MISS)",
        "y_label": "success rate",
        "episodes_per_task": args.episodes_per_task,
        "tasks": tasks,
        "series": series,
        "anchors": anchors,
        "teacher_only": {
            "label": "teacher-only (no cache)",
            "per_task": {t: round(float(v["sr"]), 6) for t, v in tref["tasks"].items()},
        },
    }

    out = pathlib.Path(args.out_dir) / f"{args.figure_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n")
    n = sum(len(s["points"]) for s in series)
    print(f"wrote {out}  ({len(tasks)} tasks, {n} arm points, {len(anchors)} anchor)")


if __name__ == "__main__":
    main()
