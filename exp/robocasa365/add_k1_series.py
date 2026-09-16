"""Fold the k=1 control family into an existing figure spec.

The k=1 sweep ran months after the k=2/k=3 spec was first written and hand
adjusted, so this cannot go through ``build_rit_figure_spec``: regenerating
would silently discard every coordinate the owner moved. This reads the spec
that is already on disk, leaves its series untouched, and appends k=1 as one
more series.

Appending rather than prepending matters. ``plot_rit_pareto_four`` picks a
dash pattern by series index, so putting k=1 first would re-dash k=2 and k=3
and the combined figure would stop matching the one already published.

Usage:
  uv run python -m exp.robocasa365.add_k1_series \\
      --frontier <frontier_k1_all13.json> \\
      --spec exp/robocasa365/analysis/figures/rit_groot_all13.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

#: Third categorical hue, assigned in fixed order after the two already in the
#: spec (#2E6FD9 / #C8641E). Validated against both existing hues: CVD dE 26.7
#: protan / 20.3 tritan, normal-vision dE 27.6, all six checks pass.
K1_COLOR = "#7A4FBF"
K1_KEY = "k=1  FULL only (no warm rung)"
ANCHOR_CID = "always_hit"


def per_task_block(pt: dict, tasks: list[str]) -> dict:
    """Per-task coordinate plus the row count the overall panel pools by.

    Same shape as ``build_rit_figure_spec.per_task_block`` -- the overall panel
    is derived from these, so a k=1 block that carried different fields would
    read as a missing task rather than as an error.
    """
    out = {}
    for task in tasks:
        sub = pt.get("per_task", {}).get(task)
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
    ap.add_argument("--frontier", required=True, help="fused k=1 frontier json")
    ap.add_argument("--spec", required=True, help="figure spec to fold it into, edited in place")
    args = ap.parse_args()

    spec_path = pathlib.Path(args.spec)
    spec = json.loads(spec_path.read_text())
    fr = json.loads(pathlib.Path(args.frontier).read_text())
    tasks = spec["tasks"]

    pts = []
    for cid, pt in fr["points"].items():
        # The all-FULL_HIT arm is tagged k=1 in the frontier because it spends
        # exactly one tier, but the spec already carries it as an anchor drawn
        # as a diamond. Letting it through would put the same measurement on
        # the plot twice, once as a diamond and once as a k=1 marker.
        if cid == ANCHOR_CID or "missing" in pt or pt.get("k") != 1:
            continue
        block = per_task_block(pt, tasks)
        if not block:
            continue
        target = pt.get("target_ir")
        pts.append({
            "id": cid,
            "label": f"IR={target:.0f}" if target is not None else cid,
            "per_task": block,
        })
    if not pts:
        raise SystemExit(f"{args.frontier} carries no k=1 points")
    pts.sort(key=lambda p: min(v["x"] for v in p["per_task"].values()))

    # Replacing rather than duplicating keeps the tool idempotent, but any hand
    # adjustment made to a previous k=1 series is lost -- say so rather than
    # let it vanish quietly.
    existing = [i for i, s in enumerate(spec["series"]) if s["key"] == K1_KEY]
    for i in reversed(existing):
        print(f"replacing the k=1 series already in {spec_path.name} "
              "(any hand-moved k=1 points in it are discarded)")
        spec["series"].pop(i)

    spec["series"].append({"key": K1_KEY, "color": K1_COLOR, "points": pts})
    spec_path.write_text(json.dumps(spec, indent=1))
    print(f"wrote {spec_path}  ({len(pts)} k=1 points, "
          f"{len(spec['series'])} series total: {[s['key'].split()[0] for s in spec['series']]})")


if __name__ == "__main__":
    main()
