"""Emit RIT arms at score cuts given directly, instead of addressing them by IR.

``emit_rit_rc.py arms`` inverts an inference-ratio target through the risk fit,
which assumes the calibration score distribution predicts the deployed one. For
pi0.5 it does not: the six IR-addressed cuts partitioned the deployed
distribution 100/98.6/92.3/84.3/83.9/2.6 percent, so five of them cut in a
region deployment never visits and the frontier collapsed onto two points.

When the addressing assumption fails, the figure does not actually need exact
IR targets -- it needs COVERAGE. Cuts chosen on the measured deployed
distribution give coverage by construction, and where each one lands is then
simply measured. That removes the prediction step entirely.

The ladder shape, judge, gate and provenance are the emitter's, unchanged: only
the source of the cuts differs.

Usage:
  uv run python -m exp.robocasa365.emit_cuts_rc --teacher pi05 \\
      --calibration <normalizers.json> --cuts <cuts.json> --cost <cost.json> \\
      --out-dir <dir> --record <record.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from exp.dispatch_surface.emit_precheck_yamls import LAYER_SECONDARY, gate_section
from exp.gate_threshold_pareto.solve_gtp import THETA_TOP_FRACTION
from exp.robocasa365 import emit_ws_search2_yamls as ws2
from exp.robocasa365 import rit_cost_rc as rc
from exp.robocasa365.emit_rit_rc import (
    DEFAULT_LIBRARY_DIR,
    FROZEN_WEIGHT_CID,
    WARM_TS,
    build_arm_cell,
    build_calibration_cell,
    schedule_of,
    w13_spec,
)
from exp.verdict_factor_judge.phase3.threshold_solver import derive_thresholds


def _cost(path: str, teacher: str) -> rc.StageCost:
    d = json.loads(Path(path).read_text())
    return rc.StageCost(
        teacher=teacher, schedule=schedule_of(teacher), stage1_ms=d["stage1_ms"],
        stage2_ms=d["stage2_ms"], stage3_head_ms=d["stage3_head_ms"],
        stage3_step_ms=d["stage3_step_ms"], provenance=d["provenance"],
        linear_stage3=d.get("linear_stage3", False),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True, choices=("groot_tp", "pi05"))
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--library-dir", default=DEFAULT_LIBRARY_DIR)
    ap.add_argument("--weight-cid", default="")
    ap.add_argument("--cuts", required=True, help="json with a 'cuts' list of FULL_HIT thresholds")
    ap.add_argument("--gate-scores", default="", help="json list of calibration scores for the gate theta")
    ap.add_argument("--gate-theta", type=float, default=None)
    ap.add_argument("--cost", required=True)
    ap.add_argument("--ks", default="1")
    ap.add_argument("--warm-offsets", default="0.5",
                    help="each warm rung's cut as a fraction of the gap below the FULL cut")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--record", required=True)
    args = ap.parse_args()

    calib = json.loads(Path(args.calibration).read_text())
    spec = w13_spec(args.teacher, args.library_dir)
    weight_cid = args.weight_cid or FROZEN_WEIGHT_CID[args.teacher]
    cost = _cost(args.cost, args.teacher)
    blob = json.loads(Path(args.cuts).read_text())
    cuts = [float(c) for c in blob["cuts"]]
    warm_ts = [round(t, 4) for t in WARM_TS[args.teacher]]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    offs = [float(x) for x in args.warm_offsets.split(",") if x.strip()]

    if args.gate_theta is not None:
        gate_theta = float(args.gate_theta)
    else:
        scores = json.loads(Path(args.gate_scores).read_text())
        gate_theta = float(derive_thresholds(scores, THETA_TOP_FRACTION, 0.0)[0])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {}
    for k in ks:
        tiers = rc.ladder(cost, warm_ts[: k - 1])
        warm_names = [t.name for t in tiers if t.hit_type == "WARM_START"]
        for cut in cuts:
            thetas = {"full": cut}
            # Warm rungs sit below the FULL cut, spaced into the score mass that
            # remains under it. Straight-line placement keeps the nesting the
            # loader requires without pretending a risk fit produced it.
            for i, name in enumerate(warm_names):
                thetas[name] = cut * (1.0 - offs[min(i, len(offs) - 1)] * (i + 1))
            cfg, dropped = build_arm_cell(
                args.teacher, calib, spec, weight_cid, tiers=tiers,
                thetas=thetas, gate_theta=gate_theta, schedule=schedule_of(args.teacher),
            )
            cid = f"c{k}__t{cut:.5f}".replace(".", "p")
            path = out_dir / f"{cid}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            ws2.validate_on_disk(path)
            index[cid] = {"file": path.name, "k": k, "cut": cut, "thetas": thetas,
                          "gate_theta": gate_theta, "dropped_rungs": dropped}

    cfg = build_calibration_cell(args.teacher, calib, spec, weight_cid)
    cfg["checkpoints"]["cp1"]["gate"] = gate_section(LAYER_SECONDARY, float(gate_theta))
    path = out_dir / "always_hit.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    ws2.validate_on_disk(path)
    index["always_hit"] = {"file": path.name, "k": 1, "gate_theta": gate_theta,
                           "predicted_ir": 100.0 * cost.stage1_ms / rc.miss_cost(cost)}

    (out_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    # The pinned pick lane's identity gate reads this file; without it the
    # driver dies before graph construction and the lane produces nothing.
    (out_dir / "provenance.json").write_text(json.dumps(
        {"teacher": args.teacher, "weight_cid": weight_cid,
         "library": spec.get("preload"), "gate_theta": gate_theta,
         "cuts_source": blob.get("source"),
         "cells": {cid: {"file": m["file"]} for cid, m in index.items()}},
        indent=1, sort_keys=True))
    Path(args.record).write_text(json.dumps(
        {"teacher": args.teacher, "weight_cid": weight_cid, "gate": {"theta": gate_theta},
         "cuts_source": blob.get("source"), "gate_skip": blob.get("gate_skip"),
         "reachable_ir": blob.get("reachable"), "targets_ir": blob.get("targets"),
         "arms": index}, indent=1, sort_keys=True))
    print(json.dumps({"arms": len(index), "ks": ks, "cuts": len(cuts),
                      "gate_theta": round(gate_theta, 6)}, indent=1))


if __name__ == "__main__":
    main()
