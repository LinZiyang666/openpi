"""Emit the LIBERO x GR00T arm set: two anchors, the GST grid, the RIT ladder.

All three rules deploy through the production ``threshold`` judge, which is the
only three-way verdict on GR00T's serviceable-judge whitelist
(``dispatch_surface`` is not). They differ only in where the cuts come from:

  anchor  two constants outside the [0, 1] score domain. ``-1`` admits every
          step at FULL_HIT (pure cache, the cheap end of the axis); ``2``
          admits none (pure teacher, the all-MISS end). Expressing them as
          thresholds rather than as ``always_hit`` / no-cache keeps them inside
          the same validator and the same runner as the sweep arms, so the
          anchors and the frontier cannot diverge in how they were produced.
  gst     percent triples on the step-20 simplex, cut as descending score
          quantiles of the warmup scores (the GTP / tgrid convention).
  rit     ``rit_cost_rc`` fitted on the warmup rows: one tolerance per
          addressed inference ratio, inverted into one cut per rung.

The anchors carry ``always_search``; the sweep arms carry the hysteresis gate
at the 0.85 score quantile with j=3 / probe_interval=3 / L=6. The anchors are
deliberately ungated: the gate's lockout forces a teacher call every L steps,
which would make "pure cache" not pure and move the very end point the
frontier is measured against. ``run_gtp`` validates one gate per matrix, so the
two families are emitted as two matrices.

The RIT rule needs a stage-cost ledger and refuses to guess one: the whole
point of IR addressing is that the x axis is a measured cost, so a borrowed
constant would silently relabel every arm. ``--rules`` therefore lets the
cost-free families be emitted before the ledger exists.

Usage:
  uv run python -m exp.libero_groot.emit_rit_arms \
      --suite libero_spatial --rules anchors,gst \
      --shadow exp/libero_groot/data/rit/shadow/libero_spatial/shadow_rows.jsonl \
      --out-dir exp/libero_groot/config/rit
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import pathlib

import numpy as np
import yaml

from openpi.cache.config import load_cache_config
from openpi.cache.types import groot_n15_schedule

from exp.dispatch_surface.emit_precheck_yamls import (
    LAYER_PRIMARY,
    LAYER_SECONDARY,
    gate_section,
)
from exp.gate_threshold_pareto.solve_gtp import THETA_TOP_FRACTION
from exp.robocasa365 import rit_cost_rc as rc
from exp.robocasa365.emit_rit_rc import fit_ladders
from exp.verdict_factor_judge.phase3.threshold_solver import derive_thresholds

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOCOL = "libero_groot_rit_arms_v1"
SUITE_TAG = {"libero_spatial": "sp", "libero_10": "l10"}
RULES = ("anchors", "gst", "rit")
DENOISING_STEPS = 8
#: Ladder rungs, cheapest first: t=0.75 leaves 2 of 8 steps, t=0.5 leaves 4.
WARM_TS = (0.75, 0.5)
DEFAULT_KS = (1, 2, 3)
#: The Pi0.5 LIBERO line's grid, matched so the two teachers' frontiers are
#: addressed at the same budgets.
DEFAULT_TARGETS = tuple(float(x) for x in range(20, 100, 5))
GST_STEP = 20
GST_MAX_SUM = 80
DEFAULT_ALPHA = 0.05
#: Outside the normalizer's [0, 1] range in both directions, so the verdict is
#: constant by construction rather than by where the score distribution happens
#: to sit.
CUT_ALL = -1.0
CUT_NONE = 2.0


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: pathlib.Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path}: no shadow rows")
    return rows


def scores_of(rows: list[dict]) -> np.ndarray:
    s = np.array(
        [float(r["s"]) for r in rows if r.get("s") is not None and math.isfinite(float(r["s"]))],
        dtype=np.float64,
    )
    if s.size == 0:
        raise SystemExit("no usable scores in the shadow rows")
    return s


def gate_theta(scores: np.ndarray) -> float:
    """The hysteresis gate's cut, at the project's fixed 0.85 convention.

    Solved with the same ``derive_thresholds`` every earlier threshold in this
    project came from, so this line's theta is comparable to theirs rather than
    merely similar.
    """
    return float(derive_thresholds(scores.tolist(), THETA_TOP_FRACTION, 0.0)[0])


def gst_cells(step: int = GST_STEP, max_sum: int = GST_MAX_SUM) -> list[tuple[int, int, int]]:
    """(full, warm_a, warm_b) percent triples on the simplex; all-zero excluded."""
    vals = list(range(0, max_sum + 1, step))
    return sorted(c for c in itertools.product(vals, repeat=3) if 0 < sum(c) <= max_sum)


def gst_cuts(scores: np.ndarray, shares: tuple[float, float, float]) -> list[float]:
    """Descending-quantile cuts for cumulative shares (tgrid convention)."""
    arr = np.sort(scores)[::-1]
    n = arr.size
    out, cum = [], 0.0
    for share in shares:
        cum += share
        i = max(0, min(n - 1, int(cum * n) - 1))
        out.append(float(arr[i]) if share > 0 else math.inf)
    return out


def _judge_from_cuts(cuts: list[float], warm_ts: tuple[float, ...]) -> tuple[dict, list[str]]:
    """A ``threshold`` judge from cuts in ladder order, dropping dead rungs.

    ``ThresholdJudge`` walks the ladder and takes the first rung the score
    clears, so a rung whose cut is not strictly below the one above it can
    never fire. The loader rejects that shape, so such a rung is dropped and
    its name recorded: an unreachable rung changes neither the verdict nor the
    cost, and dropping it states the same rule honestly.

    The FULL rung is the exception and must never be dropped. It is the judge's
    own ``threshold`` field, so removing it would leave the warm tiers with
    nothing above them; a cell that gives FULL_HIT a zero share (30 of the 34
    GST cells do) is expressed by putting that threshold *above* the score
    domain, where it can never fire while the warm tiers below it still can.
    Dropping it instead silently turns every such cell into the all-MISS arm.
    """
    names = ["full"] + [f"warm{int(round(t * 100))}" for t in warm_ts]
    dropped: list[str] = []
    full_cut = cuts[0] if math.isfinite(cuts[0]) else CUT_NONE
    judge = {"type": "threshold", "threshold": float(full_cut)}
    warm, prev = [], full_cut
    for name, cut, start_t in zip(names[1:], cuts[1:], warm_ts):
        if not math.isfinite(cut) or cut >= prev:
            dropped.append(name)
            continue
        warm.append({"threshold": float(cut), "start_t": float(start_t)})
        prev = cut
    if warm:
        judge["warm_tiers"] = warm
    return judge, dropped


def build_arm(template: dict, judge: dict, *, layer: str, theta: float) -> dict:
    doc = copy.deepcopy(template)
    cp1 = doc["checkpoints"]["cp1"]
    cp1["judge"] = judge
    cp1["gate"] = gate_section(layer, float(theta))
    doc["write_policy"] = {"type": "never"}
    return doc


def write_arm(doc: dict, path: pathlib.Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    # Load it back through the production loader: a judge shape that the
    # validator rejects must fail here, not when the fleet is already up.
    load_cache_config(str(path))
    return _sha(path)


def load_cost(path: pathlib.Path) -> rc.StageCost:
    d = json.loads(path.read_text(encoding="utf-8"))
    schedule = groot_n15_schedule(int(d.get("num_steps", DENOISING_STEPS)))
    if schedule.schedule_id != d.get("schedule_id", schedule.schedule_id):
        raise SystemExit(
            f"{path}: schedule_id {d.get('schedule_id')!r} != {schedule.schedule_id!r}"
        )
    return rc.StageCost(
        teacher=d.get("teacher", "groot_libero"),
        schedule=schedule,
        stage1_ms=float(d["stage1_ms"]),
        stage2_ms=float(d["stage2_ms"]),
        stage3_head_ms=float(d["stage3_head_ms"]),
        stage3_step_ms=float(d["stage3_step_ms"]),
        provenance=str(d["provenance"]),
        linear_stage3=bool(d.get("linear_stage3", False)),
    )


def emit(args) -> dict:
    tag = SUITE_TAG[args.suite]
    out_dir = REPO_ROOT / args.out_dir / args.suite
    template_path = out_dir / "template.yaml"
    if not template_path.exists():
        raise SystemExit(f"missing {template_path}; run emit_rit_template first")
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    arms_dir = out_dir / "arms"
    arms_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(pathlib.Path(args.shadow))
    s = scores_of(rows)
    theta = gate_theta(s)

    record: dict = {
        "protocol": PROTOCOL,
        "suite": args.suite,
        "rules": list(args.rules),
        "template": str(template_path.relative_to(REPO_ROOT)),
        "template_sha256": _sha(template_path),
        "shadow": args.shadow,
        "shadow_sha256": _sha(pathlib.Path(args.shadow)),
        "n_shadow_rows": len(rows),
        "n_scored": int(s.size),
        "gate_theta": theta,
        "gate_theta_fraction": THETA_TOP_FRACTION,
        "warm_ts": list(WARM_TS),
        "denoising_steps": DENOISING_STEPS,
        "arms": {},
    }
    matrices: dict[str, list[dict]] = {}

    def add(arm: str, doc: dict, matrix: str, meta: dict) -> None:
        path = arms_dir / f"{arm}.yaml"
        meta["yaml"] = str(path.relative_to(REPO_ROOT))
        meta["sha256"] = write_arm(doc, path)
        record["arms"][arm] = meta
        matrices.setdefault(matrix, []).append(
            {"arm": arm, "yaml": str(path), "suite": args.suite}
        )

    if "anchors" in args.rules:
        for arm, cut, what in (
            (f"{tag}_anchor_cache", CUT_ALL, "every step FULL_HIT (pure cache)"),
            (f"{tag}_anchor_teacher", CUT_NONE, "every step MISS (pure teacher)"),
        ):
            doc = build_arm(
                template, {"type": "threshold", "threshold": cut},
                layer=LAYER_PRIMARY, theta=theta,
            )
            add(arm, doc, "anchor", {"rule": "anchor", "threshold": cut, "note": what})

    if "gst" in args.rules:
        for full, wa, wb in gst_cells(args.gst_step, args.gst_max_sum):
            cuts = gst_cuts(s, (full / 100.0, wa / 100.0, wb / 100.0))
            judge, dropped = _judge_from_cuts(cuts, WARM_TS)
            arm = f"{tag}_gst_f{full:02d}w{wa:02d}v{wb:02d}"
            doc = build_arm(template, judge, layer=LAYER_SECONDARY, theta=theta)
            add(arm, doc, "hg", {
                "rule": "gst", "cell": [full, wa, wb],
                "cuts": [None if not math.isfinite(c) else c for c in cuts],
                "dropped_rungs": dropped,
            })

    if "rit" in args.rules:
        if not args.cost:
            raise SystemExit(
                "--rules rit needs --cost: the IR grid is an axis of measured "
                "milliseconds, and borrowing another benchmark's constants would "
                "relabel every arm without failing anything"
            )
        cost = load_cost(pathlib.Path(args.cost))
        record["cost"] = {
            "stage1_ms": cost.stage1_ms, "stage2_ms": cost.stage2_ms,
            "stage3_head_ms": cost.stage3_head_ms, "stage3_step_ms": cost.stage3_step_ms,
            "miss_ms": rc.miss_cost(cost), "provenance": cost.provenance,
        }
        # The ladder is fitted on the calibration rows; the tolerance-to-IR
        # bookkeeping runs on the measured deployed scores. Those are the same
        # sample here (calibration and evaluation share the 500-init pool by
        # owner ruling), but passing it explicitly keeps the two roles separate
        # -- addressing on knot quantiles is what put every Pi0.5 v1 cut in a
        # region deployment never visited.
        fits = fit_ladders(rows, cost, list(WARM_TS), list(args.ks), args.alpha, ir_sample=s)
        record["fits"] = {}
        for k in args.ks:
            f = fits[k]
            lo, hi = f["ir_range"]
            record["fits"][str(k)] = {
                "n_rows": f["n_rows"], "n_seg_req": f["n_seg_req"],
                "ir_range": [lo, hi], "knots": f["knots"],
                "tiers": [t.name for t in f["tiers"]],
                "tier_cost_ms": [t.cost_ms for t in f["tiers"]],
            }
            for target in args.targets:
                if not lo <= target <= hi:
                    record["fits"][str(k)].setdefault("unreachable", []).append(target)
                    continue
                sol = rc.delta_for_ir(f["fit"], f["ir_s"], target, cost)
                cuts = rc.cuts_for(f["fit"], sol["delta"])
                judge, dropped = _judge_from_cuts(cuts, WARM_TS[: k - 1])
                arm = f"{tag}_rit_k{k}_ir{int(round(target))}"
                doc = build_arm(template, judge, layer=LAYER_SECONDARY, theta=theta)
                add(arm, doc, "hg", {
                    "rule": "rit", "k": k, "target_ir": target,
                    "delta": sol["delta"], "predicted_ir": sol["predicted_ir"],
                    "ir_gap": sol["predicted_ir"] - target,
                    "cuts": [None if not math.isfinite(c) else c for c in cuts],
                    "dropped_rungs": dropped,
                })

    for matrix, arms in matrices.items():
        path = out_dir / f"arm_matrix_{matrix}.yaml"
        path.write_text(yaml.safe_dump({"arms": sorted(arms, key=lambda r: r["arm"])},
                                       sort_keys=False), encoding="utf-8")
        record.setdefault("matrices", {})[matrix] = str(path.relative_to(REPO_ROOT))
        print(f"wrote {path}  ({len(arms)} arms)")

    rec_path = out_dir / "arm_record.json"
    rec_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {rec_path}  gate_theta={theta:.6f}  arms={len(record['arms'])}")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, choices=tuple(SUITE_TAG))
    ap.add_argument("--shadow", required=True, help="concatenated shadow rows jsonl")
    ap.add_argument("--out-dir", default="exp/libero_groot/config/rit")
    ap.add_argument("--rules", default="anchors,gst,rit")
    ap.add_argument("--cost", default="", help="stage-cost json (required for rit)")
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--target-ir", default=",".join(f"{t:g}" for t in DEFAULT_TARGETS))
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--gst-step", type=int, default=GST_STEP)
    ap.add_argument("--gst-max-sum", type=int, default=GST_MAX_SUM)
    args = ap.parse_args()

    args.rules = tuple(r.strip() for r in args.rules.split(",") if r.strip())
    bad = [r for r in args.rules if r not in RULES]
    if bad:
        raise SystemExit(f"unknown rule(s) {bad}; known {RULES}")
    args.ks = [int(k) for k in args.ks.split(",") if k.strip()]
    args.targets = [float(t) for t in args.target_ir.split(",") if t.strip()]
    emit(args)


if __name__ == "__main__":
    main()
