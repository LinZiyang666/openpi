"""Emit the RoboCasa365 RIT ladder: the calibration cell and the IR-addressed arms.

Two subcommands, in the order they are run.

``calib`` writes the single cell the shadow pass serves: the frozen retrieval
recipe (one weight configuration, text-IVF, the W13 library) with
``gate: always_search`` / ``judge: always_hit`` / ``write_policy: never``. Its
verdict is never applied -- ``rit_shadow`` only reads the score and the winner
-- but running the real orchestrator is what makes the calibration scores the
same quantity the deployed arms cut on.

``arms`` fits the ladder on the shadow rows and writes one yaml per grid point.
The grid is laid on **inference ratio**, not on the tolerance: a target IR is
inverted through ``delta_for_ir`` to a tolerance and then to one score cut per
rung, so the frontier is addressed by the budget it spends rather than by an
abstract risk knob. Deployment is a ``threshold`` judge plus ``warm_tiers`` --
the s-only ladder is exactly a nested set of score cuts, and ``threshold`` is
on GR00T's serviceable-judge whitelist while ``dispatch_surface`` is not.

Every arm carries the same hysteresis gate as the LIBERO line (theta at the
GTP convention, j=3 / probe_interval=3 / L=6), cut from the same calibration
scores the ladder is fitted on.

Public interface: ``w13_spec``, ``build_calibration_cell``, ``build_arm_cell``,
``fit_ladders``, ``emit_arms``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from openpi.cache.types import PI05_V1, DenoiseSchedule, groot_n15_schedule

from exp.dispatch_surface.emit_precheck_yamls import LAYER_SECONDARY, gate_section
from exp.gate_threshold_pareto.solve_gtp import THETA_TOP_FRACTION
from exp.robocasa365 import emit_ws_search2_yamls as ws2
from exp.dispatch_surface.rit_pl import MIN_SEG_SAMPLES, segment_index
from exp.rit_pareto.rit_k import fit_record_fields, floor_info
from exp.robocasa365 import rit_cost_rc as rc
from exp.robocasa365.emit_ws_search_yamls import weight_matrix
from exp.verdict_factor_judge.phase3.threshold_solver import derive_thresholds

DEFAULT_LIBRARY_DIR = "/data/robocasa365_cache/cache_artifacts_w13"
LIBRARY_TAG = "w13"
DEFAULT_ALPHA = 0.05

#: The weight configuration each teacher's ladder is frozen at (owner ruling,
#: 2026-09-08). Chosen from the round-2 search: GR00T's is the cell that leads
#: the 416-episode densify arm and sits on the smoothed ridge; pi0.5's is the
#: only cell where its raw and smoothed argmax agree.
FROZEN_WEIGHT_CID = {
    "groot_tp": "grid3_vision_0@12_vision_2@37_robot_state@50",
    "pi05": "grid_vision_1@87_robot_state@12",
}

#: Resume timesteps per teacher, nearest-to-clean first (owner ruling).
#: GR00T's schedule ascends, so 0.75 leaves one of four steps; pi0.5's
#: descends, so 0.3 leaves three of ten. The two ladders are matched on the
#: fraction of stage 3 they re-run (25%/30% and 50%/50%), not on step count.
WARM_TS = {"groot_tp": [0.75, 0.5], "pi05": [0.3, 0.5]}


def schedule_of(teacher: str) -> DenoiseSchedule:
    return groot_n15_schedule(4) if teacher == "groot_tp" else PI05_V1


def w13_spec(teacher: str, library_dir: str = DEFAULT_LIBRARY_DIR) -> dict:
    """Teacher facts pointed at the W13 library.

    The merged 13-task library carries no ``pin_id``: the pick lane was
    collected pinned and the contact lane was not, and the merge only keeps an
    id every input agrees on. The pin is therefore asserted at collection time
    and in the run plan, not by the artifact binding.
    """
    # The stem is the artifact's file stem, which is also the key the Phase-1
    # normalizer calibration writes: the two must agree or the calibration
    # lookup fails at emit time rather than at serve time.
    stem = f"{teacher}_spatial_pool_16_{LIBRARY_TAG}_full"
    return {
        **ws2.TEACHERS[teacher],
        "stem": stem,
        "preload": f"{library_dir.rstrip('/')}/{stem}.pkl",
    }


def _base_cell(teacher: str, calib: dict, spec: dict, weight_cid: str) -> dict:
    configs = weight_matrix()
    if weight_cid not in configs:
        raise SystemExit(f"weight cell {weight_cid!r} is not in the round-1 matrix")
    entry = ws2.calibration_entry(calib, spec, teacher)
    cfg = ws2.build_cell(configs[weight_cid], entry, teacher, text_ivf=True, spec=spec)
    ws2.verify_cell(cfg, weight_cid, teacher, text_ivf=True, spec=spec)
    return cfg


def build_calibration_cell(
    teacher: str, calib: dict, spec: dict, weight_cid: str,
    schedule: DenoiseSchedule | None = None,
) -> dict:
    """The shadow pass's cell: real retrieval, verdict read but never applied.

    The schedule is stamped even though an ``always_hit`` recipe is not a warm
    one and the guard would not ask for it: the shadow resumes the winner's
    intermediates at every rung, so this cell does depend on the library's loop
    identity even though its verdict does not.
    """
    cfg = _base_cell(teacher, calib, spec, weight_cid)
    cp1 = cfg["checkpoints"]["cp1"]
    cp1["gate"] = {"type": "always_search"}
    cp1["judge"] = {"type": "always_hit"}
    cfg["write_policy"] = {"type": "never"}
    schedule = schedule_of(teacher) if schedule is None else schedule
    if schedule is not PI05_V1:
        cfg["denoise_schedule"] = schedule.schedule_id
    return cfg


def build_teacher_cell(teacher: str, calib: dict, spec: dict, weight_cid: str,
                       schedule: DenoiseSchedule | None = None) -> dict:
    """A reference arm whose every step is full inference, at the run's budget.

    Written as a threshold judge whose cut is unreachable rather than by
    disabling the checkpoint: the serving guard requires exactly ``cp1``
    enabled, so a disabled-cache yaml is refused outright. The search still
    runs and still returns a winner; the verdict is always MISS, so the
    executed action is the teacher's own -- which is the only thing this arm is
    measuring. Its gate is ``always_search`` so no step is skipped for a reason
    that has nothing to do with the reference.
    """
    cfg = _base_cell(teacher, calib, spec, weight_cid)
    cp1 = cfg["checkpoints"]["cp1"]
    cp1["gate"] = {"type": "always_search"}
    cp1["judge"] = {"type": "threshold", "threshold": UNREACHABLE_CUT}
    cfg["write_policy"] = {"type": "never"}
    schedule = schedule_of(teacher) if schedule is None else schedule
    if schedule is not PI05_V1:
        cfg["denoise_schedule"] = schedule.schedule_id
    return cfg


def build_arm_cell(
    teacher: str,
    calib: dict,
    spec: dict,
    weight_cid: str,
    *,
    tiers: tuple[rc.RCTier, ...],
    thetas: dict[str, float],
    gate_theta: float,
    schedule: DenoiseSchedule,
) -> dict:
    """One deployed rung ladder: threshold judge + warm tiers, behind the H gate.

    ``warm_tiers`` are written cheapest-first, which is also
    highest-threshold-first: ``ThresholdJudge`` takes the first rung whose
    threshold the score clears, so that order is what makes the walk pick the
    cheapest admissible tier rather than the safest one.
    """
    cfg = _base_cell(teacher, calib, spec, weight_cid)
    cp1 = cfg["checkpoints"]["cp1"]
    full = [t for t in tiers if t.hit_type == "FULL_HIT"]
    warm = [t for t in tiers if t.hit_type == "WARM_START"]
    if len(full) != 1:
        raise ValueError("the ladder needs exactly one FULL_HIT rung")
    # At the cheap end of the grid the fit puts two rungs at the same cut. The
    # judge walks the ladder and takes the first rung the score clears, so a
    # rung whose cut is not strictly below the one above it can never fire;
    # the loader rejects that shape rather than serving a dead rung. Drop the
    # dead ones instead of nudging the cuts: an unreachable rung changes
    # neither the cost nor the verdict, so dropping it emits the same rule
    # honestly, and the dropped names are recorded on the arm.
    kept, dropped, prev = [], [], None
    for tier in [full[0]] + sorted(warm, key=lambda t: t.cost_ms):
        cut = _cut(thetas[tier.name])
        if prev is not None and cut >= prev:
            dropped.append(tier.name)
            continue
        kept.append((tier, cut))
        prev = cut
    judge = {"type": "threshold", "threshold": kept[0][1]}
    warm_kept = [(t, c) for t, c in kept[1:]]
    if warm_kept:
        judge["warm_tiers"] = [
            {"threshold": c, "start_t": float(t.start_t)} for t, c in warm_kept
        ]
    cp1["judge"] = judge
    cfg["_dropped_rungs"] = dropped
    cp1["gate"] = gate_section(LAYER_SECONDARY, float(gate_theta))
    cfg["write_policy"] = {"type": "never"}
    if schedule is not PI05_V1:
        cfg["denoise_schedule"] = schedule.schedule_id
    dropped = cfg.pop("_dropped_rungs")
    _verify_arm(cfg, tiers, thetas)
    return cfg, dropped


#: A cut the fit puts out of reach. Per-field z-score with a tanh squash bounds
#: every field to [0, 1] and the weights sum to one, so the fused score can
#: never reach 2.0 -- writing it keeps the rung in the ladder (its identity and
#: its cost still describe the arm) while guaranteeing it never fires. Infinity
#: is not written: yaml round-trips it as ``.inf`` and the config loader has no
#: reason to accept a non-finite threshold.
UNREACHABLE_CUT = 2.0


def _cut(value: float) -> float:
    v = float(value)
    return UNREACHABLE_CUT if not math.isfinite(v) else v


def _verify_arm(cfg: dict, tiers: tuple[rc.RCTier, ...], thetas: dict[str, float]) -> None:
    """Shape invariants a mis-ordered ladder would otherwise pass silently."""
    judge = cfg["checkpoints"]["cp1"]["judge"]
    assert judge["type"] == "threshold", judge
    cuts = [judge["threshold"]] + [t["threshold"] for t in judge.get("warm_tiers", [])]
    # The loader demands strictly decreasing, not merely non-increasing: two
    # rungs at the same cut would make the cheaper one unreachable, and a
    # ladder whose rungs cannot all fire is not the ladder that was priced.
    if any(b >= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError(f"cuts must strictly decrease down the ladder: {cuts}")
    gate = cfg["checkpoints"]["cp1"]["gate"]
    assert gate["type"] == "score_hysteresis", gate
    missing = [t.name for t in tiers if t.name not in thetas]
    if missing:
        raise ValueError(f"no cut for rungs {missing}")


# ------------------------------------------------------------------
# Fit
# ------------------------------------------------------------------


def load_shadow(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def usable(rows: list[dict], y_keys: list[str]) -> list[dict]:
    """Rows carrying a finite score and every risk column the ladder needs."""
    out = []
    for r in rows:
        s = r.get("s")
        if s is None or not math.isfinite(float(s)):
            continue
        ys = [r.get(k) for k in y_keys]
        if any(y is None or not math.isfinite(float(y)) for y in ys):
            continue
        out.append(r)
    return out


def choose_knots_pooled(knot_sample, fit_sample, ladder) -> tuple[np.ndarray, int] | None:
    """Knots placed on one distribution, occupancy checked on another.

    ``choose_knots`` puts equal-frequency knots on the sample it is given and
    backs down its ladder until every segment holds enough of THAT sample. Both
    jobs are the same sample only when calibration and deployment agree. When
    they do not, the two have to be split: the knots belong on the distribution
    the rule will be APPLIED to (otherwise the curve has no resolution where it
    is used), while the occupancy floor belongs on the rows the curve is FITTED
    from (otherwise a segment is interpolated from a handful of labels).

    Returns the finest ladder rung satisfying both, or None when no rung does.
    """
    knot_sample = np.asarray(knot_sample, dtype=np.float64)
    fit_sample = np.asarray(fit_sample, dtype=np.float64)
    for n_req in ladder:
        if int(n_req) < 2:
            raise ValueError("every ladder rung must request at least two segments")
        knots = np.unique(
            np.quantile(knot_sample, np.linspace(0.0, 1.0, int(n_req) + 1), method="linear")
        )
        # The fitted rows must reach both ends, or the outermost segments are
        # extrapolation dressed as a fit.
        knots = np.unique(np.concatenate([[min(knots[0], fit_sample.min())], knots,
                                          [max(knots[-1], fit_sample.max())]]))
        if len(knots) - 1 < 2:
            continue
        counts = np.bincount(segment_index(knots, fit_sample), minlength=len(knots) - 1)
        if counts.min() >= MIN_SEG_SAMPLES:
            return knots, int(n_req)
    return None


def fit_ladders(rows: list[dict], cost: rc.StageCost, warm_ts: list[float],
                ks: list[int], alpha: float, ir_sample=None, knot_sample=None,
                knot_ladder=None) -> dict:
    """One fit per ladder depth, each on the rows that carry its risk columns.

    ``ir_sample`` is the score sample the delta-to-IR bookkeeping runs on, and
    it is deliberately separable from the rows the risk curves are fitted on.
    q(s) is a property of the retrieval score and stays fitted on the
    calibration pairs; the IR of a given tolerance is instead a property of the
    score DISTRIBUTION the rule will meet at serving time. For a teacher whose
    trajectory drifts away from the library those two distributions differ
    enough that addressing on the calibration one puts every cut in a region
    deployment never visits. Passing the measured deployed scores fixes the
    addressing without touching the ladder construction: one delta still yields
    every rung's cut through the same nested inversion.
    """
    fits = {}
    for k in ks:
        tiers = rc.ladder(cost, warm_ts[: k - 1])
        y_keys = [t.y_key for t in tiers]
        sub = usable(rows, y_keys)
        if not sub:
            raise SystemExit(f"k={k}: no usable calibration rows for columns {y_keys}")
        s = np.array([float(r["s"]) for r in sub], dtype=np.float64)
        ys = {t.y_key: np.array([float(r[t.y_key]) for r in sub]) for t in tiers}
        ys = {t.name: ys[t.y_key] for t in tiers}
        ladder = knot_ladder or rc.KNOT_LADDER
        picked = (rc.choose_knots(s, ladder) if knot_sample is None
                  else choose_knots_pooled(knot_sample, s, ladder))
        if picked is None:
            raise SystemExit(f"k={k}: the knot ladder is exhausted on {len(s)} rows")
        knots, n_seg_req = picked
        fit = rc.fit(s, ys, knots, tiers=tiers, n_seg_req=n_seg_req, alpha=alpha)
        ir_s = s if ir_sample is None else np.asarray(ir_sample, dtype=np.float64)
        lo, hi = rc.attainable_range(fit, ir_s, cost)
        fits[k] = {"fit": fit, "tiers": tiers, "s": s, "ir_s": ir_s, "n_rows": len(sub),
                   "ir_range": (lo, hi), "knots": knots.tolist(), "n_seg_req": n_seg_req}
    return fits


def per_task_stats(rows: list[dict], warm_ts: list[float], cost: rc.StageCost) -> dict:
    """Per-task calibration summary: row counts, score quantiles, risk means.

    The deployed ladder in this run is global -- one set of cuts over all
    thirteen tasks. Tasks differ enormously in how well the cache answers them,
    so a per-task ladder is a live alternative; recording the per-task view now
    means that variant can be fitted from the same shadow rows instead of
    costing another calibration pass.
    """
    y_keys = [t.y_key for t in rc.ladder(cost, warm_ts)]
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r.get("task", ""), []).append(r)
    out = {}
    for task, rs in sorted(by_task.items()):
        s = np.array([float(r["s"]) for r in rs
                      if r.get("s") is not None and math.isfinite(float(r["s"]))])
        entry = {"n_rows": len(rs), "n_scored": int(s.size)}
        if s.size:
            entry["score_quantiles"] = {
                str(q): float(np.quantile(s, q)) for q in (0.0, 0.05, 0.15, 0.5, 0.85, 0.95, 1.0)
            }
            entry["gate_theta_if_per_task"] = float(
                derive_thresholds(s.tolist(), THETA_TOP_FRACTION, 0.0)[0]
            )
        for key in y_keys:
            vals = np.array([float(r[key]) for r in rs
                             if r.get(key) is not None and math.isfinite(float(r[key]))])
            if vals.size:
                entry[key] = {"n": int(vals.size), "mean": float(vals.mean()),
                              "q95": float(np.quantile(vals, 0.95))}
        out[task] = entry
    return out


def emit_arms(out_dir: Path, teacher: str, calib: dict, spec: dict, weight_cid: str,
              fits: dict, cost: rc.StageCost, targets: list[float],
              gate_theta: float, schedule: DenoiseSchedule) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for k, blob in sorted(fits.items()):
        for target in targets:
            lo, hi = blob["ir_range"]
            if not (lo <= target <= hi):
                index[f"k{k}__ir{target:05.1f}"] = {"skipped": "outside attainable range",
                                                    "ir_range": [lo, hi]}
                continue
            sol = rc.delta_for_ir(blob["fit"], blob.get("ir_s", blob["s"]), target, cost)
            cfg, dropped = build_arm_cell(
                teacher, calib, spec, weight_cid, tiers=blob["tiers"],
                thetas=sol["thetas"], gate_theta=gate_theta, schedule=schedule,
            )
            cid = f"k{k}__ir{target:05.1f}"
            path = out_dir / f"{cid}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            ws2.validate_on_disk(path)
            index[cid] = {"file": path.name, "k": k, "target_ir": target,
                          "predicted_ir": sol["predicted_ir"], "delta": sol["delta"],
                          "thetas": sol["thetas"], "gate_theta": gate_theta,
                          "unreachable_rungs": [n for n, v in sol["thetas"].items()
                                                if not math.isfinite(float(v))],
                          # Whether each cut is carried by data or is resting on
                          # the LP's strict-monotonicity floor: a cut on the floor
                          # is an artefact of invertibility, not evidence.
                          "floor": floor_info(blob["fit"], blob["s"], sol["delta"]),
                          "dropped_rungs": dropped}
    # The all-FULL_HIT reference: the cheapest point the ladder can ever reach.
    cfg = build_calibration_cell(teacher, calib, spec, weight_cid)
    cfg["checkpoints"]["cp1"]["gate"] = gate_section(LAYER_SECONDARY, float(gate_theta))
    path = out_dir / "always_hit.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    ws2.validate_on_disk(path)
    index["always_hit"] = {"file": path.name, "k": 1,
                           "predicted_ir": 100.0 * cost.stage1_ms / rc.miss_cost(cost),
                           "gate_theta": gate_theta}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    # Freeze record: the eval driver re-hashes every dispatched yaml against
    # this before it ships one to a server, so a hand-edit or a re-emit after
    # the freeze stops the run instead of silently changing what was measured.
    cells = {
        cid: hashlib.sha256((out_dir / meta["file"]).read_text().encode()).hexdigest()
        for cid, meta in index.items()
        if "file" in meta
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "cells": cells,
                "emitter_sha256": {
                    Path(m.__file__).name: hashlib.sha256(
                        Path(m.__file__).read_text().encode()
                    ).hexdigest()
                    for m in (ws2, rc)
                }
                | {
                    "emit_rit_rc.py": hashlib.sha256(
                        Path(__file__).read_text().encode()
                    ).hexdigest()
                },
            },
            indent=1,
            sort_keys=True,
        )
    )
    return index


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _cost_from_json(path: str, teacher: str) -> rc.StageCost:
    d = json.loads(Path(path).read_text())
    if d["teacher"] != teacher:
        raise SystemExit(f"cost record is for {d['teacher']!r}, not {teacher!r}")
    sched = schedule_of(teacher)
    if d["schedule_id"] != sched.schedule_id:
        raise SystemExit(f"cost record schedule {d['schedule_id']!r} != {sched.schedule_id!r}")
    return rc.StageCost(teacher=teacher, schedule=sched, stage1_ms=d["stage1_ms"],
                        stage2_ms=d["stage2_ms"], stage3_head_ms=d["stage3_head_ms"],
                        stage3_step_ms=d["stage3_step_ms"],
                        provenance=d["provenance"], linear_stage3=d.get("linear_stage3", False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calib")
    c.add_argument("--teacher", required=True, choices=("groot_tp", "pi05"))
    c.add_argument("--calibration", required=True)
    c.add_argument("--library-dir", default=DEFAULT_LIBRARY_DIR)
    c.add_argument("--weight-cid", default="")
    c.add_argument("--out", required=True)

    t = sub.add_parser("teacher")
    t.add_argument("--teacher", required=True, choices=("groot_tp", "pi05"))
    t.add_argument("--calibration", required=True)
    t.add_argument("--library-dir", default=DEFAULT_LIBRARY_DIR)
    t.add_argument("--weight-cid", default="")
    t.add_argument("--out-dir", required=True)

    a = sub.add_parser("arms")
    a.add_argument("--teacher", required=True, choices=("groot_tp", "pi05"))
    a.add_argument("--calibration", required=True)
    a.add_argument("--library-dir", default=DEFAULT_LIBRARY_DIR)
    a.add_argument("--weight-cid", default="")
    a.add_argument("--shadow", required=True)
    a.add_argument("--cost", required=True)
    a.add_argument("--ks", default="2,3")
    a.add_argument("--n-targets", type=int, default=6)
    a.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    a.add_argument("--ir-scores", default="",
                   help="json list of scores the delta-to-IR bookkeeping runs on "
                        "(default: the calibration rows). Use the MEASURED deployed "
                        "scores when calibration does not predict deployment.")
    a.add_argument("--knot-scores", default="",
                   help="json list the knots are placed on; occupancy is still "
                        "checked against the calibration rows")
    a.add_argument("--knot-ladder", default="",
                   help="comma-separated segment counts to try, e.g. 12,6")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--record", required=True)

    args = ap.parse_args()
    teacher = args.teacher
    weight_cid = args.weight_cid or FROZEN_WEIGHT_CID[teacher]
    calib = json.loads(Path(args.calibration).read_text())
    spec = w13_spec(teacher, args.library_dir)

    if args.cmd == "teacher":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = build_teacher_cell(teacher, calib, spec, weight_cid)
        path = out_dir / "teacher_only.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        ws2.validate_on_disk(path)
        index = {"teacher_only": {"file": path.name, "k": 0,
                                  "predicted_ir": 100.0,
                                  "note": "every step is full inference"}}
        (out_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
        (out_dir / "provenance.json").write_text(json.dumps(
            {"cells": {"teacher_only": hashlib.sha256(path.read_text().encode()).hexdigest()},
             "emitter_sha256": {"emit_rit_rc.py": hashlib.sha256(
                 Path(__file__).read_text().encode()).hexdigest()}},
            indent=1, sort_keys=True))
        print(f"wrote {path}")
        return

    if args.cmd == "calib":
        cfg = build_calibration_cell(teacher, calib, spec, weight_cid)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        ws2.validate_on_disk(out)
        print(f"wrote {out} (weights {weight_cid}, library {spec['preload']})")
        return

    cost = _cost_from_json(args.cost, teacher)
    rows = load_shadow(args.shadow)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    ir_sample = (json.loads(Path(args.ir_scores).read_text()) if args.ir_scores else None)
    knot_sample = (json.loads(Path(args.knot_scores).read_text()) if args.knot_scores else None)
    knot_ladder = (tuple(int(x) for x in args.knot_ladder.split(",") if x.strip())
                   if args.knot_ladder else None)
    fits = fit_ladders(rows, cost, WARM_TS[teacher], ks, args.alpha,
                       ir_sample=ir_sample, knot_sample=knot_sample, knot_ladder=knot_ladder)
    ranges = [blob["ir_range"] for blob in fits.values()]
    targets = rc.common_grid(ranges, args.n_targets)
    all_s = [float(r["s"]) for r in rows if r.get("s") is not None and math.isfinite(float(r["s"]))]
    gate_theta = float(derive_thresholds(all_s, THETA_TOP_FRACTION, 0.0)[0])
    per_task = per_task_stats(rows, WARM_TS[teacher], cost)
    index = emit_arms(Path(args.out_dir), teacher, calib, spec, weight_cid, fits, cost,
                      targets, gate_theta, schedule_of(teacher))
    record = {
        "teacher": teacher, "weight_cid": weight_cid, "library": spec["preload"],
        "alpha": args.alpha, "targets_ir": targets,
        "gate": {"theta": gate_theta, "top_fraction": THETA_TOP_FRACTION,
                 "n_scores": len(all_s)},
        # Kept so a per-task variant of the ladder needs no second calibration
        # pass: the shadow rows already carry the task, and these are the
        # summaries a per-task cut would be derived from. This run deploys ONE
        # ladder over all tasks; the per-task theta here is recorded, not used.
        "per_task": per_task,
        "cost": {"stage1_ms": cost.stage1_ms, "stage2_ms": cost.stage2_ms,
                 "stage3_head_ms": cost.stage3_head_ms, "stage3_step_ms": cost.stage3_step_ms,
                 "miss_ms": rc.miss_cost(cost), "provenance": cost.provenance},
        "ladders": {
            str(k): {"tiers": [{"name": t.name, "hit_type": t.hit_type, "start_t": t.start_t,
                                "cost_ms": t.cost_ms, "ir_share": 100 * t.cost_ms / rc.miss_cost(cost)}
                               for t in blob["tiers"]],
                     "n_rows": blob["n_rows"], "ir_range": list(blob["ir_range"]),
                     "knots": blob["knots"], "n_seg_req": blob["n_seg_req"]}
            for k, blob in sorted(fits.items())},
        # The fitted curve itself: knot positions and the q value at each knot,
        # plus the monotone floor. Between knots the curve is linear and the
        # cut is solved analytically inside its segment, so these fields are
        # the whole continuous q(s) -- reproducing a cut needs nothing else.
        "fits": {str(k): fit_record_fields(blob["fit"]) for k, blob in sorted(fits.items())},
        "arms": index,
    }
    Path(args.record).write_text(json.dumps(record, indent=1, sort_keys=True))
    print(json.dumps({"targets": [round(t, 2) for t in targets],
                      "gate_theta": round(gate_theta, 6),
                      "ir_ranges": {str(k): [round(v, 2) for v in b["ir_range"]]
                                    for k, b in fits.items()},
                      "arms": len(index)}, indent=1))


if __name__ == "__main__":
    main()
