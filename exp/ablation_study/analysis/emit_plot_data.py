"""Collect every plotted point of the executor-substitution figures into one file.

The figures used to read the conductor journals and per-step exports directly,
which tied them to a 200 MB gitignored tree: re-plotting after the raw data
moved off this disk was impossible, and a re-run meant re-deriving the same
aggregates inside the plotting code. The pipeline is now

    journals / per_step / anchors / analyze_ablation.py
        --> emit_plot_data.py --> plot_data.json --> plot_ablation.py

and ``plot_data.json`` is the only file the figures read. One suite is one
family block; re-collecting a suite replaces exactly that block and leaves the
other untouched, so the two suites can be refreshed independently.

Provenance and the division of labour are explicit, because this collector is
*not* purely a copier the way the cache-size one is:

* Main-matrix SR / Wilson CI are copied verbatim from ``analyze_ablation.py``
  output when it is supplied, and the aggregate recomputed here is asserted
  against it -- a mismatch is an error, never a silent overwrite.
* The 4b kinematic sweep arms, the FULL_HIT rates and the teacher anchor have
  no upstream analyzer, so they are aggregated here. Every such point records
  ``success_rate_source`` so a reader can tell copied numbers from derived ones.

The x axis of the Pareto figure is *semantic*, not a plain function of the hit
rate: arms whose MISS slot runs a student never touch the teacher, so their
teacher-inference rate is 0 regardless of how often they hit. Each point
therefore carries ``teacher_inference_rate_basis`` stating which rule produced
its abscissa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from collections import defaultdict

SCHEMA = 1

# Main-matrix arm order along the fig-1 x axis.
MAIN_ARMS = (
    "cache_baseline",
    "hit_act",
    "hit_smolvla",
    "miss_act",
    "miss_smolvla",
    "pure_act",
    "pure_smolvla",
)

# 4b kinematic-verdict sweep, in decreasing-threshold order = the connect order.
SWEEP_TAGS = ("fh67", "fh49", "fh40", "fh25", "fh11")

# Abscissa rules. The Pareto x axis counts steps that pay the teacher's full
# Stage1-3 pipeline, which is a property of the routing semantics rather than of
# the measured hit rate.
X_FROM_HIT_RATE = "1 - full_hit_rate"
X_NO_TEACHER = "no teacher in the loop"
X_ALL_TEACHER = "all steps on teacher"

# arm -> (series, label, hit-slot executor, miss-slot executor, abscissa rule)
ARM_SEMANTICS: dict[str, tuple[str, str, str, str, str]] = {
    "cache_baseline": (
        "cache_replay",
        "cache replay at hit (full cache system)",
        "cache payload replay",
        "teacher Pi0.5",
        X_FROM_HIT_RATE,
    ),
    "hit_act": (
        "retrieval_threshold_routing",
        "retrieval threshold -> ACT at hit",
        "ACT student (sidecar)",
        "teacher Pi0.5",
        X_FROM_HIT_RATE,
    ),
    "hit_smolvla": (
        "retrieval_threshold_routing",
        "retrieval threshold -> SmolVLA at hit",
        "SmolVLA student (sidecar)",
        "teacher Pi0.5",
        X_FROM_HIT_RATE,
    ),
    "miss_act": (
        "student_at_miss",
        "replay at hit + ACT at miss",
        "cache payload replay",
        "ACT student (sidecar)",
        X_NO_TEACHER,
    ),
    "miss_smolvla": (
        "student_at_miss",
        "replay at hit + SmolVLA at miss",
        "cache payload replay",
        "SmolVLA student (sidecar)",
        X_NO_TEACHER,
    ),
    "pure_act": (
        "pure_student",
        "pure ACT (no teacher steps)",
        "ACT student (gate always_skip)",
        "ACT student (gate always_skip)",
        X_NO_TEACHER,
    ),
    "pure_smolvla": (
        "pure_student",
        "pure SmolVLA (no teacher steps)",
        "SmolVLA student (gate always_skip)",
        "SmolVLA student (gate always_skip)",
        X_NO_TEACHER,
    ),
}


# ------------------------------------------------------------------
# Aggregation primitives
# ------------------------------------------------------------------
def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, the same construction analyze_ablation.py uses."""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def arm_outcomes(journal: pathlib.Path) -> dict[str, tuple[int, int]]:
    """Return ``{yaml_id: (n_episodes, n_success)}`` from a conductor journal.

    Failed episodes carry ``status: "failed"`` with ``success: false``; both
    statuses count towards the denominator, so the filter is on ``phase`` only.
    """
    n: dict[str, int] = defaultdict(int)
    k: dict[str, int] = defaultdict(int)
    for line in journal.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") != "eval":
            continue
        arm = row["yaml_id"]
        n[arm] += 1
        k[arm] += 1 if row.get("success") else 0
    return {arm: (n[arm], k[arm]) for arm in n}


def full_hit_rates(per_step: pathlib.Path) -> dict[str, float]:
    """FULL_HIT fraction of inference calls, per arm, from a per-step export."""
    total: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)
    for line in per_step.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        arm = row.get("yaml_id")
        total[arm] += 1
        if "FULL_HIT" in str(row.get("hit_type")):
            hits[arm] += 1
    return {arm: hits[arm] / total[arm] for arm in total if total[arm]}


def teacher_anchor(anchor_dir: pathlib.Path) -> tuple[int, int, list[str]]:
    """Aggregate the Phase-3 teacher anchor shards into ``(n, n_success, files)``."""
    files = sorted(anchor_dir.glob("results_tasks*.json"))
    if not files:
        raise SystemExit(f"{anchor_dir}: no results_tasks*.json anchor shards found")
    n = k = 0
    for path in files:
        for row in json.loads(path.read_text()):
            n += 1
            k += 1 if row["success"] else 0
    return n, k, [str(p) for p in files]


# ------------------------------------------------------------------
# Family assembly
# ------------------------------------------------------------------
def _point(
    arm: str,
    *,
    n: int,
    k: int,
    hit_rate: float | None,
    sr_source: str,
    sr: float | None = None,
    ci: list[float] | None = None,
    extra: dict | None = None,
) -> dict:
    series, label, hit_exec, miss_exec, basis = ARM_SEMANTICS.get(
        arm,
        (
            "kinematic_verdict_sweep",
            "kinematic verdict -> ACT at hit",
            "ACT student (sidecar)",
            "teacher Pi0.5",
            X_FROM_HIT_RATE,
        ),
    )
    if basis == X_FROM_HIT_RATE:
        if hit_rate is None:
            raise SystemExit(
                f"{arm}: abscissa needs a FULL_HIT rate but the per-step export "
                "carries none for this arm"
            )
        x = 1.0 - hit_rate
    elif basis == X_NO_TEACHER:
        x = 0.0
    else:
        x = 1.0
    point = {
        "arm": arm,
        "series": series,
        "label": label,
        "hit_slot_executor": hit_exec,
        "miss_slot_executor": miss_exec,
        "success_rate": sr if sr is not None else k / n,
        "success_rate_ci95": ci if ci is not None else list(wilson(k, n)),
        "success_rate_source": sr_source,
        "n_episodes": n,
        "n_success": k,
        "teacher_inference_rate": x,
        "teacher_inference_rate_basis": basis,
    }
    if hit_rate is not None:
        point["full_hit_rate"] = hit_rate
    point.update(extra or {})
    return point


def collect_family(
    suite: str,
    *,
    main_journal: pathlib.Path,
    main_per_step: pathlib.Path,
    sweep_journal: pathlib.Path,
    sweep_per_step: pathlib.Path,
    anchor_dir: pathlib.Path,
    paired_analysis: pathlib.Path | None = None,
) -> dict:
    """Assemble one suite's family block: main matrix + 4b sweep + teacher anchor.

    Main-matrix SR is taken from ``paired_analysis`` when given, and the
    aggregate derived from the journal must agree with it -- the check exists so
    a stale journal/analysis pair fails loudly instead of producing a figure
    that disagrees with the report's statistics.
    """
    main_counts = arm_outcomes(main_journal)
    main_hits = full_hit_rates(main_per_step)
    sweep_counts = arm_outcomes(sweep_journal)
    sweep_hits = full_hit_rates(sweep_per_step)

    upstream: dict[str, dict] = {}
    if paired_analysis is not None:
        upstream = json.loads(paired_analysis.read_text()).get("arms", {})

    points: list[dict] = []
    for index, arm in enumerate(MAIN_ARMS):
        if arm not in main_counts:
            raise SystemExit(f"{main_journal}: main-matrix arm {arm!r} absent")
        n, k = main_counts[arm]
        sr = ci = None
        source = "aggregated from journal"
        if arm in upstream:
            sr, ci = upstream[arm]["sr"], list(upstream[arm]["wilson_ci95"])
            source = "verbatim from analyze_ablation.py"
            if upstream[arm]["n"] != n or abs(sr - k / n) > 1e-9:
                raise SystemExit(
                    f"{arm}: analyze_ablation.py reports {upstream[arm]['sr']!r} over "
                    f"n={upstream[arm]['n']} but the journal aggregates to {k / n!r} "
                    f"over n={n}; the journal and the analysis JSON are out of sync"
                )
        points.append(
            _point(
                arm,
                n=n,
                k=k,
                hit_rate=main_hits.get(arm),
                sr_source=source,
                sr=sr,
                ci=ci,
                extra={"matrix_index": index},
            )
        )

    for order, tag in enumerate(SWEEP_TAGS):
        arm = f"kinroute_act_{tag}"
        if arm not in sweep_counts:
            raise SystemExit(f"{sweep_journal}: kinematic sweep arm {arm!r} absent")
        n, k = sweep_counts[arm]
        points.append(
            _point(
                arm,
                n=n,
                k=k,
                hit_rate=sweep_hits.get(arm),
                sr_source="aggregated from journal (no upstream analyzer for 4b)",
                extra={"threshold_tag": tag, "sweep_order": order},
            )
        )

    anchor_n, anchor_k, anchor_files = teacher_anchor(anchor_dir)
    sources = {
        "main_matrix_journal": {
            "path": str(main_journal),
            "sha256": _sha256(main_journal),
        },
        "main_matrix_per_step": {
            "path": str(main_per_step),
            "sha256": _sha256(main_per_step),
        },
        "kinematic_sweep_journal": {
            "path": str(sweep_journal),
            "sha256": _sha256(sweep_journal),
        },
        "kinematic_sweep_per_step": {
            "path": str(sweep_per_step),
            "sha256": _sha256(sweep_per_step),
        },
        "teacher_anchor_shards": anchor_files,
    }
    if paired_analysis is not None:
        sources["paired_analysis"] = {
            "path": str(paired_analysis),
            "sha256": _sha256(paired_analysis),
        }

    return {
        "suite": suite,
        "x_axis": "teacher_inference_rate",
        "x_axis_meaning": (
            "fraction of inference calls executed by the teacher's full Stage1-3 "
            "pipeline; owner-ruled axis, the historical warmup-cost axis is not used"
        ),
        "matrix_arm_order": list(MAIN_ARMS),
        "sweep_tag_order": list(SWEEP_TAGS),
        "teacher_anchor": {
            "arm": "teacher",
            "series": "teacher_anchor",
            "label": "teacher (all steps Pi0.5)",
            "success_rate": anchor_k / anchor_n,
            "success_rate_ci95": list(wilson(anchor_k, anchor_n)),
            "success_rate_source": "aggregated from Phase-3 anchor shards",
            "n_episodes": anchor_n,
            "n_success": anchor_k,
            "teacher_inference_rate": 1.0,
            "teacher_inference_rate_basis": X_ALL_TEACHER,
        },
        "sources": sources,
        "points": points,
    }


def load_data(path: pathlib.Path) -> dict:
    """Load an existing data file for merging, refusing cross-schema merges."""
    if not path.exists():
        return {"schema": SCHEMA, "families": {}}
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA:
        raise SystemExit(
            f"{path}: schema {data.get('schema')!r} != {SCHEMA}; refusing to merge "
            "across schema versions -- migrate the file first"
        )
    return data


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="plot_data.json to create or merge into")
    ap.add_argument("--suite", required=True, help="libero_spatial | libero_10")
    ap.add_argument("--main-journal", required=True)
    ap.add_argument("--main-per-step", required=True)
    ap.add_argument("--sweep-journal", required=True, help="4b kinematic sweep journal")
    ap.add_argument("--sweep-per-step", required=True)
    ap.add_argument("--anchor-dir", required=True,
                    help="data/anchors/<suite>_teacher/ holding results_tasks*.json")
    ap.add_argument("--paired-analysis", default=None,
                    help="analyze_ablation.py output; its arm SR/CI are copied verbatim")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    data = load_data(out)
    data["families"][args.suite] = collect_family(
        args.suite,
        main_journal=pathlib.Path(args.main_journal),
        main_per_step=pathlib.Path(args.main_per_step),
        sweep_journal=pathlib.Path(args.sweep_journal),
        sweep_per_step=pathlib.Path(args.sweep_per_step),
        anchor_dir=pathlib.Path(args.anchor_dir),
        paired_analysis=(pathlib.Path(args.paired_analysis)
                         if args.paired_analysis else None),
    )
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"{out}: {len(data['families'])} families "
          f"({', '.join(sorted(data['families']))})")


if __name__ == "__main__":
    main()
