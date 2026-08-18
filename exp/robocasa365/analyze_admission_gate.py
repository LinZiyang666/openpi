"""Admission-gate analysis for the RoboCasa365 cross-scene study.

Implements the criteria pre-registered in
`logs/benchmark_and_teacher_selection.log.md` §12-2, and adds the two-teacher
usable-task intersection that defines the experiment's evaluation task subset.

    P1  admission verdict : Wilson 95% lower bound of SR_B (all tasks pooled) > 20%
    P2  per-task class    : U0 (both arms 0) / U1 (exactly one arm 0) / U2 (both > 0)
    P3  reporting         : every task occupies a row, including U0 and errored arms

This is the version-controlled successor to `analyze_step0b.py`, which produced
the pi0.5 verdict but lived only on the run host. The statistics here are
byte-identical in convention to that script, so the numbers already recorded in
the selection log reproduce exactly; `--self-check` asserts that.

The input schema is teacher-agnostic -- both the pi0.5 harness and
`groot_rollout_client.py` write the same shape::

    {"sceneA": [l, s], "sceneB": [l, s], "n_trials": K,
     "tasks": {"<Task>": {"A": {"succ": int, "n": int, "wall_s": float},
                          "B": {...}}}}

An arm that failed carries ``{"error": ...}`` instead of ``succ``. Such an arm
is reported as ERR and excluded from the statistics -- it must never be folded
in as "0 successes", which would read an infrastructure fault as teacher
incompetence.

Usage::

    # one teacher
    python analyze_admission_gate.py --teacher pi05=/path/step0b_full.json

    # both, plus the usable-task intersection
    python analyze_admission_gate.py \
        --teacher pi05=/path/step0b_full.json \
        --teacher groot=/path/groot_gate_180ep.json

    # reproduce the pi0.5 numbers recorded in the selection log
    python analyze_admission_gate.py --teacher pi05=/path/step0b_full.json --self-check
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys
from typing import Any

Z = 1.96
CRASH_LINE = 0.20  # P1 threshold: SR_B Wilson lower bound must clear this

# Two-sided t critical values at 95%, keyed by degrees of freedom. Carried over
# verbatim from the pi0.5 analysis so the paired-gap CI stays comparable.
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 25: 2.060, 30: 2.042,
}

# The pi0.5 180-ep verdict as recorded in the selection log §12-2. --self-check
# recomputes these from the raw JSON; a mismatch means the log and the data have
# drifted apart and one of them is wrong.
PI05_RECORDED = {
    "sr_a": 37 / 90,
    "sr_b": 45 / 90,
    "wilson_b_lo": 0.399,
    "gap_pp": -8.9,
    "gap_ci_pp": (-20.3, 2.5),
    "u0": 4,
    "u1": 2,
    "u2": 12,
}


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval at 95%. Returns (lo, hi), or NaNs when n == 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (centre - half, centre + half)


def classify(a_succ: int, b_succ: int) -> str:
    """P2: U0 = both arms zero, U1 = exactly one arm zero, U2 = both non-zero."""
    if a_succ == 0 and b_succ == 0:
        return "U0"
    if a_succ == 0 or b_succ == 0:
        return "U1"
    return "U2"


def t_critical(df: int) -> float:
    """95% two-sided t critical value, nearest tabulated df (as in the original)."""
    return T95.get(df) or T95[min(T95, key=lambda key: abs(key - df))]


def load_run(path: pathlib.Path) -> dict[str, Any]:
    """Parse one gate result file into meta + per-task rows.

    Rows are emitted for every task in the file, errored arms included, so the
    P3 obligation (no task silently vanishes) holds at the data-structure level
    rather than depending on the formatter.
    """
    payload = json.loads(path.read_text())
    trials = payload["n_trials"]
    rows = []
    for name, arms in payload["tasks"].items():
        a, b = arms.get("A", {}), arms.get("B", {})
        if "succ" not in a or "succ" not in b:
            rows.append({
                "task": name,
                "a": a.get("succ"),
                "b": b.get("succ"),
                "n": trials,
                "cls": "ERR",
                "gap": None,
                "wall": (a.get("wall_s") or 0.0) + (b.get("wall_s") or 0.0),
                "error_a": a.get("error", ""),
                "error_b": b.get("error", ""),
            })
            continue
        rows.append({
            "task": name,
            "a": a["succ"],
            "b": b["succ"],
            "n": a["n"],
            "cls": classify(a["succ"], b["succ"]),
            "gap": (a["succ"] - b["succ"]) / a["n"],
            "wall": a.get("wall_s", 0.0) + b.get("wall_s", 0.0),
            "error_a": "",
            "error_b": "",
        })
    return {
        "path": path,
        "scene_a": tuple(payload["sceneA"]),
        "scene_b": tuple(payload["sceneB"]),
        "n_trials": trials,
        "rows": rows,
        "metadata": payload.get("server_metadata", {}),
    }


def pooled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled SR per arm over the analysable rows, plus the paired gap."""
    ok = [r for r in rows if r["cls"] != "ERR"]
    if not ok:
        return {"n": 0, "ok_rows": []}
    n = sum(r["n"] for r in ok)
    ka = sum(r["a"] for r in ok)
    kb = sum(r["b"] for r in ok)
    lo_a, hi_a = wilson(ka, n)
    lo_b, hi_b = wilson(kb, n)
    stats = {
        "n": n,
        "ok_rows": ok,
        "k_a": ka,
        "k_b": kb,
        "sr_a": ka / n,
        "sr_b": kb / n,
        "ci_a": (lo_a, hi_a),
        "ci_b": (lo_b, hi_b),
        "p1_pass": lo_b > CRASH_LINE,
    }
    gaps = [r["gap"] for r in ok]
    mean = sum(gaps) / len(gaps)
    stats["gap"] = mean
    if len(gaps) > 1:
        sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / (len(gaps) - 1))
        se = sd / math.sqrt(len(gaps))
        crit = t_critical(len(gaps) - 1)
        stats["gap_sd"] = sd
        stats["gap_ci"] = (mean - crit * se, mean + crit * se)
    return stats


def usable_tasks(rows: list[dict[str, Any]]) -> set[str]:
    """Tasks a teacher can actually do something on: class U1 or U2.

    U0 is excluded per the pre-registered P2. ERR is excluded too, but for a
    different reason -- unknown, not zero -- and the caller reports the two
    separately so an infrastructure failure cannot masquerade as a finding.
    """
    return {r["task"] for r in rows if r["cls"] in ("U1", "U2")}


def print_teacher_report(label: str, run: dict[str, Any],
                         expect_tasks: int | None = None) -> dict[str, Any]:
    rows = run["rows"]
    stats = pooled(rows)
    print(
        f"\n{'=' * 74}\n"
        f"=== {label}   scene A={run['scene_a']}  scene B={run['scene_b']}  "
        f"K={run['n_trials']}  tasks={len(rows)}"
    )
    if run["metadata"]:
        print(f"    server: {run['metadata']}")
    print("=" * 74)

    # P3: full per-task table, U0 and ERR rows included.
    print(f"\n{'task':<30} {'A':>6} {'B':>6} {'gap':>9}  class")
    print("-" * 64)
    order = {"U2": 0, "U1": 1, "U0": 2, "ERR": 3}
    for r in sorted(rows, key=lambda x: (order[x["cls"]], -(x["gap"] if x["gap"] is not None else -9))):
        a = f"{r['a']}/{r['n']}" if r["a"] is not None else "ERR"
        b = f"{r['b']}/{r['n']}" if r["b"] is not None else "ERR"
        gap = f"{r['gap'] * 100:>+8.1f}pp" if r["gap"] is not None else f"{'--':>10}"
        print(f"{r['task']:<30} {a:>6} {b:>6} {gap}  {r['cls']}")

    if stats["n"] == 0:
        print("\n  no analysable rows -- every arm errored")
        return {"stats": stats, "rows": rows}

    print(f"\n--- pooled (n={stats['n']} per arm) ---")
    print(f"  SR_A = {stats['k_a']}/{stats['n']} = {stats['sr_a']:.3f}"
          f"   95% CI [{stats['ci_a'][0]:.3f}, {stats['ci_a'][1]:.3f}]")
    print(f"  SR_B = {stats['k_b']}/{stats['n']} = {stats['sr_b']:.3f}"
          f"   95% CI [{stats['ci_b'][0]:.3f}, {stats['ci_b'][1]:.3f}]")
    if "gap_ci" in stats:
        print(f"  paired gap = {stats['gap'] * 100:+.1f}pp  SD={stats['gap_sd'] * 100:.1f}"
              f"  95% CI [{stats['gap_ci'][0] * 100:+.1f}, {stats['gap_ci'][1] * 100:+.1f}] pp"
              f"   (descriptive only -- not an admission criterion)")

    verdict = "PASS" if stats["p1_pass"] else "FAIL"
    print(f"\n--- P1 (admission) ---")
    print(f"  SR_B Wilson 95% lower bound = {stats['ci_b'][0]:.3f}"
          f"  vs crash line {CRASH_LINE:.2f}  =>  **{verdict}**")
    if expect_tasks and len(rows) < expect_tasks:
        # Results are written atomically per task, so a partial file is normal
        # mid-run -- and tasks run in the order they were listed, so a partial
        # verdict is biased by whichever tasks happen to come first, not by a
        # random sample. P1 is defined on the complete pre-registered set.
        print(f"  !! PARTIAL RUN: {len(rows)}/{expect_tasks} tasks. This verdict is")
        print(f"     NOT the admission decision -- tasks run in list order, so the")
        print(f"     subset is systematic, not random. Wait for all {expect_tasks}.")

    counts = collections.Counter(r["cls"] for r in rows)
    print(f"\n--- P2 (per-task class) ---")
    for cls in ("U0", "U1", "U2", "ERR"):
        if not counts[cls]:
            continue
        names = sorted(r["task"] for r in rows if r["cls"] == cls)
        print(f"  {cls:<3} {counts[cls]:2d}  {names}")
    if counts["U0"]:
        pfa = 0.9 ** (2 * run["n_trials"])
        print(f"  ! K={run['n_trials']}: a task whose true per-arm SR is 10% is "
              f"mislabelled U0 with probability {pfa:.2f}")

    errored = [r for r in rows if r["cls"] == "ERR"]
    if errored:
        print(f"\n--- errored arms ({len(errored)}) ---")
        for r in errored:
            print(f"  {r['task']}: A={r['error_a'][:60]} B={r['error_b'][:60]}")

    print(f"\n  wall clock = {sum(r['wall'] for r in rows) / 3600:.2f} h")
    return {"stats": stats, "rows": rows}


def intersect_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute the usable-task intersection across teachers.

    Kept separate from the printing so the rule that defines the formal
    experiment's evaluation task subset is testable on its own. Returns the intersection plus
    every caveat the caller must surface: scene-pair mismatches, tasks absent
    from a run, and the tasks only one teacher can do.
    """
    labels = list(reports)
    rows_of = {label: reports[label]["rows"] for label in labels}
    cls_of = {
        label: {r["task"]: r["cls"] for r in rows_of[label]} for label in labels
    }
    task_sets = {label: set(cls_of[label]) for label in labels}
    common = set.intersection(*task_sets.values())
    usable = {label: usable_tasks(rows_of[label]) for label in labels}
    # Counted over `common` so the per-teacher tally and the intersection share
    # a denominator; comparing a whole-run count against a common-task total
    # reads as nonsense when a run is incomplete.
    usable_common = {label: usable[label] & common for label in labels}
    keep = set.intersection(*usable.values()) & common

    scenes = {
        (reports[label]["run"]["scene_a"], reports[label]["run"]["scene_b"])
        for label in labels
    }
    trials = {reports[label]["run"]["n_trials"] for label in labels}
    # A task one teacher can do and the other cannot is evidence about the
    # teachers, not about cross-scene transfer -- worth calling out separately.
    split = sorted(
        task for task in common
        if len({task in usable[label] for label in labels}) > 1
    )
    return {
        "labels": labels,
        "cls_of": cls_of,
        "common": common,
        "missing": sorted(set().union(*task_sets.values()) - common),
        "usable": usable,
        "usable_common": usable_common,
        "keep": keep,
        "dropped": sorted(common - keep),
        "split": split,
        "scene_mismatch": scenes if len(scenes) > 1 else None,
        "n_trials": trials.pop() if len(trials) == 1 else None,
    }


def print_intersection(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Render the usable-task intersection -- the experiment's evaluation task subset.

    This rule is NOT part of the pre-registration, which covered a single
    teacher. P3 requires any exclusion introduced outside §12-2 to be recorded
    as a post-hoc decision, so the header says so in as many words.
    """
    result = intersect_reports(reports)
    labels = result["labels"]
    cls_of, common, keep = result["cls_of"], result["common"], result["keep"]

    print(f"\n{'=' * 74}\n=== usable-task intersection: {' x '.join(labels)}\n{'=' * 74}")
    print("  ! POST-HOC DECISION. The pre-registered P2 excludes U0 for one")
    print("    teacher; intersecting across teachers is an additional rule")
    print("    introduced after seeing the pi0.5 data, and P3 requires it to be")
    print("    recorded as such in the log.")

    if result["scene_mismatch"]:
        print(f"\n  !! teachers were evaluated on different scene pairs: "
              f"{result['scene_mismatch']}")
        print("     the intersection below is not comparable -- fix the runs first")

    if result["missing"]:
        print(f"\n  !! {len(result['missing'])} task(s) absent from at least one run: "
              f"{result['missing']}")
        print("     (an incomplete run, not a finding -- excluded from the table below)")

    print(f"\n{'task':<30} " + " ".join(f"{label:>8}" for label in labels) + "   keep")
    print("-" * (34 + 9 * len(labels) + 7))
    for task in sorted(common, key=lambda t: (t not in keep, t)):
        marks = " ".join(f"{cls_of[label][task]:>8}" for label in labels)
        print(f"{task:<30} {marks}   {'yes' if task in keep else 'no'}")

    print(f"\n--- evaluation task subset (of {len(common)} tasks run by both) ---")
    for label in labels:
        print(f"  usable under {label:<8} : {len(result['usable_common'][label]):2d}/{len(common)}")
    print(f"  intersection             : {len(keep):2d}/{len(common)}  {sorted(keep)}")

    if result["dropped"]:
        print(f"\n--- dropped ({len(result['dropped'])}) ---")
        for task in result["dropped"]:
            why = ", ".join(f"{label}={cls_of[label][task]}" for label in labels)
            print(f"  {task:<30} {why}")

    if result["split"]:
        print(f"\n--- teacher-specific competence ({len(result['split'])}) ---")
        print("    one teacher works on these, the other does not; they say")
        print("    something about the teachers, not about scene transfer")
        for task in result["split"]:
            why = "  ".join(f"{label}={cls_of[label][task]}" for label in labels)
            print(f"  {task:<30} {why}")

    if result["n_trials"]:
        k = result["n_trials"]
        per_teacher = 0.9 ** (2 * k)
        both = 1 - (1 - per_teacher) ** len(labels)
        print(f"\n  ! screening twice compounds the K={k} false-positive rate: a task")
        print(f"    whose true per-arm SR is 10% is dropped by at least one teacher")
        print(f"    with probability {both:.2f} (vs {per_teacher:.2f} for one teacher)")
    return result


def self_check(reports: dict[str, dict[str, Any]]) -> int:
    """Assert the pi0.5 numbers match what the selection log records."""
    if "pi05" not in reports:
        print("\n--self-check needs a teacher labelled 'pi05'", file=sys.stderr)
        return 2
    stats = reports["pi05"]["stats"]
    counts = collections.Counter(r["cls"] for r in reports["pi05"]["rows"])
    checks = [
        ("SR_A", stats["sr_a"], PI05_RECORDED["sr_a"], 1e-9),
        ("SR_B", stats["sr_b"], PI05_RECORDED["sr_b"], 1e-9),
        ("Wilson B lower", stats["ci_b"][0], PI05_RECORDED["wilson_b_lo"], 5e-4),
        ("gap pp", stats["gap"] * 100, PI05_RECORDED["gap_pp"], 0.05),
        ("gap CI lo pp", stats["gap_ci"][0] * 100, PI05_RECORDED["gap_ci_pp"][0], 0.05),
        ("gap CI hi pp", stats["gap_ci"][1] * 100, PI05_RECORDED["gap_ci_pp"][1], 0.05),
        ("U0", counts["U0"], PI05_RECORDED["u0"], 0),
        ("U1", counts["U1"], PI05_RECORDED["u1"], 0),
        ("U2", counts["U2"], PI05_RECORDED["u2"], 0),
    ]
    print(f"\n{'=' * 74}\n=== self-check against the selection log §12-2\n{'=' * 74}")
    bad = 0
    for name, got, want, tol in checks:
        ok = abs(got - want) <= tol
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {name:<16} got {got:>10.4f}  logged {want:>10.4f}")
    print(f"\n  {'all match' if not bad else f'{bad} MISMATCH -- log and data disagree'}")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="a gate result JSON, labelled. Repeat for the cross-teacher intersection.",
    )
    parser.add_argument(
        "--expect-tasks",
        type=int,
        default=18,
        help="pre-registered task count; a smaller run is flagged as partial (0 disables)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="assert the pi0.5 numbers reproduce those recorded in the selection log",
    )
    args = parser.parse_args()

    reports: dict[str, dict[str, Any]] = {}
    for spec in args.teacher:
        if "=" not in spec:
            parser.error(f"--teacher wants LABEL=PATH, got {spec!r}")
        label, _, raw = spec.partition("=")
        path = pathlib.Path(raw)
        if not path.exists():
            parser.error(f"no such file: {path}")
        run = load_run(path)
        report = print_teacher_report(label, run, args.expect_tasks)
        report["run"] = run
        reports[label] = report

    if len(reports) > 1:
        print_intersection(reports)

    if args.self_check:
        return self_check(reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
