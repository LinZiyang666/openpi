"""Aggregate the gate-threshold sweep into Pareto points, and plot the frontier.

Two subcommands, split by where they can run:

``aggregate``
    Pure standard library, so it runs on the client node next to the raw
    per-step JSONL (tens of MB per arm -- aggregate there, move the summary).
    **Every arm passes a fail-closed integrity gate before a number is
    computed**; see :func:`check_arm_integrity`.

``plot``
    Needs matplotlib, so it runs wherever the figures are wanted.

Why the integrity gate is not optional
--------------------------------------
``teacher_ratio`` is this experiment's x-axis and its denominator is the number
of gate decisions, which is read from the per-step evidence. If some of that
evidence is missing, the ratio is still a finite number in a plausible range --
the Pareto point simply moves, with nothing anywhere reporting a failure. A row
count cannot establish the denominator's provenance either: one worker's
per-step file can disappear entirely while other, longer episodes push the
total past any threshold a reader would think to set. The gate therefore checks
*set equality of episode identities* between the results side and the per-step
side, which is the property that actually has to hold.

Only ``inference_ratio`` is deliberately absent. The pi0.5 line maps teacher
ratio onto it with an affine constant derived from that model's CUDA-Graph
stage split; GR00T's stage split has not been measured, and reusing a Pi0.5
constant would produce a second axis that looks quantitative and is not.

Public interface: ``IntegrityError``, ``check_arm_integrity``, ``aggregate``,
``pareto_front``, ``plot_teacher_pareto``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib

#: Rows carrying this key are gate decisions; rows without it are not verdicts.
_HIT_TYPE = "hit_type"


class IntegrityError(RuntimeError):
    """One arm's evidence is incomplete; no number may be derived from it."""


# ------------------------------------------------------------------
# Integrity gate (I1-I6)
# ------------------------------------------------------------------


def check_arm_integrity(
    results: list[dict],
    per_step: list[dict],
    *,
    expect_ep: int,
    merge: dict,
    arm: str = "",
) -> None:
    """Raise unless this arm's evidence supports a teacher ratio.

    Args:
        results: rows from the merged ``--save-episode-results`` json.
        per_step: rows from the merged per-step jsonl.
        expect_ep: episodes this arm was supposed to run.
        merge: the merge sidecar written beside the per-step file.
            **Required, with no default**: ``lanes_found < lanes_expected`` is
            the one failure a row count provably cannot see, so a caller that
            omitted the sidecar would silently be running a weaker gate than
            the one this function advertises. An absent sidecar is itself
            missing evidence and must be raised by the caller.
        arm: name, for the error message only.

    Raises:
        IntegrityError: naming the violated invariant.
    """
    where = f"{arm}: " if arm else ""

    # I6 first: a missing worker file explains every downstream mismatch, and
    # reporting the mismatch instead would send the reader to the wrong place.
    expected = merge.get("lanes_expected")
    found = merge.get("lanes_found")
    if expected is None or found is None:
        raise IntegrityError(
            f"{where}merge sidecar lacks lanes_expected/lanes_found; cannot "
            "establish that every worker contributed"
        )
    if found != expected:
        raise IntegrityError(
            f"{where}per-step merge saw {found} of {expected} worker files. "
            "The missing worker's episodes are absent from the denominator; "
            "a row count would not have shown this."
        )

    # I1: the arm ran the episodes it claims.
    if len(results) != expect_ep:
        raise IntegrityError(
            f"{where}{len(results)} episode results, expected {expect_ep}"
        )

    # I2: no episode counted twice on the results side.
    keys = [(row["task_id"], row["orig_init_state_idx"]) for row in results]
    duplicates = [k for k, n in collections.Counter(keys).items() if n > 1]
    if duplicates:
        raise IntegrityError(
            f"{where}duplicate (task_id, orig_init_state_idx) in results: "
            f"{sorted(duplicates)[:5]}"
        )

    # I3: the two sides describe the same episodes -- the core invariant.
    result_ids = {row["episode_id"] for row in results}
    step_ids = {row["episode_id"] for row in per_step}
    missing = result_ids - step_ids
    extra = step_ids - result_ids
    if missing or extra:
        raise IntegrityError(
            f"{where}per-step episode set != results episode set "
            f"(missing {len(missing)}, extra {len(extra)}; "
            f"e.g. missing={sorted(missing)[:3]} extra={sorted(extra)[:3]})"
        )

    # I4: every episode contributed at least one decision.
    with_verdict = {
        row["episode_id"] for row in per_step if row.get(_HIT_TYPE) is not None
    }
    silent = result_ids - with_verdict
    if silent:
        raise IntegrityError(
            f"{where}{len(silent)} episode(s) have no verdict row: "
            f"{sorted(silent)[:5]}"
        )

    # I5: no decision counted twice.
    step_keys = [(row["episode_id"], row["step_idx"]) for row in per_step]
    dup_steps = [k for k, n in collections.Counter(step_keys).items() if n > 1]
    if dup_steps:
        raise IntegrityError(
            f"{where}duplicate (episode_id, step_idx) rows: {sorted(dup_steps)[:5]}"
        )


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_arm(
    results: list[dict], per_step: list[dict], *, expect_ep: int, arm: str = ""
) -> dict:
    """One arm's Pareto point. Assumes :func:`check_arm_integrity` has passed."""
    decisions = [row for row in per_step if row.get(_HIT_TYPE) is not None]
    total = len(decisions)
    # A gate-skipped step (searched=False) is a MISS: the teacher ran. Counting
    # only judge-rejected steps would understate the x-axis by exactly the
    # gate's own intervention rate, which is the quantity this line studies.
    miss = sum(1 for row in decisions if row[_HIT_TYPE] == "MISS")
    return {
        "n_ep": len(results),
        "success_rate": sum(1 for row in results if row["success"]) / len(results),
        "teacher_ratio": (miss / total) if total else None,
        "decisions": total,
        "misses": miss,
        "arm": arm,
    }


def arms_from_config(yaml_dir: pathlib.Path) -> set[str]:
    """The arms a phase was supposed to run, read off its emitted recipes.

    The emitted YAML set is the authoritative statement of intent: it is what
    the scheduler was pointed at, so an arm that has a recipe and no result is
    exactly the "never ran" case that enumerating results cannot see.
    """
    arms = {p.stem for p in yaml_dir.glob("*.yaml")}
    if not arms:
        raise IntegrityError(f"no arm recipes in {yaml_dir}")
    return arms


def aggregate(
    results_dir: pathlib.Path,
    per_step_dir: pathlib.Path,
    *,
    expect_ep: int,
    expect_arms: set[str],
) -> dict:
    """Aggregate exactly ``expect_arms``, failing closed on anything missing.

    Enumerating whatever results happen to exist cannot detect a whole arm that
    never produced one -- the summary simply comes back one point smaller, and
    a frontier drawn through the remaining points looks entirely reasonable.
    The expected set is therefore an input, not something inferred from the
    directory being checked.
    """
    if not expect_arms:
        raise IntegrityError("expect_arms is empty; refusing to aggregate nothing")

    present = {
        p.stem
        for p in results_dir.glob("*.json")
        if not p.name.endswith(".partial.json")
    }
    missing = expect_arms - present
    extra = present - expect_arms
    if missing:
        raise IntegrityError(
            f"{len(missing)} arm(s) produced no complete results file: "
            f"{sorted(missing)}"
        )
    if extra:
        raise IntegrityError(
            f"unexpected result files not in the phase's arm set: {sorted(extra)}"
        )

    out = {}
    for arm in sorted(expect_arms):
        step_path = per_step_dir / f"{arm}.jsonl"
        if not step_path.is_file():
            raise IntegrityError(f"{arm}: per-step file missing: {step_path}")
        merge_path = per_step_dir / f"{arm}.merge.json"
        if not merge_path.is_file():
            # Without it there is no evidence that every worker contributed,
            # and I6 would be silently skipped for this arm.
            raise IntegrityError(f"{arm}: merge sidecar missing: {merge_path}")
        results = json.loads((results_dir / f"{arm}.json").read_text(encoding="utf-8"))
        per_step = _read_jsonl(step_path)
        check_arm_integrity(
            results,
            per_step,
            expect_ep=expect_ep,
            merge=json.loads(merge_path.read_text(encoding="utf-8")),
            arm=arm,
        )
        out[arm] = aggregate_arm(results, per_step, expect_ep=expect_ep, arm=arm)
    return out


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points not dominated by another with lower x and higher y."""
    ordered = sorted(points)
    front: list[tuple[float, float]] = []
    best = float("-inf")
    for x, y in ordered:
        if y > best:
            front.append((x, y))
            best = y
    return front


def _f_fh_of(arm: str) -> float | None:
    """``gp_sp_fh35`` -> 0.35; anything else -> None."""
    marker = "_fh"
    if marker not in arm:
        return None
    try:
        return int(arm.rsplit(marker, 1)[1]) / 100.0
    except ValueError:
        return None


def plot_teacher_pareto(
    suite: str,
    arms: dict,
    out_dir: pathlib.Path,
    *,
    gate_only: dict | None = None,
    status: str = "",
) -> pathlib.Path:
    """Scatter the sweep, draw its Pareto front, star the gate-only point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [
        (v["teacher_ratio"], v["success_rate"], _f_fh_of(k))
        for k, v in sorted(arms.items())
        if v["teacher_ratio"] is not None and _f_fh_of(k) is not None
    ]
    if not points:
        raise SystemExit(f"{suite}: no sweep arms to plot")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        [p[0] for p in points], [p[1] for p in points],
        s=30, color="#1f77b4", alpha=0.45, zorder=2, label="sweep cell",
    )
    front = pareto_front([(p[0], p[1]) for p in points])
    # Markers as well as a line: a frontier can legitimately collapse to a
    # single point (one arm dominating every other), and a bare line would then
    # draw nothing while the legend still claimed one.
    ax.plot(
        [p[0] for p in front], [p[1] for p in front],
        color="#1f77b4", linewidth=1.8, marker="o", markersize=5,
        zorder=3, label="Pareto front",
    )
    for x, y, f_fh in points:
        ax.annotate(f"{f_fh:.2f}", (x, y), fontsize=7, alpha=0.65,
                    xytext=(3, 3), textcoords="offset points")
    if gate_only:
        ax.scatter(
            [gate_only["teacher_ratio"]], [gate_only["success_rate"]],
            marker="*", s=220, color="#d62728", zorder=4,
            label="gate-only (verdict disabled)",
        )

    ax.set_xlabel("teacher ratio  (MISS decisions / all decisions)")
    ax.set_ylabel("success rate")
    ax.set_title(f"GR00T N1.5 x {suite} — gate-threshold Pareto")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    if status:
        fig.text(0.01, 0.01, status, fontsize=7, alpha=0.6)
    fig.tight_layout()

    png = out_dir / f"pareto_{suite}.png"
    fig.savefig(png, dpi=160)
    fig.savefig(out_dir / f"pareto_{suite}.pdf")
    plt.close(fig)
    return png


# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------


def write_manifest(out_dir: pathlib.Path, sources: dict[str, str]) -> pathlib.Path:
    """Digest every generated file in ``out_dir`` and record where it came from.

    A figure with no recorded provenance is an orphan: nothing connects it to
    the run that produced it, and the next reader cannot tell a current plot
    from a stale one left over from an earlier sweep. ``sources`` maps a suite
    to the aggregate json its points were drawn from.
    """
    files = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        files.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {"sources": sources, "files": files}
    out = out_dir / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gate-threshold Pareto analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)

    agg = sub.add_parser("aggregate", help="integrity-check + aggregate (stdlib only)")
    agg.add_argument("results_dir")
    agg.add_argument("per_step_dir")
    agg.add_argument("out_json")
    agg.add_argument("--expect-ep", type=int, required=True)
    # One of these is required: the phase's arm set has to come from outside
    # the directory being checked, or a missing arm is undetectable.
    agg.add_argument("--arms-from", default="",
                     help="directory of emitted arm YAMLs (authoritative set)")
    agg.add_argument("--expect-arms", default="",
                     help="comma-separated arm names, if no YAML dir is handy")

    plot = sub.add_parser("plot", help="teacher-rate Pareto figure")
    plot.add_argument("--suite", action="append", required=True,
                      help="suite=<aggregate.json>, repeatable")
    plot.add_argument("--out-dir", required=True)
    plot.add_argument("--status", default="")

    args = ap.parse_args(argv)

    if args.cmd == "aggregate":
        if bool(args.arms_from) == bool(args.expect_arms):
            raise SystemExit(
                "pass exactly one of --arms-from / --expect-arms: the expected "
                "arm set must come from outside the results directory, or an "
                "arm that never ran cannot be detected"
            )
        expect_arms = (
            arms_from_config(pathlib.Path(args.arms_from))
            if args.arms_from
            else {a for a in args.expect_arms.split(",") if a}
        )
        summary = aggregate(
            pathlib.Path(args.results_dir),
            pathlib.Path(args.per_step_dir),
            expect_ep=args.expect_ep,
            expect_arms=expect_arms,
        )
        pathlib.Path(args.out_json).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        for arm, row in sorted(summary.items()):
            print(f"{arm:18s} sr={row['success_rate']:.3f} "
                  f"tr={row['teacher_ratio']:.3f} n={row['decisions']}")
        print(f"\nwrote {args.out_json}")
        return 0

    out_dir = pathlib.Path(args.out_dir)
    plot_data = {}
    sources = {}
    for spec in args.suite:
        suite, _, path = spec.partition("=")
        sources[suite] = path
        arms = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        gate_only_arms = {k: v for k, v in arms.items() if k.startswith("gpgo_")}
        sweep = {k: v for k, v in arms.items() if k.startswith("gp_")}
        gate_only = next(iter(gate_only_arms.values()), None)
        png = plot_teacher_pareto(
            suite, sweep, out_dir, gate_only=gate_only, status=args.status
        )
        plot_data[suite] = {"arms": arms, "figure": png.name}
        print(f"{suite}: {png}")

    (out_dir / "plot_data.json").write_text(
        json.dumps(
            {"status": args.status, "x_axis": "teacher_ratio", "suites": plot_data},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = write_manifest(out_dir, sources)
    print(f"manifest: {manifest} ({len(json.loads(manifest.read_text())['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
