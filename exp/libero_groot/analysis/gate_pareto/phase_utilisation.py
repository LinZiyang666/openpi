"""Did the fleet stay busy? Phase utilisation from the conductor's ledger.

A capacity change is judged on whether workers stopped idling, and that is not
answerable from a wall clock alone: a phase can halve its duration because the
episodes got easier. The quantity that isolates scheduling from everything else
is

    utilisation = sum(episode wall clock) / (workers x phase wall clock)

which is 1.0 when no worker ever waited and 1/N when one of N did all the work.
The single-arm phases this line cares about sat near 1/6 by construction: the
cell scheduler gave one arm to one slot.

Both terms come from the journal, which is the only per-episode artifact a run
leaves behind -- ``duration_s`` per terminal record, and the span of ``ts``
across the records selected. The health aggregator is in-memory and the monitor
renders a string, so neither survives the run.

    python phase_utilisation.py <journal.jsonl> --workers 48 [--arms a,b,...]

What the numerator is, exactly
------------------------------
``duration_s`` is written **per accepted terminal attempt**. Two classes of
real worker time are therefore *not* in it, and both are reported rather than
absorbed:

  * **Retriable failures.** ``WorkerLoop`` times every attempt including
    failures, but the driver journals only terminal results, and a retriable
    failure is not terminal -- it is requeued. The worker really was busy for
    that attempt; the ledger has no row for it.
  * **Fenced stale attempts.** A superseded dispatch that reports late *is*
    journaled, with ``accepted: false``. That worker was busy too, but the run
    did not use the result, so counting it would credit discarded work.

This module excludes the second class (via ``Journal.record_counts_as_done``)
and cannot see the first. Both omissions push the ratio **down**, so *when the
denominator is a measured phase wall clock* the figure is a lower bound on true
occupancy -- tight when retries are rare, loose when they are not, which is why
the fenced count is printed beside it.

⚠ **Without ``--phase-wall-clock-s`` there is no bound in either direction.**
The fallback denominator is the span between the first and last completion,
which omits the opening batch's runtime and so biases the ratio *up*, against
the numerator's downward bias. That case reports ``bound: "unknown"`` and must
not be used to pass or fail an acceptance threshold.

Caveats it reports rather than hides:
  * records without ``duration_s`` (written before the field existed) are
    counted in the denominator's span but not the numerator, which biases
    utilisation *down*; the count is printed so a reader can discount it.
  * ``ts`` is the terminal-record time, so the span starts at the *first
    completion*, not the first dispatch. For a phase of many short episodes the
    difference is one episode; it is reported as ``span_note``.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

from openpi.conductor.journal import Journal as _Journal


def load_terminal_records(path: pathlib.Path) -> tuple[list[dict], int]:
    """Accepted terminal records, plus the count of fenced ones excluded.

    The fenced count is returned rather than dropped because it is the reader's
    only signal for how loose the lower bound is: many rejections mean a lot of
    real worker time is missing from the numerator.
    """
    rows: list[dict] = []
    fenced = 0
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") not in ("done", "failed"):
                continue
            if _Journal.record_counts_as_done(rec):
                rows.append(rec)
            else:
                fenced += 1
    return rows, fenced


def utilisation(
    records: list[dict], *, workers: int, phase_wall_clock_s: float | None = None
) -> dict:
    """Busy worker-seconds over available worker-seconds, plus its inputs.

    ``phase_wall_clock_s`` is the measured duration of the phase. Supply it and
    the ratio is a genuine lower bound on occupancy: the denominator is then the
    real capacity offered, and the numerator omits only work the ledger cannot
    see (retriable attempts) or must not credit (fenced ones), both of which
    push the ratio *down*.

    Omit it and the fallback is the span between the first and last terminal
    record -- which is **not** the phase duration: it starts at the first
    *completion*, so it misses however long the opening batch ran, and the
    denominator comes out short. That bias pushes the ratio *up*, against the
    numerator's downward bias, so the result has no known direction and can
    exceed 1. It is reported as ``bound: "unknown"`` and must not be used to
    fail an acceptance threshold.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if not records:
        return {"episodes": 0, "utilisation": None, "bound": "unknown"}
    durations = [r["duration_s"] for r in records if r.get("duration_s") is not None]
    busy = sum(durations)

    if phase_wall_clock_s is not None:
        if phase_wall_clock_s <= 0:
            raise ValueError(f"phase_wall_clock_s must be > 0, got {phase_wall_clock_s}")
        span = float(phase_wall_clock_s)
        bound = "lower"
    else:
        stamps = [r["ts"] for r in records if "ts" in r]
        span = (max(stamps) - min(stamps)) if len(stamps) > 1 else 0.0
        bound = "unknown"

    available = workers * span
    ratio = (busy / available) if available else None
    if ratio is not None and bound == "lower" and ratio > 1.0:
        # A true wall clock cannot yield more busy time than capacity offered.
        # Getting here means an input is wrong (wrong worker count, wrong phase
        # window, durations from another run) -- reporting it as occupancy would
        # launder that into a plausible-looking number.
        raise ValueError(
            f"occupancy {ratio:.3f} exceeds 1.0 with a measured wall clock: "
            f"{busy:.1f} busy worker-seconds over {workers} x {span:.1f}s. "
            "Check the worker count and the phase window."
        )
    return {
        "episodes": len(records),
        "episodes_timed": len(durations),
        "episodes_untimed": len(records) - len(durations),
        "phase_wall_clock_s": round(span, 1),
        "wall_clock_source": "measured" if phase_wall_clock_s is not None else "terminal-record span",
        "busy_worker_s": round(busy, 1),
        "available_worker_s": round(available, 1),
        "utilisation": ratio,
        "bound": bound,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("journal")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--arms", default="", help="comma-separated yaml_ids; default: all")
    ap.add_argument("--per-arm", action="store_true", help="also break down by arm")
    ap.add_argument(
        "--phase-wall-clock-s",
        type=float,
        default=None,
        help="measured phase duration. Without it the figure has no known "
        "direction and cannot be used to fail an acceptance threshold.",
    )
    args = ap.parse_args()

    records, fenced = load_terminal_records(pathlib.Path(args.journal))
    if args.arms:
        wanted = set(args.arms.split(","))
        records = [r for r in records if r.get("yaml_id") in wanted]

    overall = utilisation(
        records, workers=args.workers, phase_wall_clock_s=args.phase_wall_clock_s
    )
    overall["fenced_records_excluded"] = fenced
    print(json.dumps({"overall": overall}, indent=2))
    if overall.get("bound") == "unknown":
        print(
            "note: no measured phase wall clock, so the denominator is the span "
            "between first and last completion -- it misses the opening batch's "
            "runtime and biases the ratio UP, against the numerator's downward "
            "bias. Direction unknown; do not gate on this value."
        )
    if fenced:
        tail = (
            "so the figure above is a lower bound"
            if overall.get("bound") == "lower"
            else "which biases the figure down, while the fallback denominator "
            "biases it up -- hence bound: unknown"
        )
        print(
            f"note: {fenced} fenced (accepted:false) record(s) excluded, and "
            f"retriable attempts are not journaled at all -- both were real "
            f"worker time, {tail}"
        )
    if overall.get("episodes_untimed"):
        print(
            f"note: {overall['episodes_untimed']} record(s) carry no duration_s "
            "and are excluded from busy time, which biases utilisation down"
        )

    if args.per_arm:
        by_arm: dict[str, list[dict]] = collections.defaultdict(list)
        for rec in records:
            by_arm[rec.get("yaml_id", "?")].append(rec)
        for arm in sorted(by_arm):
            point = utilisation(by_arm[arm], workers=args.workers)
            u = point["utilisation"]
            print(f"{arm:20s} episodes={point['episodes']:5d} "
                  f"wall={point['phase_wall_clock_s']:8.1f}s "
                  f"util={'n/a' if u is None else f'{u:.3f}'}")


if __name__ == "__main__":
    main()
