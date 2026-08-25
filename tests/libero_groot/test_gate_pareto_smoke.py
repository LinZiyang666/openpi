"""Structural smoke gate for the first real gate run on GR00T.

The hysteresis gate has never run against this executor. What has to be
established before committing 17,200 episodes is that the mechanism is
*connected*: the gate decides, its skips reach the teacher, and the evidence
recording all of it survives the trip back.

Why the assertions are shaped the way they are
----------------------------------------------
*Direction is not asserted.* Whether a low ``f_FH`` arm shows more teacher calls
than a high one is a result, reported from the 500-episode paired arms. On the
tens of episodes a smoke run can afford the comparison has almost no power --
rollouts diverge the moment one arm replays a cached chunk the other did not --
so asserting it would reject a correct implementation on noise.

*Gate-skip presence is asserted per arm only where it is mechanically
guaranteed.* The V2 injection fires after ``L`` consecutive **FULL_HIT**
verdicts, so it depends on the judge: under a strict threshold the run counter
resets constantly and injection never happens, and N1 additionally only skips
after ``j`` consecutive sub-theta scores. A sweep arm with zero gate skips is
therefore legitimate, not broken. The **gate-only** arm is the opposite case --
its judge accepts everything, so every searched step is a FULL_HIT and
injection is guaranteed within ``L + 1`` decisions. That arm is required to be
in the smoke set, and it is where gate-skip presence is demanded.

Run against a completed smoke run:

    GP_SMOKE_RESULTS_DIR=... GP_SMOKE_PER_STEP_DIR=... GP_SMOKE_EXPECT_EP=20 \\
        GP_SMOKE_ARMS=gpgo_sp,gp_sp_fh80 \\
        uv run pytest tests/libero_groot/test_gate_pareto_smoke.py --run-manual
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from exp.libero_groot.analysis.gate_pareto.analyze_gate_pareto import (
    IntegrityError,
    _read_jsonl,
    aggregate_arm,
    check_arm_integrity,
)

pytestmark = pytest.mark.manual

_ENV = ("GP_SMOKE_RESULTS_DIR", "GP_SMOKE_PER_STEP_DIR", "GP_SMOKE_ARMS")


def _arms() -> dict[str, tuple]:
    """Load the smoke run, or fail.

    Skipping is only correct when the harness was never pointed at a run at
    all. Once it has been, incomplete data is the failure this gate exists to
    catch -- a skip there would report green for a run that produced nothing.
    """
    missing_env = [name for name in _ENV if not os.environ.get(name)]
    if len(missing_env) == len(_ENV):
        pytest.skip(f"set {', '.join(_ENV)} to a completed smoke run")
    if missing_env:
        pytest.fail(f"smoke run is half-configured; missing {missing_env}")

    results_dir = pathlib.Path(os.environ["GP_SMOKE_RESULTS_DIR"])
    per_step_dir = pathlib.Path(os.environ["GP_SMOKE_PER_STEP_DIR"])
    expect = int(os.environ.get("GP_SMOKE_EXPECT_EP", "20"))
    wanted = [a for a in os.environ["GP_SMOKE_ARMS"].split(",") if a]

    if len(wanted) < 2:
        pytest.fail(f"smoke needs at least two arms, got {wanted}")
    if not any(a.startswith("gpgo_") for a in wanted):
        pytest.fail(
            f"smoke set {wanted} has no gate-only arm; it is the one arm whose "
            "gate-skip behaviour is mechanically guaranteed, so without it the "
            "gate could be inert and every assertion here would still pass"
        )

    out = {}
    for arm in wanted:
        results_path = results_dir / f"{arm}.json"
        step_path = per_step_dir / f"{arm}.jsonl"
        merge_path = per_step_dir / f"{arm}.merge.json"
        for path in (results_path, step_path, merge_path):
            if not path.is_file():
                pytest.fail(f"{arm}: smoke evidence missing: {path}")
        out[arm] = (
            json.loads(results_path.read_text(encoding="utf-8")),
            _read_jsonl(step_path),
            json.loads(merge_path.read_text(encoding="utf-8")),
            expect,
        )
    return out


def test_every_smoke_arm_passes_the_integrity_gate():
    for arm, (results, rows, merge, expect) in _arms().items():
        try:
            check_arm_integrity(
                results, rows, expect_ep=expect, merge=merge, arm=arm
            )
        except IntegrityError as exc:
            pytest.fail(str(exc))


def test_every_smoke_arm_produced_legal_verdicts():
    legal = {"FULL_HIT", "MISS"}
    for arm, (_, rows, _, _) in _arms().items():
        seen = {r["hit_type"] for r in rows if r.get("hit_type") is not None}
        assert seen, f"{arm}: no verdict rows at all"
        assert seen <= legal, f"{arm}: unexpected verdicts {seen - legal}"


def test_no_arm_is_stuck_shut():
    # A gate that never searches again after its first lockout produces a run
    # that looks like a very expensive teacher-only baseline, with no error.
    for arm, (_, rows, _, _) in _arms().items():
        decisions = [r for r in rows if r.get("hit_type") is not None]
        searched = [r for r in decisions if r.get("searched") is not False]
        assert searched, f"{arm}: every decision was a gate skip (stuck shut)"


def test_the_gate_only_arm_actually_skipped():
    # With the verdict disabled every searched step is a FULL_HIT, so the V2
    # injection must fire within L+1 decisions. Zero skips here means the gate
    # is wired but inert -- the failure mode that would silently collapse every
    # Pareto point onto the judge's own curve.
    for arm, (_, rows, _, _) in _arms().items():
        if not arm.startswith("gpgo_"):
            continue
        skips = sum(1 for r in rows if r.get("searched") is False)
        assert skips > 0, (
            f"{arm}: gate-only arm produced no gate-skip rows; with every "
            "search accepted the V2 injection is guaranteed, so the gate is "
            "not actually intervening"
        )


def test_every_arm_reports_a_ratio_in_range():
    for arm, (results, rows, _, expect) in _arms().items():
        point = aggregate_arm(results, rows, expect_ep=expect, arm=arm)
        assert point["teacher_ratio"] is not None, f"{arm}: no decisions at all"
        assert 0.0 <= point["teacher_ratio"] <= 1.0
