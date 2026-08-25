#!/usr/bin/env python3
"""X15 U6 — the gate every evaluation launch must pass first.

    python launch_evaluation.py --ledger x15_init_ledger.json --pool a \
        --requires p_hat,tau_star --at 2026-08-24T09:00:00Z \
        -- python run_rl_router.py --run-id ...

The statistical protocol says p-hat and the tau grid are frozen before anything
on the evaluation side runs. Checking that at analysis time can only refuse to
quote a result that already exists; by then the episodes have been observed and
the parameters could have been chosen from them. So the check moves to the
launcher: nothing starts until the ledger shows the parameters frozen, and the
launcher records the first-touch stamp itself, once.

Ordering, not just presence, is what is enforced — a field can always be
back-filled, but ``freeze_parameter`` refuses to write after any evaluation pool
has been touched, and ``record_pool_touch`` refuses to revise a stamp. Between
them the ledger's freeze/touch sequence is auditable rather than asserted.

This is a WRAPPER, not a preflight check. Everything after ``--`` is the real
runner, and it is executed by this process only after the gate passes — so a
launch cannot be performed without the gate the way a separate "check first,
then run" step always can be. Without a command it still works as a bare check,
but the campaign runbook uses the wrapping form.

Exit codes: 0 = the wrapped runner's own exit code, 2 = refused (with reason).

Key dependency: ``exp.rl_router.analysis.cluster_stats`` (the guards).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from exp.rl_router.analysis.cluster_stats import (
    EVALUATION_POOLS,
    PreregistrationError,
    assert_pool_isolation,
    record_pool_touch,
)


def clear_for_launch(
    ledger_path: str,
    *,
    pool: str,
    requires: list[str],
    phase: str,
    at: str,
) -> dict:
    """Refuse or clear an evaluation launch, stamping first touch on success.

    The stamp is written last: a launch that fails the freeze check must not
    leave a touch record behind, or the pool would look consumed by a run that
    never happened.
    """
    path = pathlib.Path(ledger_path)
    if not path.exists():
        raise PreregistrationError(
            f"ledger {ledger_path!r} does not exist; the evaluation pools' "
            "freeze/touch history has to be recorded somewhere auditable"
        )
    ledger = json.loads(path.read_text(encoding="utf-8"))

    assert_pool_isolation(ledger, phase=phase, reads=pool)

    if pool in EVALUATION_POOLS:
        already = ledger.get("touched", {}).get(pool)
        if already:
            raise PreregistrationError(
                f"pool {pool!r} was already first touched at {already!r}. Each "
                "evaluation pool is measured once; a second launch would make "
                "the reported result a best-of-N."
            )
        for key in requires:
            # Checked BEFORE the stamp, so a refused launch leaves no trace.
            _assert_frozen_now(ledger, key, phase=phase, at=at)
        return record_pool_touch(ledger_path, pool, at=at)
    return ledger


def _assert_frozen_now(ledger: dict, key: str, *, phase: str, at: str) -> None:
    """Freeze check for a pool that has not been touched yet.

    ``assert_frozen_before`` compares against recorded touch stamps; at launch
    time the relevant stamp is the one about to be written, so the comparison
    uses ``at`` directly.
    """
    frozen = ledger.get("frozen", {})
    if key not in frozen:
        raise PreregistrationError(
            f"cannot launch {phase!r}: {key!r} is not frozen in the ledger. It "
            "must be committed from B-side data before the evaluation runs."
        )
    entry = frozen[key]
    if not isinstance(entry, dict) or "value" not in entry or "at" not in entry:
        raise PreregistrationError(
            f"frozen[{key!r}] must record both 'value' and 'at'"
        )
    if str(entry["at"]) >= str(at):
        raise PreregistrationError(
            f"{key!r} carries freeze time {entry['at']!r}, which is not before "
            f"this launch at {at!r}; the ordering it certifies would be false"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--pool", required=True, help="pool this launch will read")
    ap.add_argument("--phase", default="evaluate_a")
    ap.add_argument("--requires", default="p_hat",
                    help="comma-separated parameters that must already be frozen")
    ap.add_argument("--at", required=True, help="launch timestamp, ISO-8601 UTC")
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="the runner to execute once cleared, after `--`")
    args = ap.parse_args()

    try:
        clear_for_launch(
            args.ledger,
            pool=args.pool,
            requires=[k for k in args.requires.split(",") if k],
            phase=args.phase,
            at=args.at,
        )
    except PreregistrationError as exc:
        print(f"LAUNCH REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)

    command = [c for c in args.command if c != "--"]
    if not command:
        print(f"cleared: {args.phase} may read {args.pool} "
              f"(first touch stamped {args.at})")
        return
    print(f"cleared; launching: {' '.join(command)}")
    # Replacing this process keeps the runner's exit code, signals and stdio
    # exactly as if it had been invoked directly — the gate adds a
    # precondition, not a layer the operator has to reason about.
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
