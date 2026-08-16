"""Per-episode FULL_HIT evidence, joined on ``(task_uid, accepted attempt)``.

Shared by the runner's exit gate and the analyzer so the two cannot drift: an
arm-level FULL_HIT ratio is a different contract from "every one of the 500
accepted episodes was served entirely from cache", and only the latter licenses
the pure-cache claim. One FULL_HIT row would give a perfect arm-level rate.

The attempt half of the key is not decoration. ``ConductorDriver.handle_result``
writes per-step rows for **stale** attempts exactly as it writes them for the
live one, stamping each row with its ``attempt``; the scheduler's ``accepted``
verdict is what distinguishes them, and it lives in the journal, not here. Join
on the uid alone and a stale attempt's rows stand in as the witness for an
accepted episode that produced none -- or a stale MISS condemns an accepted run
that was entirely cache-served.
"""

from __future__ import annotations

import json
import pathlib


def load_per_episode_hits(
    per_step_path: str | pathlib.Path,
    *,
    require_attempt: bool = True,
) -> tuple[dict[str, dict[tuple[str, int | None], list[str]]], dict[str, int]]:
    """``({yaml_id: {(task_uid, attempt): [hit_type, ...]}}, {arm: row_count})``.

    Rejects a repeated ``(task_uid, attempt, step_idx)`` outright rather than
    keeping either copy: the runner's canonical merge already de-duplicates, so a
    survivor means two different rows claim the same step, and picking one would
    fabricate the trace.
    """
    out: dict[str, dict[tuple[str, int | None], list[str]]] = {}
    seen: set[tuple[str, str, int | None, int | None]] = set()
    rows: dict[str, int] = {}
    with open(per_step_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            uid = rec.get("task_uid")
            hit = rec.get("hit_type")
            if uid is None or hit is None:
                continue
            arm = rec.get("yaml_id") or uid.rsplit(":", 3)[0]
            attempt = rec.get("attempt")
            step = rec.get("step_idx")
            if require_attempt and attempt is None:
                raise SystemExit(
                    f"{arm}: per-step row for {uid} carries no 'attempt'. The driver "
                    "stamps every row it forwards, so an unstamped row cannot be "
                    "matched against the accepted attempt and may be stale."
                )
            if require_attempt and step is None:
                raise SystemExit(
                    f"{arm}: per-step row for {uid} carries no 'step_idx'; duplicate "
                    "rows for one step could not be detected"
                )
            ident = (arm, uid, attempt, step)
            if step is not None and ident in seen:
                raise SystemExit(
                    f"{arm}: duplicate per-step row for {uid} attempt={attempt} "
                    f"step={step}. Two rows claim the same step; the trace is ambiguous."
                )
            seen.add(ident)
            out.setdefault(arm, {}).setdefault((uid, attempt), []).append(hit)
            rows[arm] = rows.get(arm, 0) + 1
    return out, rows


def assert_full_hit_per_episode(
    arm: str,
    accepted_attempts: dict[str, int | None],
    hits: dict[tuple[str, int | None], list[str]],
) -> dict:
    """Every accepted episode needs a non-empty trace *at its accepted attempt*.

    Rows at other attempts are filtered out, not treated as errors -- a retried
    episode legitimately leaves them behind -- but they can never stand in for
    the accepted attempt's own evidence.
    """
    joined: dict[str, list[str]] = {}
    stale_rows = 0
    unknown = set()
    for (uid, attempt), trace in hits.items():
        if uid not in accepted_attempts:
            unknown.add(uid)
            continue
        if attempt != accepted_attempts[uid]:
            stale_rows += len(trace)
            continue
        joined[uid] = trace

    if unknown:
        raise SystemExit(
            f"{arm}: {len(unknown)} per-step episode(s) are absent from the accepted "
            f"ledger (e.g. {sorted(unknown)[:5]}); evidence for episodes this run "
            "never accepted cannot be part of its result"
        )

    missing = sorted(set(accepted_attempts) - set(joined))
    if missing:
        stale_only = [u for u in missing
                      if any(k[0] == u for k in hits)]
        detail = ""
        if stale_only:
            detail = (f" {len(stale_only)} of them do have rows at a *different* "
                      f"attempt (e.g. {stale_only[:3]}) -- stale evidence, which "
                      "cannot witness the accepted run.")
        raise SystemExit(
            f"{arm}: {len(missing)} accepted episode(s) have no inference rows at "
            f"their accepted attempt (e.g. {missing[:5]})." + detail
        )

    empty = sorted(uid for uid, h in joined.items() if not h)
    if empty:
        raise SystemExit(f"{arm}: {len(empty)} episode(s) have an empty trace")

    offenders = {uid: sorted(set(h)) for uid, h in joined.items()
                 if any(x != "FULL_HIT" for x in h)}
    if offenders:
        sample = dict(list(offenders.items())[:5])
        raise SystemExit(
            f"{arm}: {len(offenders)} episode(s) served non-FULL_HIT steps "
            f"(e.g. {sample}). Retrieval is task-scoped, so this means some task had "
            "no library entries and fell back to the teacher, inflating the arm's SR."
        )
    return {
        "episodes": len(joined),
        "steps": sum(len(h) for h in joined.values()),
        "stale_rows_ignored": stale_rows,
    }
