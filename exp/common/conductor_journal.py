"""Accepted-aware reader for the conductor journal.

``Journal.record`` may write several terminal rows for one ``task_uid``: a stale
attempt from a superseded dispatch is journaled *exactly like* the live one, and
is distinguished only by the optional ``attempt`` / ``accepted`` fields. A
plain last-wins read therefore lets a late-arriving stale row overwrite the real
outcome and silently change a success rate.

This module is the shared, accepted-aware reader:

*   when ``accepted`` is present, only accepted rows count;
*   among accepted rows, the highest ``attempt`` wins, falling back to file order
    when attempts are absent (the pre-X14 contract, where every terminal row was
    by construction the live one);
*   both terminal statuses (``done`` and ``failed``) stay in the denominator --
    a failed episode is an ordinary unsuccessful rollout, not a missing one.

It lives in ``exp/common`` rather than beside one experiment's analyzer so that
experiments do not take a dependency on a sibling experiment's module.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Iterable

TERMINAL_STATUSES = ("done", "failed")


@dataclass(frozen=True)
class EpisodeRecord:
    """The winning terminal row for one episode."""

    task_uid: str
    yaml_id: str
    task_id: int
    episode_idx: int
    success: bool
    status: str
    attempt: int | None = None
    accepted: bool | None = None


def parse_task_uid(task_uid: str) -> tuple[int, int]:
    """``<yaml_id>:<phase>:<task_id>:<episode_idx>`` -> (task_id, episode_idx)."""
    parts = task_uid.rsplit(":", 3)
    if len(parts) != 4:
        raise ValueError(f"unrecognised task_uid {task_uid!r}")
    return int(parts[2]), int(parts[3])


def load_accepted(paths: Iterable[str | pathlib.Path]) -> dict[str, dict[str, EpisodeRecord]]:
    """Read journals into ``{yaml_id: {task_uid: EpisodeRecord}}``.

    Rejects rather than silently resolves a genuine conflict: two *accepted* rows
    for one uid with the same attempt but different outcomes means the journal
    itself is inconsistent, and picking either one would fabricate a result.
    """
    winners: dict[str, dict[str, EpisodeRecord]] = {}
    order: dict[tuple[str, str], int] = {}
    seq = 0

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("status") not in TERMINAL_STATUSES:
                    continue
                accepted = rec.get("accepted")
                if accepted is False:
                    continue  # stale attempt from a superseded dispatch
                uid, arm = rec["task_uid"], rec["yaml_id"]
                task_id, episode_idx = parse_task_uid(uid)
                cand = EpisodeRecord(
                    task_uid=uid,
                    yaml_id=arm,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    success=bool(rec["success"]),
                    status=rec["status"],
                    attempt=rec.get("attempt"),
                    accepted=accepted,
                )
                seq += 1
                prev = winners.setdefault(arm, {}).get(uid)
                if prev is None:
                    winners[arm][uid] = cand
                    order[(arm, uid)] = seq
                    continue

                if prev.attempt is not None and cand.attempt is not None:
                    if cand.attempt > prev.attempt:
                        winners[arm][uid] = cand
                        order[(arm, uid)] = seq
                    elif cand.attempt == prev.attempt and cand.success != prev.success:
                        raise ValueError(
                            f"{arm} {uid}: two accepted rows for attempt {cand.attempt} "
                            f"disagree (success={prev.success} vs {cand.success}); "
                            "the journal is inconsistent and cannot be resolved here"
                        )
                else:
                    # Pre-X14 rows carry no attempt: file order is append order.
                    winners[arm][uid] = cand
                    order[(arm, uid)] = seq

    return winners


def success_map(records: dict[str, EpisodeRecord]) -> dict[tuple[int, int], bool]:
    """``{task_uid: EpisodeRecord}`` -> ``{(task_id, episode_idx): success}``."""
    return {(r.task_id, r.episode_idx): r.success for r in records.values()}
