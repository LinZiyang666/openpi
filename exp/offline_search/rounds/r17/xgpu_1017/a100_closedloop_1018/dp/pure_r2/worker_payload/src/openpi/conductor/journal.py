"""Append-only journal for crash-safe resume (plan §8.1).

Each terminal episode appends one JSON line. On restart the driver replays the
journal: episodes recorded ``done`` are skipped (idempotent via the deterministic
``task_uid``). Warmup episodes are recorded too but resume is stage-atomic
(plan §8.2), so the scheduler only consumes *eval* done-uids from a replay.

Coupling map:
  DEPENDS ON:  standard library (json, pathlib, threading)
  CONSUMED BY: driver.py
  IF CHANGED:  resume semantics in driver / scheduler.mark_preexisting_done
"""

from __future__ import annotations

import json
import pathlib
import threading
import time


class Journal:
    """Append-only JSONL ledger. Thread-safe; one line per terminal episode."""

    def __init__(self, path: str | pathlib.Path) -> None:
        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> pathlib.Path:
        return self._path

    def record(
        self,
        *,
        task_uid: str,
        yaml_id: str,
        phase: str,
        status: str,
        success: bool,
        attempt: int | None = None,
        accepted: bool | None = None,
        error: str | None = None,
        duration_s: float | None = None,
        run_id: str | None = None,
    ) -> None:
        """Append one terminal-episode record. ``status`` is ``done`` | ``failed``.

        ``attempt`` / ``accepted`` / ``error`` (X14) are optional and omitted
        from the line when None, so records written by pre-existing callers stay
        byte-identical. They exist because ``status`` alone cannot tell an
        offline consumer whether the scheduler took this result as the live
        dispatch's outcome: a stale attempt from a superseded dispatch is
        journaled exactly like the real one. The RL router's batch packager
        admits an episode into a training batch only when ``accepted`` is True
        and ``error`` is None.

        ``duration_s`` is the wall clock the worker spent on the episode. It is
        recorded because a capacity change is judged on worker utilisation, and
        the ledger is the only per-episode artifact a phase leaves behind -- the
        health aggregator is in-memory and the monitor renders a string. Without
        it, "did the fleet stay busy" is not answerable from anything the run
        wrote down.
        """
        record: dict = {
            "task_uid": task_uid,
            "yaml_id": yaml_id,
            "phase": phase,
            "status": status,
            "success": success,
            "ts": time.time(),
        }
        for key, value in (
            ("attempt", attempt),
            ("accepted", accepted),
            ("error", error),
            # Rounded: milliseconds are already finer than anything a
            # utilisation figure resolves, and full float repr would double the
            # size of a ledger with hundreds of thousands of lines.
            ("duration_s", None if duration_s is None else round(float(duration_s), 3)),
            # Which driver process produced this line. Dispatch generations
            # restart at 1 after a crash, so without it a re-run episode's
            # record is indistinguishable from the pre-crash one.
            ("run_id", run_id),
        ):
            if value is not None:
                record[key] = value
        line = json.dumps(record, ensure_ascii=False)
        # Open per-append + flush so a crash mid-run cannot lose a flushed record;
        # throughput is fine at episode granularity.
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    @staticmethod
    def record_counts_as_done(rec: dict) -> bool:
        """Does this terminal record describe work that actually completed?

        The driver journals *rejected* results too -- a timed-out episode is
        re-dispatched at a higher generation, and when the original worker
        finally reports, the scheduler fences that result and it is written
        with ``accepted: false``. Treating such a record as completed work is
        how a crash between the fence and the live retry turns into an episode
        that is never run again: resume skips it, and the arm is then short by
        one with nothing reporting a failure.

        Records written before the field existed carry neither value, so the
        test is ``is False`` rather than falsiness -- absent means "unknown,
        assume real", which is what keeps older ledgers replaying as they did.

        This is the single definition of "counts as done"; every consumer
        (resume replay, arm filtering, outcome selection, utilisation) must
        route through it or they will disagree about the same ledger.
        """
        return rec.get("status") in ("done", "failed") and rec.get("accepted") is not False

    def replay_done_uids(self) -> set[str]:
        """Return task_uids with any TERMINAL record (``done`` OR ``failed``).

        Both are completed episodes that must NOT be re-run on resume:
        ``done`` = solved, ``failed`` = unsolved but NON-RETRIABLE terminal
        (driver journals ``failed`` only when ``not retriable``; retriable
        transport errors are never journaled, they are requeued in-memory).
        Previously only ``done`` was skipped, so every ``failed`` episode was
        re-run on resume — deterministic rollouts just fail again, inflating
        the journal while distinct/full100 never advances (a resume livelock).

        Records the scheduler rejected (``accepted: false``) are excluded --
        see :meth:`record_counts_as_done`. Without that, an episode whose only
        terminal line is a fenced stale result is skipped forever.
        """
        if not self._path.exists():
            return set()
        done: set[str] = set()
        with self._path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn last line from a hard crash
                uid = rec.get("task_uid")
                if uid is None:
                    continue
                # Only the positive case adds: a uid can carry both a fenced
                # record and a real one in either order on disk, and a rejected
                # line must never cancel an accepted one.
                if self.record_counts_as_done(rec):
                    done.add(uid)
        return done
