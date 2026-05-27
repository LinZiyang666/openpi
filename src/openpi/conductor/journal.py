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
    ) -> None:
        """Append one terminal-episode record. ``status`` is ``done`` | ``failed``."""
        line = json.dumps(
            {
                "task_uid": task_uid,
                "yaml_id": yaml_id,
                "phase": phase,
                "status": status,
                "success": success,
                "ts": time.time(),
            },
            ensure_ascii=False,
        )
        # Open per-append + flush so a crash mid-run cannot lose a flushed record;
        # throughput is fine at episode granularity.
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    def replay_done_uids(self) -> set[str]:
        """Return task_uids with any TERMINAL record (``done`` OR ``failed``).

        Both are completed episodes that must NOT be re-run on resume:
        ``done`` = solved, ``failed`` = unsolved but NON-RETRIABLE terminal
        (driver journals ``failed`` only when ``not retriable``; retriable
        transport errors are never journaled, they are requeued in-memory).
        Previously only ``done`` was skipped, so every ``failed`` episode was
        re-run on resume — deterministic rollouts just fail again, inflating
        the journal while distinct/full100 never advances (a resume livelock).
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
                if rec.get("status") in ("done", "failed"):
                    done.add(uid)
        return done
