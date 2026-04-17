"""BaseRunState: persistent unit-of-work skeleton shared by experiment runners.

This module extracts the state-tracking pattern previously inlined in
``exp/common/run_cache_experiments.py`` (class ``RunState`` + ``_retry_failed_runs``)
into a reusable base class. New Step 1b / Step 2 / Step 3 runners for the
trajectory-deviation experiment subclass it instead of re-implementing load /
save / resume / retry logic.

Public interface
----------------
- ``UnitState``: a JSON-serializable dataclass describing one unit of work.
- ``BaseRunState``: abstract base class. Subclasses override ``build_units()``
  and ``execute_unit()``; the base class owns persistence (lock + atomic
  rename), resume, and retry orchestration.

Key design notes
----------------
- ``unit_key`` is always a string so the state JSON round-trips cleanly.
  Subclasses stringify their own tuple-like keys (for example
  ``"cfg:ep007:s42:n5:k0"``).
- ``save()`` is protected by a process-local ``threading.Lock`` and uses a
  ``<path>.tmp`` + ``replace()`` atomic rename so concurrent workers inside
  the same process (or external readers tailing the file during resume) never
  observe a half-written JSON.
- Retry strategy is "in-place re-queue": failed units flip back to
  ``pending`` with ``retry_count += 1``; no separate retry directory is kept.
  This differs from the original ``_retry_failed_runs`` because Step 2 / 3
  are not YAML-directory based.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Unit record
# ------------------------------------------------------------------


@dataclass
class UnitState:
    """Single unit of work persisted to the state JSON.

    ``unit_key`` is the stable identifier (unique within one runner). All
    other fields are populated by the base driver; ``result`` is the
    per-unit free-form payload owned by the subclass (for example
    ``{"successes": s, "total": t}`` or ``{"error": "..."}``).
    """

    unit_key: str
    status: str = "pending"  # pending | running | done | failed
    start_time: str | None = None
    end_time: str | None = None
    retry_count: int = 0
    result: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Base runner
# ------------------------------------------------------------------


class BaseRunState(ABC):
    """Skeleton for runner state files.

    Subclass responsibilities:
      - implement ``build_units()`` to enumerate the initial unit list.
      - implement ``execute_unit(unit)`` to run one unit (return a dict on
        success, raise on failure).

    Base class responsibilities:
      - load / save JSON with thread lock + atomic rename.
      - per-unit status transitions (pending -> running -> done|failed).
      - bounded retry pass for failed units.
    """

    def __init__(self, state_path: Path, *, max_retries: int = 2) -> None:
        self.state_path = Path(state_path)
        self.max_retries = max_retries
        self.units: dict[str, UnitState] = {}
        # Serializes concurrent save() calls within one process. Content
        # mutation on individual ``UnitState`` objects is not locked — the
        # convention is that each worker only mutates its own unit.
        self._save_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load units from ``state_path`` if the file exists."""
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            self.units = {k: UnitState(**v) for k, v in data.items()}

    def save(self) -> None:
        """Persist the full unit map atomically.

        Uses a ``.tmp`` sibling and ``Path.replace()`` so a concurrent
        reader (or an interrupted save) never observes a partially written
        JSON. The lock prevents two threads from racing on the tmp file.
        """
        with self._save_lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps({k: asdict(u) for k, u in self.units.items()}, indent=2)
            )
            tmp_path.replace(self.state_path)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def is_done(self, key: str) -> bool:
        return key in self.units and self.units[key].status == "done"

    def pending_units(self) -> list[UnitState]:
        return [u for u in self.units.values() if u.status != "done"]

    def failed_units(self) -> list[UnitState]:
        return [u for u in self.units.values() if u.status == "failed"]

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def build_units(self) -> list[UnitState]:
        """Enumerate the initial unit list (called when state file is empty)."""

    @abstractmethod
    def execute_unit(self, unit: UnitState) -> dict:
        """Run one unit.

        Return a result dict on success (stored in ``unit.result``); raise on
        failure (the base driver catches the exception, records
        ``{"error": str(e)}`` and flips status to ``"failed"``). Subclasses
        should log context before raising so the traceback is preserved.
        """

    # ------------------------------------------------------------------
    # Main driver
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        resume: bool = False,
        unit_filter: Callable[[UnitState], bool] | None = None,
    ) -> None:
        """Primary entry point. Runs primary pass then up to ``max_retries`` retry passes.

        ``unit_filter`` — when given — scopes **both** the primary pass and the
        retry passes. Without this the retry pass would pick up failed units
        outside the filter (for example, units left over from an earlier shard
        run), which breaks resume/sharded execution. Always filter retries
        with the same predicate used for the primary pass.
        """
        self._run_impl(resume=resume, unit_filter=unit_filter, executor=None)

    def parallel_run(
        self,
        *,
        num_workers: int,
        resume: bool = False,
        unit_filter: Callable[[UnitState], bool] | None = None,
    ) -> None:
        """Thread-pool variant of ``run`` (plan §10.1 Step 2 requirement).

        Each worker owns one unit through ``_execute_one``; the base class'
        ``save()`` lock + atomic rename keep the shared state file consistent
        across threads. Because ``execute_unit`` is expected to be I/O-bound
        (websocket round-trip per step), a thread pool is sufficient — we do
        not need subprocesses.

        Retry semantics mirror ``run``: up to ``max_retries`` passes, each
        pass flips any still-failed unit back to ``pending`` and re-submits
        it to the pool.
        """
        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
            self._run_impl(resume=resume, unit_filter=unit_filter, executor=pool)

    def _run_impl(
        self,
        *,
        resume: bool,
        unit_filter: Callable[[UnitState], bool] | None,
        executor: ThreadPoolExecutor | None,
    ) -> None:
        """Shared driver for ``run`` / ``parallel_run``.

        ``executor=None`` dispatches serially; a pool dispatches through it.
        Retry save cadence is deliberately asymmetric between the two modes
        (locked by plan §2 P1-1): serial flips-and-saves per unit so a crash
        between two retry units only exposes the one currently being re-queued;
        parallel flips all still-failed units, saves once, then redispatches
        the pool batch.
        """
        if resume:
            self.load()
        if not self.units:
            for u in self.build_units():
                self.units[u.unit_key] = u
        self.save()

        def _in_scope(u: UnitState) -> bool:
            return unit_filter is None or unit_filter(u)

        queue = [u for u in self.units.values() if u.status != "done" and _in_scope(u)]
        self._dispatch(queue, executor)

        for attempt in range(1, self.max_retries + 1):
            still_failed = [u for u in self.failed_units() if _in_scope(u)]
            if not still_failed:
                break
            logger.info(
                "[%s %d/%d] %d failed units",
                "retry" if executor is None else "parallel retry",
                attempt, self.max_retries, len(still_failed),
            )
            if executor is None:
                for u in still_failed:
                    u.retry_count = attempt
                    u.status = "pending"
                    self.save()
                    self._execute_one(u)
            else:
                for u in still_failed:
                    u.retry_count = attempt
                    u.status = "pending"
                self.save()
                self._dispatch(still_failed, executor)

    def _dispatch(
        self,
        units: list[UnitState],
        executor: ThreadPoolExecutor | None,
    ) -> None:
        """Run ``_execute_one`` over each unit, serially or via executor."""
        if not units:
            return
        if executor is None:
            for u in units:
                self._execute_one(u)
            return
        futures = [executor.submit(self._execute_one, u) for u in units]
        for _ in as_completed(futures):
            pass  # _execute_one persists state on its own

    def _execute_one(self, u: UnitState) -> None:
        u.status = "running"
        u.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        try:
            result = self.execute_unit(u)
            u.result = result
            u.status = "done"
        # Broad catch is intentional: subclasses are expected to log context
        # before raising, and the base driver records only a string summary
        # so the state JSON stays JSON-serializable.
        except Exception as e:
            u.status = "failed"
            u.result = {"error": str(e)}
            logger.warning("unit %s failed: %s", u.unit_key, e, exc_info=True)
        u.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
