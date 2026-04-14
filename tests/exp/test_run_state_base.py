"""Unit tests for ``exp/_run_state_base.py``.

Covers:
- Unit enumeration and persistence round-trip.
- Resume behaviour: already-done units are not re-executed.
- Retry pass: failed units are re-run up to ``max_retries`` times.
- Atomic write semantics: ``.tmp`` sibling does not leak, rename is atomic.
- Thread-safety: concurrent ``save()`` calls do not corrupt the state file.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
import threading

from concurrent.futures import ThreadPoolExecutor

from exp._run_state_base import BaseRunState
from exp._run_state_base import UnitState

# ------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------


@dataclass
class _Runner(BaseRunState):
    """Minimal runner used across tests.

    ``_fail_once`` units raise on the first attempt and succeed on retry;
    ``_always_fail`` units always raise.
    """

    state_path: Path = Path("unused")
    unit_ids: tuple[str, ...] = ()
    fail_once: frozenset[str] = field(default_factory=frozenset)
    always_fail: frozenset[str] = field(default_factory=frozenset)
    executed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        BaseRunState.__init__(self, self.state_path, max_retries=2)

    def build_units(self) -> list[UnitState]:
        return [UnitState(unit_key=uid) for uid in self.unit_ids]

    def execute_unit(self, unit: UnitState) -> dict:
        self.executed.append(unit.unit_key)
        if unit.unit_key in self.always_fail:
            raise RuntimeError(f"unit {unit.unit_key} always fails")
        if unit.unit_key in self.fail_once and unit.retry_count == 0:
            raise RuntimeError(f"unit {unit.unit_key} fails on first attempt")
        return {"ok": True, "key": unit.unit_key}


# ------------------------------------------------------------------
# Load / save round-trip
# ------------------------------------------------------------------


def test_build_units_populates_state_file(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    runner = _Runner(state_path=state_path, unit_ids=("u0", "u1", "u2"))
    runner.run()

    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert set(data.keys()) == {"u0", "u1", "u2"}
    assert all(v["status"] == "done" for v in data.values())


def test_load_round_trip_preserves_unit_fields(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    r1 = _Runner(state_path=state_path, unit_ids=("u0",))
    r1.run()

    r2 = _Runner(state_path=state_path, unit_ids=("u0",))
    r2.load()
    assert "u0" in r2.units
    assert r2.units["u0"].status == "done"
    assert r2.units["u0"].result == {"ok": True, "key": "u0"}


# ------------------------------------------------------------------
# Resume
# ------------------------------------------------------------------


def test_resume_skips_done_units(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    r1 = _Runner(state_path=state_path, unit_ids=("u0", "u1"))
    r1.run()
    assert r1.executed == ["u0", "u1"]

    r2 = _Runner(state_path=state_path, unit_ids=("u0", "u1"))
    r2.run(resume=True)
    # Done units from r1 must not be re-executed.
    assert r2.executed == []


def test_resume_reexecutes_non_done_units(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    r1 = _Runner(
        state_path=state_path,
        unit_ids=("u0", "u1"),
        always_fail=frozenset({"u1"}),
    )
    r1.run()
    # u1 fails on every attempt, including the retry passes.
    assert r1.units["u1"].status == "failed"

    r2 = _Runner(state_path=state_path, unit_ids=("u0", "u1"))
    r2.run(resume=True)
    # u0 was done and should be skipped; u1 should be re-attempted.
    assert r2.executed == ["u1"]
    assert r2.units["u1"].status == "done"


# ------------------------------------------------------------------
# Retry pass
# ------------------------------------------------------------------


def test_retry_recovers_fail_once_unit(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    runner = _Runner(
        state_path=state_path,
        unit_ids=("u0",),
        fail_once=frozenset({"u0"}),
    )
    runner.run()

    assert runner.units["u0"].status == "done"
    assert runner.units["u0"].retry_count == 1
    # First attempt + one retry = two executions.
    assert runner.executed == ["u0", "u0"]


def test_retry_gives_up_after_max_retries(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    runner = _Runner(
        state_path=state_path,
        unit_ids=("u0",),
        always_fail=frozenset({"u0"}),
    )
    runner.run()

    assert runner.units["u0"].status == "failed"
    # max_retries=2 -> one primary attempt + 2 retries = 3 executions.
    assert runner.executed == ["u0", "u0", "u0"]
    assert "error" in runner.units["u0"].result


# ------------------------------------------------------------------
# Atomic write
# ------------------------------------------------------------------


def test_save_uses_atomic_rename(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    runner = _Runner(state_path=state_path, unit_ids=("u0",))
    runner.units = {"u0": UnitState(unit_key="u0")}
    runner.save()

    # After save the tmp sibling must be gone.
    tmp_sibling = state_path.with_suffix(state_path.suffix + ".tmp")
    assert not tmp_sibling.exists()
    assert state_path.exists()
    assert "u0" in json.loads(state_path.read_text())


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------


def test_concurrent_save_does_not_corrupt_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    runner = _Runner(
        state_path=state_path,
        unit_ids=tuple(f"u{i}" for i in range(16)),
    )
    # Populate units directly, bypassing ``run()`` so we can stress ``save()``.
    for uid in runner.unit_ids:
        runner.units[uid] = UnitState(unit_key=uid)

    errors: list[BaseException] = []

    def _hammer() -> None:
        try:
            for _ in range(20):
                runner.save()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # Final state file must still parse as JSON with all 16 units.
    data = json.loads(state_path.read_text())
    assert set(data.keys()) == {f"u{i}" for i in range(16)}


# ------------------------------------------------------------------
# Unit filter
# ------------------------------------------------------------------


def test_unit_filter_limits_primary_pass(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    runner = _Runner(state_path=state_path, unit_ids=("u0", "u1", "u2"))
    runner.run(unit_filter=lambda u: u.unit_key == "u1")

    assert runner.executed == ["u1"]
    # Only u1 was selected by the filter, so u0 and u2 stay pending.
    assert runner.units["u0"].status == "pending"
    assert runner.units["u1"].status == "done"
    assert runner.units["u2"].status == "pending"


def test_unit_filter_also_scopes_retry_pass(tmp_path: Path) -> None:
    """Regression: ``unit_filter`` must scope **both** the primary pass and
    the retry passes. A pre-existing failed unit outside the filter (e.g.
    left over from an earlier shard) must NOT be picked up by the current
    shard's retry pass.
    """
    state_path = tmp_path / "state.json"

    # --- Shard A: runs u0, leaves it as "failed". ---
    shard_a = _Runner(
        state_path=state_path,
        unit_ids=("u0", "u1"),
        always_fail=frozenset({"u0"}),
    )
    shard_a.run(unit_filter=lambda u: u.unit_key == "u0")
    assert shard_a.units["u0"].status == "failed"
    assert shard_a.units["u1"].status == "pending"

    # --- Shard B resumes and targets u1 only. u0 must remain untouched. ---
    shard_b = _Runner(
        state_path=state_path,
        unit_ids=("u0", "u1"),
        fail_once=frozenset({"u1"}),
    )
    shard_b.run(resume=True, unit_filter=lambda u: u.unit_key == "u1")

    # u0 was failed BEFORE shard_b; shard_b must not retry it.
    assert "u0" not in shard_b.executed
    assert shard_b.units["u0"].status == "failed"
    # u1 recovers via retry (fails once then succeeds); retry stayed in scope.
    assert shard_b.executed == ["u1", "u1"]
    assert shard_b.units["u1"].status == "done"
    assert shard_b.units["u1"].retry_count == 1


# ------------------------------------------------------------------
# cleanup/06 §2 P1-1: retry save cadence is asymmetric serial vs parallel
# ------------------------------------------------------------------


def test_serial_retry_saves_per_unit_flip(tmp_path: Path) -> None:
    """Plan §2 P1-1 lock: in serial ``run()``, every still-failed unit must
    be flipped to ``pending`` AND persisted individually before its retry
    runs. A crash between two retry flips should expose only the one unit
    that was mid-transition, not all of them at once."""
    state_path = tmp_path / "state.json"
    runner = _Runner(
        state_path=state_path,
        unit_ids=("u0", "u1"),
        # Both fail on primary + retry-1, succeed on retry-2.
        fail_once=frozenset({"u0", "u1"}),
    )

    # Spy on save() and record (for each call) a snapshot of statuses and
    # retry_counts. We only care about the first retry attempt here.
    snapshots: list[dict[str, tuple[str, int]]] = []
    real_save = runner.save

    def _spy_save() -> None:
        snapshots.append(
            {k: (u.status, u.retry_count) for k, u in runner.units.items()}
        )
        real_save()

    runner.save = _spy_save  # type: ignore[method-assign]
    runner.run()

    # Locate the first post-primary snapshot where u0 has flipped to
    # ``pending`` for attempt=1 — at that exact save, u1 must still be
    # ``failed`` (its per-unit flip-save comes later).
    matched = False
    for snap in snapshots:
        if snap["u0"] == ("pending", 1) and snap["u1"] == ("failed", 0):
            matched = True
            break
    assert matched, (
        "serial retry must save u0's pending flip before touching u1; "
        f"got snapshots={snapshots}"
    )


def test_parallel_retry_batch_flips_before_single_save(tmp_path: Path) -> None:
    """Parallel mode flips ALL still-failed units to pending first, saves
    once, then redispatches. The batch snapshot where both units are
    ``pending`` simultaneously (and retry_count=1) must exist — serial mode
    could never produce that snapshot."""
    state_path = tmp_path / "state.json"
    runner = _Runner(
        state_path=state_path,
        unit_ids=("u0", "u1"),
        fail_once=frozenset({"u0", "u1"}),
    )

    snapshots: list[dict[str, tuple[str, int]]] = []
    real_save = runner.save

    def _spy_save() -> None:
        snapshots.append(
            {k: (u.status, u.retry_count) for k, u in runner.units.items()}
        )
        real_save()

    runner.save = _spy_save  # type: ignore[method-assign]
    runner.parallel_run(num_workers=2)

    # The batch flip writes exactly one save where BOTH units are pending
    # with retry_count=1. Serial mode cannot produce this snapshot.
    assert any(
        snap["u0"] == ("pending", 1) and snap["u1"] == ("pending", 1)
        for snap in snapshots
    ), f"parallel retry must show a batch-pending save; got snapshots={snapshots}"


# ------------------------------------------------------------------
# cleanup/06: parallel_run shares the same driver as run
# ------------------------------------------------------------------


def test_parallel_run_executes_all_units_via_executor(tmp_path: Path) -> None:
    """Cleanup/06 lock: parallel_run submits every primary-pass unit through
    the pool and waits for completion before returning."""
    state_path = tmp_path / "state.json"
    runner = _Runner(state_path=state_path, unit_ids=tuple(f"u{i}" for i in range(4)))
    runner.parallel_run(num_workers=2)

    assert set(runner.executed) == {"u0", "u1", "u2", "u3"}
    assert all(runner.units[k].status == "done" for k in runner.unit_ids)


def test_parallel_run_retries_fail_once_units(tmp_path: Path) -> None:
    """Parallel retry path must flip failed units to ``pending`` and redispatch
    them through the pool, just like the serial retry loop."""
    state_path = tmp_path / "state.json"
    runner = _Runner(
        state_path=state_path,
        unit_ids=("u0", "u1"),
        fail_once=frozenset({"u0"}),
    )
    runner.parallel_run(num_workers=2)

    assert runner.units["u0"].status == "done"
    assert runner.units["u0"].retry_count == 1
    assert runner.units["u1"].status == "done"
    # u0 ran twice (primary fail + retry), u1 once.
    assert runner.executed.count("u0") == 2
    assert runner.executed.count("u1") == 1


def test_dispatch_uses_executor_when_provided(tmp_path: Path) -> None:
    """Structural lock: the private ``_dispatch`` helper goes through the
    supplied ``ThreadPoolExecutor`` rather than calling ``_execute_one``
    inline. We observe this by tracking how many distinct threads ran the
    units — with ``max_workers=4`` and a small barrier inside ``execute_unit``
    we expect ``>1`` worker thread to participate.
    """
    state_path = tmp_path / "state.json"

    barrier = threading.Barrier(4, timeout=2.0)
    seen_threads: set[int] = set()
    lock = threading.Lock()

    class _ParallelRunner(BaseRunState):
        def __init__(self, path: Path) -> None:
            super().__init__(path, max_retries=0)

        def build_units(self):
            return [UnitState(unit_key=f"u{i}") for i in range(4)]

        def execute_unit(self, unit):
            barrier.wait()
            with lock:
                seen_threads.add(threading.get_ident())
            return {"ok": True}

    runner = _ParallelRunner(state_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        runner._run_impl(resume=False, unit_filter=None, executor=pool)

    # All 4 units done, and the barrier only unblocks when 4 threads reach it —
    # proving the executor actually parallelised execution.
    assert all(u.status == "done" for u in runner.units.values())
    assert len(seen_threads) == 4
