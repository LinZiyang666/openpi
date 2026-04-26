"""Cache session lifecycle tests at the Orchestrator + Interceptor level (plan §7.10).

These tests exercise the orchestrator-side broadcast / cleanup helpers
introduced for the cross-step score memo:

  - `CacheOrchestrator._broadcast_episode_start()` (single helper called
    from both `on_task_begin` and `on_episode_start`)
  - `CacheOrchestrator._close_current_search_sessions()` (single cleanup
    helper called from `on_episode_end` finally, `on_task_end`, and the
    stale-clear step inside `_broadcast_episode_start`)
  - `InferenceInterceptor.on_task_end()` forwarding cleanup so that
    connection-close paths without `on_episode_end` still release every
    registered strategy session.

Sub-cases (a)-(g) directly mirror plan §7.10. The tests intentionally
talk to the orchestrator and storage rather than the backend directly,
because the regression risk addressed by the rewrite is *who registers /
unregisters the per-strategy sid*, not the backend's set semantics
(those are covered separately in test_in_memory_backend_trajectory.py).
"""

from __future__ import annotations

import pytest
import torch

from openpi.cache.backends.in_memory_backend import (
    SearchSessionActiveError,
    InMemoryBackend,
)
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import ThresholdJudge
from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.components.search_strategy import WeightedScoreSumKnnStrategy
from openpi.cache.components.write_policy import OnAnyMissWritePolicy
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID

from tests.cache.conftest import _wrap_per_checkpoint


# ---------------------------------------------------------------------------
# Wiring helper — orchestrator with two real Trajectory strategies (one per
# checkpoint) so we can verify per-strategy sid collection.
# ---------------------------------------------------------------------------


def _make_orch() -> tuple[CacheOrchestrator, InMemoryBackend, CacheStorage,
                          dict[CheckpointID, WeightedScoreSumKnnStrategy]]:
    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    kb = PlaceholderKeyBuilder()
    gate = AlwaysSearchGate()
    judge = ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95)
    timer = SystemTimer(enabled=False)
    strategies = {
        CheckpointID.CP1: WeightedScoreSumKnnStrategy(
            storage,
            top_k=1,
            fusion_weights={"robot_state": 1.0},
            trajectory_depth=2,
            trajectory_weights=[0.6, 0.4],
        ),
        CheckpointID.CP3: WeightedScoreSumKnnStrategy(
            storage,
            top_k=1,
            fusion_weights={"robot_state": 1.0},
            trajectory_depth=2,
            trajectory_weights=[0.6, 0.4],
        ),
    }
    orch = CacheOrchestrator(
        storage,
        kb,
        gates=_wrap_per_checkpoint(gate),
        judges=_wrap_per_checkpoint(judge),
        search_strategies=strategies,
        timer=timer,
        write_policy=None,
    )
    return orch, backend, storage, strategies


# ---------------------------------------------------------------------------
# (a) Normal episode_start → episode_end cleans every layer
# ---------------------------------------------------------------------------


def test_episode_start_then_end_releases_sessions():
    orch, backend, _, strategies = _make_orch()

    orch.on_episode_start(task_key="t", episode_id="e")
    # Both strategies minted a sid; orchestrator registered both.
    assert len(orch._current_strategy_session_ids) == 2
    for sid in orch._current_strategy_session_ids:
        assert sid in backend._active_search_sessions

    orch.on_episode_end()
    assert orch._current_strategy_session_ids == []
    assert backend._active_search_sessions == set()
    for s in strategies.values():
        # Strategy keeps the last sid as its local state until the next
        # episode_start; that is intentional and not relevant to backend
        # cleanup. What matters is the backend forgot the sid.
        sid = s.get_search_session_id()
        assert sid is not None
        assert sid not in backend._active_search_sessions
        assert sid not in backend._score_memo


# ---------------------------------------------------------------------------
# (b) Connection close path: on_task_end fires without on_episode_end
# ---------------------------------------------------------------------------


def test_on_task_end_releases_sessions_without_episode_end():
    orch, backend, _, _ = _make_orch()

    orch.on_episode_start()
    assert backend._active_search_sessions, "sessions should be registered after episode_start"

    # Simulate WebSocket disconnect: no on_episode_end, just on_task_end.
    orch.on_task_end()
    assert orch._current_strategy_session_ids == []
    assert backend._active_search_sessions == set()

    # Mutation must again flow freely.
    from openpi.cache.storage_types import CacheEntry, CachePayload

    e = CacheEntry(
        id="e0",
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.zeros(32)},
        payload=CachePayload(action_chunk=torch.zeros(50, 32)),
    )
    backend.insert(e)  # would have raised under an active session
    backend.delete(["e0"])  # ditto


# ---------------------------------------------------------------------------
# (c) Repeated on_episode_start without intervening on_episode_end →
#     stale-clear inside _broadcast_episode_start cleans the previous run
# ---------------------------------------------------------------------------


def test_repeated_episode_start_cleans_stale_sessions():
    orch, backend, _, _ = _make_orch()

    orch.on_episode_start()
    stale_sids = list(orch._current_strategy_session_ids)
    assert stale_sids, "first episode_start should have registered sids"

    # Without calling on_episode_end:
    orch.on_episode_start()
    fresh_sids = list(orch._current_strategy_session_ids)

    # Stale sids removed from backend; only fresh sids remain registered.
    for sid in stale_sids:
        assert sid not in backend._active_search_sessions
        assert sid not in backend._score_memo
    assert set(fresh_sids).issubset(backend._active_search_sessions)
    # Strategies are guaranteed to mint *new* sids on each episode_start
    # (uuid4 collision probability is negligible).
    assert set(stale_sids).isdisjoint(set(fresh_sids))


# ---------------------------------------------------------------------------
# (d) Five sequential episodes — invariants hold across the loop
# ---------------------------------------------------------------------------


def test_five_serial_episodes_invariant():
    orch, backend, _, _ = _make_orch()

    for i in range(5):
        orch.on_episode_start(task_key=f"t{i}", episode_id=f"e{i}")
        assert len(backend._active_search_sessions) == 2, (
            f"episode {i}: expected 2 active sessions, got {backend._active_search_sessions}"
        )
        orch.on_episode_end()
        assert backend._active_search_sessions == set(), (
            f"episode {i}: sessions should be released after on_episode_end"
        )
        assert orch._current_strategy_session_ids == []


# ---------------------------------------------------------------------------
# (e) on_task_begin path — same broadcast helper, registers + cleans up
# ---------------------------------------------------------------------------


def test_on_task_begin_registers_and_on_task_end_cleans():
    orch, backend, _, _ = _make_orch()

    # Server connection open: only on_task_begin (no on_episode_start yet).
    orch.on_task_begin(task_key="task")
    assert len(orch._current_strategy_session_ids) == 2
    for sid in orch._current_strategy_session_ids:
        assert sid in backend._active_search_sessions

    # Mutation guard active.
    from openpi.cache.storage_types import CacheEntry, CachePayload

    e0 = CacheEntry(
        id="existing",
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.zeros(32)},
        payload=CachePayload(action_chunk=torch.zeros(50, 32)),
    )
    backend._entries["existing"] = e0  # bypass insert just to seed
    with pytest.raises(SearchSessionActiveError):
        backend.insert(e0)  # upsert blocked

    # Connection close.
    orch.on_task_end()
    assert backend._active_search_sessions == set()
    backend.insert(e0)  # now allowed


# ---------------------------------------------------------------------------
# (f) on_episode_end early-return paths still cleanup via try/finally
# ---------------------------------------------------------------------------


def test_episode_end_empty_steps_still_cleans():
    orch, backend, _, _ = _make_orch()

    orch.on_episode_start()
    assert backend._active_search_sessions  # sanity

    # No check() calls → _episode_steps stays empty → first early return.
    orch.on_episode_end()
    assert backend._active_search_sessions == set()
    assert orch._current_strategy_session_ids == []


def test_episode_end_no_write_policy_still_cleans():
    """When write_policy is None, on_episode_end early-returns after the
    empty-steps branch but BEFORE the write path. The finally block must
    still run.
    """
    orch, backend, _, _ = _make_orch()
    # write_policy=None already — replicate the second early-return branch
    # by forcing a non-empty episode buffer and then calling on_episode_end.
    orch.on_episode_start()

    # Forge a step record so we skip the first early return and hit the
    # `_write_policy is None` branch instead.
    from openpi.cache.storage_types import CacheEntry, CachePayload, StepRecord

    orch._episode_steps.append(
        StepRecord(
            query_keys={"robot_state": torch.zeros(32)},
            action_chunk=torch.zeros(50, 32),
        )
    )

    orch.on_episode_end()
    assert backend._active_search_sessions == set()
    assert orch._current_strategy_session_ids == []


def test_episode_end_write_policy_declines_still_cleans():
    """Third early-return branch: write_policy.should_write() returns False."""

    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    kb = PlaceholderKeyBuilder()
    gate = AlwaysSearchGate()
    judge = ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95)
    timer = SystemTimer(enabled=False)
    strategies = {
        CheckpointID.CP1: WeightedScoreSumKnnStrategy(
            storage,
            top_k=1,
            fusion_weights={"robot_state": 1.0},
            trajectory_depth=2,
            trajectory_weights=[0.6, 0.4],
        ),
    }
    orch = CacheOrchestrator(
        storage, kb,
        gates=_wrap_per_checkpoint(gate),
        judges=_wrap_per_checkpoint(judge),
        search_strategies=strategies,
        timer=timer,
        write_policy=OnAnyMissWritePolicy(),
    )

    orch.on_episode_start()
    # Fake a step record but report 0 misses; OnAnyMiss returns False.
    from openpi.cache.storage_types import StepRecord

    orch._episode_steps.append(
        StepRecord(
            query_keys={"robot_state": torch.zeros(32)},
            action_chunk=torch.zeros(50, 32),
        )
    )
    # No misses recorded → should_write returns False.
    orch.on_episode_end()
    assert backend._active_search_sessions == set()
    assert orch._current_strategy_session_ids == []


# ---------------------------------------------------------------------------
# (g) Helper uniqueness — only the two helpers ever touch the list
# ---------------------------------------------------------------------------


def test_only_helpers_mutate_session_id_list():
    """Static structural check: every reference to
    `self._current_strategy_session_ids` outside `__init__` must live
    inside one of the two helpers. Any other lifecycle path that touches
    the list directly is a process violation.
    """
    import ast
    import textwrap
    from pathlib import Path

    src = Path(
        "src/openpi/cache/orchestrator.py"
    ).read_text()
    tree = ast.parse(src)

    allowed = {
        "_broadcast_episode_start",
        "_close_current_search_sessions",
        "__init__",
    }

    offenders: list[tuple[str, int]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.fn_stack: list[str] = []

        def visit_FunctionDef(self, node):
            self.fn_stack.append(node.name)
            self.generic_visit(node)
            self.fn_stack.pop()

        def visit_Attribute(self, node):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr == "_current_strategy_session_ids"
            ):
                fn = self.fn_stack[-1] if self.fn_stack else "<module>"
                if fn not in allowed:
                    offenders.append((fn, node.lineno))
            self.generic_visit(node)

    _Visitor().visit(tree)

    assert not offenders, textwrap.dedent(
        f"""
        Direct references to `self._current_strategy_session_ids` found
        outside the two designated helpers and __init__:
          {offenders}
        Per plan §7.10 (g), all mutations must funnel through
        `_broadcast_episode_start` and `_close_current_search_sessions`.
        """
    )


# ---------------------------------------------------------------------------
# Interceptor forwarding — on_task_end on InferenceInterceptor reaches
# CacheOrchestrator.on_task_end. Constructed without a real Policy.
# ---------------------------------------------------------------------------


def test_interceptor_on_task_end_forwards_to_orchestrator():
    """Direct call against the production `InferenceInterceptor.on_task_end`.

    Bypasses `__init__` (which would require a real PI0 policy + GPU model)
    via `object.__new__` and seeds only the two attributes the method
    actually reads (`_timer`, `_orchestrator`). This way a future regression
    in `src/openpi/cache/interceptor.py` — e.g. dropping the orchestrator
    forward — fails this test, whereas a hand-written stub would not.
    """
    from openpi.cache.interceptor import InferenceInterceptor

    orch, backend, _, _ = _make_orch()
    orch.on_episode_start()
    assert backend._active_search_sessions

    interceptor = object.__new__(InferenceInterceptor)
    timer_calls: list[str] = []

    class _StubTimer:
        def on_task_end(self_inner):
            timer_calls.append("on_task_end")

    interceptor._timer = _StubTimer()
    interceptor._orchestrator = orch

    InferenceInterceptor.on_task_end(interceptor)

    # SystemTimer.on_task_end was hit (proves we're on the production body).
    assert timer_calls == ["on_task_end"]
    # Orchestrator forwarding cleared the active sessions on the backend.
    assert backend._active_search_sessions == set()
