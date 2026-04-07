"""Tests for episode-level write path: buffer_for_write + on_episode_end."""

import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import HitType, ThresholdJudge
from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.components.write_policy import (
    AlwaysWritePolicy,
    NeverWritePolicy,
    OnAnyMissWritePolicy,
)
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import CachePayload, EpisodeRecord, StepRecord
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID

from tests.cache.conftest import (
    TestStorageSearchStrategy,
    _wrap_per_checkpoint,
    make_orchestrator,
    make_stage1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orch_with_policy(write_policy) -> tuple[CacheOrchestrator, InMemoryBackend, CacheStorage]:
    return make_orchestrator(write_policy=write_policy)


# ---------------------------------------------------------------------------
# WritePolicy unit tests
# ---------------------------------------------------------------------------


class TestWritePolicy:
    def test_on_any_miss_with_misses(self):
        record = EpisodeRecord(
            steps=[], task_key="t", total_steps=5,
            miss_by_checkpoint={CheckpointID.CP1: 3},
        )
        assert OnAnyMissWritePolicy().should_write(record) is True

    def test_on_any_miss_no_misses(self):
        record = EpisodeRecord(
            steps=[], task_key="t", total_steps=5,
            miss_by_checkpoint={},
        )
        assert OnAnyMissWritePolicy().should_write(record) is False

    def test_on_any_miss_zero_cp1_misses(self):
        record = EpisodeRecord(
            steps=[], task_key="t", total_steps=5,
            miss_by_checkpoint={CheckpointID.CP1: 0},
        )
        assert OnAnyMissWritePolicy().should_write(record) is False

    def test_always_write(self):
        record = EpisodeRecord(
            steps=[], task_key="t", total_steps=0,
            miss_by_checkpoint={},
        )
        assert AlwaysWritePolicy().should_write(record) is True

    def test_never_write(self):
        record = EpisodeRecord(
            steps=[], task_key="t", total_steps=10,
            miss_by_checkpoint={CheckpointID.CP1: 100},
        )
        assert NeverWritePolicy().should_write(record) is False


# ---------------------------------------------------------------------------
# buffer_for_write accumulation
# ---------------------------------------------------------------------------


class TestBufferForWrite:
    def test_accumulates_steps(self):
        orch, _, _ = _make_orch_with_policy(AlwaysWritePolicy())
        assert len(orch._episode_steps) == 0

        keys1 = {"robot_state": torch.randn(32)}
        action1 = torch.randn(50, 32)
        orch.buffer_for_write(keys1, action1)
        assert len(orch._episode_steps) == 1

        keys2 = {"robot_state": torch.randn(32)}
        action2 = torch.randn(50, 32)
        orch.buffer_for_write(keys2, action2)
        assert len(orch._episode_steps) == 2

    def test_step_record_contents(self):
        orch, _, _ = _make_orch_with_policy(AlwaysWritePolicy())
        keys = {"robot_state": torch.randn(32)}
        action = torch.randn(50, 32)
        orch.buffer_for_write(keys, action)

        step = orch._episode_steps[0]
        assert torch.equal(step.query_keys["robot_state"], keys["robot_state"])
        assert torch.equal(step.action_chunk, action)


# ---------------------------------------------------------------------------
# on_episode_end with different policies
# ---------------------------------------------------------------------------


class TestOnEpisodeEnd:
    def test_always_policy_writes_entries(self):
        orch, backend, _ = _make_orch_with_policy(AlwaysWritePolicy())
        orch.on_episode_start(task_key="test_task")

        # Buffer 3 steps
        for i in range(3):
            keys = {"robot_state": torch.randn(32)}
            action = torch.randn(50, 32)
            orch.buffer_for_write(keys, action)

        # Trigger miss so write policy sees misses
        orch._miss_by_checkpoint[CheckpointID.CP1] = 1

        orch.on_episode_end()
        assert backend.count() == 3

    def test_never_policy_writes_nothing(self):
        orch, backend, _ = _make_orch_with_policy(NeverWritePolicy())
        orch.on_episode_start(task_key="test_task")

        for i in range(3):
            orch.buffer_for_write({"robot_state": torch.randn(32)}, torch.randn(50, 32))
        orch._miss_by_checkpoint[CheckpointID.CP1] = 5

        orch.on_episode_end()
        assert backend.count() == 0

    def test_on_any_miss_with_misses_writes(self):
        orch, backend, _ = _make_orch_with_policy(OnAnyMissWritePolicy())
        orch.on_episode_start(task_key="test_task")

        for i in range(2):
            orch.buffer_for_write({"robot_state": torch.randn(32)}, torch.randn(50, 32))
        orch._miss_by_checkpoint[CheckpointID.CP1] = 2

        orch.on_episode_end()
        assert backend.count() == 2

    def test_on_any_miss_no_misses_skips(self):
        orch, backend, _ = _make_orch_with_policy(OnAnyMissWritePolicy())
        orch.on_episode_start(task_key="test_task")

        for i in range(3):
            orch.buffer_for_write({"robot_state": torch.randn(32)}, torch.randn(50, 32))
        # No misses recorded

        orch.on_episode_end()
        assert backend.count() == 0

    def test_no_write_policy_writes_nothing(self):
        """Orchestrator without write_policy should skip writes."""
        orch, backend, _ = make_orchestrator()  # write_policy=None
        orch.on_episode_start(task_key="test_task")

        for i in range(3):
            orch.buffer_for_write({"robot_state": torch.randn(32)}, torch.randn(50, 32))
        orch._miss_by_checkpoint[CheckpointID.CP1] = 3

        orch.on_episode_end()
        assert backend.count() == 0

    def test_empty_episode_writes_nothing(self):
        orch, backend, _ = _make_orch_with_policy(AlwaysWritePolicy())
        orch.on_episode_start(task_key="test_task")
        # No buffer_for_write calls
        orch.on_episode_end()
        assert backend.count() == 0


# ---------------------------------------------------------------------------
# Entry chain linking
# ---------------------------------------------------------------------------


class TestEntryChainLinking:
    def test_prev_next_ids_correct(self):
        """Entries written by on_episode_end have correct prev_ids/next_ids."""
        orch, backend, _ = _make_orch_with_policy(AlwaysWritePolicy())
        orch.on_episode_start(task_key="test_task")

        for i in range(4):
            orch.buffer_for_write({"robot_state": torch.randn(32)}, torch.randn(50, 32))
        orch._miss_by_checkpoint[CheckpointID.CP1] = 1

        orch.on_episode_end()
        assert backend.count() == 4

        entries = list(backend._entries.values())
        # Sort by step_idx
        entries.sort(key=lambda e: e.step_idx)

        # First entry: no prev, has next
        assert entries[0].prev_ids == []
        assert len(entries[0].next_ids) == 1
        assert entries[0].next_ids[0] == entries[1].id

        # Middle entries: have both prev and next
        for i in range(1, 3):
            assert entries[i].prev_ids == [entries[i - 1].id]
            assert entries[i].next_ids == [entries[i + 1].id]

        # Last entry: has prev, no next
        assert entries[3].prev_ids == [entries[2].id]
        assert entries[3].next_ids == []

    def test_trajectory_id_shared(self):
        """All entries in a chain share the same trajectory_id."""
        orch, backend, _ = _make_orch_with_policy(AlwaysWritePolicy())
        orch.on_episode_start(task_key="test_task")

        for i in range(3):
            orch.buffer_for_write({"robot_state": torch.randn(32)}, torch.randn(50, 32))
        orch._miss_by_checkpoint[CheckpointID.CP1] = 1

        orch.on_episode_end()

        entries = list(backend._entries.values())
        traj_ids = {e.trajectory_id for e in entries}
        assert len(traj_ids) == 1
        assert None not in traj_ids

    def test_entry_id_format(self):
        """Entry ids follow '{trajectory_id}:{step_idx}' format."""
        orch, backend, _ = _make_orch_with_policy(AlwaysWritePolicy())
        orch.on_episode_start(task_key="test_task")

        for i in range(2):
            orch.buffer_for_write({"robot_state": torch.randn(32)}, torch.randn(50, 32))
        orch._miss_by_checkpoint[CheckpointID.CP1] = 1

        orch.on_episode_end()

        entries = sorted(backend._entries.values(), key=lambda e: e.step_idx)
        traj_id = entries[0].trajectory_id
        assert entries[0].id == f"{traj_id}:0"
        assert entries[1].id == f"{traj_id}:1"

    def test_task_key_stored_in_payload(self):
        """Entries carry the task_key from the episode."""
        orch, backend, _ = _make_orch_with_policy(AlwaysWritePolicy())
        orch.on_episode_start(task_key="my_task")

        orch.buffer_for_write({"robot_state": torch.randn(32)}, torch.randn(50, 32))
        orch._miss_by_checkpoint[CheckpointID.CP1] = 1

        orch.on_episode_end()

        entry = list(backend._entries.values())[0]
        assert entry.payload.task_key == "my_task"


# ---------------------------------------------------------------------------
# Episode lifecycle resets
# ---------------------------------------------------------------------------


class TestEpisodeReset:
    def test_on_episode_start_clears_buffers(self):
        orch, _, _ = _make_orch_with_policy(AlwaysWritePolicy())

        orch.buffer_for_write({"robot_state": torch.randn(32)}, torch.randn(50, 32))
        orch._miss_by_checkpoint[CheckpointID.CP1] = 5
        assert len(orch._episode_steps) == 1

        orch.on_episode_start(task_key="new_task")
        assert len(orch._episode_steps) == 0
        assert len(orch._miss_by_checkpoint) == 0

    def test_on_episode_end_clears_buffers(self):
        orch, _, _ = _make_orch_with_policy(AlwaysWritePolicy())
        orch.on_episode_start(task_key="task")

        orch.buffer_for_write({"robot_state": torch.randn(32)}, torch.randn(50, 32))
        orch._miss_by_checkpoint[CheckpointID.CP1] = 1

        orch.on_episode_end()
        assert len(orch._episode_steps) == 0
        assert len(orch._miss_by_checkpoint) == 0

    def test_multiple_episodes_independent(self):
        """Two consecutive episodes produce independent entry sets."""
        orch, backend, _ = _make_orch_with_policy(AlwaysWritePolicy())

        # Episode 1: 2 steps
        orch.on_episode_start(task_key="task1")
        for _ in range(2):
            orch.buffer_for_write({"robot_state": torch.randn(32)}, torch.randn(50, 32))
        orch._miss_by_checkpoint[CheckpointID.CP1] = 1
        orch.on_episode_end()
        assert backend.count() == 2

        # Episode 2: 3 steps
        orch.on_episode_start(task_key="task2")
        for _ in range(3):
            orch.buffer_for_write({"robot_state": torch.randn(32)}, torch.randn(50, 32))
        orch._miss_by_checkpoint[CheckpointID.CP1] = 1
        orch.on_episode_end()
        assert backend.count() == 5

        # Verify different trajectory_ids
        traj_ids = {e.trajectory_id for e in backend._entries.values()}
        assert len(traj_ids) == 2


# ---------------------------------------------------------------------------
# Miss counting integration
# ---------------------------------------------------------------------------


class TestMissCounting:
    def test_check_miss_increments_counter(self):
        """check() returning MISS increments _miss_by_checkpoint."""
        orch, _, _ = make_orchestrator()
        orch.on_episode_start(task_key="task")

        state = torch.randn(1, 32)
        orch.check(CheckpointID.CP1, stage1=make_stage1(state))
        orch.clear()

        assert orch._miss_by_checkpoint.get(CheckpointID.CP1, 0) == 1

    def test_gate_skip_increments_miss(self):
        """Gate skip also counts as a miss."""
        dims = {"robot_state": 32}
        backend = InMemoryBackend(dims)
        storage = CacheStorage(backend)

        class NeverGate:
            def __call__(self, *a, **kw):
                return False

        strategy = TestStorageSearchStrategy(storage, top_k=1)
        orch = CacheOrchestrator(
            storage,
            PlaceholderKeyBuilder(),
            gates=_wrap_per_checkpoint(NeverGate()),
            judges=_wrap_per_checkpoint(ThresholdJudge()),
            search_strategies=_wrap_per_checkpoint(strategy),
            timer=SystemTimer(enabled=False),
        )
        orch.on_episode_start(task_key="task")

        state = torch.randn(1, 32)
        orch.check(CheckpointID.CP1, stage1=make_stage1(state))
        orch.clear()

        assert orch._miss_by_checkpoint.get(CheckpointID.CP1, 0) == 1
