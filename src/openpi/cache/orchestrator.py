"""CacheOrchestrator: coordinate cache check and write operations.

Combines pluggable components (KeyBuilder, Gate, Judge, SearchStrategy,
WritePolicy) with CacheStorage. All storage interaction goes through
CacheStorage facade -- never touches VectorStoreBackend directly.

Data flow overview:
  check():  Interceptor -> collect -> gate -> build -> SearchContext
            -> search_strategy.search(ctx) -> judge -> CheckResult
  buffer_for_write() + on_episode_end():
            Interceptor -> buffer steps -> episode end -> WritePolicy
            -> build entry chain -> batch_insert

Coupling map:
  DEPENDS ON:  QueryKeyBuilder, GateFunction, SimilarityJudge, SearchStrategy,
               WritePolicy (components),
               CacheStorage facade -- insert/fetch_payload/batch_insert,
               SystemTimer -- optional timing
  CONSUMED BY: InferenceInterceptor (calls check/buffer_for_write/broadcast_action/
               on_episode_start/on_episode_end at cache checkpoint slots)
  DOES NOT depend on: VectorStoreBackend, Qdrant, or any specific backend
  SHARES: CacheStorage instance with SearchStrategy
  IF CHANGED:  Interceptor's cache integration logic may need updating
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch

from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import GateFunction
from openpi.cache.components.judge import HitType, SimilarityJudge
from openpi.cache.components.key_builder import QueryKeyBuilder
from openpi.cache.components.search_strategy import SearchContext, SearchStrategy
from openpi.cache.storage_types import (
    CacheEntry,
    CachePayload,
    EpisodeRecord,
    StepRecord,
)
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a cache check at one checkpoint.

    Data flow: Orchestrator.check() -> CheckResult -> Interceptor (decision point)
    Coupling:
      - CONSUMED BY: InferenceInterceptor (reads hit_type to decide stage skip,
                     reads query_keys for buffer_for_write)
      - CONTAINS: CachePayload from CacheStorage.fetch_payload() (Step 3 type)
    """

    hit_type: HitType
    payload: Optional[CachePayload] = None  # non-None on FULL_HIT or WARM_START
    score: Optional[float] = None
    entry_id: Optional[str] = None
    query_keys: Optional[dict[str, torch.Tensor]] = None  # filled on all paths
    start_t: float | None = None  # Judge-decided start_t, only on WARM_START


class CacheOrchestrator:
    """Orchestrate cache check and episode-level write operations.

    Combines pluggable components (KeyBuilder, Gate, Judge, SearchStrategy,
    WritePolicy) with CacheStorage.

    Data flow overview:
      check():  Interceptor -> collect -> gate -> build -> SearchContext
                -> search_strategy.search(ctx) -> judge -> CheckResult
      buffer_for_write() + on_episode_end():
                Interceptor -> buffer steps -> episode end -> WritePolicy
                -> build entry chain -> batch_insert

    Per-checkpoint dispatch:
      gates, judges, search_strategies are dict[CheckpointID, Component].
      CP1 and CP3 can use different Gate/Judge/SearchStrategy instances.
      key_builder is shared across checkpoints (same data extraction logic).

    Step counter:
      _step_counter tracks inference cycles within a task (client connection).
      Reset by on_task_begin()/on_episode_start(), incremented by check() on CP1.
    """

    def __init__(
        self,
        storage: CacheStorage,
        key_builder: QueryKeyBuilder,
        gates: dict[CheckpointID, GateFunction],
        judges: dict[CheckpointID, SimilarityJudge],
        search_strategies: dict[CheckpointID, SearchStrategy],
        timer: Optional[SystemTimer] = None,
        write_policy=None,
    ) -> None:
        self._storage = storage
        self._key_builder = key_builder
        self._gates = gates
        self._judges = judges
        self._search_strategies = search_strategies
        self._timer = timer if timer is not None else SystemTimer(enabled=False)
        self._write_policy = write_policy
        self._step_counter: int = 0

        # Episode-level buffers
        self._episode_steps: list[StepRecord] = []
        self._miss_by_checkpoint: dict[CheckpointID, int] = {}
        self._current_task_key: str = ""
        self._current_episode_id: str = ""

        # Register fine-grained probes for each checkpoint's sub-steps.
        for cp in ("cp1", "cp3"):
            for step in ("collect", "gate", "build", "search", "judge", "fetch"):
                self._timer.register_probe(f"{cp}_{step}", backend="cpu")

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def on_task_begin(self, task_key: str = "") -> None:
        """Reset per-task state. Called when a client connection opens."""
        self._step_counter = 0
        self._current_task_key = task_key
        self._reset_episode_buffer()
        self._broadcast_episode_start()

    def on_episode_start(self, task_key: str = "", episode_id: str = "") -> None:
        """Reset per-episode state. Called when simulator sends episode_start."""
        self._step_counter = 0
        if task_key:
            self._current_task_key = task_key
        self._current_episode_id = episode_id
        self._reset_episode_buffer()
        self._broadcast_episode_start()

    def _broadcast_episode_start(self) -> None:
        """Notify all components to clear their history buffers."""
        if hasattr(self._key_builder, 'on_episode_start'):
            self._key_builder.on_episode_start()
        for strategy in self._search_strategies.values():
            if hasattr(strategy, 'on_episode_start'):
                strategy.on_episode_start()
        for gate in self._gates.values():
            if hasattr(gate, 'on_episode_start'):
                gate.on_episode_start()
        for judge in self._judges.values():
            if hasattr(judge, 'on_episode_start'):
                judge.on_episode_start()

    def _reset_episode_buffer(self) -> None:
        self._episode_steps.clear()
        self._miss_by_checkpoint.clear()

    # ------------------------------------------------------------------
    # Action broadcast
    # ------------------------------------------------------------------

    def broadcast_action(self, action_chunk: torch.Tensor) -> None:
        """Broadcast action to all components for trajectory history.

        Called by Interceptor after action is produced (cache hit or inference).
        Must be called after check() returns (all locks released).
        """
        for strategy in self._search_strategies.values():
            if hasattr(strategy, 'record_action'):
                strategy.record_action(action_chunk)
        for gate in self._gates.values():
            if hasattr(gate, 'record_action'):
                gate.record_action(action_chunk)
        for judge in self._judges.values():
            if hasattr(judge, 'record_action'):
                judge.record_action(action_chunk)

    # ------------------------------------------------------------------
    # Cache check pipeline
    # ------------------------------------------------------------------

    def check(self, checkpoint_id: CheckpointID, **stage_outputs) -> CheckResult:
        """Cache check pipeline: collect -> gate -> build -> search -> judge -> fetch.

        For CP1, Judge returns JudgeResult with three possible outcomes:
        FULL_HIT, WARM_START (with start_t), or MISS.  On WARM_START,
        payload completeness is validated here; incomplete payloads are
        downgraded to MISS.

        CheckResult.query_keys is filled on all return paths.
        """
        prefix = checkpoint_id.name.lower()

        # If this checkpoint is not configured, skip gracefully.
        if checkpoint_id not in self._gates:
            if checkpoint_id == CheckpointID.CP1:
                self._step_counter += 1
            return CheckResult(hit_type=HitType.MISS)

        gate = self._gates[checkpoint_id]
        judge = self._judges[checkpoint_id]
        strategy = self._search_strategies[checkpoint_id]

        with self._timer.measure(f"{prefix}_collect"):
            self._key_builder.collect(checkpoint_id, **stage_outputs)

        with self._timer.measure(f"{prefix}_gate"):
            should_search = gate(checkpoint_id, self._key_builder.cached_data)
        logger.info("[step %d] %s gate: %s", self._step_counter, prefix, "SEARCH" if should_search else "SKIP")

        # build() always executes (even on gate skip) for trajectory completeness
        with self._timer.measure(f"{prefix}_build"):
            query_keys = self._key_builder.build(checkpoint_id)

        if not should_search:
            # Gate skip: record query_keys to strategy history (trajectory gap-free)
            if hasattr(strategy, 'record_query_keys'):
                strategy.record_query_keys(query_keys)
            self._miss_by_checkpoint[checkpoint_id] = self._miss_by_checkpoint.get(checkpoint_id, 0) + 1
            if checkpoint_id == CheckpointID.CP1:
                self._step_counter += 1
            return CheckResult(hit_type=HitType.MISS, query_keys=query_keys)

        with self._timer.measure(f"{prefix}_search"):
            ctx = SearchContext(
                query_keys=query_keys,
                checkpoint_id=checkpoint_id,
                current_step=self._step_counter,
                task_key=self._current_task_key or None,
            )
            results = strategy.search(ctx)

        with self._timer.measure(f"{prefix}_judge"):
            judge_result = judge(
                results, checkpoint_id, self._key_builder.cached_data
            )
        hit_type = judge_result.hit_type
        winner_id = judge_result.winner_id
        start_t = judge_result.start_t
        top_score = results[0].score if results else None
        logger.info("[step %d] %s judge: %s (top_score=%s, winner=%s)", self._step_counter, prefix, hit_type.name, top_score, winner_id)

        if hit_type == HitType.MISS:
            self._miss_by_checkpoint[checkpoint_id] = self._miss_by_checkpoint.get(checkpoint_id, 0) + 1

        if checkpoint_id == CheckpointID.CP1:
            self._step_counter += 1

        if hit_type in (HitType.FULL_HIT, HitType.WARM_START) and winner_id is not None:
            with self._timer.measure(f"{prefix}_fetch"):
                payload = self._storage.fetch_payload(winner_id)

            if hit_type == HitType.WARM_START:
                if (not payload.intermediates
                        or payload.denoising_num_steps is None
                        or start_t not in payload.intermediates):
                    logger.debug(
                        "[step %d] WARM_START payload incomplete (start_t=%s, "
                        "has_intermediates=%s), downgrade to MISS.",
                        self._step_counter - 1, start_t,
                        payload.intermediates is not None,
                    )
                    self._miss_by_checkpoint[checkpoint_id] = (
                        self._miss_by_checkpoint.get(checkpoint_id, 0) + 1
                    )
                    return CheckResult(
                        hit_type=HitType.MISS, query_keys=query_keys,
                        score=results[0].score, entry_id=winner_id,
                    )

            return CheckResult(
                hit_type=hit_type, payload=payload, start_t=start_t,
                score=results[0].score, entry_id=winner_id, query_keys=query_keys,
            )

        return CheckResult(hit_type=HitType.MISS, query_keys=query_keys)

    def build_keys(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        """Build query keys from already-collected stage outputs.

        Must be called in the main thread (accesses GPU tensors via key_builder).
        Returns CPU tensors safe for cross-thread use.
        """
        return self._key_builder.build(checkpoint_id)

    # ------------------------------------------------------------------
    # Episode-level write path
    # ------------------------------------------------------------------

    def buffer_for_write(
        self,
        query_keys: dict[str, torch.Tensor],
        action_chunk: torch.Tensor,
        intermediates: Optional[dict[float, torch.Tensor]] = None,
        denoising_num_steps: Optional[int] = None,
    ) -> None:
        """Buffer current step for episode-end batch write.

        Called by Interceptor after action is produced (cache hit or inference).
        Not called inside check() — action is not available at check time.
        """
        self._episode_steps.append(StepRecord(
            query_keys=query_keys,
            action_chunk=action_chunk,
            intermediates=intermediates,
            denoising_num_steps=denoising_num_steps,
        ))

    def on_episode_end(self) -> None:
        """Episode ended. Consult WritePolicy and optionally batch-write trajectory.

        Write flow:
          1. Build EpisodeRecord from accumulated buffers
          2. WritePolicy.should_write() decides
          3. If yes: build linked CacheEntry chain, batch_insert
          4. Reset buffers
        """
        if not self._episode_steps:
            self._reset_episode_buffer()
            return

        if self._write_policy is None:
            self._reset_episode_buffer()
            return

        record = EpisodeRecord(
            steps=self._episode_steps,
            task_key=self._current_task_key,
            miss_by_checkpoint=dict(self._miss_by_checkpoint),
            total_steps=len(self._episode_steps),
        )

        if not self._write_policy.should_write(record):
            self._reset_episode_buffer()
            return

        entries = self._build_entry_chain(record)
        if entries:
            self._storage.batch_insert(entries)

        self._reset_episode_buffer()

    def _build_entry_chain(self, record: EpisodeRecord) -> list[CacheEntry]:
        """Convert EpisodeRecord to a linked list of CacheEntry objects."""
        trajectory_id = str(uuid.uuid4())
        entries: list[CacheEntry] = []

        for step_idx, step in enumerate(record.steps):
            entry_id = f"{trajectory_id}:{step_idx}"
            entry = CacheEntry(
                id=entry_id,
                checkpoint_id=CheckpointID.CP1,
                query_keys=step.query_keys,
                payload=CachePayload(
                    action_chunk=step.action_chunk,
                    task_key=record.task_key,
                    intermediates=step.intermediates,
                    denoising_num_steps=step.denoising_num_steps,
                ),
                step_idx=step_idx,
                trajectory_id=trajectory_id,
            )
            entries.append(entry)

        # Link prev_ids / next_ids
        for i in range(len(entries)):
            if i > 0:
                entries[i].prev_ids = [entries[i - 1].id]
            if i < len(entries) - 1:
                entries[i].next_ids = [entries[i + 1].id]

        return entries

    def clear(self) -> None:
        """Release per-cycle state. Called at end of each inference cycle."""
        self._key_builder.clear()
