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

import inspect
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch

from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.factors.base import HistoryView, LibraryStats, OfflineWriter
from openpi.cache.components.gate import ClientControlledGate, GateFunction
from openpi.cache.components.judge import HitType, SimilarityJudge
from openpi.cache.components.key_builder import QueryKeyBuilder
from openpi.cache.components.payload_view import StoragePayloadView
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
    # Diagnostic per-step factor dump forwarded from the inner JudgeResult.
    # Populated only when the configured judge is a CompositeJudge with
    # ``export_factor_outputs: true``; otherwise None and Interceptor skips
    # the hit_meta side-channel.
    factor_outputs: Optional[dict] = None
    # True when this checkpoint actually searched the cache. Only a gate-skip
    # return sets this False. The gate-research collector filters non-searched
    # steps by this flag (selection bias C5) instead of inferring from
    # score/entry_id, which a cold-start / empty-library always-search MISS
    # also leaves None.
    searched: bool = True


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
        offline_writers: tuple["OfflineWriter", ...] = (),
        library_stats: Optional["LibraryStats"] = None,
    ) -> None:
        self._storage = storage
        self._key_builder = key_builder
        self._gates = gates
        self._judges = judges
        self._search_strategies = search_strategies
        self._timer = timer if timer is not None else SystemTimer(enabled=False)
        self._write_policy = write_policy
        self._step_counter: int = 0

        # B2 wiring — populated by config builder when composite judges
        # include OfflineWriter-capable factors (currently F1b-A / F1b-T).
        # Empty default keeps backward compatibility: episode-end fast-paths
        # via the merge loop's `if self._offline_writers and ...` guard.
        self._offline_writers: tuple["OfflineWriter", ...] = tuple(offline_writers)
        self._library_stats: Optional["LibraryStats"] = library_stats

        # True when at least one configured gate consumes the client-side
        # gate_decision signal. Interceptor uses this flag to fail loud on
        # config mismatch when obs carries '__gate_decision__'.
        self.accepts_client_signal: bool = any(
            isinstance(g, ClientControlledGate) for g in self._gates.values()
        )

        # Episode-level buffers
        self._episode_steps: list[StepRecord] = []
        self._miss_by_checkpoint: dict[CheckpointID, int] = {}
        self._current_task_key: str = ""
        self._current_episode_id: str = ""
        # Per-episode identity payload from `episode_start.extra_metadata`
        # (e.g. {"task_id": int, "orig_init_state_idx": int}). Cleared on
        # episode_end so cross-episode bleed is impossible. Read by
        # DumpingJudge through the lifecycle broadcast below.
        self._current_episode_extra: dict = {}

        # B1 — verdict-factor history buffers. Populated by broadcast_action
        # (actions) and check() at the anchor checkpoint (states); read by
        # CompositeJudge factor extractors via HistoryView. Cleared by
        # _reset_episode_buffer (covers on_task_begin / on_episode_start /
        # on_episode_end / on_task_end leak fix).
        self._action_history: list[torch.Tensor] = []
        self._state_history: list[torch.Tensor] = []
        # State append happens at the lowest-value enabled checkpoint so
        # multi-checkpoint configs (CP1 + CP3) record state exactly once
        # per inference cycle. Falls back to None when no gates are wired
        # (e.g. unit tests that only exercise lifecycle helpers).
        if self._gates:
            self._state_history_anchor_cp: Optional[CheckpointID] = min(
                self._gates.keys(), key=lambda cp: cp.value
            )
        else:
            self._state_history_anchor_cp = None

        # Per-strategy search session ids registered with the backend in the
        # current episode. Populated by `_broadcast_episode_start` after each
        # strategy mints its own sid; drained by `_close_current_search_sessions`
        # on episode_end / task_end / stale clear. NEVER mutated outside those
        # two helpers.
        self._current_strategy_session_ids: list[str] = []

        # Register fine-grained probes for each checkpoint's sub-steps.
        for cp in ("cp1", "cp3"):
            for step in ("collect", "gate", "build", "search", "judge", "fetch"):
                self._timer.register_probe(f"{cp}_{step}", backend="cpu")

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def key_builder(self) -> QueryKeyBuilder:
        """Public accessor for the bound KeyBuilder instance.

        Used by InferenceInterceptor to perform optional `attach_model`
        hook on type-specific builders (e.g. cp1_llm_layer_extract). The
        underlying reference is set once in __init__ and immutable;
        exposing it as a property keeps the interceptor → builder wiring
        explicit and unit-testable.
        """
        return self._key_builder

    # ------------------------------------------------------------------
    # Prefill mode (delegated to the underlying storage)
    # ------------------------------------------------------------------

    @contextmanager
    def prefill_mode(self, payload: CachePayload):
        """Scope storage prefill mode to a ``with`` block.

        While the block is active, the underlying CacheStorage returns a
        synthetic FULL_HIT carrying ``payload`` for any search, regardless
        of the actual vector store. Used by
        ``InferenceInterceptor.prefill_trajectory`` to drive history from
        ground-truth obs+action pairs.

        Exit is unconditional — even if the enclosed block raises, storage
        leaves prefill mode before the exception propagates.
        """
        self._storage.enter_prefill_mode(payload)
        try:
            yield
        finally:
            self._storage.exit_prefill_mode()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def on_task_begin(self, task_key: str = "") -> None:
        """Reset per-task state. Called when a client connection opens."""
        self._step_counter = 0
        self._current_task_key = task_key
        self._reset_episode_buffer()
        self._broadcast_episode_start()

    def on_episode_start(
        self,
        task_key: str = "",
        episode_id: str = "",
        extra_metadata: dict | None = None,
    ) -> None:
        """Reset per-episode state. Called when simulator sends episode_start.

        ``extra_metadata`` carries episode-level identity injected through the
        wire-level ``episode_start.__extra__`` channel (see
        ``InferenceInterceptor.on_episode_start``); it is stashed on
        ``self._current_episode_extra`` and broadcast to lifecycle-aware
        components (currently only ``DumpingJudge``) via the existing
        ``_safe_call_lifecycle`` filtering path.
        """
        self._step_counter = 0
        if task_key:
            self._current_task_key = task_key
        self._current_episode_id = episode_id
        # Reset first (clears stale extra from previous episode), then set
        # the new identity so it survives the broadcast below.
        self._reset_episode_buffer()
        self._current_episode_extra = dict(extra_metadata or {})
        self._broadcast_episode_start()

    def _broadcast_episode_start(self) -> None:
        """Lifecycle entry point shared by on_task_begin and on_episode_start.

        Atomically performs three steps so all broadcast paths obtain the
        same session-register guarantee:
          (1) close any stale strategy sessions left over from a previous
              episode (defensive — handles repeated task_begin or skipped
              episode_end);
          (2) broadcast on_episode_start to every component (each strategy
              mints its own per-episode search session id);
          (3) collect each strategy's freshly-minted sid via
              get_search_session_id() and register it with the backend
              through `storage.open_search_session(sid)` so the backend's
              mutation guard activates *before* any search runs.
        """
        # (1) Close any stale sessions (idempotent).
        self._close_current_search_sessions()

        # (2) Broadcast lifecycle to all components.
        # Pass `extra_metadata` only to judges (DumpingJudge consumes it for
        # JSONL identity); _safe_call_lifecycle filters via inspect.signature
        # so existing kwargs-free judges (AlwaysHit / AlwaysWarmStart /
        # Threshold / CompositeJudge) silently swallow the kwarg.
        self._safe_call_lifecycle(self._key_builder, "on_episode_start")
        for strategy in self._search_strategies.values():
            self._safe_call_lifecycle(strategy, "on_episode_start")
        for gate in self._gates.values():
            # G0a: broadcast task_key to the gate. _safe_call_lifecycle filters
            # by signature, so legacy gates (on_episode_start(self)) ignore it;
            # gates that declare on_episode_start(self, task_key="") receive it.
            self._safe_call_lifecycle(
                gate, "on_episode_start", task_key=self._current_task_key
            )
        for judge in self._judges.values():
            self._safe_call_lifecycle(
                judge, "on_episode_start",
                extra_metadata=self._current_episode_extra,
            )

        # (3) Collect strategy-minted sids and register with the backend.
        # Sequential single-thread loop here; no contention with concurrent
        # search because no search is in flight inside lifecycle hooks.
        for strategy in self._search_strategies.values():
            getter = getattr(strategy, "get_search_session_id", None)
            if getter is None:
                continue
            sid = getter()
            if sid is None:
                continue
            self._storage.open_search_session(sid)
            self._current_strategy_session_ids.append(sid)

    def _close_current_search_sessions(self) -> None:
        """Close all currently-registered strategy search sessions.

        Idempotent. The single cleanup entry point — on_episode_end (via
        finally), on_task_end, and `_broadcast_episode_start` step (1) all
        funnel through here. NO other path may mutate
        `_current_strategy_session_ids` directly.
        """
        for sid in self._current_strategy_session_ids:
            self._storage.close_search_session(sid)
        self._current_strategy_session_ids = []

    @staticmethod
    def _safe_call_lifecycle(component, method_name: str, **kwargs) -> None:
        """Call component.method_name(**kwargs) passing only kwargs the
        callee actually accepts. Older lifecycle implementations without the
        new kwargs continue to work unchanged.
        """
        method = getattr(component, method_name, None)
        if method is None:
            return
        if kwargs:
            sig = inspect.signature(method)
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        method(**kwargs)

    def _reset_episode_buffer(self) -> None:
        self._episode_steps.clear()
        self._miss_by_checkpoint.clear()
        # B1 — clear verdict-factor history. Required by HistoryView's
        # gap-free semantics (next episode should start with empty
        # action / state lists, not bleed across episode boundaries).
        self._action_history.clear()
        self._state_history.clear()
        # Identity payload from `episode_start.extra_metadata` is also
        # cleared here so any stray verdict between on_episode_end and the
        # next on_episode_start (defensive) sees an empty dict rather than
        # last episode's identity.
        self._current_episode_extra = {}

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
        # B1 — record into the orchestrator-owned action history (read by
        # CompositeJudge factors via HistoryView). Detaches the chunk so
        # the buffer never holds onto autograd state from inference.
        # Pi05 broadcasts the full action_chunk [chunk_len, A] per inference,
        # but F1a-A's `_build_action_splice` and the test contract in
        # tests/cache/components/factors/test_runtime_continuity.py both
        # expect per-step [A]-shaped tensors that mirror _state_history's
        # single-state-per-step convention. Reduce to action_chunk[0] (the
        # first action, which is what the LIBERO eval loop pops and
        # env.step()s next from this cycle's chunk).
        chunk_cpu = action_chunk.detach().cpu()
        first_action = chunk_cpu[0] if chunk_cpu.dim() >= 2 else chunk_cpu
        self._action_history.append(first_action)

    def _feed_verdict_to_gate(
        self,
        checkpoint_id: CheckpointID,
        *,
        hit_type: HitType,
        cp1_score: Optional[float],
        winner_id: Optional[str],
        start_t: Optional[float],
        searched: bool,
    ) -> None:
        """G0a: feed this step's verdict back to the checkpoint's gate.

        Called on every check() return path so a stateful gate (e.g.
        ScoreHysteresisGate) can condition the NEXT step's decision on the
        verdict the judge just produced. Per-checkpoint (only the gate that ran
        this checkpoint); guarded by hasattr so gates without record_verdict
        (all legacy gates) are unaffected. Pure gate-local state mutation — no
        storage lock, no wire. searched=False marks a gate-skip step (no search
        ran; cp1_score is None).
        """
        gate = self._gates.get(checkpoint_id)
        if gate is not None and hasattr(gate, "record_verdict"):
            gate.record_verdict(
                checkpoint_id,
                hit_type=hit_type,
                cp1_score=cp1_score,
                winner_id=winner_id,
                start_t=start_t,
                searched=searched,
            )

    # ------------------------------------------------------------------
    # Cache check pipeline
    # ------------------------------------------------------------------

    def check(
        self,
        checkpoint_id: CheckpointID,
        *,
        request_context: dict | None = None,
        **stage_outputs,
    ) -> CheckResult:
        """Cache check pipeline: collect -> gate -> build -> search -> judge -> fetch.

        For CP1, Judge returns JudgeResult with three possible outcomes:
        FULL_HIT, WARM_START (with start_t), or MISS.  On WARM_START,
        payload completeness is validated here; incomplete payloads are
        downgraded to MISS.

        Args:
            checkpoint_id: CP1 or CP3.
            request_context: Optional per-request dict forwarded to the gate.
                Default gates ignore it; ``ClientControlledGate`` consumes
                ``gate_decision``. Kwarg-only to keep it out of the
                ``**stage_outputs`` passthrough.
            **stage_outputs: Stage tensors forwarded to ``KeyBuilder.collect``.

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
            should_search = gate(
                checkpoint_id, self._key_builder.cached_data, request_context
            )
        logger.info("[step %d] %s gate: %s", self._step_counter, prefix, "SEARCH" if should_search else "SKIP")

        # build() always executes (even on gate skip) for trajectory completeness
        with self._timer.measure(f"{prefix}_build"):
            query_keys = self._key_builder.build(checkpoint_id)

        # B1 — record state history at the anchor checkpoint.
        # Anchor = lowest-value enabled checkpoint, so multi-CP configs
        # (CP1+CP3) only append once per inference cycle. Fires regardless
        # of gate / hit-type outcome so the history stays gap-free across
        # the episode (verdict factors use windowed scans that need
        # contiguous samples).
        if (
            checkpoint_id == self._state_history_anchor_cp
            and "robot_state" in query_keys
        ):
            self._state_history.append(query_keys["robot_state"].detach().cpu())

        if not should_search:
            # Gate skip: record query_keys to strategy history (trajectory gap-free)
            if hasattr(strategy, 'record_query_keys'):
                strategy.record_query_keys(query_keys)
            self._miss_by_checkpoint[checkpoint_id] = self._miss_by_checkpoint.get(checkpoint_id, 0) + 1
            if checkpoint_id == CheckpointID.CP1:
                self._step_counter += 1
            # searched=False: gate skipped the search. Distinguishes this from a
            # real always-search MISS (which leaves searched=True) for the
            # gate-research collector's C5 selection-bias filter.
            self._feed_verdict_to_gate(
                checkpoint_id, hit_type=HitType.MISS, cp1_score=None,
                winner_id=None, start_t=None, searched=False,
            )
            return CheckResult(hit_type=HitType.MISS, query_keys=query_keys, searched=False)

        with self._timer.measure(f"{prefix}_search"):
            ctx = SearchContext(
                query_keys=query_keys,
                checkpoint_id=checkpoint_id,
                current_step=self._step_counter,
                task_key=self._current_task_key or None,
            )
            results = strategy.search(ctx)

        # B1 — build PayloadView + HistoryView and inject into the judge.
        # PayloadView is per-check() so its memo (entry_id -> payload) gets
        # GC'd when this call returns. HistoryView snapshots the current
        # buffers (no aliasing — list copies; tensors stay shared).
        view = StoragePayloadView(self._storage)
        history = HistoryView(
            actions=list(self._action_history),
            states=list(self._state_history),
        )

        with self._timer.measure(f"{prefix}_judge"):
            judge_result = judge(
                results, checkpoint_id, self._key_builder.cached_data,
                view=view, history=history,
            )
        hit_type = judge_result.hit_type
        winner_id = judge_result.winner_id
        start_t = judge_result.start_t
        # Forward the optional CompositeJudge diagnostic dump on every exit
        # path below (FULL_HIT / WARM_START / WARM_START-downgrade-to-MISS /
        # post-judge MISS). Early gate / cache-disabled returns above never
        # touched a judge and stay None.
        factor_outputs = getattr(judge_result, "factor_outputs", None)
        top_score = results[0].score if results else None
        logger.info("[step %d] %s judge: %s (top_score=%s, winner=%s)", self._step_counter, prefix, hit_type.name, top_score, winner_id)

        if hit_type == HitType.MISS:
            self._miss_by_checkpoint[checkpoint_id] = self._miss_by_checkpoint.get(checkpoint_id, 0) + 1

        if checkpoint_id == CheckpointID.CP1:
            self._step_counter += 1

        if hit_type in (HitType.FULL_HIT, HitType.WARM_START) and winner_id is not None:
            with self._timer.measure(f"{prefix}_fetch"):
                # Route through PayloadView so a Judge that already
                # fetched the winner during extract (e.g. F1a-T touched
                # winner.query_keys, F1b read winner.factors) doesn't
                # incur a duplicate storage roundtrip.
                payload = view.get(winner_id)

            if hit_type == HitType.WARM_START:
                if (not payload.intermediates
                        or payload.denoising_num_steps is None
                        or start_t not in payload.intermediates):
                    logger.warning(
                        "[step %d] WARM_START payload incomplete (start_t=%s, "
                        "has_intermediates=%s), downgrade to MISS.",
                        self._step_counter - 1, start_t,
                        payload.intermediates is not None,
                    )
                    self._miss_by_checkpoint[checkpoint_id] = (
                        self._miss_by_checkpoint.get(checkpoint_id, 0) + 1
                    )
                    self._feed_verdict_to_gate(
                        checkpoint_id, hit_type=HitType.MISS,
                        cp1_score=results[0].score, winner_id=winner_id,
                        start_t=start_t, searched=True,
                    )
                    return CheckResult(
                        hit_type=HitType.MISS, query_keys=query_keys,
                        score=results[0].score, entry_id=winner_id,
                        factor_outputs=factor_outputs,
                    )

            self._feed_verdict_to_gate(
                checkpoint_id, hit_type=hit_type, cp1_score=results[0].score,
                winner_id=winner_id, start_t=start_t, searched=True,
            )
            return CheckResult(
                hit_type=hit_type, payload=payload, start_t=start_t,
                score=results[0].score, entry_id=winner_id, query_keys=query_keys,
                factor_outputs=factor_outputs,
            )

        self._feed_verdict_to_gate(
            checkpoint_id, hit_type=HitType.MISS, cp1_score=top_score,
            winner_id=winner_id, start_t=start_t, searched=True,
        )
        return CheckResult(
            hit_type=HitType.MISS, query_keys=query_keys,
            score=top_score, entry_id=winner_id,
            factor_outputs=factor_outputs,
        )

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

        Cache sessions are closed in `finally` so that early-return branches
        (no steps / no write_policy / write_policy declines) still release
        every registered sid; otherwise the backend mutation guard would stay
        raised after the episode has ended.
        """
        try:
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
        finally:
            self._close_current_search_sessions()

    def on_task_end(self) -> None:
        """Connection close / abnormal-exit cleanup.

        Idempotent bottom guard for paths where on_episode_end may not fire
        (e.g. WebSocket disconnect mid-episode). Forwarded to from
        InferenceInterceptor.on_task_end().

        B1 — also clears the per-episode buffers (action / state history,
        episode steps, miss counters). Pre-existing behavior left these
        live until the next on_task_begin / on_episode_start, which leaks
        across reconnects. Cleared here so verdict-factor history can't
        bleed from one connection's tail into the next connection's head.
        """
        self._close_current_search_sessions()
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

        # B2 — invoke each OfflineWriter once over the freshly built chain
        # and merge per-entry factor dicts into payload.factors. Skipped
        # in the default (no-writer) configuration so existing call sites
        # see byte-identical behavior.
        if self._offline_writers and self._library_stats is not None:
            for writer in self._offline_writers:
                per_entry_factors = writer.compute_for_episode(
                    entries, self._library_stats,
                )
                for entry, factors in zip(entries, per_entry_factors, strict=True):
                    if entry.payload.factors is None:
                        entry.payload.factors = {}
                    entry.payload.factors.update(factors)

        return entries

    def clear(self) -> None:
        """Release per-cycle state. Called at end of each inference cycle."""
        self._key_builder.clear()
