"""Data types of the trace serving mode (plan ``logs/cache_trace_mode_plan.log.md``).

Everything the trace path hands around is defined here so the writer, the
per-model adapters and the orchestrator agree on one vocabulary:

* ``TracePlan`` -- what to compute and record, frozen per connection;
* ``EpisodeIdentity`` -- who the episode is (client-provided identity);
* ``PerFieldTrace`` / ``SearchTrace`` / ``StepTrace`` -- one decision's record;
* ``TraceSink`` -- the per-connection protocol the interceptor talks to;
* ``TraceRuntime`` -- plan + sink + twin component set + private RNG helper;
* ``TraceWriteError`` / ``DrainReport`` / ``TraceWriteFailed`` -- the failure
  protocol of the asynchronous writer (plan §7.4).

The module is deliberately jax-free and does not import the orchestrator, the
interceptor, the models or the serving stack: the GR00T island imports it.
Depends on numpy, torch, ``openpi.cache.types`` and
``openpi.collect.data_collector.InferenceEmbeddings``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import torch

from openpi.cache.types import DenoiseSchedule
from openpi.collect.data_collector import InferenceEmbeddings

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

TRACE_SCHEMA_VERSION = 1

#: File-level attrs written by the trace writer (all ``trace_`` prefixed so a
#: legacy reader never sees an unknown top-level key it might misread).
ATTR_SCHEMA_VERSION = "trace_schema_version"
ATTR_NOISE_RECORDED = "trace_noise_actions_recorded"
ATTR_CLOSED_OK = "trace_closed_ok"
ATTR_TERMINAL = "trace_terminal"
ATTR_WRITE_ERRORS = "trace_write_errors"

#: Datasets inside ``step_XXXX/trace/`` may never look like a library snapshot.
NOISE_ACTION_RE = re.compile(r"^noise_action_\d+$")

#: JSON attrs above this many bytes are written as a vlen-string dataset under
#: ``trace/json/<name>`` and the attr is set to this marker (HDF5 attrs are
#: capped at 64 KiB).
ATTR_JSON_MAX_BYTES = 60000
ATTR_JSON_DATASET_MARKER = "@dataset"

TRACE_GROUP = "trace"

#: Executed-arm names.
ARM_FULL_HIT = "full_hit"
ARM_FULL_INFERENCE = "full_inference"
ARM_WARM_EXEC = "warm_exec"


def warm_arm_name(snapshot_index: int) -> str:
    """Dataset / arm name of the warm variant resumed from loop step ``snapshot_index``."""
    return f"warm_{int(snapshot_index):02d}"


# ---------------------------------------------------------------------------
# Plan and identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TracePlan:
    """What one connection computes and records on every decision.

    ``warm_tiers`` holds the executable warm snapshot indices (loop-step
    indices of ``schedule``, not the float ``t``), sorted in loop-execution
    order; empty when there is no orchestrator. ``save_timesteps`` is the
    snapshot set requested from the full inference: the whole schedule for a
    build, ``None`` (adapter default, no kwarg passed) otherwise.
    """

    model: str
    schedule: DenoiseSchedule
    checkpoint: Optional[str]
    warm_tiers: tuple[int, ...]
    record_noise_actions: bool
    save_timesteps: Optional[tuple[float, ...]]
    record_prefix_tokens: bool
    record_raw_images: bool
    record_model_images: bool
    record_query_keys: bool
    record_search: bool
    record_tokenized_prompt: bool
    raw_image_keys: tuple[str, ...]
    rng_isolation: str
    sidecar_jsonl: bool
    yaml_id: str = ""
    yaml_sha256: str = ""
    bundle_id: str = "default"
    connection_id: str = ""
    key_builder_type: Optional[str] = None
    library_sha256: Optional[str] = None
    concurrent: bool = False
    fail_loud: bool = False

    @property
    def fetch_top1(self) -> bool:
        """The FULL_HIT variant needs the top-1 payload whenever a checkpoint exists."""
        return self.checkpoint is not None


@dataclass(frozen=True)
class EpisodeIdentity:
    """Episode identity as the client declared it on ``episode_start``."""

    experiment: str
    task: str
    episode_id: int
    episode_name: str
    extra_metadata: dict[str, Any]

    @property
    def task_uid(self) -> str:
        return str(self.extra_metadata.get("task_uid", self.episode_id))

    @property
    def attempt(self) -> int:
        return int(self.extra_metadata.get("attempt", 1))

    @property
    def has_conductor_identity(self) -> bool:
        """True when ``task_uid`` was really sent, not derived from ``episode_id``."""
        return "task_uid" in self.extra_metadata


# ---------------------------------------------------------------------------
# Per-decision records
# ---------------------------------------------------------------------------


@dataclass
class PerFieldTrace:
    """Per-field similarity of the final twin top-k (plan §4.4).

    ``scores[j, f]`` is the score of candidate ``ids[j]`` on field
    ``fields[f]`` under ``kind_by_field[fields[f]]``; ``present[j, f]`` is
    False when the entry lacks that field (the score is then 0).
    ``current_step_wss`` is the fused weighted-score-sum of the current query
    only (``None`` unless the strategy is a weighted score sum).
    """

    fields: tuple[str, ...]
    ids: tuple[str, ...]
    scores: np.ndarray
    present: np.ndarray
    kind_by_field: dict[str, str]
    current_step_wss: Optional[np.ndarray] = None
    not_applicable_reason: Optional[str] = None
    field_metadata: Optional[dict[str, dict[str, Any]]] = None


@dataclass
class SearchTrace:
    """Search + verdict record of one checkpoint on one decision."""

    checkpoint: str
    gate_real_should_search: Optional[bool]
    gate_twin_should_search: bool
    real_topk_ids: Optional[list[str]]
    real_topk_scores: Optional[list[float]]
    real_verdict_json: Optional[str]
    twin_topk_ids: list[str]
    twin_topk_scores: list[float]
    twin_per_field: Optional[PerFieldTrace]
    twin_chain_scores: Optional[list[float]]
    twin_winner_per_field: dict[str, float]
    twin_field_own_margin: dict[str, float]
    twin_fused_margin: Optional[float]
    twin_n_results: Optional[int]
    twin_retrieval_signals_json: Optional[str]
    twin_proposed_verdict_json: str
    twin_verdict_json: str
    twin_validation_error: Optional[str]
    twin_replay_target: Optional[str]
    top1_entry_id: Optional[str]


@dataclass
class StepTrace:
    """Everything recorded for one decision (plan §7.2)."""

    legacy: InferenceEmbeddings
    raw_images: dict[str, np.ndarray]
    prompt: Optional[str]
    raw_state: Optional[np.ndarray]
    model_images: Optional[dict[str, np.ndarray]]
    image_mask: Optional[dict[str, bool]]
    tokenized_prompt: Optional[np.ndarray]
    query_keys: Optional[dict[str, np.ndarray]]
    search: Optional[SearchTrace]
    cp3_twin_json: Optional[str]
    action_full_hit: Optional[np.ndarray]
    action_warm: dict[int, Optional[np.ndarray]]
    action_warm_exec: Optional[np.ndarray]
    action_executed: np.ndarray
    executed_arm: str
    verdict: dict[str, Any]
    tier_status: dict[str, str]
    error_proxies: dict[str, float]
    timing_ms: dict[str, float]
    warm_index_map: dict[int, dict[str, Any]] = field(default_factory=dict)
    real_continuation_json: Optional[str] = None
    twin_continuation_json: Optional[str] = None
    #: Layout of ``raw_state`` when it is a concatenation of several wire keys
    #: (GR00T ``state.*``): ``(key, width)`` in concatenation order.
    raw_state_layout: Optional[tuple[tuple[str, int], ...]] = None


@dataclass
class CheckTrace:
    """What ``CacheOrchestrator.check(trace=True)`` records about one checkpoint.

    ``real_*`` fields are copied from the real pipeline's locals (``None``
    when the real gate skipped); ``twin_*`` fields come from the twin
    component set driven with ``force_search=True``. ``twin_verdict`` is the
    effective twin verdict (a proposed WARM that failed the snapshot check is
    downgraded to MISS and ``twin_validation_error`` says why). ``top1_*`` is
    the twin's top-1 candidate, the base of every warm variant.
    """

    checkpoint: str
    gate_real_should_search: Optional[bool]
    gate_twin_should_search: bool
    real_results: Optional[list[Any]]
    real_judge_result: Optional[Any]
    twin_results: list[Any]
    twin_step_features: Optional[Any]
    twin_per_field: Optional[PerFieldTrace]
    twin_chain_scores: Optional[list[float]]
    twin_retrieval_signals: Optional[Any]
    twin_proposed_verdict: Any
    twin_verdict: Any
    twin_validation_error: Optional[str]
    twin_replay_target: Optional[str]
    top1_entry_id: Optional[str]
    top1_payload: Optional[Any]


# ---------------------------------------------------------------------------
# Writer failure protocol
# ---------------------------------------------------------------------------


class TraceWriteFailed(RuntimeError):
    """Raised on the inference path when a build-mode writer has failed.

    Sticky: once the writer failed for a connection's episode, every later
    ``record_step`` / ``on_episode_start`` of that connection raises until the
    process is restarted (plan §7.4-4).
    """


@dataclass(frozen=True)
class TraceWriteError:
    token: str
    episode: str
    stage: str
    message: str
    timestamp: float


@dataclass
class DrainReport:
    """Outcome of ``TraceWriter.drain`` (plan §7.4-5)."""

    unfinished: tuple[str, ...]
    errors: tuple[TraceWriteError, ...]
    timed_out: bool
    thread_alive: bool
    fatal_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.unfinished and not self.errors and not self.timed_out and self.fatal_error is None


# ---------------------------------------------------------------------------
# Sink protocol and runtime
# ---------------------------------------------------------------------------


@runtime_checkable
class TraceSink(Protocol):
    """Per-connection sink the interceptor records into.

    Ordering contract: ``on_episode_start`` -> (``begin_step`` -> ``record_step``
    -> ``finish_step``)* -> ``on_episode_end`` | ``on_task_end``. The server
    processes one connection's frames sequentially, so a Close can never
    overlap an in-flight step; the sink asserts that invariant.
    """

    def on_episode_start(self, identity: EpisodeIdentity) -> None: ...

    def begin_step(self) -> None: ...

    def finish_step(self) -> None: ...

    def record_step(self, step: StepTrace) -> None: ...

    def on_episode_end(self, success: bool) -> None: ...

    def on_task_end(self) -> None: ...

    def close(self) -> None: ...


@dataclass
class TraceRuntime:
    """Plan + sink + twin component set for one connection."""

    plan: TracePlan
    sink: TraceSink
    twins: Optional[Any] = None

    def noise_generator(
        self,
        *,
        device: torch.device | str,
        identity: Optional[EpisodeIdentity],
        step_idx: int,
    ) -> Optional[torch.Generator]:
        """Private generator for a hit-step full inference (``verdict_aware``).

        Seeded from ``(task_uid, attempt, step_idx)`` through the same stable
        digest ``shadow_teacher`` uses, so the extra inference never touches
        the global RNG stream and a run can be replayed from its identity.
        Returns ``None`` under ``rng_isolation="global"``.
        """
        if self.plan.rng_isolation != "verdict_aware":
            return None
        from openpi.cache.shadow_teacher import stable_seed

        task_uid = identity.task_uid if identity is not None else "no-episode"
        attempt = identity.attempt if identity is not None else 1
        dev = torch.device(device)
        gen = torch.Generator(device=dev)
        gen.manual_seed(stable_seed(task_uid, attempt, int(step_idx)))
        return gen
