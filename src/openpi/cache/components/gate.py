"""Gate: decide whether to perform a cache lookup.

Data flow: KeyBuilder.cached_data (GPU) -> gate() -> bool

Coupling map:
  DEPENDS ON:  nothing (leaf component)
  MAY DEPEND ON: KeyBuilder.cached_data (read-only, for future state-change gates)
  CONSUMED BY: CacheOrchestrator.check() — if False, skip search entirely
  DOES NOT interact with: CacheStorage or any Step 3 component
  IF CHANGED:  Only affects Orchestrator's search frequency, no downstream impact
"""

from __future__ import annotations

import logging
import math
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from openpi.cache.components.judge import HitType
from openpi.cache.types import CheckpointID

logger = logging.getLogger(__name__)


@runtime_checkable
class GateFunction(Protocol):
    """Decide whether to perform a cache lookup.

    Data flow: KeyBuilder.cached_data (GPU) -> gate() -> bool
    Coupling:
      - MAY DEPEND ON: KeyBuilder.cached_data (read-only, for state-change gates)
      - CONSUMED BY: CacheOrchestrator.check() — if False, skip search entirely
      - DOES NOT interact with: CacheStorage or any Step 3 component
      - IF CHANGED: Only affects Orchestrator's search frequency, no downstream impact

    Optional lifecycle hooks (additive; a gate implements only what it needs —
    the orchestrator guards each call so gates that omit them are unaffected):
      - ``on_episode_start(self, task_key: str = "")`` — reset per-episode state.
        The orchestrator broadcasts ``task_key`` via ``inspect.signature``
        filtering, so a no-arg ``on_episode_start(self)`` silently ignores it.
      - ``record_action(self, action_chunk)`` — receive the broadcast action.
      - ``record_verdict(self, checkpoint_id, *, hit_type, cp1_score, winner_id,
        start_t, searched)`` — receive this step's own verdict AFTER the judge
        runs, so a stateful gate can condition the NEXT step's decision on it.
        Broadcast by ``CacheOrchestrator.check()`` under a ``hasattr`` guard;
        stateless gates simply omit the method. Consumed by
        ``ScoreHysteresisGate`` (server-side N1).
    """

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        request_context: dict | None = None,
    ) -> bool:
        """Return True if cache should be searched, False to skip.

        Args:
            checkpoint_id: CP1 or CP3.
            cached_data: Raw tensors from KeyBuilder.cached_data (on GPU).
            request_context: Optional per-request dict forwarded by the
                interceptor. Default gates ignore it; gates like
                ``ClientControlledGate`` consume ``gate_decision``.
        """
        ...


class AlwaysSearchGate:
    """Always search the cache. Simplest gate for initial development.

    Data flow: (no data consumed) -> always True
    Coupling: None — no dependencies on any other component.
    """

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        request_context: dict | None = None,
    ) -> bool:
        return True

    def on_episode_start(self) -> None:
        """Clear internal history buffer. Called by Orchestrator at episode start.

        Current: no-op. AlwaysSearchGate has no history state.
        Future: trajectory-aware gate can clear cached_data history here.
        """

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action. Pure local buffer op.

        Current: no-op. AlwaysSearchGate does not use action data.
        Future: trajectory-aware gate can buffer action history for drift detection.
        """


class AlwaysSkipGate:
    """Always skip search: orchestrator treats this as a gate-miss path.

    Net effect: the orchestrator still records ``query_keys`` to the strategy
    history (so trajectory buffers stay gap-free) and returns
    ``HitType.MISS`` for every checkpoint query, forcing the interceptor to
    fall through the full inference path. ``broadcast_action`` then feeds
    the real inference output back into all components. The cache framework
    is effectively transparent at this checkpoint while keeping trajectory
    history semantics intact.

    Use case: Step 2 of the trajectory-deviation experiment (background L2
    sampling) where we need M independent full-inference rollouts over a
    GT observation sequence with trajectory history preserved — see
    ``logs/trajectory_deviation_corrective_experiment.log.md`` §13.4 and
    ``logs/trajectory_deviation_corrective_implementation.log.md`` §8.1.
    """

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        request_context: dict | None = None,
    ) -> bool:
        return False

    def on_episode_start(self) -> None:
        """No-op. Signature matches GateFunction protocol."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """No-op. Signature matches GateFunction protocol."""


class RandomGate:
    """Server-side random-skip gate with per-connection RNG determinism.

    Each gate call samples an independent Bernoulli draw: with probability
    ``p_inference`` the gate returns False (skip cache -> fall-through to
    real inference); otherwise returns True (search cache).

    Reproducibility scope (intentional, per plan §5.3):
      - Constructed with ``seed``; at every ``on_episode_start()`` an
        internal ``ep_idx`` counter is incremented and the RNG is
        re-seeded with ``seed * 10_000 + ep_idx``.
      - Cache connections each get their own gate instance via
        ``build_per_connection_components``, so the stream is deterministic
        **per-connection** (same connection, same N-th episode => same
        stream). It is NOT deterministic across worker reassignment or
        resume — only the aggregate stochasticity across the 500-ep
        sweep is the metric of interest.

    Coupling:
      - UNAFFECTED BY: request_context, cached_data.
      - CONSUMED BY: CacheOrchestrator.check() — same skip semantics as
        AlwaysSkipGate.
    """

    def __init__(self, p_inference: float, seed: int) -> None:
        # bool is a subclass of int; reject it explicitly so ``seed=True``
        # does not silently degrade to seed=1.
        if isinstance(p_inference, bool) or not isinstance(p_inference, (int, float)):
            raise TypeError(
                f"RandomGate p_inference must be a real number, "
                f"got {type(p_inference).__name__}"
            )
        if not (0.0 <= p_inference <= 1.0):
            raise ValueError(
                f"RandomGate p_inference must be in [0, 1], got {p_inference}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(
                f"RandomGate seed must be a non-negative int, "
                f"got {type(seed).__name__}"
            )
        if seed < 0:
            raise ValueError(f"RandomGate seed must be >= 0, got {seed}")
        self._p_inference = float(p_inference)
        self._seed = int(seed)
        self._ep_idx = 0
        self._rng = np.random.default_rng(self._seed)

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        request_context: dict | None = None,
    ) -> bool:
        # Skip (run inference) with probability p_inference; otherwise search.
        return bool(self._rng.random() >= self._p_inference)

    def on_episode_start(self) -> None:
        """Advance ep_idx and re-seed RNG for per-connection determinism."""
        self._ep_idx += 1
        self._rng = np.random.default_rng(self._seed * 10_000 + self._ep_idx)

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """No-op. Signature matches GateFunction protocol."""


class PeriodicGate:
    """Server-side periodic-skip gate.

    Each episode begins with ``cache_len`` cache searches, followed by
    ``inference_len`` forced skips; the cycle repeats until the episode
    ends. ``on_episode_start`` resets the counter so every episode
    starts with a cache block.

    Cost metric derivation:
      - The decision at cycle ``c`` is exactly
        ``c % (cache_len + inference_len) < cache_len``. Given
        ``total_cycles`` (known to the runner), the runner derives
        ``num_inference_cycles`` via the same closed-form formula;
        no server-side stat transport is required.

    Coupling:
      - UNAFFECTED BY: request_context, cached_data.
      - CONSUMED BY: CacheOrchestrator.check() — same skip semantics as
        AlwaysSkipGate.
    """

    def __init__(self, cache_len: int, inference_len: int) -> None:
        if isinstance(cache_len, bool) or not isinstance(cache_len, int):
            raise TypeError(
                f"PeriodicGate cache_len must be an int >= 1, "
                f"got {type(cache_len).__name__}"
            )
        if isinstance(inference_len, bool) or not isinstance(inference_len, int):
            raise TypeError(
                f"PeriodicGate inference_len must be an int >= 1, "
                f"got {type(inference_len).__name__}"
            )
        if cache_len < 1 or inference_len < 1:
            raise ValueError(
                "PeriodicGate requires cache_len >= 1 and inference_len >= 1, "
                f"got cache_len={cache_len}, inference_len={inference_len}"
            )
        self._cache_len = int(cache_len)
        self._inference_len = int(inference_len)
        self._period = self._cache_len + self._inference_len
        self._counter = 0

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        request_context: dict | None = None,
    ) -> bool:
        pos = self._counter % self._period
        self._counter += 1
        return pos < self._cache_len   # True first k positions, False next n

    def on_episode_start(self) -> None:
        """Reset the cycle counter so the next episode starts with cache."""
        self._counter = 0

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """No-op. Signature matches GateFunction protocol."""


class ClientControlledGate:
    """Gate whose skip/search decision is driven by a per-request client signal.

    The client injects ``{"__gate_decision__": "skip" | "search"}`` into each
    obs. ``InferenceInterceptor.infer()`` pops that field before
    ``_input_transform`` and forwards it as
    ``request_context={"gate_decision": <value>}`` through
    ``CacheOrchestrator.check()`` to this gate.

    Coupling:
      - REQUIRES: request_context with key "gate_decision".
      - FAILS LOUD: raises ValueError on missing / unknown value.
      - UNAFFECTED BY: cached_data.
      - Skip path: orchestrator treats the return identically to
        AlwaysSkipGate (record_query_keys + broadcast_action preserved).
    """

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        request_context: dict | None = None,
    ) -> bool:
        if request_context is None or "gate_decision" not in request_context:
            raise ValueError(
                "ClientControlledGate requires request_context['gate_decision']. "
                "Ensure obs carries '__gate_decision__' and that "
                "InferenceInterceptor.infer() forwards it to orchestrator.check(). "
                "Verify the cache YAML sets gate.type='client_controlled'."
            )
        decision = request_context["gate_decision"]
        if decision == "skip":
            return False
        if decision == "search":
            return True
        raise ValueError(
            f"ClientControlledGate: unknown gate_decision={decision!r}. "
            "Expected 'skip' or 'search'."
        )

    def on_episode_start(self) -> None:
        """No-op. Signature matches GateFunction protocol."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """No-op. Signature matches GateFunction protocol."""


class ScoreHysteresisGate:
    """Server-side N1 score-hysteresis gate, plus the optional N4 V2 injection.

    Serverizes exp-layer ``N1GateState`` (roadmap N1); with ``L`` set it also
    serverizes the Stage-3a ``N4GateState`` (roadmap N4, winning point L=6). The
    V2 branch caps a continuous cache-execution (FULL_HIT) run at ``L``: once
    ``_fh_run >= L`` the gate forces one skip (a fresh inference) that breaks the
    run. ``L=None`` disables V2 -> behavior is identical to pure N1 (the latency
    deployment profile is ``L`` omitted; the SR profile is ``L: 6``).

    Mechanism (roadmap N1): remember the last searched step's ``cp1_score``.
    After ``j`` consecutive searched steps score below ``theta_low`` the gate
    stops searching (that step runs full inference); during the skip stretch it
    probes (forces a search) every ``probe_interval`` steps; a probe scoring
    ``>= theta_high`` recovers to normal searching (dual-threshold hysteresis
    guards against chatter). ``probe_interval=None`` means never probe (a skip
    stretch, once entered, is permanent for the episode).

    Decide/observe are split across two orchestrator calls that together
    reproduce the exp-layer ``N1GateState`` step-for-step:
      - ``__call__`` is PURE (no state mutation) -> the decision only.
      - ``record_verdict`` applies the score-driven transition, branching on the
        searching/skipping state left unchanged by ``__call__``.
    The orchestrator feeds ``record_verdict`` on EVERY per-checkpoint check()
    path: searched steps carry the real ``cp1_score``; skip steps carry
    ``cp1_score=None`` with ``searched=False``. So the decide/observe pairing
    matches the client loop and the same golden traces validate both.

    Coupling:
      - CONSUMES: its own verdict feedback via ``record_verdict``. No dependency
        on request_context / cached_data / CacheStorage.
      - Per-connection instance (build_per_connection_components); single-thread
        per connection -> state mutation is lock-free.
    """

    def __init__(
        self,
        theta_low: float,
        theta_high: float,
        j: int,
        probe_interval: int | None,
        L: int | None = None,
        include_ws: bool = False,
    ) -> None:
        # bool is an int subclass; reject it explicitly on every numeric param so
        # a YAML/CLI ``true``/``false`` misconfig fails loud instead of silently
        # degrading to 1/0 (mirrors RandomGate / PeriodicGate).
        for name, value in (("theta_low", theta_low), ("theta_high", theta_high)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"ScoreHysteresisGate {name} must be a real number, "
                    f"got {type(value).__name__}"
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"ScoreHysteresisGate {name} must be finite, got {value}"
                )
        if float(theta_high) < float(theta_low):
            raise ValueError(
                "ScoreHysteresisGate requires theta_high >= theta_low, "
                f"got theta_low={theta_low}, theta_high={theta_high}"
            )
        if isinstance(j, bool) or not isinstance(j, int):
            raise TypeError(
                f"ScoreHysteresisGate j must be an int >= 1, got {type(j).__name__}"
            )
        if j < 1:
            raise ValueError(f"ScoreHysteresisGate j must be >= 1, got {j}")
        if probe_interval is not None:
            if isinstance(probe_interval, bool) or not isinstance(probe_interval, int):
                raise TypeError(
                    "ScoreHysteresisGate probe_interval must be None or an int "
                    f">= 1, got {type(probe_interval).__name__}"
                )
            if probe_interval < 1:
                raise ValueError(
                    "ScoreHysteresisGate probe_interval must be >= 1 (or None), "
                    f"got {probe_interval}"
                )
        # V2 injection threshold (Stage 3b): None = pure N1 (V2 disabled, exact
        # backward-compatible behavior). An int L caps a continuous cache-execution
        # run at L before forcing one skip (inject a fresh inference).
        if L is not None:
            if isinstance(L, bool) or not isinstance(L, int):
                raise TypeError(
                    f"ScoreHysteresisGate L must be None or an int >= 1, "
                    f"got {type(L).__name__}"
                )
            if L < 1:
                raise ValueError(f"ScoreHysteresisGate L must be >= 1 (or None), got {L}")
        if not isinstance(include_ws, bool):
            raise TypeError(
                f"ScoreHysteresisGate include_ws must be a bool, got {type(include_ws).__name__}"
            )
        self._theta_low = float(theta_low)
        self._theta_high = float(theta_high)
        self._j = int(j)
        self._probe_interval = None if probe_interval is None else int(probe_interval)
        self._L = None if L is None else int(L)
        self._include_ws = bool(include_ws)
        # State machine registers (reset by on_episode_start).
        self._searching = True
        self._low_run = 0
        self._since_probe = 0
        # V2: length of the current continuous cache-execution (FULL_HIT) run.
        self._fh_run = 0
        self._task_key = ""

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        request_context: dict | None = None,
    ) -> bool:
        # PURE decide (no mutation): searching -> search; else probe when the
        # skip interval is up; else skip. Mutation happens only in record_verdict.
        if self._searching:
            # V2 injection (Stage 3b): after L continuous cache-execution steps,
            # force one skip (fresh inference) to break the run. Reads _fh_run/_L
            # only. A searching-state skip is the UNIQUE V2 source, which is how
            # record_verdict reconstructs V1 vs V2 without a _last_v2 flag.
            if self._L is not None and self._fh_run >= self._L:
                return False  # V2 inject
            return True
        if (
            self._probe_interval is not None
            and self._since_probe + 1 >= self._probe_interval
        ):
            return True  # probe
        return False

    def record_verdict(
        self,
        checkpoint_id: CheckpointID,
        *,
        hit_type,
        cp1_score: float | None,
        winner_id: str | None,
        start_t,
        searched: bool,
    ) -> None:
        """Observe this step's verdict; advance the state machine.

        Dispatches on ``searched`` first so the V2 injection (a skip taken while
        the N1 machine is still in the searching state) does not perturb the N1
        hysteresis. ``hit_type`` is a server-side ``HitType`` enum and drives the
        V2 cache-execution run counter (``_fh_run``); ``winner_id`` / ``start_t``
        remain forward-compat only. When ``L is None`` (pure N1) a searching-state
        skip never occurs, so this reduces to the original N1 transitions.
        """
        if not searched:
            # Skip step = fresh inference -> break the cache-execution run.
            self._fh_run = 0
            if not self._searching:
                # V1 skip (skipping state): advance the probe counter (N1).
                self._since_probe += 1
            # else: V2 injection (searching state, Stage 3b) -> freeze N1; the
            # gate did not decide this skip via the hysteresis, so leave
            # _searching / _low_run / _since_probe untouched.
            return

        # Searched step: a real cache search happened. Advance the V2 run counter
        # (pure FULL_HIT replay by default; WARM_START only if include_ws), then
        # the N1 hysteresis exactly as before.
        is_cache_exec = hit_type == HitType.FULL_HIT or (
            self._include_ws and hit_type == HitType.WARM_START
        )
        self._fh_run = self._fh_run + 1 if is_cache_exec else 0

        if cp1_score is None:
            # searched step with an empty result set (legit MISS): treat as the
            # lowest possible score so it counts toward low_run and never
            # triggers recovery. Matches N1GateState's None-as-(-inf) rule.
            score = -math.inf
        elif not math.isfinite(cp1_score):
            # cp1_score is normalized cosine in [-1, 1]; a non-finite value is a
            # contract violation that should never occur server-side. Fail open
            # to full searching (and reset the V2 run) rather than risk a wrongful
            # skip or injection.
            logger.warning(
                "ScoreHysteresisGate got non-finite cp1_score=%r (task=%r); "
                "failing open to full search.",
                cp1_score, self._task_key,
            )
            self._searching = True
            self._low_run = 0
            self._since_probe = 0
            self._fh_run = 0
            return
        else:
            score = float(cp1_score)

        if self._searching:
            if score < self._theta_low:
                self._low_run += 1
                if self._low_run >= self._j:
                    self._searching = False
                    self._since_probe = 0
            else:
                self._low_run = 0
        else:  # skipping: this searched step was a probe
            self._since_probe = 0
            if score >= self._theta_high:
                self._searching = True
                self._low_run = 0

    def on_episode_start(self, task_key: str = "") -> None:
        """Reset the state machine at each episode; stash task_key.

        ``task_key`` is broadcast by the orchestrator (G0a) and kept for
        diagnostics / future per-task parametrization; the N1 decision does not
        depend on it.
        """
        self._searching = True
        self._low_run = 0
        self._since_probe = 0
        self._fh_run = 0
        self._task_key = task_key
        logger.debug("ScoreHysteresisGate reset for task=%r", task_key)

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """No-op. N1 does not use action history. Signature matches protocol."""
