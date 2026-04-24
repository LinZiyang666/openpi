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

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from openpi.cache.types import CheckpointID


@runtime_checkable
class GateFunction(Protocol):
    """Decide whether to perform a cache lookup.

    Data flow: KeyBuilder.cached_data (GPU) -> gate() -> bool
    Coupling:
      - MAY DEPEND ON: KeyBuilder.cached_data (read-only, for state-change gates)
      - CONSUMED BY: CacheOrchestrator.check() — if False, skip search entirely
      - DOES NOT interact with: CacheStorage or any Step 3 component
      - IF CHANGED: Only affects Orchestrator's search frequency, no downstream impact
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
