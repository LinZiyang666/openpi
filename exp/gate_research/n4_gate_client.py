"""N4 hybrid-gate live client (Stage 3a).

N4 overlays a V2 injection branch on top of the Stage-1b N1 score-hysteresis
gate, entirely on the CLIENT side, driving the same server-side
``ClientControlledGate`` with **zero src change**. Per CP1 decision step it emits
``search`` unless either branch fires:

- **V1 (latency, reused N1)**: the N1 dual-threshold hysteresis predicts a MISS ->
  ``skip``. Skipping a predicted-MISS step is inference-neutral (a MISS would run
  full inference anyway); it only saves the search+judge+fetch overhead.
- **V2 (SR gain, new)**: after ``L`` consecutive cache-execution steps (FULL_HIT
  replays) the gate forces a ``skip``, which turns that step into a fresh
  inference -- a low-dose injection that breaks the long cache-replay run (roadmap
  F12 / Stage 2a H1: long pure-replay runs suppress SR; a small injection lets the
  trajectory return to its own manifold).

This module provides, mirroring ``n1_gate_client``:

- ``N4GateState`` -- a pure state machine that EMBEDS an unmodified ``N1GateState``
  as its V1 sub-machine (single source of the hysteresis logic) and layers the V2
  cache-execution run counter on top. The V2 injection is transparent to the N1
  sub-machine (see ``_last_v2``) so the V1 golden traces are unchanged.
- ``N4GateClient`` -- a thin websocket-client wrapper. Per step it reads BOTH
  ``result["__hit_meta__"]["cp1_score"]`` (advances the hysteresis) AND
  ``result["__hit_meta__"]["hit_type"]`` (advances the V2 run counter). The
  ``hit_type`` field is guaranteed present on the wire by
  ``InferenceInterceptor._build_hit_meta`` (FULL_HIT / WARM_START / MISS), which is
  what makes the zero-src V2 count possible.
- ``make_n4_client_factory`` / ``n4_params_from_env`` -- glue for the N4 worker
  entry.

Authoritative-``searched`` and fail-open contracts are inherited verbatim from
``N1GateClient``: skip is determined ONLY by the client decision (recorded via
``__collect_meta__``); a searched step with a non-finite score or missing
``__hit_meta__`` fails OPEN (resume full searching, reset the V2 run) instead of
raising; ``inner.infer`` exceptions propagate unchanged.

"step" throughout = one CP1 decision step (one action chunk ~= 10 env steps).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from exp.gate_research.n1_gate_client import (
    COLLECT_META_KEY,
    GATE_DECISION_KEY,
    HIT_META_KEY,
    SEARCH,
    SKIP,
    N1GateClient,
    N1GateState,
)

logger = logging.getLogger(__name__)

# hit_type name (see openpi.cache.components.judge.HitType) that counts as a
# cache-execution step (pure cached-action replay). WARM_START is a partial
# warm-start inference (start_t > 0) and by default breaks the pure-replay run.
FULL_HIT = "FULL_HIT"
WARM_START = "WARM_START"


# ----------------------------------------------------------------------
# Pure state machine (importable, no side effects)
# ----------------------------------------------------------------------
class N4GateState:
    """N4 hybrid gate: N1 V1 hysteresis (embedded) + V2 cache-execution cap.

    ``decide()`` returns the action for the upcoming step; ``observe(decision,
    hit_type, score)`` advances the state after the server verdict is seen. The
    embedded ``N1GateState`` is the single source of the V1 decide/observe logic
    and is never modified; ``N4GateState`` only adds the V2 run counter and a
    ``_last_v2`` flag that keeps a V2-injected skip from perturbing the N1 phase.
    """

    def __init__(
        self,
        theta_low: float,
        theta_high: float,
        j: int,
        M: int | None,
        L: int,
        include_ws: bool = False,
    ) -> None:
        # The embedded N1 sub-machine validates theta_low/theta_high/j/M (single
        # source of truth); a bad V1 param crashes here at worker startup.
        self._n1 = N1GateState(theta_low, theta_high, j, M)
        # bool is an int subclass; reject it explicitly so a YAML/CLI true/false
        # never silently degrades L to 1/0 (mirrors N1GateState's j validation).
        if isinstance(L, bool) or not isinstance(L, int) or L < 1:
            raise ValueError(f"L must be an int >= 1, got {L!r}")
        if not isinstance(include_ws, bool):
            raise ValueError(f"include_ws must be a bool, got {include_ws!r}")
        self.L = int(L)
        self.include_ws = bool(include_ws)
        self.reset()

    def reset(self) -> None:
        """Reset per-episode state: reset the V1 sub-machine and clear the V2 run."""
        self._n1.reset()
        self.fh_run = 0
        self._last_v2 = False

    def decide(self) -> str:
        """Return ``"search"`` or ``"skip"`` for the upcoming step.

        V1 (N1 skip) takes precedence over V2; both yield ``skip`` so the output
        is identical either way, but the branch is recorded in ``_last_v2`` so
        ``observe`` can route the state update. When N1 decides skip, ``fh_run``
        is already 0 (a skip is not a cache execution), so the V2 test is moot.
        """
        base = self._n1.decide()
        if base == SKIP:
            self._last_v2 = False
            return SKIP
        if self.fh_run >= self.L:
            self._last_v2 = True
            return SKIP
        self._last_v2 = False
        return SEARCH

    def observe(self, decision: str, hit_type: str | None, score: float | None) -> None:
        """Advance the state after the server verdict for ``decision`` is seen.

        On a searched step, advance the N1 hysteresis (score) and the V2 run
        counter (hit_type). On a skip, the step ran a fresh inference so the
        cache-execution run resets; a V1 skip advances the N1 machine, while a V2
        injection leaves N1 frozen (it did not decide that skip -- feeding it a
        spurious skip would corrupt ``since_probe``).
        """
        if decision == SEARCH:
            self._n1.observe(SEARCH, score)
            is_cache_exec = hit_type == FULL_HIT or (self.include_ws and hit_type == WARM_START)
            self.fh_run = self.fh_run + 1 if is_cache_exec else 0
        else:  # SKIP
            # A skip runs fresh inference -> the cache-execution run is broken.
            self.fh_run = 0
            if not self._last_v2:
                # V1 skip: N1 decided it -> advance the sub-machine as N1 would.
                self._n1.observe(SKIP, None)
            # V2 injection (_last_v2): keep N1 frozen (transparent overlay).

    def force_search(self) -> None:
        """Fail-open reset (searched step returned anomalous data): resume full
        searching via the N1 sub-machine and clear the V2 run so the gate
        degrades to always-search, never mis-skips."""
        self._n1.force_search()
        self.fh_run = 0
        self._last_v2 = False


# ----------------------------------------------------------------------
# Live client wrapper
# ----------------------------------------------------------------------
class N4GateClient:
    """Wrap an inner websocket client policy and drive the N4 gate per step.

    Mirrors ``N1GateClient`` but reads ``hit_type`` in addition to ``cp1_score``
    so the V2 run counter can advance. Delegates every method other than
    ``infer`` / ``episode_start`` to the inner client (drop-in for
    ``LiberoEpisodeRunner``'s client contract).
    """

    def __init__(self, inner: Any, state: N4GateState) -> None:
        self._inner = inner
        self._state = state

    def infer(self, obs: dict) -> dict:
        """Inject the gate decision, run inference, advance the state machine,
        and stamp the authoritative ``searched`` provenance onto the result."""
        decision = self._state.decide()
        # Copy so the caller's obs dict is never mutated.
        out_obs = {**obs, GATE_DECISION_KEY: decision}
        result = self._inner.infer(out_obs)  # inner exceptions propagate

        if decision == SEARCH:
            # Reuse N1's score/anomaly contract verbatim (single source of truth).
            score, anomaly = N1GateClient._read_score(result)  # noqa: SLF001 - shared helper
            if anomaly:
                # Server contract violated on a searched step: fail OPEN (resume
                # searching, reset the V2 run) instead of raising, matching N1.
                logger.warning("N4GateClient: anomalous searched-step result, "
                               "failing open to full search")
                self._state.force_search()
            else:
                # score is a finite float or None (legit empty-search MISS);
                # hit_type drives the V2 run counter (missing -> non-cache-exec).
                self._state.observe(SEARCH, self._read_hit_type(result), score)
        else:
            self._state.observe(SKIP, None, None)

        # Record seam: the conductor recorder reads __collect_meta__ and persists
        # row["searched"] -- the ONLY authoritative skip signal for the analyzer.
        if isinstance(result, dict):
            result[COLLECT_META_KEY] = {"searched": decision == SEARCH}
        return result

    def episode_start(self, *args, **kwargs):
        """Reset the gate state at each episode boundary, then delegate."""
        self._state.reset()
        return self._inner.episode_start(*args, **kwargs)

    @staticmethod
    def _read_hit_type(result: Any) -> str | None:
        """Extract the searched-step ``hit_type`` (FULL_HIT / WARM_START / MISS).

        Returns None on a missing / malformed hit-meta so the V2 counter treats
        the step as non-cache-execution (conservative: never over-extends a run).
        """
        if not isinstance(result, dict):
            return None
        meta = result.get(HIT_META_KEY)
        if not isinstance(meta, dict):
            return None
        ht = meta.get("hit_type")
        return ht if isinstance(ht, str) else None

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on N4GateClient itself. Guard
        # against recursion before _inner is assigned.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


# ----------------------------------------------------------------------
# Factory + env parameter loading (used by the N4 worker entry)
# ----------------------------------------------------------------------
def n4_params_from_env(env: dict | None = None) -> dict:
    """Read N4 parameters from environment variables, failing loudly on
    missing / malformed values. Returns ``{theta_low, theta_high, j, M, L}``.
    ``N4_M`` may be the literal ``"none"`` (never probe). ``include_ws`` is NOT
    read from env: 3a live runs are fixed to ``include_ws=False`` (it is a
    unit-test-only parameter), so a live run never depends on an env toggle."""
    env = os.environ if env is None else env
    missing = [k for k in ("N4_THETA_LOW", "N4_THETA_HIGH", "N4_J", "N4_M", "N4_L") if k not in env]
    if missing:
        raise ValueError(f"missing N4 env vars: {missing}")
    try:
        theta_low = float(env["N4_THETA_LOW"])
        theta_high = float(env["N4_THETA_HIGH"])
        j = int(env["N4_J"])
        L = int(env["N4_L"])
    except ValueError as exc:
        raise ValueError(f"malformed N4 env value: {exc}") from exc
    m_raw = env["N4_M"].strip()
    M = None if m_raw.lower() == "none" else int(m_raw)
    # Delegate range validation to N4GateState (single source of truth).
    N4GateState(theta_low, theta_high, j, M, L)
    return {"theta_low": theta_low, "theta_high": theta_high, "j": j, "M": M, "L": L}


def make_n4_client_factory(params: dict, inner_factory=None):
    """Build a client factory ``(server) -> N4GateClient`` for the conductor.

    ``inner_factory`` defaults to ``examples.libero.episode_runner.default_client_factory``
    (imported lazily so this module stays importable without the examples deps).
    A fresh ``N4GateState`` is created per factory call (one per worker client).
    """
    def factory(server):
        nonlocal inner_factory
        if inner_factory is None:
            from examples.libero.episode_runner import default_client_factory
            inner_factory = default_client_factory
        inner = inner_factory(server)
        state = N4GateState(
            params["theta_low"], params["theta_high"], params["j"], params["M"], params["L"]
        )
        return N4GateClient(inner, state)

    return factory
