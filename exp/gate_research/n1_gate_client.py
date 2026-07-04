"""N1 ScoreHysteresisGate live client (Stage 1b).

This module carries the entire N1 gate logic on the CLIENT side so the live
LIBERO rollout can drive a server-side ``ClientControlledGate`` without any src
or shared-``examples/`` change. It provides:

- ``N1GateState`` — a pure, importable state machine (no argv, no I/O at import)
  identical in step semantics to the Stage-1a offline ``n1_sim``. It is the
  single source of the decide/observe logic, shared by the live wrapper and the
  unit tests.
- ``N1GateClient`` — a thin wrapper around a websocket client policy. Per step it
  reads the gate decision from the state machine, injects
  ``obs["__gate_decision__"]``, reads back ``result["__hit_meta__"]["cp1_score"]``
  to advance the machine, and stamps ``result["__collect_meta__"]={"searched": ...}``
  so the existing conductor recorder persists an authoritative ``searched`` field.
- ``make_n1_client_factory`` / ``n1_params_from_env`` — glue for the N1 worker
  entry.

Authoritative-``searched`` invariant: skip is determined ONLY by the client
decision (recorded via ``__collect_meta__``); ``cp1_score is None`` does NOT imply
skip (an empty searched-MISS also yields ``cp1_score=None``). Exception contract:
construction validates params (fail-fast); a searched step with a None score is a
legitimate empty-search MISS (treated as ``-inf``); a searched step with a
non-finite score or a missing ``__hit_meta__`` is an anomaly that fails OPEN (the
machine is forced back to full searching, never mis-skips, never raises inside
``infer``); ``inner.infer`` exceptions propagate unchanged.

"step" throughout = one CP1 decision step (one action chunk ~= 10 env steps).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

# Reserved obs / result keys shared with the server contract.
GATE_DECISION_KEY = "__gate_decision__"
HIT_META_KEY = "__hit_meta__"
COLLECT_META_KEY = "__collect_meta__"
SEARCH = "search"
SKIP = "skip"


# ----------------------------------------------------------------------
# Pure state machine (importable, no side effects)
# ----------------------------------------------------------------------
class N1GateState:
    """Dual-threshold hysteresis gate state machine (roadmap N1).

    Mealy machine: ``decide()`` returns the action for the upcoming step from
    the current state; ``observe(decision, score)`` advances the state after the
    server's verdict is seen. Step-for-step identical to the Stage-1a offline
    ``n1_sim``: enter SKIPPING after ``j`` consecutive searched steps below
    ``theta_low``; while skipping, probe a real search every ``M`` steps and
    resume once a probe scores ``>= theta_high``.
    """

    def __init__(self, theta_low: float, theta_high: float, j: int, M: int | None) -> None:
        # Fail-fast validation: a bad config must crash at worker startup, before
        # any rollout, rather than silently mis-gate and pollute success rate.
        if not (isinstance(theta_low, (int, float)) and not isinstance(theta_low, bool)
                and math.isfinite(theta_low)):
            raise ValueError(f"theta_low must be a finite number, got {theta_low!r}")
        if not (isinstance(theta_high, (int, float)) and not isinstance(theta_high, bool)
                and math.isfinite(theta_high)):
            raise ValueError(f"theta_high must be a finite number, got {theta_high!r}")
        if theta_high < theta_low:
            raise ValueError(f"theta_high ({theta_high}) must be >= theta_low ({theta_low})")
        if isinstance(j, bool) or not isinstance(j, int) or j < 1:
            raise ValueError(f"j must be an int >= 1, got {j!r}")
        if M is not None and (isinstance(M, bool) or not isinstance(M, int) or M < 1):
            raise ValueError(f"M must be None or an int >= 1, got {M!r}")
        self.theta_low = float(theta_low)
        self.theta_high = float(theta_high)
        self.j = int(j)
        self.M = None if M is None else int(M)
        self.reset()

    def reset(self) -> None:
        """Reset per-episode state; the first ``decide()`` always searches."""
        self.searching = True
        self.low_run = 0
        self.since_probe = 0

    def decide(self) -> str:
        """Return ``"search"`` or ``"skip"`` for the upcoming step."""
        if self.searching:
            return SEARCH
        # Skipping: this step is a probe (a real search) iff the probe interval
        # is due; +1 mirrors the offline loop's pre-increment of the counter.
        if self.M is not None and self.since_probe + 1 >= self.M:
            return SEARCH
        return SKIP

    def observe(self, decision: str, score: float | None) -> None:
        """Advance the state after seeing the server verdict for this step.

        A None ``score`` on a searched step is a legitimate empty-search MISS and
        is treated as ``-inf`` (below any threshold). Non-finite scores are
        handled upstream (fail-open) and never reach here.
        """
        if decision == SEARCH:
            s = float("-inf") if score is None else float(score)
            if self.searching:
                if s < self.theta_low:
                    self.low_run += 1
                    if self.low_run >= self.j:
                        self.searching = False
                        self.since_probe = 0
                else:
                    self.low_run = 0
            else:
                # Probe step: a real search happened while stopped.
                self.since_probe = 0
                if s >= self.theta_high:
                    self.searching = True
                    self.low_run = 0
        else:  # SKIP
            self.since_probe += 1

    def force_search(self) -> None:
        """Fail-open reset used when a searched step returns anomalous data:
        resume full searching so the gate degrades to always-search, never
        mis-skips."""
        self.searching = True
        self.low_run = 0
        self.since_probe = 0


# ----------------------------------------------------------------------
# Live client wrapper
# ----------------------------------------------------------------------
class N1GateClient:
    """Wrap an inner websocket client policy and drive the N1 gate per step.

    Delegates every method other than ``infer`` / ``episode_start`` to the inner
    client, so it is a drop-in for ``LiberoEpisodeRunner``'s client contract
    (``select_bundle`` / ``episode_end`` / ``close`` / ``get_server_metadata``).
    """

    def __init__(self, inner: Any, state: N1GateState) -> None:
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
            score, anomaly = self._read_score(result)
            if anomaly:
                # Non-finite score or missing hit-meta on a searched step: the
                # server contract was violated. Fail OPEN (resume searching)
                # instead of raising (a raise would be swallowed by
                # main._run_episode's broad except and fail the episode).
                logger.warning("N1GateClient: anomalous searched-step result, "
                               "failing open to full search")
                self._state.force_search()
            else:
                # score is a finite float or None (legitimate empty-search MISS).
                self._state.observe(SEARCH, score)
        else:
            self._state.observe(SKIP, None)

        # Record seam: the existing conductor recorder reads __collect_meta__
        # and persists row["searched"]. This is the ONLY authoritative skip
        # signal for the analyzer (cp1_score None is NOT sufficient).
        if isinstance(result, dict):
            result[COLLECT_META_KEY] = {"searched": decision == SEARCH}
        return result

    def episode_start(self, *args, **kwargs):
        """Reset the gate state at each episode boundary, then delegate."""
        self._state.reset()
        return self._inner.episode_start(*args, **kwargs)

    @staticmethod
    def _read_score(result: Any) -> tuple[float | None, bool]:
        """Extract the searched-step score. Returns ``(score, anomaly)`` where
        ``score`` is a finite float or None (legit empty-search MISS), and
        ``anomaly`` flags a contract violation (missing hit-meta / non-finite
        score) that must fail open."""
        if not isinstance(result, dict):
            return None, True
        meta = result.get(HIT_META_KEY)
        if not isinstance(meta, dict) or "cp1_score" not in meta:
            return None, True
        score = meta["cp1_score"]
        if score is None:
            return None, False  # empty-search MISS: legitimate
        try:
            f = float(score)
        except (TypeError, ValueError):
            return None, True
        if not math.isfinite(f):
            return None, True
        return f, False

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on N1GateClient itself. Guard
        # against recursion before _inner is assigned.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


# ----------------------------------------------------------------------
# Factory + env parameter loading (used by the N1 worker entry)
# ----------------------------------------------------------------------
def n1_params_from_env(env: dict | None = None) -> dict:
    """Read N1 parameters from environment variables, failing loudly on
    missing / malformed values. Returns ``{theta_low, theta_high, j, M}``.
    ``N1_M`` may be the literal ``"none"`` (never probe)."""
    env = os.environ if env is None else env
    missing = [k for k in ("N1_THETA_LOW", "N1_THETA_HIGH", "N1_J", "N1_M") if k not in env]
    if missing:
        raise ValueError(f"missing N1 env vars: {missing}")
    try:
        theta_low = float(env["N1_THETA_LOW"])
        theta_high = float(env["N1_THETA_HIGH"])
        j = int(env["N1_J"])
    except ValueError as exc:
        raise ValueError(f"malformed N1 env value: {exc}") from exc
    m_raw = env["N1_M"].strip()
    M = None if m_raw.lower() == "none" else int(m_raw)
    # Delegate range validation to N1GateState (single source of truth).
    N1GateState(theta_low, theta_high, j, M)
    return {"theta_low": theta_low, "theta_high": theta_high, "j": j, "M": M}


def make_n1_client_factory(params: dict, inner_factory=None):
    """Build a client factory ``(server) -> N1GateClient`` for the conductor.

    ``inner_factory`` defaults to ``examples.libero.episode_runner.default_client_factory``
    (imported lazily so this module stays importable without the examples deps).
    A fresh ``N1GateState`` is created per factory call (one per worker client).
    """
    def factory(server):
        nonlocal inner_factory
        if inner_factory is None:
            from examples.libero.episode_runner import default_client_factory
            inner_factory = default_client_factory
        inner = inner_factory(server)
        state = N1GateState(params["theta_low"], params["theta_high"], params["j"], params["M"])
        return N1GateClient(inner, state)

    return factory
