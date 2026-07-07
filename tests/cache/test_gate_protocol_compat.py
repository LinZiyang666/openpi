"""Runtime-Protocol compatibility guard for the gate hooks (Stage 4a Blocking②).

``GateFunction`` is a ``@runtime_checkable`` Protocol. Its optional lifecycle
hooks (``on_episode_start`` / ``record_action`` / ``record_verdict`` and now
``replay_target``) are documented in the docstring ONLY and are discovered by
the orchestrator via ``hasattr`` -- they are deliberately NOT declared in the
Protocol body. A ``@runtime_checkable`` Protocol cannot express optional
members: declaring one would make ``isinstance(gate, GateFunction)`` require it,
breaking every legacy gate that implements only ``__call__``.

This test locks that invariant so a future refactor cannot silently promote
``replay_target`` (or any hook) into a required Protocol member.
"""

from __future__ import annotations

from openpi.cache.components.gate import (
    AlwaysSearchGate,
    AlwaysSkipGate,
    ClientControlledGate,
    FollowWinnerGate,
    GateFunction,
    PeriodicGate,
    RandomGate,
    ScoreHysteresisGate,
)


_ALL_GATES = [
    AlwaysSearchGate(),
    AlwaysSkipGate(),
    ClientControlledGate(),
    RandomGate(p_inference=0.5, seed=0),
    PeriodicGate(cache_len=1, inference_len=1),
    ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=1, probe_interval=1),
    FollowWinnerGate(lock_streak=2, budget=3),
]


def test_all_gates_conform_to_runtime_protocol():
    for gate in _ALL_GATES:
        assert isinstance(gate, GateFunction), f"{type(gate).__name__} lost GateFunction"


def test_minimal_call_only_object_still_conforms():
    """A gate implementing ONLY ``__call__`` must remain a valid GateFunction.

    This fails if any optional hook (e.g. ``replay_target``) is wrongly added to
    the Protocol body, which would make ``@runtime_checkable`` require it.
    """

    class _CallOnlyGate:
        def __call__(self, checkpoint_id, cached_data, request_context=None):
            return True

    assert isinstance(_CallOnlyGate(), GateFunction)


def test_replay_target_is_only_on_follow_winner():
    """Only FollowWinnerGate exposes ``replay_target``; legacy gates must not.

    The orchestrator gates the blind-replay branch on ``hasattr(gate,
    'replay_target')``, so a legacy gate must NOT accidentally grow the method.
    """
    for gate in _ALL_GATES:
        has = hasattr(gate, "replay_target")
        assert has is isinstance(gate, FollowWinnerGate), (
            f"{type(gate).__name__}: replay_target presence must match FollowWinnerGate"
        )
    # And it returns None until the gate has locked.
    assert FollowWinnerGate(lock_streak=2, budget=3).replay_target() is None
