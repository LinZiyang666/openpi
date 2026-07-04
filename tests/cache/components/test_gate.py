"""Tests for gate implementations."""

import math

import pytest
import torch

from openpi.cache.components.gate import (
    AlwaysSearchGate,
    AlwaysSkipGate,
    ClientControlledGate,
    GateFunction,
    PeriodicGate,
    RandomGate,
    ScoreHysteresisGate,
)
from openpi.cache.types import CheckpointID


def test_always_search_gate_returns_true():
    gate = AlwaysSearchGate()
    assert gate(CheckpointID.CP1, {}) is True


def test_always_search_gate_cp3_returns_true():
    gate = AlwaysSearchGate()
    assert gate(CheckpointID.CP3, {"state": torch.randn(32)}) is True


def test_always_search_gate_conforms_to_protocol():
    assert isinstance(AlwaysSearchGate(), GateFunction)


def test_always_search_gate_ignores_request_context():
    # Default gates accept request_context for protocol conformance but
    # never consume it; presence must not alter the decision.
    gate = AlwaysSearchGate()
    assert gate(CheckpointID.CP1, {}, {"gate_decision": "skip"}) is True
    assert gate(CheckpointID.CP1, {}, None) is True


def test_always_skip_gate_returns_false():
    gate = AlwaysSkipGate()
    assert gate(CheckpointID.CP1, {}) is False


def test_always_skip_gate_cp3_returns_false():
    gate = AlwaysSkipGate()
    assert gate(CheckpointID.CP3, {"state": torch.randn(32)}) is False


def test_always_skip_gate_conforms_to_protocol():
    assert isinstance(AlwaysSkipGate(), GateFunction)


def test_always_skip_gate_lifecycle_hooks_are_noop():
    # Both hooks should be callable and return None without touching state.
    gate = AlwaysSkipGate()
    assert gate.on_episode_start() is None
    assert gate.record_action(torch.zeros(1)) is None


def test_always_skip_gate_ignores_request_context():
    gate = AlwaysSkipGate()
    assert gate(CheckpointID.CP1, {}, {"gate_decision": "search"}) is False
    assert gate(CheckpointID.CP1, {}, None) is False


# ---------------------------------------------------------------------------
# ClientControlledGate
# ---------------------------------------------------------------------------


def test_client_controlled_gate_skip_returns_false():
    gate = ClientControlledGate()
    assert gate(CheckpointID.CP1, {}, {"gate_decision": "skip"}) is False


def test_client_controlled_gate_search_returns_true():
    gate = ClientControlledGate()
    assert gate(CheckpointID.CP1, {}, {"gate_decision": "search"}) is True


def test_client_controlled_gate_ignores_cached_data():
    # Decision depends only on the request signal. Different cached_data
    # must not flip the outcome.
    gate = ClientControlledGate()
    assert gate(CheckpointID.CP3, {}, {"gate_decision": "skip"}) is False
    assert (
        gate(CheckpointID.CP3, {"state": torch.randn(32)}, {"gate_decision": "skip"})
        is False
    )


def test_client_controlled_gate_missing_request_context_raises():
    gate = ClientControlledGate()
    with pytest.raises(ValueError, match="requires request_context"):
        gate(CheckpointID.CP1, {}, None)


def test_client_controlled_gate_missing_gate_decision_raises():
    gate = ClientControlledGate()
    with pytest.raises(ValueError, match="requires request_context"):
        gate(CheckpointID.CP1, {}, {"something_else": "skip"})


def test_client_controlled_gate_unknown_value_raises():
    gate = ClientControlledGate()
    with pytest.raises(ValueError, match="unknown gate_decision"):
        gate(CheckpointID.CP1, {}, {"gate_decision": "maybe"})


def test_client_controlled_gate_conforms_to_protocol():
    assert isinstance(ClientControlledGate(), GateFunction)


def test_client_controlled_gate_lifecycle_hooks_are_noop():
    gate = ClientControlledGate()
    assert gate.on_episode_start() is None
    assert gate.record_action(torch.zeros(1)) is None


# ---------------------------------------------------------------------------
# RandomGate
# ---------------------------------------------------------------------------


def test_random_gate_rejects_invalid_p_inference():
    with pytest.raises(ValueError, match="p_inference"):
        RandomGate(p_inference=-0.1, seed=0)
    with pytest.raises(ValueError, match="p_inference"):
        RandomGate(p_inference=1.5, seed=0)


def test_random_gate_rejects_negative_seed():
    with pytest.raises(ValueError, match="seed"):
        RandomGate(p_inference=0.3, seed=-1)


def test_random_gate_p_zero_always_searches():
    gate = RandomGate(p_inference=0.0, seed=0)
    for _ in range(200):
        assert gate(CheckpointID.CP1, {}) is True


def test_random_gate_p_one_always_skips():
    gate = RandomGate(p_inference=1.0, seed=0)
    for _ in range(200):
        assert gate(CheckpointID.CP1, {}) is False


def test_random_gate_per_connection_determinism():
    # Fixed seed + identical on_episode_start count => byte-identical stream.
    def draw(seed: int, ep_count: int, n: int) -> list[bool]:
        g = RandomGate(p_inference=0.5, seed=seed)
        for _ in range(ep_count):
            g.on_episode_start()
        return [g(CheckpointID.CP1, {}) for _ in range(n)]

    assert draw(seed=7, ep_count=3, n=32) == draw(seed=7, ep_count=3, n=32)


def test_random_gate_consecutive_episodes_have_distinct_streams():
    g = RandomGate(p_inference=0.5, seed=7)
    g.on_episode_start()
    ep1 = [g(CheckpointID.CP1, {}) for _ in range(64)]
    g.on_episode_start()
    ep2 = [g(CheckpointID.CP1, {}) for _ in range(64)]
    assert ep1 != ep2


def test_random_gate_ignores_request_context_and_cached_data():
    # Decision depends only on internal RNG; extra inputs must not alter
    # the sequence.
    g1 = RandomGate(p_inference=0.5, seed=0)
    g1.on_episode_start()
    seq1 = [g1(CheckpointID.CP1, {}, None) for _ in range(32)]

    g2 = RandomGate(p_inference=0.5, seed=0)
    g2.on_episode_start()
    seq2 = [
        g2(CheckpointID.CP1, {"state": torch.randn(32)}, {"gate_decision": "skip"})
        for _ in range(32)
    ]
    assert seq1 == seq2


def test_random_gate_conforms_to_protocol():
    assert isinstance(RandomGate(p_inference=0.3, seed=0), GateFunction)


def test_random_gate_record_action_is_noop():
    g = RandomGate(p_inference=0.3, seed=0)
    assert g.record_action(torch.zeros(1)) is None


# ---------------------------------------------------------------------------
# PeriodicGate
# ---------------------------------------------------------------------------


def test_periodic_gate_rejects_invalid_lengths():
    with pytest.raises(ValueError, match="cache_len"):
        PeriodicGate(cache_len=0, inference_len=1)
    with pytest.raises(ValueError, match="inference_len"):
        PeriodicGate(cache_len=1, inference_len=0)


def test_periodic_gate_duty_cycle():
    gate = PeriodicGate(cache_len=3, inference_len=2)
    # One full period: TTTFF.
    expected = [True, True, True, False, False]
    got = [gate(CheckpointID.CP1, {}) for _ in range(5)]
    assert got == expected
    # Second period replays identical pattern.
    got2 = [gate(CheckpointID.CP1, {}) for _ in range(5)]
    assert got2 == expected


def test_periodic_gate_k1_n1_alternates():
    gate = PeriodicGate(cache_len=1, inference_len=1)
    seq = [gate(CheckpointID.CP1, {}) for _ in range(6)]
    assert seq == [True, False, True, False, True, False]


def test_periodic_gate_on_episode_start_resets():
    gate = PeriodicGate(cache_len=2, inference_len=3)
    # Advance mid-period.
    for _ in range(3):
        gate(CheckpointID.CP1, {})
    # Reset restores cache-first ordering.
    gate.on_episode_start()
    assert gate(CheckpointID.CP1, {}) is True   # first of new cache block
    assert gate(CheckpointID.CP1, {}) is True


def test_periodic_gate_ignores_request_context_and_cached_data():
    gate = PeriodicGate(cache_len=1, inference_len=1)
    # Same sequence regardless of extra inputs.
    seq = [gate(CheckpointID.CP1, {}, {"gate_decision": "skip"}) for _ in range(4)]
    assert seq == [True, False, True, False]


def test_periodic_gate_conforms_to_protocol():
    assert isinstance(PeriodicGate(cache_len=2, inference_len=1), GateFunction)


def test_periodic_gate_record_action_is_noop():
    gate = PeriodicGate(cache_len=2, inference_len=1)
    assert gate.record_action(torch.zeros(1)) is None


# ---------------------------------------------------------------------------
# Constructor-level int regressions (G2 Round 1 Blocking #2 defense in depth)
# ---------------------------------------------------------------------------


def test_random_gate_ctor_rejects_fractional_seed():
    with pytest.raises(TypeError, match="seed"):
        RandomGate(p_inference=0.3, seed=0.5)  # type: ignore[arg-type]


def test_random_gate_ctor_rejects_bool_seed():
    with pytest.raises(TypeError, match="seed"):
        RandomGate(p_inference=0.3, seed=True)  # type: ignore[arg-type]


def test_random_gate_ctor_rejects_non_numeric_p():
    with pytest.raises(TypeError, match="p_inference"):
        RandomGate(p_inference="0.3", seed=0)  # type: ignore[arg-type]


def test_periodic_gate_ctor_rejects_fractional_cache_len():
    with pytest.raises(TypeError, match="cache_len"):
        PeriodicGate(cache_len=1.5, inference_len=1)  # type: ignore[arg-type]


def test_periodic_gate_ctor_rejects_bool_inference_len():
    with pytest.raises(TypeError, match="inference_len"):
        PeriodicGate(cache_len=1, inference_len=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ScoreHysteresisGate (server-side N1)
# ---------------------------------------------------------------------------


def _drive(gate: ScoreHysteresisGate, scores):
    """Replay the orchestrator loop: decide (__call__) then feed the verdict.

    ``scores[i]`` is the cp1_score the step would observe *if it searches*.
    A skip step observes cp1_score=None (no search ran), exactly as the
    orchestrator feeds it. Returns the list of searched decisions.
    """
    out = []
    for s in scores:
        searched = gate(CheckpointID.CP1, {})
        out.append(searched)
        cp1 = s if searched else None
        gate.record_verdict(
            CheckpointID.CP1, hit_type=None, cp1_score=cp1,
            winner_id=None, start_t=None, searched=searched,
        )
    return out


def _n1_reference(scores, *, theta_low, theta_high, j, probe_interval):
    """Independent restatement of the N1 decide/observe spec (no exp import).

    Cross-checks ScoreHysteresisGate against a hand-written implementation of
    the same state machine (roadmap N1 / Stage-1b N1GateState).
    """
    searching, low_run, since_probe = True, 0, 0
    out = []
    for s in scores:
        if searching:
            searched = True
        elif probe_interval is not None and since_probe + 1 >= probe_interval:
            searched = True
        else:
            searched = False
        out.append(searched)
        val = float(s) if (searched and s is not None) else -math.inf
        if searching:
            if val < theta_low:
                low_run += 1
                if low_run >= j:
                    searching = False
                    since_probe = 0
            else:
                low_run = 0
        else:
            if searched:
                since_probe = 0
                if val >= theta_high:
                    searching = True
                    low_run = 0
            else:
                since_probe += 1
    return out


def test_score_hysteresis_first_step_searches():
    gate = ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=3, probe_interval=2)
    # No history -> the first decision is always search.
    assert gate(CheckpointID.CP1, {}) is True


def test_score_hysteresis_golden_enter_probe_recover():
    # Hand-computed golden trace (θ_low=0.5, θ_high=0.8, j=3, probe_interval=2):
    #  0..2 search @0.1 (<θ_low) -> low_run hits j=3 -> stop searching
    #  3    skip (since_probe 0->needs +1>=2? no)
    #  4    probe @0.1 (mid<θ_high) -> stay skipping
    #  5    skip
    #  6    probe @0.9 (>=θ_high) -> recover to searching
    #  7    search @0.9
    gate = ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=3, probe_interval=2)
    scores = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9]
    got = _drive(gate, scores)
    assert got == [True, True, True, False, True, False, True, True]


def test_score_hysteresis_band_blocks_chatter():
    # A probe scoring inside the hysteresis band (θ_low <= s < θ_high) must
    # NOT recover: the gate stays skipping.
    gate = ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=2, probe_interval=2)
    #  0,1 search @0.1 -> stop; 2 skip; 3 probe @0.65 (band) -> stay skipping;
    #  4 skip; 5 probe @0.65 -> still skipping.
    scores = [0.1, 0.1, 0.1, 0.65, 0.1, 0.65]
    got = _drive(gate, scores)
    assert got == [True, True, False, True, False, True]


def test_score_hysteresis_probe_none_permanent_skip():
    # probe_interval=None -> once skipping, never probe again this episode.
    gate = ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=2, probe_interval=None)
    scores = [0.1] * 6
    got = _drive(gate, scores)
    assert got == [True, True, False, False, False, False]


def test_score_hysteresis_none_score_counts_as_miss():
    # A searched step with an empty result set (cp1_score=None) is treated as
    # -inf: it counts toward low_run just like a genuine low score.
    gate = ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=3, probe_interval=None)
    scores = [None, None, None, 0.1]
    got = _drive(gate, scores)
    # 3 None-verdicts drive low_run to j -> the 4th step is a skip.
    assert got == [True, True, True, False]


def test_score_hysteresis_call_is_pure():
    # __call__ must not mutate state: repeated calls without record_verdict
    # return the same decision and leave the registers untouched.
    gate = ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=3, probe_interval=2)
    first = gate(CheckpointID.CP1, {})
    snapshot = (gate._searching, gate._low_run, gate._since_probe)
    for _ in range(5):
        assert gate(CheckpointID.CP1, {}) == first
    assert (gate._searching, gate._low_run, gate._since_probe) == snapshot


def test_score_hysteresis_non_finite_fails_open():
    # A non-finite cp1_score (contract violation) forces recovery to full
    # searching rather than risking a wrongful skip.
    gate = ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=1, probe_interval=None)
    # One low score drives into skipping (j=1).
    assert gate(CheckpointID.CP1, {}) is True
    gate.record_verdict(
        CheckpointID.CP1, hit_type=None, cp1_score=0.1,
        winner_id=None, start_t=None, searched=True,
    )
    assert gate(CheckpointID.CP1, {}) is False  # now skipping
    # Feed a NaN verdict on a (probe) searched step -> fail open.
    gate.record_verdict(
        CheckpointID.CP1, hit_type=None, cp1_score=float("nan"),
        winner_id=None, start_t=None, searched=True,
    )
    assert gate._searching is True
    assert gate(CheckpointID.CP1, {}) is True


def test_score_hysteresis_on_episode_start_resets_and_stores_task_key():
    gate = ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=1, probe_interval=None)
    # Drive into skipping.
    _drive(gate, [0.1, 0.1])
    assert gate._searching is False
    gate.on_episode_start(task_key="pick_up_the_cup")
    assert gate._searching is True
    assert gate._low_run == 0 and gate._since_probe == 0
    assert gate._task_key == "pick_up_the_cup"
    # First decision after reset is search again.
    assert gate(CheckpointID.CP1, {}) is True


def test_score_hysteresis_matches_reference_on_long_sequence():
    # Cross-check the gate against an independent restatement of the spec over
    # a fixed, adversarial-ish score sequence (no randomness, no exp import).
    theta_low, theta_high, j, probe_interval = 0.5, 0.8, 3, 2
    scores = [
        0.9, 0.9, 0.2, 0.1, 0.3, 0.85, 0.2, 0.2, 0.2, 0.95,
        0.1, 0.1, 0.4, 0.9, 0.9, 0.1, 0.1, 0.1, 0.7, 0.82,
    ]
    gate = ScoreHysteresisGate(
        theta_low=theta_low, theta_high=theta_high, j=j, probe_interval=probe_interval
    )
    got = _drive(gate, scores)
    expected = _n1_reference(
        scores, theta_low=theta_low, theta_high=theta_high, j=j,
        probe_interval=probe_interval,
    )
    assert got == expected


def test_score_hysteresis_conforms_to_protocol():
    gate = ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=1, probe_interval=1)
    assert isinstance(gate, GateFunction)


def test_score_hysteresis_record_action_is_noop():
    gate = ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=1, probe_interval=1)
    assert gate.record_action(torch.zeros(1)) is None


# --- ScoreHysteresisGate construction validation ---


def test_score_hysteresis_ctor_rejects_theta_high_below_low():
    with pytest.raises(ValueError, match="theta_high >= theta_low"):
        ScoreHysteresisGate(theta_low=0.8, theta_high=0.5, j=1, probe_interval=1)


def test_score_hysteresis_ctor_rejects_non_finite_theta():
    with pytest.raises(ValueError, match="finite"):
        ScoreHysteresisGate(
            theta_low=float("nan"), theta_high=0.5, j=1, probe_interval=1
        )
    with pytest.raises(ValueError, match="finite"):
        ScoreHysteresisGate(
            theta_low=0.5, theta_high=float("inf"), j=1, probe_interval=1
        )


def test_score_hysteresis_ctor_rejects_bad_j():
    with pytest.raises(ValueError, match="j must be >= 1"):
        ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=0, probe_interval=1)


def test_score_hysteresis_ctor_rejects_zero_probe_interval():
    with pytest.raises(ValueError, match="probe_interval"):
        ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=1, probe_interval=0)


def test_score_hysteresis_ctor_allows_none_probe_interval():
    gate = ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=1, probe_interval=None)
    assert gate(CheckpointID.CP1, {}) is True


def test_score_hysteresis_ctor_rejects_bool_params():
    # bool is an int subclass; reject it on every numeric param.
    with pytest.raises(TypeError, match="theta_low"):
        ScoreHysteresisGate(theta_low=True, theta_high=0.8, j=1, probe_interval=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="theta_high"):
        ScoreHysteresisGate(theta_low=0.5, theta_high=False, j=1, probe_interval=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="j"):
        ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=True, probe_interval=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="probe_interval"):
        ScoreHysteresisGate(theta_low=0.5, theta_high=0.8, j=1, probe_interval=True)  # type: ignore[arg-type]


def test_score_hysteresis_ctor_rejects_non_numeric_theta():
    with pytest.raises(TypeError, match="theta_low"):
        ScoreHysteresisGate(theta_low="0.5", theta_high=0.8, j=1, probe_interval=1)  # type: ignore[arg-type]
