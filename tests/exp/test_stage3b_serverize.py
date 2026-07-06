"""Stage 3b equivalence: server ScoreHysteresisGate(L) == client N4GateState(L).

Cross-layer test (allowed in tests/): drives the src server gate and the exp
client machine in lockstep over shared (score, hit_type) traces and asserts the
decision sequences match step for step. The server gate compares the ``HitType``
enum; the client machine reads the wire string, so the enum -> string translation
happens ONLY on the client (N4) reference side (``hit.name``). All non-manual.
"""

from __future__ import annotations

import pytest

from openpi.cache.components.gate import ScoreHysteresisGate
from openpi.cache.components.judge import HitType
from openpi.cache.types import CheckpointID

from exp.gate_research.n4_gate_client import N4GateState

# (theta_low, theta_high, j, probe_interval/M, L)
_PARAM_SETS = [
    (0.968929, 0.968929, 3, 3, 6),   # 3a spatial winning point
    (0.996873, 0.996873, 3, 3, 6),   # 3a l10 winning point
    (0.5, 0.8, 2, 3, 4),             # synthetic, tighter thresholds
    (0.5, 0.8, 1, None, 3),          # never-probe
    (0.5, 0.8, 2, 2, 2),             # aggressive L
]

# A fixed, adversarial-ish (score, hit_type enum) trace exercising: long FULL_HIT
# runs (V2 injection), low-score runs (V1 enter skipping), probes, WS/MISS resets.
_HTS = [HitType.FULL_HIT, HitType.WARM_START, HitType.MISS]
_TRACE = [
    (0.99, HitType.FULL_HIT), (0.99, HitType.FULL_HIT), (0.99, HitType.FULL_HIT),
    (0.99, HitType.FULL_HIT), (0.99, HitType.FULL_HIT), (0.99, HitType.FULL_HIT),
    (0.1, HitType.MISS), (0.1, HitType.MISS), (0.1, HitType.MISS),
    (0.2, HitType.WARM_START), (0.95, HitType.FULL_HIT), (0.99, HitType.FULL_HIT),
    (0.99, HitType.FULL_HIT), (0.3, HitType.MISS), (0.99, HitType.FULL_HIT),
    (0.99, HitType.WARM_START), (0.99, HitType.FULL_HIT), (0.99, HitType.FULL_HIT),
    (0.4, HitType.MISS), (0.5, HitType.FULL_HIT), (0.99, HitType.FULL_HIT),
]


def _drive_pair(params, include_ws=False):
    tl, th, j, pi, L = params
    server = ScoreHysteresisGate(theta_low=tl, theta_high=th, j=j, probe_interval=pi,
                                 L=L, include_ws=include_ws)
    client = N4GateState(tl, th, j, pi, L, include_ws=include_ws)
    s_seq, c_seq = [], []
    for score, ht in _TRACE:
        # server decide (bool) vs client decide (str)
        s_searched = server(CheckpointID.CP1, {})
        c_decision = client.decide()
        s_seq.append(s_searched)
        c_seq.append(c_decision == "search")
        # observe both: server gets the enum; client gets the wire string (.name)
        server.record_verdict(
            CheckpointID.CP1, hit_type=ht,
            cp1_score=(score if s_searched else None),
            winner_id=None, start_t=None, searched=s_searched,
        )
        if c_decision == "search":
            client.observe("search", ht.name, score)
        else:
            client.observe("skip", None, None)
    return s_seq, c_seq


@pytest.mark.parametrize("params", _PARAM_SETS)
def test_server_equals_client_default(params):
    s_seq, c_seq = _drive_pair(params, include_ws=False)
    assert s_seq == c_seq


@pytest.mark.parametrize("params", _PARAM_SETS)
def test_server_equals_client_include_ws(params):
    s_seq, c_seq = _drive_pair(params, include_ws=True)
    assert s_seq == c_seq


def test_server_equals_client_L_none_is_pure_n1():
    # With L=None the two must also agree (pure N1 on both sides).
    tl, th, j, pi = 0.5, 0.8, 2, 3
    server = ScoreHysteresisGate(theta_low=tl, theta_high=th, j=j, probe_interval=pi, L=None)
    client = N4GateState(tl, th, j, pi, L=10**9)  # client has no None; huge L never fires
    s_seq, c_seq = [], []
    for score, ht in _TRACE:
        s_searched = server(CheckpointID.CP1, {})
        c_decision = client.decide()
        s_seq.append(s_searched)
        c_seq.append(c_decision == "search")
        server.record_verdict(CheckpointID.CP1, hit_type=ht,
                              cp1_score=(score if s_searched else None),
                              winner_id=None, start_t=None, searched=s_searched)
        if c_decision == "search":
            client.observe("search", ht.name, score)
        else:
            client.observe("skip", None, None)
    assert s_seq == c_seq
