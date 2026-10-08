"""Serving controller; exact R11 coins and accepted-history convention."""
from __future__ import annotations
import math
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r11.knob.recipe import history, uniform

REASON_TASK_PERIODIC = 73

def caps(reference, priority, bias=0., gain=.35):
    if gain == 0 and bias == 0:
        return np.full(10, reference, dtype=float)
    return np.maximum(np.expm1(np.log1p(reference) + bias - gain * np.asarray(priority)), 0.)

def propose(q, guard, settings):
    task = int(q.task_id)
    if not 0 <= task < 10 or task != int(q.episode.task_id):
        raise api.ContractError('R14 needs the original instruction task')
    _, last, run, ledger = history(q)
    K = float(settings[task])
    if len(settings) != 10 or not math.isfinite(K) or K < 0:
        raise api.ContractError('R14 needs ten finite nonnegative task caps')
    u = uniform('R11-gap-v1', 0, task, int(q.episode.init), last, 'cap')
    floor = math.floor(K)
    cap = floor + int(K - floor > 0 and u < K - floor)
    knob = not guard and run >= cap
    return dict(miss=bool(guard or knob), guard=bool(guard), knob=bool(knob),
                K=K, cap=cap, coin=u, run=run, N=ledger.decisions,
                V=ledger.looks, M=ledger.calls)
