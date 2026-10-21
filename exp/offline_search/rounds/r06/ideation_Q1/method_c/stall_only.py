"""Q3 stall gate as the R10Recipe inner judge (owner 10-13 / 10-20: the stall gate is Q3 only).

Takes the place of the removed R8 TriggerCommitJudge (noprog_span) in R15/R16 fits. The fitted
cache (``base``) is unchanged. At every fresh query, i.e. a segment boundary, the R6 Q3 tracker
observes the query at control index step*5 (R6-C clock). slow_confirmed makes this decision a
real model call (os_force_miss=1, os_reason=63), except on the anchor right after a stall call
(R6-C-v2 one-anchor cooldown; nominal lottery 0, so nothing else calls). slow_ambiguous adds no
mid-segment LOOK (as in R17 serving). Blind steps are the base's own; the policy tail is C10's
lifecycle gate (CommitJudge facade, as in methods.CalibratedRescue). The plugin turns the
forced flag into a MISS with ``--os-judge stall_only``; a knob layer above may add its own.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np

from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge
from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallTracker
from .stall_bridge import STATE, call_probability, starts_cooldown

BLOCK_CONTROLS = 5   # one plugin decision step = five controls (R6-C methods.py:181)
STALL_REASON = 63


class Q3StallInner:
    family = 'q3_stall_inner'

    def __init__(self, base, C, stall_model):
        self.base, self.C, self.stall_model = base, C, stall_model
        self.name = 'Q3Stall__' + base.name
        self._clear()

    def __getattr__(self, name):
        # Everything else (invalidate_anchor, bytes_per_entry, last_blind_extras, ...) is the base's.
        if name.startswith('__') or name in ('base', 'C', 'stall_model'):
            raise AttributeError(name)
        base = self.__dict__.get('base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __deepcopy__(self, memo):
        # Per-connection clone: the fitted Q3 model is read-only and shared.
        new = object.__new__(type(self))
        memo[id(self)] = new
        for key, value in vars(self).items():
            setattr(new, key, value if key == 'stall_model' else copy.deepcopy(value, memo))
        return new

    def _clear(self):
        self.tracker = None
        self._identity = None
        self._anchor_index = 0
        self._cooldown_anchor = None
        self._policy_gate_anchor = None
        self._memo_step, self._memo = -1, None

    def reset(self, episode):
        self.base.reset(episode)
        self._clear()
        self._identity = (str(episode.uid), int(episode.task_id), int(episode.init))
        self.tracker = StallTracker(self.stall_model, int(episode.task_id))

    def query(self, q):
        if self._identity != (str(q.episode.uid), int(q.task_id), int(q.episode.init)):
            self.reset(q.episode)
        step = int(q.step)
        if step == self._memo_step:
            return copy.deepcopy(self._memo)
        res = self.base.query(q)
        self._policy_gate_anchor = copy.deepcopy(self.base._anchor)
        self.tracker.observe(q, step * BLOCK_CONTROLS)
        status = self.tracker.status()
        state = status['state']
        if state not in STATE:
            raise ValueError(f'unknown Q3 stall state {state!r}')
        cooled = self._cooldown_anchor is not None and self._anchor_index == self._cooldown_anchor + 1
        call = call_probability(state, cooled, 0.) == 1.
        if starts_cooldown(call, state, 'stall'):
            self._cooldown_anchor = self._anchor_index
        log = dict(os_q3_state=float(STATE[state]), os_q3_call=float(call), os_q3_cooldown=float(cooled),
                   os_q3_anchor=float(self._anchor_index), os_q3_control=float(step * BLOCK_CONTROLS))
        for field in ('delta_hat', 'e90', 'a10', 'window_span', 'W', 'reference_episodes'):
            value = status.get(field)
            if value is not None and np.isfinite(value):
                log['os_q3_' + field] = float(value)
        extras = res.extras or {}
        res.extras = {'os_force_miss': float(call), 'os_reason': float(STALL_REASON if call else 0), **log,
                      **{k: v for k, v in extras.items() if k not in ('os_force_miss', 'os_reason')}}
        self._anchor_index += 1
        self._memo_step, self._memo = step, copy.deepcopy(res)
        return res

    def blind_step(self, bq):
        return self.base.blind_step(bq)

    def policy_tail_step(self, bq):
        facade = SimpleNamespace(policy_tail_gate='lifecycle', monitor='off', base=self.base,
                                 _policy_gate_anchor=self._policy_gate_anchor)
        try:
            return CommitJudge.policy_tail_step(facade, bq)
        finally:
            self._policy_gate_anchor = facade._policy_gate_anchor
