"""R13 distance feedback on the unchanged R10/R11 serving lifecycle.

No fitting or exploration module is imported when loading a serving artifact.
Off and periodic dispatch literally to R11. Pure C10 is periodic with K=0;
pure C5 is emitted with the existing contextual pure_inference arm mode.
"""
from __future__ import annotations

import copy
import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r10.recipe.recipe import R10Recipe
from exp.offline_search.rounds.r11.knob.recipe import R11Knob, history

COSTS = {'pi05': (.152, .848), 'groot': (.148, .852)}
METHODS = ('off', 'periodic', 'distance_feedback', 'distance_static_b', 'pure_c10')
REASONS = {'distance_feedback': 69, 'distance_static_b': 70}


def accepted_history(q):
    """R11 history, additionally checked against the live plugin's commits."""
    out = history(q)
    if np.any((~np.asarray(q.hist_has_vision, bool)) & (np.asarray(q.hist_hit) != 1)):
        raise api.ContractError('blind slots must be accepted HITs')
    session = getattr(q, '_s', None)
    if session is not None:
        n = int(q.step)
        if (session.b_aex.n != n or len(session.hits) != n
                or len(session.has_vision) != n + 1
                or not np.array_equal(q.hist_hit, session.hits)
                or not np.array_equal(q.hist_has_vision, session.has_vision[:n])):
            raise api.ContractError('R13 history differs from accepted plugin decisions')
        if session.rt.opts.os_log_inputs:
            if len(session._recs) != n or any(
                    bool(r['hit']) != bool(session.hits[i])
                    or bool(r['vision']) != bool(session.has_vision[i])
                    for i, r in enumerate(session._recs)):
                raise api.ContractError('R13 decision records differ from accepted history')
    return out


class R13Knob(R11Knob):
    family = 'r13_knob'

    def __init__(self, library='current', episode_subset=None, method='off', target=None):
        if method not in METHODS:
            raise ValueError(method)
        underlying = 'periodic' if method == 'pure_c10' else (method if method in ('off', 'periodic') else 'off')
        super().__init__(library, episode_subset, method=underlying,
                         target=.5 if method == 'pure_c10' else target)
        if method.startswith('distance') and (target is None or not np.isfinite(target) or not 0 < target <= .5):
            raise ValueError('distance knob needs a target IR in (0,.5]')
        self.knob_method = 'periodic' if method == 'pure_c10' else method
        self._ledger_checkpoint = (0, 0, 0)
        self._accepted_prefix = ((), ())

    def fit(self, lib, ctx):
        raise api.ContractError('R13 serving requires an ordinary frozen fit artifact')

    def configure(self, settings, model):
        if self.knob_method in ('off', 'periodic'):
            return super().configure(settings, model)
        cfg = copy.deepcopy(settings)
        required = ('lo', 'hi', 'width', 'bias') if self.knob_method == 'distance_feedback' else ('threshold',)
        if model not in COSTS or any(not np.isfinite(cfg[k]) for k in required):
            raise ValueError('invalid R13 scalar settings')
        if self.knob_method == 'distance_feedback' and (cfg['hi'] <= cfg['lo'] or cfg['width'] != 1.):
            raise ValueError('frozen R13 scale/width violated')
        if self.knob_method == 'distance_static_b' and cfg['threshold'] < 0:
            raise ValueError('static raw threshold must be nonnegative')
        self.knob_settings, self.knob_model = cfg, model
        self.name = f'R13Knob_{self.knob_method}_{self.target}__{self.name}'

    def reset(self, episode):
        super().reset(episode)
        self._ledger_checkpoint = (0, 0, 0)
        self._accepted_prefix = ((), ())

    def query(self, q):
        if self.knob_method in ('off', 'periodic'):
            return super().query(q)
        step = int(q.step)
        if step == self._proposal_step:
            return copy.deepcopy(self._proposal)
        if step < self._proposal_step:
            raise api.ContractError('reset R13 on a new episode')
        _, _, _, ledger = accepted_history(q)
        N, V, M = ledger.decisions, ledger.looks, ledger.calls
        oldN, oldV, oldM = self._ledger_checkpoint
        if not 0 <= M-oldM <= V-oldV <= N-oldN:
            raise api.ContractError('accepted R13 counters must advance monotonically')
        hv, hits = tuple(map(bool, q.hist_has_vision)), tuple(map(int, q.hist_hit))
        if (hv[:oldN], hits[:oldN]) != self._accepted_prefix:
            raise api.ContractError('previous accepted history was rewritten')
        # Exactly one inherited query. Its corrected proposal and gate anchor
        # remain alive until the plugin commits the real policy or cache action.
        res = R10Recipe.query(self, q)
        ex = dict(res.extras)
        guard = bool(ex.get('os_force_miss', 0))
        d = float(np.min(np.asarray(self.inner.base._dist(q)[7], np.float64)))
        finite = bool(np.isfinite(d) and d >= 0)
        cfg = self.knob_settings
        cv, cm = COSTS[self.knob_model]
        debt = (cv*V + cm*M - self.target*N) / cm
        if self.knob_method == 'distance_feedback':
            u = float(np.clip((np.log1p(d)-cfg['lo'])/(cfg['hi']-cfg['lo']), 0., 1.)) if finite else float('nan')
            theta = cfg['bias'] + debt/cfg['width']
            trigger = not finite or u > theta
            width, bias = cfg['width'], cfg['bias']
        else:
            u, theta = d, cfg['threshold']
            trigger = not finite or d > theta
            width, bias = 0., 0.
        knob = bool(trigger and not guard)
        if knob:
            ex.update(os_force_miss=1., os_reason=float(REASONS[self.knob_method]))
        # All SPEC audit scalars precede even inherited os_* fields (40-field
        # mixed-log cap). Nonfinite d/u survive in the NPZ and flags explain null
        # JSON scalars; no fabricated finite score replaces the invalid distance.
        audit = dict(os_r13_raw_distance=d, os_r13_u=u, os_r13_theta=float(theta),
                     os_r13_D=float(debt), os_r13_target=float(self.target),
                     os_r13_width=float(width), os_r13_bias=float(bias),
                     os_r13_N=float(N), os_r13_V=float(V), os_r13_M=float(M),
                     os_r13_guard=float(guard), os_r13_knob_call=float(knob),
                     os_r13_nonfinite=float(not finite),
                     os_r13_saturated_low=float(theta < 0) if self.knob_method == 'distance_feedback' else 0.,
                     os_r13_saturated_high=float(theta >= 1) if self.knob_method == 'distance_feedback' else 0.,
                     os_r13_version=1., os_r13_static=float(self.knob_method == 'distance_static_b'))
        res.extras = {**audit, **{k:v for k,v in ex.items() if k.startswith('os_')},
                      **{k:v for k,v in ex.items() if not k.startswith('os_')}}
        self._ledger_checkpoint = (N, V, M)
        self._accepted_prefix = (hv, hits)
        self._proposal_step, self._proposal = step, copy.deepcopy(res)
        return res
