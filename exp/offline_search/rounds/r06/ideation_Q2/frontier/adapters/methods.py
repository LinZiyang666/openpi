"""Deployment-only Q2 extra calls and library-budget lottery.

No policy, model, shadow, simulator, profiler, or server is created here. The
existing guard_only plugin executes os_force_miss and owns policy-tail transport.
All stochastic choices are keyed SHA256 uniforms independent of global RNGs.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge
from exp.offline_search.rounds.r06.p1_groot_commit.judge import GrootCommitJudge
from .budget import DOSES, load_calibration, solve

VERSION = 'Q2-deploy-v1'
EXTRA_REASON = 61


def uniform(key, seed, task, init, step, domain):
    """Half-open [0,1), exact 53-bit double; no UID, arrival order, or outcome."""
    if init < 0 or task < 0:
        raise api.ContractError('Q2 coins require original nonnegative task/init metadata')
    payload = [VERSION, key, int(seed), int(task), int(init), int(step), domain]
    bits = int.from_bytes(hashlib.sha256(json.dumps(payload, separators=(',', ':'), ensure_ascii=True).encode()).digest()[:8], 'big')
    # Discard low 11 bits: dividing a full 64-bit int can round to exactly 1.
    return (bits >> 11) * 2.**-53


def validate_random(seed, key):
    if type(seed) is not int or not 0 <= seed < 2**53:
        raise ValueError('random_seed must be an integer in [0, 2**53)')
    if not isinstance(key, str) or not key:
        raise ValueError('randomization_key must be a nonempty frozen campaign identifier')


def validate_dose(dose):
    if isinstance(dose, bool) or not np.isfinite(dose) or not 0 <= dose <= 1:
        raise ValueError('dose must be a finite probability in [0,1]')
    return float(dose)


class _ExtraDose:
    """Mixin: evaluate unchanged B first, then add a source decision only."""
    family = 'r6_q2_extra_dose'

    def __init__(self, dose=0., random_seed=0, randomization_key='Q2-extra-call-v1', **kwargs):
        self.q2_dose = validate_dose(dose)
        validate_random(random_seed, randomization_key)
        self.q2_seed, self.q2_key = random_seed, randomization_key
        defaults = dict(policy_tail_gate='lifecycle', monitor='off', progress_guard='noprog_span',
                        events='none', stuck_guard='vision_confirmed')
        for k, v in defaults.items():
            if k in kwargs and kwargs[k] != v:
                raise ValueError(f'extra-dose deployment B requires {k}={v!r}')
            kwargs[k] = v
        super().__init__(**kwargs)
        if self.base.budget != 1 or self.base.serving != 'anchor_tail' or self.base.gates != 'budget_only':
            raise ValueError('B must use deployed anchor_tail/budget1/budget_only')
        if not self.guards or self.burst != 0:
            raise ValueError('B must retain deployed guards and burst=0')

    def query(self, q):
        res = super().query(q)  # includes unmodified model-specific guard and tail preparation
        if self.q2_dose == 0:
            return res         # exact original Result, extras, and mutable B state
        forced = bool(res.extras.get('os_force_miss', 0))
        u = uniform(self.q2_key, self.q2_seed, q.task_id, q.episode.init, q.step, 'extra-anchor')
        extra = not forced and u < self.q2_dose
        ex = dict(res.extras)
        ex.update(os_q2_version=1., os_q2_dose=self.q2_dose, os_q2_coin=u,
                  os_q2_eligible=float(not forced), os_q2_extra=float(extra),
                  os_q2_base_miss=float(forced), os_q2_p_call=1. if forced else self.q2_dose,
                  os_q2_seed=float(self.q2_seed))
        if extra:
            ex.update(os_force_miss=1., os_reason=float(EXTRA_REASON))
        # Do not rewrite guard flags/progress memos: they still describe B's verdict.
        res.extras = ex
        return res


class ExtraDosePi05(_ExtraDose, CommitJudge):
    """Exactly pi05 B plus independent calls on B-cache anchors."""


class ExtraDoseGroot(_ExtraDose, GrootCommitJudge):
    """Exactly GR00T B; gripper semantics remain in GrootCommitJudge."""


class CacheDose(BlindAWM):
    """Exact A retrieval with a fixed anchor dose and C10 policy lifecycle.

    Keeping A's own retrieval (not B with guards=False) preserves its tie rule.
    """
    family = 'r6_q2_cache_dose'

    def __init__(self, dose=0., random_seed=0, randomization_key='Q2-extra-call-v1', **kwargs):
        self.q2_dose = validate_dose(dose)
        validate_random(random_seed, randomization_key)
        self.q2_seed, self.q2_key = random_seed, randomization_key
        for k, v in dict(serving='anchor_tail', budget=1, gates='budget_only').items():
            if k in kwargs and kwargs[k] != v:
                raise ValueError(f'Q2 A deployment requires {k}={v!r}')
            kwargs[k] = v
        super().__init__(**kwargs)
        self._policy_gate_anchor = None
        self.q2_episode = None

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        self.finish_adapter_fit(lib, ctx)

    def finish_adapter_fit(self, lib, ctx):
        if self.H < 10:
            raise api.ContractError('10 controls exceed fitted action horizon')

    def reset(self, episode):
        super().reset(episode)
        self._policy_gate_anchor = None
        self.q2_episode = (str(episode.uid), int(episode.task_id), int(episode.init))
        self._reset_assignment(episode)

    def _reset_assignment(self, episode):
        self.q2_episode_dose = self.q2_dose
        self.q2_episode_propensity = 1.
        self.q2_episode_coin = 0.
        self.q2_task_p = self.q2_dose

    def _zero_identity(self):
        return self.q2_dose == 0

    def _assignment_extras(self):
        return dict(os_q2_episode_dose=self.q2_episode_dose,
                    os_q2_episode_propensity=self.q2_episode_propensity,
                    os_q2_episode_coin=self.q2_episode_coin, os_q2_task_p=self.q2_task_p)

    def query(self, q):
        ident = (str(q.episode.uid), int(q.task_id), int(q.episode.init))
        if self.q2_episode != ident:
            self.reset(q.episode)
        res = super().query(q)
        if self._zero_identity():
            return res
        # Same metadata C10 saves before plugin.invalidate_anchor on a MISS.
        a = self._anchor
        self._policy_gate_anchor = dict(step=a['step'], episode=a['episode'], task=a['task'],
                                        rows=a['rows'].copy(), weights=a['weights'].copy())
        u = uniform(self.q2_key, self.q2_seed, q.task_id, q.episode.init, q.step, 'dose-anchor')
        call = u < self.q2_episode_dose
        res.extras = {**res.extras, **self._assignment_extras(),
                      'os_q2_version': 1., 'os_q2_coin': u, 'os_q2_seed': float(self.q2_seed),
                      'os_q2_dose': self.q2_episode_dose, 'os_q2_p_call': self.q2_episode_dose,
                      'os_q2_eligible': 1., 'os_q2_extra': float(call), 'os_q2_base_miss': 0.,
                      'os_force_miss': float(call), 'os_reason': float(EXTRA_REASON if call else 0)}
        return res

    def policy_tail_step(self, bq):
        # Reuse C10's lifecycle verbatim, through an ephemeral facade. No cycles
        # are retained in the fitted object or per-connection clone.
        gate = SimpleNamespace(policy_tail_gate='lifecycle', monitor='off',
                               _policy_gate_anchor=self._policy_gate_anchor, base=self)
        try:
            return CommitJudge.policy_tail_step(gate, bq)
        finally:
            self._policy_gate_anchor = gate._policy_gate_anchor


class RiskLottery(CacheDose):
    """One episode dose lottery from a frozen library-only target-IR solve."""
    family = 'r6_q2_risk_lottery'

    def __init__(self, rho=0., allocation='risk', calibration_path=None,
                 calibration_sha256=None, **kwargs):
        if isinstance(rho, bool) or not np.isfinite(rho) or rho < 0:
            raise ValueError('rho must be finite and nonnegative')
        if allocation not in ('risk', 'uniform'):
            raise ValueError('allocation must be risk or uniform')
        if 'dose' in kwargs:
            raise ValueError('RiskLottery sets dose from rho; do not supply dose')
        if not calibration_path or not calibration_sha256:
            raise ValueError('frozen library calibration path and SHA required')
        self.q2_rho, self.q2_allocation = float(rho), allocation
        self.q2_calibration_path, self.q2_calibration_sha = str(calibration_path), str(calibration_sha256)
        self.q2_budget = None
        super().__init__(dose=0., **kwargs)

    def finish_adapter_fit(self, lib, ctx):
        super().finish_adapter_fit(lib, ctx)
        deployed = lib if self.cand_name == 'current' else ctx.open_library(self.cand_name)
        cal = load_calibration(self.q2_calibration_path, self.q2_calibration_sha, deployed, ctx)
        self.q2_budget = solve(cal['tasks'], self.q2_rho, cal['c1'], self.q2_allocation)

    def _zero_identity(self):
        return self.q2_rho == 0

    def _reset_assignment(self, episode):
        if self.q2_budget is None:
            raise api.ContractError('RiskLottery must be fitted before reset/query')
        task = self.q2_budget['tasks'][int(episode.task_id)]
        weights = np.asarray(task['weights'])
        u = uniform(self.q2_key, self.q2_seed, episode.task_id, episode.init, -1, 'episode-dose')
        j = int(np.searchsorted(np.cumsum(weights), u, side='right'))
        j = min(j, len(DOSES)-1)
        if weights[j] <= 0:
            raise api.ContractError('lottery selected a zero-probability dose')
        self.q2_episode_dose, self.q2_episode_propensity = float(DOSES[j]), float(weights[j])
        self.q2_episode_coin, self.q2_task_p = u, task['p']

    def _assignment_extras(self):
        return {**super()._assignment_extras(), 'os_q2_rho': self.q2_rho,
                'os_q2_clamped': float(self.q2_budget['clamped']),
                'os_q2_predicted_IR': self.q2_budget['predicted_IR'],
                'os_q2_uniform': float(self.q2_allocation == 'uniform')}
