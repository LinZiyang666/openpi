"""C: exact A retrieval + calibrated rescue decisions, no deployment profiling.

The adapter delegates policy-tail validation to C10 unchanged. R6-C-v2 applies
one free-anchor cooldown only after a stall call. Lottery calls have no cooldown.
The old all-call scope remains available only for in-memory replay tests.
SELECTION §7b (still R6-C-v2): a slow_ambiguous anchor draws the ordinary R /
uniform lottery like an ok anchor; its extra LOOK is scheduled only when that
lottery does not call. The rule lives in stall_bridge and is shared with budget.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.closed_loop.blind import BlindResult, LookReason
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge
from exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.methods import uniform, validate_random
from .common import SCHEMA, CONTROLLER_VERSION, load_base, read_bank, sha
from .stall_bridge import (load_stall, extra_look, STATE, verify_stall, AMBIGUOUS_RULE,
                           call_probability, starts_cooldown, scheduled_look)


class _MetricCodeMatrix:
    """Capture A's already computed code without changing its matrix multiply."""
    def __init__(self, matrix):
        self.matrix = matrix
        self.code = None

    def __matmul__(self, code):
        self.code = code
        return self.matrix @ code


class CalibratedRescue(api.Method):
    controller_version = CONTROLLER_VERSION
    tier = 'T1'
    family = 'r6_calibrated_rescue'
    uses_gt = False
    uses_nonlibrary_action = False  # plugin supplies policy actions, as in B

    def __init__(self, rho, placement='R', stall_model_path=None, calibration_path=None,
                 random_seed=0, randomization_key=CONTROLLER_VERSION, cooldown_scope='stall'):
        if isinstance(rho, bool) or not np.isfinite(rho) or rho < 0:
            raise ValueError('rho must be finite and nonnegative')
        if placement not in ('uniform', 'R') or not calibration_path:
            raise ValueError('placement must be uniform/R and calibration_path is required')
        validate_random(random_seed, randomization_key)
        if cooldown_scope not in ('stall','all'):
            raise ValueError('cooldown_scope must be stall/all; all is for tests only')
        self.cooldown_scope = cooldown_scope
        self.rho, self.placement = float(rho), placement
        self.stall_model_path, self.calibration_path = stall_model_path, str(calibration_path)
        self.random_seed, self.randomization_key = random_seed, randomization_key
        self.name = f'{CONTROLLER_VERSION}_{placement}_rho{rho:g}'
        self.base = None
        self._episode_identity = None

    def fit(self, lib, ctx):
        self._load(ctx.cell, allow_dryrun=False)

    def _load(self, cell, *, allow_dryrun=False):
        path = Path(self.calibration_path)
        cal = json.loads(path.read_text())
        if cal['schema'] != SCHEMA + '.calibration':
            raise ValueError('unknown C calibration schema')
        if cal['status'] != 'NONTEST_BVAL' and not allow_dryrun:
            raise ValueError('DRYRUN_TEST_INITS artifacts are forbidden for deployment/validation')
        if self.cooldown_scope=='all' and not allow_dryrun:
            raise ValueError('all-call cooldown is test-only; deployment requires stall scope')
        if cal.get('controller_version')!=CONTROLLER_VERSION or cal.get('cooldown_scope')!=self.cooldown_scope:
            raise ValueError('calibration controller version/cooldown scope mismatch; refit for R6-C-v2')
        if cal.get('ambiguous_rule') != AMBIGUOUS_RULE:
            raise ValueError('calibration predates SELECTION 7b slow_ambiguous lottery rule; refit')
        if cal['cell'].rsplit('_', 1)[0] + '_cache' != cell:
            raise ValueError('calibration cell mismatch')
        self.base, _ = load_base(cal['base_source'])
        bank_path = path.parent / cal['r_bank_path']
        if sha(bank_path) != cal['r_bank_sha256']:
            raise ValueError('calibration R bank metadata hash mismatch')
        self.bank, arrays = read_bank(bank_path, self.base)
        self.r_bank = arrays['r']
        self.a, self.b = float(cal['intercept']), float(cal['slope'])
        self.block_controls = int(self.bank['interface']['block_controls'])
        self.commit_controls = int(self.bank['interface']['commit_controls'])
        # Existing plugin contract is explicitly five controls/request, one tail.
        # Refuse rather than silently mishandle another plugin interface.
        if self.block_controls != 5 or self.commit_controls != 10:
            raise api.ContractError('current A/plugin adapter supports b=5, h=10; port its tail interface for another block length')
        self.stall_model, self.tracker_class, identity = load_stall(self.stall_model_path)
        verify_stall(self.stall_model,self.base,self.bank)
        mode = 'stall' if self.stall_model_path else 'no_stall'
        if identity != cal['stall_fingerprint'][mode]:
            raise ValueError('stall artifact differs from budget calibration')
        self.budget = cal['solutions'][mode][self.placement].get(format(self.rho, '.12g'))
        if self.budget is None:
            raise ValueError('target rho was not calibrated; run fit_calibration for this target')
        if not self.budget['feasible']:
            raise ValueError(f"infeasible target rho={self.rho}: [{self.budget['floor']}, {self.budget['ceiling']}]")
        if identity == 'inactive_stub_missing_module' and not allow_dryrun:
            raise ValueError('configured stall module is unavailable; stub cannot be a validation C arm')
        self.parameter = self.budget['parameter']
        self.calibration_digest = sha(path)
        self.fit_info = dict(c_calibration_sha256=self.calibration_digest,
                             r_bank_fingerprint=self.bank['retrieval_fingerprint'],
                             stall_fingerprint=identity, budget=self.budget,
                             controller_version=CONTROLLER_VERSION, cooldown_scope=self.cooldown_scope,
                             ambiguous_rule=AMBIGUOUS_RULE,
                             cooldown='one free anchor after a stall call' if self.cooldown_scope=='stall' else 'TEST ONLY: after every call')

    def reset(self, episode):
        if self.base is None:
            raise api.ContractError('C must be fitted before reset/query')
        self.base.reset(episode)
        self._episode_identity = (str(episode.uid), int(episode.task_id), int(episode.init))
        self.tracker = self.tracker_class(self.stall_model, int(episode.task_id))
        self._policy_gate_anchor = None
        self._anchor_index = 0
        self._last_cooldown_anchor = None
        self._last_extra_control = None
        self._look_due_step = None
        self._last_log = {}

    @property
    def last_blind_extras(self):
        return {**getattr(self.base, 'last_blind_extras', {}), **self._carried_log()}

    def _carried_log(self):
        return {**self._last_log, 'os_c_carried_anchor_score': 1.,
                'os_c_fresh': 0., 'os_c_anchor_p': self._last_log.get('os_c_p',0.),
                'os_c_p': 0., 'os_c_call': 0., 'os_c_stall_call': 0., 'os_c_extra_look': 0.}

    def invalidate_anchor(self):
        self.base.invalidate_anchor()

    def _metric_distance(self, original, task, name, capture, query):
        facade = copy.copy(self.base)
        facade.tasks = dict(self.base.tasks)
        table = copy.copy(task)
        setattr(table, name, capture)
        facade.tasks[int(query.task_id)] = table
        return original.__func__(facade, query)

    def _query_with_metric_code(self, q):
        """Run the original A query once and retain its early/main query code.

        A._dist keeps the code local. A query-local shallow facade substitutes
        only the matrix consuming that code; its multiplication still executes
        on the original ndarray. Fitted arrays/tasks are never mutated. The
        temporary _dist hook belongs to this episode's stateful base and is
        restored even on failure; no hook is retained in a fitted artifact.
        """
        base = self.base
        original = base._dist
        task = base.tasks[int(q.task_id)]
        early = int(q.step) == 0 and base.early
        name = ('Z0' if task.Z0 is not None else 'A0') if early else 'Z'
        capture = _MetricCodeMatrix(getattr(task, name))

        def distance(query):
            return self._metric_distance(original, task, name, capture, query)

        previous = base.__dict__.get('_dist')
        had_override = '_dist' in base.__dict__
        base._dist = distance
        try:
            result = base.query(q)
        finally:
            if had_override:
                base._dist = previous
            else:
                del base._dist
        return result, dict(metric_code=capture.code, metric='early' if early else 'main')

    def query(self, q):
        identity = (str(q.episode.uid), int(q.task_id), int(q.episode.init))
        if self._episode_identity != identity:
            self.reset(q.episode)
        control_index = int(q.step) * self.block_controls
        if self.stall_model is None:
            self.tracker.observe(q, control_index)
            result = self.base.query(q)
        else:
            try:
                result, key = self._query_with_metric_code(q)
            except Exception:
                # Preserve invalid-observation window clearing on a failed A
                # query. Successful queries never repeat the raw projection.
                self.tracker.observe(q, control_index)
                raise
            self.tracker.observe(key, control_index)  # includes fresh extra LOOKs
        status = self.tracker.status()
        if status['state'] not in STATE:
            raise api.ContractError('unknown stall state')
        anchor = self.base._anchor
        R = float(np.asarray(anchor['weights'], float) @ self.r_bank[anchor['rows']])
        estimate = self.a + self.b*R
        nominal = float(self.parameter if self.placement == 'uniform' else min(1., self.parameter*estimate))
        cooled = self._last_cooldown_anchor is not None and self._anchor_index == self._last_cooldown_anchor + 1
        look_eligible = extra_look(status, control_index, self._last_extra_control, self.commit_controls)
        p = call_probability(status['state'], cooled, nominal)  # §7b: ambiguous draws the lottery
        coin = uniform(self.randomization_key, self.random_seed, q.task_id, q.episode.init, q.step, 'dose-anchor')
        call = coin < p
        stall_call = call and status['state']=='slow_confirmed'
        look = scheduled_look(look_eligible, call)  # §7b: LOOK only when the lottery did not call
        if call:
            if starts_cooldown(call, status['state'], self.cooldown_scope):
                self._last_cooldown_anchor = self._anchor_index
            self._policy_gate_anchor = dict(step=anchor['step'], episode=anchor['episode'], task=anchor['task'],
                rows=anchor['rows'].copy(), weights=anchor['weights'].copy())
        else:
            self._policy_gate_anchor = None
        if look:
            self._last_extra_control = control_index
            self._look_due_step = int(q.step) + 1  # min(b,L) controls; same blind-LOOK veto as B
        else:
            self._look_due_step = None
        self._last_log = dict(os_c_version=2., os_c_stall_call=float(stall_call), os_c_fresh=1., os_c_R=R, os_c_Ehat=estimate, os_c_p=p, os_c_nominal_p=nominal,
            os_c_coin=coin, os_c_stall_state=float(STATE[status['state']]), os_c_extra_look=float(look),
            os_c_cooldown=float(cooled), os_c_anchor=float(self._anchor_index), os_c_rho=self.rho,
            os_c_commit_controls=float(self.commit_controls if not look else self.block_controls),
            os_c_control_index=float(control_index), os_c_hold=1., os_c_call=float(call))
        for field in ('delta_hat', 'e90', 'a10', 'window_span', 'W'):
            value = status.get(field)
            if value is not None and np.isfinite(value):
                self._last_log['os_c_stall_' + field] = float(value)
        result.extras = {**self._last_log, 'os_force_miss': float(call),
                         'os_reason': float(63 if call and status['state']=='slow_confirmed' else 62 if call else 0),
                         **result.extras}
        self._anchor_index += 1
        return result

    def blind_step(self, bq):
        if self._look_due_step == int(bq.step):
            self._look_due_step = None
            self.base.last_blind_extras = {**self._last_log, 'os_c_forced_look': 1.}
            return LookReason(8, 'c_slow_ambiguous')
        result = self.base.blind_step(bq)
        if isinstance(result, BlindResult):
            return BlindResult(result.action, result.rows, result.weights, result.library,
                               {**result.extras, **self._carried_log()})
        return result

    def policy_tail_step(self, bq):
        facade = SimpleNamespace(policy_tail_gate='lifecycle', monitor='off', base=self.base,
                                 _policy_gate_anchor=self._policy_gate_anchor)
        try:
            result = CommitJudge.policy_tail_step(facade, bq)
        finally:
            self._policy_gate_anchor = facade._policy_gate_anchor
        if isinstance(result, BlindResult):
            return BlindResult(result.action, result.rows, result.weights, result.library,
                               {**result.extras, **self._carried_log()})
        return result

    def bytes_per_entry(self):
        return self.base.bytes_per_entry() + 8 if self.base is not None else 8
