"""Opt-in library-risk H2 guard and wrist-first M2 over the frozen R10/R11 stack.

Fitted coefficients, library normalization and labelled dev thresholds are
embedded in each artifact. Serving imports no explorer or calibration code.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r11.knob.recipe import R11Knob
from .lookhalf import R12LookHalf, NeedCamera, _View
from .lookhalf_math import correct, freeze_fit

METHODS = ('off', 'h2', 'm2')
FEATURES = ('W_d1rel', 'step', 'lagrel', 'speed2', 'osc', 'mismatch', 'W_adv')


def feature_matrix(table, *, fill=None):
    """Apply Opus's seven-feature transform; optional frozen library imputation."""
    columns = []
    for i, name in enumerate(FEATURES):
        v = np.atleast_1d(np.asarray(table[name], dtype=np.float64)).copy()
        if name == 'W_adv':
            v = np.clip(np.nan_to_num(v, nan=5.), -5., 8.)
        elif name == 'step':
            v /= 100.
        elif name == 'mismatch':
            v = np.log1p(np.nan_to_num(v, nan=1.))
        default = fill[i] if fill is not None else (np.nanmedian(v) if np.isfinite(v).any() else 0.)
        columns.append(np.nan_to_num(v, nan=default))
    return np.stack(columns, axis=1)


def risk_score(X, coef):
    """Evaluate the standardized logistic risk, with intercept first."""
    X = np.atleast_2d(np.asarray(X, np.float64))
    z = np.c_[np.ones(len(X)), (X-np.asarray(coef['mu']))/np.asarray(coef['sd'])]
    return 1./(1.+np.exp(-(z@np.asarray(coef['w']))))


def h2_flags(step, advance, risk, theta_a, theta_b):
    """Frozen stall-or-risk rule, eligible only from decision four."""
    eligible = np.asarray(step) >= 4
    stall = eligible & (np.nan_to_num(np.asarray(advance, float), nan=9.) <= .5)
    return (stall & (np.asarray(risk) >= theta_a)) | (eligible & (np.asarray(risk) >= theta_b))


class _H2Judge:
    """Replace layer two before R11 sees it, preserving the cache and its memos."""

    def __init__(self, deployed):
        self.deployed = deployed
        self.decision = None

    def __getattr__(self, name):
        if name.startswith('__') or name == 'deployed':
            raise AttributeError(name)
        return getattr(self.deployed, name)

    def query(self, q):
        """Commit the underlying synthesis once and substitute H2's guard bit."""
        if self.decision is None:
            raise api.ContractError('H2 verdict must be prepared before synthesis')
        res = self.deployed.query(q)
        ex = dict(res.extras)
        flags = (int(ex['os_flags']) & ~8) | (8 if self.decision['guard_call'] else 0)
        ex.update(os_flags=float(flags), os_force_miss=float(bool(flags)),
                  os_reason=float((flags & -flags).bit_length() if flags else 0))
        self.deployed._s['flag'][-1] = int(bool(flags))
        for name, value in self.decision['scalars'].items():
            ex[f'os_h2_{name}'] = float(value)
        res.extras = ex
        return res

    def blind_step(self, q):
        """Use the cache lifecycle/tail; old no-progress cannot veto H2's tail."""
        return self.deployed.base.blind_step(q)


class R12Guard(R12LookHalf):
    """Explicit off/H2/M2 wrapper, with off directly delegating to R10/R11."""
    family = 'r12_guard'

    def __init__(self, library='current', episode_subset=None, method='off',
                 knob_method='off', target=None):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__(library, episode_subset, method='off', knob_method=knob_method, target=target)
        self.guard_method = method
        self.risk_fit = None
        self._guard_history = []
        self._pending_guard = None

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.risk_fit is not None:
            freeze_fit(self.risk_fit)

    def fit(self, lib, ctx):
        """Require explicit CPU building before this method is served."""
        from exp.offline_search.rounds.r12.guard.build import fit_wrapper
        fit_wrapper(self, ctx)

    def reset(self, episode):
        """Clear per-episode history, acquisition memory and retry state."""
        super().reset(episode)
        self._guard_history = []
        self._pending_guard = None

    def invalidate_anchor(self):
        """Invalidate an uncommitted request without carrying its risk forward."""
        super().invalidate_anchor()
        self._pending_guard = None

    def plan_cameras(self, q, *, look_reason, executed_steps, execution_audit, exec_ok,
                     previous_source=None, previous_ack=None):
        """M2 uses wrist first after every valid full memory, including a call."""
        due = self.periodic_due(q)
        identity = (str(q.episode.uid), int(q.task_id))
        mem = self._memory
        lifecycle = bool(exec_ok and execution_audit and executed_steps == 5)
        if (mem is not None and mem['identity'] != identity) or not lifecycle:
            self._memory = mem = None
        mode = 'full'
        if self.guard_method == 'm2' and int(q.step) > 0 and mem is not None and lifecycle and not due:
            mode = 'wrist_only'
        self.lookhalf_diagnostics = dict(plan='wrist_first' if mode == 'wrist_only' else 'full_planned',
                                        completion_reason='planned', periodic_due=due)
        self.next_camera_mode = mode
        return mode

    def _prepare_guard(self, q):
        step = int(q.step)
        if self._pending_guard is not None and self._pending_guard['step'] == step:
            return self._pending_guard
        b, C, fit = self.inner.base, self.inner.C, self.risk_fit
        rs = np.asarray(q.rs, np.float64)[:8]
        p1 = b.B1T@np.asarray(q.key_v1, np.float32)-b.muB1
        W = self._query_metric('opus_w', int(q.task_id), np.r_[p1, rs.astype(np.float32)], q)
        row = int(W['rows'][0])
        prog, length = float(C.prog[row]), max(int(C.ep_len[row]), 1)
        hist = self._guard_history
        states = [r['rs'] for r in hist[-3:]]+[rs]
        vec = [now[:3]-old[:3] for old, now in zip(states[:-1], states[1:])]
        speed = float(np.mean([np.linalg.norm(v) for v in vec[-2:]])) if vec else np.nan
        osc = sum(float(np.dot(a, c) < 0) for a, c in zip(vec[:-1], vec[1:]))
        mismatch = np.nan
        if hist and step-hist[-1]['step'] == 2:
            prev = hist[-1]
            # Accepted executed heads include policy calls and their actual tail.
            actions = np.asarray(q.hist_a_exec, np.float64)
            a0 = actions[prev['step'], :5, :3].sum(0)
            a1 = actions[prev['step']+1, :5, :3].sum(0)
            x = np.r_[1., a0, a1, prev['rs']]
            mismatch = float(np.linalg.norm(rs[:3]-prev['rs'][:3]-x@fit['mismatch_B']))/fit['mismatch_scale']
        advance = (prog-hist[-1]['served_prog'])*max(length-1, 1) if hist else np.nan
        raw = dict(W_d1rel=W['distance']/fit['distance_median'], step=step,
                   lagrel=(step-prog*max(length-1, 1))/length,
                   speed2=speed, osc=osc, mismatch=mismatch, W_adv=advance)
        X = feature_matrix(raw, fill=fit['coef']['mu'])
        risk = float(risk_score(X, fit['coef'])[0])
        ta, tb = fit['theta_a'], fit['theta_b']
        call = bool(h2_flags(step, advance, risk, ta, tb))
        scalars = dict(risk=risk, theta_a=ta, theta_b=tb, advance=advance if np.isfinite(advance) else 9.,
                       eligible=step >= 4, stall_branch=step >= 4 and advance <= .5 and risk >= ta,
                       risk_branch=step >= 4 and risk >= tb, call=call,
                       distance_median=fit['distance_median'], wrist_top1=self.row_subset[row])
        scalars.update({f'feature_{i}': value for i, value in enumerate(X[0])})
        pending = dict(step=step, rs=rs.copy(), p1=p1, W=W, guard_call=call, scalars=scalars,
                       features=X[0].tolist(), feature_names=list(FEATURES), raw=raw)
        self._pending_guard = pending
        return pending

    def _m2(self, q, pending):
        if pending['guard_call'] or self.periodic_due(q):
            return NeedCamera(reason='h2_call' if pending['guard_call'] else 'knob_call')
        mem = self._memory
        if mem is None:
            return NeedCamera(reason='invalid_memory')
        lag = int(q.step)-mem['step']
        self.lookhalf_diagnostics.update(lag=lag, tau=.10)
        if lag > 8 or f'opus_l{max(2, lag)}' not in self.half_fit['metrics']:
            return NeedCamera(reason='lag')
        W, p1 = pending['W'], pending['p1']
        rs = np.asarray(q.rs, np.float32)[:8]
        name = f'opus_l{max(2, lag)}'
        WL = self._query_metric(name, int(q.task_id), np.r_[mem['p0'], p1, rs], q)
        score = float(np.sqrt(np.mean(((W['raw'][:10, :6]-WL['raw'][:10, :6])/self.inner.base.sig[:6])**2))
                      + np.mean(np.sign(W['raw'][:5, 6]) != np.sign(WL['raw'][:5, 6])))
        self.lookhalf_diagnostics.update(score=score, w_top1=int(self.row_subset[W['rows'][0]]),
                                        wl_top1=int(self.row_subset[WL['rows'][0]]))
        if not np.isfinite(score) or score > .10:
            return NeedCamera(reason='consistency')
        proposal = dict(WL, distance=W['distance'])
        WL['action'], factor = correct(proposal, self.half_fit['heads']['opus_w'][int(q.task_id)],
            self.half_fit['scales']['opus_w'], np.r_[p1, rs], int(q.step), int(q.task_id), self.inner.base.sig_head)
        WL['metric_name'] = name
        self.lookhalf_diagnostics['factor_w'] = factor
        return WL

    @contextmanager
    def _verdict(self, pending):
        self.inner.decision = pending
        try:
            yield
        finally:
            self.inner.decision = None

    def query(self, q):
        """Prepare risk before completion; commit H2/cache/knob once per look."""
        if self.guard_method == 'off':
            return R11Knob.query(self, q)
        step = int(q.step)
        if step == self._lh_step:
            return copy.deepcopy(self._lh_result)
        if step < self._lh_step:
            raise api.ContractError('reset H2 on episode change')
        pending = self._prepare_guard(q)
        self.lookhalf_diagnostics['h2'] = {k: v for k, v in pending.items()
            if k in ('guard_call', 'scalars', 'features', 'feature_names')}
        proposal = None
        with self._verdict(pending):
            if self._camera_mode == 'wrist_only':
                proposal = self._m2(q, pending)
                if isinstance(proposal, NeedCamera):
                    self.lookhalf_diagnostics['completion_reason'] = proposal.reason
                    return proposal
                safe = _View(q, key_v0=self._memory['key0'])
                with self._chosen_base(safe, proposal):
                    res = R11Knob.query(self, safe)
            else:
                res = R11Knob.query(self, q)
        b, C = self.inner.base, self.inner.C
        called = bool(res.extras.get('os_force_miss', 0))
        anchor = b._anchor
        self._guard_history.append(dict(step=step, rs=pending['rs'], served_prog=float(C.prog[anchor['rows'][0]])))
        self._guard_history = self._guard_history[-3:]
        if self._camera_mode == 'full':
            self._memory = dict(step=step, identity=(str(q.episode.uid), int(q.task_id)),
                p0=b.B0T@np.asarray(q.key_v0, np.float32)-b.muB0,
                key0=np.array(q.key_v0, np.float32, copy=True), rs=pending['rs'].copy(),
                rows=anchor['rows'].copy(), weights=anchor['weights'].copy(), called=called)
        self.lookhalf_diagnostics.update(accepted_half=proposal is not None, called=called)
        self._lh_step, self._lh_result = step, copy.deepcopy(res)
        self._pending_guard = None
        return res
