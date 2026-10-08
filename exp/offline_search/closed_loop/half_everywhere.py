"""Frozen wrist-sub cache + wrist LOEO head + H2, per task, over R11."""
from __future__ import annotations
import copy
import numpy as np
from exp.offline_search.harness import api, dims
from exp.offline_search.rounds.r10.recipe.recipe import _CorrectedCache, predict
from exp.offline_search.rounds.r11.knob.recipe import R11Knob
from .cache_ablate import _CameraCache
from .guard import R12Guard, _H2Judge
from .lookhalf import R12LookHalf, NeedCamera
from .lookhalf_math import freeze_fit
from .half_everywhere_config import load_table
from exp.offline_search.rounds.r08.abl.judge import TriggerCommitJudge, TriggerGrootCommitJudge


class _WristDiagnostics:
    """Disabled deployed visual diagnostics cannot acquire the absent view."""
    def _self_change(self, q):
        if self.base.camera[str(int(q.task_id))] == 'both':
            return super()._self_change(q)
        if int(q.step) == 0:
            return float('nan'), float('nan')
        now = np.asarray(dims.valid_state(q.rs, self.model), np.float32)
        old = np.asarray(dims.valid_state(q.hist_rs[-1], self.model), np.float32)
        return float(np.linalg.norm(now-old)), float('nan')

    def confirmed_stuck(self, q):
        if self.base.camera[str(int(q.task_id))] == 'both':
            return super().confirmed_stuck(q)
        return 0  # stuck/overtime bits are disabled; H2 supplies the wrist verdict


class _PiJudge(_WristDiagnostics, TriggerCommitJudge):
    pass


class _GrootJudge(_WristDiagnostics, TriggerGrootCommitJudge):
    pass


class _SubCache(_CorrectedCache):
    """Both dispatch is inherited; wrist arithmetic never reads the other key."""
    selected_camera = 1

    def _dist(self, q):
        if self.camera[str(int(q.task_id))] == 'both':
            return super()._dist(q)
        task, step = int(q.task_id), int(q.step)
        T = self.tasks[task]
        key = np.asarray(q.key_v1, np.float32)
        p1 = self.B1T @ key - self.muB1
        rs = np.asarray(dims.valid_state(q.rs, self.model), np.float32)
        d = self.camera_metrics[task].distance(np.r_[p1, rs], initial=step == 0)
        med = float(np.median(d)) + 1e-12
        regime = 0 if step == 0 else (1 if q.prev_hit is False else 2)
        c, dt = None, d
        if regime == 1:
            tail = (np.asarray(q.prev_a_exec[5:10, :7], np.float32)/self.sig).ravel()
            c = np.sqrt(np.maximum(T.h2-2*(T.HD@tail)+float(tail@tail), 0)/35)
            dt = d/med+self.lam_c*c/T.s_c
        return T, step, regime, None, key, p1, rs, d, med, c, dt

    def os_score_all(self, q):
        if self.camera[str(int(q.task_id))] == 'both':
            return super().os_score_all(q)
        T, _, _, _, _, p1, _, d, _, c, dt = self._dist(q)
        ds = dt.astype(np.float64)
        kr = min(self.kref, min(self.k, len(ds)))
        ref = max(float(np.partition(ds, kr-1)[kr-1]-ds.min()), 1e-6)
        pc = p1-T.Vm1
        # The old judge's non-serving diagnostics have no current agentview.
        aux = dict(awm_d=d, vis_v0=np.zeros(len(d), np.float32),
                   vis_v1=T.V1@(pc/max(float(np.sqrt(pc@pc)), 1e-12)))
        if c is not None:
            aux['awm_c'] = c
        return T.rows, -((ds-ds.min())/ref)**2, aux

    def _correction(self, q, action):
        if self.camera[str(int(q.task_id))] == 'both':
            return super()._correction(q, action)
        self.sub_factor = 0.
        if int(q.step) == 0 or not self.corrector_enabled:
            return np.zeros((10, 6), np.float32)
        T, _, _, _, _, p1, rs, d, _, _, _ = self._dist(q)
        pos = np.searchsorted(T.rows, self._anchor['rows'])
        if not np.array_equal(T.rows[pos], self._anchor['rows']):
            raise api.ContractError('wrist corrector anchor outside task')
        ratio = float(np.min(d[pos]))/self.scale_sub[int(q.task_id)]
        factor = float(np.clip((2.-ratio)/1.25, 0., 1.)) if np.isfinite(ratio) else 0.
        self.sub_factor = factor
        if not factor:
            return np.zeros((10, 6), np.float32)
        X = self._features(q, p1, rs, action)
        corr = predict(self.wrist_heads[int(q.task_id)], X)[0].reshape(10, 6)*self.sig_head[:6]
        return corr*np.float32(factor)

    def pure_query(self, q):
        return _CameraCache.query(self, q)


class _TaskJudge(_H2Judge):
    def query(self, q):
        if self.deployed.base.camera[str(int(q.task_id))] == 'both':
            return self.deployed.query(q)
        return super().query(q)

    def blind_step(self, q):
        if self.deployed.base.camera[str(int(q.task_id))] == 'both':
            return self.deployed.blind_step(q)
        return self.deployed.base.blind_step(q)


class R12HalfEverywhere(R11Knob):
    family = 'r12_half_everywhere'
    allows_wrist_after_miss = True
    allows_wrist_every_look = True
    periodic_due = R12LookHalf.periodic_due
    _query_metric = R12LookHalf._query_metric
    _prepare_guard = R12Guard._prepare_guard

    def __init__(self, library='current', episode_subset=None, camera_table=None,
                 knob_method='off', target=None):
        if knob_method not in ('off', 'periodic'):
            raise ValueError('half-everywhere requires off/periodic knob')
        table = load_table(camera_table)
        super().__init__(library, episode_subset, method=knob_method, target=target)
        self.camera_table = dict(camera_table)
        self.camera = dict(table['camera'])
        self.table_cell = table['cell']
        self.next_camera_mode = self._camera_mode = 'full'
        self.half_fit = self.risk_fit = None
        self._guard_history, self._pending_guard = [], None
        self._half_step, self._half_result = -1, None
        self.lookhalf_diagnostics = {}
        self.guard_enabled = True

    def __setstate__(self, state):
        self.__dict__.update(state)
        freeze_fit(self.half_fit)
        freeze_fit(self.risk_fit)
        if self.inner is not None:
            freeze_fit(self.inner.base.wrist_heads)
            for metric in self.inner.base.camera_metrics.values():
                freeze_fit(vars(metric))

    def fit(self, lib, ctx):
        raise api.ContractError('R12HalfEverywhere requires an explicit B-library frozen fit artifact')

    def reset(self, episode):
        super().reset(episode)
        self._guard_history, self._pending_guard = [], None
        self._half_step, self._half_result = -1, None
        self.next_camera_mode = self._camera_mode = 'full'
        self.lookhalf_diagnostics = {}

    def set_camera_mode(self, mode):
        if mode not in ('full', 'wrist_only'):
            raise ValueError(mode)
        self._camera_mode = mode

    def invalidate_anchor(self):
        self.inner.invalidate_anchor()
        self._pending_guard = None

    def on_policy_executed(self):
        self.inner.invalidate_anchor()

    def plan_cameras(self, q, *, look_reason, executed_steps, execution_audit, exec_ok,
                     previous_source=None, previous_ack=None):
        mode = self.camera[str(int(q.task_id))]
        due = self.periodic_due(q)
        planned = 'full' if mode == 'both' or due else 'wrist_only'
        self.next_camera_mode = planned
        self.lookhalf_diagnostics = dict(task_mode=mode, plan=planned, periodic_due=due,
            completion_reason='planned', execution_audit=bool(execution_audit),
            exec_ok=bool(exec_ok), executed_steps=int(executed_steps), look_reason=int(look_reason))
        return planned

    def query(self, q):
        task, step = int(q.task_id), int(q.step)
        if self.camera[str(task)] == 'both':
            res = R11Knob.query(self, q)
            b = self.inner.base
            d = b._dist(q)[7]
            pos = np.searchsorted(b.tasks[task].rows, b._anchor['rows'])
            d_min = float(np.min(d[pos]))
            ratio = d_min/b.distance_scale
            factor = float(np.clip((2.-ratio)/1.25, 0., 1.)) if step and np.isfinite(ratio) else 0.
            self.lookhalf_diagnostics.update(served_top1=int(res.topk[0]),
                d_min=d_min, factor=factor, deployed_span=int(self.inner._noprog_span),
                called=bool(res.extras.get('os_force_miss', 0)))
            return res
        if step == self._half_step:
            return copy.deepcopy(self._half_result)
        if step < self._half_step:
            raise api.ContractError('reset half-everywhere on episode change')
        b = self.inner.base
        if not self.guard_enabled and not b.corrector_enabled and self.knob_method == 'off':
            r = b.pure_query(q)
            res = api.Result(self.row_subset[r.topk], r.scores, r.confidence,
                             action=r.action, library=self.library, extras=r.extras)
            pending = None
        else:
            pending = self._prepare_guard(q)
            self.lookhalf_diagnostics['h2'] = {k: pending[k] for k in ('guard_call', 'scalars', 'features', 'feature_names')}
            if pending['guard_call'] and self.guard_enabled and self._camera_mode == 'wrist_only':
                self.lookhalf_diagnostics['completion_reason'] = 'h2_call'
                return NeedCamera(reason='h2_call')
            verdict = pending if self.guard_enabled else dict(pending, guard_call=False)
            self.inner.decision = verdict
            try:
                res = R11Knob.query(self, q)
            finally:
                self.inner.decision = None
        if pending is not None:
            self._guard_history.append(dict(step=step, rs=pending['rs'],
                served_prog=float(self.inner.C.prog[b._anchor['rows'][0]])))
            self._guard_history = self._guard_history[-3:]
        d = b._dist(q)[7]
        pos = np.searchsorted(b.tasks[task].rows, b._anchor['rows'])
        self.lookhalf_diagnostics.update(served_top1=int(res.topk[0]), d_min=float(np.min(d[pos])),
            factor=float(getattr(b, 'sub_factor', 0.)), called=bool(res.extras.get('os_force_miss', 0)))
        self._half_step, self._half_result = step, copy.deepcopy(res)
        self._pending_guard = None
        return res

    def bytes_per_entry(self):
        return super().bytes_per_entry() + self.half_bytes/len(self.row_subset)
