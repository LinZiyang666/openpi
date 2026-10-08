"""Opt-in R12 acquisition transaction around the unchanged R10/R11 stack."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import math
import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r11.knob.recipe import R11Knob, history, uniform
from .lookhalf_math import atlas_plan, retrieve, correct, freeze_fit

METHODS = ('off', 'opus_m1', 'astra_consensus2', 'astra_strict')


@dataclass(frozen=True)
class NeedCamera:
    camera: str = 'base'
    reason: str = 'consistency'


class _View:
    def __init__(self, q, **fields):
        self.q, self.fields = q, fields
    def __getattr__(self, name):
        return self.fields[name] if name in self.fields else getattr(self.q, name)


class R12LookHalf(R11Knob):
    family = 'r12_lookhalf'
    allows_wrist_after_miss = True

    def __init__(self, library='current', episode_subset=None, method='off',
                 knob_method='off', target=None):
        if method not in METHODS or knob_method not in ('off', 'periodic'):
            raise ValueError('R12 supports off/M1/consensus2/strict with off or periodic knob')
        super().__init__(library, episode_subset, method=knob_method, target=target)
        self.lookhalf_method = method
        self.half_fit = None
        self.next_camera_mode = 'full'
        self._camera_mode = 'full'
        self._memory = None
        self._previous_look_called = False
        self._lh_step, self._lh_result = -1, None
        self.lookhalf_diagnostics = {}
        self._execution = {}

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.half_fit is not None:
            freeze_fit(self.half_fit)

    def fit(self, lib, ctx):
        from exp.offline_search.rounds.r12.lookhalf.build import fit_wrapper
        fit_wrapper(self, ctx)

    def reset(self, episode):
        super().reset(episode)
        self._memory = None
        self._previous_look_called = False
        self._lh_step, self._lh_result = -1, None
        self.next_camera_mode = self._camera_mode = 'full'
        self.lookhalf_diagnostics = {}
        self._execution = {}

    def invalidate_anchor(self):
        self.inner.invalidate_anchor()
        self._memory = None
        self.next_camera_mode = 'full'
        self._lh_step, self._lh_result = -1, None

    def set_camera_mode(self, mode):
        if mode not in ('full', 'wrist_only'):
            raise ValueError(mode)
        self._camera_mode = mode

    def on_policy_executed(self):
        # Retain genuine full policy memory, invalidating only the cache tail.
        self.inner.invalidate_anchor()

    def periodic_due(self, q):
        if self.knob_method == 'off':
            return False
        _, last, run, _ = history(q)
        setting = self.knob_settings['setting']
        u = uniform('R11-gap-v1', 0, int(q.task_id), int(q.episode.init), last, 'cap')
        cap = math.floor(setting)+int(u < setting%1)
        return run >= cap

    def plan_cameras(self, q, *, look_reason, executed_steps, execution_audit, exec_ok,
                     previous_source=None, previous_ack=None):
        """Called after CPU state preparation, before any image encoding."""
        due = self.periodic_due(q)
        identity = (str(q.episode.uid), int(q.task_id))
        mem = self._memory
        if mem is not None and mem['identity'] != identity:
            self._memory = mem = None
        step = int(q.step)
        # Failed or ambiguous execution must not license a stale camera.
        lifecycle = bool(exec_ok and execution_audit and executed_steps == 5)
        if not lifecycle:
            self._memory = mem = None
        diag = dict(plan='full_planned', completion_reason='planned', periodic_due=due)
        mode = 'full'
        if self.lookhalf_method == 'opus_m1':
            if step > 0 and mem is not None and lifecycle and not due and not self._previous_look_called:
                mode = 'wrist_only'
        elif self.lookhalf_method.startswith('astra'):
            tail = bool(mem is not None and mem.get('called') and step == mem['step']+2
                        and previous_source == 'policy_tail' and previous_ack is True)
            legal = lifecycle and (look_reason not in (6, 7, 8) or (look_reason == 6 and tail))
            diag.update(atlas_plan(step=step, memory=mem, rs=np.asarray(q.rs, np.float32)[:8],
                                   table=self.half_fit['atlas'], lifecycle=legal, due=due,
                                   span=self.inner._noprog_span))
            if diag['half']:
                mode = 'wrist_only'
        diag['plan'] = 'wrist_first' if mode == 'wrist_only' else 'full_planned'
        self.lookhalf_diagnostics = diag
        self.next_camera_mode = mode
        self._execution[step] = lifecycle
        return mode

    def _query_metric(self, name, task, x, q):
        b = self.inner.base
        m, rows = self.half_fit['metrics'][name][task]
        d = m.dist(np.atleast_2d(x), initial=int(q.step) == 0)[0]
        # M1's W/WL use the unchanged AWM post-MISS continuity regime.
        if name.startswith('opus') and int(q.step) and q.prev_hit is False:
            T = b.tasks[task]
            tail = (np.asarray(q.prev_a_exec[5:10, :7], np.float32)/b.sig).ravel()
            c = np.sqrt(np.maximum(T.h2-2*(T.HD@tail)+float(tail@tail), 0)/35)
            d = d/(float(np.median(d))+1e-12)+b.lam_c*c/T.s_c
        result = retrieve(m, rows, x, b.act, b.kref, distances=d)
        # Match the explorer's witness arithmetic on the committed controls;
        # the authoritative H payload still retains its complete cache tail.
        if name in ('astra_w', 'astra_f'):
            result['raw'] = np.einsum('k,ktc->tc', result['weights'],
                b.act[result['rows'], :10, :7], optimize=False)
        elif name == 'astra_h':
            result['raw'][:10, :7] = np.tensordot(result['weights'],
                b.act[result['rows'], :10, :7], 1)
        return result

    def _opus(self, q, p1, rs):
        task = int(q.task_id)
        mem = self._memory
        lag = int(q.step)-mem['step']
        diag = self.lookhalf_diagnostics
        diag.update(lag=lag, tau=.10, margin=1.)
        if lag > 8:
            return NeedCamera(reason='lag')
        # Frozen lag metrics are defined only at 2,4,6,8 slots. An unusual
        # acquisition cadence completes rather than silently inventing a fit.
        L = max(2, lag)
        if f'opus_l{L}' not in self.half_fit['metrics']:
            return NeedCamera(reason='lag')
        W = self._query_metric('opus_w', task, np.r_[p1, rs], q)
        WL = self._query_metric(f'opus_l{L}', task, np.r_[mem['p0'], p1, rs], q)
        C = self.inner.C
        prev = self.inner._vision_progress[-1][1] if self.inner._vision_progress else None
        advance = lambda r: (float(C.prog[r])-prev)*max(int(C.ep_len[r])-1, 1) if prev is not None else math.inf
        aw, al = advance(W['rows'][0]), advance(WL['rows'][0])
        diag.update(w_top1=int(self.row_subset[W['rows'][0]]), wl_top1=int(self.row_subset[WL['rows'][0]]),
                    advance_w=aw, advance_wl=al, d_w_min=W['distance'])
        if min(aw, al) <= 1.5:
            return NeedCamera(reason='suspicion')
        a, c = W['raw'], WL['raw']
        score = float(np.sqrt(np.mean(((a[:10, :6]-c[:10, :6])/self.inner.base.sig[:6])**2))
                      + np.mean(np.sign(a[:5, 6]) != np.sign(c[:5, 6])))
        diag['score'] = score
        if score > .10:
            return NeedCamera(reason='consistency')
        # Head consumes the authoritative WL synthesis, attenuation uses W.
        headproposal = dict(WL, distance=W['distance'])
        WL['action'], diag['factor_w'] = correct(headproposal, self.half_fit['heads']['opus_w'][task],
            self.half_fit['scales']['opus_w'], np.r_[p1, rs], int(q.step), task, self.inner.base.sig_head)
        WL['metric_name'] = f'opus_l{L}'
        return WL

    def _astra(self, q, p1, rs):
        task, step, mem = int(q.task_id), int(q.step), self._memory
        X = np.r_[mem['p0'], p1, rs, rs-mem['rs']].astype(np.float32)
        H = self._query_metric('astra_h', task, X, q)
        sig = self.half_fit['sigma']
        H['action'], factor = correct(H, self.half_fit['heads']['astra_h'][task],
                                     self.half_fit['scales']['astra_h'], X, step, task, sig)
        W = self._query_metric('astra_w', task, X[64:], q)
        wa, _ = correct(W, self.half_fit['heads']['astra_w'][task],
                        self.half_fit['scales']['astra_w'], X[64:], step, task, sig)
        P, past = self.half_fit['P'], self.half_fit['past2']
        delta = np.einsum('k,kd->d', H['weights'], P[H['rows'], :64]-P[past[H['rows']], :64])
        XF = np.r_[mem['p0']+delta, p1, rs].astype(np.float32)
        F = self._query_metric('astra_f', task, XF, q)
        fa, _ = correct(F, self.inner.base.heads[str(task)], self.inner.base.distance_scale,
                        XF, step, task, self.inner.base.sig_head)
        action = H['action']
        rms = [float(np.sqrt(np.mean(((a[:10, :6]-action[:10, :6])/sig[:6])**2))) for a in (wa, fa)]
        grip = [bool(np.any((a[:10, 6]>=0) != (action[:10, 6]>=0))) for a in (wa, fa)]
        C = self.inner.C
        h, witnesses = H['rows'][0], [W['rows'][0], F['rows'][0]]
        pp = [abs(float(C.prog[h])-float(C.prog[r]))*max(int(C.ep_len[h])-1, int(C.ep_len[r])-1, 1) for r in witnesses]
        advance = (float(C.prog[h])-float(C.prog[mem['rows'][0]]))*max(int(C.ep_len[h])-1, 1)
        tau = .05 if self.lookhalf_method == 'astra_strict' else .10
        self.lookhalf_diagnostics.update(lag=step-mem['step'], tau=tau, witness_rms=rms,
            witness_grip=grip, witness_progress=pp, advance_h=advance, factor_h=factor,
            h_top1=int(self.row_subset[h]), witness_top1=[int(self.row_subset[r]) for r in witnesses])
        if max(rms) > tau or any(grip) or max(pp) > .5 or advance <= .5:
            return NeedCamera(reason='consensus')
        H['metric_name'] = 'astra_h'
        return H

    @contextmanager
    def _chosen_base(self, q, proposal):
        """One chosen synthesis enters the unchanged guard/knob exactly once."""
        b = self.inner.base
        keys = ('os_score_all', 'os_synth')
        originals = {k: vars(b).get(k) for k in keys}
        score = b.os_score_all
        task_rows = self.half_fit['metrics'][proposal['metric_name']][int(q.task_id)][1]
        def scores(view):
            _, _, aux = score(view)  # remembered-view diagnostics, never hidden current pixels
            return task_rows, proposal['all_scores'], dict(aux, awm_d=proposal['all_distances'])
        def synth(view, rows, weights):
            if not np.array_equal(rows, proposal['rows']):
                raise api.ContractError('R12 requires unchanged no-recovery donor selection')
            a = proposal['action'].copy()
            b._remember_anchor(view, rows, proposal['weights'], a)
            return a
        b.os_score_all, b.os_synth = scores, synth
        try:
            yield
        finally:
            for k, value in originals.items():
                if value is None:
                    delattr(b, k)
                else:
                    setattr(b, k, value)

    def query(self, q):
        if self.lookhalf_method == 'off':
            return super().query(q)
        if int(q.step) == self._lh_step:
            return copy.deepcopy(self._lh_result)
        if int(q.step) < self._lh_step:
            raise api.ContractError('reset look-half on episode change')
        b = self.inner.base
        rs = np.asarray(q.rs, np.float32)[:8]
        proposal = None
        if self._camera_mode == 'wrist_only':
            if self._memory is None:
                return NeedCamera(reason='invalid_memory')
            p1 = b.B1T@np.asarray(q.key_v1, np.float32)-b.muB1
            proposal = self._opus(q, p1, rs) if self.lookhalf_method == 'opus_m1' else self._astra(q, p1, rs)
            if isinstance(proposal, NeedCamera):
                self.lookhalf_diagnostics['completion_reason'] = proposal.reason
                return proposal
            safeq = _View(q, key_v0=self._memory['key0'])
            with self._chosen_base(safeq, proposal):
                res = super().query(safeq)
        else:
            res = super().query(q)
        called = bool(res.extras.get('os_force_miss', 0))
        self._previous_look_called = called
        if self._camera_mode == 'full':
            a = b._anchor
            self._memory = dict(step=int(q.step), identity=(str(q.episode.uid), int(q.task_id)),
                p0=b.B0T@np.asarray(q.key_v0, np.float32)-b.muB0,
                p1=b.B1T@np.asarray(q.key_v1, np.float32)-b.muB1,
                key0=np.array(q.key_v0, np.float32, copy=True), rs=rs.copy(),
                rows=a['rows'].copy(), weights=a['weights'].copy(), called=called)
        elif called:
            # H donors cannot be installed as genuinely full donors.
            self._memory = None
        self.lookhalf_diagnostics.update(accepted_half=proposal is not None, called=called)
        self._lh_step, self._lh_result = int(q.step), copy.deepcopy(res)
        return res

    def bytes_per_entry(self):
        # Frozen fitted arrays are shared between independent connection clones.
        return super().bytes_per_entry() + (0 if self.half_fit is None else self.half_fit.get('bytes', 0)/len(self.row_subset))
