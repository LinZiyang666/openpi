"""Composition-only takeover: frozen r3c judge/corrector are imported unchanged.

The new wrapper never owns a judge field (in particular burst). The task-blind
2b detector sees the stack's proposed corrected chunk; its weights and threshold
are frozen, so transfer to corrected actions is an explicitly untested shift.
The pace12 alternative changes only takeover duration: same lag12/deadline80,
twelve fresh policy calls, one entry, no rearm. Guard calls remain unrestricted.
"""
import hashlib
import importlib
import pickle

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r09.explore_fable.round3.tools import methods as frozen
from exp.offline_search.rounds.r09.explore_opus.round2.methods import _Escalation
from ...round2.inference import predict
from ...round2b.inference import Monitor, LatchedGate


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class Pace12(_Escalation):
    """Task-blind bounded persistence; use the imported pace trigger verbatim."""
    def __init__(self, base):
        self.base = base
        self._esc_init(12, 80)
        self.used = 0

    def observe(self, q, row, force=False):
        if force and self._esc_step is None:
            self._esc_step = int(q.step)
            self._esc_identity = (str(getattr(q.episode, 'uid', q.episode)),)
        before = self._esc_step
        active = self._esc_update(q, row)
        start = active and (before is None or force)
        call = active and self.used < 12
        self.used += int(call)
        return call, start


class StackTakeover(api.Method):
    tier = 'T1'
    family = 'r9_astra_r5_takeover'
    uses_gt = False
    uses_nonlibrary_action = True

    def __init__(self, stack_class, stack_kwargs, head_path, phase_path, threshold,
                 pins, mode='latch', force_at=()):
        if stack_class not in ('NpGraspStack3', 'NpGraspStackGroot3'):
            raise ValueError('only imported no-progress/corrector stacks allowed')
        if mode not in ('latch', 'pace12'):
            raise ValueError('unknown takeover')
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('invalid threshold')
        self.tk_class, self.tk_kwargs = stack_class, dict(stack_kwargs)
        self.tk_head_path, self.tk_phase_path = str(head_path), str(phase_path)
        self.tk_threshold, self.tk_pins = float(threshold), dict(pins)
        self.tk_mode, self.tk_force_at = mode, tuple(map(int, force_at))
        self.name = 'R9_astra_r5_'+mode
        self.tk_stack = None

    def fit(self, lib, ctx):
        for module, want in self.tk_pins['sources'].items():
            if digest(importlib.import_module(module).__file__) != want:
                raise api.ContractError('frozen source changed: '+module)
        for path, want in self.tk_pins['assets'].items():
            if digest(path) != want:
                raise api.ContractError('frozen asset changed: '+path)
        kw = self.tk_kwargs
        if kw.get('max_calls') != 0 or kw.get('force_trigger_at'):
            raise api.ContractError('empty-grasp intervention must be disabled')
        self.tk_stack = getattr(frozen, self.tk_class)(**kw)
        self.tk_stack.fit(lib, ctx)
        s = self.tk_stack
        with open(kw['onlynp_fit'], 'rb') as f:
            src = FitUnpickler(f).load()['method']
        # Compare every inherited fitted field, not merely the famous burst flag.
        compared = []
        for key, value in vars(src).items():
            if key in ('base', 'name', 'fit_info', 'prof'):
                continue
            if pickle.dumps(getattr(s, key), protocol=4) != pickle.dumps(value, protocol=4):
                raise api.ContractError('frozen judge field changed: '+key)
            compared.append(key)
        if set(vars(self)) & set(vars(src)) - {'name', 'prof', 'fit_info'}:
            raise api.ContractError('wrapper shadows a judge field')
        assert type(s.base) is frozen.CorrectedCacheJ
        assert s.base.blend == .5 and not s.base.correct_gripper
        assert s.burst == src.burst == 0 and s.gm_max_calls == 0
        assert tuple(s.disabled_guards) == ('stuck', 'terminal', 'overtime')
        with np.load(self.tk_head_path, allow_pickle=False) as z:
            self.tk_head = {k: z[k] for k in z.files}
        self.tk_phase = np.load(self.tk_phase_path, allow_pickle=False)
        assert self.tk_phase.shape == (len(s.base.act),)
        for a in [self.tk_phase, *self.tk_head.values()]:
            a.flags.writeable = False
        self.tk_identity = None
        self.tk_log = {}
        self.fit_info = dict(fit_inits=list(range(20)), eval_inits=list(range(20,30)),
            reused_frozen_detector=True, new_task_conditioning=False, mode=self.tk_mode,
            unchanged_judge_fields=compared, corrector_path='os_synth', judge_burst=0,
            takeover_cap=12, guard_cap=None, trigger_threshold=self.tk_threshold)

    def reset(self, episode):
        self.tk_stack.reset(episode)
        self.tk_identity = str(episode.uid)
        self.tk_monitor = Monitor()
        self.tk_gate = LatchedGate(self.tk_threshold) if self.tk_mode == 'latch' else Pace12(self.tk_stack.base)
        self.tk_log = {}

    def query(self, q):
        if self.tk_identity != str(q.episode.uid):
            self.reset(q.episode)
        base = self.tk_stack.base
        # _dist is a stateless fitted-metric read. Recover the original detector's
        # d1/median input; do not silently replace it with judge confidence or zero.
        if self.tk_mode == 'latch':
            _, _, _, _, _, xv, _, d, med, _, _ = base._dist(q)
            distance = float(np.min(d)) / med
        res = self.tk_stack.query(q)
        a = base._anchor
        if a is None or a['step'] != int(q.step):
            raise api.ContractError('missing fresh stack anchor')
        force = int(q.step) in self.tk_force_at
        score = -1.
        if self.tk_mode == 'latch':
            x = self.tk_monitor.observe(xv, q.rs, res.action, a['rows'], a['weights'],
                self.tk_phase[a['rows']], int(q.step), distance)
            score = float(predict(self.tk_head, x[None])[0,0])
            if force:  # CPU-only probe; production freezes force_at=[]
                self.tk_gate.high = 1
                self.tk_gate.warmup = 0
            call, start = self.tk_gate.observe(max(score, self.tk_threshold) if force else score, int(q.step))
        else:
            call, start = self.tk_gate.observe(q, res.topk[0], force)
        guard = bool((res.extras or {}).get('os_force_miss'))
        self.tk_log = dict(r9a5_score=score, r9a5_start=float(start), r9a5_call=float(call),
            r9a5_used=float(self.tk_gate.used), r9a5_guard=float(guard), r9a5_added=float(call and not guard))
        ex = dict(res.extras or {}, **self.tk_log)
        if call:
            if not guard:
                ex.update(os_force_miss=1., os_reason=95.,
                          os_flags=float(int(ex.get('os_flags',0)) | (1 << 12)))
            if self.tk_stack._s.get('flag'):
                self.tk_stack._s['flag'][-1] = 1
        return api.Result(res.topk, res.scores, res.confidence, action=res.action, library=res.library, extras=ex)

    def invalidate_anchor(self):
        self.tk_stack.invalidate_anchor()

    def blind_step(self, q):
        return self.tk_stack.blind_step(q)

    def policy_tail_step(self, q):
        return self.tk_stack.policy_tail_step(q)

    @property
    def last_blind_extras(self):
        return dict(self.tk_stack.last_blind_extras, **self.tk_log)

    def bytes_per_entry(self):
        return self.tk_stack.bytes_per_entry()+8
