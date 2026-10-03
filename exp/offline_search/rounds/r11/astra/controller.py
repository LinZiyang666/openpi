"""Minimal implementation reference for sol, not a serving/plugin adapter.

No task ID is accepted. All fitted parameters are cell-wide; mutable state is
one episode/connection. Ledger inputs count committed five-control requests.
"""
from dataclasses import dataclass
import hashlib
import numpy as np

from exp.offline_search.rounds.r11.astra.signals import VISION


@dataclass(frozen=True)
class Ledger:
    decisions: int
    looks: int
    calls: int


class MissController:
    def __init__(self, *, model, method, dose, target, threshold=None,
                 tie_probability=0., score_reference=None, beta=.5, eta=.2, seed=20261002):
        if method not in ('threshold', 'hybrid', 'adaptive'):
            raise ValueError(method)
        if not 0 <= dose <= 1 or not 0 <= beta <= 1 or not 0 < target <= .5:
            raise ValueError('invalid scalar knob')
        if method == 'adaptive':
            ref = np.sort(np.asarray(score_reference, float))
            if not len(ref) or not np.isfinite(ref).all():
                raise ValueError('adaptive method needs finite B-only score reference')
            self.reference = ref
        else:
            if threshold is None or not np.isfinite(threshold):
                raise ValueError('static method needs a finite threshold')
            self.reference = None
        self.a, self.b = VISION[model], 1-VISION[model]
        self.method, self.dose, self.target = method, dose, target
        self.threshold, self.tie_probability = threshold, tie_probability
        self.beta, self.eta, self.seed = beta, eta, seed
        self.reset('uninitialized')

    def reset(self, episode_uid):
        uid = hashlib.blake2b(f'{self.seed}:{episode_uid}'.encode(), digest_size=8).digest()
        self.rng = np.random.default_rng(int.from_bytes(uid, 'little'))
        self.q = self.dose
        self.previous_ledger = None
        self.last_step = -1
        self.last_proposal = None

    def propose(self, *, step, score, guard, ledger):
        # A retried proposal must not consume a new draw or charge the ledger twice.
        if step == self.last_step:
            return dict(self.last_proposal)
        if step < self.last_step or ledger.decisions != step:
            raise ValueError('reset on new episode; ledger must precede this committed step')
        if not (0 <= ledger.calls <= ledger.looks <= ledger.decisions):
            raise ValueError('invalid committed ledger')
        if self.previous_ledger is not None:
            dN = ledger.decisions-self.previous_ledger.decisions
            dV = ledger.looks-self.previous_ledger.looks
            dM = ledger.calls-self.previous_ledger.calls
            if dN <= 0 or not (0 <= dM <= dV <= dN):
                raise ValueError('ledger counters must advance monotonically')
            if self.method == 'adaptive':
                self.q = float(np.clip(self.q-self.eta*(self.a*dV+self.b*dM-self.target*dN)/self.b,0,1))
        if self.method != 'adaptive' and self.dose in (0., 1.):
            # Global endpoints also apply to finite scores outside B support.
            p = self.dose
        elif not np.isfinite(score):
            # Explicit conservative inference fallback, recorded as an exception.
            p = 1.
        elif self.method == 'adaptive':
            lo=np.searchsorted(self.reference,score,side='left')/len(self.reference)
            hi=np.searchsorted(self.reference,score,side='right')/len(self.reference)
            u=lo+self.rng.random()*(hi-lo)
            p=(1-self.beta)*self.q+self.beta*float(u>1-self.q)
        else:
            high=float(score>self.threshold)+float(score==self.threshold)*self.tie_probability
            p=high if self.method=='threshold' else (1-self.beta)*self.dose+self.beta*high
        knob=bool(self.rng.random()<p)
        result=dict(miss=bool(guard or knob), guard=bool(guard), knob=knob,
                    probability=float(p), dose=float(self.q), nonfinite=not bool(np.isfinite(score)))
        self.previous_ledger=ledger
        self.last_step=step
        self.last_proposal=result
        return dict(result)
