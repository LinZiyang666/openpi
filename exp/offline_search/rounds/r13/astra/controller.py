"""Reference distance-threshold controller; no task-specific fitted state.

Distance stays monotone at any fixed ledger. Threshold is affine in actual
committed excess cost. Out-of-range thresholds expose controller saturation.
"""
from dataclasses import dataclass
import numpy as np

COSTS = {'pi05': (.152, .848), 'groot': (.148, .852)}

@dataclass(frozen=True)
class Ledger:
    decisions: int
    looks: int
    calls: int

def score(distance, lo, hi):
    if not hi > lo:
        raise ValueError('log-distance scale must be nondegenerate')
    return np.clip((np.log1p(distance)-lo)/(hi-lo), 0., 1.)

class DistanceFeedback:
    def __init__(self, *, model, target, lo, hi, width, bias):
        if not 0 < target <= .5 or width<=0 or hi<=lo:
            raise ValueError('invalid controller scalars')
        self.cv,self.cm=COSTS[model]
        self.target,self.lo,self.hi,self.width,self.bias=target,lo,hi,width,bias
        self.reset()

    def reset(self):
        self.last_step=-1; self.last_proposal=None; self.previous=Ledger(0,0,0)

    def propose(self, *, step, distance, guard, ledger):
        if step==self.last_step:
            return dict(self.last_proposal)
        if step<self.last_step or ledger.decisions!=step:
            raise ValueError('accepted ledger must precede current proposal; reset each episode')
        if not (0<=ledger.calls<=ledger.looks<=ledger.decisions):
            raise ValueError('invalid ledger')
        dn=ledger.decisions-self.previous.decisions
        dv=ledger.looks-self.previous.looks; dm=ledger.calls-self.previous.calls
        if not 0<=dm<=dv<=dn:
            raise ValueError('ledger counters must advance monotonically')
        debt=(self.cv*ledger.looks+self.cm*ledger.calls-self.target*ledger.decisions)/self.cm
        threshold=self.bias+debt/self.width
        finite=bool(np.isfinite(distance) and distance>=0)
        u=float(score(distance,self.lo,self.hi)) if finite else float('nan')
        trigger=not finite or u>threshold
        out=dict(miss=bool(guard or trigger),guard=bool(guard),knob=bool(trigger and not guard),
                 score=u,threshold=threshold,debt=debt,nonfinite=not finite,
                 saturated_low=threshold<0,saturated_high=threshold>=1)
        self.last_step,self.last_proposal,self.previous=step,out,ledger
        return dict(out)

def replay(pack, *, model, target, lo, hi, width, bias, kind='feedback', constant=False, shuffle=False):
    """Fixed-state replay: never a counterfactual closed-loop rollout."""
    cv,cm=COSTS[model]; H,E=pack['distance'].shape
    us=score(pack['distance'],lo,hi)
    if constant: us=np.full_like(us,.5)
    if shuffle:
        us=us.copy(); rng=np.random.default_rng(20261004)
        for j in range(E):
            jj=np.flatnonzero(pack['alive'][:,j]); us[jj,j]=rng.permutation(us[jj,j])
    V=np.zeros(E); M=np.zeros(E); run=np.zeros(E,int); maxrun=np.zeros(E,int)
    eligible=interior=knobs=0; capture=0.; calls=np.zeros((H,E),bool); thresholds=np.full((H,E),np.nan)
    for i in range(H):
        active=pack['alive'][i]; g=pack['guard'][i]&active
        debt=(cv*V+cm*M-target*pack['step'][i])/cm
        threshold=bias+debt/width if kind=='feedback' else np.full(E,bias)
        trigger=(us[i]>threshold)&active
        miss=g|trigger
        ok=active&~g
        eligible+=int(ok.sum()); interior+=int((ok&(threshold>=0)&(threshold<1)).sum())
        knobs+=int((ok&trigger).sum())
        V+=active; M+=miss
        run=np.where(active,np.where(miss,0,run+1),run); maxrun=np.maximum(maxrun,run)
        calls[i]=miss; thresholds[i,active]=threshold[active]
        if 'risk' in pack:
            capture+=float(np.nansum(np.where(ok&trigger,pack['risk'][i],0)))
    N=np.asarray(pack['N']); cost=cv*V+cm*M
    return dict(ir=float(cost.sum()/N.sum()),ep_ir=cost/N,N=N,V=V,M=M,calls=calls,thresholds=thresholds,
        interior=interior/max(eligible,1),eligible=eligible,knobs=knobs,maxrun=maxrun,capture=capture,
        terminal_debt=(cost-target*N)/cm)

def fit_bias(pack, *, model, target, lo, hi, width, kind='feedback'):
    left,right=-16.,16.
    cache=[]
    for b in (left,right):
        r=replay(pack,model=model,target=target,lo=lo,hi=hi,width=width,bias=b,kind=kind)
        cache.append((abs(r['ir']-target),b,r['ir']))
    for _ in range(35):
        b=(left+right)/2
        r=replay(pack,model=model,target=target,lo=lo,hi=hi,width=width,bias=b,kind=kind)
        cache.append((abs(r['ir']-target),b,r['ir']))
        if r['ir']>target: left=b
        else: right=b
    _,b,ir=min(cache,key=lambda x:(x[0],abs(x[1]),x[1]))
    return dict(bias=b,fit_ir=ir,fit_error=ir-target)

def fit_bias_minimax(packs, *, model, target, lo, hi, width):
    """One bias shared across B calibration behavior paths, minimizing worst error."""
    left,right=-16.,16.; candidates=[]
    for _ in range(37):
        b=(left+right)/2
        values=[replay(p,model=model,target=target,lo=lo,hi=hi,width=width,bias=b)['ir'] for p in packs]
        err=max(abs(x-target) for x in values)
        candidates.append((err,abs(b),b,values))
        if (max(values)+min(values))/2>target: left=b
        else: right=b
    err,_,b,values=min(candidates)
    return dict(bias=b,calibration_irs=values,max_calibration_error=err)
