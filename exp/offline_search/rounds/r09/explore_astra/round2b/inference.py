"""Small task-blind transition monitor. No predicate, task, or outcome inputs."""
from collections import deque
import numpy as np


class Monitor:
    """Causal trailing windows at fresh looks, also used verbatim offline."""
    def __init__(self):self.history=deque(maxlen=7)

    def observe(self,visual,state,action,rows,weights,phase,step,distance):
        visual=np.asarray(visual).reshape(2,-1).astype(float)
        state=np.asarray(state)[:8].astype(float)
        action=np.asarray(action)[:10,:7].astype(float)
        weights=np.asarray(weights,dtype=float);weights=weights/weights.sum()
        phase=np.asarray(phase,dtype=float)
        meanphase=float(weights@phase)
        entropy=float(-(weights*np.log(np.maximum(weights,1e-12))).sum())
        moment=action.mean(0)
        feat=[min(step,120)/120,meanphase,float(np.sqrt(weights@((phase-meanphase)**2))),
            entropy,float(weights.max()),float(np.nan_to_num(distance,nan=0,posinf=0,neginf=0)),
            *moment,*action.std(0),float(np.mean(action[:,6]>=0)),
            float(np.mean(np.abs(np.diff(action[:,6]>=0)))),
            float(np.linalg.norm(state[6:8])),float(state[2])]
        now=dict(visual=visual,state=state,action=action,rows=np.asarray(rows).copy(),
            weights=weights,phase=meanphase,step=int(step))
        for lag in [1,3,6]:
            if len(self.history)>=lag:
                old=self.history[-lag]
                # Relative visual displacement has no fitted task scale.
                vis=np.sqrt(np.mean((visual-old['visual'])**2,axis=1))/np.maximum(
                    np.sqrt(np.mean(visual**2+old['visual']**2,axis=1)),1e-6)
                delta=state-old['state']
                overlap=float(weights[np.isin(rows,old['rows'])].sum())
                feat.extend([1.,*vis,float(np.linalg.norm(delta[:3])),float(np.linalg.norm(delta[3:6])),
                    float(np.linalg.norm(delta[6:8])),meanphase-old['phase'],overlap,
                    float(np.sqrt(np.mean((action[:,:6]-old['action'][:,:6])**2))),
                    float(np.mean((action[:,6]>=0)!=(old['action'][:,6]>=0))),
                    min(step-old['step'],24)/24])
            else:feat.extend([0.]*11)
        self.history.append(now)
        return np.asarray(feat,np.float32)


def gate_schedule(scores,steps,threshold,burst=3,cooldown=6,max_calls=12,warmup=12):
    """Factual-path accounting only. A call still commits for ten controls.

    Consecutive scores >= threshold at two fresh anchors start one burst. After
    it, wait cooldown fresh anchors; cap total calls. Same state machine online.
    """
    gate=RecoveryGate(threshold,burst,cooldown,max_calls,warmup)
    return np.array([gate.observe(float(s),int(t))[0] for s,t in zip(scores,steps)],bool)


class RecoveryGate:
    def __init__(self,threshold,burst=3,cooldown=6,max_calls=12,warmup=12):
        if burst<1 or cooldown<0 or max_calls<0:raise ValueError('invalid recovery budget')
        self.threshold=float(threshold);self.burst=int(burst);self.cooldown=int(cooldown)
        self.max_calls=int(max_calls);self.warmup=int(warmup)
        self.remaining=0;self.cool=0;self.used=0;self.high=0;self.last=-1

    def observe(self,score,step):
        if step<=self.last:raise ValueError('fresh step must increase')
        self.last=step
        eligible=step>=self.warmup and np.isfinite(score)
        self.high=self.high+1 if eligible and score>=self.threshold else 0
        started=False
        if self.cool>0:self.cool-=1;self.high=0
        elif self.remaining==0 and self.high>=2 and self.used<self.max_calls:
            self.remaining=min(self.burst,self.max_calls-self.used);started=True;self.high=0
        call=self.remaining>0
        if call:
            self.remaining-=1;self.used+=1
            if self.remaining==0:self.cool=self.cooldown
        return call,started


class LatchedGate:
    """Keep policy control until observed recovery, with a strict episode cap.

    Entry as RecoveryGate. Exit after at least 3 calls AND two fresh scores below
    half entry threshold, or after 12 total calls. Ten-control commits unchanged.
    After an early exit, six free anchors cool down before another allowed entry.
    """
    def __init__(self,threshold,min_calls=3,max_calls=12,cooldown=6,warmup=12):
        self.threshold=float(threshold);self.min_calls=min_calls;self.max_calls=max_calls
        self.cooldown=cooldown;self.warmup=warmup
        self.active=False;self.high=0;self.low=0;self.used=0;self.run=0;self.cool=0;self.last=-1

    def observe(self,score,step):
        if step<=self.last:raise ValueError('fresh step must increase')
        self.last=step;finite=np.isfinite(score)
        self.high=self.high+1 if finite and step>=self.warmup and score>=self.threshold else 0
        self.low=self.low+1 if finite and score<.5*self.threshold else 0
        started=False
        if self.active and ((self.run>=self.min_calls and self.low>=2) or self.used>=self.max_calls):
            self.active=False;self.cool=self.cooldown;self.high=0
        if not self.active:
            if self.cool>0:self.cool-=1;self.high=0
            elif self.high>=2 and self.used<self.max_calls:
                self.active=True;self.run=0;self.low=0;self.high=0;started=True
        call=self.active
        if call:self.used+=1;self.run+=1
        return call,started


class RandomBurstGate:
    """Matched-budget placement control: a fresh independent coin starts a burst."""
    def __init__(self,probability,burst=3,max_calls=12,cooldown=6,warmup=12):
        self.probability=float(probability);self.burst=burst;self.max_calls=max_calls
        self.cooldown=cooldown;self.warmup=warmup
        self.remaining=0;self.used=0;self.cool=0;self.last=-1

    def observe(self,coin,step):
        if step<=self.last:raise ValueError('fresh step must increase')
        self.last=step;started=False
        if self.cool>0:self.cool-=1
        elif self.remaining==0 and step>=self.warmup and coin<self.probability and self.used<self.max_calls:
            self.remaining=min(self.burst,self.max_calls-self.used);started=True
        call=self.remaining>0
        if call:
            self.remaining-=1;self.used+=1
            if self.remaining==0:self.cool=self.cooldown
        return call,started
