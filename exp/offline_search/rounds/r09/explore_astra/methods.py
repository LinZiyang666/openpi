"""R9 isolated deployment candidates. Existing serving files remain untouched."""
from pathlib import Path
import json
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r08.methods.methods import AnchorCalls, coin
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM


class ResidualCache(api.Method):
    """Frozen cache + cheap motion correction; retain gripper and 10-control tail."""
    tier='T1'
    family='r09_astra_residual'
    uses_nonlibrary_action=True

    def __init__(self,base_fit,base_kwargs,student_path,blend=.5):
        if not 0 <= blend <= 1:
            raise ValueError('blend must be in [0,1]')
        self.base_fit=str(base_fit);self.base_kwargs=dict(base_kwargs)
        self.student_path=str(student_path);self.blend=float(blend)
        self.base=None;self.models={};self.name=f'R9_residual_{blend:g}'

    def fit(self,lib,ctx):
        with open(self.base_fit,'rb') as f:
            artifact=FitUnpickler(f).load()
        if artifact['kwargs'] != self.base_kwargs or artifact['cell'] != ctx.cell:
            raise ValueError('base fit identity mismatch')
        self.base=artifact['method'];self.base.prof=self.prof
        if (self.base.serving,self.base.budget,self.base.gates)!=('anchor_tail',1,'budget_only'):
            raise ValueError('requires frozen ten-control cache')
        with np.load(self.student_path,allow_pickle=False) as z:
            self.models={t:{k:z[f't{t}_{k}'].copy() for k in ['mean','std','w','bias','coef','intercept']} for t in range(10)}
        for model in self.models.values():
            for x in model.values():x.flags.writeable=False
        self.fit_info=dict(base_fit=self.base_fit,student_path=self.student_path,blend=self.blend,
                           discovery_trained=True,gripper='unchanged',commit_controls=10)

    def reset(self,episode):
        self.base.reset(episode)

    def query(self,q):
        from .inference import predict
        result=self.base.query(q)
        if self.blend==0:
            return result
        visual=np.concatenate([self.base.B0T@np.asarray(q.key_v0,np.float32)-self.base.muB0,
                               self.base.B1T@np.asarray(q.key_v1,np.float32)-self.base.muB1])
        x=np.concatenate([visual,np.asarray(q.rs)[:8],result.action[:10,:7].ravel(),[min(int(q.step),120)/120]])
        correction=predict(self.models[int(q.task_id)],x[None])[0]
        result.action=result.action.copy()
        result.action[:10,:6]+=self.blend*correction
        result.extras=dict(result.extras,r9_correction_rms=float(np.sqrt(np.mean(correction**2))),r9_blend=self.blend)
        # The next blind decision must serve the corrected anchor, not the old tail.
        a=self.base._anchor
        self.base._remember_anchor(q,a['rows'],a['weights'],result.action)
        return result

    def blind_step(self,q):return self.base.blind_step(q)
    def invalidate_anchor(self):return self.base.invalidate_anchor()
    @property
    def last_blind_extras(self):return self.base.last_blind_extras
    def bytes_per_entry(self):return self.base.bytes_per_entry()


class EpisodePolicyLottery(AnchorCalls):
    """Task-only pure-policy probability, drawn once for the WHOLE episode."""
    family='r09_astra_episode_lottery'

    def __init__(self,task_probabilities,**kwargs):
        p=np.asarray(task_probabilities,float)
        if p.shape!=(10,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('ten valid task probabilities required')
        self.task_probabilities=p
        kwargs['p']=1.
        super().__init__(**kwargs)
        self.name='R9_episode_policy_lottery'

    def _assignment(self,q):
        u=coin(self.random_seed,q.task_id,q.episode.init,0,self.coin_domain)
        p=float(self.task_probabilities[int(q.task_id)])
        return p,u,dict(coin_domain=self.coin_domain,random_seed=self.random_seed,
                        episode_lottery=True,task_policy_probability=p)


class TenStepMetricCache(BlindAWM):
    """Refit the library-only Mahalanobis metric against ten-step action labels.

Confidence scales are retained solely for logging; this pure-cache candidate
MUST use judge=always. A call guard would require a new confidence calibration.
"""
    family='r09_astra_metric10'

    def __init__(self,base_fit,base_kwargs):
        super().__init__(**base_kwargs)
        self.base_fit=str(base_fit);self.base_kwargs=dict(base_kwargs)

    def fit(self,lib,ctx):
        from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric
        with open(self.base_fit,'rb') as f:blob=FitUnpickler(f).load()
        if blob['cell']!=ctx.cell or blob['kwargs']!=self.base_kwargs:
            raise ValueError('metric base identity mismatch')
        self.__dict__.update(vars(blob['method']))
        self.name='R9_ten_step_metric'
        for T in self.tasks.values():
            X=(np.asarray(T.Z,np.float64)+T.shift)@np.linalg.inv(np.asarray(T.Wf,np.float64))
            rows=T.rows;H=(self.act[rows,:10,:7]/self.sig).reshape(len(rows),70)
            mu,sd,w=fit_metric(X,H,self.lib_ep[rows],nn=self.nn,lam=self.lam)
            T.Z=np.asarray(((X-mu)/sd)@w,np.float32)
            T.z2=np.asarray((T.Z.astype(float)**2).sum(1),np.float32)
            T.Wf=np.asarray(w/sd[:,None],np.float32);T.shift=np.asarray((mu/sd)@w,np.float32)
            ii=self.lib_step[rows]<=2
            mu,sd,w=fit_metric(X[ii],H[ii],self.lib_ep[rows][ii],nn=self.nn,lam=self.lam)
            T.Z0=np.asarray(((X-mu)/sd)@w,np.float32)
            T.n20=np.asarray((T.Z0.astype(float)**2).sum(1),np.float32)
            T.W0f=np.asarray(w/sd[:,None],np.float32);T.c0=np.asarray((mu/sd)@w,np.float32)
            for value in vars(T).values():
                if isinstance(value,np.ndarray):value.flags.writeable=False
        self.invalidate_anchor()
