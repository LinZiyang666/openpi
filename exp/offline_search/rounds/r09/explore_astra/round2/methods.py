"""Frozen task-agnostic residual heads with unchanged ten-control commitment."""
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from .inference import inputs, correct


class SharedResidual(api.Method):
    tier='T1'
    family='r09_astra_shared_residual'
    uses_nonlibrary_action=True

    def __init__(self,base_fit,base_spec,base_kwargs,head_path,history=True,
                 blend=.5,gripper=False,gripper_threshold=.8,wrist=False):
        if not 0 <= blend <= 1 or not 0 < gripper_threshold <= 1:
            raise ValueError('invalid blend or gripper threshold')
        self.base_fit=str(base_fit);self.base_spec=base_spec;self.base_kwargs=dict(base_kwargs)
        self.head_path=str(head_path);self.history=bool(history);self.blend=float(blend)
        self.gripper=bool(gripper);self.gripper_threshold=float(gripper_threshold);self.wrist=bool(wrist)
        if wrist and history: raise ValueError('Frozen wrist head has no history inputs')
        self.base=None;self.head=None;self._previous=None
        self.name='R9R2_shared_wrist' if wrist else 'R9R2_shared'
        if wrist:self.camera_mode='per_request'

    def fit(self,lib,ctx):
        with open(self.base_fit,'rb') as f:blob=FitUnpickler(f).load()
        if blob['cell']!=ctx.cell or blob['kwargs']!=self.base_kwargs or blob['spec']!=self.base_spec:
            raise ValueError('base identity mismatch')
        self.base=blob['method'];self.base.prof=self.prof
        cache=self._cache()
        if (cache.serving,cache.budget,cache.gates)!=('anchor_tail',1,'budget_only'):
            raise ValueError('requires ten-control cache')
        if self.wrist:
            if type(self.base).__name__!='WristEveryLook' or self.base._r8_every_controls!=10:
                raise ValueError('requires frozen wrist-every-ten')
            # Camera shadow and serving must use the same PCA basis.
            np.testing.assert_array_equal(cache.B1T,self.base.wrist.B1T)
            np.testing.assert_array_equal(cache.muB1,self.base.wrist.muB1)
        with np.load(self.head_path,allow_pickle=False) as z:self.head={k:z[k] for k in z.files}
        width=(143 if self.wrist else 207)+(16 if self.history else 0)
        if self.head['mean'].shape!=(width,) or self.head['coef'].shape[1]!=70:
            raise ValueError('head shape mismatch')
        for value in self.head.values():value.flags.writeable=False
        self.fit_info=dict(training_inits=list(range(20)),evaluation_inits=list(range(20,30)),
            task_agnostic_head=True,commit_controls=10,blend=self.blend,gripper=self.gripper,
            gripper_threshold=self.gripper_threshold,history=self.history,wrist=self.wrist)

    def _cache(self):return self.base.base if self.wrist else self.base

    def reset(self,episode):
        self.base.reset(episode);self._previous=None

    def query(self,q):
        res=self.base.query(q);cache=self._cache()
        original=res.action.copy()
        wrist=cache.B1T@np.asarray(q.key_v1,np.float32)-cache.muB1
        visual=wrist if self.wrist else np.r_[cache.B0T@np.asarray(q.key_v0,np.float32)-cache.muB0,wrist]
        x=inputs(visual,q.rs,original,int(q.step),self._previous,self.history)
        self._previous=dict(state=np.asarray(q.rs)[:8].copy(),action=original[:10,:7].copy(),step=int(q.step))
        if self.blend==0 and not self.gripper:return res
        res.action=correct(self.head,x,original[None],self.blend,self.gripper,self.gripper_threshold)[0]
        res.extras=dict(res.extras,r9r2_correction_rms=float(np.sqrt(np.mean((res.action[:10,:6]-original[:10,:6])**2))),
            r9r2_gripper_changes=float(np.sum((res.action[:10,6]>=0)!=(original[:10,6]>=0))))
        a=cache._anchor
        cache._remember_anchor(q,a['rows'],a['weights'],res.action)
        return res

    def blind_step(self,q):return self.base.blind_step(q)
    def invalidate_anchor(self):
        self.base.invalidate_anchor();self._previous=None
    @property
    def last_blind_extras(self):return self.base.last_blind_extras
    @property
    def next_camera_mode(self):return self.base.next_camera_mode if self.wrist else 'full'
    def set_camera_mode(self,mode):
        if self.wrist:self.base.set_camera_mode(mode)
        elif mode!='full':raise ValueError('full head requires both cameras')
    def bytes_per_entry(self):return self.base.bytes_per_entry()
