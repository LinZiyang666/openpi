"""Observation-triggered, bounded policy recovery with exact ten-control tails."""
from types import SimpleNamespace
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.methods import uniform
from ..round2.inference import predict
from .inference import Monitor,RecoveryGate,LatchedGate,RandomBurstGate


class TransitionRecovery(api.Method):
    tier='T1';family='r09_astra_transition_recovery';uses_gt=False;uses_nonlibrary_action=False

    def __init__(self,base_fit,base_spec,base_kwargs,head_path,phase_path,threshold,
                 response='burst3',random_probability=0.,random_seed=20261002):
        if response not in ['burst1','burst3','latch','random3']:raise ValueError('unknown response')
        if not 0<=random_probability<=1 or not np.isfinite(threshold):raise ValueError('invalid gate parameter')
        self.base_fit=str(base_fit);self.base_spec=base_spec;self.base_kwargs=dict(base_kwargs)
        self.head_path=str(head_path);self.phase_path=str(phase_path);self.threshold=float(threshold)
        self.response=response;self.random_probability=float(random_probability);self.random_seed=int(random_seed)
        self.name=f'R9R2b_transition_{response}';self.base=None

    def fit(self,lib,ctx):
        with open(self.base_fit,'rb') as f:blob=FitUnpickler(f).load()
        if blob['cell']!=ctx.cell or blob['kwargs']!=self.base_kwargs or blob['spec']!=self.base_spec:
            raise ValueError('base identity mismatch')
        self.base=blob['method'];self.base.prof=self.prof
        if (self.base.serving,self.base.budget,self.base.gates)!=('anchor_tail',1,'budget_only'):
            raise ValueError('ten-control cache required')
        with np.load(self.head_path,allow_pickle=False) as z:self.head={k:z[k] for k in z.files}
        self.phase=np.load(self.phase_path,allow_pickle=False)
        if self.phase.shape!=(len(self.base.act),):raise ValueError('library phase mismatch')
        for a in [self.phase,*self.head.values()]:a.flags.writeable=False
        self.fit_info=dict(fit_inits=list(range(20)),eval_inits=list(range(20,30)),
            task_agnostic=True,predicate_inputs=False,response=self.response,threshold=self.threshold,
            commit_controls=10,max_policy_calls=12)

    def reset(self,episode):
        self.base.reset(episode);self.monitor=Monitor();self._policy_gate_anchor=None;self._log={}
        self._identity=(str(episode.uid),int(episode.task_id),int(episode.init))
        self.gate=(LatchedGate(self.threshold) if self.response=='latch' else
            RandomBurstGate(self.random_probability) if self.response=='random3' else
            RecoveryGate(self.threshold,burst=1 if self.response=='burst1' else 3))

    def query(self,q):
        identity=(str(q.episode.uid),int(q.task_id),int(q.episode.init))
        if getattr(self,'_identity',None)!=identity:self.reset(q.episode)
        res=self.base.query(q);a=self.base._anchor
        visual=np.r_[self.base.B0T@np.asarray(q.key_v0,np.float32)-self.base.muB0,
                     self.base.B1T@np.asarray(q.key_v1,np.float32)-self.base.muB1]
        x=self.monitor.observe(visual,q.rs,res.action,a['rows'],a['weights'],
            self.phase[a['rows']],int(q.step),res.extras.get('d1_rel',0))
        score=float(predict(self.head,x[None])[0,0])
        coin=uniform('R9R2b-transition-control',self.random_seed,q.task_id,q.episode.init,q.step,'burst-start')
        call,start=self.gate.observe(coin if self.response=='random3' else score,int(q.step))
        self._policy_gate_anchor=dict(step=a['step'],episode=a['episode'],task=a['task'],
            rows=a['rows'].copy(),weights=a['weights'].copy()) if call else None
        self._log=dict(r9b_score=score,r9b_threshold=self.threshold,r9b_call=float(call),
            r9b_start=float(start),r9b_used=float(self.gate.used),r9b_coin=float(coin))
        res.extras={**res.extras,**self._log,'os_force_miss':float(call),'os_reason':92. if call else 0.}
        return res

    def invalidate_anchor(self):
        # Plugin invokes this after every policy call. Preserve observation
        # history/recovery latch and the separately captured policy-tail gate.
        self.base.invalidate_anchor()

    @property
    def last_blind_extras(self):return {**self.base.last_blind_extras,**self._log}
    def blind_step(self,q):return self.base.blind_step(q)
    def policy_tail_step(self,q):
        facade=SimpleNamespace(policy_tail_gate='lifecycle',monitor='off',base=self.base,
            _policy_gate_anchor=self._policy_gate_anchor)
        try:return CommitJudge.policy_tail_step(facade,q)
        finally:self._policy_gate_anchor=facade._policy_gate_anchor
    def bytes_per_entry(self):return self.base.bytes_per_entry()+8
