"""CPU task-periodic reference. The real plugin adapter belongs to sol."""
import math
from exp.offline_search.rounds.r11.knob.recipe import history,uniform
from exp.offline_search.harness import api

class TaskPeriodic:
    def __init__(self,settings):
        self.K=tuple(float(v) for v in settings)
        if len(self.K)!=10 or not all(math.isfinite(k) and k>=0 for k in self.K):
            raise ValueError('need ten finite nonnegative periodic caps')
        self.reset()
    def reset(self):self.last_step=-1;self.last=None
    def propose(self,q,guard):
        step=int(q.step)
        if step==self.last_step:return dict(self.last)
        if step<self.last_step:raise api.ContractError('reset on a new episode')
        task=int(q.task_id)
        if not 0<=task<len(self.K):raise api.ContractError('unknown task')
        anchors,last,run,ledger=history(q)
        K=self.K[task];u=uniform('R11-gap-v1',0,task,int(q.episode.init),last,'cap')
        cap=math.floor(K)+int(K-math.floor(K)>0 and u<K-math.floor(K))
        knob=not guard and run>=cap
        result=dict(miss=bool(guard or knob),guard=bool(guard),knob=bool(knob),K=K,cap=cap,
                    coin=u,run=run,N=ledger.decisions,V=ledger.looks,M=ledger.calls)
        self.last_step=step;self.last=result
        return dict(result)
