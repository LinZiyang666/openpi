from common import *
fence()
from controller import TaskPeriodic
from exp.offline_search.rounds.r11.knob.recipe import R11Knob
from exp.offline_search.rounds.r10.recipe.recipe import R10Recipe
from exp.offline_search.harness import api
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np

rng=np.random.default_rng(20261004);count=retries=0
def base(self,q):return api.Result(np.arange(16),np.zeros(16),1.,action=np.zeros((10,32),np.float32),
                                  extras={'os_force_miss':float(q.fakeguard),'os_reason':4. if q.fakeguard else 0.})
with patch.object(R10Recipe,'query',base):
    for rep in range(4):
        for task in range(10):
            for K in (0.,.3,1.,1.7,40.):
                b=R11Knob(method='periodic',target=.32);b.inner=NS(reset=lambda e:None)
                b.configure(dict(setting=K),'pi05');qep=NS(init=rep,task_id=task,uid=f'{rep}:{task}')
                b.reset(qep);ctl=TaskPeriodic([K]*10);hv=[];hit=[]
                for step in range(60):
                    vision=step%2==0 or rng.random()<.1
                    if not vision:hv.append(False);hit.append(1);continue
                    q=NS(step=step,task_id=task,episode=qep,hist_has_vision=np.array(hv,bool),hist_hit=np.array(hit),fakeguard=rng.random()<.15)
                    r=b.query(q);s=ctl.propose(q,q.fakeguard)
                    assert s['miss']==bool(r.extras['os_force_miss'])
                    assert s['knob']==bool(r.extras['os_r11_knob_call']) and s['cap']==r.extras['os_r11_cap']
                    assert s==ctl.propose(q,q.fakeguard);retries+=1;count+=1
                    hv.append(True);hit.append(int(not s['miss']))
                ctl.reset();assert ctl.last_step==-1
bad=NS(step=2,hist_has_vision=[],hist_hit=[],task_id=0,episode=NS(init=0))
try:TaskPeriodic([1]*10).propose(bad,False)
except api.ContractError:pass
else:raise AssertionError('bad history was accepted')
dump(HERE/'controller_check.json',dict(timestamp=now(),reference_proposals=count,idempotent_retries=retries,
    invalid_history_refused=True,includes_extra_fresh_looks=True,zero_and_large_caps=True))
provenance('controller_check');print(count,retries)
