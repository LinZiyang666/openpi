"""Additional serving tests; trusted artifacts generated in this directory only."""
import json
import pickle
from types import SimpleNamespace
import unittest
import numpy as np
from .common import HERE,STORE
from ..methods import EpisodePolicyLottery
from exp.offline_search.harness import api


class Deployment(unittest.TestCase):
    def test_episode_assignment_is_constant_and_extremes_exact(self):
        m=EpisodePolicyLottery(task_probabilities=[0.,1.,.4,.5,.5,.5,.5,.5,.5,.5],random_seed=7,coin_domain='test')
        for task in range(10):
            for init in range(30):
                q=SimpleNamespace(task_id=task,step=0,episode=SimpleNamespace(init=init))
                p,u,_=m._assignment(q)
                for step in [1,2,30,100]:
                    q.step=step;p2,u2,_=m._assignment(q)
                    self.assertEqual((p,u),(p2,u2))
                if task==0:self.assertFalse(u<p)
                if task==1:self.assertTrue(u<p)

    def test_metric_artifact_queries_and_tails(self):
        specs=json.loads((HERE/'confirmation_specs.json').read_text())
        for spec in specs:
            if not spec['name'].endswith('_metric10'):continue
            args=spec['plugin_args'];p=args[args.index('--os-fit-artifact')+1]
            with open(p,'rb') as f:blob=pickle.load(f)
            m=blob['method'];m.prof=api.NULL_PROFILER
            lib=STORE/'library'/f"{spec['model']}_{spec['suite']}"/m.cand_name
            k0=np.load(lib/'key_v0.npy',mmap_mode='r');k1=np.load(lib/'key_v1.npy',mmap_mode='r');rs=np.load(lib/'rs.npy',mmap_mode='r')
            for task in range(10):
                ep=SimpleNamespace(uid=f'local:eval:{task}:0',task_id=task,init=0)
                m.reset(ep);r=int(m.tasks[task].rows[0])
                q=SimpleNamespace(task_id=task,step=0,episode=ep,key_v0=k0[r],key_v1=k1[r],rs=rs[r],prev_hit=None)
                a=m.query(q)
                self.assertTrue(np.isfinite(a.action).all())
                bq=SimpleNamespace(task_id=task,step=1,episode=ep,rs=rs[r],prev_hit=True,blind_age=0,executed_steps=5,hist_rs=np.array([rs[r]]))
                b=m.blind_step(bq)
                np.testing.assert_array_equal(a.action[5:10,:7],b.action[:5,:7])
                # Also exercise the ordinary metric, after the initial early fit.
                q.step=2;q.prev_hit=True;q.hist_key_v0=[k0[r]];q.hist_key_v1=[k1[r]]
                c=m.query(q)
                self.assertTrue(np.isfinite(c.action).all())


if __name__=='__main__':unittest.main()
