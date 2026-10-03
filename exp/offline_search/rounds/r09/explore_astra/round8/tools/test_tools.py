"""Safety and statistical semantics checks; synthetic fixtures never leave round8."""
import io
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from . import safe, analyze
from .corrector import fit_cuts, gate_metrics


class AdmissionTests(unittest.TestCase):
    def test_nested_identity_is_not_admission(self):
        raw='{"payload":{"init":0,"task_id":0},"task_uid":"x:eval:0:35"}'
        self.assertEqual(safe.json_identity(raw),(0,35))

    def test_escaped_payload_identity(self):
        raw=json.dumps(dict(note='"uid":"x:eval:0:0"',uid='x:eval:2:28',payload=[{},'\\']))
        self.assertEqual(safe.json_identity(raw),(2,28))

    def test_conflicting_identities_rejected(self):
        with self.assertRaises(ValueError):safe.json_identity('{"init":20,"task_id":0,"uid":"x:eval:0:31"}')

    def test_duplicate_identity_rejected(self):
        with self.assertRaises(ValueError):safe.json_identity('{"uid":"x:eval:0:20","uid":"x:eval:0:20"}')

    def test_integer_identity_required(self):
        for value in ('20.5','true','"20"'):
            with self.assertRaises(ValueError):safe.json_identity('{"task_id":0,"init":'+value+'}')

    def test_forbidden_payload_not_decoded(self):
        # This is deliberately invalid JSON. Only the identity is parsed.
        raw='{"uid":"x:eval:0:35","payload":THIS_MUST_NOT_BE_DESERIALIZED}\n'
        raw+='{"uid":"x:eval:0:20","payload":7}\n'
        with patch.object(safe.Path,'open',return_value=io.StringIO(raw)):
            self.assertEqual(list(safe.admitted_jsonl(safe.HERE/'synthetic.jsonl')),[dict(uid='x:eval:0:20',payload=7)])

    def test_requested_forbidden_init_rejected_before_open(self):
        with patch.object(safe.Path,'open',side_effect=AssertionError('opened')):
            with self.assertRaises(ValueError):list(safe.admitted_jsonl(safe.HERE/'x',inits=[30]))

    def test_forbidden_prefixes_and_owned_root(self):
        for name in ('r09_holdout','r09_holdout_extra','r09_astra_holdout2'):
            with self.assertRaises(ValueError):safe.admitted('/home/weiland/trace_runs/os_closed_loop/'+name+'/x')
        with self.assertRaises(ValueError):safe.owned(safe.HERE.parent/'must_not_write')

    def test_compact_payload_guard(self):
        class Archive:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def __getitem__(self,k):
                if k=='task':return np.array([0,0])
                if k=='init':return np.array([20,30])
                raise AssertionError('forbidden payload touched')
        with patch.object(analyze.np,'load',return_value=Archive()):
            with self.assertRaises(ValueError):analyze.compact('groot','P10',['served_chunk'])


class SemanticsTests(unittest.TestCase):
    def test_model_gripper_conventions(self):
        np.testing.assert_array_equal(analyze.is_closed([-1,1],'groot'),[True,False])
        np.testing.assert_array_equal(analyze.is_closed([-1,1],'pi05'),[False,True])

    def test_gripper_phases(self):
        self.assertEqual(analyze.phases([False,False,True,True,True,False,False,False]).tolist(),
            ['approach','grasp','grasp','carry','release','release','release','post'])
        self.assertEqual(analyze.phases([False]*4).tolist(),['approach']*4)

    def test_fit_cut_excludes_eval(self):
        d=pd.DataFrame(dict(init=[0,19],correction=[1.,2.],distance=[4.,8.]))
        self.assertAlmostEqual(fit_cuts(d)['correction'],1.9)
        d.loc[1,'init']=20
        with self.assertRaises(ValueError):fit_cuts(d)

    def test_gate_is_equal_episode_not_decision_weighted(self):
        d=pd.DataFrame(dict(variant=['A']*4,task=[0,0,0,1],init=[20]*4,
            recipe_mse=[1.]*4,cache_mse=[2.,2.,2.,0.],gate=[True]*4))
        r=gate_metrics(d,'gate')
        self.assertAlmostEqual(r['delta'],0.)
        self.assertEqual(r['episodes_improved'],1)
        self.assertEqual(r['episodes_worsened'],1)

    def test_recovery_uses_next_fresh_and_censors_end(self):
        rows=[]
        for j in range(6):
            rows.append(dict(model='groot',arm='recipe',task=0,init=20,seq=j,success=False,
                vision=j in (0,2,4),call=j in (0,4),src='policy' if j in (0,4) else 'cache',
                closed=False,close_last=False,intragrip=False,reason=4 if j in (0,4) else 0,
                dnn=.1,noprog=0 if j==2 else 3,progress=j/10,liblen=11,lag=0,
                correction=.03,grip_reconstruction=0,escalation=False,motion=.1))
        _,e,c=analyze.annotate(pd.DataFrame(rows))
        self.assertEqual(c.next_seq.iloc[0],2)
        self.assertTrue(c.next_np_clear.iloc[0])
        self.assertTrue(pd.isna(c.next_seq.iloc[1]))
        self.assertIsNone(c.next_np_clear.iloc[1])
        self.assertEqual(e.calls.iloc[0],2)

    def test_pair_direction(self):
        e=pd.DataFrame([dict(model='groot',arm=a,task=0,init=20,success=s) for a,s in [('recipe',False),('P10',True)]])
        self.assertEqual(analyze.pair_episodes(e).group.iloc[0],'lost')


class RealOutputTests(unittest.TestCase):
    def test_admitted_population_and_join(self):
        d=pd.read_parquet(safe.HERE/'results/decisions.parquet')
        self.assertTrue(d.init.between(0,29).all())
        self.assertEqual(d.groupby(['model','arm','task','init']).ngroups,1200)
        self.assertFalse(d.duplicated(['model','arm','task','init','seq']).any())

    def test_recipe_guard_and_gripper_reconstruction(self):
        d=pd.read_parquet(safe.HERE/'results/decisions.parquet')
        r=d[(d.arm=='recipe')&d.vision]
        self.assertEqual(int(((r.noprog>=2)&~r.call).sum()),0)
        self.assertEqual(int((r.call&(r.reason==4)&(r.noprog<2)).sum()),0)
        self.assertLess(r.grip_reconstruction.max(),1e-6)

    def test_recorded_summary_totals(self):
        e=pd.read_parquet(safe.HERE/'results/episodes.parquet')
        s=json.loads((safe.HERE/'results/summary.json').read_text())
        for model in ('groot','pi05'):
            for split,keep in [('fit',e.init<20),('eval',e.init>=20),('all',e.init>=0)]:
                for arm in ('recipe','P10'):
                    r=e[keep&(e.model==model)&(e.arm==arm)]
                    x=s[model]['splits'][split]['arms'][arm]
                    self.assertEqual(int(r.calls.sum()),x['calls'])
                    self.assertEqual(int(r.decisions.sum()),x['decisions'])


if __name__=='__main__':unittest.main(verbosity=2)
