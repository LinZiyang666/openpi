"""Unit tests confined to owned files; no existing tests/ or review_tests/ read."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from ..inference import Monitor,RecoveryGate,LatchedGate,RandomBurstGate
from . import data
from .labels import targets
from .metrics import ranking
from .responses import expected_random_calls
from .screen import selected_entries
from .live_results import admitted
from .bounds import coupling


class ToolsTest(unittest.TestCase):
    def test_identity_rejected_before_payload(self):
        class Fake:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def __getitem__(self,k):
                if k=='task':return np.array([0,1])
                if k=='init':return np.array([20,30])
                raise AssertionError('payload touched before forbidden identity rejection')
        with tempfile.TemporaryDirectory(dir=data.HERE/'results') as td:
            root=Path(td);p=root/'r8_pi05_l10_50_A.npz';p.touch()
            p.with_suffix('.json').write_text(json.dumps(dict(episodes=300,split='inits 0..29')))
            with patch.object(data,'COMPACT',root),patch.object(data.np,'load',return_value=Fake()):
                with self.assertRaises(ValueError):data.source('pi05_l10_50','A')
        with self.assertRaises(ValueError):data.source('../r09_astra_holdout','A')

    def test_train_eval_disjoint(self):
        tr,te=data.masks(np.arange(30))
        self.assertFalse(np.any(tr&te));self.assertEqual(tr.sum(),20);self.assertEqual(te.sum(),10)
        with self.assertRaises(ValueError):data.masks(np.array([49]))

    def test_labels_are_pre_action_and_keep_initial_goals(self):
        # Two settle controls, followed by 20 live controls; first new goal
        # arises at live control 2. At decision 0 it has NOT happened yet.
        p=np.zeros((22,2));p[:,0]=1;p[4:,1]=1
        settle=np.r_[np.ones(2,bool),np.zeros(20,bool)]
        seq=np.r_[-1,-1,np.repeat(np.arange(4),5)]
        y=targets(p,seq,settle,np.array([0,1,2,3]),True)
        np.testing.assert_array_equal(y['new_goals'],[0,1,1,1])
        np.testing.assert_array_equal(y['current_goals'],[1,2,2,2])
        np.testing.assert_array_equal(y['prior_controls'],[0,5,10,15])
        self.assertEqual(y['goal_age'][1],2)
        self.assertEqual(y['progress30'][0],1)

    def test_censoring_and_late_stall(self):
        p=np.zeros((202,2));p[4:,0]=1
        settle=np.r_[True,True,np.zeros(200,bool)]
        seq=np.r_[-1,-1,np.repeat(np.arange(40),5)]
        y=targets(p,seq,settle,np.array([0,10,14,30]),False)
        np.testing.assert_array_equal(y['late_stall'],[False,False,True,True])
        np.testing.assert_array_equal(y['valid100'],[True,True,True,False])
        self.assertTrue(y['late_trap100'][2]);self.assertEqual(y['progress100'][2],0)

    def test_monitor_reset_and_prefix_causality(self):
        def run(n):
            m=Monitor();out=[]
            for j in range(n):
                out.append(m.observe(np.ones(128)*(j+1),np.ones(8)*j,np.ones((10,7))*j,
                    np.arange(16),np.ones(16)/16,np.linspace(0,1,16),j*2,.2))
            return np.array(out)
        np.testing.assert_array_equal(run(8),run(12)[:8])
        self.assertEqual(run(1).shape,(1,57));self.assertTrue(np.isfinite(run(8)).all())

    def test_gate_warmup_burst_cap_cooldown(self):
        g=RecoveryGate(.5,burst=3,cooldown=2,max_calls=5,warmup=4)
        calls=[];starts=[]
        for step in range(0,24,2):
            c,s=g.observe(.9,step);calls.append(c);starts.append(s)
        self.assertEqual(np.flatnonzero(calls).tolist(),[3,4,5,9,10])
        self.assertEqual(np.flatnonzero(starts).tolist(),[3,9]);self.assertEqual(g.used,5)
        with self.assertRaises(ValueError):g.observe(.9,22)

    def test_latch_minimum_hysteresis_and_limit(self):
        g=LatchedGate(.5,min_calls=3,max_calls=5,cooldown=2,warmup=0)
        result=[g.observe(s,j*2)[0] for j,s in enumerate([1,1,.1,.1,.1,.1,.1,1,1,1,1])]
        self.assertEqual(np.flatnonzero(result).tolist(),[1,2,3,8,9]);self.assertEqual(g.used,5)

    def test_random_expectation_matches_exhaustive_coins(self):
        p=.3;steps=np.arange(0,14,2);total=0.
        for coins in itertools.product([False,True],repeat=len(steps)):
            g=RandomBurstGate(p,burst=2,max_calls=4,cooldown=1,warmup=2)
            n=sum(g.observe(0. if c else 1.,int(s))[0] for c,s in zip(coins,steps))
            k=sum(coins);total+=n*p**k*(1-p)**(len(steps)-k)
        got=expected_random_calls(steps,p,burst=2,cap=4,cooldown=1,warmup=2)
        self.assertAlmostEqual(got,total,places=12)

    def test_first_alert_is_pre_treatment(self):
        d=dict(episode=np.repeat([0,1],10),seq=np.tile(np.arange(0,20,2),2))
        s=np.ones(20);first,calls,starts=selected_entries(d,s,.5)
        self.assertEqual(np.flatnonzero(first).tolist(),[7,17]);self.assertEqual(calls.sum(),6)
        d['success']=np.zeros(20);d['teacher']=np.zeros((20,10,7))
        np.testing.assert_array_equal(first,selected_entries(d,s,.5)[0])

    def test_weighted_ranking_ties(self):
        self.assertEqual(ranking([0,1],[1,1]),(.5,.5))
        self.assertEqual(ranking([0,1],[0,1]),(1.,1.))
        self.assertEqual(ranking([0,1],[1,0]),(.5,0.))
        ap,auc=ranking([0,1],[1,1],[3,1]);self.assertAlmostEqual(ap,.25);self.assertEqual(auc,.5)

    def test_journal_identity_before_payload(self):
        self.assertIsNone(admitted('{"task_uid":"arm:eval:0:30", NOT_JSON_PAYLOAD'))
        self.assertIsNone(admitted('{"task_uid":"arm:eval:0:19", NOT_JSON_PAYLOAD'))
        self.assertEqual(admitted('{"task_uid":"arm:eval:4:29", NOT_JSON_PAYLOAD'),(4,29))

    def test_coupling_bounds_enclose_all_possible_responses(self):
        for rescued in range(4):
            for harmed in range(3):
                b=coupling(10,6,3,2,rescued/3,harmed/2)
                self.assertAlmostEqual(b['scenario'],(6+rescued-harmed)/10)
                self.assertGreaterEqual(b['scenario'],b['lower']);self.assertLessEqual(b['scenario'],b['upper'])
        self.assertEqual(coupling(10,6,0,0,1,1),dict(lower=.6,upper=.6,scenario=.6))

    def test_frozen_manifest_and_hashes(self):
        from .prepare_confirmation import OUT
        from .data import sha
        frozen=json.loads((data.HERE/'FROZEN_CANDIDATES.json').read_text())
        self.assertEqual(frozen['fit_inits'],list(range(20)))
        manifest=json.loads((OUT/'manifests/eval100.json').read_text())
        self.assertEqual({(r['task'],r['init']) for r in manifest['selected']},
            {(t,i) for t in range(10) for i in range(20,30)})
        self.assertEqual(len(frozen['candidates']),16)
        for candidate in frozen['candidates']:
            self.assertEqual(sha(candidate['artifact']),candidate['artifact_sha256'])
            self.assertEqual(sha(candidate['kwargs']['head_path']),candidate['head_sha256'])
            self.assertEqual(sha(candidate['kwargs']['phase_path']),candidate['phase_sha256'])
        specs=json.loads((data.HERE/'confirmation_specs.json').read_text())
        self.assertEqual(len(specs),22)
        for spec in specs:
            self.assertEqual(spec['manifest'],str(OUT/'manifests/eval100.json'))
            self.assertNotIn('--os-oracle',spec['plugin_args'])
            if spec['method'].endswith('TransitionRecovery'):
                self.assertTrue(spec['full_model']);self.assertIn('--os-policy-tail',spec['plugin_args'])


if __name__=='__main__':unittest.main()
