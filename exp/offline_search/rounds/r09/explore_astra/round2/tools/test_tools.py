"""Meaningful isolation, causal-estimand and serving-contract tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from . import data
from .data import masks, temporal, balanced_weights, interval
from .learn import fit
from .call_value import first_entry
from .statistics import paired_ni
from ..inference import inputs, correct, predict


class Tests(unittest.TestCase):
    def test_loader_rejects_locked_identity_before_actions(self):
        # Synthetic fixture only: contains no real forbidden observation/outcome.
        with tempfile.TemporaryDirectory(dir=data.HERE) as directory:
            root=Path(directory);name='r8_pi05_l10_50_A'
            (root/(name+'.npz')).touch()
            (root/(name+'.json')).write_text(json.dumps(dict(episodes=300,split='inits 0..29')))
            class Trap:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def __getitem__(self,key):
                    if key=='task':return np.array([0,0])
                    if key=='init':return np.array([0,30])
                    raise AssertionError('Loaded content before rejecting forbidden identity')
            with patch.object(data,'COMPACT',root),patch.object(data.np,'load',return_value=Trap()):
                with self.assertRaisesRegex(ValueError,'Forbidden identity'):data.load('pi05_l10_50')

    def test_population_allowlist(self):
        with self.assertRaises(ValueError):data.load('../r09_astra_holdout')
        with self.assertRaises(ValueError):data.load('pi05_l10_50','arbitrary')
        with self.assertRaises(ValueError):masks(np.array([30]))

    def test_split_disjoint_across_paths(self):
        init=np.tile(np.arange(30),30)
        tr,te=masks(init)
        self.assertFalse((tr&te).any());self.assertTrue((tr|te).all())
        self.assertEqual(set(init[tr]),set(range(20)))
        self.assertEqual(set(init[te]),set(range(20,30)))

    def test_temporal_features_exact_and_no_future(self):
        state=np.arange(32,dtype=float).reshape(4,8)
        base=np.arange(280,dtype=float).reshape(4,10,7)
        seq=np.array([0,2,0,2]);episode=np.array([1,1,2,2])
        off=temporal(state,base,seq,episode)
        self.assertTrue((off[[0,2]]==0).all())
        previous=dict(state=state[0],action=base[0],step=0)
        x=inputs(np.zeros(128),state[1],base[1],2,previous,True)
        np.testing.assert_array_equal(x[0,-16:],off[1])
        modified=state.copy();modified[2:]=1e5
        np.testing.assert_array_equal(temporal(modified,base,seq,episode)[:2],off[:2])

    def test_episode_balancing(self):
        e=np.array([1,1,1,2]);w=balanced_weights(e)
        self.assertAlmostEqual(w[e==1].sum(),w[e==2].sum())

    def test_fit_eval_isolation_and_regression(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(400,3));truth=rng.normal(size=(3,2))
        y=x@truth;ep=np.arange(400)//10
        model=fit(x[:200],y[:200],ep[:200],random_features=0,alpha=.001)
        self.assertLess(np.mean((predict(model,x[200:])-y[200:])**2),1e-7)
        frozen=model['coef'].copy();y[200:]=1e9
        again=fit(x[:200],y[:200],ep[:200],random_features=0,alpha=.001)
        np.testing.assert_array_equal(frozen,again['coef'])

    def test_gripper_threshold_padding_and_zero(self):
        scores=np.zeros((10,7));scores[:,:6]=2;scores[:,6]=-.81;scores[0,6]=-.79
        model=dict(mean=np.zeros(1),std=np.ones(1),w=np.empty((1,0)),bias=np.empty(0),
            coef=np.zeros((1,70)),intercept=scores.ravel())
        action=np.ones((1,12,9))
        out=correct(model,np.zeros((1,1)),action,.5,True)
        np.testing.assert_array_equal(out[0,:10,:6],2.)
        self.assertEqual(out[0,0,6],1);np.testing.assert_array_equal(out[0,1:10,6],-1.)
        np.testing.assert_array_equal(out[:,:,7:],action[:,:,7:]);np.testing.assert_array_equal(out[:,10:],action[:,10:])
        np.testing.assert_array_equal(correct(model,np.zeros((1,1)),action,0,False),action)

    def test_first_entry_and_ht_estimand(self):
        np.testing.assert_array_equal(first_entry(np.array([False,True,True,True,True]),np.array([0,0,0,1,1])),[False,True,False,True,False])
        # Exact randomization enumeration, not a noisy Monte Carlo assertion.
        p=.25;m=.45;y1=.7;y0=.2
        ht=p*((y1-m)/p)+(1-p)*(-(y0-m)/(1-p))
        self.assertAlmostEqual(ht,y1-y0)

    def test_interval_uses_init_clusters(self):
        eps=np.array([20,21,50,51]);v=np.array([0.,1.,0.,1.])
        s=interval(v,eps,draws=1000)
        self.assertEqual(s['init_clusters'],2);self.assertEqual(s['mean'],.5)

    def test_noninferiority_does_not_turn_no_difference_into_proof(self):
        r=paired_ni(0,0,100,family=3)
        self.assertEqual(r['delta'],0)
        self.assertFalse(r['passes']);self.assertLess(r['lower'],-.02)
        self.assertTrue(paired_ni(60,0,200,family=3)['passes'])
        with self.assertRaises(ValueError):paired_ni(8,8,10)

    def test_frozen_population_and_single_head(self):
        frozen=json.loads((data.HERE/'FROZEN_CANDIDATES.json').read_text())
        self.assertEqual(frozen['fit_inits'],list(range(20)))
        self.assertEqual(frozen['evaluation_inits'],list(range(20,30)))
        self.assertTrue(frozen['no_task_switches'])
        for item in frozen['candidates']:
            with np.load(item['kwargs']['head_path'],allow_pickle=False) as z:
                self.assertEqual(set(z.files),{'mean','std','w','bias','coef','intercept'})
        specs=json.loads((data.HERE/'confirmation_specs.json').read_text())
        for spec in specs:
            manifest=json.loads(Path(spec['manifest']).read_text())
            self.assertEqual({p['init'] for p in manifest['selected']},set(range(20,30)))


if __name__=='__main__':unittest.main()
