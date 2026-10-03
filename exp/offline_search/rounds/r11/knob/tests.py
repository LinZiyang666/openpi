"""Both SPEC audit lists: arithmetic, committed lifecycle and isolation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from exp.offline_search.harness import api
from .recipe import R11Knob, METHODS, REASONS, uniform, score_features, predictor_score, history
from .controller import MissController, Ledger

class BaseStub:
    def __init__(self):
        self.n=0; self.guard=False; self._s={'flag':[]}; self._policy_gate_anchor=None
    def reset(self,e): self.n=0; self._s={'flag':[]};self._policy_gate_anchor=None
    def query(self,q):
        self.n+=1;self._s['flag'].append(int(self.guard));self._policy_gate_anchor={'step':q.step}
        return api.Result(np.array([0]),np.array([-1.]),.8,action=np.ones((10,32),np.float32),
            library='current',extras=dict(os_force_miss=float(self.guard),os_reason=4. if self.guard else 0.,os_flags=8. if self.guard else 0.))

def make(method,setting=.5):
    obj=R11Knob(method=method,target=None if method=='off' else .32)
    obj.inner=BaseStub();obj.row_subset=np.array([42]);obj._score=lambda q:.6
    cfg=dict(setting=setting,dose=setting,threshold=.5,tie_probability=.3,beta=.5,eta=.2,
             score_reference=np.array([.1,.2,.5,.5,.8,1.]))
    obj.configure(cfg,'pi05');obj.reset(SimpleNamespace(uid='ep',init=7))
    return obj

def query(step=0,hv=None,hit=None):
    return SimpleNamespace(step=step,task_id=3,episode=SimpleNamespace(uid='ep',init=7),
        hist_has_vision=np.asarray(hv if hv is not None else [i%2==0 for i in range(step)]),
        hist_hit=np.asarray(hit if hit is not None else [1]*step))

class Tests(unittest.TestCase):
    def test_every_method_guard_precedes_knob(self):
        for method in METHODS:
            with self.subTest(method=method):
                obj=make(method,1.);obj.inner.guard=True
                r=obj.query(query())
                self.assertEqual(r.extras['os_reason'],4.)
                self.assertEqual(r.extras['os_flags'],8.)
                self.assertEqual(obj.inner._s['flag'],[1])
                self.assertEqual(obj.inner._policy_gate_anchor,{'step':0})
                if method!='off':self.assertEqual(r.extras['os_r11_knob_call'],0.)
                if method in METHODS[5:]:
                    self.assertEqual(r.extras['r11_guard'],1.)
                    self.assertTrue({'r11_knob','r11_p','r11_score','r11_dose'} <= r.extras.keys())
    def test_every_active_method_idempotent(self):
        for method in METHODS[1:]:
            with self.subTest(method=method):
                obj=make(method);a=obj.query(query());before=copy.deepcopy(obj.inner._s)
                q0=obj._controller.q if obj._controller else None
                b=obj.query(query())
                self.assertEqual(a.extras,b.extras);self.assertEqual(obj.inner.n,1)
                self.assertEqual(obj.inner._s,before)
                self.assertEqual(q0,obj._controller.q if obj._controller else None)
    def test_off_exact_fields_and_no_coin(self):
        obj=make('off');r=obj.query(query())
        self.assertEqual(set(r.extras),{'os_force_miss','os_reason','os_flags'})
        self.assertIsNone(obj._proposal)
    def test_every_active_method_reset_and_connection_isolation(self):
        for method in METHODS[1:]:
            with self.subTest(method=method):
                a=make(method,1.);b=copy.deepcopy(a)
                a.query(query());self.assertEqual(b.inner.n,0)
                self.assertIsNot(a._guard_steps,b._guard_steps)
                if a._controller:
                    self.assertIsNot(a._controller,b._controller)
                    self.assertIsNot(a._controller.rng,b._controller.rng)
                a.inner.guard=True;a.query(query(2));self.assertNotIn(2,b._guard_steps)
                a.reset(SimpleNamespace(uid='ep',init=7))
                self.assertEqual(a._guard_steps,set());self.assertEqual(a._trigger_steps,set())
                self.assertEqual(a._proposal_step,-1)
                if a._controller:self.assertEqual(a._controller.q,1.)
    def test_keyed_uniform_equals_r6_exact_payload(self):
        from exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.methods import uniform as ref
        for key in ('R11-random-v1','R11-gap-v1'):
            for step in (-1,0,2,100):
                u=uniform(key,0,9,49,step,'cap')
                self.assertEqual(u,ref(key,0,9,49,step,'cap'));self.assertTrue(0<=u<1)
        with self.assertRaises(api.ContractError):uniform('k',0,-1,0,0,'d')
        self.assertNotEqual(uniform('k',0,1,2,0,'d'),uniform('k',0,1,2,0,'e'))
    def test_random_binomial_and_common_coins(self):
        us=np.array([uniform('R11-random-v1',0,t,i,s,'knob-anchor') for t in range(10) for i in range(50) for s in range(0,50,2)])
        for rho in (.2,.4,.7):self.assertLess(abs((us<rho).mean()-rho),5*np.sqrt(rho*(1-rho)/len(us)))
        self.assertTrue(np.all((us<.2)<=(us<.4)))
    def test_periodic_bound_and_committed_call_reset(self):
        for K in (0.,.4,1.,1.7,3.):
            obj=make('periodic',K);hv=[];hits=[]
            for s in range(0,100,2):
                obj.inner.guard=s in (12,32)
                r=obj.query(query(s,hv,hits));x=r.extras
                self.assertLessEqual(x['os_r11_run'],x['os_r11_cap'])
                if not obj.inner.guard:self.assertEqual(bool(x['os_r11_knob_call']),x['os_r11_run']>=x['os_r11_cap'])
                hv.extend([True,False]);hits.extend([int(not x['os_force_miss']),1])
    def test_post_guard_and_random_tail_single_followup(self):
        a=make('periodic_pgt1',10);a.inner.guard=True;a.query(query())
        a.inner.guard=False;r=a.query(query(2,[True,False],[0,1]));self.assertEqual(r.extras['os_r11_knob_call'],1.)
        r=a.query(query(4,[True,False,True,False],[0,1,0,1]));self.assertEqual(r.extras['os_r11_knob_call'],0.)
        a=make('random_tail2',1.);r=a.query(query());self.assertEqual(r.extras['os_reason'],64.)
        a.knob_settings['setting']=0.;r=a.query(query(2,[True,False],[0,1]));self.assertEqual(r.extras['os_r11_knob_call'],1.)
        r=a.query(query(4,[True,False,True,False],[0,1,0,1]));self.assertEqual(r.extras['os_r11_knob_call'],0.)
    def test_proposed_tail_requires_actual_committed_miss(self):
        for method in ('periodic_pgt1','random_tail2'):
            a=make(method,1. if method=='random_tail2' else 20.)
            a.inner.guard=method=='periodic_pgt1';a.query(query())
            a.inner.guard=False;a.knob_settings['setting']=0. if method=='random_tail2' else 20.
            r=a.query(query(2,[True,False],[1,1]))
            self.assertEqual(r.extras['os_r11_knob_call'],0.)
    def test_every_state_method_nonfinite_conservative(self):
        for method in METHODS[5:]:
            a=make(method);a._score=lambda q:float('nan');r=a.query(query())
            self.assertEqual(r.extras['os_force_miss'],1.)
            self.assertEqual(r.extras['os_r11_nonfinite'],1.)
    def test_controller_parity_with_astra_and_retries(self):
        from exp.offline_search.rounds.r11.astra.controller import MissController as Ref,Ledger as RL
        for mode in ('threshold','hybrid','adaptive'):
            kw=dict(model='groot',method=mode,dose=.4,target=.32,threshold=.7,tie_probability=.25,
                    score_reference=np.array([.2,.7,.7,1.]))
            a,b=MissController(**kw),Ref(**kw);a.reset('uid');b.reset('uid')
            N=V=M=0
            for step in range(0,100,2):
                score=(.2,.7,1.,np.nan)[(step//2)%4];g=step%8==2
                aa=a.propose(step=step,score=score,guard=g,ledger=Ledger(N,V,M))
                bb=b.propose(step=step,score=score,guard=g,ledger=RL(N,V,M))
                self.assertEqual(aa,bb);self.assertEqual(aa,a.propose(step=step,score=score,guard=g,ledger=Ledger(N,V,M)))
                N+=2;V+=1;M+=int(aa['miss'])
    def test_adaptive_accepted_ledger_includes_anomalous_looks(self):
        a=MissController(model='pi05',method='adaptive',dose=.5,target=.32,score_reference=[0,1])
        a.propose(step=0,score=.7,guard=False,ledger=Ledger(0,0,0))
        self.assertEqual(a.q,.5)
        a.propose(step=4,score=.7,guard=True,ledger=Ledger(4,3,2))
        expected=float(np.clip(.5-.2*(.152*3+.848*2-.32*4)/.848,0,1))
        self.assertEqual(a.q,expected)
        a.propose(step=4,score=.7,guard=True,ledger=Ledger(4,3,2));self.assertEqual(a.q,expected)
        with self.assertRaises(ValueError):a.propose(step=6,score=.7,guard=False,ledger=Ledger(6,2,1))
    def test_static_endpoint_and_tied_probability(self):
        for mode in ('threshold','hybrid'):
            for dose in (0.,1.):
                a=MissController(model='pi05',method=mode,dose=dose,target=.32,threshold=.5)
                for s in (-100.,100.,float('nan')):
                    a.reset('e');r=a.propose(step=0,score=s,guard=False,ledger=Ledger(0,0,0));self.assertEqual(r['probability'],dose)
            a=MissController(model='pi05',method=mode,dose=.4,target=.32,threshold=.5,tie_probability=.3)
            r=a.propose(step=0,score=.5,guard=False,ledger=Ledger(0,0,0))
            self.assertEqual(r['probability'],.3 if mode=='threshold' else .35)
    def test_history_real_look_only_and_missing_commit_refused(self):
        anchors,last,run,L=history(query(5,[True,False,True,True,False],[0,1,1,0,1]))
        self.assertEqual((last,run,L),(3,0,Ledger(5,3,2)))
        with self.assertRaises(api.ContractError):history(query(2,[True],[0]))
        with self.assertRaises(api.ContractError):history(query(2,[True,False],[-1,1]))
    def test_features_exact_reference_uncorrected_grip_and_padding(self):
        from exp.offline_search.rounds.r11.astra.experiment import retrieve,predictor_score as refscore
        from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
        rng=np.random.default_rng(77)
        act=rng.normal(size=(17,10,7)).astype(np.float32);ds=np.linspace(1.,3.,16)[None]
        sig=rng.uniform(.2,2.,7).astype(np.float32)
        C=dict(X=np.zeros((17,8),np.float32),step=np.zeros(17,int),ep=np.arange(17),act=act,kref=5)
        met=SimpleNamespace(distances=lambda X,step:ds.copy())
        ref=retrieve(C,np.array([0]),np.arange(1,17),met,sig)
        w=_kernel_w(ds-ds[:,:1],5);w=(w/w.sum(1,keepdims=True)).astype(np.float32)[0]
        X=score_features(act[1:],w,ds[0],sig)
        np.testing.assert_array_equal(X,ref['X'][0])
        head=dict(mean=np.zeros(8),std=np.ones(8),coef=np.arange(9)/10)
        self.assertAlmostEqual(predictor_score(head,X),refscore(head,X[None])[0],places=14)
        padded=np.pad(act[1:],((0,0),(0,6),(0,25)),constant_values=100)
        np.testing.assert_array_equal(score_features(padded,w,ds[0],sig),X)
    def test_calibration_boundary_and_task_free_payload(self):
        from .calibration import bound
        with self.assertRaises(ValueError):bound('/home/weiland/trace_runs/os_closed_loop/r11_knob_1/eval500.json')
        from .build import artifact_path
        import pickle
        with artifact_path('pi05','l10',50,'adaptive_error_hybrid',.32).open('rb') as f:a=pickle.load(f)['method']
        self.assertEqual(a.knob_settings['predictor']['coef'].shape,(9,))
        self.assertEqual(a._controller.reference.ndim,1)
        self.assertNotIn('task',a.knob_settings)

if __name__=='__main__': unittest.main(verbosity=2)
