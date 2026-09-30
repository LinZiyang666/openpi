"""Numerical/contract tests; no simulator, policy, or held-out tuning."""
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .stall import StallModel, StallTracker, monotone_alignment, _quantile, _cdf, _cdf_table
from .bmech import BmechCommitJudge
from exp.offline_search.harness import api
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge

HERE=Path(__file__).resolve().parent


def fixture(E=5, n=21, R=3, H=11):
    step=np.tile(np.arange(n),E)
    episode=np.repeat(np.arange(E),n)
    codes=np.column_stack((step,episode*.01)).astype(float)
    library=dict(manifest=dict(H=H,exec_steps=R,task_map={'task':0}),task_id=np.zeros(E*n,int),
                 episode=episode,step=step,success=np.ones(E*n,bool))
    metric=dict(tasks={'0':dict(rows=np.arange(E*n),codes={'main':codes,'early':codes*2})})
    return library,metric


class AlignmentTests(unittest.TestCase):
    def test_exhaustive_oracle_and_ties(self):
        rng=np.random.default_rng(327)
        for n in range(1,7):
            for w in range(1,6):
                for _ in range(5):
                    d=rng.integers(0,4,(w,n)).astype(float)
                    paths=list(itertools.combinations_with_replacement(range(n),w))
                    best=min(paths,key=lambda p:(sum(d[i,j] for i,j in enumerate(p)),p))
                    cost,path=monotone_alignment(d)
                    self.assertEqual(tuple(path),best)
                    self.assertAlmostEqual(cost,sum(d[i,j] for i,j in enumerate(best))/w)

    def test_missing_distances_rejected(self):
        for d in (np.empty((0,2)),[[0,np.nan]],[[np.inf,1]]):
            with self.assertRaises(ValueError):monotone_alignment(d)

    def test_inverse_ecdf_and_episode_weights(self):
        self.assertEqual(_quantile([1,2,3,4],.5),2)
        table=_cdf_table([0,0,0,1],[0,0,0,1])
        self.assertAlmostEqual(_cdf(table,0),.5)
        self.assertEqual(_cdf(table,-1),0)
        self.assertEqual(_cdf(table,1),1)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib,cls.metric=fixture()
        cls.model=StallModel.fit(cls.lib,metric=cls.metric,commit_controls=6)

    def test_library_only_reproducible(self):
        second=StallModel.fit(self.lib,metric=self.metric,commit_controls=6)
        self.assertEqual(second.fingerprint,self.model.fingerprint)
        self.assertEqual(self.model.tasks['0']['W'],2)
        self.assertEqual(self.model.tasks['0']['K'],4)
        self.assertEqual(len(self.model.tasks['0']['references']),5)
        for r in self.model.tasks['0']['calibration']:
            self.assertAlmostEqual(r['delta_true'],.2)

    def test_successful_only(self):
        lib,metric=fixture();lib['success'][:21]=False
        model=StallModel.fit(lib,metric=metric,commit_controls=6)
        self.assertEqual(model.tasks['0']['E'],4)
        self.assertEqual(model.tasks['0']['K'],3)
        self.assertNotIn(0,[r['episode'] for r in model.tasks['0']['calibration']])

    def test_availability(self):
        lib,metric=fixture(E=3)
        model=StallModel.fit(lib,metric=metric,commit_controls=6)
        tracker=StallTracker(model,0)
        for i in range(5):tracker.observe([i,0],i*6)
        self.assertEqual(tracker.status()['reason'],'fewer_than_four_reference_episodes')
        self.assertEqual(StallTracker(model,'missing').status()['state'],'inactive')

    def test_roundtrip_corruption_and_no_overwrite(self):
        path=Path(tempfile.mkdtemp(prefix='_test_model_'))
        self.addCleanup(shutil.rmtree,path)
        self.model.save(path);loaded=StallModel.load(path)
        self.assertEqual(loaded.fingerprint,self.model.fingerprint)
        a,b=StallTracker(self.model,0),StallTracker(loaded,'task')
        for i in range(5):
            a.observe([i,0],i*6);b.observe([i,0],i*6)
            self.assertEqual(a.status(),b.status())
        with self.assertRaises(FileExistsError):self.model.save(path)
        (path/'stall.pkl').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):StallModel.load(path)

    def test_warmup_missing_reset_and_timestamps(self):
        tr=StallTracker(self.model,0)
        tr.observe([0,0],0);tr.observe([0,0],6)
        self.assertEqual(tr.status()['state'],'inactive')
        tr.observe([0,0],12)
        self.assertEqual(tr.status()['state'],'slow_confirmed')
        before=tr.status();before['reference_windows'][0][0]='changed'
        self.assertNotEqual(before,tr.status())
        tr.observe([np.nan,0],18)
        self.assertEqual(tr.status()['reason'],'invalid_observation')
        tr.observe([0,0],24)
        self.assertEqual(tr.status()['reason'],'warming_up')
        for bad in (24,23,-1,25.5,True):
            with self.assertRaises(ValueError):tr.observe([0,0],bad)

    def test_span_adjustment(self):
        w=[(np.array([x,0.]),'main') for x in (0,2,4)]
        p=self.model._estimate('0',w,12)
        q=self.model._estimate('0',w,24)
        self.assertAlmostEqual(p['delta_hat']/2,q['delta_hat'])
        self.assertAlmostEqual(p['distance_hat'],q['distance_hat'])

    def test_regime_dimensions_and_nonrobot_constants(self):
        tr=StallTracker(self.model,0)
        tr.observe({'metric_code':[0,0],'metric':'early'},0)
        tr.observe({'metric_code':[2,0],'metric':'main'},6)
        tr.observe({'metric_code':[4,0],'metric':'main'},12)
        self.assertEqual(tr.status()['state'],'ok')
        self.assertEqual(self.model.exec_steps,3)
        with self.assertRaises(ValueError):StallModel.fit(self.lib,metric=self.metric,commit_controls=10)

    def test_status_inequalities(self):
        model=StallModel.fit(self.lib,metric=self.metric,commit_controls=6)
        for r in model.tasks['0']['references']:
            r['residual']=np.full_like(r['residual'],.15)
            r['advance']=np.full_like(r['advance'],.1)
        pred=dict(delta_hat=.01,phase_hat=0.,distance_hat=0.,spread_hat=0.)
        self.assertEqual(model.calibrated_status('0',pred,12)['state'],'slow_ambiguous')
        pred['delta_hat']=.1
        self.assertEqual(model.calibrated_status('0',pred,12)['state'],'ok')
        pred['delta_hat']=-.1
        self.assertEqual(model.calibrated_status('0',pred,12)['state'],'slow_confirmed')


class BmechTests(unittest.TestCase):
    def new(self,enabled=True):
        return BmechCommitJudge(base_kwargs=dict(lib='current',kref=5,serving='anchor_tail',budget=1,gates='budget_only'),
            progress_guard='noprog_span',events='none',stuck_guard='vision_confirmed',policy_tail_gate='lifecycle',
            monitor='off',mask_no_progress=enabled)

    def test_mask_only_and_keep_diagnostic_memo(self):
        m=self.new();m._s=dict(burst_end=0,ret_end=0,flag=[1]);m._noprog_span=5
        for flags in range(16):
            r=api.Result(np.array([2]),np.array([.1]),.7,action=np.ones((3,2)),library='x',
                         extras=dict(os_flags=float(flags),os_reason=float((flags&-flags).bit_length()),
                                     os_force_miss=float(bool(flags)),diagnostic=2.))
            with patch.object(CommitJudge,'query',return_value=r):out=m.query(None)
            self.assertEqual(out.extras['os_flags'],flags&~8)
            self.assertEqual(out.extras['os_force_miss'],bool(flags&~8))
            self.assertIs(out.action,r.action)
            self.assertEqual(m._s['flag'],[1]);self.assertEqual(m._noprog_span,5)
        self.assertIs(BmechCommitJudge.blind_step,CommitJudge.blind_step)
        self.assertIs(BmechCommitJudge.policy_tail_step,CommitJudge.policy_tail_step)

    def test_disabled_mask_is_exact_identity(self):
        m=self.new(False)
        r=api.Result(np.array([0]),np.array([0.]),1.,extras={'os_flags':8.})
        with patch.object(CommitJudge,'query',return_value=r):self.assertIs(m.query(None),r)


if __name__=='__main__':unittest.main()
