"""Scientific-contract tests, self-contained synthetic data only.

Run: python -m unittest exp.offline_search.rounds.r09.explore_astra.tools.test_tools
"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .common import HERE, accepted, assert_discovery, mixture_ir, paired_bootstrap
from .extract import join_aug
from .routing import optimize
from .shadows import candidates, errors
from .student import fit_residual, predict


class Contracts(unittest.TestCase):
    def test_split_rejects_holdout(self):
        with self.assertRaises(ValueError):assert_discovery([0],[30])
        assert_discovery([0,9],[0,29])

    def test_attempt_acceptance_and_holdout(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path=Path(tmp)/'journal.jsonl'
            rows=[dict(task_uid=f'x:eval:{t}:{i}',attempt=1,accepted=True,status='done',success=True)
                  for t in range(10) for i in range(50)]
            # An unaccepted retry is not an outcome.
            rows.append(dict(task_uid='x:eval:0:0',attempt=2,accepted=False,status='failed',success=False))
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(len(accepted(path)),300)
            self.assertTrue(accepted(path)[0,0]['success'])
            rows.append(dict(task_uid='x:eval:0:0',attempt=2,accepted=True,status='failed',success=False))
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):accepted(path)

    def test_ratio_not_average(self):
        # The two episode choices have IR .1 and .5, but unequal lengths.
        self.assertAlmostEqual(mixture_ir([[10,5]],[[100,10]],[[.5,.5]]),15/110)
        self.assertNotAlmostEqual(15/110,.3)

    def test_aug_join_by_identity_and_missing_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            d=Path(tmp);p=d/'aug/policy_shadow';p.mkdir(parents=True)
            a=np.zeros((3,10,32));a[:,0,0]=[10,20,99]
            np.savez(p/'part_00000.npz',decision_id=np.array(['b','a','holdout']),chunk=a)
            arr,meta=join_aug(d,'policy_shadow',{'a':0,'b':1},['chunk'])
            np.testing.assert_equal(arr['chunk'][:,0,0],[20,10])
            self.assertEqual(meta['decisions'],2)
            with self.assertRaises(ValueError):join_aug(d,'policy_shadow',{'missing':0},['chunk'])

    def test_gripper_separate_and_padding_ignored(self):
        a=np.zeros((2,10,32));b=a.copy();b[:,:,7:]=np.nan;b[:,:,6]=-1
        m,g=errors(a,b)
        np.testing.assert_equal(m,0);np.testing.assert_equal(g,1)

    def test_synthesis_fallback_for_no_success(self):
        rng=np.random.default_rng(2);a=rng.normal(size=(3,16,10,7));w=np.ones((3,16))/16
        result=dict(candidates(a,w,np.zeros((3,16)),np.tile(np.arange(16),(3,1))))
        np.testing.assert_allclose(result['successful_only'],result['mean'])

    def test_routing_uses_cheaper_equivalent_task(self):
        y=np.ones((10,20,2));y[5:,:,1]=0
        c=np.broadcast_to([5.,1.],y.shape);n=np.full(y.shape,10.)
        p,d=optimize(y,c,n,prior=0,tolerance=0)
        np.testing.assert_allclose(p[:5,1],1)
        np.testing.assert_allclose(p[5:,0],1)

    def test_bootstrap_pairs_preserved(self):
        x=np.full((10,30),.123)
        b=paired_bootstrap(x,draws=50)
        self.assertAlmostEqual(b['lo'],.123);self.assertAlmostEqual(b['hi'],.123)

    def test_student_heldout_linear_signal(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(600,3));coef=rng.normal(size=(3,60))
        target=(x@coef).reshape(-1,10,6)
        model=fit_residual(x[:500],target[:500],np.arange(500)%25)
        pred=predict(model,x[500:])
        self.assertLess(np.mean((pred-target[500:])**2),.2*np.mean(target[500:]**2))


if __name__=='__main__':unittest.main()
