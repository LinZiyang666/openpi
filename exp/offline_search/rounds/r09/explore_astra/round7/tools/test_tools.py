"""Meaningful split, task-independence, judge-path and frozen-pair tests."""
import copy
import json
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from .data import HERE, RUN, STORE, CELLS, admitted, identity_arrays, sha, artifact, json_identity, admitted_jsonl, owned
from .learn import train_assert
from .methods import RowCorrectedCache
from .numeric import predict, features, confidence
from exp.offline_search.rounds.r09.explore_astra.round6.tools.numeric import residual


def load(path):
    with open(path, 'rb') as f:
        return FitUnpickler(f).load()


class AdmissionTests(unittest.TestCase):
    def test_forbidden_root_rejected_before_open(self):
        for prefix in ('r09_holdout', 'r09_astra_holdout'):
            with self.assertRaises(ValueError):
                admitted('/nonexistent/' + prefix + '_blocked/file')

    def test_mixed_npz_identity_rejected_before_payload(self):
        class IdentityOnly:
            def __getitem__(self, key):
                if key == 'task': return np.array([0, 0])
                if key == 'init': return np.array([0, 99])
                raise AssertionError('payload accessed before rejection')
        with self.assertRaises(ValueError):
            identity_arrays(IdentityOnly())

    def test_training_rejects_eval_and_onehot_inputs(self):
        with self.assertRaises(ValueError):
            train_assert(dict(init=np.array([0, 20]), x=np.zeros((2, 207))))
        with self.assertRaises(ValueError):
            train_assert(dict(init=np.array([0, 19]), x=np.zeros((2, 217))))

    def test_numeric_row_permutation_equivariance(self):
        rng = np.random.default_rng(1)
        table = rng.normal(size=(100, 60)).astype(np.float32)
        rows = rng.integers(0, 100, (4, 16))
        w = rng.uniform(size=(4, 16))
        permutation = rng.permutation(100)
        moved = np.empty_like(table)
        moved[permutation] = table
        a = residual({}, table, np.zeros((4,207)), rows, w)
        b = residual({}, moved, np.zeros((4,207)), permutation[rows], w)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a, residual({}, table, np.zeros((4,207)), rows, w*17), atol=1e-7)

    def test_json_filter_precedes_payload_decode(self):
        path=owned(HERE/'results/synthetic_admission.jsonl')
        path.write_text('{"task_id":0,"init":99,"payload":INVALID}\n' +
                        '{"task_uid":"synthetic:eval:0:20","payload":7}\n')
        self.assertEqual(list(admitted_jsonl(path)),[dict(task_uid='synthetic:eval:0:20',payload=7)])
        self.assertEqual(json_identity('{"payload":{"init":99},"task_id":1,"init":20}'),(1,20))
        with self.assertRaises(ValueError): json_identity('{"init":20,"init":99,"task_id":0}')
        with self.assertRaises(ValueError): json_identity('{"init":20,"task_id":0,"uid":"x:0:99"}')

    def test_write_boundary(self):
        with self.assertRaises(ValueError): owned(HERE.parent/'forbidden_output')

    def test_gripper_features_and_confidence(self):
        x=np.zeros((2,207),np.float32); x[:,136:206]=np.tile(np.r_[np.zeros(6),1],10)
        x[1,142:206:7]=-1
        f=features(x,True)
        self.assertEqual(f.shape,(2,231)); self.assertEqual(f[0,207],0.); self.assertEqual(f[1,207],1.)
        actions=np.zeros((2,16,32),np.float32); actions[1,:,:6]=1
        rows=np.array([[0,0],[0,1]]);w=np.ones((2,2))
        cf=confidence(actions,rows,w,.25)
        np.testing.assert_allclose(cf,[1,.5])
        actions[:,:,7:]=999; actions[:,10:,:6]=999
        np.testing.assert_array_equal(cf,confidence(actions,rows,w,.25))
        np.testing.assert_array_equal(cf,confidence(actions,rows,w*3,.25))


class FrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {a['arm']:a for a in json.loads((RUN/'arms.json').read_text())}

    def test_standard_store_manifest_and_identical_client(self):
        self.assertEqual(len(self.rows), 6)
        manifest = json.loads((RUN/'manifests/eval100.json').read_text())
        self.assertEqual({(r['task'],r['init']) for r in manifest['selected']},
                         {(t,i) for t in range(10) for i in range(20,30)})
        for cell, variant in ((c,v) for c in CELLS for v in ('capacity','confidence')):
            c, n = [self.rows[f'r9a7_{cell}_{v}'] for v in ('control',variant)]
            self.assertEqual(c['client_overrides'], n['client_overrides'])
            self.assertEqual(c['kwargs'], {k:v for k,v in n['kwargs'].items() if k!='residual_path'})
            for a in (c,n):
                p = a['plugin_args']
                self.assertEqual(p[p.index('--os-root')+1], str(STORE))
                self.assertEqual(a['manifest'], str(RUN/'manifests/eval100.json'))
                self.assertEqual(a['judge'], 'guard_only')

    def test_only_corrector_changes_in_fitted_state(self):
        for cell, variant in ((c,v) for c in CELLS for v in ('capacity','confidence')):
            c, n = [load(artifact(self.rows[f'r9a7_{cell}_{v}']))['method'] for v in ('control',variant)]
            self.assertEqual(set(vars(n))-set(vars(c)), {'ar7_residual_path'})
            for key, value in vars(c).items():
                if key not in ('base', 'fit_info'):
                    self.assertEqual(pickle.dumps(value, protocol=4), pickle.dumps(getattr(n,key), protocol=4), (cell,key))
            self.assertEqual(c.burst, n.burst)
            self.assertEqual(c.burst, 0)
            self.assertEqual(n.gm_max_calls, 0)
            self.assertIs(type(n.base), RowCorrectedCache)
            self.assertFalse(any(k in vars(n.base) for k in RowCorrectedCache._OLD_HEAD))
            np.testing.assert_array_equal(c.base.act, n.base.act)

    def test_corrector_does_not_access_task_or_init(self):
        for cell, variant in ((c,v) for c in CELLS for v in ('capacity','confidence')):
            b = load(artifact(self.rows[f'r9a7_{cell}_{variant}']))['method'].base
            class PoisonIdentity:
                key_v0 = np.zeros(b.B0T.shape[1], np.float32)
                key_v1 = np.zeros(b.B1T.shape[1], np.float32)
                rs = np.zeros(8, np.float32)
                step = 4
                @property
                def task_id(self): raise AssertionError('task identity used')
                @property
                def episode(self): raise AssertionError('episode identity used')
                @property
                def init(self): raise AssertionError('init identity used')
            rows = np.arange(16)
            a = np.array(b.act[0], copy=True)
            out = b._ar7_action(PoisonIdentity(), a, rows, np.ones(16))
            self.assertTrue(np.isfinite(out).all())
            self.assertGreater(float(np.max(np.abs(out[:10,:6]-a[:10,:6]))), 1e-7)
            np.testing.assert_array_equal(out[10:], a[10:])
            np.testing.assert_array_equal(out[:10,6:], a[:10,6:])

    def test_judge_os_synth_corrects_once_and_remembers_tail(self):
        for cell, variant in ((c,v) for c in CELLS for v in ('capacity','confidence')):
            b = load(artifact(self.rows[f'r9a7_{cell}_{variant}']))['method'].base
            q = SimpleNamespace(key_v0=np.zeros(b.B0T.shape[1],np.float32),
                key_v1=np.zeros(b.B1T.shape[1],np.float32), rs=np.zeros(8,np.float32),
                task_id=0, step=4, episode=SimpleNamespace(uid='unit:0:0'))
            rows, w = np.arange(16), np.ones(16)
            a = np.array(b.act[0], copy=True)
            expected = b._ar7_action(q,a,rows,w)
            with patch.object(BlindAWM,'os_synth',return_value=a):
                out = b.os_synth(q,rows,w)
            np.testing.assert_array_equal(out,expected)
            np.testing.assert_array_equal(b._anchor['action'],expected)
            self.assertEqual(b._anchor['step'],4)

    def test_serving_padding_tail_invariance_and_strength_bounds(self):
        for cell, variant in ((c,v) for c in CELLS for v in ('capacity','confidence')):
            b=load(artifact(self.rows[f'r9a7_{cell}_{variant}']))['method'].base
            q=SimpleNamespace(key_v0=np.zeros(b.B0T.shape[1],np.float32),key_v1=np.zeros(b.B1T.shape[1],np.float32),rs=np.zeros(8,np.float32),step=4)
            rows=np.arange(16); w=np.ones(16); a=b.act[0].copy()
            out=b._ar7_action(q,a,rows,w)
            altered=a.copy();altered[:,7:]=123.;altered[10:,:7]=-321.
            out2=b._ar7_action(q,altered,rows,w)
            np.testing.assert_array_equal(out[:10,:7],out2[:10,:7])
            np.testing.assert_array_equal(out2[10:],altered[10:])
            np.testing.assert_array_equal(out2[:,7:],altered[:,7:])
            m=b.ar7_meta
            st=.5+m['confidence_gain']*confidence(b.act,rows[None],w[None],m['dispersion_scale'])
            self.assertTrue(np.all((st>=.5)&(st<=.75)))

    def test_full_model_row_permutation_and_batch_scalar_parity(self):
        rng=np.random.default_rng(260702)
        for cell,variant in ((c,v) for c in CELLS for v in ('capacity','confidence')):
            b=load(artifact(self.rows[f'r9a7_{cell}_{variant}']))['method'].base
            x=np.repeat(b.ar7_head['mean'][None,:207],3,axis=0)
            rows=rng.integers(0,len(b.act),(3,16));w=rng.uniform(size=(3,16))
            a=predict(b.ar7_head,b.ar7_table,x,rows,w,b.act,b.ar7_meta)
            single=np.concatenate([predict(b.ar7_head,b.ar7_table,x[i:i+1],rows[i:i+1],w[i:i+1],b.act,b.ar7_meta) for i in range(3)])
            np.testing.assert_allclose(a,single,atol=2e-6)
            perm=rng.permutation(len(b.act));t=np.empty_like(b.ar7_table);acts=np.empty_like(b.act)
            t[perm]=b.ar7_table;acts[perm]=b.act
            moved=predict(b.ar7_head,t,x,perm[rows],w,acts,b.ar7_meta)
            np.testing.assert_array_equal(a,moved)

    def test_fit_rejects_attribute_shadowing(self):
        cell=CELLS[0]
        source=load(artifact(self.rows[f'r9a7_{cell}_control']))['method'].base
        source.ar7_table = None
        with self.assertRaises(api.ContractError):
            RowCorrectedCache.from_corrected(source,HERE/'artifacts'/f'{cell}_capacity.npz','pi05_l10_cache')

    def test_prediction_before_emit_and_exact_controls(self):
        f=json.loads((HERE/'FROZEN.json').read_text())
        self.assertEqual(sha(HERE/'PREDICTION.md'),f['prediction_sha256'])
        self.assertLess((HERE/'PREDICTION.md').stat().st_mtime,(RUN/'arms.json').stat().st_mtime)
        for row in f['arms']:
            self.assertEqual(sha(row['artifact']),row['sha256'])
            if row['variant']=='control':
                self.assertEqual(row['sha256'],row['control_source_sha256'])
            else:
                blob=load(row['artifact'])
                self.assertEqual(sha(blob['kwargs']['residual_path']),blob['provenance']['head_sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
