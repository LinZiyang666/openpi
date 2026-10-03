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
from .data import HERE, RUN, STORE, CELLS, admitted, identity_arrays, sha, artifact
from .learn import train_assert
from .methods import RowCorrectedCache
from .numeric import residual


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


class FrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {a['arm']:a for a in json.loads((RUN/'arms.json').read_text())}

    def test_standard_store_manifest_and_identical_client(self):
        self.assertEqual(len(self.rows), 8)
        manifest = json.loads((RUN/'manifests/eval100.json').read_text())
        self.assertEqual({(r['task'],r['init']) for r in manifest['selected']},
                         {(t,i) for t in range(10) for i in range(20,30)})
        for cell in CELLS:
            c, n = [self.rows[f'r9a6_{cell}_{v}'] for v in ('control','taskfree')]
            self.assertEqual(c['client_overrides'], n['client_overrides'])
            self.assertEqual(c['kwargs'], {k:v for k,v in n['kwargs'].items() if k!='residual_path'})
            for a in (c,n):
                p = a['plugin_args']
                self.assertEqual(p[p.index('--os-root')+1], str(STORE))
                self.assertEqual(a['manifest'], str(RUN/'manifests/eval100.json'))
                self.assertEqual(a['judge'], 'guard_only')

    def test_only_corrector_changes_in_fitted_state(self):
        for cell in CELLS:
            c, n = [load(artifact(self.rows[f'r9a6_{cell}_{v}']))['method'] for v in ('control','taskfree')]
            self.assertEqual(set(vars(n))-set(vars(c)), {'ar6_residual_path'})
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
        for cell in CELLS:
            b = load(artifact(self.rows[f'r9a6_{cell}_taskfree']))['method'].base
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
            out = b._ar6_action(PoisonIdentity(), a, rows, np.ones(16))
            self.assertTrue(np.isfinite(out).all())
            self.assertGreater(float(np.max(np.abs(out[:10,:6]-a[:10,:6]))), 1e-7)
            np.testing.assert_array_equal(out[10:], a[10:])
            np.testing.assert_array_equal(out[:10,6:], a[:10,6:])

    def test_judge_os_synth_corrects_once_and_remembers_tail(self):
        for cell in CELLS:
            b = load(artifact(self.rows[f'r9a6_{cell}_taskfree']))['method'].base
            q = SimpleNamespace(key_v0=np.zeros(b.B0T.shape[1],np.float32),
                key_v1=np.zeros(b.B1T.shape[1],np.float32), rs=np.zeros(8,np.float32),
                task_id=0, step=4, episode=SimpleNamespace(uid='unit:0:0'))
            rows, w = np.arange(16), np.ones(16)
            a = np.array(b.act[0], copy=True)
            expected = b._ar6_action(q,a,rows,w)
            with patch.object(BlindAWM,'os_synth',return_value=a):
                out = b.os_synth(q,rows,w)
            np.testing.assert_array_equal(out,expected)
            np.testing.assert_array_equal(b._anchor['action'],expected)
            self.assertEqual(b._anchor['step'],4)

    def test_fit_rejects_attribute_shadowing(self):
        cell=CELLS[0]
        source=load(artifact(self.rows[f'r9a6_{cell}_control']))['method'].base
        source.ar6_table = None
        with self.assertRaises(api.ContractError):
            RowCorrectedCache.from_corrected(source,HERE/'artifacts'/f'{cell}.npz','pi05_l10_cache')

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
