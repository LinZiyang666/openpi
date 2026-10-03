"""Semantic tests: numeric serving alignment, strength, and hard data boundaries."""
import copy
import unittest
from pathlib import Path
import numpy as np

from . import offline
from .boundary import check_path, install, RUNS, HERE
from .method import multiplier
from exp.offline_search.rounds.r10 import train
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w


class Tests(unittest.TestCase):
    def test_boundaries(self):
        for p in (RUNS / 'r10_size_pi05/runs/x/summary.json', RUNS / 'r09_recipe_full_p/eval500.json',
                  RUNS / 'r10_corr2_groot/clients/foo.json'):
            with self.assertRaises(PermissionError): check_path(p)
        for p in (RUNS / 'r10_corr2_pi05/arms.json', RUNS / 'r10_size_groot/fits/x.pkl'):
            check_path(p)
        with self.assertRaises(PermissionError): check_path(HERE.parent / 'method.py', True)
        check_path(HERE / 'tests.py', True)

    def test_strength(self):
        for rule in offline.CANDIDATES:
            r = np.linspace(0, 10, 1000)
            s = np.array([multiplier(x, 1, rule) for x in r]) * .5
            self.assertTrue((np.diff(s) <= 0).all())
            np.testing.assert_allclose(s, offline.strength(r, np.ones(len(r)), rule))
            self.assertEqual(multiplier(0, 0, rule), 0)
            self.assertEqual(multiplier(np.nan, 1, rule), 0)
            self.assertEqual(multiplier(np.inf, 1, rule), 0)
            self.assertEqual(multiplier(rule['cutoff'], 1, rule), 0)
            self.assertEqual(multiplier(0, 1, rule), 2 * rule['peak'])

    def test_episode_weights(self):
        ep = np.array([0, 0, 1, 1, 1])
        self.assertAlmostEqual(offline.weights(ep)[ep == 0].sum(), 1)
        self.assertAlmostEqual(offline.weights(ep)[ep == 1].sum(), 1)

    def test_serving_alignment(self):
        errors = []
        for model, suite in (('pi05', 'l10'), ('groot', 'spatial')):
            C = offline.load(model, suite, 50)
            lib, base, feat, table = train.load_size(model, suite, 50)
            sig = offline.sigma(C, np.arange(len(C['ep'])))
            np.testing.assert_array_equal(sig, base.sig)
            for task in (0, 1):
                cr = np.flatnonzero(C['task'] == task)
                met = offline.Metric(C['X'][cr], C['act'][cr], C['ep'][cr], C['step'][cr], sig)
                np.testing.assert_allclose(met.Z, base.tasks[task].Z, atol=2e-4, rtol=2e-5)
                qs = list(train.queries(lib, table, task))
                chosen = [qs[0], qs[1], qs[len(qs)//2], qs[-1]]
                for i, q in chosen:
                    rows, ds, xv, rs = train.neighbors(base, q, int(C['ep'][i]))
                    base.reset(q.episode)
                    raw = base.os_synth(q, rows, _kernel_w(ds.astype(float) - float(ds[0]), base.kref))
                    d, chunks = offline.serve(C, np.array([i]), cr, met)
                    np.testing.assert_allclose(d, ds[:1], rtol=2e-4, atol=2e-4)
                    np.testing.assert_allclose(chunks[0], raw[:10, :7], rtol=3e-3, atol=3e-4)
                    actual_features = feat._features(q, xv, rs, raw)
                    vector_features = offline.features(C, np.array([i]), raw[None, :10, :7], sig)
                    np.testing.assert_allclose(actual_features, vector_features, rtol=2e-4, atol=2e-4)
                    errors.append(float(np.max(np.abs(chunks[0] - raw[:10, :7]))))
        print('Max vectorized-vs-serving chunk deviation:', max(errors))


if __name__ == '__main__':
    install()
    unittest.main()
