"""R10 semantic tests on real recorded B-library queries and small numeric cases."""
import copy
import json
import pickle
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r09.recipe import recipe
from .data import HERE, STORE, RUNS, assert_fit_input, IndexedRows
from . import train
from .train import load_size, queries, neighbors, feature_row, fit_head, task_dataset, anchor_weights, radius_for


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib, cls.base, cls.features, cls.table = load_size('pi05', 'spatial', 50)

    def test_fit_boundary(self):
        for p in [RUNS / 'r08_main/x', RUNS / 'r10_size_pi05/fits/x', STORE / 'derived/r08/x']:
            with self.assertRaises(ValueError): assert_fit_input(p)
        assert_fit_input(self.lib.dir, library=True)

    def test_indexed_pickle_and_slicing(self):
        a = IndexedRows(self.lib.dir / 'action.npy', self.lib.rows)
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNone(b._array)
        self.assertLess(len(pickle.dumps(b)), len(self.lib.rows) * 9 + 1000)
        np.testing.assert_array_equal(b[[0, 2], :10, :7], self.lib.action[[0, 2], :10, :7])
        np.testing.assert_array_equal(b[0, :10, :7], self.lib.action[0, :10, :7])

    def test_nested_episode_counts_and_failure_retention(self):
        for model in ('pi05', 'groot'):
            for suite in ('l10', 'spatial'):
                prev = set()
                for size in (50, 100, 200, 300, 400, 500):
                    path = HERE / 'subsets' / f'{model}_{suite}_{size}'
                    rows = set(np.load(path.with_suffix('.npy')).tolist())
                    meta = json.loads(path.with_suffix('.json').read_text())
                    self.assertTrue(prev <= rows)
                    self.assertTrue(all(len(e) == size // 10 for e in meta['episode_ids_by_task'].values()))
                    if size == 500: self.assertEqual(len(rows), meta['parent_rows'])
                    prev = rows

    def test_pseudo_queries_use_deployed_regime2(self):
        for task in (0, 1):
            regimes = set()
            for _, q in queries(self.lib, self.table, task):
                _, step, regime, *rest = self.base._dist(q)
                regimes.add(regime)
                if step > 0:
                    self.assertIs(q.prev_hit, True)
                    self.assertEqual(regime, 2)
                else:
                    self.assertEqual(regime, 0)
            self.assertEqual(regimes, {0, 2})

    def test_loeo_matches_deployed_restricted_cache(self):
        base = self.base
        for task in (0, 1):
            for i, q in list(queries(self.lib, self.table, task))[:4]:
                ep = int(self.lib.episode[i])
                rows, d, _, _ = neighbors(base, q, ep)
                self.assertTrue(np.all(base.lib_ep[rows] != ep))
                T = base.tasks[task]
                restricted = copy.copy(T)
                valid = base.lib_ep[T.rows] != ep
                for key in ('rows', 'Z', 'z2', 'HD', 'h2', 'V0', 'V1', 'RS', 'rs2', 'Z0', 'n20'):
                    a = getattr(T, key)
                    if a is not None: setattr(restricted, key, a[valid])
                base.tasks[task] = restricted
                try:
                    base.reset(q.episode)
                    result = base.query(q)
                finally:
                    base.tasks[task] = T
                np.testing.assert_array_equal(result.topk, rows)
                np.testing.assert_array_equal(-result.scores, d.astype(np.float64))
                base.reset(q.episode)
                kd = d.astype(np.float64)
                served = base.os_synth(q, rows, _kernel_w(kd - kd[0], base.kref))
                np.testing.assert_array_equal(result.action, served)

    def test_parent_mapping_covers_policy_tail_provenance(self):
        from exp.offline_search.closed_loop.blind import BlindResult
        from exp.offline_search.rounds.r10.method import SizeController
        from types import SimpleNamespace
        m = SizeController(500, 'G')
        m.parent_name = 'bpool_all'
        m.row_subset = np.asarray([11, 29, 8000], np.int64)
        raw = BlindResult(np.zeros((16, 32), np.float32), np.asarray([2], np.int64),
                          np.ones(1, np.float32), 'current', {})
        m.inner = SimpleNamespace(policy_tail_step=lambda q: raw)
        result = m.policy_tail_step(None)
        self.assertEqual(result.library, 'bpool_all')
        self.assertEqual(result.rows.tolist(), [8000])

    def test_features_byte_identical_to_actual_serving_correction(self):
        f = self.features
        f.heads = {'all': {}}
        for task in (0, 1):
            for i, q in list(queries(self.lib, self.table, task))[:4]:
                rows, d, xv, rs8 = neighbors(self.base, q, int(self.lib.episode[i]))
                self.base.reset(q.episode)
                kd = d.astype(np.float64)
                action = self.base.os_synth(q, rows, _kernel_w(kd - kd[0], self.base.kref))
                X = feature_row(f, q, xv, rs8, action)[None]
                captured = []
                def predict(h, x):
                    captured.append(x.copy())
                    return np.zeros((1, 60), np.float32)
                with patch.object(recipe, '_predict', predict):
                    f._correction(q, action)
                self.assertEqual(X.dtype, np.dtype('float32'))
                self.assertEqual(X.tobytes(), captured[0].tobytes())

    def test_weighted_ridge_matches_direct_equation(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(91, 17)).astype(np.float32)
        Y = rng.normal(size=(91, 60)).astype(np.float32)
        w = rng.uniform(.1, 3, 91)
        h = fit_head(X, Y, w, batch=13)
        xn = np.clip((X - h['mean']) / h['std'], -8, 8)
        F = np.concatenate([xn, np.cos(xn @ h['w'] + h['bias']) * np.sqrt(2)], 1).astype(np.float64)
        # The solver must preserve supplied mass, including mass != n_pairs.
        sw = np.sqrt(w)[:, None]
        Fw, Yw = F * sw, Y.astype(np.float64) * sw
        mf = (Fw * sw).sum(0) / (sw ** 2).sum()
        my = (Yw * sw).sum(0) / (sw ** 2).sum()
        fc, yc = Fw - mf * sw, Yw - my * sw
        coef = np.linalg.solve(fc.T @ fc + 100 * np.eye(F.shape[1]), fc.T @ yc).T
        np.testing.assert_allclose(h['coef'], coef, atol=2e-7, rtol=2e-5)
        np.testing.assert_allclose(h['intercept'], my - coef @ mf, atol=2e-7, rtol=2e-5)

    def test_episode_weights_and_cross_episode_pairs(self):
        d = task_dataset(self.lib, self.base, self.features, self.table, 0, float('inf'))
        for variant, data in d.items():
            n_anchor = len(np.unique(data['row']))
            self.assertAlmostEqual(float(data['weight'].sum()), n_anchor)
            for e in np.unique(data['ep']):
                self.assertAlmostEqual(float(data['weight'][data['ep'] == e].sum()),
                                       n_anchor / len(np.unique(data['ep'])))
            if variant == 'pair': self.assertTrue(np.all(data['ep'] != data['donor']))
        # Each anchor's PAIR mass equals its LOEO row mass, irrespective of cap.
        pair = d['pair']
        for row, w in zip(d['loeo']['row'], d['loeo']['weight']):
            self.assertAlmostEqual(float(pair['weight'][pair['row'] == row].sum()), w)

    def test_pair_weights_count_anchors_with_unequal_pair_counts(self):
        rows = np.asarray([3, 3, 4, 7, 7, 7])
        ep = np.asarray([0, 0, 0, 1, 1, 1])
        w = anchor_weights(rows, ep)
        self.assertAlmostEqual(w.sum(), 3.)
        self.assertAlmostEqual(w[ep == 0].sum(), 1.5)
        self.assertAlmostEqual(w[ep == 1].sum(), 1.5)
        self.assertAlmostEqual(w[rows == 3].sum(), .75)
        self.assertAlmostEqual(w[rows == 4].sum(), .75)
        self.assertAlmostEqual(w[rows == 7].sum(), 1.5)

    def test_pair_multiplicity_preserves_ridge_strength(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(31, 17)).astype(np.float32)
        Y = rng.normal(size=(31, 60)).astype(np.float32)
        row_head = fit_head(X, Y, np.ones(31))
        pair_head = fit_head(np.repeat(X, 16, axis=0), np.repeat(Y, 16, axis=0),
                             np.full(31 * 16, 1. / 16))
        np.testing.assert_allclose(pair_head['coef'], row_head['coef'], atol=1e-6, rtol=1e-4)
        np.testing.assert_allclose(pair_head['intercept'], row_head['intercept'], atol=1e-6, rtol=1e-4)

    def test_radius_is_per_task_median_loeo16(self):
        radii, distances = radius_for(self.lib, self.base, self.table, tasks=(0, 1))
        self.assertEqual(train.PAIR_CAP, 16)
        self.assertEqual(train.RADIUS_QUANTILE, .5)
        for task in (0, 1):
            expected = [float(neighbors(self.base, q, int(self.lib.episode[i]), cap=16)[1][-1])
                        for i, q in queries(self.lib, self.table, task)]
            self.assertEqual(distances[task], expected)
            self.assertEqual(radii[task], float(np.median(expected)))
        self.assertNotEqual(radii[0], radii[1])
        d = task_dataset(self.lib, self.base, self.features, self.table, 0, radii[0])['pair']
        rows, counts = np.unique(d['row'], return_counts=True)
        self.assertTrue(np.all(counts <= 16))
        for i, q in queries(self.lib, self.table, 0):
            if i not in rows: continue
            donors, scores, _, _ = neighbors(self.base, q, int(self.lib.episode[i]))
            n = int(np.sum(d['row'] == i))
            self.assertEqual(n, int(np.sum(scores[:16] <= radii[0])))
            np.testing.assert_array_equal(d['donor'][d['row'] == i],
                                          self.lib.episode[donors[:16][scores[:16] <= radii[0]]])

    def test_fold_donors_exclude_heldout_episodes(self):
        episodes = np.unique(self.lib.episode[self.lib.task_id == 0])
        allowed = set(episodes[:3])
        for i, q in list(queries(self.lib, self.table, 0))[-4:]:
            rows, _, _, _ = neighbors(self.base, q, int(self.lib.episode[i]), allowed)
            self.assertTrue(set(self.lib.episode[rows]).issubset(allowed))
            self.assertTrue(np.all(self.lib.episode[rows] != self.lib.episode[i]))

    def test_per_task_head_shapes(self):
        paths = list(train.HEADS.glob('*.npz'))
        if len(paths) != 48:
            self.skipTest('all 48 Stage 2b heads not yet published')
        for path in paths:
            heads, meta = recipe.load_head(path)
            self.assertEqual(meta['revision'], train.REVISION)
            self.assertEqual(set(heads), set(map(str, range(10))))
            self.assertEqual(meta['chans'], 6)
            for h in heads.values():
                self.assertEqual(h['coef'].shape, (60, 601))
                self.assertEqual(h['w'].shape, (217, 384))


if __name__ == '__main__':
    unittest.main()
