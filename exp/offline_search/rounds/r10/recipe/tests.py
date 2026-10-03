"""Recipe-specific boundaries, whole episodes, folds, and artifact contracts."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from exp.offline_search.rounds.r10.data import STORE, RUNS, CELLS, SIZES
from .build import HERE, artifact_path, subset_kwargs
from .fitting import episode_selection, fit_path, library_boundary
from .recipe import ONLY_NO_PROGRESS, R10Recipe, _CorrectedCache


class Tests(unittest.TestCase):
    def test_fit_boundary_resolves_symlinks(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            link = Path(d) / 'innocent.json'
            link.symlink_to(RUNS / 'r10_size_pi05/eval500.json')
            for path in (link, RUNS / 'r10_corr3_pi05/fits/frozen.pkl',
                         STORE / 'queries/pi05_l10_cache/key_v0.npy', STORE / 'derived/r10/head.npz'):
                with self.assertRaises(ValueError):
                    fit_path(path, STORE)

    def test_actual_fit_read_boundary(self):
        directory = STORE / 'library/pi05_l10/current'
        with library_boundary(STORE, directory) as reads:
            (directory / 'episodes.json').read_bytes()
            for path in (STORE / 'library/pi05_l10/bpool_cs/episodes.json',
                         STORE / 'queries/pi05_l10_cache/episodes.json',
                         RUNS / 'r10_corr3_pi05/arms.json'):
                with self.assertRaises(ValueError):
                    path.read_bytes()
        self.assertEqual(reads, {str(directory / 'episodes.json')})

    def test_whole_episode_selection_and_order(self):
        ep = np.repeat(np.arange(60), 3)
        task = np.repeat(np.repeat(np.arange(10), 6), 3)
        chosen = {str(t): list(range(6*t, 6*t+5))[::-1] for t in range(10)}
        selected, rows = episode_selection(ep, task, chosen)
        self.assertEqual(selected, chosen)
        self.assertTrue(np.all(np.diff(rows) > 0))
        self.assertEqual(len(rows), 150)
        for episode in np.unique(ep[rows]):
            np.testing.assert_array_equal(rows[ep[rows] == episode], np.flatnonzero(ep == episode))
        bad = dict(chosen, **{'0': [6, 1, 2, 3, 4]})
        with self.assertRaises(ValueError):
            episode_selection(ep, task, bad)

    def test_all_r10_specs_are_inlined_and_bound(self):
        from exp.offline_search.rounds.r10.data import HERE as R10
        for model, suite in CELLS:
            for size in SIZES:
                kwargs = subset_kwargs(model, suite, R10 / 'subsets' / f'{model}_{suite}_{size}.json')
                self.assertEqual(set(kwargs), {'library', 'episode_subset'})
                self.assertEqual(sum(map(len, kwargs['episode_subset'].values())), size)
                R10Recipe(**kwargs)
        with self.assertRaises(ValueError):
            subset_kwargs('groot', 'l10', R10 / 'subsets/pi05_l10_50.json')

    def test_completed_artifact_contracts(self):
        paths = [(artifact_path(m, s, n), n) for m, s in CELLS for n in (*SIZES, None)]
        present = [(p, n) for p, n in paths if p.exists()]
        self.assertTrue(present, 'build at least one recipe before artifact-contract tests')
        for p, size in present:
            with p.open('rb') as f:
                blob = pickle.load(f)
            method = blob['method']
            info, base, guard = method.fit_info, method.inner.base, method.inner
            self.assertIsInstance(method, R10Recipe)
            self.assertIsInstance(base, _CorrectedCache)
            self.assertFalse(info['escalation'])
            self.assertTrue(info['library_only'])
            self.assertEqual(method.size, size or sum(map(len, info['episode_ids_by_task'].values())))
            self.assertEqual(base.kref, 5 if all(len(es) == 5 for es in info['episode_ids_by_task'].values()) else 8)
            self.assertEqual(tuple(guard.disabled_guards), ONLY_NO_PROGRESS)
            self.assertEqual(guard.burst, 0)
            self.assertEqual(base.serving, 'anchor_tail')
            self.assertEqual(base.budget, 1)
            self.assertEqual(base.blend, .5)
            self.assertEqual(base.chans, 6)
            self.assertEqual(set(base.heads), set(map(str, range(10))))
            for h in base.heads.values():
                self.assertEqual(h['coef'].shape, (60, 601))
                self.assertEqual(h['w'].shape, (217, 384))
            parent = Path(info['parent']).resolve()
            for path in info['fit_reads']:
                self.assertIn(parent, Path(path).resolve().parents)
            self.assertNotIn('source_fit', vars(method))
            self.assertNotIn('head_path', vars(base))
            self.assertNotIn('calibration_path', vars(method))
            self.assertEqual(guard.C.act.path, base.act.path)
            all_eps = set(e for ids in info['episode_ids_by_task'].values() for e in ids)
            held = []
            for fold in info['distance_folds']:
                tr, va = set(fold['train_episodes']), set(fold['heldout_episodes'])
                self.assertFalse(tr & va)
                self.assertEqual(tr | va, all_eps)
                held.extend(va)
            self.assertEqual(set(held), all_eps)
            self.assertEqual(len(held), len(all_eps))
            self.assertEqual(sum(r['rows'] for r in info['head_counts'].values()), len(method.row_subset))
            for counts in info['head_counts'].values():
                self.assertAlmostEqual(counts['weight_sum'], counts['rows'])


if __name__ == '__main__':
    unittest.main()
