"""Permanent bitwise parity tests against the frozen pre-optimization recipe."""
import copy
import itertools
from pathlib import Path
import pickle
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import stall, stall_reference as reference


def assert_identical(a, b, where=''):
    """Equality including float bits, nested keys, order and ndarray bytes."""
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (where, a.keys(), b.keys())
        for k in a:
            assert_identical(a[k], b[k], f'{where}.{k}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), (where, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_identical(x, y, f'{where}[{i}]')
    elif isinstance(a, np.ndarray):
        assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), where
    elif isinstance(a, (float, np.floating)):
        assert struct.pack('d', a) == struct.pack('d', b), (where, a, b, abs(a-b))
    else:
        assert a == b, (where, a, b)


def alignment_details(model, task, window, span, exclude=None):
    """Reference instrumentation independent of the optimized code."""
    data = model.tasks[task]
    candidates, paths, distances = [], [], []
    for order, t in enumerate(data['templates']):
        dd = []
        for z, regime in window:
            diff = t['codes'][regime] - z
            dd.append(np.sqrt(np.einsum('ij,ij->i', diff, diff)))
        dd = np.asarray(dd)
        mean, path = reference.monotone_alignment(dd)
        paths.append(path)
        distances.append(dd)
        if t['episode'] != exclude:
            candidates.append((mean, order))
    best = [v[1] for v in sorted(candidates)[:data['K']]]
    return best, paths, distances


def check_alignment(model, task, window, span, exclude=None):
    ds = np.asarray([model._distances(task, v) for v in window])
    got, best, paths = model._estimate_distances(task, ds, span, exclude, details=True)
    old = model._estimate_reference(task, window, span, exclude)
    assert_identical(got, old, 'estimate')
    selected, old_paths, old_distances = alignment_details(model, task, window, span, exclude)
    assert best.tolist() == selected, (best, selected)
    for i, t in enumerate(model.tasks[task]['templates']):
        n = len(t['phase'])
        assert_identical(ds[:, i, :n], old_distances[i], f'distances[{i}]')
        assert_identical(paths[:, i], old_paths[i], f'paths[{i}]')
    return len(old_paths)


def synthetic(seed, W, dims):
    rng = np.random.default_rng(seed)
    E = int(rng.integers(4, 10))
    lengths = rng.integers(12, 19, E) if W == 2 else rng.integers(86, 100, E)
    ep = np.repeat(np.arange(E), lengths)
    step = np.concatenate([np.arange(n) for n in lengths])
    # Integer-valued codes deliberately create point/path/template ties.
    codes = rng.integers(-2, 3, (len(ep), dims)).astype(float)
    offsets = np.r_[0, np.cumsum(lengths)]
    codes[offsets[1]:offsets[1]+min(lengths[0], lengths[1])] = codes[:min(lengths[0], lengths[1])]
    early_dims = dims+1
    early = rng.integers(-1, 2, (len(ep), early_dims)).astype(float)
    library = dict(manifest=dict(H=11, exec_steps=3, task_map={'task': 0}),
        task_id=np.zeros(len(ep), int), episode=ep, step=step, success=np.ones(len(ep), bool))
    metric = dict(tasks={'0': dict(rows=np.arange(len(ep)), codes={'main': codes, 'early': early})})
    return library, metric, rng


class FastEquivalenceTests(unittest.TestCase):
    def test_batched_dp_exhaustive_ties_and_variable_lengths(self):
        rng = np.random.default_rng(921)
        comparisons = 0
        for w in range(1, 6):
            for n in range(1, 7):
                lengths = np.array([1, n, max(1, n-1), n])
                ds = np.full((w, len(lengths), n), np.inf)
                for i, length in enumerate(lengths):
                    ds[:, i, :length] = rng.integers(0, 4, (w, length))
                costs, paths = stall._monotone_alignment_batch(ds, np.arange(n))
                for i, length in enumerate(lengths):
                    d = ds[:, i, :length]
                    cost, path = reference.monotone_alignment(d)
                    assert_identical(float(costs[i]), cost)
                    assert_identical(paths[:, i], path)
                    all_paths = itertools.combinations_with_replacement(range(length), w)
                    best = min(all_paths, key=lambda p: (sum(d[t, j] for t, j in enumerate(p)), p))
                    self.assertEqual(tuple(path), best)
                    comparisons += 1
        self.assertEqual(comparisons, 120)

    def test_randomized_fits_streams_and_exclusions(self):
        for W in (2, 3):
            for dims in (1, 2, 3, 7, 16, 17, 136):
                lib, metric, rng = synthetic(1700+W*100+dims, W, dims)
                model = stall.StallModel.fit(lib, metric=metric, commit_controls=6)
                old = reference.StallModel.fit(lib, metric=metric, commit_controls=6)
                self.assertEqual(model.tasks['0']['W'], W)
                self.assertEqual(model.fingerprint, old.fingerprint)
                assert_identical(model._payload(), old._payload())
                fast, slow = stall.StallTracker(model, 'task'), reference.StallTracker(old, 0)
                control = 0
                for i in range(45):
                    control += int(rng.integers(1, 14))
                    regime = 'early' if i % 5 == 0 else 'main'
                    dim = dims+1 if regime == 'early' else dims
                    code = rng.integers(-2, 3, dim).astype(float)
                    if i % 4 == 0: code += rng.normal(size=dim)
                    key = dict(metric_code=code, metric=regime)
                    if i in (7, 8, 21): key = [np.nan]*dims
                    fast.observe(key, control); slow.observe(key, control)
                    assert_identical(fast.status(), slow.status(), f'W{W}/d{dims}/anchor{i}')
                    if len(fast._window) == W+1:
                        window = [v[0] for v in fast._window]
                        span = fast._window[-1][1]-fast._window[0][1]
                        check_alignment(model, '0', window, span)
                        check_alignment(model, '0', window, span, exclude=model.tasks['0']['templates'][0]['episode'])

    def test_nearest_context_ties_and_lower_inverse_ecdf(self):
        lib, metric, _ = synthetic(400, 2, 3)
        model = stall.StallModel.fit(lib, metric=metric, commit_controls=6)
        for i, r in enumerate(model.tasks['0']['references']):
            r['context'] = np.zeros_like(r['context'])
            r['residual'] = np.arange(len(r['context']), dtype=float)+i
            r['advance'] = np.arange(len(r['context']), dtype=float)+i+.25
        pred = dict(delta_hat=.1, phase_hat=.2, distance_hat=.3, spread_hat=.4)
        got = model.calibrated_status('0', pred, 12)
        assert_identical(got, model.calibrated_status_reference('0', pred, 12))
        self.assertEqual(got['reference_windows'], [[r['episode'], int(r['end_control'][0])] for r in model.tasks['0']['references']])

    def test_cache_is_episode_local_and_invalid_observation_clears_it(self):
        lib, metric, _ = synthetic(401, 2, 3)
        model = stall.StallModel.fit(lib, metric=metric, commit_controls=6)
        a, b = stall.StallTracker(model, 0), stall.StallTracker(model, 0)
        with patch.object(model, '_distances', wraps=model._distances) as distance:
            for i in range(8):
                a.observe([i, 0, 1], i)
                self.assertEqual(distance.call_count, 0 if i < 2 else i+1)
            b.observe([1, 2, 3], 0)
            self.assertEqual(distance.call_count, 8)
            a.observe([np.inf, 0, 1], 8)
            self.assertEqual(len(a._distance_window), 0)
            for i in range(9, 12): a.observe([i, 0, 1], i)
            self.assertEqual(distance.call_count, 11)
            self.assertEqual(len(b._distance_window), 0)
        z = np.array([1., 2., 3.]); a.observe(z, 12); z[:] = 99
        self.assertFalse(a._window[-1][0][0].flags.writeable)
        self.assertEqual(a._window[-1][0][0].tolist(), [1, 2, 3])

    def test_public_rejections_and_recovery_match_reference(self):
        lib, metric, _ = synthetic(402, 2, 3)
        model = stall.StallModel.fit(lib, metric=metric, commit_controls=6)
        old = copy.copy(model); old.__class__ = reference.StallModel
        for task in (0, 'unknown'):
            a, b = stall.StallTracker(model, task), reference.StallTracker(old, task)
            for key, control in [([0, 0, 0], 0), ([1, 2], 1), ({'code': [0, 0, 0], 'metric': 'bad'}, 2),
                ({}, 3), (object(), 4), ([[1, 2, 3]], 5), ([0, np.nan, 1], 6), ([0, 0, 0], 7),
                ([1, 0, 0], 8), ([2, 0, 0], 9), ([0, 0, 0], 9), ([0, 0, 0], -1),
                ([0, 0, 0], True), ([0, 0, 0], 10.5), ([3, 0, 0], np.int64(10)),
                ([4, 0, 0], 10**30), ([5, 0, 0], 10**30+5)]:
                outcomes = []
                for tr in (a, b):
                    try: tr.observe(key, control); outcomes.append(None)
                    except Exception as e: outcomes.append(type(e))
                self.assertEqual(outcomes[0], outcomes[1], (key, control, outcomes))
                assert_identical(a.status(), b.status())
        a, b = stall.StallTracker(model, 0), reference.StallTracker(old, 0)
        with np.errstate(over='ignore'):
            for i, z in enumerate(([1e308, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0])):
                outcomes = []
                for tr in (a, b):
                    try: tr.observe(z, i); outcomes.append(None)
                    except Exception as e: outcomes.append(type(e))
                self.assertEqual(outcomes[0], outcomes[1])
                assert_identical(a.status(), b.status())
        for bad in ([], [[np.nan]], [[np.inf]], np.empty((0, 2)), [1, 2]):
            for function in (stall.monotone_alignment, reference.monotone_alignment):
                with self.assertRaises(ValueError): function(bad)

    def test_real_50_and_500_episode_banks(self):
        for cell in ('pi05_l10_50', 'pi05_l10_500'):
            root = Path('/tmp/q3_stall_fits')/cell
            self.assertTrue(root.is_dir(), root)
            model = stall.StallModel.load(root); old = reference.StallModel.load(root)
            for task in ('0', '8'):
                a, b = stall.StallTracker(model, task), reference.StallTracker(old, task)
                template = model.tasks[task]['templates'][0]
                for i in range(min(14, len(template['phase']))):
                    regime = 'early' if i == 0 else 'main'
                    key = dict(metric_code=template['codes'][regime][i], metric=regime)
                    control = int(template['controls'][i])
                    a.observe(key, control); b.observe(key, control)
                    assert_identical(a.status(), b.status(), f'{cell}/{task}/{i}')
                    if len(a._window) == model.tasks[task]['W']+1:
                        check_alignment(model, task, [v[0] for v in a._window], control-a._window[0][1])

    def test_derived_caches_do_not_change_serialized_artifact(self):
        lib, metric, _ = synthetic(403, 2, 3)
        model = stall.StallModel.fit(lib, metric=metric, commit_controls=6)
        payload = pickle.dumps(model._payload(), protocol=4)
        tracker = stall.StallTracker(model, 0)
        for i in range(5): tracker.observe([i, 0, 1], i)
        self.assertEqual(pickle.dumps(model._payload(), protocol=4), payload)
        scratch = Path('/tmp/codex_stall_fast'); scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='stall_fast_', dir=scratch) as path:
            model.save(path)
            self.assertEqual((Path(path)/'stall.pkl').read_bytes(), payload)
            with self.assertRaises(FileExistsError): model.save(path)
            self.assertEqual(stall.StallModel.load(path).fingerprint, model.fingerprint)


if __name__ == '__main__': unittest.main()
