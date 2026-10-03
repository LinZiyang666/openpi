"""Fit inputs are B libraries only; A metadata belongs solely to evaluation specs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from exp.offline_search.harness import api, store

HERE = Path(__file__).resolve().parent
STORE = Path('/home/weiland/trace_runs/offline_search_store')
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
SIZES = (50, 100, 200, 300, 400, 500)
CELLS = tuple((m, s) for m in ('pi05', 'groot') for s in ('l10', 'spatial'))
PARENT = {'pi05': 'bpool_cs', 'groot': 'bpool_all'}
SEED = 20261002


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def assert_fit_input(path, *, library=False):
    """Resolve symlinks before enforcing the hard test-set boundary."""
    p = Path(path).resolve()
    if p == RUNS or RUNS in p.parents:
        raise ValueError(f'A/test closed-loop path refused for fitting: {p}')
    if 'derived' in p.parts and ('r08' in p.parts or 'r8' in p.parts):
        raise ValueError(f'R8-derived fit input refused: {p}')
    if library and not (STORE / 'library') in p.parents:
        raise ValueError(f'fit raw data must be in offline_search_store/library: {p}')
    return p


class IndexedRows:
    """Lazy row gather over a parent mmap; pickle stores filename and indices only."""
    def __init__(self, path, rows):
        self.path = str(path)
        self.rows = np.asarray(rows, np.int64)
        self.rows.flags.writeable = False
        a = np.load(self.path, mmap_mode='r')
        self.shape = (len(self.rows), *a.shape[1:])
        self.dtype = a.dtype
        self.ndim = len(self.shape)
        self._array = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode='r')
        if isinstance(item, tuple):
            first, *rest = item
        else:
            first, rest = item, []
        out = self._array[self.rows[first]]
        if rest:
            out = out[(slice(None), *rest)] if np.ndim(self.rows[first]) else out[tuple(rest)]
        return out

    def __array__(self, dtype=None, copy=None):
        out = self[:]
        return np.asarray(out, dtype=dtype)

    def __getstate__(self):
        return {k: v for k, v in vars(self).items() if k != '_array'}

    def __setstate__(self, state):
        vars(self).update(state)
        self._array = None


class SubsetLibrary:
    def __init__(self, root, key, size, rows=None):
        model, suite = key.split('_')
        parent = store.LibraryView(root, key, PARENT[model])
        assert_fit_input(parent.dir, library=True)
        self.parent = parent
        self.dir, self.root, self.key = parent.dir, parent.root, key
        self.model, self.suite, self.name = model, suite, 'current'
        if rows is None:
            spec = json.loads((HERE / 'subsets' / f'{key}_{size}.json').read_text())
            if sha(parent.dir / 'episodes.json') != spec['episodes_sha256']:
                raise ValueError('parent episode metadata changed')
            rows = np.load(HERE / 'subsets' / f'{key}_{size}.npy', allow_pickle=False)
        self.rows = np.asarray(rows, np.int64)
        if not np.all(np.diff(self.rows) > 0) or self.rows.min() < 0 or self.rows.max() >= parent.L:
            raise ValueError('subset indices must be sorted, unique and in parent range')
        self.L, self.H = len(self.rows), parent.H
        self._a = {}
        self.episodes = [e for e in parent.episodes if np.isin(self.rows, np.arange(e['start'], e['end'])).any()]

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._a:
            assert_fit_input(store._npy_path(self.dir, name), library=True)
            if name.startswith('key_'):
                a = IndexedRows(store._npy_path(self.dir, name), self.rows)
            else:
                a = np.array(getattr(self.parent, name)[self.rows])
                if name in ('prev', 'next'):
                    inv = np.full(self.parent.L, -1, np.int32)
                    inv[self.rows] = np.arange(self.L)
                    a = np.where(a >= 0, inv[np.maximum(a, 0)], -1)
                a.flags.writeable = False
            self._a[name] = a
        return self._a[name]

    def tasks(self):
        return np.unique(self.task_id)

    def rows_of_task(self, task):
        return np.flatnonzero(self.task_id == task)

    def has(self, name):
        return self.parent.has(name)


class SubsetContext(api.Context):
    def __init__(self, ctx, lib):
        vars(self).update(vars(ctx))
        self.subset = lib
        self._sigma = np.maximum(np.asarray(lib.action[:, :5, :7]).std(axis=(0, 1)), 1e-6).astype(np.float32)

    @property
    def action_sigma(self):
        return self._sigma

    def open_library(self, name='current'):
        if name not in ('current', PARENT[self.model]):
            raise ValueError(f'R10 fit cannot open other library: {name}')
        return self.subset


def make_subsets():
    for model, suite in CELLS:
        key = f'{model}_{suite}'
        lib = store.LibraryView(STORE, key, PARENT[model])
        assert_fit_input(lib.dir, library=True)
        ep = np.asarray(lib.episode)
        task = np.asarray(lib.task_id)
        if len(np.unique(ep)) != 500:
            raise ValueError(f'{key}: expected 500 globally unique parent episode IDs')
        permutations = {}
        for t in range(10):
            ids = np.unique(ep[task == t])
            if len(ids) != 50:
                raise ValueError(f'{key}: task {t} has {len(ids)} episodes')
            permutations[t] = np.random.default_rng(SEED + t).permutation(ids).tolist()
        previous = set()
        for size in SIZES:
            chosen = {t: p[:size // 10] for t, p in permutations.items()}
            rows = np.flatnonzero(np.isin(ep, np.concatenate(list(chosen.values()))))
            for t in range(10):
                assert set(np.unique(ep[rows][task[rows] == t])) == set(chosen[t])
            assert previous.issubset(set(rows))
            previous = set(rows)
            if size == 500:
                assert np.array_equal(rows, np.arange(lib.L))
            d = HERE / 'subsets'
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f'{key}_{size}.npy', rows, allow_pickle=False)
            write_json(d / f'{key}_{size}.json', dict(model=model, suite=suite, size=size, seed=SEED,
                parent=str(lib.dir), parent_rows=lib.L, rows=len(rows), episode_ids_by_task=chosen,
                permutation_by_task=permutations, episodes_sha256=sha(lib.dir / 'episodes.json'),
                failed_episodes=sum(not e['success'] for e in lib.episodes if int(e['start']) in previous),
                rows_sha256=sha(d / f'{key}_{size}.npy'), fit_pool='B', evaluation_pool='A; never fit'))
