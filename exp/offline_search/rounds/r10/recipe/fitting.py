"""Library-only builder math; no frozen fit or test-result inputs."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r08.abl.judge import TriggerCommitJudge, TriggerGrootCommitJudge
from exp.offline_search.rounds.r10.data import IndexedRows, SEED, RUNS, sha
from .recipe import _CorrectedCache, ONLY_NO_PROGRESS, DISTANCE_RULE

_BOUNDARY = None
_HOOK_INSTALLED = False


def fit_path(path, root, library=None):
    p, root = Path(path).resolve(), Path(root).resolve()
    if RUNS == p or RUNS in p.parents or 'os_closed_loop' in p.parts:
        raise ValueError(f'closed-loop data refused for fitting: {p}')
    if library is not None:
        lib = Path(library).resolve()
        if lib != p and lib not in p.parents:
            raise ValueError(f'fit input is outside selected library: {p}')
    if root / 'queries' == p or root / 'queries' in p.parents or 'derived' in p.parts:
        raise ValueError(f'query/derived data refused for fitting: {p}')
    return p


@contextmanager
def library_boundary(root, library):
    """Audit every fit read, including mmap opens, and fail closed on A data."""
    global _BOUNDARY, _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        def hook(event, args):
            if event != 'open' or _BOUNDARY is None or isinstance(args[0], int):
                return
            path, _, flags = args
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                return
            p = Path(os.fsdecode(path)).resolve()
            root, lib, reads = _BOUNDARY
            fit_path(p, root)
            if p == root or root in p.parents:
                fit_path(p, root, lib)
                reads.add(str(p))
        sys.addaudithook(hook)
        _HOOK_INSTALLED = True
    if _BOUNDARY is not None:
        raise RuntimeError('fit in separate processes, not overlapping threads')
    reads = set()
    _BOUNDARY = (Path(root).resolve(), Path(library).resolve(), reads)
    try:
        yield reads
    finally:
        _BOUNDARY = None


def episode_selection(episode, task, selected=None):
    if set(np.unique(task)) != set(range(10)):
        raise ValueError('recipe requires the ten suite tasks')
    if selected is not None and set(selected) != set(map(str, range(10))):
        raise ValueError('episode subset must specify all ten tasks')
    chosen = {}
    for t in range(10):
        ids = np.unique(episode[task == t])
        vals = np.random.default_rng(SEED + t).permutation(ids).tolist() if selected is None else selected[str(t)]
        if any(not isinstance(e, (int, np.integer)) or isinstance(e, bool) for e in vals):
            raise ValueError('episode IDs must be integers')
        if len(vals) < 3 or len(vals) != len(set(vals)) or not set(vals) <= set(ids):
            raise ValueError(f'task {t}: need at least three distinct episodes belonging to this task')
        chosen[str(t)] = list(map(int, vals))
    flat = [e for ids in chosen.values() for e in ids]
    if len(flat) != len(set(flat)):
        raise ValueError('episode IDs must be globally unique across tasks')
    rows = np.flatnonzero(np.isin(episode, flat))
    return chosen, rows


class _Library:
    """A compact view of whole selected episodes, with local predecessor IDs."""
    def __init__(self, root, key, name, selected):
        self.parent = store.LibraryView(root, key, name)
        self.root, self.key, self.dir = self.parent.root, key, self.parent.dir
        fit_path(self.dir, root, Path(root) / 'library' / key / name)
        if not (Path(root).resolve() / 'library') in self.dir.resolve().parents:
            raise ValueError('selected library must resolve inside the library store')
        self.model, self.suite = key.split('_')
        self.name = 'current'  # both candidate and fit routing use this same view
        self._a = {}
        self.chosen, self.rows = episode_selection(self.parent.episode, self.parent.task_id, selected)
        self.rows.flags.writeable = False
        self.L, self.H = len(self.rows), self.parent.H
        ids = set(e for es in self.chosen.values() for e in es)
        self.episodes = [e for e in self.parent.episodes if int(self.parent.episode[int(e['start'])]) in ids]

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._a:
            path = fit_path(store._npy_path(self.dir, name), self.root, self.dir)
            if name.startswith('key_'):
                a = IndexedRows(path, self.rows)
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


class _Context(api.Context):
    def __init__(self, ctx, lib):
        vars(self).update(vars(ctx))
        self.selected = lib
        self._sigma = np.maximum(np.asarray(lib.action[:, :5, :7]).std(axis=(0, 1)), 1e-6).astype(np.float32)

    @property
    def action_sigma(self):
        return self._sigma

    def open_library(self, name='current'):
        if name != 'current':
            raise ValueError('fitting can only open the selected episode view')
        return self.selected


def loeo_heads(lib, base, table):
    from exp.offline_search.rounds.r10 import train
    feature = _CorrectedCache.__new__(_CorrectedCache)
    vars(feature).update(vars(base))
    feature.sig_head = base.sig.copy()
    heads, counts = {}, {}
    for task in range(10):
        X, Y, episodes, rows = [], [], [], []
        for i, q in train.queries(lib, table, task):
            ep = int(lib.episode[i])
            donors, distances, xv, rs8 = train.neighbors(base, q, ep, cap=16)
            base.reset(q.episode)
            kd = distances.astype(np.float64)
            cached = base.os_synth(q, donors, awm._kernel_w(kd - kd[0], base.kref))
            X.append(feature._features(q, xv, rs8, cached)[0])
            target = np.asarray(lib.action[i, :10, :6], np.float32)
            Y.append(((target - cached[:10, :6]) / base.sig[:6]).ravel())
            episodes.append(ep); rows.append(i)
        X, Y = np.asarray(X, np.float32), np.asarray(Y, np.float32)
        weights = train.anchor_weights(np.asarray(rows), np.asarray(episodes))
        heads[str(task)] = train.fit_head(X, Y, weights)
        counts[str(task)] = dict(rows=len(rows), episodes=len(set(episodes)), weight_sum=float(weights.sum()))
        print(f'HEAD task={task} rows={len(rows)}', flush=True)
    for h in heads.values():
        for v in h.values():
            v.flags.writeable = False
    return heads, counts


def distance_scale(lib, projections, kref):
    # Reproduce astra's final outer-fold calibration, with the same fixed PCA,
    # fold assignment, training-only sigma/metric and balanced-median arithmetic.
    from exp.offline_search.rounds.r10.astra.offline import Metric, median, serve, sigma
    ef = {e: i % 5 for ids in lib.chosen.values() for i, e in enumerate(ids)}
    P = np.concatenate([projections['v0'], projections['v1']], 1).astype(np.float32)
    C = dict(P=P, X=np.concatenate([P, lib.rs[:, :8]], 1).astype(np.float32),
             act=np.asarray(lib.action[:, :10, :7], np.float32), ep=np.array(lib.episode),
             step=np.array(lib.step), task=np.array(lib.task_id), kref=kref)
    folds = np.asarray([ef[int(e)] for e in C['ep']])
    distances, episodes, evidence = [], [], []
    for f in range(5):
        tr, va = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        sig = sigma(C, tr)
        for task in range(10):
            cr = tr[C['task'][tr] == task]
            # Include step 0 in the exact frozen vectorized batch boundaries,
            # then discard it before the balanced median.
            qr = va[C['task'][va] == task]
            if not len(qr):
                continue  # fewer than five episodes: this task has no query in this fold
            met = Metric(C['X'][cr], C['act'][cr], C['ep'][cr], C['step'][cr], sig)
            d, _ = serve(C, qr, cr, met)
            keep = C['step'][qr] > 0
            distances.append(d[keep]); episodes.append(C['ep'][qr][keep])
        evidence.append(dict(fold=f, train_episodes=np.unique(C['ep'][tr]).tolist(),
                             heldout_episodes=np.unique(C['ep'][va]).tolist()))
        print(f'DISTANCE fold={f}', flush=True)
    value = median(np.concatenate(distances), np.concatenate(episodes))
    if not np.isfinite(value) or value <= 0:
        raise ValueError('invalid library-held-out distance scale')
    return value, evidence


def fit_recipe(recipe, ctx):
    directory = Path(ctx.root) / 'library' / ctx.lib_key / recipe.library
    fit_path(directory, ctx.root)
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    with library_boundary(ctx.root, directory) as reads:
        lib = _Library(ctx.root, ctx.lib_key, recipe.library, recipe.episode_subset)
        sc = _Context(ctx, lib)
        kref = 5 if all(len(es) == 5 for es in lib.chosen.values()) else 8
        cls = TriggerCommitJudge if ctx.model == 'pi05' else TriggerGrootCommitJudge
        judge = cls(base_kwargs=dict(lib='current', kref=kref, serving='anchor_tail', budget=1, gates='budget_only'),
                    progress_guard='noprog_span', events='none', stuck_guard='vision_confirmed',
                    policy_tail_gate='lifecycle', monitor='off', disabled_guards=ONLY_NO_PROGRESS)
        judge.prof = api.NULL_PROFILER
        judge.base.lib = judge.base.fit_src = 'current'
        projections = {}
        def fresh_pca(L, key, field):
            print(f'PCA {ctx.lib_key}/{recipe.library} {field} rows={lib.L}', flush=True)
            result = awm.pca_fit(getattr(L, f'key_{field}'))
            projections[field] = result[2]
            return result
        with patch.object(awm, 'pca_current', fresh_pca):
            judge.fit(lib, sc)
        print(f'GUARD {ctx.lib_key} kref={kref}', flush=True)
        base = judge.base
        heads, counts = loeo_heads(lib, base, judge.C)
        scale, folds = distance_scale(lib, projections, kref)
        corrected = _CorrectedCache.__new__(_CorrectedCache)
        vars(corrected).update(vars(base))
        corrected.heads = heads
        corrected.sig_head = base.sig.copy()
        corrected.head_meta = dict(task_onehot=True, chans=6, sigma=base.sig.tolist(), library_only=True)
        corrected.blend, corrected.chans, corrected.correct_gripper = .5, 6, False
        corrected.distance_scale, corrected.distance_rule = scale, dict(DISTANCE_RULE)
        corrected.act = IndexedRows(store._npy_path(lib.dir, 'action'), lib.rows)
        corrected.invalidate_anchor()
        judge.base, judge.C.act = corrected, corrected.act
        # Reset training-time memos before publishing the deployment object.
        judge.reset(api.EpisodeView('recipe-build', '', 0, 0, 0, 0))
        recipe.inner = judge
        recipe.row_subset = lib.rows.copy()
        recipe.row_subset.flags.writeable = False
        recipe.size = sum(len(es) for es in lib.chosen.values())
        recipe.name = f'R10Recipe_{recipe.library}_{recipe.size}__{judge.name}'
        recipe.fit_info = dict(library_only=True, fit_version=1, library=recipe.library,
            parent=str(lib.dir), episodes=recipe.size, rows=lib.L, episode_ids_by_task=lib.chosen,
            row_subset_sha256=hashlib.sha256(recipe.row_subset.tobytes()).hexdigest(),
            episodes_sha256=sha(lib.dir / 'episodes.json'), kref=kref,
            pca='fresh selected-library randomized SVD seed0', guard='R8 library LOEO only-no-progress',
            loeo_revision='stage2b_regime2_anchor_mass_task_median16', head_counts=counts,
            head_parameters=dict(rff=384, alpha=100., seed=0, channels=6, steps=10),
            distance_scale=scale, distance_rule=dict(DISTANCE_RULE), distance_folds=folds,
            distance_reference='episode-balanced median of non-step0 five-fold held-out d1; fixed library PCA',
            escalation=False, fit_reads=sorted(reads))
