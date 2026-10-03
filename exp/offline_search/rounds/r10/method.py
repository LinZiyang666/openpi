"""Subset adapter: R4/R8 algorithms unchanged, local IDs mapped to parent IDs."""
from __future__ import annotations

import pickle
import fcntl
import json
from unittest.mock import patch
import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult, LookReason
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r08.abl.judge import TriggerCommitJudge, TriggerGrootCommitJudge
from exp.offline_search.rounds.r09.recipe.recipe import RecipeCorrectedBase, load_head
from .data import HERE, PARENT, SubsetLibrary, SubsetContext, IndexedRows, assert_fit_input, sha


class SizeController:
    tier = 'T1'
    family = 'r10_size'
    uses_gt = False

    def __init__(self, size=500, variant='A', base_kwargs=None, guard_kwargs=None, source_fit='', head_path=''):
        if size not in (50, 100, 200, 300, 400, 500) or variant not in ('A', 'G', 'GC_loeo', 'GC_pair'):
            raise ValueError('invalid R10 cell size/variant')
        self.size, self.variant = size, variant
        self.base_kwargs = dict(base_kwargs or {})
        self.guard_kwargs = dict(guard_kwargs or {})
        self.source_fit, self.head_path = str(source_fit), str(head_path)
        self.name = f'R10_{variant}_{size}'
        self.prof, self.inner = api.NULL_PROFILER, None
        self.uses_nonlibrary_action = variant.startswith('GC_')

    def __getattr__(self, name):
        if name.startswith('__') or name == 'inner':
            raise AttributeError(name)
        inner = self.__dict__.get('inner')
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def fit(self, lib, ctx):
        # Fit corpus is solely the selected rows of the parent B library.
        subset = SubsetLibrary(ctx.root, ctx.lib_key, self.size)
        sc = SubsetContext(ctx, subset)
        self.row_subset = subset.rows.copy()
        self.row_subset.flags.writeable = False
        self.parent_name = PARENT[ctx.model]
        if self.variant.startswith('GC_'):
            path = assert_fit_input(self.source_fit)
            with path.open('rb') as f:
                blob = pickle.load(f)
            src = blob['method']
            if (src.variant != 'G' or src.size != self.size or blob['cell'] != ctx.cell
                    or src.fit_info.get('fit_version') != 2 or not src.fit_info.get('library_only')
                    or not np.array_equal(src.row_subset, subset.rows)):
                raise ValueError('corrector must use matching fresh R10 G fit')
            self.inner = src.inner
            base = RecipeCorrectedBase.__new__(RecipeCorrectedBase)
            vars(base).update(vars(self.inner.base))
            assert_fit_input(self.head_path)
            base.heads, base.head_meta = load_head(self.head_path)
            if base.head_meta['cell'] != ctx.cell or base.head_meta['size'] != self.size:
                raise ValueError('head cell/size mismatch')
            if (base.head_meta.get('fit_pool') != 'B' or base.head_meta.get('variant') != self.variant[3:]
                    or base.head_meta.get('source_fit_sha256') != sha(path)):
                raise ValueError('head provenance must match this R10 B-only G fit')
            base.sig_head = np.asarray(base.head_meta['sigma'], np.float32)
            base.chans, base.blend, base.correct_gripper = 6, .5, False
            base.head_path, base.base_fit = self.head_path, self.source_fit
            base.invalidate_anchor()
            self.inner.base = base
        else:
            kw = dict(self.base_kwargs)
            if self.variant == 'A':
                self.inner = BlindAWM(**kw)
            else:
                cls = TriggerCommitJudge if ctx.model == 'pi05' else TriggerGrootCommitJudge
                self.inner = cls(base_kwargs=kw, **self.guard_kwargs)
            self.inner.prof = api.NULL_PROFILER
            # Both fit and candidate names must resolve to the SAME subset.
            # Otherwise V7 treats this as borrowed fitting information and tries
            # to remap global episode IDs through a compact episode-stem table.
            fitted_base = self.inner if self.variant == 'A' else self.inner.base
            fitted_base.lib = fitted_base.fit_src = 'current'
            # No previously fitted PCA cache is read, including for size 500.
            # This is the unchanged R4/R8 randomized-SVD recipe on B keys.
            def subset_pca(L, key, field):
                # A and G share only this freshly computed subset-only PCA.
                # Fitted serving pickles carry the basis; h100 needs no PCA files.
                d = HERE / 'pca' / key / str(self.size)
                d.mkdir(parents=True, exist_ok=True)
                path = d / f'{field}.npz'
                fingerprint = sha(HERE / 'subsets' / f'{key}_{self.size}.npy')
                with (d / f'{field}.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if path.exists():
                        with np.load(path, allow_pickle=False) as z:
                            if str(z['rows_sha256']) != fingerprint:
                                raise ValueError('R10 PCA subset fingerprint mismatch')
                            return tuple(np.array(z[k]) for k in ('mean', 'basis', 'proj'))
                    result = awm.pca_fit(getattr(L, f'key_{field}'))
                    np.savez(path, mean=result[0], basis=result[1], proj=result[2], rows_sha256=fingerprint)
                    return result
            with patch.object(awm, 'pca_current', subset_pca):
                self.inner.fit(subset, sc)  # Standard R8 -> R6 calibration path.
        base = self.inner if self.variant == 'A' else self.inner.base
        base.act = IndexedRows(store._npy_path(subset.dir, 'action'), subset.rows)
        if self.variant != 'A':
            self.inner.C.act = base.act
        self.fit_info = dict(fit_pool='B', library_only=True, no_Bval_used=True, size=self.size,
                             parent=str(subset.dir), rows=len(subset.rows), normalization='subset action[:5,:7] std',
                             kref=base.kref, escalation=False, fit_version=2, pca='fresh subset-only rsvd seed0')
        self.name += '__' + self.inner.name

    def reset(self, episode):
        self.inner.reset(episode)

    def query(self, q):
        r = self.inner.query(q)
        return api.Result(self.row_subset[r.topk], r.scores, r.confidence, action=r.action,
                          library=self.parent_name, extras=r.extras)

    def blind_step(self, q):
        r = self.inner.blind_step(q)
        return self._parent_blind(r)

    def _parent_blind(self, r):
        if isinstance(r, BlindResult):
            return BlindResult(r.action, self.row_subset[r.rows], r.weights, self.parent_name, r.extras)
        return r

    def policy_tail_step(self, q):
        hook = getattr(self.inner, 'policy_tail_step', None)
        if not callable(hook):
            return LookReason(8, 'no_policy_tail_hook')
        # Tail actions come from the policy; their row IDs describe the cache
        # gate's provenance and need the same parent mapping as cache blinds.
        return self._parent_blind(hook(q))

    def bytes_per_entry(self):
        return self.inner.bytes_per_entry()
