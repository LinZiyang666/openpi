"""A with a configurable token pooling grid before per-camera PCA-64.

Only the PCA input changes. Stock AWM fitting, learned metrics, normalization,
step-0 branch, top-16 synthesis, and K1 blind continuation execute unchanged.
"""
from __future__ import annotations

import hashlib
import json
import mmap
from pathlib import Path
import resource
import time
from types import FunctionType

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM

CACHE = Path('/home/weiland/trace_runs/os_closed_loop/r06_abl/pca')
RECIPE = 'centered_rsvd_k128_o32_power3_seed0_keep64_chunk128_v1'


def pool_tokens(tokens, grid):
    """Row-major spatial mean in f32; grid=16 is lossless flattening, no pooling."""
    t = np.asarray(tokens, np.float32)
    if t.shape[-2:] != (256, 2048) or grid not in (1, 2, 4, 8, 16):
        raise ValueError(f'invalid token shape/grid: {t.shape}, {grid}')
    if grid == 16:
        return t.reshape(*t.shape[:-2], -1)
    block = 16 // grid
    return t.reshape(*t.shape[:-2], grid, block, grid, block, 2048).mean(axis=(-4, -2)).reshape(*t.shape[:-2], -1)


class TokenMatrix:
    """Bounded-residency memmap reader; each chunk is detached before dropping its pages."""
    def __init__(self, path, grid):
        self.path, self.grid = Path(path), grid
        self.a = np.load(path, mmap_mode='r')
        assert self.a.dtype == np.float16 and self.a.shape[1:] == (256, 2048)
        self.shape = (len(self.a), grid * grid * 2048)

    def read(self, lo, hi):
        # f16 -> f32 conversion already detaches the data from the memmap.
        x = pool_tokens(self.a[lo:hi], self.grid)
        # DONTNEED drops this process's file-backed resident pages, not another
        # experiment's mappings. Never retain the 31 GB token file in process RAM.
        self.a._mmap.madvise(mmap.MADV_DONTNEED)
        return x


def streaming_pca(path, grid, chunk=128):
    """Stock AWM randomized-PCA recipe, bounded token blocks instead of a dense key matrix."""
    t0 = time.perf_counter()
    X = TokenMatrix(path, grid)
    n, D = X.shape
    cuts = [(lo, min(n, lo+chunk)) for lo in range(0, n, chunk)]
    mu64 = np.zeros(D, np.float64)
    sumsq = 0.
    for lo, hi in cuts:
        x = X.read(lo, hi)
        mu64 += x.sum(0, dtype=np.float64)
        sumsq += float(np.square(x, dtype=np.float64).sum())
    mu64 /= n
    mu = mu64.astype(np.float32)
    total_ss = sumsq - n * float(mu64 @ mu64)
    del x, mu64

    def xmul(M):
        out = np.empty((n, M.shape[1]), np.float32)
        muM = mu @ M
        for lo, hi in cuts:
            out[lo:hi] = X.read(lo, hi) @ M - muM
        return out

    def xtmul(Y):
        out = np.zeros((D, Y.shape[1]), np.float32)
        for lo, hi in cuts:
            out += X.read(lo, hi).T @ Y[lo:hi]
        out -= np.outer(mu, Y.sum(0))
        return out

    r = min(160, n, D)
    omega = np.random.default_rng(0).standard_normal((D, r), dtype=np.float64).astype(np.float32)
    Q, _ = np.linalg.qr(xmul(omega))
    del omega
    for iteration in range(3):
        Z, _ = np.linalg.qr(xtmul(Q))
        Q, _ = np.linalg.qr(xmul(Z))
        del Z
        print(json.dumps(dict(event='pca_power', path=str(path), grid=grid, iteration=iteration+1,
                              elapsed_s=time.perf_counter()-t0)), flush=True)
    _, singular, Vt = np.linalg.svd(xtmul(Q).T, full_matrices=False)
    B = np.ascontiguousarray(Vt[:64].T, np.float32)
    del Vt, Q
    proj = xmul(B)
    # Actual projected centered energy, not just the sketch singular values.
    captured = np.sum(np.asarray(proj, np.float64)**2, axis=0)
    info = dict(recipe=RECIPE, n=n, dims=D, grid=grid, keep=64, sketch_rank=r, power=3,
                chunk=chunk, total_centered_ss=total_ss, explained_variance_ratio=(captured/total_ss).tolist(),
                explained_variance_ratio_sum=float(captured.sum()/total_ss),
                sketch_singular_values=singular[:64].tolist(), basis_bytes=int(B.nbytes),
                mean_bytes=int(mu.nbytes), wall_s=time.perf_counter()-t0,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                token_file=str(path), token_file_bytes=Path(path).stat().st_size)
    return mu, B, proj, info


def fingerprint(lib):
    return dict(n=lib.L, ids_sha256=hashlib.sha256((lib.dir/'ids.json').read_bytes()).hexdigest(),
                token_shapes={v:list(lib.tok(v).shape) for v in ('v0','v1')})


def prepare_pca(lib, lib_key, grid, cache=CACHE):
    root = Path(cache) / f'grid{grid}' / lib_key / lib.dir.name
    fp = fingerprint(lib)
    manifest = json.loads((lib.dir/'manifest.json').read_text())
    if not manifest.get('tok_complete'):
        raise api.ContractError('token PCA requires a complete library token table')
    rows = np.load(lib.dir/'tok/rows.npy', mmap_mode='r')
    if not np.array_equal(rows, np.arange(lib.L)):
        raise api.ContractError('token rows must be exactly aligned with all library rows')
    for camera in ('v0', 'v1'):
        d = root / camera
        if (d/'meta.json').exists():
            meta = json.loads((d/'meta.json').read_text())
            assert meta['fingerprint'] == fp and meta['recipe'] == RECIPE and meta['grid'] == grid
            continue
        d.mkdir(parents=True, exist_ok=True)
        mu, B, proj, meta = streaming_pca(lib.dir/'tok'/f'{camera}.npy', grid)
        for name, value in [('mean',mu), ('basis',B), ('proj',proj)]:
            np.save(d/f'{name}.npy', value)
        meta.update(fingerprint=fp, camera=camera)
        (d/'meta.json').write_text(json.dumps(meta, indent=1)+'\n')
        print(json.dumps(dict(event='pca_done', directory=str(d), **meta)), flush=True)
    return root.parent.parent


class _TokenQuery:
    def __init__(self, q, grid, previous):
        self.q = q
        self.key_v0 = pool_tokens(q.tok_v0, grid)
        self.key_v1 = pool_tokens(q.tok_v1, grid)
        self.previous_available = (previous is not None and previous[0] == q.episode.uid
                                   and previous[1] == int(q.step)-1)
        # Only AWM's unused adjacent-key cosine diagnostic reads these [-1]
        # entries. Dense state/action histories are forwarded without change.
        self.hist_key_v0 = (previous[2] if self.previous_available else self.key_v0)[None]
        self.hist_key_v1 = (previous[3] if self.previous_available else self.key_v1)[None]

    def __getattr__(self, key):
        return getattr(self.q, key)


class TokenPCAAWM(BlindAWM):
    family = 'r6_p2_token_pca'

    def __init__(self, pooling_grid=16, pca_cache=str(CACHE), **kwargs):
        super().__init__(**kwargs)
        if pooling_grid not in (1,2,4,8,16) or int(pooling_grid) != pooling_grid:
            raise ValueError('pooling_grid must be 1, 2, 4, 8, or 16')
        if self.fit_data != 'same' or self.features != 'joint' or not self.early or self.codes:
            raise ValueError('token ablation preserves A: fit_data=same, joint, early, codes=0')
        self.pooling_grid, self.pca_cache = int(pooling_grid), str(pca_cache)
        self.name = f'P2_tokens_grid{pooling_grid}__' + self.name
        self._previous_tokens = None

    def fit(self, lib, ctx):
        if self.pooling_grid == 4:
            # Exact deployed A control, including its original PCA cache and
            # model-specific online pooling arithmetic (pi05 pools in bf16).
            super().fit(lib, ctx)
            self.token_pca_info = {'control': 'unchanged deployed A pooled-key path'}
            return
        deployed = lib if self.lib == 'current' else ctx.open_library(awm.BIG[ctx.model])
        root = prepare_pca(deployed, ctx.lib_key, self.pooling_grid, self.pca_cache)

        def current(L, key, camera):
            d = root/key/'current'/camera
            return tuple(np.array(np.load(d/f'{name}.npy'), np.float32, copy=True)
                         for name in ('mean','basis','proj'))

        fit = FunctionType(awm.AWM.fit.__code__, {**awm.AWM.fit.__globals__, 'PCA_BIG':root, 'pca_current':current},
                           argdefs=awm.AWM.fit.__defaults__, closure=awm.AWM.fit.__closure__)
        fit(self, lib, ctx)
        self._fit_blind(deployed)
        self.token_pca_info = {v:json.loads((root/ctx.lib_key/deployed.dir.name/v/'meta.json').read_text())
                               for v in ('v0','v1')}

    def reset(self, episode):
        super().reset(episode)
        self._previous_tokens = None

    def _token_query(self, q):
        return q if isinstance(q, _TokenQuery) else _TokenQuery(q, self.pooling_grid, self._previous_tokens)

    def _dist(self, q):
        return super()._dist(q if self.pooling_grid == 4 else self._token_query(q))

    def query(self, q):
        if self.pooling_grid == 4:
            return super().query(q)
        tq = self._token_query(q)
        result = super().query(tq)
        if not tq.previous_available:
            result.extras.pop('still', None)  # never fabricate a visual-history diagnostic across missing tokens
        self._previous_tokens = (q.episode.uid, int(q.step), tq.key_v0, tq.key_v1)
        return result
