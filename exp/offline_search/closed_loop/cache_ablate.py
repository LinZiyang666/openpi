"""R12 layer-one-only cache; fixed acquisition, no judge or correction."""
from __future__ import annotations

import numpy as np
from exp.offline_search.harness import api, dims
from exp.offline_search.closed_loop.blind import BlindResult
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM

ARMS = ('both', 'wrist_sub', 'wrist_refit', 'agent_sub', 'agent_refit')
SPEC = 'exp.offline_search.closed_loop.cache_ablate:R12CacheAblate'


class CameraMetric:
    """Frozen reduced metric, with deployed float32 affine early representation."""
    def distance(self, x, initial=False):
        if hasattr(self, 'restricted_X'):
            transform = self.early_transform if initial else self.transform
            delta = (np.asarray(x, np.float64)-self.restricted_X) @ transform
            return np.sqrt(np.sum(delta*delta, axis=1)).astype(np.float32)
        if initial:
            y = x @ self.W0 - self.c0
            cross = self.Z @ (self.A0 @ y)
            d2 = self.n20 - 2.0 * cross + float(y @ y)
        else:
            z = x @ self.W - self.shift
            d2 = self.z2 - 2.0 * (self.Z @ z) + float(z @ z)
        return np.sqrt(np.maximum(d2, 0.0))


class _CameraCache(BlindAWM):
    def query(self, q):
        # Same AWM top-k/kernel/synthesis and anchor bookkeeping. The removed
        # adjacent-key diagnostic is the only code that read the unused view.
        task, step = self.tasks[int(q.task_id)], int(q.step)
        camera = self.selected_camera
        key = np.asarray(getattr(q, f'key_v{camera}'), np.float32)
        p = getattr(self, f'B{camera}T') @ key - getattr(self, f'muB{camera}')
        rs = np.asarray(dims.valid_state(q.rs, self.model), np.float32)
        d = self.camera_metrics[int(q.task_id)].distance(np.r_[p, rs], initial=step == 0)
        n, k = len(d), min(self.k, len(d))
        idx = np.argpartition(d, k - 1)[:k] if k < n else np.arange(n)
        idx = idx[np.lexsort((idx, d[idx]))]
        dk = d[idx].astype(np.float64)
        rows = task.rows[idx]
        wn, chunks, action = self._mix(rows, _kernel_w(dk-dk[0], self.kref))
        disp = self._disp5(task, action, idx[:5])
        dst = self._dst(task, rs)
        regime = 0 if step == 0 else 2  # Pure cache can never enter post-MISS.
        conf = self._conf(task, regime, float(d.min()), disp, dst, float('nan'))
        ex = dict(d1=float(d.min()), d1_rel=float(d.min())/(float(np.median(d))+1e-12),
                  disp5=disp, dst=dst, regime=float(regime),
                  lib_ep=float(self.lib_ep[rows[0]]), lib_step=float(self.lib_step[rows[0]]),
                  w_eff=float(1/(wn.astype(np.float64)**2).sum()))
        self._insure(action, chunks, wn, step, ex)
        result = api.Result(rows.astype(np.int64), -dk, float(conf), action=action,
                            library=self.cand_name, extras=ex)
        # Preserve _BlindMixin's score-to-anchor arithmetic exactly.
        weights = _kernel_w(dk-dk[0], self.kref)
        self._remember_anchor(q, rows, (weights/weights.sum()).astype(np.float32), action)
        return result


class R12CacheAblate:
    family, tier, uses_gt, uses_nonlibrary_action = 'r12_cache_ablate', 'T1', False, True

    def __init__(self, library='current', episode_subset=None, method='both'):
        if method not in ARMS:
            raise ValueError(method)
        self.library, self.episode_subset, self.ablate_method = library, episode_subset, method
        self.inner = None
        self.prof = api.NULL_PROFILER
        self.next_camera_mode = ('full' if method == 'both' else
                                 'wrist_only' if method.startswith('wrist') else 'agent_only')
        self.name = f'R12CacheAblate_{method}'

    def fit(self, lib, ctx):
        raise api.ContractError('R12CacheAblate requires a frozen B-library --os-fit-artifact')

    def reset(self, episode):
        self.inner.reset(episode)

    def set_camera_mode(self, mode):
        if mode != self.next_camera_mode:
            raise ValueError('pure-cache ablation acquisition cannot change cameras')

    def invalidate_anchor(self):
        self.inner.invalidate_anchor()

    @property
    def last_blind_extras(self):
        return self.inner.last_blind_extras

    def query(self, q):
        r = self.inner.query(q)
        return api.Result(self.row_subset[r.topk], r.scores, r.confidence,
                          action=r.action, library=self.library, extras=r.extras)

    def blind_step(self, q):
        r = self.inner.blind_step(q)
        if isinstance(r, BlindResult):
            return BlindResult(r.action, self.row_subset[r.rows], r.weights, self.library, r.extras)
        return r

    def bytes_per_entry(self):
        return self.inner.bytes_per_entry()
