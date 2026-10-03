"""One deployment entry point: cache, only-no-progress guard, distance LOEO head.

Serving imports only frozen cache/judge code and R10's indexed payload view.
All learned state is embedded in the plugin artifact; only the selected parent
library's action array remains an external payload dependency. No fit/head/
calibration artifact is loaded by fit() or by serving.
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult, LookReason
from exp.offline_search.harness import api
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM

ONLY_NO_PROGRESS = ('stuck', 'terminal', 'overtime')
DISTANCE_RULE = dict(peak=.5, plateau=.75, cutoff=2.)


def predict(head, X):
    # Frozen R9 equation, also used by R10 Stage 2b; no exploration imports.
    Xn = np.clip((X - head['mean']) / head['std'], -8, 8)
    F = np.concatenate([Xn, np.cos(Xn @ head['w'] + head['bias']) * np.sqrt(2)], 1)
    return F @ head['coef'].T + head['intercept']


class _CorrectedCache(BlindAWM):
    """Frozen R10 GC_dist serving arithmetic, including its operation order."""
    family = 'r10_recipe_corrected_cache'
    uses_nonlibrary_action = True

    def _features(self, q, xv, rs8, action):
        cols = [xv, rs8, (action[:10, :7] / self.sig_head).ravel(),
                np.array([min(int(q.step), 120) / 120.], np.float32),
                np.eye(10, dtype=np.float32)[int(q.task_id)]]
        return np.concatenate(cols).astype(np.float32)[None]

    def _correction(self, q, action):
        if int(q.step) == 0:
            return np.zeros((10, self.chans), np.float32)
        T, _, _, _, _, _, _, d, _, _, _ = self._dist(q)
        anchor = self._anchor
        if anchor is None:
            return np.zeros((10, self.chans), np.float32)
        positions = np.searchsorted(T.rows, anchor['rows'])
        if np.any(positions >= len(T.rows)) or not np.array_equal(T.rows[positions], anchor['rows']):
            raise ValueError('distance corrector anchor is not in the query task')
        ratio = float(np.min(d[positions])) / self.distance_scale
        # Multiply the residual by the gate, then by blend=.5 in _apply:
        # this preserves GC_dist's float32 rounding as well as its strength.
        factor = 0. if not np.isfinite(ratio) else float(np.clip((2. - ratio) / 1.25, 0., 1.))
        if factor == 0:
            return np.zeros((10, self.chans), np.float32)
        k0, k1 = np.asarray(q.key_v0, np.float32), np.asarray(q.key_v1, np.float32)
        xv = np.concatenate([self.B0T @ k0 - self.muB0, self.B1T @ k1 - self.muB1])
        x = self._features(q, xv, np.asarray(q.rs, np.float32)[:8], action)
        corr = predict(self.heads[str(int(q.task_id))], x)[0].reshape(10, self.chans) * self.sig_head[:self.chans]
        return corr * np.float32(factor)

    def _apply(self, action, corr):
        action = np.array(action, dtype=np.float32, copy=True)
        action[:10, :6] += self.blend * corr[:, :6]
        return action

    def query(self, q):
        res = super().query(q)
        corr = self._correction(q, res.action)
        action = self._apply(res.action, corr)
        res.action = action
        res.extras = dict(res.extras, r9f_corr_rms=float(np.sqrt(np.mean(corr[:, :6] ** 2))), r9f_blend=self.blend)
        a = self._anchor
        self._remember_anchor(q, a['rows'], a['weights'], action)
        return res

    def os_synth(self, q, rows, w):
        action = super().os_synth(q, rows, w)
        action = self._apply(action, self._correction(q, action))
        a = self._anchor
        if a is not None:
            self._remember_anchor(q, a['rows'], a['weights'], action)
        return action


class R10Recipe:
    """Fit the identical three-layer rule to a full library or whole episodes.

    ``library`` is a stored library name. ``episode_subset`` is either None or
    {task_id: [parent_episode_ids, ...]}; ordering fixes the five-fold distance
    calibration. Each task needs at least three complete episodes. No parameters
    vary by cell except the frozen model-aware R8 judge and learned library state.
    """
    family = 'r10_recipe'
    tier = 'T1'
    uses_gt = False
    uses_nonlibrary_action = True

    def __init__(self, library='current', episode_subset=None):
        if not isinstance(library, str) or not library or '/' in library or library in ('.', '..'):
            raise ValueError('library must be a stored library name')
        if episode_subset is not None and not isinstance(episode_subset, dict):
            raise ValueError('episode_subset must map tasks to parent episode IDs')
        self.library = library
        self.episode_subset = None if episode_subset is None else {
            str(t): list(ids) for t, ids in episode_subset.items()}
        self.inner = None
        self.prof = api.NULL_PROFILER
        self.name = 'R10Recipe_unfitted'

    def __getattr__(self, name):
        if name.startswith('__') or name == 'inner':
            raise AttributeError(name)
        inner = self.__dict__.get('inner')
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def fit(self, lib, ctx):
        # Import fitting dependencies only when building, never at serving load.
        from .fitting import fit_recipe
        fit_recipe(self, ctx)

    def reset(self, episode):
        self.inner.reset(episode)

    def query(self, q):
        r = self.inner.query(q)
        return api.Result(self.row_subset[r.topk], r.scores, r.confidence, action=r.action,
                          library=self.library, extras=r.extras)

    def _parent_blind(self, r):
        if isinstance(r, BlindResult):
            return BlindResult(r.action, self.row_subset[r.rows], r.weights, self.library, r.extras)
        return r

    def blind_step(self, q):
        return self._parent_blind(self.inner.blind_step(q))

    def policy_tail_step(self, q):
        hook = getattr(self.inner, 'policy_tail_step', None)
        if not callable(hook):
            return LookReason(8, 'no_policy_tail_hook')
        return self._parent_blind(hook(q))

    def bytes_per_entry(self):
        base = self.inner.base
        head_bytes = sum(v.nbytes for h in base.heads.values() for v in h.values())
        return self.inner.bytes_per_entry() + head_bytes / len(self.row_subset)
