"""Library-only AWM/guard/LOEO-corrector/periodic shared across layouts.

This array-only entry cannot open query/outcome files. P6 seals the source
allowlist and content manifests before constructing LibraryData. Per-task
descent is the adjacent data-only gate; P2 owns dispatch and attempt fencing.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np

from ..contracts import ActionStateLayout, RequestContext
from ..query_view import QueryView, RecipeResult
from .guard import RETIRED
from .math import _kernel_w, anchor_weights, fit_head, fit_metric, pca_fit, predict
from .periodic import periodic


@dataclass(frozen=True)
class LibraryData:
    features: dict[str, np.ndarray]
    state: np.ndarray
    action: np.ndarray
    task_id: np.ndarray
    episode: np.ndarray
    step: np.ndarray
    partition: str = "library"
    source_digest: str = ""


def _readonly(a, dtype):
    a = np.array(a, dtype=dtype, order="C", copy=True)
    if not np.isfinite(a).all():
        raise ValueError("library contains nonfinite values")
    a.flags.writeable = False
    return a


class LayoutRecipe:
    version = "r17.layout_recipe.v1"

    def __init__(self, layout: ActionStateLayout, task_count: int, *, commit_horizon=None, allow_partial_commit=False,
                 k=16, kref=8, corrector=True, guard=False, periodic_gap=None, early=True):
        self.layout, self.task_count = layout, task_count
        self.commit = max(layout.replan_steps, min(10, layout.horizon)) if commit_horizon is None else commit_horizon
        layout.validate_commit(self.commit, allow_partial=allow_partial_commit)
        if task_count <= 0 or k <= 0 or kref <= 0:
            raise ValueError("positive task_count/k/kref required")
        self.k, self.kref = k, kref
        if guard:
            raise ValueError(RETIRED)
        self.use_corrector, self.use_guard = corrector, False
        self.periodic_gap, self.early = periodic_gap, early
        self.serve_loeo_episodes = {}
        self.reset(None)

    def reset(self, episode):
        self._memo_step = -1
        self._memo = None

    def fork(self):
        result = copy.copy(self)
        result.reset(None)
        return result

    def fit(self, lib: LibraryData):
        if not isinstance(lib, LibraryData) or lib.partition != "library":
            raise ValueError("fit requires selected library data, never dev/test/query partitions")
        self.action = _readonly(lib.action, np.float32)
        n = len(self.action)
        if self.action.shape != (n, self.layout.horizon, self.layout.action_width):
            raise ValueError("library action geometry mismatch")
        self.state = _readonly(lib.state, np.float32)
        self.task = _readonly(lib.task_id, np.int64)
        self.episode = _readonly(lib.episode, np.int64)
        self.step = _readonly(lib.step, np.int64)
        if any(a.shape != (n,) for a in (self.task, self.episode, self.step)) or len(self.state) != n:
            raise ValueError("library row alignment mismatch")
        if np.any(self.task < 0) or np.any(self.task >= self.task_count):
            raise ValueError("library task id outside declared task table")
        if len(self.layout.valid_state_indices) > self.state.shape[1] or max(self.layout.valid_state_indices) >= self.state.shape[1]:
            raise ValueError("library active state indices outside state width")
        for ep in np.unique(self.episode):
            rows = np.flatnonzero(self.episode == ep)
            if len(np.unique(self.task[rows])) != 1 or not np.array_equal(self.step[rows], np.arange(len(rows))):
                raise ValueError("selected library must contain complete ordered episodes starting at zero")
        self.fields = tuple(k for k in lib.features if k != "robot_state")
        if not self.fields:
            raise ValueError("recipe requires real policy features")
        self.pca, projections = {}, []
        for name in self.fields:
            f = _readonly(lib.features[name], np.float32)
            if f.ndim != 2 or len(f) != n:
                raise ValueError("feature library geometry mismatch")
            keep = min(64, f.shape[1], n - 1)
            if keep < 1:
                raise ValueError("insufficient library PCA rank")
            mean, basis, proj = pca_fit(f, keep=keep)
            self.pca[name] = (_readonly(basis.T, np.float32), _readonly(mean @ basis, np.float32))
            projections.append(proj)
        self.projected = _readonly(np.concatenate(projections, 1), np.float32)
        rs = self.state[:, self.layout.valid_state_indices]
        X = np.concatenate([self.projected, rs], 1).astype(np.float64)
        active = self.layout.active_actions
        self.sigma = np.maximum(self.action[:, :self.layout.replan_steps, active].std(axis=(0, 1)), 1e-6).astype(np.float32)
        heads = (self.action[:, :self.layout.replan_steps, active].astype(np.float64) / self.sigma.astype(np.float64)).reshape(n, -1)
        self.tasks, self.fit_evidence = {}, {"partition": "library", "source_digest": lib.source_digest,
                                           "loeo": [], "layout_identity": self.layout.identity}
        for task in np.unique(self.task):
            rows = np.flatnonzero(self.task == task)
            eps = self.episode[rows]
            if len(np.unique(eps)) < (3 if self.use_corrector else 2):
                raise ValueError("corrector requires >=3 complete episodes/task; raw retrieval requires >=2")
            T = self._metric(X[rows], heads[rows], eps)
            T.rows = rows
            T.HD = np.ascontiguousarray(heads[rows], np.float32)
            T.h2 = (heads[rows] ** 2).sum(1).astype(np.float32)
            T.s_c = None
            if self.layout.horizon >= 2 * self.layout.replan_steps:
                tail = (self.action[rows][:, self.layout.replan_steps:2*self.layout.replan_steps, active].astype(np.float64) / self.sigma.astype(np.float64)).reshape(len(rows), -1)
                h = heads[rows]
                dist = (tail * tail).sum(1)[:, None] - 2 * tail @ h.T + (h * h).sum(1)[None]
                dist[self.episode[rows, None] == self.episode[rows][None]] = np.inf
                T.s_c = float(np.median(np.sqrt(np.maximum(dist.min(1), 0) / h.shape[1]))) + 1e-6
            T.early = None
            if self.early:
                mask = self.step[rows] <= 2
                mn, sd, W = fit_metric(X[rows], heads[rows], eps)
                mn0, sd0, W0 = fit_metric(X[rows][mask], heads[rows][mask], eps[mask])
                A = np.linalg.inv(W) @ (np.diag(sd / sd0) @ W0)
                b = ((mn - mn0) / sd0) @ W0
                Y = T.Z.astype(np.float64) @ A
                T.early = SimpleNamespace(A0=A.astype(np.float32), W0f=(W0 / sd0[:, None]).astype(np.float32),
                                          c0=((mn0 / sd0) @ W0 + b).astype(np.float32),
                                          n20=(Y * Y).sum(1).astype(np.float32))
            self.tasks[int(task)] = T
        self.heads = {}
        if self.use_corrector:
            distances = []
            for task in self.tasks:
                Xh, Yh, eps, row_ids = [], [], [], []
                for row in self.tasks[task].rows:
                    q = QueryView(RequestContext("fit", "library", str(task), 0, int(self.step[row]), task_id=task),
                                  {name: lib.features[name][row] for name in self.fields}, self.state[row], prev_hit=True)
                    proposal, xv, rs = self._retrieve(q, exclude_episode=int(self.episode[row]))
                    self.fit_evidence["loeo"].append({"row": int(row), "query_episode": int(self.episode[row]),
                                                      "donor_episodes": self.episode[proposal.rows].tolist()})
                    Xh.append(self._features(q, xv, rs, proposal.action)[0])
                    target = self.action[row, :self.commit][:, self.layout.motion_channels]
                    cached = proposal.action[:self.commit][:, self.layout.motion_channels]
                    motion_sigma = self.sigma[[active.index(c) for c in self.layout.motion_channels]]
                    Yh.append(((target - cached) / motion_sigma).ravel())
                    eps.append(self.episode[row]); row_ids.append(row)
                    if q.step > 0:
                        distances.append((float(proposal.distances[0]), int(self.episode[row])))
                self.heads[task] = fit_head(np.asarray(Xh, np.float32), np.asarray(Yh, np.float32),
                                            anchor_weights(np.asarray(row_ids), np.asarray(eps)))
            self.distance_scale = self._distance_calibration(lib, X)
            if not np.isfinite(self.distance_scale) or self.distance_scale <= 0:
                raise ValueError("library held-out corrector distance scale is invalid")
            self.fit_evidence["distance_calibration"] = "R10.five_fold_fixed_library_pca"
        for T in self.tasks.values():
            for obj in (T, T.early):
                if obj:
                    for a in vars(obj).values():
                        if isinstance(a, np.ndarray):
                            a.flags.writeable = False
        for h in self.heads.values():
            for a in h.values():
                a.flags.writeable = False
        self.reset(None)
        return self

    @staticmethod
    def _balanced_median(values, episodes):
        if not len(values):
            raise ValueError("held-out library has no nonzero-step rows")
        _, inv, counts = np.unique(episodes, return_inverse=True, return_counts=True)
        weights = 1 / counts[inv]
        order = np.argsort(values, kind="stable")
        return float(values[order[np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]])

    @staticmethod
    def _metric(X, heads, ep, candidates=None):
        mean, std, W = fit_metric(X, heads, ep)
        Z = (((X if candidates is None else candidates) - mean) / std) @ W
        z = np.ascontiguousarray(Z, np.float32)
        return SimpleNamespace(Z=z, z2=(z.astype(np.float64) ** 2).sum(1).astype(np.float32),
                               Wf=np.ascontiguousarray(W / std[:, None], np.float32),
                               shift=((mean / std) @ W).astype(np.float32))

    def _distance_calibration(self, lib, X):
        # Preserve R10 fold assignment, global training-only sigma, batch
        # boundaries (including step0 before dropping), and weighted median.
        fold = np.zeros(len(self.episode), np.int64)
        evidence = []
        for task in self.tasks:
            eps = list(dict.fromkeys(self.episode[self.task == task].tolist()))
            for i, ep in enumerate(eps):
                fold[self.episode == ep] = i % 5
        Cx = X.astype(np.float32)
        active = self.layout.active_actions
        actions = self.action[:, :self.commit, active]
        ds, ep_ds = [], []
        for f in range(5):
            train, held = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            if not len(held):
                continue
            sig = np.maximum(actions[train, :self.layout.replan_steps].std(axis=(0, 1)), 1e-6).astype(np.float32)
            for task in self.tasks:
                cr = train[self.task[train] == task]
                qr = held[self.task[held] == task]
                if not len(qr):
                    continue
                heads = (actions[cr, :self.layout.replan_steps].astype(np.float64) / sig.astype(np.float64)).reshape(len(cr), -1)
                T = self._metric(Cx[cr].astype(np.float64), heads, self.episode[cr])
                for lo in range(0, len(qr), 256):
                    q = qr[lo:lo + 256]
                    z = Cx[q] @ T.Wf - T.shift
                    D = T.z2[None] - 2 * (z @ T.Z.T) + np.sum(z * z, 1)[:, None]
                    D = np.sqrt(np.maximum(D, 0.))
                    D[self.episode[q, None] == self.episode[cr][None]] = np.inf
                    nearest = np.min(D, axis=1)
                    keep = self.step[q] > 0
                    ds.extend(nearest[keep]); ep_ds.extend(self.episode[q][keep])
            evidence.append({"fold": f, "train_episodes": np.unique(self.episode[train]).tolist(),
                             "heldout_episodes": np.unique(self.episode[held]).tolist()})
        self.fit_evidence["distance_folds"] = evidence
        return self._balanced_median(np.asarray(ds), np.asarray(ep_ds))

    def _project(self, q):
        cols = []
        for name in self.fields:
            B, shift = self.pca[name]
            value = np.asarray(q.features[name], np.float32)
            if value.shape != (B.shape[1],) or not np.isfinite(value).all():
                raise ValueError("query feature geometry/finite check failed")
            cols.append(B @ value - shift)
        rs = np.asarray(q.state, np.float32)[list(self.layout.valid_state_indices)]
        if not np.isfinite(rs).all():
            raise ValueError("query state nonfinite")
        return np.concatenate(cols), rs

    def _retrieve(self, q, exclude_episode=None):
        T = self.tasks[q.task_id]
        xv, rs = self._project(q)
        x = np.concatenate([xv, rs])
        if q.step == 0 and T.early is not None:
            y = x @ T.early.W0f - T.early.c0
            d2 = T.early.n20 - 2.0 * (T.Z @ (T.early.A0 @ y)) + float(y @ y)
        else:
            z = x @ T.Wf - T.shift
            d2 = T.z2 - 2.0 * (T.Z @ z) + float(z @ z)
        d = np.sqrt(np.maximum(d2, 0.0))
        raw_d = d
        pos = np.flatnonzero(self.episode[T.rows] != exclude_episode) if exclude_episode is not None else np.arange(len(T.rows))
        if not len(pos):
            raise ValueError("no cross-episode donors")
        if q.step > 0 and q.prev_hit is False and q.previous_action is not None and T.s_c is not None:
            tail = (np.asarray(q.previous_action, np.float32)[self.layout.replan_steps:2*self.layout.replan_steps][:, self.layout.active_actions] / self.sigma).ravel()
            c = np.sqrt(np.maximum(T.h2 - 2.0 * (T.HD @ tail) + float(tail @ tail), 0.0) / len(tail))
            d = d / (float(np.median(d[pos])) + 1e-12) + .5 * c / T.s_c
        chosen = pos[np.lexsort((pos, d[pos]))[:self.k]]
        distances = d[chosen].astype(np.float64)
        w = _kernel_w(distances - distances[0], self.kref)
        wn = (w / w.sum()).astype(np.float32)
        rows = T.rows[chosen]
        # Same float32 operation order as frozen AWM._mix.
        action = np.tensordot(wn, self.action[rows], 1)
        result = RecipeResult(action, rows, wn, distances,
                              extras={"raw_anchor_distance": float(np.min(raw_d[chosen]))})
        return result, xv, rs

    def _features(self, q, xv, rs, action):
        cols = [xv, rs, (action[:self.commit][:, self.layout.active_actions] / self.sigma).ravel(),
                np.array([min(int(q.step), 120) / 120.], np.float32),
                np.eye(self.task_count, dtype=np.float32)[q.task_id]]
        return np.concatenate(cols).astype(np.float32)[None]

    def query(self, q, *, exclude_episode=None):
        if exclude_episode is not None:
            return self._retrieve(q, exclude_episode)[0]
        if q.step == self._memo_step:
            return copy.deepcopy(self._memo)
        if q.step < self._memo_step:
            raise ValueError("reset recipe before a new episode")
        exclude_episode = self.serve_loeo_episodes.get((q.request.task_id, q.request.original_init_idx))
        result, xv, rs = self._retrieve(q, exclude_episode=exclude_episode)
        top = result.rows[0]
        length = int(np.count_nonzero(self.episode == self.episode[top]))
        progress = self.step[top] / max(length - 1, 1)
        prev = q.prev_hit
        if prev is None and q.hist_hit:
            prev = bool(q.hist_hit[-1])
        if self.use_guard:
            raise ValueError(RETIRED)
        guard = False
        knob, ex = periodic(q.request, q.hist_has_vision, q.hist_hit, self.periodic_gap, guard=guard)
        correction_norm = 0.
        if self.use_corrector and q.step > 0:
            ratio = result.extras["raw_anchor_distance"] / self.distance_scale
            factor = 0. if not np.isfinite(ratio) else float(np.clip((2. - ratio) / 1.25, 0., 1.))
            motion = self.layout.motion_channels
            sigma = self.sigma[[self.layout.active_actions.index(c) for c in motion]]
            corr = predict(self.heads[q.task_id], self._features(q, xv, rs, result.action))[0].reshape(self.commit, len(motion)) * sigma
            corr = corr * np.float32(factor)
            correction_norm = float(np.linalg.norm(.5 * corr))
            action = np.array(result.action, dtype=np.float32, copy=True)
            action[:self.commit, list(motion)] += .5 * corr
            result.action = action
        result.force_miss = bool(guard or knob)
        result.extras = dict(result.extras, **ex, guard=guard, noprog_span=0,
                             os_force_miss=float(result.force_miss), os_reason=4. if guard else (62. if knob else 0.))
        if getattr(self, "study_diagnostics", False):
            result.extras.update(corrector_enabled=self.use_corrector,
                corrector_applied=bool(self.use_corrector and q.step > 0),
                corrector_delta_norm=correction_norm)
        self._memo_step, self._memo = q.step, copy.deepcopy(result)
        return result

    def force_look(self, request, has_vision, hit):
        force, _ = periodic(request, has_vision, hit, self.periodic_gap)
        if self.use_guard:
            raise ValueError(RETIRED)
        return force


def fit_library(library: LibraryData, *, layout: ActionStateLayout, task_count: int, **settings):
    return LayoutRecipe(layout, task_count, **settings).fit(library)


def make_method(*, layout: ActionStateLayout, task_count: int, **settings):
    return LayoutRecipe(layout, task_count, **settings)
