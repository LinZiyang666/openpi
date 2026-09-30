"""Frozen gripper-event stages and candidate-LOEO state calibration (R7 §3).

No online outcome, image, task text, or episode fraction enters this table.
See STAGES_API.md for the calibration conventions and the optional frozen fit.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

SCHEMA = "r7.stages.v1"
FIELDS = ("action", "rs", "task_id", "episode", "step", "success", "next")


def _array(library, name):
    return np.asarray(library[name] if isinstance(library, dict) else getattr(library, name))


def content_fingerprint(library, manifest):
    h = hashlib.sha256()
    geometry = {k: manifest[k] for k in ("exec_steps", "H", "act_valid_dims", "rs_valid_dims", "gripper_dim")}
    h.update(json.dumps(geometry, sort_keys=True).encode())
    for name in FIELDS:
        a = _array(library, name)
        h.update(name.encode()); h.update(str((a.shape, a.dtype.str)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def two_means(x):
    """E1's deterministic two-means; mode 0 is the lower learned center."""
    x = np.asarray(x, float)
    c = np.quantile(x, [.25, .75])
    if c[0] == c[1]:
        c = np.array([x.min(), x.max()])
    for _ in range(100):
        labels = x > c.mean()
        nc = np.array([x[~labels].mean() if (~labels).any() else c[0],
                       x[labels].mean() if labels.any() else c[1]])
        if np.allclose(c, nc):
            break
        c = nc
    return c


def weighted_quantile(values, weights, q):
    ix = np.argsort(values, kind="stable")
    v, w = np.asarray(values)[ix], np.asarray(weights, float)[ix]
    return float(v[min(np.searchsorted(np.cumsum(w), q * w.sum()), len(v) - 1)])


class StageTable:
    schema = SCHEMA

    @classmethod
    def fit(cls, library, *, manifest):
        """Fit successful demonstrations; optional manifest['_retrieval'] is frozen A.

        Without it, an ordinary LibraryView supplies the root for a library-only
        A fit. Synthetic/dict libraries must supply the retrieval fit explicitly.
        """
        t = cls()
        t.manifest = {k: manifest[k] for k in ("exec_steps", "H", "act_valid_dims", "rs_valid_dims", "gripper_dim")}
        t.fingerprint = content_fingerprint(library, manifest)
        t.exec_steps = int(manifest["exec_steps"])
        t.reference_blocks = 2  # selected A commitment: anchor plus one blind block
        if t.exec_steps < 1 or not 0 <= int(manifest["gripper_dim"]) < int(manifest["act_valid_dims"]):
            raise ValueError("invalid manifest execution/gripper geometry")
        t.task_id, t.episode, t.step = (_array(library, k).copy() for k in ("task_id", "episode", "step"))
        t.success = _array(library, "success").astype(bool)
        if not t.success.any():
            raise ValueError("stage calibration needs successful library rows")
        t.rs = np.array(_array(library, "rs")[:, :int(manifest["rs_valid_dims"])], float, copy=True)
        t.next = _array(library, "next").astype(np.int64, copy=True)
        r = np.flatnonzero((t.next >= 0) & (t.next < len(t.next)))
        n = t.next[r]
        valid = ((t.episode[r] == t.episode[n]) & (t.task_id[r] == t.task_id[n]) & (t.step[n] == t.step[r] + 1))
        t.next[(t.next < 0) | (t.next >= len(t.next))] = -1
        t.next[r[~valid]] = -1
        t.successor_valid = t.next >= 0
        command = np.median(_array(library, "action")[:, :t.exec_steps, int(manifest["gripper_dim"])], axis=1)
        t.gripper_centers = two_means(command[t.success])
        t.gripper_threshold = float(t.gripper_centers.mean())
        t.mode = np.full(len(t.next), -1, np.int8)
        t.event_near = np.zeros(len(t.next), bool)
        t.rows_to_event = np.full(len(t.next), -1, np.int32)
        t.stage_run = np.full(len(t.next), -1, np.int32)
        t.event_occupancy = {}
        for task in np.unique(t.task_id):
            good = np.flatnonzero((t.task_id == task) & t.success)
            for ep in np.unique(t.episode[good]):
                rows = good[t.episode[good] == ep]
                rows = rows[np.argsort(t.step[rows], kind="stable")]
                if len(rows) > 1 and not np.all(np.diff(t.step[rows]) == 1):
                    raise ValueError("successful episode has missing decision rows")
                mode = command[rows] > t.gripper_threshold
                if len(mode) > 2:
                    mode = np.r_[mode[0], (mode[:-2].astype(int) + mode[1:-1] + mode[2:]) >= 2, mode[-1]]
                events = np.flatnonzero(np.diff(mode)) + 1
                pos = np.arange(len(rows))
                t.mode[rows] = mode
                t.stage_run[rows] = np.cumsum(np.r_[0, np.diff(mode) != 0])
                if len(events):
                    t.event_near[rows] = (abs(pos[:, None] - events).min(axis=1) <= 1)
                # At an event row the *next* event is meant. Episode end is a
                # boundary after its last recorded head, i.e. one row past it.
                boundaries = np.r_[events, len(rows)]
                t.rows_to_event[rows] = boundaries[np.searchsorted(boundaries, pos, side="right")] - pos
            t.event_occupancy[int(task)] = float(t.event_near[good].mean()) if len(good) else 0.
        t.h = t.event_occupancy
        t.state_scale = np.std(t.rs[t.success], axis=0)
        t.state_active = t.state_scale > np.finfo(float).eps * max(float(t.state_scale.max()), 1.)
        t.state_scale[~t.state_active] = 1.
        if not t.state_active.any():
            raise ValueError("no varying valid proprioceptive coordinates")
        retrieval = manifest.get("_retrieval")
        if retrieval is None:
            from exp.offline_search.harness import api
            from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
            if isinstance(library, dict):
                raise ValueError("dict library needs manifest['_retrieval'] for exact A-LOEO")
            retrieval = BlindAWM(lib="current" if library.name == "current" else "big",
                                 kref=5 if library.name == "current" else 8,
                                 serving="anchor_tail", budget=1, gates="budget_only")
            ctx = api.Context(root=library.root, cell=library.key + "_cache", seed=0,
                              scratch=Path("/tmp/r7_C1/stage_fit"))
            retrieval.fit(library, ctx)
        t._calibrate(retrieval)
        t._freeze()
        return t

    def advance(self, rows, blocks):
        """True successors; -1 propagates. Never terminal-clamp or drop members."""
        out = np.asarray(rows, np.int64).copy()
        if blocks < 0:
            raise ValueError("negative successor depth")
        for _ in range(int(blocks)):
            ok = (out >= 0) & (out < len(self.next))
            new = np.full(out.shape, -1, np.int64)
            new[ok] = self.next[out[ok]]
            out = new
        return out

    def online(self, rows, weights, cmd_mode=None):
        rows, w = np.asarray(rows, np.int64), np.asarray(weights, float)
        if rows.ndim != 1 or w.shape != rows.shape or not len(rows) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("invalid stage kernel")
        w = w / w.sum()  # diagnostic masses only; never changes action weights
        valid = (rows >= 0) & (rows < len(self.mode))
        modes = np.full(rows.shape, -1, np.int8)
        modes[valid] = self.mode[rows[valid]]
        mass = [float(w @ (modes == m)) for m in (0, 1)]
        unanimous = bool((modes >= 0).all() and np.all(modes == modes[0]))
        mode = int(modes[0]) if unanimous else -1
        if cmd_mode is not None:
            unanimous = unanimous and mode == int(cmd_mode)
        known = valid & (modes >= 0)
        return dict(mode_mass=mass, mode0_mass=mass[0], mode1_mass=mass[1], mode=mode,
                    unanimous=unanimous, event_mass=float(w[known] @ self.event_near[rows[known]]),
                    min_rows_to_event=int(self.rows_to_event[rows].min()) if known.all() else -1,
                    unknown_mass=float(w @ (modes < 0)))

    def displacement(self, current, anchor, rows, weights, blocks):
        future = self.advance(rows, blocks)
        if (future < 0).any():
            return float("inf"), False
        w = np.asarray(weights, float)
        expected = np.einsum("k,kd->d", w, self.rs[future] - self.rs[rows])
        d = ((np.asarray(current)[:len(self.state_scale)] - np.asarray(anchor)[:len(self.state_scale)] - expected)
             / self.state_scale)[self.state_active]
        return float(np.sqrt(np.mean(d*d))), bool(np.isfinite(d).all())

    def deviation(self, current, rows, weights):
        """Absolute current-state residual for CT's entry latch."""
        d = ((np.asarray(current)[:len(self.state_scale)] - np.einsum("k,kd->d", weights, self.rs[rows]))
             / self.state_scale)[self.state_active]
        return float(np.sqrt(np.mean(d*d)))

    def _calibrate(self, base):
        from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
        from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import fingerprint
        self.retrieval_fingerprint = fingerprint(base)
        samples, absolute, sample_task, sample_ep = [], [], [], []
        for task, fitted in sorted(base.tasks.items()):
            rows = np.asarray(fitted.rows)
            z = np.asarray(fitted.Z, float)
            z2 = np.sum(z*z, axis=1)
            # E3: A-cadence even rows, successful held-out source episode,
            # frozen PCA/metric and unmodified A candidate set.
            for i in np.flatnonzero(self.success[rows] & (self.step[rows] % self.reference_blocks == 0)):
                if self.step[rows[i]] == 0 and base.early:
                    zz = np.asarray(fitted.Z0, float) if fitted.Z0 is not None else z @ fitted.A0
                    if fitted.Z0 is None and fitted.As0 is not None:
                        zz = zz + np.asarray(fitted.RS, float) @ fitted.As0
                    dist = np.linalg.norm(zz - zz[i], axis=1)
                else:
                    dist = np.sqrt(np.maximum(z2 + z2[i] - 2*z @ z[i], 0))
                dist[self.episode[rows] == self.episode[rows[i]]] = np.inf
                if np.isfinite(dist).sum() < base.k:
                    continue
                ix = np.argpartition(dist, base.k - 1)[:base.k]
                ix = ix[np.lexsort((rows[ix], dist[ix]))]
                members = rows[ix]
                w = _kernel_w(dist[ix] - dist[ix[0]], base.kref)
                w = w / w.sum()
                abs0 = self.deviation(self.rs[rows[i]], members, w)
                ds = []
                for age in range(1, self.reference_blocks + 1):
                    target = self.advance([rows[i]], age)[0]
                    if target < 0:
                        break
                    d, ok = self.displacement(self.rs[target], self.rs[rows[i]], members, w, age)
                    if not ok:
                        break
                    ds.append(d)
                absolute.append(abs0); sample_task.append(int(task)); sample_ep.append(int(self.episode[rows[i]]))
                samples.append(max(ds) if len(ds) == self.reference_blocks else np.nan)
        d, a = np.asarray(samples), np.asarray(absolute)
        tasks, eps = np.asarray(sample_task), np.asarray(sample_ep)
        valid = np.isfinite(d)
        if not valid.any():
            raise ValueError("no full-support LOEO reference commitments")
        self.valve_radius_row = float(np.quantile(d[valid], .95, method="higher"))
        # E3's episode convention: quantile over each episode's maximum.
        emax = [d[valid & (eps == e)].max() for e in np.unique(eps[valid])]
        self.valve_radius_episode = float(np.quantile(emax, .95, method="higher"))
        self.valve_radius = self.valve_radius_row
        self.valve_radius_by_task, self.deviation_p75, self.deviation_occupancy = {}, {}, {}
        for task in np.unique(tasks):
            mask = tasks == task
            dv = d[mask & valid]
            self.valve_radius_by_task[int(task)] = float(np.quantile(dv, .95, method="higher")) if len(dv) else self.valve_radius
            p = float(np.quantile(a[mask], .75, method="higher"))
            self.deviation_p75[int(task)] = p
            self.deviation_occupancy[int(task)] = float(np.mean(a[mask] > p))
        self.calibration = dict(convention="frozen-A candidate-LOEO, successful even-row sources; maximum over reference commitment",
                                anchors=int(valid.sum()), attempted_anchors=len(d), episodes=len(emax),
                                q_row95=self.valve_radius_row, q_episode95=self.valve_radius_episode,
                                reference_controls=self.exec_steps * self.reference_blocks)

    def _freeze(self):
        for value in self.__dict__.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False

    def save(self, path):
        payload = pickle.dumps(self, protocol=4)
        envelope = dict(schema=SCHEMA, content_sha256=hashlib.sha256(payload).hexdigest(), payload=payload)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(pickle.dumps(envelope, protocol=4))

    @classmethod
    def load(cls, path, *, library=None, manifest=None):
        envelope = pickle.loads(Path(path).read_bytes())
        if envelope["schema"] != SCHEMA or hashlib.sha256(envelope["payload"]).hexdigest() != envelope["content_sha256"]:
            raise ValueError("stage artifact schema/content mismatch")
        t = pickle.loads(envelope["payload"])
        if library is not None and (manifest is None or content_fingerprint(library, manifest) != t.fingerprint):
            raise ValueError("stage artifact belongs to a different library")
        t._freeze()
        return t
