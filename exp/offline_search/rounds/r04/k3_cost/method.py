"""Same-library wrist-only AWM and library-LOEO V7/guard recalibration.

Camera deletion is exact on stored pooled keys: no tower inference, interpolation,
query labels, or larger-library borrowing. The inherited action metric is refitted
in 72 dimensions (PCA-64 wrist + 8 valid state). Both kernel and fresh-MISS
continuity are inherited. Camera-aware views duplicate wrist only for the legacy
TWO-camera diagnostic API, never in the metric itself.
"""
from __future__ import annotations
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r02.g1_awm.awm import AWM
from exp.offline_search.rounds.r03.h3_judge.judge import MixedJudge


class WristView:
    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, {"key_v0": "key_v1", "hist_key_v0": "hist_key_v1"}.get(name, name))


class WristAWM(AWM):
    camera_mode = "wrist_only"
    family = "k3_cost"

    def __init__(self, lib="current", kref=None, **kwargs):
        if kwargs.get("fit_data", "same") != "same":
            raise ValueError("WristAWM must refit on the deployed library (fit_data=same)")
        if kwargs.get("features", "joint") != "joint" or kwargs.get("step0_joint", False):
            raise ValueError("WristAWM supports the joint 72-dimensional recipe")
        super().__init__(lib=lib, kref=(5 if lib == "current" else 8) if kref is None else kref, **kwargs)
        self.name = "Wrist_" + self.name

    def _feats(self, P0, P1, rs, which):
        return np.concatenate([P1, rs], axis=1).astype(np.float64)

    def fit(self, lib, ctx):
        if ctx.model != "pi05":
            raise api.SkipCell("WristAWM is the pi05 camera-1 adapter")
        super().fit(lib, ctx)
        # The base fitter consults both cached PCA bases; only P1 enters _feats.
        # Remove the unused camera's fixed and per-entry arrays from deployment.
        self.B0T = np.empty((0, self.B1T.shape[1]), np.float32)
        self.muB0 = np.empty(0, np.float32)
        self.mu0 = self.mu1
        for T in self.tasks.values():
            T.V0, T.Vm0 = T.V1, T.Vm1
        self.fixed_bytes = int(self.mu1.nbytes + self.B1T.nbytes + self.muB1.nbytes + sum(
            x.nbytes for T in self.tasks.values() for k, x in vars(T).items()
            if isinstance(x, np.ndarray) and k in ("Wf", "shift", "W0f", "A0", "c0", "Vm1")))
        self.B0T.flags.writeable = self.muB0.flags.writeable = False

    def query(self, q):
        return super().query(WristView(q))

    def os_score_all(self, q):
        T, _, _, _, _, xv, _, d, _, c, dt = self._dist(WristView(q))
        dt64 = dt.astype(np.float64)
        kr = min(self.kref, self.k, len(dt64))
        ref = max(float(np.partition(dt64, kr - 1)[kr - 1] - dt64.min()), 1e-6)
        S = -((dt64 - dt64.min()) / ref) ** 2
        pc = xv - T.Vm1
        vis = T.V1 @ (pc / max(float(np.sqrt(pc @ pc)), 1e-12))
        aux = {"awm_d": d, "vis_v0": vis, "vis_v1": vis}
        if c is not None:
            aux["awm_c"] = c
        return T.rows, S, aux

    def bytes_per_entry(self):
        return float(4 * ((self.codes or 72) + 8 + ((self.codes or 1) if self.early else 0)))


class WristMixedJudge(MixedJudge):
    """Recompute V7 pseudo-query tables and stuck thresholds using wrist alone."""
    camera_mode = "wrist_only"
    family = "k3_cost"

    def __init__(self, lib="current", kref=None, base_kwargs=None, **kwargs):
        if "base" in kwargs:
            raise ValueError("WristMixedJudge fixes its base to WristAWM")
        bk = {"lib": lib, **(base_kwargs or {})}
        if kref is not None:
            bk["kref"] = kref
        super().__init__(base="exp.offline_search.rounds.r04.k3_cost.method:WristAWM", base_kwargs=bk, **kwargs)

    def _fit_thresholds(self, Fv, FT):
        return super()._fit_thresholds(WristView(Fv), FT)

    def _calibrate(self, Fv, FT):
        return super()._calibrate(WristView(Fv), FT)

    def query(self, q):
        return super().query(WristView(q))
