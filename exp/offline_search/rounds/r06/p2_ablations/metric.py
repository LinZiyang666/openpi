"""Exact M_tau = I ablation; all other AWM fitting/query/blind code stays stock."""
from types import FunctionType

import numpy as np

from exp.offline_search.rounds.r02.g1_awm.awm import AWM
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM


def identity_metric(X, H, ep, nn=3, lam=.1, rank=0):
    """Stock per-task z-score, identity in the full 136-dimensional feature space."""
    if X.shape[1] != 136 or rank:
        raise ValueError("identity ablation requires full 136-dimensional joint features")
    return X.mean(0), X.std(0) + 1e-6, np.eye(X.shape[1], dtype=np.float64)


# AWM.fit has no metric hook. Bind its UNCHANGED code to a private globals dict;
# do not monkeypatch its module (safe for concurrent ordinary AWM fits).
_identity_fit = FunctionType(AWM.fit.__code__, {**AWM.fit.__globals__, "fit_metric": identity_metric},
                             name="identity_awm_fit", argdefs=AWM.fit.__defaults__, closure=AWM.fit.__closure__)


class IdentityBlindAWM(BlindAWM):
    family = "r6_p2_metric"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if (self.features != "joint" or self.codes != 0 or self.state_scale != 1.
                or not self.early or self.step0_joint):
            raise ValueError("P2 preserves A's joint/full-rank/unit-state-scale/early metric structure")
        self.name = "P2_identity__" + self.name

    def fit(self, lib, ctx):
        if self.fit_data != "same":
            raise ValueError("P2 requires A's fit_data=same")
        records = []

        def record_identity(X, *args, **kwargs):
            mean, std, W = identity_metric(X, *args, **kwargs)
            records.append((X, mean, std))
            return mean, std, W

        fit = FunctionType(AWM.fit.__code__, {**_identity_fit.__globals__, "fit_metric": record_identity},
                           argdefs=AWM.fit.__defaults__, closure=AWM.fit.__closure__)
        fit(self, lib, ctx)
        # The stock early affine distance uses float32 differences of large
        # terms. Direct float64 differences avoid ranking inversions at small
        # early standard deviations. Branches and their fit rows stay identical.
        assert len(records) == 2 * len(self.tasks)
        for i, T in enumerate(self.tasks.values()):
            X, mean, std = records[2 * i]
            _, mean0, std0 = records[2 * i + 1]
            T.euclidean = ((mean0, std0, (X - mean0) / std0),
                           (mean, std, (X - mean) / std))
            for params in T.euclidean:
                for arr in params:
                    arr.flags.writeable = False
        deployed = lib if self.cand_name == "current" else ctx.open_library(self.cand_name)
        self._fit_blind(deployed)

    def _dist(self, q):
        # Preserve stock projection, regime selection, and post-MISS continuity.
        # Only distance evaluation changes (same identity quadratic form).
        T, step, regime, k0, k1, xv, rs8, _, _, c, _ = super()._dist(q)
        mean, std, Z = T.euclidean[0 if regime == 0 else 1]
        x = np.concatenate([xv, rs8]).astype(np.float64)
        d = np.linalg.norm(Z - (x - mean) / std, axis=1)
        med = float(np.median(d)) + 1e-12
        dt = d / med + self.lam_c * c / T.s_c if regime == 1 else d
        return T, step, regime, k0, k1, xv, rs8, d, med, c, dt

    def bytes_per_entry(self):
        # Two 136-d float64 normalized candidate arrays, retained in addition
        # to stock codes so the inherited fitting/confidence path is unchanged.
        return super().bytes_per_entry() + 2 * 136 * 8
