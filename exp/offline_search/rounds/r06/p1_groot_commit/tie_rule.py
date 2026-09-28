"""TEST-ONLY control for the nesting check (never deployed): A with the G3 wrapper's documented top-k tie rule.

AWM.query picks its k=16 kernel members with ``np.argpartition(dt, k-1)[:k]`` and then sorts them by (dt, index).
At an exact float32 distance tie across the k-th slot, argpartition keeps an arbitrary one of the tied rows. The G3
wrapper under MixedJudge / K7 / K10 / C10 / P1 (``g3_core.topk_pos``) keeps the lowest position ("identical to
np.argsort(-S, kind='stable')[:k] ... including exact tie handling"). ``StableTieBlindAWM`` is BlindAWM whose
AWM.query uses ``np.argsort(dt, kind='stable')[:k]``: the same (dt, index) order, lowest index among ties.
Everything else is AWM.query verbatim (awm.py:442-475), so it differs from A only at k-th-slot ties.
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r02.g1_awm.awm import AWM, _kernel_w
from exp.offline_search.rounds.r04.k1_blind.blind_awm import _BlindMixin


class StableTieAWM(AWM):
    def query(self, q):
        T, step, regime, k0, k1, xv, rs8, d, med, c, dt = self._dist(q)
        with self.prof.section("synth"):
            n = dt.shape[0]
            k = min(self.k, n)
            idx = np.argsort(dt, kind="stable")[:k]          # the only change: stable ties (lowest index)
            dk = dt[idx].astype(np.float64)
            w = _kernel_w(dk - dk[0], self.kref)
            rows = T.rows[idx]
            wn, Ck, a = self._mix(rows, w)
            disp5 = self._disp5(T, a, idx[:5])
        d1 = float(d.min())
        dst = self._dst(T, rs8)
        c0 = float(c[idx[0]]) if c is not None else float("nan")
        conf = self._conf(T, regime, d1, disp5, dst, c0)
        ex = {"d1": d1, "d1_rel": d1 / med, "disp5": disp5, "dst": dst, "regime": float(regime),
              "lib_ep": float(self.lib_ep[rows[0]]), "lib_step": float(self.lib_step[rows[0]]),
              "w_eff": float(1.0 / (wn.astype(np.float64) ** 2).sum())}
        if regime == 1:
            ex["c0"] = c0
        if step > 0:
            h0, h1 = q.hist_key_v0[-1], q.hist_key_v1[-1]
            s = 0.0
            for cur, prv, mu in ((k0, h0, self.mu0), (k1, h1, self.mu1)):
                u = cur - mu
                v = np.asarray(prv, np.float32) - mu
                s += float(u @ v) / max(float(np.sqrt((u @ u) * (v @ v))), 1e-12)
            ex["still"] = s
        self._insure(a, Ck, wn, step, ex)
        return api.Result(topk=rows.astype(np.int64), scores=-dt[idx].astype(np.float64), confidence=float(conf),
                          action=a, library=self.cand_name, extras=ex)


class StableTieBlindAWM(_BlindMixin, StableTieAWM):
    """Test-only A control: BlindAWM with the stable k-th-slot tie rule."""
