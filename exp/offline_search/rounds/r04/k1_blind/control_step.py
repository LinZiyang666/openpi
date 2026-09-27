"""R4-C virtual control-step index: G (geometry) and GS (geometry + recorded-control splice)."""
from __future__ import annotations
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r02.g1_awm.awm import AWM, _kernel_w, _masked_min
from exp.offline_search.rounds.r04.k1_blind.blind_awm import consecutive_next


class ControlStepLibrary(AWM):
    family = "k1_blind"

    def __init__(self, ablation="GS", offsets=(0, 1, 2, 3, 4), **kw):
        if ablation not in ("G", "GS"):
            raise ValueError("ablation must be G or GS")
        if tuple(offsets) not in ((0, 1, 2, 3, 4), (0, 2, 4)):
            raise ValueError("offsets must be [0,1,2,3,4] or [0,2,4]")
        if kw.get("norm_cap", 0) or kw.get("hyst", 0) or kw.get("insure", False):
            raise ValueError("control_step_library is plain AWM with no insurance")
        super().__init__(**kw)
        self.ablation, self.offsets = ablation, tuple(offsets)
        self.name = f"CSL_{ablation}_o{''.join(map(str, offsets))}__{self.name}"

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        lib = lib if self.cand_name == "current" else ctx.open_library(self.cand_name)
        self._fit_edges(lib)
        self._fit_virtual_scales()

    def _fit_edges(self, lib):
        self.control_next = consecutive_next(lib)
        self.edge_l2 = np.zeros(lib.L, np.float32)
        for T in self.tasks.values():
            valid = self.control_next[T.rows] >= 0
            pos = np.flatnonzero(valid)
            npos = np.searchsorted(T.rows, self.control_next[T.rows[pos]])
            d = T.Z[npos].astype(np.float64) - T.Z[pos].astype(np.float64)
            self.edge_l2[T.rows[pos]] = np.sum(d * d, axis=1).astype(np.float32)
        self.control_next.flags.writeable = False
        self.edge_l2.flags.writeable = False

    def _virtual(self, T, d2):
        """Analytic chord projection, rounded to one offset per parent; ties go lower."""
        nxt = self.control_next[T.rows]
        pos = np.searchsorted(T.rows, np.where(nxt >= 0, nxt, T.rows))
        ell = self.edge_l2[T.rows].astype(np.float64)
        d2 = np.asarray(d2, np.float64)
        alpha = (d2 - d2[..., pos] + ell) / (2 * np.where(ell > 0, ell, 1.))
        stride = 1 if len(self.offsets) == 5 else 2
        offsets = (np.clip(np.ceil(np.clip(5 * alpha, 0, 4) / stride - .5), 0, 4 // stride)
                   * stride).astype(np.int8)
        offsets[..., (ell == 0) | (nxt < 0)] = 0
        a = offsets.astype(np.float64) / 5.
        value = (1 - a) * d2 + a * d2[..., pos] - a * (1 - a) * ell
        return np.sqrt(np.maximum(value, 0)), offsets

    def aligned_chunks(self, rows, offsets):
        chunks = self.act[rows].copy()
        if self.ablation == "G":
            return chunks
        for i, (row, offset) in enumerate(zip(rows, offsets)):
            if not offset:
                continue  # offset zero is the entire original chunk, including its tail
            row, offset = int(row), int(offset)
            written = 0
            chunks[i] = 0
            while written < self.H:
                n = min(5 - offset, self.H - written)
                chunks[i, written:written + n, :7] = self.act[row, offset:offset + n, :7]
                written += n
                nxt = int(self.control_next[row])
                if nxt < 0:
                    # End of actually executed library controls: hold final head control.
                    chunks[i, written:, :7] = self.act[row, 4, :7]
                    break
                row, offset = nxt, 0
        return chunks

    def _fit_virtual_scales(self):
        pq = {key: [] for key in ("d1", "disp", "dst")}
        for T in self.tasks.values():
            Z = T.Z.astype(np.float64)
            z2 = np.sum(Z * Z, axis=1)
            ep = self.lib_ep[T.rows]
            kk = min(self.k, len(ep) - int(np.bincount(np.unique(ep, return_inverse=True)[1]).max()))
            kk = max(kk, 1)
            for lo in range(0, len(Z), 128):
                hi = min(lo + 128, len(Z))
                d2 = np.maximum(z2[lo:hi, None] - 2 * Z[lo:hi] @ Z.T + z2, 0)
                d, offsets = self._virtual(T, d2)
                d[ep[lo:hi, None] == ep[None, :]] = np.inf
                idx = np.argsort(d, axis=1, kind="stable")[:, :kk]
                dk = np.take_along_axis(d, idx, 1)
                ww = _kernel_w(dk - dk[:, :1], self.kref)
                pq["d1"].extend(dk[:, 0])
                for j in range(hi - lo):
                    chunks = self.aligned_chunks(T.rows[idx[j]], offsets[j, idx[j]])
                    heads = (chunks[:, :5, :7].astype(np.float64) / self.sig).reshape(kk, 35)
                    head = (heads * ww[j, :, None]).sum(0) / ww[j].sum()
                    pq["disp"].append(float(np.sqrt(np.mean((heads[:5] - head) ** 2))))
            rs = T.RS.astype(np.float64)
            pq["dst"].extend(_masked_min(rs, rs, ep, ep) / T.s_d)
        v = {key: np.asarray(value) for key, value in pq.items()}
        self.s_a = float(np.median(v["disp"])) + 1e-6
        self.zmu = {key: float(np.mean(-value)) for key, value in v.items()}
        self.zsd = {key: float(np.std(-value)) + 1e-9 for key, value in v.items()}
        zs = (-v["d1"] - self.zmu["d1"]) / self.zsd["d1"] + (-v["disp"] - self.zmu["disp"]) / self.zsd["disp"]
        if self.features == "joint":
            zs += (-v["dst"] - self.zmu["dst"]) / self.zsd["dst"]
        self.zs_sd = float(np.std(zs)) + 1e-9

    def query(self, q):
        if q.step == 0 or q.prev_hit is False:
            return super().query(q)
        T, step, regime, k0, k1, xv, rs, d, med, c, dt = self._dist(q)
        dv, offsets = self._virtual(T, d.astype(np.float64) ** 2)
        idx = np.argsort(dv, kind="stable")[:self.k]
        rows, off = T.rows[idx], offsets[idx]
        dk = dv[idx]
        w = _kernel_w(dk - dk[0], self.kref)
        wn = (w / w.sum()).astype(np.float32)
        chunks = self.aligned_chunks(rows, off)
        action = np.tensordot(wn, chunks, 1)
        hd = (chunks[:5, :5, :7] / self.sig).reshape(-1, 35)
        disp = float(np.sqrt(np.mean((hd - (action[:5, :7] / self.sig).ravel()) ** 2)))
        dst = self._dst(T, rs)
        conf = self._conf(T, regime, float(dv.min()), disp, dst, float("nan"))
        ex = {"d1": float(dv.min()), "disp5": disp, "dst": dst, "regime": float(regime),
              "offset_nonzero_mass": float(wn @ (off > 0)),
              "fractional_lib_step": float(wn @ (self.lib_step[rows] + off / 5.)),
              "top1_fractional_step": float(self.lib_step[rows[0]] + off[0] / 5.),
              "head_grip_transitions": float(np.count_nonzero(np.diff(action[:5, 6] >= 0))),
              "lib_step": float(self.lib_step[rows[0]])}
        if q.prev_a_exec is not None:
            previous = np.asarray(q.prev_a_exec[:5, :7], np.float32)
            ex["head_rms_from_previous"] = float(np.sqrt(np.mean(((action[:5, :7] - previous) / self.sig) ** 2)))
            signs = action[:5, 6] >= 0
            ex["grip_event_replayed"] = float(np.any(np.diff(signs)) and np.array_equal(signs, previous[:, 6] >= 0))
        for i, value in enumerate(off):
            ex[f"offset_{i}"] = float(value)
        for i, value in enumerate(action[:5, :7].ravel()):
            ex[f"served_head_{i}"] = float(value)
        return api.Result(rows.astype(np.int64), -dk, float(conf), action=action, library=self.cand_name, extras=ex)

    def os_score_all(self, q):
        raise api.ContractError("ControlStepLibrary is pure-cache only: static MixedJudge heads do not describe splices")

    def bytes_per_entry(self):
        return super().bytes_per_entry() + 8
