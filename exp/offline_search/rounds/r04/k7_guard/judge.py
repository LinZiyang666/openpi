"""Stock stuck detection on vision; endpoint-confirmed motion across blind gaps.

For consecutive real vision anchors a < b, the stock min-camera centred cosine
confirms every transition in (a, b] iff it meets the stock c_thr. Each transition
also needs its own raw valid-state L2 motion < stock m_thr. The trailing run of
these confirmed transitions is stuck_n. An unbounded prefix is never confirmed.
Blind decisions never evaluate this guard or acquire fabricated keys. Endpoint
agreement cannot exclude out-and-back visual motion inside a gap.

Calibration, blind stepping, and span progress remain K1's. With an all-vision
history, execute stock MixedJudge directly, retaining even its diagnostic fields.
"""
from __future__ import annotations

import math
import numpy as np

from exp.offline_search.harness import api, dims
from exp.offline_search.rounds.r03.h3_judge.judge import MixedJudge, V7, core
from exp.offline_search.rounds.r04.k1_blind.judge import BlindMixedJudge as K1BlindMixedJudge


class VisionConfirmedBlindMixedJudge(K1BlindMixedJudge):
    family = "k7_guard"

    def __init__(self, stuck_guard="vision_confirmed", **kwargs):
        if stuck_guard not in ("vision_confirmed", "dense"):
            raise ValueError("stuck_guard must be vision_confirmed or dense")
        self.stuck_guard = stuck_guard
        super().__init__(**kwargs)
        if stuck_guard == "vision_confirmed" and self.memo_reset_after_miss:
            raise ValueError("stock parity requires memo_reset_after_miss=False")
        self.name = f"K7_{stuck_guard}__{self.name}"

    def fit(self, lib, ctx):
        # K1 corrects GR00T terminal sign whereas stock does not. Do not silently
        # claim stock parity for two different terminal definitions.
        if ctx.model != "pi05":
            raise api.SkipCell("K7 supports pi05 only; GR00T stock terminal semantics differ")
        super().fit(lib, ctx)

    def reset(self, episode):
        super().reset(episode)
        self._guard_identity = (str(episode.uid), int(episode.task_id))

    def confirmed_stuck(self, q):
        """Pure online-history computation; no calibration/anchor state mutation."""
        step, task = int(q.step), int(q.task_id)
        if not step or task not in self.M0:
            return 0
        hv = np.asarray(getattr(q, "hist_has_vision", np.ones(step, bool)), bool)
        if len(hv) != step or len(q.hist_rs) != step:
            raise api.ContractError("K7 requires dense step-aligned state and vision histories")
        anchors = np.r_[np.flatnonzero(hv), step]
        count = 0
        for a, b in zip(anchors[-2::-1], anchors[:0:-1]):
            # Stop before touching keys when no trailing motion can qualify.
            for j in range(int(b), int(a), -1):
                now = q.rs if j == step else q.hist_rs[j]
                prev = q.hist_rs[j - 1]
                motion = float(np.linalg.norm(
                    dims.valid_state(np.asarray(now, np.float32), self.model)
                    - dims.valid_state(np.asarray(prev, np.float32), self.model)))
                if not motion < self.m_thr:
                    return count
                if j == b:
                    v0 = q.key_v0 if b == step else q.hist_key_v0[b]
                    v1 = q.key_v1 if b == step else q.hist_key_v1[b]
                    cosine = min(core.centred_cos(v0, q.hist_key_v0[a], self.M0[task]),
                                 core.centred_cos(v1, q.hist_key_v1[a], self.M1[task]))
                    if not cosine >= self.c_thr:
                        return count
                count += 1
        return count

    def query(self, q):
        identity = (str(q.episode.uid), int(q.task_id))
        if getattr(self, "_guard_identity", None) != identity:
            self.reset(q.episode)
        if self.stuck_guard == "dense":
            return super().query(q)
        hv = getattr(q, "hist_has_vision", None)
        if hv is None or np.all(hv):
            # No normalized-motion substitution and no V7 feature substitution.
            res = MixedJudge.query(self, q)
            self._progress(q, int(res.topk[0]))
            return res
        if self.progress_guard != "noprog_span":
            raise api.ContractError("blind gaps require progress_guard=noprog_span")
        count = self.confirmed_stuck(q)
        self._dense_stuck = count  # inherited feature hook; stock library fit unchanged
        try:
            res = V7.query(self, q)
        finally:
            self._dense_stuck = None
        # The following is K1's span guard, replacing ONLY its stuck count and
        # motion diagnostic. Overtime intentionally consumes the corrected count.
        ex, s, C = res.extras, self._s, self.C
        s["stuck_n"] = count
        step, top1 = int(q.step), int(res.topk[0])
        span = self._progress(q, top1)
        pa = q.prev_a_exec
        gexec = 0. if not step or pa is None else (1. if pa[4, 6] >= 0 else -1.)
        closed = gexec < 0 if self.model == "groot" else gexec > 0
        w = core.kernel_weights(np.asarray(res.scores, np.float64), self.T)
        vote = float((w / w.sum()) @ np.where(C.HD[res.topk, 6] >= 0, 1., -1.))
        gprop = 1. if res.action[0, 6] >= 0 else -1.
        term = bool(C.nxt[top1] < 0)
        overtime, lag = ex.get("overtime", float("nan")), ex.get("lag", float("nan"))
        disp = ex.get("disp", float("nan"))
        flags = 0
        if self.guards:
            if count >= self.stuck_thr:
                flags |= 1
            if term and closed:
                flags |= 2
            if overtime > 1 and lag > self.lag_thr and count >= 1:
                flags |= 4
            if span >= self.noprog_n - 1:
                flags |= 8
        if "disp" in self.events and disp >= self.disp_thr:
            flags |= 16
        if "grip" in self.events and gexec and gexec != gprop and abs(vote) < self.vote_thr:
            flags |= 32
        self._memo_sync(s, step)
        s["prog"].append((float(C.prog[top1]), max(int(C.ep_len[top1]) - 1, 1)))
        s["flag"].append(int(bool(flags)))
        reason, phase, conf = 0, 0, float(res.confidence)
        if flags:
            reason = (flags & -flags).bit_length()
            if self.burst > 1:
                s["burst_end"], s["ret_end"] = step + self.burst, step + self.burst + self.ret_hold
        elif step < s["burst_end"]:
            reason, phase = 7, 1
        elif step < s["ret_end"]:
            if int((np.asarray(q.hist_hit)[s["burst_end"]:step] == 1).sum()) < self.ret_n:
                phase, conf = 2, conf - self.ret_margin
            else:
                s["ret_end"] = 0
        out = dict(os_force_miss=float(reason != 0), os_reason=float(reason), os_flags=float(flags),
                   os_phase=float(phase), os_conf_raw=float(res.confidence), noprog_span=float(span),
                   dense_motion_guard=0., stuck_n=float(count), motion=ex.get("motion", float("nan")),
                   top1_prog=float(C.prog[top1]), term1=float(term), gexec=gexec, gprop=gprop, vote=vote,
                   overtime=overtime, lag=lag, disp=disp,
                   burst_left=float(max(s["burst_end"] - step - 1, 0)), vision_confirmed_guard=1.)
        for key in ("pred_err", "zsum", "regime", "dnn", "vis", "vself"):
            if key in ex:
                out[key] = ex[key]
        out = {k: float(v) for k, v in out.items() if math.isfinite(v)}
        return api.Result(res.topk, res.scores, conf, action=res.action, library=res.library, extras=out)


BlindMixedJudge = VisionConfirmedBlindMixedJudge
