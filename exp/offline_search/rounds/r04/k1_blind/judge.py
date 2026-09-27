"""R4 gap-aware MixedJudge and opt-in progress memo reset (B2)."""
from __future__ import annotations
import math
import numpy as np
from exp.offline_search.harness import api, dims
from exp.offline_search.rounds.r03.h3_judge.judge import MixedJudge, V7, core
from exp.offline_search.rounds.r04.k1_blind.blind_awm import LookReason

BASE = "exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM"


class BlindMixedJudge(MixedJudge):
    family = "k1_blind"

    def __init__(self, base=BASE, base_kwargs=None, progress_guard="noprog_span", memo_reset_after_miss=False,
                 guards=True, events="none", **kw):
        if progress_guard not in ("noprog_span", "noprog_n"):
            raise ValueError("progress_guard must be noprog_span or noprog_n")
        self.progress_guard = progress_guard
        self.memo_reset_after_miss = bool(memo_reset_after_miss)
        super().__init__(base=base, base_kwargs=base_kwargs, guards=guards, events=events, **kw)
        if progress_guard == "noprog_span" and not hasattr(self.base, "blind_step"):
            raise ValueError("noprog_span needs a blind-capable base with library state statistics")
        self.name = f"R4_{progress_guard}_mr{int(self.memo_reset_after_miss)}__{self.name}"
        self._dense_stuck = None

    def reset(self, episode):
        super().reset(episode)
        self._vision_progress = []
        self._noprog_span = 0
        self._dense_stuck = None

    def invalidate_anchor(self):
        if hasattr(self.base, "invalidate_anchor"):
            self.base.invalidate_anchor()

    @property
    def last_blind_extras(self):
        return getattr(self.base, "last_blind_extras", {})

    def blind_step(self, bq):
        if not hasattr(self.base, "blind_step"):
            return LookReason(8, "base_has_no_blind_step")
        reason = self.base.lifecycle_reason(bq)
        if reason is not None:
            return reason
        if self.progress_guard == "noprog_n":
            # The unchanged legacy guard assumes adjacent visual observations.
            return LookReason(8, "noprog_n_requires_vision")
        # Once a real anchor shows no advancement, keep looking until it advances
        # (or the explicitly selected memo-reset variant clears it after a MISS).
        if self.guards and self.progress_guard == "noprog_span" and self._noprog_span > 0:
            return LookReason(8, "noprog_span")
        return self.base.blind_step(bq)

    def _self_change(self, q):
        if self.progress_guard == "noprog_n":
            return super()._self_change(q)
        hv = getattr(q, "hist_has_vision", None)
        if q.step and hv is not None and not bool(hv[-1]):
            rs = dims.valid_state(np.asarray(q.rs, np.float32), self.model)
            prev = dims.valid_state(np.asarray(q.hist_rs[-1], np.float32), self.model)
            return float(np.linalg.norm(rs - prev)), float("nan")
        return super()._self_change(q)

    def _features(self, *args, **kw):
        if self.progress_guard == "noprog_span" and self._dense_stuck is not None:
            args = list(args)
            # V7 positional argument `stuck` is index 10; library calibration stays stock.
            if len(args) > 10:
                args[10] = min(self._dense_stuck, 5)
            else:
                kw["stuck"] = min(self._dense_stuck, 5)
        return super()._features(*args, **kw)

    def _progress(self, q, top1):
        step = int(q.step)
        self._vision_progress = [x for x in self._vision_progress if x[0] < step]
        if self.memo_reset_after_miss and q.prev_hit is False:
            self._vision_progress = []
        self._vision_progress.append((step, float(self.C.prog[top1]), max(int(self.C.ep_len[top1]) - 1, 1)))
        span = 0
        for prev, now in zip(self._vision_progress[:-1], self._vision_progress[1:]):
            span = span + now[0] - prev[0] if (now[1] - prev[1]) * now[2] <= self.prog_eps else 0
        self._noprog_span = span
        return span

    def query(self, q):
        if self.progress_guard == "noprog_n":
            if self.memo_reset_after_miss and q.prev_hit is False:
                # Clear only progress comparisons. Hard guards, burst and visual state survive.
                self._s["prog"] = [(float("nan"), float("nan"))] * int(q.step)
            return super().query(q)
        motion = self.base.dense_motion(q)
        count = 0
        for value in motion[::-1]:
            if not value < self.base.motion10[int(q.task_id)]:
                break
            count += 1
        self._dense_stuck = count
        try:
            res = V7.query(self, q)  # unchanged selection/synthesis and V7, new dense motion feature
        finally:
            self._dense_stuck = None
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
                   dense_motion_guard=1., stuck_n=float(count), motion=float(motion[-1]) if len(motion) else 0.,
                   top1_prog=float(C.prog[top1]), term1=float(term), gexec=gexec, gprop=gprop, vote=vote,
                   overtime=overtime, lag=lag, disp=disp,
                   burst_left=float(max(s["burst_end"] - step - 1, 0)))
        for key in ("pred_err", "zsum", "regime", "dnn", "vis", "vself"):
            if key in ex:
                out[key] = ex[key]
        out = {k: float(v) for k, v in out.items() if math.isfinite(v)}
        return api.Result(res.topk, res.scores, conf, action=res.action, library=res.library, extras=out)


class MemoResetMixedJudge(BlindMixedJudge):
    """B2: stock MixedJudge except clearing progress comparisons after an executed MISS."""
    def __init__(self, **kw):
        kw.setdefault("progress_guard", "noprog_n")
        kw.setdefault("memo_reset_after_miss", True)
        super().__init__(**kw)
