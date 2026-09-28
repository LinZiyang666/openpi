"""K7's exact motion/gap rule, confirmed only by the available wrist key.

WristView maps both legacy camera slots to camera 1. Thus min(c, c) = c,
and library calibration computes the wrist distribution's own p95. The raw
state L2 threshold, strict motion comparison, inclusive cosine comparison,
trailing run, and missing-left-anchor semantics are inherited from K7.
No missing base-camera key or blind-row visual key is consumed.
"""
from exp.offline_search.rounds.r04.k3_cost.method import WristView
from exp.offline_search.rounds.r04.k7_guard.judge import VisionConfirmedBlindMixedJudge


class WristVisionConfirmedBlindJudge(VisionConfirmedBlindMixedJudge):
    camera_mode = "wrist_only"
    family = "q6_wrist_blind"

    def __init__(self, base_kwargs=None, **kwargs):
        if "base" in kwargs:
            raise ValueError("Q6 fixes the base to K1 BlindWristAWM / K3 WristAWM")
        if kwargs.get("stuck_guard", "vision_confirmed") != "vision_confirmed":
            raise ValueError("Q6 requires the wrist vision-confirmed stuck guard")
        if kwargs.get("progress_guard", "noprog_span") != "noprog_span":
            raise ValueError("Q6 blind serving requires progress_guard=noprog_span")
        if kwargs.get("c_pct", 95.) != 95. or kwargs.get("m_pct", 10.) != 10.:
            raise ValueError("Q6 uses library wrist p95 and K7 motion p10")
        bk = dict(serving="anchor_tail", budget=1, gates="budget_only")
        bk.update(base_kwargs or {})
        super().__init__(base="exp.offline_search.rounds.r04.k1_blind.wrist:BlindWristAWM",
                         base_kwargs=bk, **kwargs)
        self.name = "Q6_wrist__" + self.name

    def _fit_thresholds(self, Fv, FT):
        return super()._fit_thresholds(WristView(Fv), FT)

    def _calibrate(self, Fv, FT):
        return super()._calibrate(WristView(Fv), FT)

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        self.fit_info["q6_confirmation"] = dict(camera="wrist_only", percentile=95.,
            fit_library=self.fitname, deployed_library=self.libname,
            c_thr=self.c_thr, m_thr=self.m_thr, motion="K7_raw_valid_state_L2")

    def confirmed_stuck(self, q):
        return super().confirmed_stuck(WristView(q))

    def query(self, q):
        return super().query(WristView(q))
