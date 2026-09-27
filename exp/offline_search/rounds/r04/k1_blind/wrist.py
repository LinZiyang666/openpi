"""Composition for batch four: K3's wrist metric + K1's blind and gap guard adapters."""
from exp.offline_search.rounds.r04.k3_cost.method import WristAWM, WristView
from exp.offline_search.rounds.r04.k1_blind.blind_awm import _BlindMixin
from exp.offline_search.rounds.r04.k1_blind.judge import BlindMixedJudge


class BlindWristAWM(_BlindMixin, WristAWM):
    pass


class BlindWristMixedJudge(BlindMixedJudge):
    camera_mode = "wrist_only"

    def __init__(self, base_kwargs=None, **kw):
        super().__init__(base="exp.offline_search.rounds.r04.k1_blind.wrist:BlindWristAWM",
                         base_kwargs=base_kwargs, **kw)

    def _fit_thresholds(self, Fv, FT):
        return super()._fit_thresholds(WristView(Fv), FT)

    def _calibrate(self, Fv, FT):
        return super()._calibrate(WristView(Fv), FT)

    def query(self, q):
        return super().query(WristView(q))
