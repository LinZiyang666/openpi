"""Round 8 (fable): pace-gated wrist looks on top of R7's stage-wrist / stage-follow pure-cache stack (pi0.5, 500-demo cells).

``PaceWrist`` is R7's ``StageWrist`` (per-request camera plan over an A or StageFollow base, wrist retrieval in the refitted
72-D wrist metric) with one change in the camera plan: a look is wrist-only when the chain to the next anchor is structurally
valid and the state is inside the library valve (R7 reasons 3 and 5 absent) AND the episode is on the demonstration's pace
(top-1 library step lag at the anchor <= ``pace_lag`` decisions); R7's gripper-stage unanimity / event-proximity conditions
(reasons 2 and 4) are not required.  This is the round-5 "pace" gate, now without any judge or policy call in the loop.
New class only; the R7 classes are untouched.
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.rounds.r07.c2_wrist.method import StageWrist

FOLLOW_BASE = "exp.offline_search.rounds.r07.c1_follow.methods:StageFollow"


class PaceWrist(StageWrist):
    family = "r9f8_pace_wrist"

    def __init__(self, *, pace_lag=1, **kw):
        if isinstance(pace_lag, bool) or not np.isfinite(pace_lag) or pace_lag < 0:
            raise ValueError("pace_lag must be a finite nonnegative number of decisions")
        super().__init__(**kw)
        self.pace_lag = float(pace_lag)
        self.name = f"PW_L{self.pace_lag:g}" + self.name[2:]

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        if self.enabled and not hasattr(self.base, "lib_step"):
            raise ValueError("pace gate needs the base's library step table")
        self.fit_info = dict(getattr(self, "fit_info", {}) or {}, pace_lag=self.pace_lag, camera_gate="pace+chain+valve",
                             follow_blocks=int(getattr(self.base, "follow_extend_blocks", 0)))

    def _plan(self, current, elapsed, target, command_action=None):
        super()._plan(current, elapsed, target, command_action)
        a = self._anchor
        if not self.enabled or a is None:
            return
        reason = int(self._plan_extras.get("os_sw_reason", 1))
        lag = float(int(a["step"]) - int(self.base.lib_step[int(a["rows"][0])]))
        wrist = reason in (0, 2, 4) and lag <= self.pace_lag
        self.next_camera_mode = "wrist_only" if wrist else "full"
        self._plan_extras.update(os_sw_next_camera=float(wrist), os_pw_lag=lag, os_pw_pace=float(lag <= self.pace_lag))
