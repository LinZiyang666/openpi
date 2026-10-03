"""Round 7 (fable): GR00T gated follow without the rescue delay.  New class only; round-5/6 classes untouched.

Round-6 finding: with gated follow GR00T's first no-progress call comes ~5 decisions later than the control's on the same
episodes (fewer looks, and the look after a follow block retrieves inflated progress), and the GR00T stack has no second
rescue.  This variant removes the delay in two task-agnostic ways:
  1. a follow block is granted only when the stage gate and state valve pass (as fg) AND the episode is on the
     demonstration's pace at the anchor (lag <= ``pace_lag`` decisions) AND the no-progress memo is clean (span 0);
  2. the look right after a follow block runs a pace-consistency check: if the pace lag grew by >= ``lag_jump`` decisions
     (default 2: the three-decision follow stretch advanced the retrieved library progress by at most one step)
     over the follow stretch (the 15 blind controls delivered less than their share of progress), the policy is called at
     that look (forced MISS, reason 94) instead of waiting for the no-progress span to accumulate.
Everything else (judge, corrector, follow serving, lifecycle) is the round-5 LookCostGroot.
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult
from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r07.c1_follow.methods import ExtensionPlan
from exp.offline_search.rounds.r09.explore_fable.round5.tools.methods import LookCostGroot

POSTFOLLOW_REASON = 94.0          # os_reason of the post-follow pace MISS (93 = empty grasp, 91 = escalation)
POSTFOLLOW_BIT = 1 << 12


def _ident(obj):
    ep = getattr(obj, "episode", obj)
    return str(getattr(ep, "uid", ep))


class LookCostGrootPace(LookCostGroot):
    family = "r9f7_lookcost_groot_pace"
    _LC7_OWN = ("lc7_lag_jump", "lc7_pending", "lc7_n_revoked", "lc7_n_forced")

    def __init__(self, pace_lag=1, lag_jump=2.0, **kw):
        if kw.get("follow_blocks", 0) != 1:
            raise ValueError("LookCostGrootPace is the gated-follow variant: follow_blocks must be 1")
        if isinstance(lag_jump, bool) or not np.isfinite(lag_jump) or lag_jump <= 0:
            raise ValueError("lag_jump must be a positive finite number of decisions")
        super().__init__(pace_lag=pace_lag, **kw)
        self.lc7_lag_jump = float(lag_jump)
        self.lc7_pending = None
        self.lc7_n_revoked = self.lc7_n_forced = 0
        self.gm_name = f"R9F7_lookcost_groot_pace_L{float(pace_lag):g}_J{self.lc7_lag_jump:g}"

    def fit(self, lib, ctx):
        with open(self.gm_onlynp_fit, "rb") as f:
            src = FitUnpickler(f).load()["method"]
        clash = sorted(k for k in self._LC7_OWN if k in vars(src))
        if clash:
            raise api.ContractError("round-7 attributes shadow frozen judge attributes: %s" % clash)
        own = {k: getattr(self, k) for k in self._LC7_OWN}
        super().fit(lib, ctx)
        vars(self).update(own)
        if self.lc_follow is None or self.lc_wrist is not None:
            raise api.ContractError("LookCostGrootPace needs gated follow and no wrist looks")
        if getattr(self, "progress_guard", None) != "noprog_span" or not callable(getattr(self, "_progress", None)):
            raise api.ContractError("frozen judge has no no-progress span memo")   # _noprog_span itself is set at reset()
        self.fit_info = dict(self.fit_info, lc7_pace_lag=self.lc_pace_lag, lc7_lag_jump=self.lc7_lag_jump,
                             lc7_follow_gate="stage+valve+pace+clean_memo", lc7_post_follow_check="lag_jump_forced_miss")

    def reset(self, episode):
        super().reset(episode)
        self.lc7_pending = None
        self.lc7_n_revoked = self.lc7_n_forced = 0

    def invalidate_anchor(self):
        super().invalidate_anchor()
        self.lc7_pending = None

    def query(self, q):
        pending, self.lc7_pending = self.lc7_pending, None
        res = super().query(q)
        ex = dict(res.extras or {})
        lag = float(self.lc_last_lag)
        span = float(getattr(self, "_noprog_span", 0) or 0)
        revoked = 0.0
        plan = self.lc_follow_plan
        if plan is not None and plan.blocks and (not np.isfinite(lag) or lag > self.lc_pace_lag or span > 0):
            self.lc_follow_plan = ExtensionPlan(0, plan.structural, plan.stage_ok, {**plan.extras, "os_sf_granted": 0.0})
            self.lc_plan_extras["os_lc_fgrant"] = 0.0
            ex["os_lc_fgrant"] = 0.0
            revoked, self.lc7_n_revoked = 1.0, self.lc7_n_revoked + 1
        forced, jump = 0.0, float("nan")
        post = pending is not None and pending["episode"] == _ident(q) and int(q.step) == pending["step"] + 1
        if post:
            jump = lag - pending["anchor_lag"]
            if np.isfinite(jump) and jump >= self.lc7_lag_jump and not ex.get("os_force_miss"):
                ex.update(os_force_miss=1.0, os_reason=POSTFOLLOW_REASON, os_flags=float(int(ex.get("os_flags", 0)) | POSTFOLLOW_BIT))
                if self._s.get("flag"):
                    self._s["flag"][-1] = 1
                forced, self.lc7_n_forced = 1.0, self.lc7_n_forced + 1
        ex = {"os_lc7_revoked": revoked, "os_lc7_post": float(post), "os_lc7_jump": jump, "os_lc7_forced": forced, **ex}
        return api.Result(res.topk, res.scores, res.confidence, action=res.action, library=res.library, extras=ex)

    def blind_step(self, bq):
        res = super().blind_step(bq)
        if isinstance(res, BlindResult) and float((res.extras or {}).get("os_sf_source", 0) or 0) > 0:
            self.lc7_pending = dict(episode=_ident(bq), step=int(bq.step), anchor_lag=float(self.lc_last_lag))
        return res
