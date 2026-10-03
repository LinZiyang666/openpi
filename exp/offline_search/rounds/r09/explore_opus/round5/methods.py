"""Opus round-5: two task-agnostic gates on the guard calls of fable's leading round-3c stacks (IR reduction).

The stacks are used exactly as frozen (``explore_fable/round3/tools/methods.py``, sha fa881a57...): this module only
subclasses them and post-processes the verdict their ``query`` returns.  Nothing is task-indexed.

Gate P -- "on-pace silence" (``pace_lag_max`` = L0):
    A no-progress verdict (``os_reason == 4``) at a look whose pace lag ``decision index - library step of the top-1
    row`` is <= L0 is dropped: the decision is served by the (corrected) cache as a HIT, and the no-progress blind
    veto is lifted for the next decision, so the controller is exactly the cache until the next regular look.  The
    guard's own span statistic keeps accumulating, so a stall that persists is re-checked at the next look with a
    larger lag and called then.  Why: on discovery inits 0-19 a stall that is at most two decisions behind the
    demo's pace recovers at the next look as often without a call as with one (pure-cache replay vs real calls).
Gate C -- "call budget" (``call_budget`` = C):
    At most C guard calls (any reason: no-progress, escalation, empty grasp) per episode, counted from the committed
    history (vision decisions that were MISSes).  Afterwards every verdict is dropped and the blind veto lifted:
    the rest of the episode is the (corrected) cache.  Why: on inits 0-19 episodes that need more than ~20 calls
    almost never succeed, and those calls are a large share of the guard's cost.

Both thresholds are constants chosen on discovery inits 0-19 (``tools/gatesim.py``); ``None`` disables a gate, in
which case the outputs are identical to the parent stack (extras gain only the ``r9o5_*`` diagnostics; tested).

``force_noprog_at`` (debug / selftest only, production ``()``): at those decision indices a no-progress verdict is
forced before gating, so the gate paths are exercised in CPU selftests.
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r09.explore_fable.round3.tools.methods import NpGraspEsc3, NpGraspStackGroot3

NOPROG_REASON = 4.0
GATE_NONE, GATE_PACE, GATE_BUDGET = 0.0, 1.0, 2.0


def committed_calls(q):
    """Guard calls already executed in this episode: vision decisions whose committed outcome was a MISS."""
    step = int(q.step)
    if step <= 0:
        return 0
    hit = np.asarray(q.hist_hit)[:step]
    hv = np.asarray(getattr(q, "hist_has_vision", np.ones(step, bool)), bool)[:step]
    if len(hit) != step or len(hv) != step:
        raise api.ContractError("call budget needs step-aligned hit / vision histories")
    return int(np.sum((hit == 0) & hv))


class _CallGate:
    """Mixin over a fitted r3c stack: gate P and gate C on the final verdict; state under ``cg_``."""
    _CG_OWN = ("cg_pace_lag_max", "cg_budget", "cg_force_np", "cg_identity", "cg_quiet_step", "cg_name")

    def _cg_init(self, pace_lag_max, call_budget, force_noprog_at):
        if pace_lag_max is not None and (isinstance(pace_lag_max, bool) or not np.isfinite(pace_lag_max)):
            raise ValueError("pace_lag_max must be None or a finite number (decisions)")
        if call_budget is not None and (type(call_budget) is not int or call_budget < 0):
            raise ValueError("call_budget must be None or a nonnegative int")
        self.cg_pace_lag_max = None if pace_lag_max is None else float(pace_lag_max)
        self.cg_budget = call_budget
        self.cg_force_np = tuple(int(v) for v in (force_noprog_at or ()))
        self.cg_identity, self.cg_quiet_step = None, None
        tag = (f"_P{self.cg_pace_lag_max:g}" if self.cg_pace_lag_max is not None else "") + \
              (f"_C{call_budget}" if call_budget is not None else "")
        self.cg_name = "R9O5_gate" + (tag or "_off")

    def _cg_check_clash(self, keep):
        """The parent fit copies the frozen judge's fields over ``self``; none of them may shadow a gate field."""
        clash = sorted(k for k, v in keep.items() if k not in vars(self) or vars(self)[k] is not v)
        if clash:
            raise api.ContractError(f"frozen judge fields shadow gate fields: {clash}")

    def _cg_fit(self):
        if not hasattr(self.base, "lib_step"):
            raise api.ContractError("gate P needs the base's library steps")
        if getattr(self, "progress_guard", None) != "noprog_span":
            raise api.ContractError("gates need the only-no-progress (noprog_span) judge")
        self.name = f"{self.cg_name}__{self.name}"
        self.fit_info = dict(self.fit_info or {}, r9o5_pace_lag_max=self.cg_pace_lag_max, r9o5_call_budget=self.cg_budget,
                             r9o5_force_noprog_at=list(self.cg_force_np))

    def _cg_reset(self, episode):
        self.cg_identity = (str(getattr(episode, "uid", episode)),)
        self.cg_quiet_step = None

    def _cg_sync(self, q):
        ident = (str(getattr(q.episode, "uid", q.episode)),)
        if ident != self.cg_identity:
            self._cg_reset(q.episode)

    def _cg_apply(self, res, q):
        self._cg_sync(q)
        step = int(q.step)
        ex = dict(res.extras or {})
        if self.cg_force_np and step in self.cg_force_np and not ex.get("os_force_miss"):
            ex.update(os_force_miss=1.0, os_reason=NOPROG_REASON, os_flags=float(int(ex.get("os_flags", 0)) | 8))
            if self._s.get("flag"):
                self._s["flag"][-1] = 1
        lag = float(step - int(self.base.lib_step[int(res.topk[0])]))
        n_calls = committed_calls(q)
        reason = float(ex.get("os_reason", 0.0) or 0.0)
        gate = GATE_NONE
        if ex.get("os_force_miss"):
            if self.cg_budget is not None and n_calls >= self.cg_budget:
                gate = GATE_BUDGET
            elif self.cg_pace_lag_max is not None and reason == NOPROG_REASON and lag <= self.cg_pace_lag_max:
                gate = GATE_PACE
        if gate != GATE_NONE:
            ex.update(os_force_miss=0.0, os_reason=0.0, r9o5_gated_reason=reason)
            if self._s.get("flag"):
                self._s["flag"][-1] = 0
            self.cg_quiet_step = step
        else:
            self.cg_quiet_step = None
        ex.update(r9o5_lag=lag, r9o5_calls_before=float(n_calls), r9o5_gate=gate)
        return api.Result(res.topk, res.scores, res.confidence, action=res.action, library=res.library, extras=ex)

    def _cg_quiet(self, bq):
        step = int(bq.step)
        if self.cg_quiet_step is not None and step == self.cg_quiet_step + 1:
            return True
        return self.cg_budget is not None and committed_calls(bq) >= self.cg_budget

    def blind_step(self, bq):
        self._cg_sync(bq)
        if self._cg_quiet(bq) and getattr(self, "_noprog_span", 0) > 0:
            saved = self._noprog_span
            self._noprog_span = 0          # lift ONLY the no-progress look veto; lifecycle and budget gates unchanged
            try:
                return super().blind_step(bq)
            finally:
                self._noprog_span = saved
        return super().blind_step(bq)


class GatedNpGraspEsc3(_CallGate, NpGraspEsc3):
    """pi0.5 leading stack (only-no-progress + corrector on the judge path + pace-lag escalation) + gates P / C."""
    family = "r9o5_gated_np_corr_esc"

    def __init__(self, onlynp_fit="", corrected_fit="", lag_threshold=12, deadline=80, empty_aperture=0.001, closed_sign=1.0,
                 hold_decisions=2, burst=2, max_calls=2, force_trigger_at=(), pace_lag_max=None, call_budget=None,
                 force_noprog_at=()):
        NpGraspEsc3.__init__(self, onlynp_fit=onlynp_fit, corrected_fit=corrected_fit, lag_threshold=lag_threshold,
                             deadline=deadline, empty_aperture=empty_aperture, closed_sign=closed_sign,
                             hold_decisions=hold_decisions, burst=burst, max_calls=max_calls, force_trigger_at=force_trigger_at)
        self._cg_init(pace_lag_max, call_budget, force_noprog_at)

    def fit(self, lib, ctx):
        keep = {k: getattr(self, k) for k in self._CG_OWN}
        NpGraspEsc3.fit(self, lib, ctx)
        self._cg_check_clash(keep)
        self._cg_fit()

    def reset(self, episode):
        super().reset(episode)
        self._cg_reset(episode)

    def query(self, q):
        return self._cg_apply(super().query(q), q)


class GatedNpGraspStackGroot3(_CallGate, NpGraspStackGroot3):
    """GR00T leading stack (only-no-progress + corrector on the judge path) + gates P / C."""
    family = "r9o5_gated_np_corr_groot"

    def __init__(self, onlynp_fit="", corrected_fit="", empty_aperture=0.0009, closed_sign=-1.0, hold_decisions=2, burst=2,
                 max_calls=2, force_trigger_at=(), pace_lag_max=None, call_budget=None, force_noprog_at=()):
        NpGraspStackGroot3.__init__(self, onlynp_fit=onlynp_fit, corrected_fit=corrected_fit, empty_aperture=empty_aperture,
                                    closed_sign=closed_sign, hold_decisions=hold_decisions, burst=burst, max_calls=max_calls,
                                    force_trigger_at=force_trigger_at)
        self._cg_init(pace_lag_max, call_budget, force_noprog_at)

    def fit(self, lib, ctx):
        keep = {k: getattr(self, k) for k in self._CG_OWN}
        NpGraspStackGroot3.fit(self, lib, ctx)
        self._cg_check_clash(keep)
        self._cg_fit()

    def reset(self, episode):
        super().reset(episode)
        self._cg_reset(episode)

    def query(self, q):
        return self._cg_apply(super().query(q), q)
