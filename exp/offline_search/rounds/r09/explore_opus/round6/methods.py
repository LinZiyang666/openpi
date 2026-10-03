"""Opus round-6: "script exhausted -> bounded policy takeover" on fable's leading r3c stacks.

Signal (task-agnostic, no simulator truth): an episode ends on success, so if the cache's retrieved top-1 row is the
LAST row of its demonstration (rows to the demo end <= ``end_rows``) while the episode is still running ``dwell``
decisions after it first got there, the scripted demonstration has been played out without achieving the goal.
Action: a bounded takeover -- every fresh decision of the next ``window`` decisions is a policy call (the policy's
chunk then also serves the following blind decision via the existing policy-tail lifecycle), then control returns to
the unchanged stack.  At most one takeover per episode.  Verdicts the stack already forces are left as they are.

Classes (subclasses of the frozen r3c classes, ``explore_fable/round3/tools/methods.py`` sha fa881a57...):
  ExhaustNpGraspEsc3          pi0.5 stack (only-no-progress + corrector on the judge path + pace-lag escalation) + takeover
  ExhaustNpGraspStack3        pi0.5 only-no-progress + corrector, escalation REPLACED by the takeover
  ExhaustNpGraspStackGroot3   GR00T stack (only-no-progress + corrector) + takeover
Thresholds (end_rows, dwell) are fitted on discovery inits 0-19 (``tools/exhaust.py``); window = 24 decisions is the
round-2 constant (not tuned).  ``force_exhaust_at`` (debug / selftest only, production ``()``) starts the takeover at
those decision indices.
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r09.explore_fable.round3.tools.methods import NpGraspEsc3, NpGraspStack3, NpGraspStackGroot3

EXHAUST_REASON = 95.0


class _Exhaust:
    """Mixin over a fitted r3c stack; all state under ``xt_``."""
    _XT_OWN = ("xt_end_rows", "xt_dwell", "xt_window", "xt_force", "xt_identity", "xt_t_end", "xt_start", "xt_rte",
               "xt_name")

    def _xt_init(self, end_rows, dwell, window, force_exhaust_at):
        for v, nm in ((end_rows, "end_rows"), (dwell, "dwell"), (window, "window")):
            if type(v) is not int or v < 0:
                raise ValueError(f"{nm} must be a nonnegative int")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.xt_end_rows, self.xt_dwell, self.xt_window = end_rows, dwell, window
        self.xt_force = tuple(int(s) for s in (force_exhaust_at or ()))
        self.xt_identity, self.xt_t_end, self.xt_start, self.xt_rte = None, None, None, None
        self.xt_name = f"R9O6_exhaust_E{end_rows}_D{dwell}_W{window}"

    def _xt_check_clash(self, keep):
        clash = sorted(k for k, v in keep.items() if k not in vars(self) or vars(self)[k] is not v)
        if clash:
            raise api.ContractError(f"frozen judge fields shadow takeover fields: {clash}")

    def _xt_fit(self):
        lib_step = np.asarray(self.base.lib_step, np.int64)
        ep_len = np.asarray(self.C.ep_len, np.int64)
        if lib_step.shape != ep_len.shape:
            raise api.ContractError("library step / episode length arrays differ in size")
        rte = ep_len - 1 - lib_step
        if (rte < 0).any():
            raise api.ContractError("library step beyond its demonstration length")
        nxt = np.asarray(self.C.nxt)
        if not np.array_equal(rte == 0, nxt < 0):
            raise api.ContractError("last-row definition disagrees with the library successor table")
        self.xt_rte = rte
        self.name = f"{self.xt_name}__{self.name}"
        self.fit_info = dict(self.fit_info or {}, r9o6_end_rows=self.xt_end_rows, r9o6_dwell=self.xt_dwell,
                             r9o6_window=self.xt_window, r9o6_force_exhaust_at=list(self.xt_force))

    def _xt_reset(self, episode):
        self.xt_identity = (str(getattr(episode, "uid", episode)),)
        self.xt_t_end, self.xt_start = None, None

    def _xt_apply(self, res, q):
        ident = (str(getattr(q.episode, "uid", q.episode)),)
        if ident != self.xt_identity:
            self._xt_reset(q.episode)
        step = int(q.step)
        rte = int(self.xt_rte[int(res.topk[0])])
        at_end = rte <= self.xt_end_rows
        if at_end and self.xt_t_end is None:
            self.xt_t_end = step
        fire = False
        if self.xt_start is None:
            if self.xt_force and step in self.xt_force:
                fire = True
            elif at_end and self.xt_t_end is not None and step - self.xt_t_end >= self.xt_dwell:
                fire = True
            if fire:
                self.xt_start = step
        active = self.xt_start is not None and self.xt_start <= step < self.xt_start + self.xt_window
        ex = dict(res.extras or {}, r9o6_rte=float(rte), r9o6_t_end=float(self.xt_t_end if self.xt_t_end is not None else -1),
                  r9o6_start=float(self.xt_start if self.xt_start is not None else -1), r9o6_trigger=float(fire),
                  r9o6_takeover=float(active))
        if active and not ex.get("os_force_miss"):
            ex.update(os_force_miss=1.0, os_reason=EXHAUST_REASON, os_flags=float(int(ex.get("os_flags", 0)) | (1 << 12)))
            if self._s.get("flag"):
                self._s["flag"][-1] = 1
        return api.Result(res.topk, res.scores, res.confidence, action=res.action, library=res.library, extras=ex)


def _fit(self, parent, lib, ctx):
    keep = {k: getattr(self, k) for k in self._XT_OWN}
    parent.fit(self, lib, ctx)
    self._xt_check_clash(keep)
    self._xt_fit()


class ExhaustNpGraspEsc3(_Exhaust, NpGraspEsc3):
    family = "r9o6_exhaust_np_corr_esc"

    def __init__(self, onlynp_fit="", corrected_fit="", lag_threshold=12, deadline=80, empty_aperture=0.001, closed_sign=1.0,
                 hold_decisions=2, burst=2, max_calls=2, force_trigger_at=(), end_rows=0, dwell=2, window=24, force_exhaust_at=()):
        NpGraspEsc3.__init__(self, onlynp_fit=onlynp_fit, corrected_fit=corrected_fit, lag_threshold=lag_threshold,
                             deadline=deadline, empty_aperture=empty_aperture, closed_sign=closed_sign,
                             hold_decisions=hold_decisions, burst=burst, max_calls=max_calls, force_trigger_at=force_trigger_at)
        self._xt_init(end_rows, dwell, window, force_exhaust_at)

    def fit(self, lib, ctx):
        _fit(self, NpGraspEsc3, lib, ctx)

    def reset(self, episode):
        super().reset(episode)
        self._xt_reset(episode)

    def query(self, q):
        return self._xt_apply(super().query(q), q)


class ExhaustNpGraspStack3(_Exhaust, NpGraspStack3):
    family = "r9o6_exhaust_np_corr"

    def __init__(self, onlynp_fit="", corrected_fit="", empty_aperture=0.001, closed_sign=1.0, hold_decisions=2, burst=2,
                 max_calls=2, force_trigger_at=(), end_rows=0, dwell=2, window=24, force_exhaust_at=()):
        NpGraspStack3.__init__(self, onlynp_fit=onlynp_fit, corrected_fit=corrected_fit, empty_aperture=empty_aperture,
                               closed_sign=closed_sign, hold_decisions=hold_decisions, burst=burst, max_calls=max_calls,
                               force_trigger_at=force_trigger_at)
        self._xt_init(end_rows, dwell, window, force_exhaust_at)

    def fit(self, lib, ctx):
        _fit(self, NpGraspStack3, lib, ctx)

    def reset(self, episode):
        super().reset(episode)
        self._xt_reset(episode)

    def query(self, q):
        return self._xt_apply(super().query(q), q)


class ExhaustNpGraspStackGroot3(_Exhaust, NpGraspStackGroot3):
    family = "r9o6_exhaust_np_corr_groot"

    def __init__(self, onlynp_fit="", corrected_fit="", empty_aperture=0.0009, closed_sign=-1.0, hold_decisions=2, burst=2,
                 max_calls=2, force_trigger_at=(), end_rows=0, dwell=2, window=24, force_exhaust_at=()):
        NpGraspStackGroot3.__init__(self, onlynp_fit=onlynp_fit, corrected_fit=corrected_fit, empty_aperture=empty_aperture,
                                    closed_sign=closed_sign, hold_decisions=hold_decisions, burst=burst, max_calls=max_calls,
                                    force_trigger_at=force_trigger_at)
        self._xt_init(end_rows, dwell, window, force_exhaust_at)

    def fit(self, lib, ctx):
        _fit(self, NpGraspStackGroot3, lib, ctx)

    def reset(self, episode):
        super().reset(episode)
        self._xt_reset(episode)

    def query(self, q):
        return self._xt_apply(super().query(q), q)
