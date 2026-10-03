"""Opus round-2 serving candidates: persistent escalation to the policy on a library-pace trouble signal.

Lever: *how long* the policy keeps control once the episode is in trouble -- not where single calls go.
Trouble signal (task-agnostic, no task id, no policy, no simulator truth): at every fresh decision the cache's own
retrieval gives the top-1 library row; ``lag = decision index - library step of that row`` measures how far the
episode has fallen behind the pace of the demonstration it is imitating.  Successful cache episodes keep lag ~ 0
(they follow the demo's pace); failing ones run through the demo script and then stall, so lag grows.
Once ``lag >= lag_threshold`` at a fresh decision with index ``<= deadline``, the episode is *escalated*: every
later fresh decision is a policy call whose chunk is committed for 10 controls (the pure-policy P10 pattern, CU's
policy-tail lifecycle).  Escalation is never undone (persistent).  Before escalation the controller is exactly
the frozen base (pure cache, or B with only the no-progress guard).

Classes
  EscalateCalls          pure cache + escalation (subclass of R8 AnchorCalls; base = frozen BlindAWM prefit)
  EscalateOnlyNP         pi0.5 B (no-progress guard only) + escalation
  EscalateOnlyNPGroot    GR00T B (no-progress guard only) + escalation

The thresholds are fixed constants chosen on discovery inits 0-19 only (lag 12, deadline decision 80); see
``tools/triggers.py``.  Nothing here is fitted on evaluation inits.
"""
from __future__ import annotations

import pickle

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r08.abl.judge import TriggerCommitJudge, TriggerGrootCommitJudge
from exp.offline_search.rounds.r08.methods.methods import AnchorCalls, coin

ESC_REASON = 91.0          # os_reason code of an escalation MISS (outside every existing reason code)


def _check(lag_threshold, deadline, window=None):
    if isinstance(lag_threshold, bool) or not np.isfinite(lag_threshold) or lag_threshold <= 0:
        raise ValueError("lag_threshold must be a positive finite number")
    if deadline is not None and (type(deadline) is not int or deadline < 0):
        raise ValueError("deadline must be None or a nonnegative int (decision index)")
    if window is not None and (type(window) is not int or window < 1):
        raise ValueError("window must be None (persistent) or a positive int (decisions)")


class _Escalation:
    """Per-episode escalation state; mixed into a method whose ``self.base`` is a fitted BlindAWM."""

    def _esc_init(self, lag_threshold, deadline, window=None):
        _check(lag_threshold, deadline, window)
        self.lag_threshold, self.deadline, self.window = float(lag_threshold), deadline, window
        self._esc_identity, self._esc_step, self._esc_lag = None, None, float("nan")

    def _esc_reset(self, episode):
        self._esc_identity = (str(getattr(episode, "uid", episode)),)
        self._esc_step, self._esc_lag = None, float("nan")

    def _esc_update(self, q, top1):
        """Update with the top-1 library row of this fresh decision; returns True when escalated."""
        ident = (str(getattr(q.episode, "uid", q.episode)),)
        if ident != self._esc_identity:
            self._esc_reset(q.episode)
        step = int(q.step)
        lag = float(step - int(self.base.lib_step[int(top1)]))
        self._esc_lag = lag
        if (self._esc_step is None and lag >= self.lag_threshold
                and (self.deadline is None or step <= self.deadline)):
            self._esc_step = step
        if self._esc_step is None:
            return False
        # persistent (window None) or calls only during ``window`` decisions after the trigger; never re-armed
        return getattr(self, "window", None) is None or step < self._esc_step + self.window

    def _esc_extras(self):
        return dict(r9o_lag=self._esc_lag, r9o_escalated=float(self._esc_step is not None),
                    r9o_esc_step=float(self._esc_step) if self._esc_step is not None else -1.0,
                    r9o_lag_threshold=self.lag_threshold)


class EscalateCalls(_Escalation, AnchorCalls):
    """Frozen pure cache; after the pace-lag trigger, the policy at every fresh decision (P10 pattern)."""
    family = "r9o_escalate_cache"

    def __init__(self, lag_threshold=12, deadline=80, base_kwargs=None, base_fit="",
                 random_seed=26100201, coin_domain="R9O/escalate", window=None):
        super().__init__(p=1.0, base_kwargs=base_kwargs, base_fit=base_fit,
                         random_seed=random_seed, coin_domain=coin_domain)
        self._esc_init(lag_threshold, deadline, window)
        self.name = f"R9O_ESC_cache_L{self.lag_threshold:g}_D{deadline}" + (f"_W{window}" if window else "")

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        self.fit_info.update(lag_threshold=self.lag_threshold, deadline=self.deadline,
                             persistent=self.window is None, window=self.window,
                             fixed_p=None, trigger="top1 library-pace lag")

    def reset(self, episode):
        super().reset(episode)
        self._esc_reset(episode)

    def _assignment(self, q):
        a = self.base._anchor
        if a is None or a.get("step") != int(q.step):
            raise api.ContractError("escalation needs the base anchor of this decision")
        esc = self._esc_update(q, a["rows"][0])
        u = coin(self.random_seed, q.task_id, q.episode.init, q.step, self.coin_domain)
        return (1.0 if esc else 0.0), u, dict(coin_domain=self.coin_domain, random_seed=self.random_seed,
                                              **self._esc_extras())

    def query(self, q):
        result = super().query(q)
        result.extras = dict(result.extras or {}, **self._esc_extras())
        return result


class _EscalateJudge(_Escalation):
    """B (only the no-progress guard) + escalation; forced MISS after the trigger."""

    def __init__(self, onlynp_fit="", lag_threshold=12, deadline=80):
        # The fitted state is copied from the frozen R8 'only no-progress' artifact in fit(); __init__ of the
        # judge hierarchy is not re-run (it needs a library).  Only the escalation parameters live here.
        self.onlynp_fit = str(onlynp_fit)
        self._esc_init(lag_threshold, deadline)
        self.name = f"R9O_ESC_onlynp_L{self.lag_threshold:g}_D{deadline}"

    def fit(self, lib, ctx):
        with open(self.onlynp_fit, "rb") as f:
            blob = FitUnpickler(f).load()
        src = blob["method"]
        if not isinstance(src, self._BASE_CLASS) or blob["cell"] != ctx.cell:
            raise ValueError("only-no-progress artifact class/cell mismatch")
        if tuple(src.disabled_guards) != ("stuck", "terminal", "overtime"):
            raise ValueError("base must be B with only the no-progress guard enabled")
        keep = dict(onlynp_fit=self.onlynp_fit, lag_threshold=self.lag_threshold, deadline=self.deadline,
                    _esc_identity=None, _esc_step=None, _esc_lag=float("nan"), name=self.name)
        vars(self).update(vars(src))
        vars(self).update(keep)
        self.prof = api.NULL_PROFILER
        self.fit_info = dict(getattr(src, "fit_info", {}) or {}, onlynp_fit=self.onlynp_fit,
                             lag_threshold=self.lag_threshold, deadline=self.deadline, persistent=True)

    def reset(self, episode):
        super().reset(episode)
        self._esc_reset(episode)

    def query(self, q):
        res = super().query(q)
        esc = self._esc_update(q, res.topk[0])
        ex = dict(res.extras or {}, **self._esc_extras())
        if esc:
            ex.update(os_force_miss=1.0, os_reason=ESC_REASON,
                      os_flags=float(int(ex.get("os_flags", 0)) | (1 << 10)))
            if self._s.get("flag"):
                self._s["flag"][-1] = 1
        return api.Result(res.topk, res.scores, res.confidence, action=res.action, library=res.library, extras=ex)


class EscalateOnlyNP(_EscalateJudge, TriggerCommitJudge):
    family = "r9o_escalate_onlynp"
    _BASE_CLASS = TriggerCommitJudge


class EscalateOnlyNPGroot(_EscalateJudge, TriggerGrootCommitJudge):
    family = "r9o_escalate_onlynp_groot"
    _BASE_CLASS = TriggerGrootCommitJudge


def publish(path, method, spec, kwargs, cell, provenance):
    blob = dict(method=method, registered={}, spec=spec, kwargs=kwargs, cell=cell, fit_s=0.0, provenance=provenance)
    with open(path, "wb") as f:
        pickle.dump(blob, f, protocol=4)
    return str(path)
