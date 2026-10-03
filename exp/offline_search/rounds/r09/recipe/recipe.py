"""R9Recipe: one serving class for the R9 leading configuration of every LIBERO cell (consolidation for deployment).

Per cell (mode chosen by the constructor kwargs, frozen artifacts only; nothing is fitted here):
  * 50-demo libraries, ``mode="stack"``:
      the frozen R8 "only no-progress" judge (``r8abl_onlynp_*`` artifact: ``TriggerCommitJudge`` pi0.5 /
      ``TriggerGrootCommitJudge`` GR00T, guards stuck / terminal / overtime disabled)
      + its cache base replaced by the half-strength residual corrector applied on the JUDGE path
        (semantics of fable's ``CorrectedCacheJ``: correction added inside ``os_synth`` and remembered as the anchor,
        so the blind tail serves the corrected chunk)
      + optionally (default: pi0.5 LIBERO-10-50 only) the pace-lag escalation: once ``decision - library step of the
        top-1 row >= lag_threshold`` at a fresh decision ``<= deadline``, every later fresh decision is a policy call
        (persistent), or at most ``escalation_max_calls`` such calls per episode, after which the stack (guard still
        active) serves again
      + optionally a per-episode guard-call budget (``call_budget``, default None = off): after C committed guard calls
        every verdict is dropped and the no-progress look veto lifted (the rest of the episode is the corrected cache).
  * ``mode="pace_wrist"`` (pi0.5 500-demo "look saving"): fable's round-8 ``PaceWrist`` -- R7's ``StageWrist`` per-request
    camera plan over R7's gated ``StageFollow`` base, a look is wrist-only when the chain/valve checks pass and the
    episode is on the demo's pace (top-1 library-step lag <= ``pace_lag``); no judge, no policy call.  Served with the
    plugin flags ``--os-request-cameras --os-tokens off``.
  * ``mode="cache"``: the frozen plain cache A (``BlindAWM`` prefit), served unchanged.
Cell defaults (see ``tools/build_eq.py``): every 50-demo cell and LIBERO-10-500 = stack (escalation only for pi0.5
LIBERO-10-50); both Spatial-500 cells = cache.  Look saving (pace_wrist) is an optional setting for the two pi0.5
500-demo cells, default OFF everywhere (full-500 closed loop: it costs success).  Optional, non-default GR00T
LIBERO-10-50 settings: escalation as on pi0.5 (persistent) or capped at 12 escalation calls per episode.

Self-contained at serving time: this module imports only the harness API, numpy and the non-exploratory round
modules (R4 ``BlindAWM``, R6 ``FitUnpickler``, R7 ``StageWrist`` (which itself loads R7 ``StageFollow``), R8 judge
classes).  The corrector, escalation, budget and pace-wrist logic is copied (with attribution) from the R9 exploration
rounds; nothing under ``explore_*`` is imported.  A fitted
``R9Recipe`` pickles to an object graph of recipe / R4 / R8 classes only (tested).

NOTE for deployment: the corrector heads of the leading configuration are per task (``heads[str(task_id)]``) and
take a task one-hot as input (fable round 2, ``motion_pertask`` heads).  The recipe reproduces them unchanged; they
are the only task-indexed component of the recipe.
"""
from __future__ import annotations

import json

import numpy as np

from exp.offline_search.closed_loop.blind import LookReason
from exp.offline_search.harness import api
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r07.c2_wrist.method import StageWrist
from exp.offline_search.rounds.r08.abl.judge import TriggerCommitJudge, TriggerGrootCommitJudge

BASE_SPEC = "exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM"
JUDGE_SPECS = {"exp.offline_search.rounds.r08.abl.judge:TriggerCommitJudge": TriggerCommitJudge,
               "exp.offline_search.rounds.r08.abl.judge:TriggerGrootCommitJudge": TriggerGrootCommitJudge}
ONLY_NO_PROGRESS = ("stuck", "terminal", "overtime")
ESC_REASON = 91.0            # os_reason of an escalation MISS (as in explore_opus/round2/methods.py)
ESC_FLAG = 1 << 10
FOLLOW_BASE = "exp.offline_search.rounds.r07.c1_follow.methods:StageFollow"


# ---------------------------------------------------------------------------------------------------------------------
# Corrector (copied from explore_fable/round2/tools/methods.py: _predict, load_head, CorrectedCache._features / query,
# and explore_fable/round3/tools/methods.py: CorrectedCacheJ.os_synth; behaviour unchanged)
# ---------------------------------------------------------------------------------------------------------------------
def _predict(model, X):
    Xn = np.clip((X - model["mean"]) / model["std"], -8, 8)
    F = np.concatenate([Xn, np.cos(Xn @ model["w"] + model["bias"]) * np.sqrt(2)], 1)
    return F @ model["coef"].T + model["intercept"]


def load_head(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        keys = ("mean", "std", "w", "bias", "coef", "intercept")
        names = sorted({k.rsplit("_", 1)[0] for k in z.files if k != "meta_json"})
        heads = {n: {k: np.array(z[f"{n}_{k}"]) for k in keys} for n in names}
    for h in heads.values():
        for v in h.values():
            v.flags.writeable = False
    return heads, meta


class RecipeCorrectedBase(BlindAWM):
    """Frozen cache A + residual head, correction applied both on ``query`` and on the judge's ``os_synth`` path."""
    family = "r9_recipe_corrected_cache"
    uses_nonlibrary_action = True

    @classmethod
    def from_artifacts(cls, base_fit, head_path, blend, cell, correct_gripper=False):
        """Exactly fable's ``CorrectedCache.fit`` (state of the frozen A prefit + the head), without its class."""
        if not 0.0 <= float(blend) <= 1.5:
            raise ValueError("blend must be in [0, 1.5]")
        with open(base_fit, "rb") as f:
            blob = FitUnpickler(f).load()
        if blob["spec"] != BASE_SPEC or blob["cell"] != cell:
            raise ValueError("base artifact spec/cell mismatch")
        self = cls.__new__(cls)
        vars(self).update(vars(blob["method"]))
        vars(self).update(head_path=str(head_path), blend=float(blend), base_fit=str(base_fit), correct_gripper=bool(correct_gripper))
        self.prof = api.NULL_PROFILER
        self.heads, self.head_meta = load_head(self.head_path)
        if self.head_meta["cell"].rsplit("_", 1)[0] + "_cache" != cell:
            raise ValueError("head cell mismatch")
        self.chans = int(self.head_meta["chans"])
        self.sig_head = np.asarray(self.head_meta["sigma"], np.float32)
        if self.correct_gripper and self.chans != 7:
            raise ValueError("gripper correction needs a 7-channel head")
        self.name = f"R9F_corr_b{self.blend:g}{'_grip' if self.correct_gripper else ''}__" + self.name
        self.invalidate_anchor()
        return self

    def _features(self, q, xv, rs8, action):
        step = min(int(q.step), 120) / 120.0
        cols = [xv, rs8, (action[:10, :7] / self.sig_head).ravel(), np.array([step], np.float32)]
        if self.head_meta.get("task_onehot", True):
            cols.append(np.eye(10, dtype=np.float32)[int(q.task_id)])
        return np.concatenate(cols).astype(np.float32)[None]

    def _correction(self, q, action):
        k0 = np.asarray(q.key_v0, np.float32)
        k1 = np.asarray(q.key_v1, np.float32)
        xv = np.concatenate([self.B0T @ k0 - self.muB0, self.B1T @ k1 - self.muB1])
        rs8 = np.asarray(q.rs, np.float32)[:8]
        x = self._features(q, xv, rs8, action)
        head = self.heads["all"] if "all" in self.heads else self.heads[str(int(q.task_id))]
        return _predict(head, x)[0].reshape(10, self.chans) * self.sig_head[: self.chans]

    def _apply(self, action, corr):
        action = np.array(action, dtype=np.float32, copy=True)
        action[:10, :6] += self.blend * corr[:, :6]
        if self.correct_gripper:
            g = action[:10, 6] + self.blend * corr[:, 6]
            action[:10, 6] = np.where(g >= 0, 1.0, -1.0).astype(np.float32)
        return action

    def query(self, q):                      # CorrectedCache.query (standalone path; judges do not call it)
        res = super().query(q)
        if self.blend == 0:
            return res
        corr = self._correction(q, res.action)
        action = self._apply(res.action, corr)
        res.action = action
        res.extras = dict(res.extras, r9f_corr_rms=float(np.sqrt(np.mean(corr[:, :6] ** 2))), r9f_blend=self.blend)
        a = self._anchor
        self._remember_anchor(q, a["rows"], a["weights"], action)
        return res

    def os_synth(self, q, rows, w):          # CorrectedCacheJ.os_synth (the judge path)
        action = super().os_synth(q, rows, w)
        if self.blend == 0:
            return action
        action = self._apply(action, self._correction(q, action))
        a = self._anchor
        if a is not None:
            self._remember_anchor(q, a["rows"], a["weights"], action)
        return action


class RecipePaceWrist(StageWrist):
    """Copied from explore_fable/round8/tools/methods.py (PaceWrist, sha cc35e98a...); behaviour unchanged.
    R7 StageWrist's camera plan, but a look is wrist-only iff R7 reasons 3 / 5 are absent (chain valid, state inside the
    library valve) AND the anchor's top-1 library-step lag <= ``pace_lag``; R7 reasons 2 / 4 are not required."""
    family = "r9_recipe_pace_wrist"

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


def committed_calls(q):
    """Guard calls already executed in this episode (vision decisions whose committed outcome was a MISS).
    Copied from explore_opus/round5/methods.py."""
    step = int(q.step)
    if step <= 0:
        return 0
    hit = np.asarray(q.hist_hit)[:step]
    hv = np.asarray(getattr(q, "hist_has_vision", np.ones(step, bool)), bool)[:step]
    if len(hit) != step or len(hv) != step:
        raise api.ContractError("call budget needs step-aligned hit / vision histories")
    return int(np.sum((hit == 0) & hv))


# ---------------------------------------------------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------------------------------------------------
class R9Recipe:
    """One serving class; composition over the frozen judge (stack mode) or the frozen cache A (cache mode)."""
    family = "r9_recipe"
    uses_gt = False

    def __init__(self, mode="stack", judge_fit="", base_fit="", head_path="", blend=0.5, escalation=False,
                 lag_threshold=12, deadline=80, call_budget=None, follow_kwargs=None, wrist_fit="", stage_fit="", pace_lag=1,
                 escalation_max_calls=None, force_escalation_at=()):
        if mode not in ("stack", "cache", "pace_wrist"):
            raise ValueError("mode must be 'stack', 'cache' or 'pace_wrist'")
        if mode in ("cache", "pace_wrist") and (judge_fit or head_path or escalation or call_budget is not None):
            raise ValueError(f"{mode} mode takes no judge / corrector / escalation / budget")
        if mode != "pace_wrist" and (follow_kwargs or wrist_fit or stage_fit):
            raise ValueError("follow / wrist / stage artifacts belong to pace_wrist mode")
        if mode == "pace_wrist" and not (base_fit and wrist_fit and stage_fit and follow_kwargs):
            raise ValueError("pace_wrist mode needs base_fit, wrist_fit, stage_fit and follow_kwargs")
        if isinstance(pace_lag, bool) or not np.isfinite(pace_lag) or pace_lag < 0:
            raise ValueError("pace_lag must be a finite nonnegative number")
        if escalation_max_calls is not None and (type(escalation_max_calls) is not int or escalation_max_calls < 1
                                                 or not escalation or mode != "stack"):
            raise ValueError("escalation_max_calls must be None or a positive int, with escalation on in stack mode")
        if force_escalation_at and not escalation:
            raise ValueError("force_escalation_at (debug / selftest only) needs escalation on")
        if isinstance(lag_threshold, bool) or not np.isfinite(lag_threshold) or lag_threshold <= 0:
            raise ValueError("lag_threshold must be a positive finite number")
        if deadline is not None and (type(deadline) is not int or deadline < 0):
            raise ValueError("deadline must be None or a nonnegative int")
        if call_budget is not None and (type(call_budget) is not int or call_budget < 0):
            raise ValueError("call_budget must be None or a nonnegative int")
        self.mode, self.judge_fit, self.base_fit, self.head_path = mode, str(judge_fit), str(base_fit), str(head_path)
        self.blend, self.escalation = float(blend), bool(escalation)
        self.lag_threshold, self.deadline, self.call_budget = float(lag_threshold), deadline, call_budget
        self.follow_kwargs, self.wrist_fit, self.stage_fit = dict(follow_kwargs or {}), str(wrist_fit), str(stage_fit)
        self.pace_lag = pace_lag
        self.escalation_max_calls = escalation_max_calls
        self.force_escalation_at = tuple(int(v) for v in (force_escalation_at or ()))   # debug / selftest only
        self.inner = None
        self.name = "R9Recipe_unfitted"
        self.tier = "T1"
        self.prof = api.NULL_PROFILER
        self._ep, self._esc_step, self._esc_lag, self._quiet_step = None, None, float("nan"), None

    # -- plugin-facing attributes not defined here are those of the served object (judge or cache)
    def __getattr__(self, name):
        if name == "inner" or name.startswith("__"):
            raise AttributeError(name)
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    # -- fit: load frozen artifacts only
    def fit(self, lib, ctx):
        if self.mode == "cache":
            with open(self.base_fit, "rb") as f:
                blob = FitUnpickler(f).load()
            if blob["spec"] != BASE_SPEC or blob["cell"] != ctx.cell:
                raise ValueError("cache artifact spec/cell mismatch")
            self.inner = blob["method"]
            tag = "cache"
        elif self.mode == "pace_wrist":
            inner = RecipePaceWrist(enabled=True, base_spec=FOLLOW_BASE, base_kwargs=self.follow_kwargs, base_fit=self.base_fit,
                                    wrist_fit=self.wrist_fit, stage_fit=self.stage_fit, pace_lag=self.pace_lag)
            inner.prof = api.NULL_PROFILER
            inner.fit(lib, ctx)               # R7 StageWrist.fit: needs the deployed library (plugin prefit path)
            self.inner = inner
            tag = f"pace_wrist{float(self.pace_lag):g}"
        else:
            with open(self.judge_fit, "rb") as f:
                blob = FitUnpickler(f).load()
            judge = blob["method"]
            want = JUDGE_SPECS.get(blob["spec"])
            if want is None or type(judge) is not want or blob["cell"] != ctx.cell:
                raise ValueError("judge artifact must be the frozen R8 only-no-progress judge of this cell")
            if tuple(judge.disabled_guards) != ONLY_NO_PROGRESS:
                raise ValueError("judge must have only the no-progress guard enabled")
            base = RecipeCorrectedBase.from_artifacts(self.base_fit, self.head_path, self.blend, ctx.cell)
            if not np.array_equal(np.asarray(judge.base.act), np.asarray(base.act)):
                raise ValueError("corrected cache and judge base use different libraries")
            burst = int(judge.burst)
            judge.base = base
            judge.base.prof = api.NULL_PROFILER
            judge.prof = api.NULL_PROFILER
            if int(judge.burst) != burst or burst != 0:
                raise api.ContractError("the frozen judge must keep burst = 0")
            self.inner = judge
            cap = self.__dict__.get("escalation_max_calls")
            tag = "stack" + (f"_esc{self.lag_threshold:g}_{self.deadline}" if self.escalation else "") + \
                  (f"_cap{cap}" if cap is not None else "") + \
                  (f"_budget{self.call_budget}" if self.call_budget is not None else "")
        self.inner.prof = api.NULL_PROFILER
        self.tier = getattr(self.inner, "tier", "T1")
        self.name = f"R9Recipe_{tag}__{self.inner.name}"
        self.fit_info = dict(getattr(self.inner, "fit_info", {}) or {}, recipe_mode=self.mode, judge_fit=self.judge_fit,
                             base_fit=self.base_fit, head_path=self.head_path, blend=self.blend, escalation=self.escalation,
                             lag_threshold=self.lag_threshold, deadline=self.deadline, call_budget=self.call_budget,
                             follow_kwargs=self.follow_kwargs, wrist_fit=self.wrist_fit, stage_fit=self.stage_fit,
                             pace_lag=self.pace_lag, escalation_max_calls=self.__dict__.get("escalation_max_calls"))

    def bytes_per_entry(self):
        return self.inner.bytes_per_entry()

    # -- per-episode state
    def _sync(self, episode):
        ident = str(getattr(episode, "uid", episode))
        if ident != self._ep:
            self._ep, self._esc_step, self._esc_lag, self._quiet_step = ident, None, float("nan"), None
            self._esc_calls = 0

    def reset(self, episode):
        self.inner.reset(episode)
        self._ep = None
        self._sync(episode)

    def invalidate_anchor(self):
        if hasattr(self.inner, "invalidate_anchor"):
            self.inner.invalidate_anchor()

    @property
    def last_blind_extras(self):
        return getattr(self.inner, "last_blind_extras", {})

    # -- decisions
    def query(self, q):
        res = self.inner.query(q)
        if self.mode != "stack":
            return res
        self._sync(q.episode)
        step = int(q.step)
        ex = dict(res.extras or {})
        if self.escalation:
            # explore_opus/round2/methods.py (_Escalation._esc_update, persistent; _EscalateJudge.query)
            lag = float(step - int(self.inner.base.lib_step[int(res.topk[0])]))
            self._esc_lag = lag
            if (self._esc_step is None and lag >= self.lag_threshold
                    and (self.deadline is None or step <= self.deadline)):
                self._esc_step = step
            if self._esc_step is None and step in self.__dict__.get("force_escalation_at", ()):
                self._esc_step = step                  # debug-forced trigger (production: never)
            ex.update(r9o_lag=lag, r9o_escalated=float(self._esc_step is not None),
                      r9o_esc_step=float(self._esc_step) if self._esc_step is not None else -1.0,
                      r9o_lag_threshold=self.lag_threshold)
            # optional per-episode cap on escalation calls (attributes read via __dict__: artifacts pickled before
            # this option existed keep their persistent behaviour)
            cap = self.__dict__.get("escalation_max_calls")
            n_esc = self.__dict__.get("_esc_calls", 0)
            if self._esc_step is not None and (cap is None or n_esc < cap):
                ex.update(os_force_miss=1.0, os_reason=ESC_REASON, os_flags=float(int(ex.get("os_flags", 0)) | ESC_FLAG))
                if self.inner._s.get("flag"):
                    self.inner._s["flag"][-1] = 1
                self._esc_calls = n_esc + 1
            if cap is not None:
                ex.update(r9o_esc_calls=float(self.__dict__.get("_esc_calls", 0)), r9o_esc_cap=float(cap))
        self._quiet_step = None
        if self.call_budget is not None:
            # explore_opus/round5/methods.py (gate C only)
            n_calls = committed_calls(q)
            ex.update(r9o5_calls_before=float(n_calls), r9o5_gate=0.0)
            if ex.get("os_force_miss") and n_calls >= self.call_budget:
                ex.update(os_force_miss=0.0, r9o5_gated_reason=float(ex.get("os_reason", 0.0) or 0.0), os_reason=0.0,
                          r9o5_gate=2.0)
                if self.inner._s.get("flag"):
                    self.inner._s["flag"][-1] = 0
                self._quiet_step = step
        return api.Result(res.topk, res.scores, res.confidence, action=res.action, library=res.library, extras=ex)

    def _quiet(self, bq):
        if self.call_budget is None:
            return False
        if self._quiet_step is not None and int(bq.step) == self._quiet_step + 1:
            return True
        return committed_calls(bq) >= self.call_budget

    def blind_step(self, bq):
        if self.mode == "stack":
            self._sync(bq.episode)
            if self._quiet(bq) and getattr(self.inner, "_noprog_span", 0) > 0:
                saved = self.inner._noprog_span
                self.inner._noprog_span = 0     # lift only the no-progress look veto once the budget is spent
                try:
                    return self.inner.blind_step(bq)
                finally:
                    self.inner._noprog_span = saved
        return self.inner.blind_step(bq)

    def policy_tail_step(self, bq):
        hook = getattr(self.inner, "policy_tail_step", None)
        if not callable(hook):
            return LookReason(8, "method has no policy_tail_step")
        return hook(bq)
