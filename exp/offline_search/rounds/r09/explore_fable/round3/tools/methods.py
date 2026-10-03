"""Round-3 stacks (fable): only-no-progress judge x half-strength corrector x empty-grasp recovery x escalation.

Composition rules (no existing file edited):
* the judge is the frozen R8 "A + only no-progress" artifact (pi0.5 ``TriggerCommitJudge`` / GR00T
  ``TriggerGrootCommitJudge``), copied field by field as opus's ``_EscalateJudge`` does;
* its cache base is swapped for a fitted ``CorrectedCache`` artifact (round 2; half strength, heads fitted on
  inits 0-19), so retrieval, blind tail and the guard's progress bookkeeping see the corrected chunks;
* the empty-grasp trigger (round 2, robot-only: close held >= 2 decisions and finger aperture below the empty
  threshold; ``closed_sign`` +1 pi0.5 / -1 GR00T) forces a policy MISS at this and the next ``burst-1`` fresh
  decisions (CU policy-tail lifecycle), at most ``max_calls`` triggers per episode; ``max_calls=0`` turns the
  trigger off (pure "no-progress + corrector" stack);
* ``*Esc`` variants inherit opus's persistent pace-lag escalation (``EscalateOnlyNP``), unmodified.
Forced misses follow the existing extras contract (``os_force_miss``, ``os_reason``, ``_s['flag']``).
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r08.abl.judge import TriggerCommitJudge, TriggerGrootCommitJudge
from exp.offline_search.rounds.r09.explore_fable.round2.tools.methods import CorrectedCache, _predict
from exp.offline_search.rounds.r09.explore_opus.round2.methods import EscalateOnlyNP

GRASP_REASON = 93.0
CORRECTED_SPEC = "exp.offline_search.rounds.r09.explore_fable.round2.tools.methods:CorrectedCache"


def load_corrected(path, cell):
    with open(path, "rb") as f:
        blob = FitUnpickler(f).load()
    if blob["spec"] != CORRECTED_SPEC or blob["cell"] != cell:
        raise ValueError("corrected-cache artifact spec/cell mismatch")
    m = blob["method"]
    if not isinstance(m, CorrectedCache) or (m.serving, m.budget, m.gates) != ("anchor_tail", 1, "budget_only"):
        raise api.ContractError("stack needs the frozen ten-control CorrectedCache")
    return m, blob


class _GraspStack:
    """Mixin: copy the frozen only-no-progress judge, swap its base for the corrected cache, add the trigger."""
    _BASE_CLASS = None

    def _stack_init(self, onlynp_fit, corrected_fit, empty_aperture, closed_sign, hold_decisions, burst, max_calls):
        if closed_sign not in (1.0, -1.0, 1, -1) or hold_decisions < 1 or burst < 1 or max_calls < 0:
            raise ValueError("invalid stack parameters")
        self.onlynp_fit, self.corrected_fit = str(onlynp_fit), str(corrected_fit)
        self.empty_aperture, self.closed_sign = float(empty_aperture), float(closed_sign)
        self.hold_decisions, self.burst, self.max_calls = int(hold_decisions), int(burst), int(max_calls)
        self._gm_identity, self._gm_n, self._gm_burst_left = None, 0, 0

    def _stack_fit(self, ctx):
        with open(self.onlynp_fit, "rb") as f:
            blob = FitUnpickler(f).load()
        src = blob["method"]
        if not isinstance(src, self._BASE_CLASS) or blob["cell"] != ctx.cell:
            raise ValueError("only-no-progress artifact class/cell mismatch")
        if tuple(src.disabled_guards) != ("stuck", "terminal", "overtime"):
            raise ValueError("base must be B with only the no-progress guard enabled")
        corrected, cblob = load_corrected(self.corrected_fit, ctx.cell)
        if not np.array_equal(np.asarray(src.base.act), np.asarray(corrected.act)):
            raise ValueError("corrected cache and judge base use different libraries")
        keep = {k: getattr(self, k) for k in ("onlynp_fit", "corrected_fit", "empty_aperture", "closed_sign", "hold_decisions", "burst",
                                             "max_calls", "_gm_identity", "_gm_n", "_gm_burst_left", "name")}
        keep.update({k: getattr(self, k) for k in ("lag_threshold", "deadline", "window", "_esc_identity", "_esc_step", "_esc_lag") if hasattr(self, k)})
        vars(self).update(vars(src))
        vars(self).update(keep)
        self.base = corrected
        self.base.prof = api.NULL_PROFILER
        self.prof = api.NULL_PROFILER
        self.fit_info = dict(getattr(src, "fit_info", {}) or {}, onlynp_fit=self.onlynp_fit, corrected_fit=self.corrected_fit,
                             corrector_blend=float(corrected.blend), grasp_trigger=dict(empty_aperture=self.empty_aperture, closed_sign=self.closed_sign,
                             hold_decisions=self.hold_decisions, burst=self.burst, max_calls=self.max_calls))

    def _gm_reset(self, episode):
        self._gm_identity = (str(getattr(episode, "uid", episode)),)
        self._gm_n, self._gm_burst_left = 0, 0

    def empty_grasp(self, q):
        step = int(q.step)
        if step < self.hold_decisions or self._gm_n >= self.max_calls:
            return False
        hist = np.asarray(q.hist_a_exec)[step - self.hold_decisions:step, :5, 6] * self.closed_sign
        if not np.all(hist > 0):
            return False
        raw = np.asarray(q.raw_state, np.float64)
        return float(raw[6] - raw[7]) / 2.0 < self.empty_aperture

    def _gm_update(self, q):
        ident = (str(getattr(q.episode, "uid", q.episode)),)
        if ident != self._gm_identity:
            self._gm_reset(q.episode)
        trig = self._gm_burst_left == 0 and self.empty_grasp(q)
        if trig:
            self._gm_n += 1
            self._gm_burst_left = self.burst
        in_burst = self._gm_burst_left > 0
        if in_burst:
            self._gm_burst_left -= 1
        return trig, in_burst

    def _stack_query(self, res, q):
        trig, in_burst = self._gm_update(q)
        ex = dict(res.extras or {}, r9f3_grasp_trigger=float(trig), r9f3_grasp_burst=float(in_burst), r9f3_n_trigger=float(self._gm_n))
        if in_burst and not ex.get("os_force_miss"):
            ex.update(os_force_miss=1.0, os_reason=GRASP_REASON, os_flags=float(int(ex.get("os_flags", 0)) | (1 << 11)))
            if self._s.get("flag"):
                self._s["flag"][-1] = 1
        return api.Result(res.topk, res.scores, res.confidence, action=res.action, library=res.library, extras=ex)


class NpGraspStack(_GraspStack, TriggerCommitJudge):
    """pi0.5: only-no-progress judge + half corrector (+ empty-grasp recovery burst when max_calls > 0)."""
    family = "r9f3_np_corr_grasp"
    _BASE_CLASS = TriggerCommitJudge

    def __init__(self, onlynp_fit="", corrected_fit="", empty_aperture=0.001, closed_sign=1.0, hold_decisions=2, burst=2, max_calls=2):
        self._stack_init(onlynp_fit, corrected_fit, empty_aperture, closed_sign, hold_decisions, burst, max_calls)
        self.name = f"R9F3_np_corr_gm{max_calls}"

    def fit(self, lib, ctx):
        self._stack_fit(ctx)

    def reset(self, episode):
        super().reset(episode)
        self._gm_reset(episode)

    def query(self, q):
        return self._stack_query(super().query(q), q)


class NpGraspStackGroot(NpGraspStack, TriggerGrootCommitJudge):
    """GR00T variant (model-aware terminal sign in the judge; closed_sign must be -1)."""
    family = "r9f3_np_corr_grasp_groot"
    _BASE_CLASS = TriggerGrootCommitJudge


class NpGraspEsc(_GraspStack, EscalateOnlyNP):
    """pi0.5: opus's only-no-progress + persistent pace-lag escalation, with the corrected base (+ optional trigger)."""
    family = "r9f3_np_corr_grasp_esc"
    _BASE_CLASS = TriggerCommitJudge

    def __init__(self, onlynp_fit="", corrected_fit="", lag_threshold=12, deadline=80, empty_aperture=0.001, closed_sign=1.0,
                 hold_decisions=2, burst=2, max_calls=2):
        EscalateOnlyNP.__init__(self, onlynp_fit=onlynp_fit, lag_threshold=lag_threshold, deadline=deadline)
        self._stack_init(onlynp_fit, corrected_fit, empty_aperture, closed_sign, hold_decisions, burst, max_calls)
        self.name = f"R9F3_np_corr_esc_L{lag_threshold:g}_D{deadline}_gm{max_calls}"

    def fit(self, lib, ctx):
        self._stack_fit(ctx)                       # copies the judge, swaps the base; escalation fields are kept

    def reset(self, episode):
        super().reset(episode)                     # EscalateOnlyNP.reset -> judge reset + escalation reset
        self._gm_reset(episode)

    def query(self, q):
        return self._stack_query(super().query(q), q)   # EscalateOnlyNP.query (escalation) then the grasp trigger


# ----------------------------------------------------------------------------- v2 (r3b): single judge family per class
# Root cause of the r3 failure: ``NpGraspStackGroot(NpGraspStack, TriggerGrootCommitJudge)`` linearized BOTH judge
# families (CommitJudge and GrootCommitJudge) into one MRO, so one decision ran two judge query chains; the frozen P2
# trigger mask then saw inconsistent window state and raised "P2 does not support active burst/return windows".
# The v2 classes below mix ``_GraspStack`` into exactly one judge family each. ``force_trigger_at`` (debug only,
# default empty) fires the empty-grasp trigger at the listed decision indices regardless of the state, so tests and
# CPU plugin selftests can exercise the trigger path together with the guard path in the same episode.
class _GraspStack2(_GraspStack):
    def _stack_init2(self, force_trigger_at=()):
        self.force_trigger_at = tuple(int(v) for v in (force_trigger_at or ()))

    def empty_grasp(self, q):
        if self.force_trigger_at and int(q.step) in self.force_trigger_at and self._gm_n < self.max_calls:
            return True
        return _GraspStack.empty_grasp(self, q)

    def _stack_fit(self, ctx):
        _GraspStack._stack_fit(self, ctx)
        # exactly one frozen R8 judge family may be linearized (GrootCommitJudge itself subclasses CommitJudge, so
        # the test is on the two R8 ablation classes, not on the base judge names)
        if sum(c in type(self).__mro__ for c in (TriggerCommitJudge, TriggerGrootCommitJudge)) != 1:
            raise api.ContractError("stack class mixes two judge families")
        self.fit_info["force_trigger_at"] = list(self.force_trigger_at)


class NpGraspStack2(_GraspStack2, TriggerCommitJudge):
    """pi0.5: only-no-progress judge + half corrector (+ empty-grasp recovery when max_calls > 0)."""
    family = "r9f3b_np_corr_grasp"
    _BASE_CLASS = TriggerCommitJudge

    def __init__(self, onlynp_fit="", corrected_fit="", empty_aperture=0.001, closed_sign=1.0, hold_decisions=2, burst=2, max_calls=2,
                 force_trigger_at=()):
        self._stack_init(onlynp_fit, corrected_fit, empty_aperture, closed_sign, hold_decisions, burst, max_calls)
        self._stack_init2(force_trigger_at)
        self.name = f"R9F3b_np_corr_gm{max_calls}"

    def fit(self, lib, ctx):
        self._stack_fit(ctx)

    def reset(self, episode):
        super().reset(episode)
        self._gm_reset(episode)

    def query(self, q):
        return self._stack_query(super().query(q), q)


class NpGraspStackGroot2(_GraspStack2, TriggerGrootCommitJudge):
    """GR00T: only-no-progress judge (model-aware terminal sign) + half corrector (+ recovery); closed_sign must be -1."""
    family = "r9f3b_np_corr_grasp_groot"
    _BASE_CLASS = TriggerGrootCommitJudge

    def __init__(self, onlynp_fit="", corrected_fit="", empty_aperture=0.0009, closed_sign=-1.0, hold_decisions=2, burst=2, max_calls=2,
                 force_trigger_at=()):
        if float(closed_sign) != -1.0:
            raise ValueError("GR00T stacks require closed_sign=-1 (normalized gripper -1 = close)")
        self._stack_init(onlynp_fit, corrected_fit, empty_aperture, closed_sign, hold_decisions, burst, max_calls)
        self._stack_init2(force_trigger_at)
        self.name = f"R9F3b_groot_np_corr_gm{max_calls}"

    def fit(self, lib, ctx):
        self._stack_fit(ctx)

    def reset(self, episode):
        super().reset(episode)
        self._gm_reset(episode)

    def query(self, q):
        return self._stack_query(super().query(q), q)


class NpGraspEsc2(_GraspStack2, EscalateOnlyNP):
    """pi0.5: opus's only-no-progress + persistent pace-lag escalation, corrected base (+ optional recovery)."""
    family = "r9f3b_np_corr_grasp_esc"
    _BASE_CLASS = TriggerCommitJudge

    def __init__(self, onlynp_fit="", corrected_fit="", lag_threshold=12, deadline=80, empty_aperture=0.001, closed_sign=1.0,
                 hold_decisions=2, burst=2, max_calls=2, force_trigger_at=()):
        EscalateOnlyNP.__init__(self, onlynp_fit=onlynp_fit, lag_threshold=lag_threshold, deadline=deadline)
        self._stack_init(onlynp_fit, corrected_fit, empty_aperture, closed_sign, hold_decisions, burst, max_calls)
        self._stack_init2(force_trigger_at)
        self.name = f"R9F3b_np_corr_esc_L{lag_threshold:g}_D{deadline}_gm{max_calls}"

    def fit(self, lib, ctx):
        self._stack_fit(ctx)

    def reset(self, episode):
        super().reset(episode)
        self._gm_reset(episode)

    def query(self, q):
        return self._stack_query(super().query(q), q)


# ----------------------------------------------------------------------------- v3 (r3c): no attribute collisions, corrector active under the judge
# Root cause of the r3/r3b crashes (verified on the fitted artifacts): ``_GraspStack`` stored its recovery-burst length as
# ``self.burst`` and kept it over ``vars(src)``, so the frozen judge's own ``burst`` (0 in the artifact; k7 arms its
# burst/return windows only when ``self.burst > 1``) became 2. Every no-progress firing then armed a window, and the
# P2 trigger mask raised "P2 does not support active burst/return windows" as soon as a disabled guard fired later.
# The v3 mixin prefixes every stack attribute with ``gm_`` and refuses, at fit time, any attribute name that already
# exists on the frozen judge. Second fix: a judge retrieves through the G3 base contract (``os_score_all`` /
# ``os_synth``), never through ``base.query``, so a plain ``CorrectedCache`` base left the corrector inactive inside
# every judge stack (including round 2's ``np_corr05pt``). ``CorrectedCacheJ`` applies the head in ``os_synth``.
class CorrectedCacheJ(CorrectedCache):
    """CorrectedCache whose correction is also applied on the G3 base-contract path used by judges."""
    family = "r9f3c_corrected_cache_judge_path"

    def os_synth(self, q, rows, w):
        action = super().os_synth(q, rows, w)             # BlindAWM: kernel mean + anchor remembered (uncorrected)
        if self.blend == 0:
            return action
        k0 = np.asarray(q.key_v0, np.float32)
        k1 = np.asarray(q.key_v1, np.float32)
        xv = np.concatenate([self.B0T @ k0 - self.muB0, self.B1T @ k1 - self.muB1])
        rs8 = np.asarray(q.rs, np.float32)[:8]
        x = self._features(q, xv, rs8, action)
        head = self.heads["all"] if "all" in self.heads else self.heads[str(int(q.task_id))]
        corr = _predict(head, x)[0].reshape(10, self.chans) * self.sig_head[: self.chans]
        action = np.array(action, dtype=np.float32, copy=True)
        action[:10, :6] += self.blend * corr[:, :6]
        if self.correct_gripper:
            g = action[:10, 6] + self.blend * corr[:, 6]
            action[:10, 6] = np.where(g >= 0, 1.0, -1.0).astype(np.float32)
        a = self._anchor
        if a is not None:
            self._remember_anchor(q, a["rows"], a["weights"], action)
        return action


def load_corrected_for_judge(path, cell):
    m, blob = load_corrected(path, cell)
    m.__class__ = CorrectedCacheJ                           # fitted state unchanged; os_synth now applies the head
    return m, blob


class _GraspStack3:
    """Mixin: frozen only-no-progress judge + CorrectedCacheJ base + empty-grasp recovery; all state under ``gm_``."""
    _BASE_CLASS = None
    _OWN = ("gm_onlynp_fit", "gm_corrected_fit", "gm_aperture", "gm_sign", "gm_hold", "gm_burst", "gm_max_calls", "gm_force_at",
            "gm_identity", "gm_n", "gm_burst_left", "gm_name")

    def _gm_init(self, onlynp_fit, corrected_fit, empty_aperture, closed_sign, hold_decisions, burst, max_calls, force_trigger_at, name):
        if closed_sign not in (1.0, -1.0, 1, -1) or hold_decisions < 1 or burst < 1 or max_calls < 0:
            raise ValueError("invalid stack parameters")
        self.gm_onlynp_fit, self.gm_corrected_fit = str(onlynp_fit), str(corrected_fit)
        self.gm_aperture, self.gm_sign = float(empty_aperture), float(closed_sign)
        self.gm_hold, self.gm_burst, self.gm_max_calls = int(hold_decisions), int(burst), int(max_calls)
        self.gm_force_at = tuple(int(v) for v in (force_trigger_at or ()))
        self.gm_identity, self.gm_n, self.gm_burst_left = None, 0, 0
        self.gm_name = name

    def _gm_fit(self, ctx):
        with open(self.gm_onlynp_fit, "rb") as f:
            blob = FitUnpickler(f).load()
        src = blob["method"]
        if not isinstance(src, self._BASE_CLASS) or blob["cell"] != ctx.cell:
            raise ValueError("only-no-progress artifact class/cell mismatch")
        if tuple(src.disabled_guards) != ("stuck", "terminal", "overtime"):
            raise ValueError("base must be B with only the no-progress guard enabled")
        if sum(c in type(self).__mro__ for c in (TriggerCommitJudge, TriggerGrootCommitJudge)) != 1:
            raise api.ContractError("stack class mixes two judge families")
        clash = sorted(set(self._OWN) & set(vars(src)))
        if clash:
            raise api.ContractError("stack attributes shadow frozen judge attributes: %s" % clash)
        corrected, _ = load_corrected_for_judge(self.gm_corrected_fit, ctx.cell)
        if not np.array_equal(np.asarray(src.base.act), np.asarray(corrected.act)):
            raise ValueError("corrected cache and judge base use different libraries")
        keep = {k: getattr(self, k) for k in self._OWN}
        keep.update({k: getattr(self, k) for k in ("lag_threshold", "deadline", "window", "_esc_identity", "_esc_step", "_esc_lag", "onlynp_fit") if hasattr(self, k)})
        vars(self).update(vars(src))                         # judge's own fields, including burst = 0, win untouched
        vars(self).update(keep)
        self.name = self.gm_name
        self.base = corrected
        self.base.prof = api.NULL_PROFILER
        self.prof = api.NULL_PROFILER
        if int(self.burst) != int(src.burst):
            raise api.ContractError("judge burst length changed by the stack")
        self.fit_info = dict(getattr(src, "fit_info", {}) or {}, onlynp_fit=self.gm_onlynp_fit, corrected_fit=self.gm_corrected_fit,
                             corrector_blend=float(corrected.blend), corrector_path="os_synth", judge_burst=int(self.burst),
                             grasp_trigger=dict(empty_aperture=self.gm_aperture, closed_sign=self.gm_sign, hold_decisions=self.gm_hold,
                                                burst=self.gm_burst, max_calls=self.gm_max_calls, force_trigger_at=list(self.gm_force_at)))

    def _gm_reset(self, episode):
        self.gm_identity = (str(getattr(episode, "uid", episode)),)
        self.gm_n, self.gm_burst_left = 0, 0

    def gm_empty_grasp(self, q):
        step = int(q.step)
        if self.gm_n >= self.gm_max_calls:
            return False
        if self.gm_force_at and step in self.gm_force_at:
            return True
        if step < self.gm_hold:
            return False
        hist = np.asarray(q.hist_a_exec)[step - self.gm_hold:step, :5, 6] * self.gm_sign
        if not np.all(hist > 0):
            return False
        raw = np.asarray(q.raw_state, np.float64)
        return float(raw[6] - raw[7]) / 2.0 < self.gm_aperture

    def _gm_query(self, res, q):
        ident = (str(getattr(q.episode, "uid", q.episode)),)
        if ident != self.gm_identity:
            self._gm_reset(q.episode)
        trig = self.gm_burst_left == 0 and self.gm_empty_grasp(q)
        if trig:
            self.gm_n += 1
            self.gm_burst_left = self.gm_burst
        in_burst = self.gm_burst_left > 0
        if in_burst:
            self.gm_burst_left -= 1
        ex = dict(res.extras or {}, r9f3_grasp_trigger=float(trig), r9f3_grasp_burst=float(in_burst), r9f3_n_trigger=float(self.gm_n))
        if in_burst and not ex.get("os_force_miss"):
            ex.update(os_force_miss=1.0, os_reason=GRASP_REASON, os_flags=float(int(ex.get("os_flags", 0)) | (1 << 11)))
            if self._s.get("flag"):
                self._s["flag"][-1] = 1
        return api.Result(res.topk, res.scores, res.confidence, action=res.action, library=res.library, extras=ex)


class NpGraspStack3(_GraspStack3, TriggerCommitJudge):
    """pi0.5: only-no-progress judge + corrector (judge path) + optional empty-grasp recovery."""
    family = "r9f3c_np_corr_grasp"
    _BASE_CLASS = TriggerCommitJudge

    def __init__(self, onlynp_fit="", corrected_fit="", empty_aperture=0.001, closed_sign=1.0, hold_decisions=2, burst=2, max_calls=2,
                 force_trigger_at=()):
        self._gm_init(onlynp_fit, corrected_fit, empty_aperture, closed_sign, hold_decisions, burst, max_calls, force_trigger_at,
                      f"R9F3c_np_corr_gm{max_calls}")

    def fit(self, lib, ctx):
        self._gm_fit(ctx)

    def reset(self, episode):
        super().reset(episode)
        self._gm_reset(episode)

    def query(self, q):
        return self._gm_query(super().query(q), q)


class NpGraspStackGroot3(_GraspStack3, TriggerGrootCommitJudge):
    """GR00T: only-no-progress judge (model-aware terminal sign) + corrector (judge path) + optional recovery."""
    family = "r9f3c_np_corr_grasp_groot"
    _BASE_CLASS = TriggerGrootCommitJudge

    def __init__(self, onlynp_fit="", corrected_fit="", empty_aperture=0.0009, closed_sign=-1.0, hold_decisions=2, burst=2, max_calls=2,
                 force_trigger_at=()):
        if float(closed_sign) != -1.0:
            raise ValueError("GR00T stacks require closed_sign=-1")
        self._gm_init(onlynp_fit, corrected_fit, empty_aperture, closed_sign, hold_decisions, burst, max_calls, force_trigger_at,
                      f"R9F3c_groot_np_corr_gm{max_calls}")

    def fit(self, lib, ctx):
        self._gm_fit(ctx)

    def reset(self, episode):
        super().reset(episode)
        self._gm_reset(episode)

    def query(self, q):
        return self._gm_query(super().query(q), q)


class NpGraspEsc3(_GraspStack3, EscalateOnlyNP):
    """pi0.5: opus's only-no-progress + escalation, corrector on the judge path, optional recovery."""
    family = "r9f3c_np_corr_grasp_esc"
    _BASE_CLASS = TriggerCommitJudge

    def __init__(self, onlynp_fit="", corrected_fit="", lag_threshold=12, deadline=80, empty_aperture=0.001, closed_sign=1.0,
                 hold_decisions=2, burst=2, max_calls=2, force_trigger_at=()):
        EscalateOnlyNP.__init__(self, onlynp_fit=onlynp_fit, lag_threshold=lag_threshold, deadline=deadline)
        self._gm_init(onlynp_fit, corrected_fit, empty_aperture, closed_sign, hold_decisions, burst, max_calls, force_trigger_at,
                      f"R9F3c_np_corr_esc_L{lag_threshold:g}_D{deadline}_gm{max_calls}")

    def fit(self, lib, ctx):
        self._gm_fit(ctx)

    def reset(self, episode):
        super().reset(episode)
        self._gm_reset(episode)

    def query(self, q):
        return self._gm_query(super().query(q), q)
