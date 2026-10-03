"""Round-2 serving candidates (task-agnostic; fitted on inits 0-19 only).

``CorrectedCache`` -- the frozen pure cache (BlindAWM) whose served chunk receives a
ridge/RFF residual correction (motion channels; optional gripper channel) at a fixed
strength. It *is* a BlindAWM (class swap on the fitted object), so every blind /
anchor-tail / judge hook keeps working unchanged. The head was fitted by
``round2/tools/corrector.py`` on discovery inits 0-19.

``GraspMissCalls`` -- CorrectedCache plus an observation-keyed recovery call: when
the executed gripper command has been "close" for the last two decisions and the
finger aperture (robot state) has collapsed below the empty-close threshold (the
fingers met nothing), the next anchor is a forced policy call (10-control commit,
CU's policy tail). At most ``max_calls`` such calls per episode. No task knowledge
is used; thresholds come from the library's own successful demonstrations.
``p_uniform`` adds an independent uniform call coin on top (0 = off).
"""
from __future__ import annotations

import copy
import json
import pickle

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r08.methods.methods import AnchorCalls, coin

BASE_SPEC = "exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM"


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


class CorrectedCache(BlindAWM):
    """Frozen A + residual head; kwargs = A's kwargs plus head_path / blend / correct_gripper."""
    family = "r9f_r2_corrected_cache"
    uses_nonlibrary_action = True

    def __init__(self, head_path="", blend=1.0, base_fit="", correct_gripper=False, **base_kwargs):
        if not 0.0 <= float(blend) <= 1.5:
            raise ValueError("blend must be in [0, 1.5]")
        super().__init__(**base_kwargs)
        self.head_path, self.blend, self.base_fit = str(head_path), float(blend), str(base_fit)
        self.correct_gripper = bool(correct_gripper)
        self.heads, self.head_meta = None, None

    def fit(self, lib, ctx):
        with open(self.base_fit, "rb") as f:
            blob = FitUnpickler(f).load()
        if blob["spec"] != BASE_SPEC or blob["cell"] != ctx.cell:
            raise ValueError("base artifact spec/cell mismatch")
        keep = dict(head_path=self.head_path, blend=self.blend, base_fit=self.base_fit, correct_gripper=self.correct_gripper)
        vars(self).update(vars(blob["method"]))
        vars(self).update(keep)
        self.prof = api.NULL_PROFILER
        self.heads, self.head_meta = load_head(self.head_path)
        if self.head_meta["cell"].rsplit("_", 1)[0] + "_cache" != ctx.cell:
            raise ValueError("head cell mismatch")
        self.chans = int(self.head_meta["chans"])
        self.sig_head = np.asarray(self.head_meta["sigma"], np.float32)
        if self.correct_gripper and self.chans != 7:
            raise ValueError("gripper correction needs a 7-channel head")
        self.name = f"R9F_corr_b{self.blend:g}{'_grip' if self.correct_gripper else ''}__" + self.name
        self.invalidate_anchor()

    def _features(self, q, xv, rs8, action):
        step = min(int(q.step), 120) / 120.0
        cols = [xv, rs8, (action[:10, :7] / self.sig_head).ravel(), np.array([step], np.float32)]
        if self.head_meta.get("task_onehot", True):
            cols.append(np.eye(10, dtype=np.float32)[int(q.task_id)])
        return np.concatenate(cols).astype(np.float32)[None]

    def query(self, q):
        res = super().query(q)
        if self.blend == 0:
            return res
        k0 = np.asarray(q.key_v0, np.float32)
        k1 = np.asarray(q.key_v1, np.float32)
        xv = np.concatenate([self.B0T @ k0 - self.muB0, self.B1T @ k1 - self.muB1])
        rs8 = np.asarray(q.rs, np.float32)[:8]
        x = self._features(q, xv, rs8, res.action)
        head = self.heads["all"] if "all" in self.heads else self.heads[str(int(q.task_id))]
        corr = _predict(head, x)[0].reshape(10, self.chans) * self.sig_head[: self.chans]
        action = np.array(res.action, dtype=np.float32, copy=True)
        nchan = 6
        action[:10, :nchan] += self.blend * corr[:, :nchan]
        if self.correct_gripper:
            g = action[:10, 6] + self.blend * corr[:, 6]
            action[:10, 6] = np.where(g >= 0, 1.0, -1.0).astype(np.float32)
        res.action = action
        res.extras = dict(res.extras, r9f_corr_rms=float(np.sqrt(np.mean(corr[:, :nchan] ** 2))), r9f_blend=self.blend)
        a = self._anchor
        self._remember_anchor(q, a["rows"], a["weights"], action)   # the blind tail must serve the corrected chunk
        return res


class GraspMissCalls(AnchorCalls):
    """CorrectedCache + observation-keyed recovery calls after an empty grasp (+ optional uniform coin)."""
    family = "r9f_r2_grasp_miss_calls"

    def __init__(self, corrected_fit, corrected_kwargs, empty_aperture, p_uniform=0.0, max_calls=2, hold_decisions=2,
                 burst=2, random_seed=26100202, coin_domain="R9F/r2/graspmiss"):
        if not 0 <= p_uniform <= 1 or max_calls < 0 or hold_decisions < 1 or burst < 1:
            raise ValueError("invalid trigger parameters")
        self.corrected_fit, self.corrected_kwargs = str(corrected_fit), dict(corrected_kwargs)
        self.empty_aperture, self.p_uniform = float(empty_aperture), float(p_uniform)
        self.max_calls, self.hold_decisions, self.burst = int(max_calls), int(hold_decisions), int(burst)
        self._n_trigger, self._burst_left = 0, 0
        super().__init__(p=max(p_uniform, 1.0), base_kwargs={}, base_fit="", random_seed=random_seed, coin_domain=coin_domain)
        self.name = f"R9F_graspmiss_ap{self.empty_aperture:g}_pu{self.p_uniform:g}_max{self.max_calls}_b{self.burst}"

    def fit(self, lib, ctx):
        with open(self.corrected_fit, "rb") as f:
            blob = FitUnpickler(f).load()
        want = "exp.offline_search.rounds.r09.explore_fable.round2.tools.methods:CorrectedCache"
        if blob["spec"] != want or blob["kwargs"] != self.corrected_kwargs or blob["cell"] != ctx.cell:
            raise ValueError("corrected artifact spec/kwargs/cell mismatch")
        self.base = blob["method"]
        self.base.prof = self.prof
        if (self.base.serving, self.base.budget, self.base.gates) != ("anchor_tail", 1, "budget_only"):
            raise api.ContractError("requires the frozen ten-control cache")
        self.fit_info = dict(corrected_fit=self.corrected_fit, empty_aperture=self.empty_aperture, p_uniform=self.p_uniform,
                             max_calls=self.max_calls, hold_decisions=self.hold_decisions, stall=False, cooldown=False,
                             policy_commit_controls=10, policy_tail_blocks=1)

    def reset(self, episode):
        super().reset(episode)
        self._n_trigger, self._burst_left = 0, 0

    def empty_grasp(self, q):
        """Closed command held for ``hold_decisions`` decisions and fingers collapsed below the empty threshold."""
        step = int(q.step)
        if step < self.hold_decisions or self._n_trigger >= self.max_calls:
            return False
        hist = np.asarray(q.hist_a_exec)
        closed = all(np.all(hist[step - k, :5, 6] > 0) for k in range(1, self.hold_decisions + 1))   # +1 = close
        if not closed:
            return False
        raw = np.asarray(q.raw_state, np.float64)
        aperture = float(raw[6] - raw[7]) / 2.0
        return aperture < self.empty_aperture

    def _assignment(self, q):
        u = coin(self.random_seed, q.task_id, q.episode.init, q.step, self.coin_domain)
        trig = self._burst_left == 0 and self.empty_grasp(q)
        if trig:
            self._n_trigger += 1
            self._burst_left = self.burst                  # this anchor and the next burst-1 anchors are policy calls
        in_burst = self._burst_left > 0
        if in_burst:
            self._burst_left -= 1
        p = 1.0 if in_burst else self.p_uniform
        return p, u, dict(coin_domain=self.coin_domain, random_seed=self.random_seed, grasp_miss_trigger=float(trig),
                          grasp_miss_burst=float(in_burst), n_trigger=self._n_trigger)


class GraspMissCalls2(GraspMissCalls):
    """GraspMissCalls plus a second robot-only trigger: the close command held for ``long_carry_decisions``
    consecutive decisions (object held but never placed; 0 false alarms in successful held-out episodes at 30
    decisions = 150 controls). Both triggers share the burst / cap. Separate class so pickled GraspMissCalls
    objects stay untouched."""
    family = "r9f_r2_grasp_miss_calls2"

    def __init__(self, long_carry_decisions=30, **kwargs):
        if long_carry_decisions < 2:
            raise ValueError("long_carry_decisions must be >= 2")
        self.long_carry_decisions = int(long_carry_decisions)
        super().__init__(**kwargs)
        self.name = self.name.replace("R9F_graspmiss", "R9F_graspmiss2_lc%d" % self.long_carry_decisions)

    def long_carry(self, q):
        step = int(q.step)
        if step < self.long_carry_decisions or self._n_trigger >= self.max_calls:
            return False
        hist = np.asarray(q.hist_a_exec)[step - self.long_carry_decisions:step, :5, 6]
        return bool(np.all(hist > 0))

    def _assignment(self, q):
        u = coin(self.random_seed, q.task_id, q.episode.init, q.step, self.coin_domain)
        empty = self._burst_left == 0 and self.empty_grasp(q)
        carry = self._burst_left == 0 and not empty and self.long_carry(q)
        trig = empty or carry
        if trig:
            self._n_trigger += 1
            self._burst_left = self.burst
        in_burst = self._burst_left > 0
        if in_burst:
            self._burst_left -= 1
        p = 1.0 if in_burst else self.p_uniform
        return p, u, dict(coin_domain=self.coin_domain, random_seed=self.random_seed, grasp_miss_trigger=float(empty),
                          long_carry_trigger=float(carry), grasp_miss_burst=float(in_burst), n_trigger=self._n_trigger)


# ----------------------------------------------------------------------------- round 2c: cache-side responses (no policy call)
class GraspMissReanchor(CorrectedCache):
    """Cache-side open-and-retry after a flagged empty grasp (no policy call; IR unchanged).

    At a look decision whose history shows the close command held for ``hold_decisions`` decisions while the finger
    aperture is below ``empty_aperture`` (fingers met nothing), the served 10-control commit is replaced by a
    *retreat*: the last ``retreat_decisions`` executed decisions replayed in reverse with negated motion (in wire
    units, through the affine normalized->wire map ``norm_scale/norm_shift``) and the gripper commanded open.
    The anchor is set to that chunk, so the blind decision serves its tail; the next decision is a fresh look from
    the retreated, open-gripper state and retrieval re-attempts the grasp. At most ``max_retries`` per episode and
    never within ``cooldown_decisions`` of the previous retreat. ``closed_sign`` is the model's normalized gripper
    sign that means "close" (+1 pi0.5, -1 GR00T). Optional union triggers: ``long_carry_decisions`` (close held
    that many decisions: retreat keeping the gripper closed) and ``stall_decisions`` with ``stall_path`` (normalized
    end-effector path over that many decisions below the threshold: retreat keeping the gripper state).
    """
    family = "r9f_r2_grasp_miss_reanchor"

    def __init__(self, empty_aperture=0.001, hold_decisions=2, max_retries=2, retreat_decisions=2, cooldown_decisions=3,
                 closed_sign=1.0, norm_scale=None, norm_shift=None, long_carry_decisions=0, stall_decisions=0, stall_path=0.0,
                 **kwargs):
        if hold_decisions < 1 or max_retries < 0 or retreat_decisions < 1 or cooldown_decisions < 1 or closed_sign not in (1.0, -1.0, 1, -1):
            raise ValueError("invalid re-anchor parameters")
        if norm_scale is None or norm_shift is None or len(norm_scale) != 7 or len(norm_shift) != 7:
            raise ValueError("norm_scale/norm_shift (7 each, normalized->wire affine map) are required")
        super().__init__(**kwargs)
        self.empty_aperture, self.hold_decisions = float(empty_aperture), int(hold_decisions)
        self.max_retries, self.retreat_decisions, self.cooldown_decisions = int(max_retries), int(retreat_decisions), int(cooldown_decisions)
        self.closed_sign = float(closed_sign)
        self.norm_scale, self.norm_shift = [float(v) for v in norm_scale], [float(v) for v in norm_shift]
        self.long_carry_decisions, self.stall_decisions, self.stall_path = int(long_carry_decisions), int(stall_decisions), float(stall_path)
        self._n_retry, self._last_retry_step = 0, -10 ** 6

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        self.name = "R9F_reanchor_ap%g_m%d_lc%d_st%d__" % (self.empty_aperture, self.max_retries, self.long_carry_decisions, self.stall_decisions) + self.name

    def reset(self, episode):
        super().reset(episode)
        self._n_retry, self._last_retry_step = 0, -10 ** 6

    def _closed_hist(self, q, n):
        step = int(q.step)
        if step < n:
            return False
        hist = np.asarray(q.hist_a_exec)[step - n:step, :5, 6] * self.closed_sign
        return bool(np.all(hist > 0))

    def flag(self, q):
        """'empty', 'carry', 'stall' or None (checked at look decisions only)."""
        step = int(q.step)
        if self._n_retry >= self.max_retries or step - self._last_retry_step < self.cooldown_decisions:
            return None
        if self._closed_hist(q, self.hold_decisions):
            raw = np.asarray(q.raw_state, np.float64)
            if float(raw[6] - raw[7]) / 2.0 < self.empty_aperture:
                return "empty"
        if self.long_carry_decisions and self._closed_hist(q, self.long_carry_decisions):
            return "carry"
        if self.stall_decisions and step >= self.stall_decisions:
            rs = np.asarray(q.hist_rs)[step - self.stall_decisions:step, :3]
            path = float(np.linalg.norm(np.diff(np.vstack([rs, np.asarray(q.rs)[None, :3]]), axis=0), axis=1).sum())
            if path < self.stall_path:
                return "stall"
        return None

    def retreat_chunk(self, q, open_gripper):
        """Reverse replay of the last executed decisions with negated motion, in wire units; (H, 32) normalized."""
        step = int(q.step)
        n = min(self.retreat_decisions, step)
        hist = np.asarray(q.hist_a_exec)[step - n:step, :5, :7]            # executed controls, normalized
        ctrls = hist.reshape(-1, 7)[::-1]                                   # reverse execution order
        sc, sh = np.asarray(self.norm_scale, np.float32), np.asarray(self.norm_shift, np.float32)
        wire = ctrls * sc + sh
        wire[:, :6] = -wire[:, :6]                                          # undo the motion
        norm = (wire - sh) / sc
        chunk = np.zeros((self.H, 32), np.float32)
        m = min(len(norm), self.H)
        chunk[:m, :6] = norm[:m, :6]
        grip = -self.closed_sign if open_gripper else float(np.sign(ctrls[0, 6]) or self.closed_sign)
        chunk[:, 6] = grip
        return chunk

    def query(self, q):
        res = super().query(q)
        kind = self.flag(q)
        if kind is None:
            return res
        self._n_retry += 1
        self._last_retry_step = int(q.step)
        chunk = self.retreat_chunk(q, open_gripper=(kind == "empty"))
        res.action = chunk
        res.extras = dict(res.extras, r9f_retry=float({"empty": 1, "carry": 2, "stall": 3}[kind]), r9f_n_retry=float(self._n_retry))
        a = self._anchor
        self._remember_anchor(q, a["rows"], a["weights"], chunk)           # the blind tail serves the retreat too
        return res


class GraspMissCallsSigned(GraspMissCalls2):
    """GraspMissCalls2 with an explicit normalized closed-gripper sign (``closed_sign``: +1 pi0.5, -1 GR00T).

    The earlier classes test ``hist > 0`` for "closed", which is pi0.5's convention only; GR00T's normalized
    gripper is +1 = open / -1 = close (verified on recorded apertures), so their GR00T arms never trigger.
    """
    family = "r9f_r2_grasp_miss_calls_signed"

    def __init__(self, closed_sign=1.0, **kwargs):
        if closed_sign not in (1.0, -1.0, 1, -1):
            raise ValueError("closed_sign must be +1 or -1")
        self.closed_sign = float(closed_sign)
        super().__init__(**kwargs)
        self.name = self.name.replace("R9F_graspmiss2", "R9F_graspmissS%+d" % int(self.closed_sign))

    def empty_grasp(self, q):
        step = int(q.step)
        if step < self.hold_decisions or self._n_trigger >= self.max_calls:
            return False
        hist = np.asarray(q.hist_a_exec)[step - self.hold_decisions:step, :5, 6] * self.closed_sign
        if not np.all(hist > 0):
            return False
        raw = np.asarray(q.raw_state, np.float64)
        return float(raw[6] - raw[7]) / 2.0 < self.empty_aperture

    def long_carry(self, q):
        step = int(q.step)
        if step < self.long_carry_decisions or self._n_trigger >= self.max_calls:
            return False
        hist = np.asarray(q.hist_a_exec)[step - self.long_carry_decisions:step, :5, 6] * self.closed_sign
        return bool(np.all(hist > 0))
