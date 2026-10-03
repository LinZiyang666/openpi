"""Opus round 3: condition the policy takeover on an in-distribution arm configuration ("homing").

Same trouble trigger as round 2 (top-1 library-pace lag >= 12 at a fresh decision <= 80).  New: before the policy takes
over, the robot is driven back to the end-effector position it had at the start of the episode -- the configuration
every demonstration and every pure-policy episode starts from -- with the gripper open: first up by ``lift`` metres,
then across at that height, then down.  The homing commands are scripted from the robot's own state (no policy, no
vision model, no task knowledge); each fresh decision commits a 10-control homing segment computed from the current
state, and the blind decision serves its tail, exactly like a cache chunk.  Afterwards either the policy controls for
``window`` decisions (``window > 0``) and the cache resumes, or the cache resumes immediately (``window = 0``).

Why: on inits 0-19 / 20-29 data, a policy that takes over a troubled pi0.5 episode recovers it far more often when
the arm is near its start pose (round-3 DATA_ANALYSIS §3); escalated episodes start the takeover a median 0.25-0.32 m
away from it.

Maps (fitted on inits 0-19, ``tools/action_map.py``): wire_d = a_d * model_d + b_d for d < 6 (exact, residual 1e-7) and
delta_eef_pos per control ~= M @ wire[0:3] (about 1 cm per unit command).  Rotation is left unchanged (zero wire
rotation); the gripper is commanded open (pi0.5 model -1, GR00T model +1).
"""
from __future__ import annotations

import json
import math

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r08.methods.methods import AnchorCalls, coin

MODES = {"cache": 0, "homing": 1, "window": 2, "post": 3}


def homing_command(pos, target, M_inv, umax):
    """Constant wire position command (per control) that covers (target - pos) in ten controls, saturated."""
    u = M_inv @ ((np.asarray(target, float) - np.asarray(pos, float)) / 10.0)
    m = float(np.max(np.abs(u))) if u.size else 0.0
    return u * (umax / m) if m > umax else u


def waypoints(start, home, lift):
    start, home = np.asarray(start, float), np.asarray(home, float)
    z = max(start[2], home[2]) + lift
    return [np.array([start[0], start[1], z]), np.array([home[0], home[1], z]), home.copy()]


class HomingEscalation(AnchorCalls):
    """Pure cache; on the pace-lag trigger: scripted homing, then ``window`` policy decisions, then cache."""
    family = "r9o3_homing_escalation"
    uses_nonlibrary_action = True       # homing segments are scripted, not library actions

    def __init__(self, lag_threshold=12, deadline=80, window=24, home_max=8, home_tol=0.03, lift=0.05, umax=0.8,
                 action_map="", base_kwargs=None, base_fit="", random_seed=26100301, coin_domain="R9O3/homing"):
        super().__init__(p=1.0, base_kwargs=base_kwargs, base_fit=base_fit, random_seed=random_seed,
                         coin_domain=coin_domain)
        for name, v, lo in (("lag_threshold", lag_threshold, 1e-9), ("home_tol", home_tol, 1e-9), ("umax", umax, 1e-9)):
            if isinstance(v, bool) or not np.isfinite(v) or v < lo:
                raise ValueError(f"{name} must be positive")
        for name, v, lo in (("deadline", deadline, 0), ("window", window, 0), ("home_max", home_max, 1)):
            if type(v) is not int or v < lo:
                raise ValueError(f"{name} must be an int >= {lo}")
        if not np.isfinite(lift) or lift < 0:
            raise ValueError("lift must be >= 0")
        self.lag_threshold, self.deadline, self.window = float(lag_threshold), deadline, window
        self.home_max, self.home_tol, self.lift, self.umax = home_max, float(home_tol), float(lift), float(umax)
        self.action_map = str(action_map)
        self.name = f"R9O3_HOME_L{self.lag_threshold:g}_D{deadline}_W{window}"
        self._clear()

    def _clear(self):
        self._ident = None
        self._mode, self._home, self._wps, self._wp = "cache", None, None, 0
        self._t_trig = self._t_home = self._t_win = None
        self._lag, self._dist = float("nan"), float("nan")

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        m = json.loads(open(self.action_map).read())
        if m["model"] != ctx.cell.split("_")[0]:
            raise ValueError("action map model mismatch")
        self._a = np.array([f["a"] for f in m["affine"][:6]], np.float64)
        self._b = np.array([f["b"] for f in m["affine"][:6]], np.float64)
        if np.max([f["max_abs_residual"] for f in m["affine"][:6]]) > 1e-4:
            raise ValueError("action map is not affine on dims 0-5")
        self._M_inv = np.linalg.inv(np.asarray(m["M"], np.float64))
        self._grip_open = -1.0 if m["model"] == "pi05" else 1.0
        for x in (self._a, self._b, self._M_inv):
            x.flags.writeable = False
        self.fit_info.update(lag_threshold=self.lag_threshold, deadline=self.deadline, window=self.window,
                             home_max=self.home_max, home_tol=self.home_tol, lift=self.lift, umax=self.umax,
                             action_map=self.action_map, trigger="top1 library-pace lag", homing="scripted")

    def reset(self, episode):
        super().reset(episode)
        self._clear()

    # ------------------------------------------------------------------ state machine (one call per fresh decision)
    def _pos(self, q):
        raw = np.asarray(getattr(q, "raw_state", None), np.float64)
        if raw.shape != (8,) or not np.isfinite(raw).all():
            raise api.ContractError("homing needs the finite 8-d wire state")
        return raw[:3]

    def _advance(self, q, top1):
        ident = str(getattr(q.episode, "uid", q.episode))
        if ident != self._ident:
            self._clear()
            self._ident = ident
        step = int(q.step)
        pos = self._pos(q)
        if self._home is None:
            if step != 0:
                raise api.ContractError("the start pose must be recorded at decision 0")
            self._home = pos.copy()
        self._lag = float(step - int(self.base.lib_step[int(top1)]))
        self._dist = float(np.linalg.norm(pos - self._home))
        if self._mode == "cache" and self._lag >= self.lag_threshold and step <= self.deadline:
            self._mode, self._t_trig, self._t_home = "homing", step, step
            self._wps, self._wp = waypoints(pos, self._home, self.lift), 0
        if self._mode == "homing":
            while self._wp < len(self._wps) and np.linalg.norm(pos - self._wps[self._wp]) < self.home_tol:
                self._wp += 1
            if self._wp >= len(self._wps) or (step - self._t_home) // 2 >= self.home_max:
                self._mode = "window" if self.window > 0 else "post"
                self._t_win = step
        if self._mode == "window" and step >= self._t_win + self.window:
            self._mode = "post"
        return self._mode

    def _assignment(self, q):
        a = self.base._anchor
        if a is None or a.get("step") != int(q.step):
            raise api.ContractError("homing escalation needs the base anchor of this decision")
        mode = self._advance(q, a["rows"][0])
        u = coin(self.random_seed, q.task_id, q.episode.init, q.step, self.coin_domain)
        return (1.0 if mode == "window" else 0.0), u, dict(coin_domain=self.coin_domain, random_seed=self.random_seed,
                                                           **self._extras())

    def _extras(self):
        return dict(r9o_lag=self._lag, r9o3_mode=float(MODES[self._mode]), r9o3_dist=self._dist,
                    r9o3_t_trig=float(self._t_trig) if self._t_trig is not None else -1.0,
                    r9o3_t_win=float(self._t_win) if self._t_win is not None else -1.0)

    def homing_chunk(self, q, template):
        """Model-space chunk (H, 32) that moves toward the current waypoint with the gripper open."""
        pos = self._pos(q)
        target = self._wps[min(self._wp, len(self._wps) - 1)]
        u = homing_command(pos, target, self._M_inv, self.umax)
        wire = np.zeros(6)
        wire[:3] = u
        model6 = (wire - self._b) / self._a
        out = np.array(template, dtype=np.float32, copy=True)
        out[:, :6] = model6.astype(np.float32)
        out[:, 6] = self._grip_open
        return out

    def query(self, q):
        result = super().query(q)                       # base retrieval + _assignment (mode update, call decision)
        if self._mode == "homing":
            action = self.homing_chunk(q, result.action)
            result.action = action
            a = self.base._anchor
            self.base._remember_anchor(q, a["rows"], a["weights"], action)   # the blind decision serves the homing tail
        result.extras = dict(result.extras or {}, **self._extras())
        return result
