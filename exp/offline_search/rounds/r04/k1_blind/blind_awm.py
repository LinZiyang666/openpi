"""Fixed AWM anchors with bounded, vision-free library continuation (R4-A)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from exp.offline_search.harness import api, dims
from exp.offline_search.rounds.r02.g1_awm.awm import AWM, _kernel_w
from exp.offline_search.rounds.r03.h1_trap.awm3 import AWM3

try:
    from exp.offline_search.closed_loop.blind import BlindResult, LookReason
except ImportError:  # K2 installs this shared contract; fallback has identical fields.
    @dataclass(frozen=True, slots=True)
    class LookReason:
        code: int
        name: str

    @dataclass(frozen=True, slots=True)
    class BlindResult:
        action: np.ndarray
        rows: np.ndarray
        weights: np.ndarray
        library: str
        extras: dict[str, float]


def episode_id(q):
    return getattr(q.episode, "uid", q.episode)


def consecutive_next(lib):
    """Only real consecutive decision edges, never across an episode/task boundary."""
    nxt = np.array(lib.next, dtype=np.int32, copy=True)
    r = np.flatnonzero(nxt >= 0)
    n = nxt[r]
    good = ((lib.episode[r] == lib.episode[n]) & (lib.task_id[r] == lib.task_id[n])
            & (lib.step[n] == lib.step[r] + 1))
    nxt[r[~good]] = -1
    return nxt


class _BlindMixin:
    family = "k1_blind"

    def __init__(self, serving="phase_particles", budget=2, gates="all", residual_threshold=.5, **kw):
        if serving not in ("phase_particles", "kernel_clock", "top1_clock", "anchor_tail"):
            raise ValueError("unknown blind serving variant")
        if int(budget) != budget or not 0 <= budget <= 4:
            raise ValueError("budget must be 0, 1, 2, 3 or 4")
        if gates not in ("all", "budget_only"):
            raise ValueError("gates must be all or budget_only")
        if not np.isfinite(residual_threshold) or residual_threshold <= 0:
            raise ValueError("residual_threshold must be positive and finite")
        if kw.get("k", 16) != 16:
            raise ValueError("blind anchors require the fixed 16-member AWM kernel")
        super().__init__(**kw)
        self.serving, self.budget, self.gates = serving, int(budget), gates
        self.residual_threshold = float(residual_threshold)
        self.name = f"BL_{serving}_B{budget}_{gates}_r{residual_threshold:g}__{self.name}"
        self._anchor = None
        self.last_blind_extras = {}

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        deployed = lib if self.cand_name == "current" else ctx.open_library(self.cand_name)
        self._fit_blind(deployed)

    def _fit_blind(self, lib):
        self.blind_next = consecutive_next(lib)
        r = np.arange(lib.L)
        nxt = np.where(self.blind_next >= 0, self.blind_next, r)
        prv = r.copy()
        edge = np.flatnonzero(self.blind_next >= 0)
        prv[self.blind_next[edge]] = edge
        self.blind_rs = np.array(dims.valid_state(lib.rs, self.model), dtype=np.float32, copy=True)
        g = np.asarray(lib.action[:, :5, 6]) >= 0
        self.blind_event = ((g != g[:, :1]).any(1) | (g[:, 0] != g[prv, 4]) | (g[:, 4] != g[nxt, 0]))
        self.blind_terminal = np.asarray(lib.step >= lib.ep_len - 2, dtype=bool)
        self.state_scale_by_task, self.motion10 = {}, {}
        for task in lib.tasks():
            rows = lib.rows_of_task(task)
            scale = np.maximum(np.std(self.blind_rs[rows], axis=0), .05).astype(np.float32)
            edges = rows[self.blind_next[rows] >= 0]
            motion = np.sqrt(np.mean(((self.blind_rs[nxt[edges]] - self.blind_rs[edges]) / scale) ** 2, axis=1))
            self.state_scale_by_task[int(task)] = scale
            self.motion10[int(task)] = float(np.percentile(motion, 10)) if len(motion) else 0.
        for a in [self.blind_next, self.blind_rs, self.blind_event, self.blind_terminal,
                  *self.state_scale_by_task.values()]:
            a.flags.writeable = False

    def reset(self, episode):
        super().reset(episode)
        self.invalidate_anchor()
        self.last_blind_extras = {}

    def invalidate_anchor(self):
        self._anchor = None

    def _remember_anchor(self, q, rows, weights, action):
        rows = np.array(rows, dtype=np.int64, copy=True)
        weights = np.array(weights, dtype=np.float32, copy=True)
        self._anchor = dict(rows=rows, weights=weights, phase=rows.copy(),
                            action=np.array(action, dtype=np.float32, copy=True),
                            rs=np.array(dims.valid_state(q.rs, self.model), dtype=np.float32, copy=True),
                            task=int(q.task_id), episode=episode_id(q), step=int(q.step), last_step=int(q.step))

    def query(self, q):
        res = super().query(q)
        d = -np.asarray(res.scores, np.float64)
        w = _kernel_w(d - d[0], self.kref)
        self._remember_anchor(q, res.topk, (w / w.sum()).astype(np.float32), res.action)
        # AWM's adjacent-key diagnostic is unavailable across a blind gap. Selection,
        # action, score and confidence remain exactly the superclass result.
        hv = getattr(q, "hist_has_vision", None)
        if q.step > 0 and hv is not None and not bool(hv[-1]):
            res.extras.pop("still", None)
        return res

    def os_synth(self, q, rows, w):
        action = super().os_synth(q, rows, w)
        w = np.asarray(w, np.float64)
        self._remember_anchor(q, rows, (w / w.sum()).astype(np.float32), action)
        return action

    def _advance(self, rows, n):
        rows = np.asarray(rows, np.int64).copy()
        for _ in range(n):
            nxt = self.blind_next[rows]
            rows = np.where(nxt >= 0, nxt, rows)
        return rows

    def lifecycle_reason(self, bq):
        a = self._anchor
        reason = None
        if int(bq.step) == 0:
            reason = "first_decision"
        elif bq.prev_hit is False or (bq.prev_hit is not None and not bool(bq.prev_hit)):
            reason = "after_miss"
        elif a is None:
            reason = "invalid_anchor"
        elif int(bq.task_id) != a["task"] or episode_id(bq) != a["episode"]:
            reason = "episode_or_task_change"
        elif int(bq.step) != a["last_step"] + 1 or int(bq.blind_age) != int(bq.step) - a["step"] - 1:
            reason = "decision_discontinuity"
        elif getattr(bq, "executed_steps", 5) != 5:
            reason = "executed_steps"
        elif not np.isfinite(dims.valid_state(bq.rs, self.model)).all():
            reason = "invalid_state"
        if reason is not None:
            self.invalidate_anchor()
            return LookReason(6, reason)
        return None

    def dense_motion(self, q):
        """Dense normalized movements, including the current interval, from online state only."""
        scale = self.state_scale_by_task[int(q.task_id)]
        rs = np.asarray(dims.valid_state(q.hist_rs, self.model), np.float32)
        now = np.asarray(dims.valid_state(q.rs, self.model), np.float32)
        if not len(rs):
            return np.empty(0, np.float32)
        seq = np.concatenate([rs, now[None]], axis=0)
        return np.sqrt(np.mean((np.diff(seq, axis=0) / scale) ** 2, axis=1))

    def blind_step(self, bq):
        reason = self.lifecycle_reason(bq)
        if reason is not None:
            self.last_blind_extras = {"look_reason": 6.}
            return reason
        a = self._anchor
        h = int(bq.step) - a["step"]
        ex = {"anchor_step": float(a["step"]), "blind_age": float(h),
              "gate_budget": float(h > self.budget), "gate_grip": 0., "gate_terminal": 0.,
              "gate_motion": 0., "gate_residual": 0.}
        self.last_blind_extras = ex
        if h > self.budget:
            return LookReason(1, "budget")
        if self.serving == "anchor_tail" and (h + 1) * 5 > self.H:
            return LookReason(6, "tail_exhausted")
        rows, w = a["rows"], a["weights"]
        nominal = self._advance(rows, h)
        scale = self.state_scale_by_task[a["task"]]
        current = np.asarray(dims.valid_state(bq.rs, self.model), np.float32)
        expected = np.einsum("k,kd->d", w, self.blind_rs[nominal] - self.blind_rs[rows])
        residual = float(np.sqrt(np.mean(((current - a["rs"] - expected) / scale) ** 2)))
        motion = self.dense_motion(bq)
        grip_mass = float(w @ (self.blind_event[nominal] | self.blind_event[self._advance(nominal, 1)]))
        terminal_mass = float(w @ self.blind_terminal[nominal])
        low = len(motion) >= 2 and bool(np.all(motion[-2:] < self.motion10[a["task"]]))
        ex.update(grip_mass=grip_mass, terminal_mass=terminal_mass, displacement_residual=residual,
                  motion=float(motion[-1]) if len(motion) else 0., motion10=self.motion10[a["task"]],
                  gate_grip=float(grip_mass >= .20), gate_terminal=float(terminal_mass >= .20),
                  gate_motion=float(low), gate_residual=float(residual > self.residual_threshold))
        for j in range(len(scale)):
            ex[f"state_{j}"] = float(current[j])
            ex[f"expected_delta_{j}"] = float(expected[j])
        if self.gates == "all":
            for code, key, name in ((2, "gate_grip", "gripper_event_ahead"), (3, "gate_terminal", "near_terminal"),
                                    (4, "gate_motion", "low_motion"), (5, "gate_residual", "displacement_residual")):
                if ex[key]:
                    return LookReason(code, name)
        selected = nominal
        if self.serving == "phase_particles":
            offsets = np.arange(h - 1, h + 2)
            candidates = np.stack([self._advance(rows, int(off)) for off in offsets])
            cost = np.mean(((self.blind_rs[candidates] - current) / scale) ** 2, axis=-1)
            cost += (.05 * (offsets - h) ** 2)[:, None]
            phase0 = self.lib_step[a["phase"]]
            allowed = ((self.lib_step[candidates] >= phase0) & (self.lib_step[candidates] <= phase0 + 2))
            cost = np.where(allowed, cost, np.inf)
            if not np.isfinite(cost).any(axis=0).all():
                self.invalidate_anchor()
                return LookReason(6, "invalid_phase")
            selected = np.take_along_axis(candidates, np.argmin(cost, axis=0)[None], axis=0)[0]
        action = np.zeros((self.H, 32), np.float32)
        if self.serving == "anchor_tail":
            tail = a["action"][h * 5:, :7]
            action[:len(tail), :7] = tail
            action[len(tail):, :7] = tail[-1]  # wire padding; never extends tail eligibility
            selected = rows
        elif self.serving == "top1_clock":
            selected = nominal[:1]
            w = np.ones(1, np.float32)
            action[:, :7] = self.act[selected[0], :, :7]
        else:
            action[:, :7] = np.tensordot(w, self.act[selected, :, :7], 1)
        a["phase"] = selected.copy() if self.serving == "phase_particles" else nominal.copy()
        a["last_step"] = int(bq.step)
        ex["phase_mean"] = float(a["weights"] @ (self.lib_step[a["phase"]] - self.lib_step[rows]))
        for i, row in enumerate(a["phase"]):
            ex[f"phase_{i}"] = float(self.lib_step[row] - self.lib_step[rows[i]])
        return BlindResult(action, selected.astype(np.int64), w.copy(), self.cand_name, ex)

    def bytes_per_entry(self):
        # State is already retained by joint AWM; vision-only variants need the eight dims.
        return super().bytes_per_entry() + 6 + (0 if self.features == "joint" or self.step0_joint else 32)


class BlindAWM(_BlindMixin, AWM):
    """Ordinary query is exactly AWM; blind_step uses only the most recent vision anchor."""


class BlindAWM3(_BlindMixin, AWM3):
    """Optional AWM3-compatible anchor adapter; ordinary query retains AWM3 behavior."""
