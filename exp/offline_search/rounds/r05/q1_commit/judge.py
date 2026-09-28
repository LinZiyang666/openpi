"""R5 C10 policy-plan completion and D1 command-matched grasp inspection.

Execution counts are enforced by the plugin before calling the blind hook. An
absent count follows its documented L=5 client contract. No histories or guard
memos are rewritten here. Mutable state belongs to each cloned connection.
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult, LookReason, policy_tail_chunk
from exp.offline_search.harness import api
from exp.offline_search.rounds.r04.k10_policy_tail.judge import PolicyTailJudge
from exp.offline_search.rounds.r04.k7_guard.judge import VisionConfirmedBlindMixedJudge
from exp.offline_search.rounds.r04.k1_blind.blind_awm import consecutive_next


def fit_contact(state, action, next_row):
    """D's reference arithmetic and dtypes, fitted on the deployed library only."""
    aperture = (state[:, 6].astype(float) - state[:, 7]) / 2
    lo, hi = np.percentile(aperture, [1, 99]).astype(np.float32)
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("D1 requires finite, nondegenerate library aperture quantiles")
    width = ((aperture - lo) / (hi - lo)).astype(np.float32)
    pattern = np.sum((action[:, :5, 6] >= 0).astype(np.uint8)
                     * (1 << np.arange(5, dtype=np.uint8)), axis=1).astype(np.uint8)
    table = dict(width=width, pattern=pattern, next=np.array(next_row, np.int32), lo=lo, hi=hi)
    for key in ("width", "pattern", "next"):
        table[key].flags.writeable = False
    return table


def contact_alarm(table, state, last_two_heads, rows, weights):
    """Exact pi05 D1 thresholds; malformed/nonfinite inputs cannot raise an alarm."""
    heads = np.asarray(last_two_heads)
    rows, weights = np.asarray(rows), np.asarray(weights)
    if (heads.ndim != 3 or len(heads) < 2 or heads.shape[1] < 5 or heads.shape[2] < 7
            or not np.isfinite(heads[-2:, :5, :7]).all()
            or not np.isfinite(state).all()):
        return False
    if not np.all(heads[-2:, :5, 6] > 0):
        return False
    width = ((float(state[6]) - float(state[7])) / 2 - table['lo']) / (table['hi'] - table['lo'])
    if width >= .05:
        return False
    if (rows.ndim != 1 or not len(rows) or rows.dtype.kind not in 'iu'
            or weights.shape != rows.shape or np.any(rows < 0)
            or np.any(rows >= len(table['next'])) or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        return False
    pattern = int(np.sum((heads[-1, :5, 6] >= 0) * (1 << np.arange(5))))
    nr = table['next'][rows]
    valid = (nr >= 0) & (table['pattern'][rows] == pattern)
    mass = weights[valid].sum()
    if mass < .5:
        return False
    ww, expected = weights[valid] / mass, table['width'][nr[valid]]
    return bool(ww @ expected > .25 and ww @ (expected > .25) >= .75)


def fit_monitor(lib):
    """Ideation A's own-library LOEO ridge and xyz p99, without evaluation data."""
    rs = np.asarray(lib.rs[:, :8], float)
    nxt = consecutive_next(lib)
    r = np.flatnonzero(nxt >= 0)
    X = np.column_stack((np.ones(len(r)), rs[r], np.asarray(lib.action[r, :5, :7]).mean(1)))
    Y = rs[nxt[r], :3] - rs[r, :3]
    labels = np.asarray(lib.episode)[r]
    if len(np.unique(labels)) < 2 or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("loeo_xyz99 requires finite edges in at least two library episodes")

    def solve(x, y):
        gram = x.T @ x
        reg = np.eye(16) * (.01 * np.trace(gram[1:, 1:]) / 15)
        reg[0, 0] = 0
        return np.linalg.solve(gram + reg, x.T @ y)

    error = np.empty(len(r))
    for ep in np.unique(labels):
        test = labels == ep
        pred = X[test] @ solve(X[~test], Y[~test])
        scale = np.maximum(Y[~test].std(0), .005)
        error[test] = np.sqrt(np.mean(((Y[test] - pred) / scale) ** 2, axis=1))
    result = dict(B=solve(X, Y).astype(np.float32),
                  scale=np.maximum(Y.std(0), .005).astype(np.float32),
                  q99=np.float32(np.quantile(error, .99)))
    result['B'].flags.writeable = result['scale'].flags.writeable = False
    return result


class CommitJudge(PolicyTailJudge):
    family = "r5_commit"

    def __init__(self, policy_tail_gate="inherited", monitor="off", **kwargs):
        if policy_tail_gate not in ("inherited", "lifecycle"):
            raise ValueError("policy_tail_gate must be inherited or lifecycle")
        if monitor not in ("off", "loeo_xyz99"):
            raise ValueError("monitor must be off or loeo_xyz99")
        if monitor != "off" and policy_tail_gate != "lifecycle":
            raise ValueError("monitor requires policy_tail_gate=lifecycle; inherited preserves K10")
        super().__init__(**kwargs)
        self.policy_tail_gate, self.monitor = policy_tail_gate, monitor
        if policy_tail_gate != "inherited":
            self.name = f"C10_{monitor}__" + self.name

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        if self.monitor == "loeo_xyz99":
            deployed = lib if self.base.cand_name == "current" else ctx.open_library(self.base.cand_name)
            self.policy_monitor = fit_monitor(deployed)

    def policy_tail_step(self, bq):
        if self.policy_tail_gate == "inherited":
            return super().policy_tail_step(bq)
        a = self._policy_gate_anchor
        self._policy_gate_anchor = None
        step = int(bq.step)
        if (step <= 0 or bq.prev_hit is not False or bq.blind_age != 0 or a is None
                or a['step'] != step - 1 or a['episode'] != bq.episode.uid
                or a['task'] != bq.task_id or len(bq.hist_has_vision) != step
                or not bq.hist_has_vision[-1] or len(bq.hist_hit) != step
                or bq.hist_hit[-1] != 0 or len(bq.hist_rs) != step
                or len(bq.hist_a_exec) != step or bq.prev_a_exec is None
                or getattr(bq, 'executed_steps', 5) != 5
                or not np.isfinite(bq.rs).all() or not np.isfinite(bq.raw_state).all()):
            return LookReason(6, "policy_tail_lifecycle")
        try:
            action = policy_tail_chunk(bq.prev_a_exec)
        except ValueError:
            return LookReason(6, "policy_tail_invalid_chunk")
        ex = dict(policy_tail=1., policy_anchor_step=float(a['step']), policy_tail_lifecycle=1.)
        if self.monitor == "loeo_xyz99":
            old = np.asarray(bq.hist_rs[-1, :8], float)
            if not np.isfinite(old).all():
                return LookReason(6, "policy_tail_invalid_previous_state")
            x = np.r_[1., old, np.asarray(bq.prev_a_exec)[:5, :7].mean(0)]
            m = self.policy_monitor
            residual = float(np.sqrt(np.mean(((bq.rs[:3] - old[:3] - x @ m['B']) / m['scale']) ** 2)))
            ex.update(policy_xyz_residual=residual, policy_xyz_q99=float(m['q99']))
            if residual > m['q99']:
                self.base.last_blind_extras = ex
                return LookReason(5, "policy_head_xyz_residual")
        self.base.last_blind_extras = ex
        return BlindResult(action, a['rows'].copy(), a['weights'].copy(), self.base.cand_name, ex)


class GraspCheckJudge(VisionConfirmedBlindMixedJudge):
    family = "r5_grasp_check"
    LOOK_CODE = 9
    MISS_CODE = 9

    def __init__(self, base_kwargs=None, **kwargs):
        base_kwargs = {"serving": "anchor_tail", "budget": 1, "gates": "budget_only", **(base_kwargs or {})}
        if (base_kwargs['serving'], base_kwargs['budget'], base_kwargs['gates']) != ('anchor_tail', 1, 'budget_only'):
            raise ValueError("D1 requires anchor_tail, budget=1, gates=budget_only")
        if kwargs.get('memo_reset_after_miss', False):
            raise ValueError("D1 preserves progress memos")
        super().__init__(base_kwargs=base_kwargs, **kwargs)
        self.name = "D1__" + self.name

    def fit(self, lib, ctx):
        if ctx.model != "pi05":
            raise api.SkipCell("GraspCheckJudge D1 supports pi05 only; GR00T is unsupported")
        super().fit(lib, ctx)
        deployed = lib if self.base.cand_name == 'current' else ctx.open_library(self.base.cand_name)
        self.contact = fit_contact(deployed.rs, deployed.action, deployed.next)

    def bytes_per_entry(self):
        # Compact float32 aperture + uint8 command pattern + int32 successor;
        # the two scalar quantiles add another eight fixed bytes per fit.
        return super().bytes_per_entry() + 9

    def reset(self, episode):
        super().reset(episode)
        self._grasp_used = False
        self._grasp_pending = None
        self._grasp_issued = None

    def _sync_grasp(self, q):
        if getattr(self, '_guard_identity', None) != (str(q.episode.uid), int(q.task_id)):
            self.reset(q.episode)
        p = self._grasp_issued
        if p is not None and q.step > p:
            # query() is a proposal, not execution. Only committed real-vision
            # MISS history consumes the allowance (also handles skipped calls).
            hv = getattr(q, 'hist_has_vision', np.ones(q.step, bool))
            if (len(q.hist_hit) > p and len(hv) > p and len(q.hist_a_exec) > p
                    and q.hist_hit[p] == 0 and hv[p]
                    and np.isfinite(q.hist_a_exec[p]).all()):
                self._grasp_used = True
            self._grasp_pending = self._grasp_issued = None
        elif self._grasp_pending is not None and q.step > self._grasp_pending:
            self._grasp_pending = None

    def blind_step(self, bq):
        self._sync_grasp(bq)
        if self._grasp_pending == bq.step:
            return LookReason(self.LOOK_CODE, "grasp_aperture_contradiction")
        res = super().blind_step(bq)
        if (isinstance(res, BlindResult) and not self._grasp_used and bq.step >= 2
                and contact_alarm(self.contact, bq.rs, bq.hist_a_exec[-2:], res.rows, res.weights)):
            self._grasp_pending = int(bq.step)
            return LookReason(self.LOOK_CODE, "grasp_aperture_contradiction")
        return res

    def query(self, q):
        self._sync_grasp(q)
        res = super().query(q)
        if not self._grasp_used and self._grasp_pending == q.step:
            self._grasp_issued = int(q.step)
            res.extras = {**res.extras, 'grasp_base_reason': res.extras.get('os_reason', 0.),
                          'os_force_miss': 1., 'os_reason': float(self.MISS_CODE), 'grasp_check': 1.}
        return res
