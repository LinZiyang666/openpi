"""Configuration B for GR00T: C10 (Commit-Cache + committed policy rescue) with a model gripper constant.

B is R5 C10 (`rounds/r05/q1_commit/judge.py:CommitJudge`, ``policy_tail_gate="lifecycle"``): a real vision
anchor retrieves/synthesizes an AWM chunk and K1 ``anchor_tail`` budget 1 serves its rows 5..9 on the next,
vision-free decision (10 controls per cache source, no policy call). The only MISS source is the K7
vision-confirmed guard (MixedJudge guards: stuck / terminal-with-closed-gripper / overtime / no-progress,
``--os-judge guard_only``); the MISS policy chunk is then committed for 10 controls through the lifecycle policy
tail (plugin ``--os-policy-tail --os-policy-tail-blocks 1``). Nothing here changes C10's decision logic.

Why K7 (and hence C10) refused GR00T. With an all-vision history K7 executes stock ``MixedJudge.query``, whose
terminal guard (os_reason 2) reads "closed" as ``prev_a_exec[4, 6] >= 0``: LIBERO's positive-close convention,
which pi0.5's normalized actions use. GR00T's normalized gripper is OPENNESS (its wire adapter applies
``1 - 2x`` then ``sign``): a value < 0 is closed. On GR00T stock would fire "terminal while OPEN" and never fire
"terminal while still holding". K1's gap branch (used by K7 after blind decisions) already applies the
model-aware sign ``closed = gexec < 0 if groot else gexec > 0``. Every other guard input is sign free (stuck:
state motion + camera cosine; overtime/lag; no-progress; the optional grip event compares like with like).

Fix = one model constant, ``CLOSED_SIGN`` (pi05 +1, groot -1). After the stock all-vision verdict, the terminal
bit is recomputed with that sign and the verdict (os_flags / os_reason / os_force_miss) re-derived exactly as stock
derives it. For pi05 the recomputation is the identity, so this class is C10 bit for bit on pi05 (tested in
``nesting.py``). The re-derivation is implemented for ``burst <= 1`` (C10's deployed setting): a burst window would
need the pre-decision burst state, which stock overwrites, so larger bursts are refused.

Thresholds are whatever MixedJudge / V7 / K1 fit on the deployed library, i.e. the GR00T library for GR00T arms:
m_thr (library 10th percentile of consecutive-row valid-state L2 motion), c_thr (95th percentile of the
consecutive-row min-camera task-centred key cosine), V7's library-LOEO isotonic calibration, median episode length
per task, K1 per-task state scales and motion10. The guard constants stuck_thr 2, lag_thr 5, noprog_n 3,
prog_eps 0.5 are the same fixed constants as C10's.
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r04.k7_guard.judge import VisionConfirmedBlindMixedJudge
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge

# Sign of a CLOSED normalized gripper command (action dim 6) per model; gexec = sign(prev_a_exec[4, 6]) in {-1, +1}.
CLOSED_SIGN = {"pi05": 1.0, "groot": -1.0}
TERMINAL_BIT = 1 << 1          # os_flags bit of guard 2 (terminal row AND closed executed gripper)


def gripper_closed(gexec: float, model: str) -> bool:
    """Model-aware "closed" for the executed gripper sign; identical to K1's predicate."""
    return float(gexec) * CLOSED_SIGN[model] > 0


class GrootCommitJudge(CommitJudge):
    """C10 with the model gripper constant; deployable on GR00T, bit-identical to CommitJudge on pi05."""

    family = "r6_p1_commit"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.policy_tail_gate != "lifecycle":
            raise ValueError("P1 is C10: policy_tail_gate must be 'lifecycle'")
        if self.monitor != "off":
            raise ValueError("P1 keeps C10's deployed monitor='off'")
        if self.stuck_guard != "vision_confirmed" or self.progress_guard != "noprog_span":
            raise ValueError("P1 requires stuck_guard='vision_confirmed' and progress_guard='noprog_span'")
        if self.burst > 1:
            raise ValueError("the terminal-sign re-derivation supports burst <= 1 (C10 deploys burst 0)")
        self.closed_sign = None
        self.name = "P1__" + self.name

    def fit(self, lib, ctx):
        if ctx.model not in CLOSED_SIGN:
            raise api.SkipCell(f"unknown model {ctx.model!r}")
        self.closed_sign = CLOSED_SIGN[ctx.model]
        if ctx.model == "pi05":
            super().fit(lib, ctx)                      # exactly C10's fit, including K7's own model check
        else:
            # CommitJudge.fit adds only the optional monitor (refused above); K7.fit adds only the GR00T refusal
            # this class resolves. Everything else is the same MixedJudge / V7 / K1 fit on the GR00T library.
            super(VisionConfirmedBlindMixedJudge, self).fit(lib, ctx)
        self.fit_info["p1"] = {"model": ctx.model, "closed_sign": self.closed_sign, "m_thr": self.m_thr,
                               "c_thr": self.c_thr, "stuck_thr": self.stuck_thr, "lag_thr": self.lag_thr,
                               "noprog_n": self.noprog_n, "prog_eps": self.prog_eps}

    def query(self, q):
        hv = getattr(q, "hist_has_vision", None)
        stock_branch = hv is None or bool(np.all(hv))       # the K7 branch that runs stock MixedJudge.query
        res = super().query(q)
        if self.closed_sign is None:
            raise api.ContractError("P1 must be fitted before query")
        if not stock_branch or self.closed_sign > 0:
            return res                                     # K1/K7 gap branch is already sign-aware; pi05: identity
        return self._resign_terminal(res)

    def _resign_terminal(self, res):
        """Stock's verdict tail with the model's closed sign in guard 2 (burst <= 1)."""
        ex = res.extras
        flags = int(ex["os_flags"])
        closed = float(ex.get("gexec", 0.)) * self.closed_sign > 0
        want = bool(self.guards and float(ex.get("term1", 0.)) == 1. and closed)
        new = (flags & ~TERMINAL_BIT) | (TERMINAL_BIT if want else 0)
        if new == flags:
            return res
        s = self._s
        if self.burst > 1 or s["burst_end"] or s["ret_end"]:
            raise api.ContractError("terminal-sign re-derivation requires burst <= 1")
        reason = (new & -new).bit_length() if new else 0
        out = dict(ex)
        # Stock with burst <= 1: phase 0, burst_left 0, confidence unchanged; only the fired set changes.
        out.update(os_force_miss=float(reason != 0), os_reason=float(reason), os_flags=float(new),
                   os_phase=0., burst_left=0.)
        s["flag"][-1] = int(new != 0)
        return api.Result(res.topk, res.scores, res.confidence, action=res.action, library=res.library, extras=out)
