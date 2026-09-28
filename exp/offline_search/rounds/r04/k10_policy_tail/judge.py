"""K7 with one method-approved continuation of the preceding policy MISS.

HIT anchor_tail is inherited unchanged. A MISS still invalidates K1's action
anchor. Its vision proposal is retained separately ONLY to evaluate the identical
budget/proprioception/library gates; the served action comes from the policy.
K7's confirmed stuck and span progress see the dense vision mask, including the
policy-tail gap. No guard memo, actual HIT history or vision key is fabricated.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult, LookReason, policy_tail_chunk
from exp.offline_search.rounds.r04.k7_guard.judge import VisionConfirmedBlindMixedJudge


class PolicyTailJudge(VisionConfirmedBlindMixedJudge):
    family = "k10_policy_tail"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.base.serving != "anchor_tail":
            raise ValueError("K10 requires base serving=anchor_tail")
        if self.base.budget not in (0, 1):
            raise ValueError("K10 requires base budget 0 or 1")
        self.name = "K10__" + self.name
        self._policy_gate_anchor = None

    def reset(self, episode):
        super().reset(episode)
        self._policy_gate_anchor = None

    def query(self, q):
        res = super().query(q)
        self._policy_gate_anchor = copy.deepcopy(self.base._anchor)
        return res

    def policy_tail_step(self, bq):
        a = self._policy_gate_anchor
        self._policy_gate_anchor = None
        if (not bq.step or bq.prev_hit is not False or bq.blind_age != 0 or a is None
                or a["step"] != bq.step - 1 or a["episode"] != bq.episode.uid
                or a["task"] != bq.task_id or len(bq.hist_has_vision) != bq.step
                or not bq.hist_has_vision[-1] or bq.prev_a_exec is None
                or not np.isfinite(bq.rs).all() or not np.isfinite(bq.raw_state).all()):
            return LookReason(6, "policy_tail_lifecycle")
        try:
            action = policy_tail_chunk(bq.prev_a_exec)
        except ValueError:
            return LookReason(6, "policy_tail_invalid_chunk")
        # Temporarily make the old vision proposal available to K1's unchanged
        # gate evaluation. Only prev_hit on this *gate facade* is overridden;
        # hist_hit and all actual committed histories still contain the MISS.
        saved = self.base._anchor
        self.base._anchor = a
        try:
            gate = super().blind_step(replace(bq, prev_hit=True))
        finally:
            self.base._anchor = saved
        if isinstance(gate, LookReason):
            return gate
        return BlindResult(action, gate.rows, gate.weights, gate.library,
                           {**gate.extras, "policy_tail": 1., "policy_anchor_step": float(a["step"])})
