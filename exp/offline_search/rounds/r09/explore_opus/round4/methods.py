"""Opus round 4: bounded escalation window on top of fable's half-strength corrector.

The cache under the escalation controller is fable's round-2 ``CorrectedCache`` *fitted artifact*, unchanged (per-task
ridge/RFF heads fitted on inits 0-19, blend .5, motion channels only, gripper and the 10-control commit unchanged) --
the exact objects served by the standalone arms ``r9f2_{pi05,groot}_l10_50_corr05pt``.

Correction path: this controller is an R8 ``AnchorCalls`` (not a judge).  At every fresh decision it calls
``self.base.query(q)``; ``CorrectedCache.query`` applies the head and re-remembers the anchor with the corrected chunk, so
(i) a cache decision serves the corrected chunk and (ii) the following blind decision serves the corrected tail.  The
judge-path problem fable found (judges retrieve through ``os_score_all`` / ``os_synth`` and bypass ``base.query``) does
not arise here; the plugin only uses those hooks under ``--os-gpu-retrieval``, which these arms do not set.  A unit test
checks that the served chunk equals the standalone corrector's and differs from the plain cache on motion channels only.

Escalation (round 2, unchanged constants): at the first fresh decision with ``lag = decision index - library step of the
top-1 row >= 12`` and decision index ``<= 80``, every fresh decision for the next 24 decisions is a policy call
(10-control commit with the existing policy tail); then the corrected cache resumes; never re-armed.
``force_trigger_at`` is a debug-only switch (default empty, never set in a frozen arm) that escalates at the listed
decision indices so CPU selftests can exercise the call path.
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r09.explore_fable.round2.tools.methods import CorrectedCache
from exp.offline_search.rounds.r09.explore_opus.round2.methods import EscalateCalls

CORRECTED_SPEC = "exp.offline_search.rounds.r09.explore_fable.round2.tools.methods:CorrectedCache"


class CorrectedEscalateCalls(EscalateCalls):
    """Half corrector as the cache + bounded escalation window (round-2 rule)."""
    family = "r9o4_corrected_escalate_window"
    uses_nonlibrary_action = True

    def __init__(self, corrected_fit="", corrected_kwargs=None, lag_threshold=12, deadline=80, window=24,
                 random_seed=26100401, coin_domain="R9O4/corr_escalate", force_trigger_at=()):
        super().__init__(lag_threshold=lag_threshold, deadline=deadline, base_kwargs={}, base_fit="",
                         random_seed=random_seed, coin_domain=coin_domain, window=window)
        self.corrected_fit = str(corrected_fit)
        self.corrected_kwargs = dict(corrected_kwargs or {})
        self.force_trigger_at = tuple(int(v) for v in (force_trigger_at or ()))
        self.name = f"R9O4_corr_esc_L{self.lag_threshold:g}_D{deadline}_W{window}"

    def fit(self, lib, ctx):
        with open(self.corrected_fit, "rb") as f:
            blob = FitUnpickler(f).load()
        if blob["spec"] != CORRECTED_SPEC or blob["cell"] != ctx.cell or blob["kwargs"] != self.corrected_kwargs:
            raise ValueError("corrected-cache artifact spec/cell/kwargs mismatch")
        base = blob["method"]
        if type(base) is not CorrectedCache:
            raise api.ContractError("base must be exactly fable's round-2 CorrectedCache")
        if (base.serving, base.budget, base.gates) != ("anchor_tail", 1, "budget_only"):
            raise api.ContractError("requires the frozen ten-control cache")
        if base.blend != float(self.corrected_kwargs.get("blend", base.blend)) or base.correct_gripper:
            raise api.ContractError("corrector must be the motion-only artifact named in corrected_kwargs")
        self.base = base
        self.base.prof = self.prof
        self.fit_info = dict(corrected_fit=self.corrected_fit, corrector_blend=float(base.blend),
                             corrector_head=str(base.head_path), corrector_path="base.query",
                             lag_threshold=self.lag_threshold, deadline=self.deadline, window=self.window,
                             persistent=self.window is None, force_trigger_at=list(self.force_trigger_at),
                             policy_commit_controls=10, policy_tail_blocks=1)

    def _esc_update(self, q, top1):
        esc = super()._esc_update(q, top1)
        if not esc and self._esc_step is None and int(q.step) in self.force_trigger_at:
            self._esc_step = int(q.step)                     # debug-only forced trigger (selftests)
            return True
        return esc
