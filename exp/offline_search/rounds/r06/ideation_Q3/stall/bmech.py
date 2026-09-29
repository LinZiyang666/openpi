"""B mechanism control: mask only the no-progress MISS verdict, keep LOOKs.

The parent runs unchanged. In particular this class does not override blind_step,
progress tracking, feature construction, tail lifecycle, or reset. Diagnostic
history (including the raw fired memo) remains the parent's, even when its verdict
is masked. Events/bursts/memo-reset variants are refused, as in deployed B.
"""
from exp.offline_search.harness import api
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge
from exp.offline_search.rounds.r06.p1_groot_commit.judge import GrootCommitJudge

NO_PROGRESS_BIT = 1 << 3


class _NoProgressMissOnly:
    family = "r6_q3_bmech"

    def __init__(self, mask_no_progress=True, **kwargs):
        if type(mask_no_progress) is not bool:
            raise ValueError("mask_no_progress must be a boolean")
        super().__init__(**kwargs)
        if (not self.guards or self.events or self.burst != 0 or self.monitor != "off"
                or self.policy_tail_gate != "lifecycle" or self.stuck_guard != "vision_confirmed"
                or self.progress_guard != "noprog_span" or self.memo_reset_after_miss
                or self.base.serving != "anchor_tail" or self.base.budget != 1
                or self.base.gates != "budget_only"):
            raise ValueError("Bmech requires the deployed B configuration")
        self._q3_mask_no_progress = mask_no_progress

    def query(self, q):
        res = super().query(q)
        if not self._q3_mask_no_progress:
            return res
        flags = int(res.extras["os_flags"])
        masked = flags & ~NO_PROGRESS_BIT
        if masked == flags:
            return res
        if self._s["burst_end"] or self._s["ret_end"]:
            raise api.ContractError("Bmech does not support active burst/return windows")
        reason = (masked & -masked).bit_length() if masked else 0
        extras = {**res.extras, "os_flags": float(masked), "os_reason": float(reason),
                  "os_force_miss": float(bool(masked))}
        # Deliberately retain _s['flag'], _vision_progress and _noprog_span:
        # they describe the unmasked diagnosis, not the served source.
        return api.Result(res.topk, res.scores, res.confidence, action=res.action,
                          library=res.library, extras=extras)


class BmechCommitJudge(_NoProgressMissOnly, CommitJudge):
    """Deployed pi0.5 B with no-progress MISS verdict disabled only."""


class BmechGrootCommitJudge(_NoProgressMissOnly, GrootCommitJudge):
    """Deployed GR00T B, preserving its model-aware terminal semantics."""
