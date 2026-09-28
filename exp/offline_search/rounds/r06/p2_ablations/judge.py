"""B trigger leave-one-out. Compute every diagnostic, then mask only verdict bits.

The stuck counter, V7 features, and progress memos are never disabled: overtime
still consumes the same count >= 1 when the stuck MISS bit is disabled. Turning
off no-progress also removes its blind LOOK veto, but retains progress tracking.
Only B's deployed events=none/burst=0/lifecycle/monitor=off configuration is supported.
"""
from exp.offline_search.harness import api
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge
from exp.offline_search.rounds.r06.p1_groot_commit.judge import GrootCommitJudge

BITS = {"stuck": 1, "terminal": 2, "overtime": 4, "no_progress": 8}


class _TriggerMask:
    family = "r6_p2_trigger"

    def __init__(self, disabled_guard="none", **kwargs):
        if disabled_guard not in (*BITS, "all", "none"):
            raise ValueError(f"unknown disabled_guard: {disabled_guard}")
        super().__init__(**kwargs)
        if (not self.guards or self.events or self.burst != 0 or self.monitor != "off"
                or self.policy_tail_gate != "lifecycle" or self.stuck_guard != "vision_confirmed"
                or self.progress_guard != "noprog_span" or self.memo_reset_after_miss
                or self.base.serving != "anchor_tail" or self.base.budget != 1
                or self.base.gates != "budget_only"):
            raise ValueError("P2 trigger ablation requires deployed B settings; use disabled_guard='all' for B-off")
        self.disabled_guard = disabled_guard
        self.disabled_mask = 15 if disabled_guard == "all" else BITS.get(disabled_guard, 0)
        self.name = f"P2_without_{disabled_guard}__" + self.name

    def query(self, q):
        res = super().query(q)
        ex = res.extras
        old = int(ex["os_flags"])
        new = old & ~self.disabled_mask
        if new == old:
            return res
        if self._s["burst_end"] or self._s["ret_end"]:
            raise api.ContractError("P2 does not support active burst/return windows")
        reason = (new & -new).bit_length() if new else 0
        out = {**ex, "os_flags": float(new), "os_reason": float(reason), "os_force_miss": float(bool(new))}
        self._s["flag"][-1] = int(bool(new))
        return api.Result(res.topk, res.scores, res.confidence, action=res.action, library=res.library, extras=out)

    def blind_step(self, bq):
        if self.disabled_mask & BITS["no_progress"]:
            return self.base.blind_step(bq)
        return super().blind_step(bq)


class TriggerCommitJudge(_TriggerMask, CommitJudge):
    """pi0.5 B with one (or all/no) guard verdicts disabled."""


class TriggerGrootCommitJudge(_TriggerMask, GrootCommitJudge):
    """GR00T B, retaining P1's model-aware terminal sign."""
