"""Set-valued guard switch using R6's unchanged verdict and blind-veto logic."""
from exp.offline_search.rounds.r06.p2_ablations.judge import BITS
from exp.offline_search.rounds.r06.p2_ablations.judge import TriggerCommitJudge as R6TriggerCommitJudge
from exp.offline_search.rounds.r06.p2_ablations.judge import TriggerGrootCommitJudge as R6TriggerGrootCommitJudge


def canonical_guards(disabled_guards):
    if isinstance(disabled_guards, (str, bytes)):
        raise ValueError("disabled_guards must be a collection of guard names")
    try:
        guards = set(disabled_guards)
    except TypeError as exc:
        raise ValueError("disabled_guards must be a collection of guard names") from exc
    if guards - BITS.keys():
        raise ValueError(f"unknown disabled guards: {guards - BITS.keys()}")
    return tuple(g for g in BITS if g in guards)


class _DisabledGuards:
    family = "r8_abl_trigger"

    def __init__(self, disabled_guards=(), **kwargs):
        guards = canonical_guards(disabled_guards)
        if "disabled_guard" in kwargs:
            raise ValueError("use disabled_guards, not disabled_guard")
        # Reuse R6's deployed-B restriction checks and every query/blind hook.
        super().__init__(disabled_guard="none", **kwargs)
        self.disabled_guards = guards
        self.disabled_mask = 0
        for guard in guards:
            self.disabled_mask |= BITS[guard]
        self.disabled_guard = "_".join(guards) or "none"
        self.name = f"R8_without_{self.disabled_guard}__" + self.name.removeprefix("P2_without_none__")


class TriggerCommitJudge(_DisabledGuards, R6TriggerCommitJudge):
    """pi0.5 B with any subset of its four verdict guards disabled."""


class TriggerGrootCommitJudge(_DisabledGuards, R6TriggerGrootCommitJudge):
    """GR00T B retaining the deployed model-aware terminal sign."""
