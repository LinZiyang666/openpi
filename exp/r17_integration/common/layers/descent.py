"""Generic per-task first-failure descent gate consumed by P2's conductor."""
from dataclasses import dataclass, field


def passes_gate(candidate_success: int, full_success: int, n_dev: int, tolerance_pp=5):
    if n_dev <= 0 or not 0 <= candidate_success <= n_dev or not 0 <= full_success <= n_dev:
        raise ValueError("invalid complete development cohort")
    return 100 * (candidate_success - full_success) >= -tolerance_pp * n_dev


@dataclass
class TaskDescent:
    levels: tuple[str, ...]
    expected_uids: frozenset[str]
    full_success: int
    tolerance_pp: float = 5
    index: int = 0
    selected: str = "full"
    stopped: bool = False
    accepted: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.levels or len(set(self.levels)) != len(self.levels) or not self.expected_uids:
            raise ValueError("descent needs unique ordered levels and a complete nonempty cohort")
        passes_gate(self.full_success, self.full_success, len(self.expected_uids), self.tolerance_pp)

    def accept(self, uid, success, *, accepted=True, error=False):
        if self.stopped or not accepted or error or type(success) is not bool or uid not in self.expected_uids or uid in self.accepted:
            return False
        self.accepted[uid] = bool(success)
        if set(self.accepted) == set(self.expected_uids):
            passed = passes_gate(sum(self.accepted.values()), self.full_success,
                                 len(self.expected_uids), self.tolerance_pp)
            if passed:
                self.selected = self.levels[self.index]
                self.index += 1
            self.stopped = not passed or self.index == len(self.levels)
            self.accepted = {}
        return True
