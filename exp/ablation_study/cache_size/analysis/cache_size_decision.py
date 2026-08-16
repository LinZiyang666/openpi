"""The pre-registered decision tree for the cache-size ablation (plan §8.4).

Four axes, each with a fixed role:

*   ``M`` monotonicity violation -- confirmatory, from tests 1-5. Highest
    priority: if the curve itself is suspect, "plateau" and "gap" lose their
    footing, so the verdict short-circuits to branch N.
*   ``D`` direction relative to 0 -- confirmatory, from test 6.
*   ``Q`` adequacy relative to delta -- confirmatory, from tests 7/8. **This is
    the main decision axis**: the experiment asks whether pure replay is good
    enough, and non-inferiority is what "good enough" means.
*   ``P`` plateau -- **descriptive only**. It never enters the family, and any
    sentence resting on it must be labelled as not multiplicity-controlled.

D and Q are orthogonal: non-inferiority is ``U < +delta``, which says nothing
about whether the interval straddles 0. A gap CI of ``[+1pp, +4pp]`` therefore
carries *two* simultaneous facts -- the teacher is ahead, and the cache is still
within the practical margin -- and both must be reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DAxis = Literal["D-teacher", "D-cache", "D-none"]
QAxis = Literal["Q-pass", "Q-fail", "Q-inconc"]
PAxis = Literal["P-yes", "P-no", "P-inconc"]

# Frozen before any data is seen (plan §8.4.4).
DELTA = 0.05          # practical margin: smallest effect prior work treated as real
PLATEAU_EPS = 0.02    # config-noise scale from the historical AlwaysHit spread


class UnreachableCombination(RuntimeError):
    """Raised when D and Q land on a pair that no dataset can produce.

    Only the two ``D-cache`` pairs qualify: both require the point estimate to be
    simultaneously below 0 and above delta. Searching 5,876 negative-mean gap
    vectors never produced a test-8 p below 0.64, so these really are
    unreachable, and a firing here means the implementation broke.

    ``D-none x Q-fail`` was originally listed here too. It is **not**
    unreachable -- see ``_D_NONE_Q_FAIL_NOTE``.
    """


@dataclass(frozen=True)
class Verdict:
    branch: str
    d: DAxis
    q: QAxis
    p: PAxis
    m_yes: bool
    headline: str
    caveats: tuple[str, ...]


def classify_d(*, test6_rejected: bool, gap_point: float) -> DAxis:
    if not test6_rejected:
        return "D-none"
    return "D-teacher" if gap_point > 0 else "D-cache"


def classify_q(*, test7_rejected: bool, test8_rejected: bool) -> QAxis:
    if test7_rejected and test8_rejected:
        raise UnreachableCombination(
            "tests 7 and 8 both rejected: H0 'gap >= delta' and H0 'gap <= delta' "
            "overlap only at gap == delta, so both cannot be false"
        )
    if test7_rejected:
        return "Q-pass"
    if test8_rejected:
        return "Q-fail"
    return "Q-inconc"


def classify_p(
    *,
    slope5_ci_hi: float,
    slope6_ci_lo: float,
    slope6_ci_hi: float,
    degenerate: bool = False,
) -> PAxis:
    """Plateau axis. ``degenerate`` forces inconclusive.

    A fully tied slope gives ``CI = [0, 0]``, whose upper bound is trivially
    below the plateau threshold -- so a zero-information sample would otherwise
    produce ``P-yes`` and, with ``Q-fail``, the headline branch A. That directly
    contradicts how the same sample is treated on the testing side, where zero
    between-cluster spread is refused as not evaluable.
    """
    if degenerate:
        return "P-inconc"
    if slope5_ci_hi < PLATEAU_EPS and slope6_ci_hi < PLATEAU_EPS:
        return "P-yes"
    if slope6_ci_lo > PLATEAU_EPS:
        return "P-no"
    return "P-inconc"


def classify_m(adjacent_rejected: dict[str, bool], adjacent_points: dict[str, float]) -> bool:
    """M-yes iff some adjacent comparison is a Holm-confirmed *decrease*.

    A missing point estimate raises rather than defaulting to 0.0: silently
    reading absence as "not a decrease" would turn a wiring bug into a clean
    M-no, which is the branch that lets the rest of the tree be interpreted.
    """
    missing = sorted(set(adjacent_rejected) - set(adjacent_points))
    if missing:
        raise KeyError(f"no point estimate for {missing}; cannot classify monotonicity")
    return any(
        adjacent_rejected[name] and adjacent_points[name] < 0
        for name in adjacent_rejected
    )


# Confirming "gap > delta" does not imply confirming "gap != 0" at the family
# level, because the two use different tails: test 6 is two-sided and test 8 is
# one-sided. When the bootstrap null is skewed -- one task with an extreme
# difference is enough -- the two-sided count picks up left-tail mass that the
# one-sided count never sees, so p6 can exceed p8 by more than the Holm slot
# ratio absorbs. A worked case with the production code: gap mean +0.312 gives
# raw p6=0.026 / p8=0.003, Holm-adjusted 0.052 (not rejected) / 0.026 (rejected).
# That is a legitimate dataset, so it gets a reading rather than an abort.
_D_NONE_Q_FAIL_NOTE = (
    "magnitude confirmed above delta while direction was not confirmed at family "
    "level -- the two use different tails (test 8 one-sided, test 6 two-sided), so "
    "this is a real dataset shape, not an inconsistency. Report both facts and do "
    "not paraphrase the pair as agreement"
)


def _check_reachable(d: DAxis, q: QAxis) -> None:
    if d == "D-cache" and q == "Q-fail":
        raise UnreachableCombination(
            "D-cache asserts gap < 0 while Q-fail asserts gap > delta > 0"
        )
    if d == "D-cache" and q == "Q-inconc":
        raise UnreachableCombination(
            "confirming gap < 0 implies confirming gap < delta, so Q cannot be inconclusive"
        )


_SCOPE = (
    "scope: single teacher (Pi0.5), single keybuilder (cp1_spatial_pool_16), "
    "two LIBERO suites, sim-only"
)
_INDEX_LIMIT = (
    "holds for THIS index only -- a flat curve is equally consistent with "
    "'the index cannot exploit a denser library' (plan §3.3.1)"
)
_P_DESCRIPTIVE = (
    "the plateau half of this statement is descriptive, not multiplicity-controlled"
)

_CELLS: dict[tuple[PAxis, QAxis], tuple[str, str]] = {
    ("P-yes", "Q-fail"): ("A", "curve has flattened and the cache stays more than delta behind the teacher"),
    ("P-yes", "Q-pass"): ("C", "curve has flattened and the cache is already good enough (non-inferior)"),
    ("P-yes", "Q-inconc"): ("D", "curve has flattened; adequacy is undetermined"),
    ("P-no", "Q-fail"): ("E", "still climbing and more than delta behind -- data volume dominates within this benchmark's init budget"),
    ("P-no", "Q-pass"): ("F", "already good enough yet still climbing"),
    ("P-no", "Q-inconc"): ("E'", "still climbing; adequacy is undetermined"),
    ("P-inconc", "Q-fail"): ("G", "more than delta behind; whether the curve has plateaued is undetermined"),
    ("P-inconc", "Q-pass"): ("I", "good enough; whether the curve has plateaued is undetermined"),
    ("P-inconc", "Q-inconc"): ("H", "inconclusive on both axes at this budget"),
}

_D_PHRASE: dict[DAxis, str] = {
    "D-teacher": "the teacher remains statistically ahead",
    "D-cache": "the cache is statistically ahead of the teacher",
    "D-none": "the direction cannot be asserted",
}


def decide(
    *,
    d: DAxis,
    q: QAxis,
    p: PAxis,
    m_yes: bool,
    equivalence_note: bool = False,
) -> Verdict:
    """Resolve the tree. M outranks everything; otherwise (P, Q) selects the cell."""
    _check_reachable(d, q)

    caveats: list[str] = [_SCOPE]

    if m_yes:
        return Verdict(
            branch="N",
            d=d, q=q, p=p, m_yes=True,
            headline=(
                "non-monotonic curve: at least one tier is a confirmed decrease. "
                "Adjudicate the normalization-mismatch explanation via the sensitivity "
                "arms before reading plateau or gap"
            ),
            caveats=tuple(caveats + [
                "P and Q are not interpreted substantively while the curve is suspect"
            ]),
        )

    branch, cell_text = _CELLS[(p, q)]
    headline = f"{cell_text}; {_D_PHRASE[d]}"

    if branch == "A":
        caveats += [_INDEX_LIMIT, _P_DESCRIPTIVE]
    if p in ("P-yes", "P-no"):
        caveats.append(_P_DESCRIPTIVE)
    if q == "Q-inconc":
        caveats.append(
            "underpowered, NOT evidence of equivalence -- must not be read as "
            "'no difference' or 'good enough'"
        )
    if q == "Q-pass" and equivalence_note:
        caveats.append(
            "two-sided equivalence also holds (descriptive; not multiplicity-controlled)"
        )
    if d == "D-none" and q == "Q-fail":
        caveats.append(_D_NONE_Q_FAIL_NOTE)
    if d == "D-cache":
        caveats.append(
            "pure replay beating its own teacher is counter-intuitive: clear the "
            "§8.4.3 checklist (anchor protocol / success-filter selection effect / "
            "evaluation leakage) before publishing this as a finding"
        )

    # Every cell must report both axes -- reporting only one is misleading.
    seen = set()
    ordered = [c for c in caveats if not (c in seen or seen.add(c))]
    return Verdict(branch=branch, d=d, q=q, p=p, m_yes=False,
                   headline=headline, caveats=tuple(ordered))
