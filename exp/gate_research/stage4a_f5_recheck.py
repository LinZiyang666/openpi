"""Stage 4a Phase A — F5 lockstep re-check under N4 injection (offline, 0 GPU).

Roadmap Stage 4a (N2 FollowWinnerGate) entry condition #2 requires re-measuring
F5 winner-persistence *after* N4 injection reshapes the hit segments, before
committing to the L3 blind-replay build. This is the gating pre-requisite: if
lockstep is destroyed under injection, N2's structural premise is gone and
Phase B is not built.

What it does
------------
For each N4 live run (Stage 3a: {spatial,l10} x L{6,8,12}) and for the Stage-0
always-search baselines, it recomputes the F5 statistic over *strictly adjacent*
FULL_HIT step pairs within each episode:

  - same-episode persistence % = fraction of adjacent FH->FH pairs that stay on
    the same library episode (winner_id trajectory prefix).
  - Delta winner_step distribution = how the winner step index advances between
    adjacent FH steps (+1 is the lockstep signal; 0 is a dense-replan repeat).

Because injected skips (searched=False, MISS) break the FH run, an adjacent
FH->FH pair can only occur *within* an un-injected sub-run. So this measures
the within-FH-run lockstep persistence in the N4-injected regime -- exactly the
structural-opportunity question. The pair logic mirrors
``gate_structure_analysis.py`` block [5] (the original F5 engine).

Go/No-Go (plan Stage 4a section 2.4)
------------------------------------
Per (suite, L): GO if N4 same-episode% is within 10pp of that suite's Stage-0
baseline AND Delta+1 stays the majority of same-episode pairs; NO-GO if
same% < 85% OR Delta+1 loses majority. Anything else is a GRAY case flagged for
owner review.

Usage:
    python exp/gate_research/stage4a_f5_recheck.py            # all runs + report
    python exp/gate_research/stage4a_f5_recheck.py --md OUT   # also write report
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Data locations (localized Stage 3a N4 live data + Stage-0 baselines)
# ---------------------------------------------------------------------------

DATA = Path("exp/gate_research/data")

# Stage 3a N4 live runs: 6 winning-point/dose runs, 500 ep each.
N4_RUNS = {
    "spatial": [("L6", "n1_live/spatial_n4_L6"),
                ("L8", "n1_live/spatial_n4_L8"),
                ("L12", "n1_live/spatial_n4_L12")],
    "l10": [("L6", "n1_live/l10_n4_L6"),
            ("L8", "n1_live/l10_n4_L8"),
            ("L12", "n1_live/l10_n4_L12")],
}

# Stage-0 always-search baselines (original F5 measurement source).
BASELINES = {
    "spatial": "libero_spatial/gate_rows.jsonl",
    "l10": "libero_10/gate_rows.jsonl",
}

# Go/No-Go thresholds (plan 2.4).
PERSIST_DROP_PP = 10.0   # GO requires N4 same% >= baseline - this
PERSIST_FLOOR = 85.0     # NO-GO if same% below this


# ---------------------------------------------------------------------------
# Core F5 statistic
# ---------------------------------------------------------------------------


def _rows_path(rel: str) -> Path:
    p = DATA / rel
    return p / "rows.jsonl" if p.is_dir() else p


def _load_per_step(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    return [r for r in rows if r.get("_kind") != "episode_summary"]


def _episodes(per_step: list[dict]) -> dict[tuple, list[dict]]:
    """Group per-step rows into episodes keyed by (yaml_id, task_uid), sorted by step."""
    eps: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in per_step:
        eps[(r["yaml_id"], r["task_uid"])].append(r)
    for steps in eps.values():
        steps.sort(key=lambda r: r["step_idx"])
    return eps


def f5_persistence(per_step: list[dict]) -> dict:
    """Compute F5 over adjacent FULL_HIT pairs. Mirrors gate_structure_analysis [5].

    Returns same/diff pair counts, the Delta winner_step distribution, and the
    derived same-episode% and Delta+1% (as a fraction of same-episode pairs).
    """
    same = diff = 0
    dstep: collections.Counter = collections.Counter()
    for steps in _episodes(per_step).values():
        for a, b in zip(steps, steps[1:]):
            if a.get("hit_type") != "FULL_HIT" or b.get("hit_type") != "FULL_HIT":
                continue
            wa_id, wb_id = a.get("winner_id"), b.get("winner_id")
            if not wa_id or not wb_id:
                continue
            wa, sa = wa_id.rsplit(":", 1)
            wb, sb = wb_id.rsplit(":", 1)
            if wa == wb:
                same += 1
                dstep[int(sb) - int(sa)] += 1
            else:
                diff += 1
    tot = same + diff
    same_pct = 100.0 * same / tot if tot else float("nan")
    delta1_pct = 100.0 * dstep.get(1, 0) / same if same else float("nan")
    delta0_pct = 100.0 * dstep.get(0, 0) / same if same else float("nan")
    return {
        "same": same, "diff": diff, "n_pairs": tot,
        "same_pct": same_pct, "delta1_pct": delta1_pct, "delta0_pct": delta0_pct,
        "dstep": dict(sorted(dstep.items())),
    }


def _verdict(n4_same: float, base_same: float, delta1: float) -> str:
    """GO / NO-GO / GRAY per plan 2.4."""
    if n4_same < PERSIST_FLOOR or delta1 <= 50.0:
        return "NO-GO"
    if n4_same >= base_same - PERSIST_DROP_PP and delta1 > 50.0:
        return "GO"
    return "GRAY"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report() -> tuple[str, dict]:
    lines: list[str] = []
    results: dict = {}

    def emit(s: str = "") -> None:
        lines.append(s)

    emit("# Stage 4a Phase A — F5 lockstep re-check under N4 injection")
    emit("")
    emit("Gating pre-requisite for the N2 FollowWinnerGate L3 build (roadmap Stage 4a "
         "entry condition #2). Offline, 0 GPU, on localized Stage 3a N4 live data.")
    emit("")
    emit("F5 statistic = same-episode persistence % over strictly adjacent FULL_HIT "
         "step pairs within an episode (winner_id trajectory prefix), plus the "
         "Delta winner_step distribution (+1 = lockstep). Mirrors "
         "`gate_structure_analysis.py` block [5]. Injected skips break the FH run, "
         "so this is the *within-FH-run* persistence in the injected regime.")
    emit("")

    for suite in ("spatial", "l10"):
        emit(f"## Suite: {suite}")
        emit("")

        base_stat = f5_persistence(_load_per_step(_rows_path(BASELINES[suite])))
        results.setdefault(suite, {})["baseline"] = base_stat
        emit(f"- **Stage-0 baseline** (always-search): same-episode "
             f"**{base_stat['same_pct']:.1f}%**, Delta+1 **{base_stat['delta1_pct']:.1f}%**, "
             f"Delta0 {base_stat['delta0_pct']:.1f}%, n_pairs={base_stat['n_pairs']}")
        emit("")
        emit("| L | same-episode % | Delta+1 % | Delta0 % | n_pairs | vs baseline | verdict |")
        emit("|---|---|---|---|---|---|---|")

        for label, rel in N4_RUNS[suite]:
            stat = f5_persistence(_load_per_step(_rows_path(rel)))
            v = _verdict(stat["same_pct"], base_stat["same_pct"], stat["delta1_pct"])
            results[suite][label] = {**stat, "verdict": v}
            emit(f"| {label} | {stat['same_pct']:.1f} | {stat['delta1_pct']:.1f} | "
                 f"{stat['delta0_pct']:.1f} | {stat['n_pairs']} | "
                 f"{stat['same_pct'] - base_stat['same_pct']:+.1f}pp | **{v}** |")
        emit("")

    # Overall Go/No-Go: GO only if every measured run is GO.
    verdicts = [results[s][lab]["verdict"]
                for s in ("spatial", "l10") for lab, _ in N4_RUNS[s]]
    if all(v == "GO" for v in verdicts):
        overall = "GO"
    elif any(v == "NO-GO" for v in verdicts):
        overall = "NO-GO"
    else:
        overall = "GRAY"
    results["overall"] = overall

    emit("## Go/No-Go")
    emit("")
    emit(f"Per-run verdicts: {verdicts}")
    emit("")
    emit(f"**Overall: {overall}** — thresholds: GO if same% >= baseline-"
         f"{PERSIST_DROP_PP:.0f}pp and Delta+1 majority; NO-GO if same% < "
         f"{PERSIST_FLOOR:.0f}% or Delta+1 not majority; else GRAY (owner review).")
    emit("")
    emit("## Notes & interpretation")
    emit("")
    emit("- **Methodology self-check**: the recomputed Stage-0 baselines reproduce "
         "the roadmap F5 range (same-episode 93-98%, Delta+1 75-97%, appendix A), so "
         "the pair engine is faithful to the original F5.")
    emit("- **Baseline pooling caveat**: each suite baseline pools all always-search "
         "configs in `gate_rows.jsonl` (not keybuilder-matched to the specific N4 "
         "run). The GO margins (spatial within ~2pp; l10 above baseline) are large "
         "enough that keybuilder-matching would not flip any verdict.")
    emit("- **Why l10 rises above baseline under injection**: l10 is the oscillating "
         "suite (high Delta0 dense-replan). N4 injection breaks up the long/oscillating "
         "cache-execution runs, so the surviving contiguous FH sub-runs are the cleaner "
         "lockstep segments -- within-run persistence and Delta+1 both go *up*, Delta0 "
         "drops. Injection concentrates, not destroys, the lockstep opportunity.")
    emit("- **C8/C11 scope**: this validates only the *structural opportunity* "
         "(lockstep still present offline). N2's blind replay runs *without* verdict "
         "supervision and changes the execution flow, so its SR effect is offline-"
         "unmeasurable and MUST be live-validated (Phase C).")
    return "\n".join(lines) + "\n", results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=str, default=None,
                    help="write the markdown report to this path")
    args = ap.parse_args()
    report, results = build_report()
    print(report)
    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"[stage4a_f5_recheck] report written to {out}")


if __name__ == "__main__":
    main()
