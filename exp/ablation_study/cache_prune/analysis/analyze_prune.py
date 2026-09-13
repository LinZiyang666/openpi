"""Paired full-grid statistics for the cache_prune run, read off the journals.

Each family (suite x regime) is one conductor run over its ten arms. The
accepted outcome of every episode comes from the common conductor reader; its
same-attempt per-step rows supply the FULL_HIT witness and the client timing.
Per arm that gives a ledger of 500 paired episodes, and the ten ledgers go
through the pre-registered task-level paired bootstrap unchanged. A family with
an incomplete arm is reported per arm without intervals.

Usage:
  uv run python -m exp.ablation_study.cache_prune.analysis.analyze_prune \\
      --direct-root <dir with <lane>/<suite>_<regime>/journal.jsonl> \\
      --audit-root <dir with <suite>/eval_membership.json> \\
      --artifacts <cache_artifacts dir> --out <analysis.json>
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from exp.common.conductor_journal import load_accepted

from ..common import REGIMES, SEED, SUITES, read_json, require

WAIT, REPLAN = 10, 5


def read_jsonl(path: Path):
    """Yield the well-formed rows of a JSONL file; a torn last line is skipped."""
    with path.open() as handle:
        for line in handle:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def family_ledgers(family: Path) -> tuple[dict, dict]:
    """Per arm: one ledger row per accepted episode, plus its hit-type census.

    Per-step rows are joined on (task_uid, attempt) to the accepted attempt, so
    a stale attempt's rows can neither witness nor pollute an episode. Rows are
    read from the sink and from the driver's crash snapshot alike, deduplicated
    on (uid, attempt, step, kind).
    """
    journal, per_step = family / "journal.jsonl", family / "per_step.jsonl"
    if not journal.exists():
        return {}, {}
    accepted = load_accepted([journal])
    rows_by: dict = defaultdict(list)
    seen = set()
    for source in (per_step, family / "per_step.snapshot.jsonl"):
        if not source.exists():
            continue
        for row in read_jsonl(source):
            key = (row.get("task_uid"), row.get("attempt"), row.get("step_idx"), row.get("_kind"))
            if key in seen:
                continue
            seen.add(key)
            rows_by[row.get("task_uid"), row.get("attempt")].append(row)
    ledgers, witnesses = {}, {}
    for arm, records in accepted.items():
        ledger, hits, misses = [], 0, 0
        for uid, rec in records.items():
            rows = rows_by.get((uid, rec.attempt), [])
            timing = [r for r in rows if r.get("_kind") == "client_timing"]
            hit_rows = [r for r in rows if "hit_type" in r]
            hits += sum(r["hit_type"] == "FULL_HIT" for r in hit_rows)
            misses += sum(r["hit_type"] != "FULL_HIT" for r in hit_rows)
            entry = {
                "task_id": rec.task_id,
                "init_idx": rec.episode_idx,
                "success": bool(rec.success),
            }
            if timing:
                t = timing[0]
                infers = max(int(t.get("infers") or 0), 1)
                infer_ms = float(t.get("infer_ms") or 0.0)
                entry.update(
                    control_steps=int(t.get("steps") or 0) - WAIT,
                    infers=infers,
                    infer_ms=infer_ms,
                    infer_ms_per_call=infer_ms / infers,
                )
            ledger.append(entry)
        ledgers[arm] = ledger
        witnesses[arm] = {"full_hit": hits, "non_full_hit": misses}
    return ledgers, witnesses


def paired_statistics(
    ledgers: list[list[dict]],
    membership: list[dict],
    *,
    draws: int = 10000,
    seed: int = SEED,
) -> dict:
    """Compute task-equal SR and paired two-level bootstrap with common draws."""
    require(len(ledgers) == 10, "statistics require all ten points including P00")
    maps = [{(r["task_id"], r["init_idx"]): r for r in ledger} for ledger in ledgers]
    expected = {(t, i) for t in range(10) for i in range(50)}
    require(
        all(
            len(ledger) == 500 and set(m) == expected
            for ledger, m in zip(ledgers, maps)
        ),
        "unpaired or incomplete ledger",
    )
    members = {(r["task_id"], r["orig_init_state_idx"]): r for r in membership}
    require(set(members) == expected, "statistics membership incomplete")
    results = {}
    for subset in ("common_unseen", "full", "seen"):
        pools = {
            t: [
                i
                for i in range(50)
                if subset == "full"
                or members[t, i]["common_unseen"] is (subset == "common_unseen")
            ]
            for t in range(10)
        }
        if not all(pools.values()):
            require(subset == "seen", "primary subset has an empty task")
            results[subset] = {
                "estimable": False,
                "reason": "at least one task has no seen init",
                "per_task_n": {str(t): len(v) for t, v in pools.items()},
            }
            continue
        per_task = np.array(
            [
                [np.mean([m[t, i]["success"] for i in pools[t]]) for m in maps]
                for t in range(10)
            ]
        )
        rng = np.random.default_rng(seed)
        sampled_tasks = rng.integers(0, 10, size=(draws, 10))
        boot = np.zeros((draws, 10, 10), dtype=np.float64)
        for task, indices in pools.items():
            where = np.where(sampled_tasks == task)
            states = np.array(
                [[m[task, i]["success"] for m in maps] for i in indices], dtype=float
            )
            chosen = rng.integers(0, len(indices), size=(len(where[0]), len(indices)))
            boot[where] = states[chosen].mean(axis=1)
        boot = boot.mean(axis=1)
        bootstrap_delta = boot - boot[:, :1]
        points = []
        for point, current in enumerate(maps):
            keys = [(t, i) for t in range(10) for i in pools[t]]
            common_success = [
                k for k in keys if maps[0][k]["success"] and current[k]["success"]
            ]
            differences = [
                current[k]["control_steps"] - maps[0][k]["control_steps"]
                for k in common_success
            ]
            infer_ms = sum(current[k]["infer_ms"] for k in keys)
            infers = sum(current[k]["infers"] for k in keys)
            points.append(
                {
                    "point": f"P{point:02d}",
                    "n": len(keys),
                    "successes": sum(current[k]["success"] for k in keys),
                    "sr": float(per_task[:, point].mean()),
                    "sr_ci95": np.quantile(boot[:, point], [0.025, 0.975]).tolist(),
                    "delta_sr": float((per_task[:, point] - per_task[:, 0]).mean()),
                    "delta_sr_ci95": np.quantile(
                        bootstrap_delta[:, point], [0.025, 0.975]
                    ).tolist(),
                    "per_task_sr": per_task[:, point].tolist(),
                    "common_success_n": len(common_success),
                    "mean_control_steps_delta": float(np.mean(differences))
                    if differences
                    else None,
                    "lost_successes": sum(
                        maps[0][k]["success"] and not current[k]["success"]
                        for k in keys
                    ),
                    "new_successes": sum(
                        not maps[0][k]["success"] and current[k]["success"]
                        for k in keys
                    ),
                    "sum_infer_ms": infer_ms,
                    "sum_infers": infers,
                    "online_ms_per_call": infer_ms / infers,
                    "episode_mean_ms_per_call": float(
                        np.mean([current[k]["infer_ms_per_call"] for k in keys])
                    ),
                }
            )
        results[subset] = {
            "estimable": True,
            "per_task_n": {str(t): len(v) for t, v in pools.items()},
            "points": points,
        }
    return {
        "subsets": results,
        "bootstrap_draws": draws,
        "seed": seed,
        "interval_interpretation": "descriptive two-level paired bootstrap; no selection correction",
    }


def library_record(artifacts: Path, source: dict | None, arm: str, point: str) -> dict:
    """Entry count, bytes and the grid's actual deletion rate for one arm."""
    record = {"point": point, "entries": None, "bytes": None, "actual_rate": None}
    manifest = artifacts / arm / "artifact.json"
    if manifest.exists():
        data = read_json(manifest)
        record.update(entries=data.get("entries"), bytes=data["file"]["bytes"])
    elif point == "P00" and source is not None:
        record.update(entries=source["entries"], bytes=source["file"]["bytes"])
    return record


def analyze_families(direct_root: Path, memberships: dict, artifacts: Path, grids: dict) -> dict:
    """Every family found under the run root, complete ones with paired statistics."""
    curves = []
    for lane_dir in sorted(p for p in direct_root.iterdir() if p.is_dir()):
        for suite in SUITES:
            for regime in REGIMES:
                family = lane_dir / f"{suite}_{regime}"
                if not family.is_dir():
                    continue
                ledgers, witnesses = family_ledgers(family)
                arms = [f"cache_prune_{suite}_{regime}_P{i:02d}" for i in range(10)]
                rates = {}
                source = None
                if (suite, regime) in grids:
                    grid, source = grids[suite, regime]
                    rates = {p["point"]: p["actual_rate"] for p in grid["points"]}
                members = {
                    (r["task_id"], r["orig_init_state_idx"]): r for r in memberships[suite]
                }
                per_arm, libraries = [], []
                for arm in arms:
                    point = arm[-3:]
                    ledger = ledgers.get(arm, [])
                    w = witnesses.get(arm, {"full_hit": 0, "non_full_hit": 0})
                    common = [
                        r for r in ledger
                        if members.get((r["task_id"], r["init_idx"]), {}).get("common_unseen")
                    ]
                    timed = [r for r in ledger if "infer_ms" in r]
                    solved = [r for r in timed if r["success"]]
                    per_arm.append(
                        {
                            "arm": arm,
                            "point": point,
                            "n": len(ledger),
                            "sr": (sum(r["success"] for r in ledger) / len(ledger)) if ledger else None,
                            "sr_common_unseen": (
                                sum(r["success"] for r in common) / len(common)
                            ) if common else None,
                            "n_common_unseen": len(common),
                            "full_hit": w["full_hit"],
                            "non_full_hit": w["non_full_hit"],
                            "ms_per_call": (
                                sum(r["infer_ms"] for r in timed) / sum(r["infers"] for r in timed)
                            ) if timed else None,
                            "mean_control_steps": (
                                sum(r["control_steps"] for r in timed) / len(timed)
                            ) if timed else None,
                            # Length of the episodes this arm actually solved: the
                            # all-episode mean is pulled up by failures sitting at
                            # the step cap, so it says more about the failure rate
                            # than about how long a solution takes.
                            "n_success": len(solved),
                            "mean_control_steps_success": (
                                sum(r["control_steps"] for r in solved) / len(solved)
                            ) if solved else None,
                        }
                    )
                    record = library_record(artifacts, source, arm, point)
                    record["actual_rate"] = rates.get(point)
                    libraries.append(record)
                complete = all(
                    len(ledgers.get(a, [])) == 500
                    and all("infer_ms" in r for r in ledgers[a])
                    for a in arms
                )
                curve = {
                    "lane": lane_dir.name,
                    "suite": suite,
                    "regime": regime,
                    "complete": complete,
                    "arms": per_arm,
                    "libraries": libraries,
                }
                if complete:
                    curve.update(paired_statistics([ledgers[a] for a in arms], memberships[suite]))
                curves.append(curve)
    return {"kind": "cache_prune_analysis", "curves": curves}


def print_report(analysis: dict) -> None:
    for curve in analysis["curves"]:
        print(f"== {curve['lane']} {curve['suite']}/{curve['regime']} complete={curve['complete']}")
        for row, lib in zip(curve["arms"], curve["libraries"]):
            sr = "  -  " if row["sr"] is None else f"{row['sr']:.3f}"
            cu = "  -  " if row["sr_common_unseen"] is None else f"{row['sr_common_unseen']:.3f}"
            ms = "  -  " if row["ms_per_call"] is None else f"{row['ms_per_call']:.0f}"
            steps = row["full_hit"] + row["non_full_hit"]
            fh = f"{row['full_hit'] / steps:.3f}" if steps else "  -  "
            print(
                f"  {row['point']} entries={lib['entries']!s:>6} n={row['n']:4d} sr={sr} "
                f"sr_cu={cu} full_hit={fh} ms/call={ms}"
            )
        if curve["complete"]:
            for point in curve["subsets"]["common_unseen"]["points"]:
                lo, hi = point["delta_sr_ci95"]
                print(
                    f"     {point['point']} common_unseen sr={point['sr']:.3f} "
                    f"dSR={point['delta_sr']:+.3f} [{lo:+.3f},{hi:+.3f}] "
                    f"lost={point['lost_successes']} new={point['new_successes']}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    memberships = {
        s: read_json(args.audit_root / s / "eval_membership.json")["rows"] for s in SUITES
    }
    grids = {}
    for suite in SUITES:
        for regime in REGIMES:
            grid_path = args.audit_root / suite / regime / "grid_freeze.json"
            source_path = args.audit_root / suite / regime / "source.json"
            if grid_path.exists() and source_path.exists():
                grids[suite, regime] = (read_json(grid_path), read_json(source_path))
    analysis = analyze_families(args.direct_root, memberships, args.artifacts, grids)
    print_report(analysis)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(analysis, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
