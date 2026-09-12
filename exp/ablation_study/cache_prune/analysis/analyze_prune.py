"""Validate accepted episode evidence and compute paired full-grid statistics.

The common conductor reader owns outcome selection. This module joins precisely
that accepted producer/attempt to timing and FULL_HIT traces, then applies
pre-frozen init membership and shared two-level paired bootstrap draws.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from exp.ablation_study.cache_size.full_hit import assert_full_hit_per_episode
from exp.common.conductor_journal import load_accepted

from ..common import (
    REGIMES,
    SEED,
    SUITES,
    VERSION,
    file_identity,
    read_json,
    require,
    seal,
    write_json,
)
from ..emit_prune_arms import validate_freeze


def read_jsonl(path: str | Path) -> list[dict]:
    """Read complete JSONL evidence, rejecting truncated lines rather than skipping."""
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def control_length(
    timing: dict, success: bool, *, wait: int, replan: int, max_steps: int
) -> int:
    """Derive actual control steps and distinguish timeout failure from early exit."""
    steps, infers = timing.get("steps"), timing.get("infers")
    require(
        type(steps) is int and type(infers) is int,
        "timing steps/infers must be integers",
    )
    require(steps >= wait and infers > 0, "incomplete wait or no inference")
    control = steps - wait
    require(1 <= control <= max_steps, "invalid control length")
    require(
        math.ceil(control / replan) == infers,
        "control length and inference count disagree",
    )
    require(
        success or control == max_steps,
        "unsuccessful early exit is an invalid batch, not a timeout outcome",
    )
    return control


def episode_ledger(
    freeze: dict,
    arm: dict,
    journal: str | Path,
    per_step: str | Path,
    launch: dict,
    *,
    smoke: bool = False,
) -> list[dict]:
    """Validate complete per-episode outcomes, timing and same-attempt cache witnesses."""
    require(
        launch["freeze_digest"] == freeze["digest"] and launch["arm"] == arm["arm"],
        "launch/freeze mismatch",
    )
    protocol = freeze["protocol"]
    for field in ("num_steps_wait", "replan_steps"):
        require(
            launch[field] == protocol[field], f"launch {field} differs from protocol"
        )
    require(
        launch["max_steps"] == protocol["max_steps"][arm["suite"]],
        "launch max_steps differs",
    )
    trials = 1 if smoke else 50
    require(
        launch["trials"] == trials and launch["smoke"] is smoke,
        "smoke/formal denominator mismatch",
    )
    raw = read_jsonl(journal)
    terminal = [
        r
        for r in raw
        if r.get("status") in ("done", "failed") and r.get("accepted") is not False
    ]
    for record in terminal:
        require(
            record.get("accepted") is True
            and type(record.get("attempt")) is int
            and record["attempt"] > 0,
            "unstamped journal attempt",
        )
        require(
            type(record.get("success")) is bool and bool(record.get("run_id")),
            "invalid journal outcome/producer",
        )
        require(
            record["phase"] == "eval" and record["yaml_id"] == arm["arm"],
            "foreign journal phase/arm",
        )
    producers = {r["run_id"] for r in terminal}
    require(len(producers) == 1, "a batch must have exactly one conductor producer")
    records = load_accepted([journal])
    require(set(records) == {arm["arm"]}, "journal has an unknown or missing arm")
    records = records[arm["arm"]]
    expected_uids = {
        f"{arm['arm']}:eval:{task}:{index}"
        for task in range(10)
        for index in range(trials)
    }
    require(
        set(records) == expected_uids,
        "batch is missing episodes or has extra identities",
    )
    by_uid = defaultdict(list)
    for row in read_jsonl(per_step):
        uid = row.get("task_uid")
        require(uid in expected_uids, "per-step evidence has an unknown episode")
        require(
            row.get("yaml_id") == arm["arm"] and type(row.get("attempt")) is int,
            "per-step provenance missing",
        )
        if row["attempt"] != records[uid].attempt or row.get("accepted") is False:
            continue
        require(
            row.get("accepted") is True and row.get("run_id") in producers,
            "per-step producer/acceptance mismatch",
        )
        by_uid[uid].append(row)
    source = next(s for s in freeze["sources"] if s["digest"] == arm["source_digest"])
    retained = set(arm["artifact"]["selection"]["kept_ids"])
    entry_rows = {r["id"]: r for r in source["rows"] if r["id"] in retained}
    membership = {
        (r["task_id"], r["orig_init_state_idx"]): r
        for r in freeze["memberships"][arm["suite"]]["rows"]
    }
    ledger, full_hits = [], {}
    for uid, accepted in sorted(records.items()):
        evidence = [
            r
            for r in terminal
            if r["task_uid"] == uid and r["attempt"] == accepted.attempt
        ]
        require(len(evidence) == 1, "duplicate accepted terminal record")
        record = evidence[0]
        require(not record.get("error"), "accepted episode has infrastructure error")
        require(
            record["status"] == ("done" if accepted.success else "failed"),
            "journal status/outcome mismatch",
        )
        duration = record.get("duration_s")
        require(
            isinstance(duration, (int, float))
            and math.isfinite(duration)
            and duration >= 0,
            "missing episode wall time",
        )
        rows = by_uid[uid]
        timings = [r for r in rows if r.get("_kind") == "client_timing"]
        require(len(timings) == 1, "missing or duplicate client_timing")
        timing = timings[0]
        control = control_length(
            timing,
            accepted.success,
            wait=launch["num_steps_wait"],
            replan=launch["replan_steps"],
            max_steps=launch["max_steps"],
        )
        infer_ms = timing.get("infer_ms")
        require(
            isinstance(infer_ms, (int, float))
            and math.isfinite(infer_ms)
            and infer_ms >= 0,
            "invalid inference timing",
        )
        hits = [r for r in rows if "hit_type" in r]
        require(
            len(hits) == timing["infers"], "inference count lacks full step evidence"
        )
        require(
            all(type(r.get("step_idx")) is int for r in hits),
            "missing/noninteger decision step",
        )
        hits.sort(key=lambda r: r["step_idx"])
        require(
            [r["step_idx"] for r in hits]
            == list(
                range(
                    0, timing["infers"] * launch["replan_steps"], launch["replan_steps"]
                )
            ),
            "missing or duplicate decision step",
        )
        member = membership[accepted.task_id, accepted.episode_idx]
        winner_remaining, trajectories = [], []
        for hit in hits:
            require(hit.get("searched") is True, "pure-cache call did not search")
            winner = entry_rows.get(hit.get("winner_id"))
            require(
                winner is not None and winner["task_key"] == member["task_key"],
                "unknown, removed or wrong-task winner",
            )
            winner_remaining.append(winner["remaining"])
            trajectories.append(winner["trajectory_id"])
        full_hits[uid, accepted.attempt] = [h["hit_type"] for h in hits]
        ledger.append(
            {
                "arm": arm["arm"],
                "task_uid": uid,
                "task_id": accepted.task_id,
                "init_idx": accepted.episode_idx,
                "state_sha256": member["state_sha256"],
                "common_unseen": member["common_unseen"],
                "success": accepted.success,
                "attempt": accepted.attempt,
                "producer_run_id": record["run_id"],
                "steps": timing["steps"],
                "infers": timing["infers"],
                "infer_ms": infer_ms,
                "infer_ms_per_call": infer_ms / timing["infers"],
                "control_steps": control,
                "num_steps_wait": launch["num_steps_wait"],
                "replan_steps": launch["replan_steps"],
                "duration_s": duration,
                "winner_remaining": winner_remaining,
                "trajectory_switches": sum(
                    a != b for a, b in zip(trajectories, trajectories[1:])
                ),
            }
        )
    assert_full_hit_per_episode(
        arm["arm"], {uid: r.attempt for uid, r in records.items()}, full_hits
    )
    return ledger


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


def analyze_run(
    freeze_manifest: dict,
    journals: dict,
    per_step: dict,
    membership: dict,
    *,
    launches: dict,
) -> dict:
    """Analyze all 40 arms only after each whole batch has passed evidence gates."""
    from ..run_prune_eval import accepted_batches

    validate_freeze(freeze_manifest, rehash=False)
    require(
        membership == freeze_manifest["memberships"],
        "analysis attempted to change frozen subsets",
    )
    names = {a["arm"] for a in freeze_manifest["arms"]}
    require(
        set(journals) == set(per_step) == set(launches) == names,
        "analysis requires all 40 complete batches",
    )
    run_dirs = {Path(path).resolve().parent.parent for path in journals.values()}
    require(len(run_dirs) == 1, "all arms must belong to one immutable run catalog")
    batches = accepted_batches(
        next(iter(run_dirs)), freeze_manifest, require_complete=True
    )
    for name in names:
        require(
            Path(journals[name]).resolve() == Path(batches[name]["journal"]).resolve()
            and Path(per_step[name]).resolve()
            == Path(batches[name]["per_step"]).resolve()
            and launches[name] == batches[name]["launch"],
            "analysis inputs differ from the complete accepted batch",
        )
    ledgers = {
        a["arm"]: episode_ledger(
            freeze_manifest,
            a,
            journals[a["arm"]],
            per_step[a["arm"]],
            launches[a["arm"]],
        )
        for a in freeze_manifest["arms"]
    }
    curves = []
    for suite in SUITES:
        for regime in REGIMES:
            arms = sorted(
                (
                    a
                    for a in freeze_manifest["arms"]
                    if a["suite"] == suite and a["regime"] == regime
                ),
                key=lambda a: a["point"],
            )
            statistics = paired_statistics(
                [ledgers[a["arm"]] for a in arms], membership[suite]["rows"]
            )
            curves.append(
                {
                    "suite": suite,
                    "regime": regime,
                    **statistics,
                    "libraries": [
                        {
                            "point": a["point"],
                            "entries": a["artifact"]["entries"],
                            "bytes": a["artifact"]["file"]["bytes"],
                            "actual_rate": a["actual_rate"],
                            "loaded_peak_rss_bytes": read_json(
                                Path(journals[a["arm"]]).parent / "node_postflight.json"
                            )["peak_server_rss"],
                            "rss_scope": launches[a["arm"]].get(
                                "rss_scope", "single_library_server"
                            ),
                            "latency_scope": launches[a["arm"]].get(
                                "latency_scope", "single_arm_call"
                            ),
                            "shared_server_peak_rss_bytes": read_json(
                                Path(journals[a["arm"]]).parent / "node_postflight.json"
                            ).get("shared_server_peak_rss"),
                        }
                        for a in arms
                    ],
                }
            )
    return seal(
        {
            "version": VERSION,
            "kind": "analysis",
            "freeze_digest": freeze_manifest["digest"],
            "curves": curves,
            "ledgers": ledgers,
            "evidence": {
                name: {
                    "journal": file_identity(journals[name]),
                    "per_step": file_identity(per_step[name]),
                }
                for name in sorted(names)
            },
        }
    )


def main() -> None:
    """Validate a complete batch catalog and publish paired experiment statistics."""
    from ..run_prune_eval import accepted_batches

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze = read_json(args.freeze)
    batches = accepted_batches(args.run_dir, freeze, require_complete=True)
    result = analyze_run(
        freeze,
        {k: v["journal"] for k, v in batches.items()},
        {k: v["per_step"] for k, v in batches.items()},
        freeze["memberships"],
        launches={k: v["launch"] for k, v in batches.items()},
    )
    write_json(args.out, result)


if __name__ == "__main__":
    main()
