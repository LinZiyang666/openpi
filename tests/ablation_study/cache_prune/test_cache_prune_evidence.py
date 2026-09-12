"""Adversarial accepted-attempt, timing, memory and paired-statistics checks."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from exp.ablation_study.cache_prune.analysis.analyze_prune import (
    control_length,
    episode_ledger,
    paired_statistics,
)
from exp.ablation_study.cache_prune.prepare_membership import state_ids
from exp.ablation_study.cache_prune.run_prune_eval import (
    GIB,
    host_budget,
    validate_load_log,
)


@pytest.mark.parametrize(
    "steps,infers,success,expected",
    [(23, 3, True, 13), (25, 3, True, 15), (11, 1, True, 1), (230, 44, False, 220)],
)
def test_control_length_counts_wait_and_partial_action_blocks(
    steps, infers, success, expected
):
    """Check control length counts wait and partial action blocks."""
    assert (
        control_length(
            {"steps": steps, "infers": infers},
            success,
            wait=10,
            replan=5,
            max_steps=220,
        )
        == expected
    )


@pytest.mark.parametrize(
    "timing,success",
    [
        ({"steps": 9, "infers": 0}, False),
        ({"steps": 10, "infers": 1}, False),
        ({"steps": 23, "infers": 3}, False),
        ({"steps": 23, "infers": 2}, True),
        ({"steps": 23.0, "infers": 3}, True),
        ({"steps": 231, "infers": 45}, True),
    ],
)
def test_control_length_rejects_early_exit_and_inconsistent_counts(timing, success):
    """Check control length rejects early exit and inconsistent counts."""
    with pytest.raises(ValueError):
        control_length(timing, success, wait=10, replan=5, max_steps=220)


@pytest.fixture
def episode_files(tmp_path):
    """Create the real journal/step shapes for 500 accepted successful episodes."""
    name = "cache_prune_libero_spatial_rit50_P00"
    source_rows = [
        {
            "id": f"winner{t}",
            "task_key": f"task {t}",
            "trajectory_id": f"traj{t}",
            "remaining": 0,
        }
        for t in range(10)
    ]
    members, journal, per_step = [], [], []
    for task in range(10):
        for index in range(50):
            uid = f"{name}:eval:{task}:{index}"
            common = {
                "task_uid": uid,
                "yaml_id": name,
                "run_id": "producer",
                "attempt": 1,
                "accepted": True,
            }
            members.append(
                {
                    "task_id": task,
                    "orig_init_state_idx": index,
                    "state_sha256": f"state{task}_{index}",
                    "task_key": f"task {task}",
                    "common_unseen": index >= 5,
                }
            )
            journal.append(
                {
                    **common,
                    "phase": "eval",
                    "status": "done",
                    "success": True,
                    "duration_s": 1.2,
                }
            )
            per_step.extend(
                {
                    **common,
                    "step_idx": step,
                    "hit_type": "FULL_HIT",
                    "searched": True,
                    "winner_id": f"winner{task}",
                }
                for step in (0, 5, 10)
            )
            per_step.append(
                {
                    **common,
                    "_kind": "client_timing",
                    "steps": 23,
                    "infers": 3,
                    "infer_ms": 60.0,
                }
            )
    arm = {
        "arm": name,
        "suite": "libero_spatial",
        "source_digest": "source",
        "artifact": {"selection": {"kept_ids": [r["id"] for r in source_rows]}},
    }
    freeze = {
        "digest": "freeze",
        "sources": [{"digest": "source", "rows": source_rows}],
        "protocol": {
            "num_steps_wait": 10,
            "replan_steps": 5,
            "max_steps": {"libero_spatial": 220},
        },
        "memberships": {"libero_spatial": {"rows": members}},
    }
    launch = {
        "freeze_digest": "freeze",
        "arm": name,
        "num_steps_wait": 10,
        "replan_steps": 5,
        "max_steps": 220,
        "trials": 50,
        "smoke": False,
    }
    return freeze, arm, launch, journal, per_step, tmp_path


def _write_ledger(fixture):
    freeze, arm, launch, journal, rows, directory = fixture
    j, p = directory / "journal.jsonl", directory / "per_step.jsonl"
    j.write_text("".join(json.dumps(row) + "\n" for row in journal))
    p.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return episode_ledger(freeze, arm, j, p, launch)


def test_complete_ledger_uses_persisted_timing_not_journal_nsteps(episode_files):
    """Check complete ledger uses persisted timing not journal nsteps."""
    result = _write_ledger(episode_files)
    assert len(result) == 500
    assert all(
        r["control_steps"] == 13 and r["infer_ms_per_call"] == 20 for r in result
    )
    assert sum(r["common_unseen"] for r in result) == 450


@pytest.mark.parametrize(
    "problem",
    [
        "missing_first",
        "missing_middle",
        "missing_tail",
        "duplicate_step",
        "duplicate_timing",
        "missing_timing",
        "stale_only",
        "wrong_producer",
        "unknown_winner",
        "wrong_task",
        "no_search",
        "missing_search",
        "missing_task",
        "infra_error",
        "short_failure",
        "missing_attempt",
        "miss",
        "duplicate_terminal",
        "two_producers",
    ],
)
def test_invalid_evidence_never_produces_formal_ledger(episode_files, problem):
    """Check invalid evidence never produces formal ledger."""
    _, _, _, journal, rows, _ = episode_files
    if problem in ("missing_first", "missing_middle", "missing_tail"):
        rows.pop({"missing_first": 0, "missing_middle": 1, "missing_tail": 2}[problem])
    elif problem == "duplicate_step":
        rows.append(copy.deepcopy(rows[0]))
    elif problem == "duplicate_timing":
        rows.append(copy.deepcopy(rows[3]))
    elif problem == "missing_timing":
        rows.pop(3)
    elif problem == "stale_only":
        journal[0]["attempt"] = 2
    elif problem == "wrong_producer":
        rows[0]["run_id"] = "old_producer"
    elif problem == "unknown_winner":
        rows[0]["winner_id"] = "deleted"
    elif problem == "wrong_task":
        rows[0]["winner_id"] = "winner1"
    elif problem == "no_search":
        rows[0]["searched"] = False
    elif problem == "missing_search":
        del rows[0]["searched"]
    elif problem == "missing_task":
        journal[:] = journal[50:]
    elif problem == "infra_error":
        journal[0]["error"] = "worker crashed"
    elif problem == "short_failure":
        journal[0].update(status="failed", success=False)
    elif problem == "missing_attempt":
        del journal[0]["attempt"]
    elif problem == "miss":
        rows[0]["hit_type"] = "MISS"
    elif problem == "duplicate_terminal":
        journal.append(copy.deepcopy(journal[0]))
    else:
        journal[0]["run_id"] = "old_producer"
    with pytest.raises((ValueError, SystemExit)):
        _write_ledger(episode_files)


def test_stale_attempt_cannot_override_a_complete_accepted_attempt(episode_files):
    """Check stale attempt cannot override a complete accepted attempt."""
    _, _, _, journal, rows, _ = episode_files
    old = {
        **journal[0],
        "accepted": False,
        "attempt": 0,
        "success": False,
        "status": "failed",
    }
    journal.append(old)
    rows.append({**rows[0], "accepted": False, "attempt": 0, "hit_type": "MISS"})
    result = _write_ledger(episode_files)
    assert all(r["success"] for r in result)


def test_shared_bootstrap_and_common_success_length(episode_files):
    """Check shared bootstrap and common success length."""
    base = _write_ledger(episode_files)
    ledgers = [copy.deepcopy(base) for _ in range(10)]
    for point, ledger in enumerate(ledgers):
        for row in ledger:
            row["success"] = row["init_idx"] >= point
            row["control_steps"] = 13 + point
    result = paired_statistics(
        ledgers, episode_files[0]["memberships"]["libero_spatial"]["rows"], draws=200
    )
    points = result["subsets"]["full"]["points"]
    assert points[0]["delta_sr_ci95"] == [0.0, 0.0]
    assert points[5]["sr"] == pytest.approx(0.9)
    assert points[5]["delta_sr"] == pytest.approx(-0.1)
    assert (
        points[5]["common_success_n"] == 450
        and points[5]["mean_control_steps_delta"] == 5
    )
    assert result["subsets"]["common_unseen"]["points"][5]["sr"] == 1.0
    for row in ledgers[9]:
        row["success"] = False
    result = paired_statistics(
        ledgers, episode_files[0]["memberships"]["libero_spatial"]["rows"], draws=200
    )
    assert result["subsets"]["full"]["points"][9]["mean_control_steps_delta"] is None
    assert result["subsets"]["full"]["points"][9]["common_success_n"] == 0


def test_node_budget_and_actual_load_events():
    """Check node budget and actual load events."""
    assert host_budget(128 * GIB, 110 * GIB) == 94 * GIB
    assert host_budget(128 * GIB, 10 * GIB) == 0
    identity = {"path": "/tmp/library.pkl"}
    good = "load_cache_config bundle=a\nLoaded 42 entries from /tmp/library.pkl\nload_cache_config bundle=a\n"
    validate_load_log(good, identity, 42)
    for wrong in (
        good + "Loaded 42 entries from /tmp/library.pkl\n",
        good.replace("42", "41"),
        good.replace("library.pkl", "other.pkl"),
        "load_cache_config bundle=a",
    ):
        with pytest.raises(ValueError):
            validate_load_log(wrong, identity, 42)


def test_init_identity_is_state_content_not_cross_pool_index(tmp_path):
    """Check init identity is state content not cross pool index."""
    import torch

    a, b = tmp_path / "a.init", tmp_path / "b.init"
    torch.save(np.array([[1, 2], [3, 4]], dtype=np.float32), a)
    torch.save(np.array([[3, 4], [1, 2]], dtype=np.float64), b)
    assert state_ids(a)[0] == state_ids(b)[1]
    assert state_ids(a)[0] != state_ids(b)[0]
