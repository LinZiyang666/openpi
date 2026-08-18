"""Non-manual tests for verify_collection_artifacts.py + the completion rule.

Fixture trees are built with h5py in tmp_path; journals and run-plans are
plain JSON. Nothing touches a server or the simulator.
"""

from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import pytest

from exp.robocasa365.run_collect import RobocasaCollectStrategy, build_run_plan
from exp.robocasa365.run_collect import write_run_plan
from exp.robocasa365.verify_collection_artifacts import (
    audit,
    build_manifest,
    is_admissible,
    load_run_plan,
    merge_run_plans,
    min_episodes_for_target,
    prob_at_least,
)
from openpi.conductor.driver import assign_servers
from openpi.conductor.task import ServerEndpoint

TASKS = [("OpenCabinet", 2), ("CloseDrawer", 1)]


# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _run_plan(tmp_path: pathlib.Path, *, batch: int = 1, episode_lo=None, tasks=TASKS) -> dict:
    strategy = RobocasaCollectStrategy(
        teacher="pi05", layout=1, style=1, base_seed=0, replan_steps=5,
        tasks=tasks, batch=batch, episode_lo=episode_lo,
    )
    servers = [ServerEndpoint("127.0.0.1", 8010)]
    weights = {yid: n for yid, (_, n) in zip(strategy.yaml_ids, tasks)}
    graph = strategy.plan(sorted(weights), assign_servers(weights, servers, None, None))
    payload = build_run_plan(strategy, graph, str(tmp_path / "build_l1s1"))
    path = tmp_path / f"run_plan_collect_l1s1_pi05_b{batch:02d}.json"
    write_run_plan(path, payload)
    return load_run_plan(path)


def _write_h5(root: pathlib.Path, prefix: str, attempt: int, *, task: str, success: bool, steps: int = 2) -> pathlib.Path:
    path = root / f"{prefix}_a{attempt:02d}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["task"] = task
        f.attrs["success"] = success
        f.attrs["num_steps"] = steps
        for i in range(steps):
            grp = f.create_group(f"step_{i:04d}")
            for j in range(3):
                grp.create_dataset(f"vision_{j}", data=np.zeros((4, 8), dtype=np.float16))
            grp.create_dataset("prompt_emb", data=np.zeros((5, 8), dtype=np.float16))
            grp.create_dataset("robot_state", data=np.zeros(16, dtype=np.float32))
            grp.create_dataset("clean_action", data=np.zeros((50, 32), dtype=np.float32))
    return path


def _journal_row(uid: str, *, success=True, accepted=True, attempt=1, error=None) -> dict:
    return {
        "task_uid": uid, "yaml_id": uid.split(":")[0], "phase": "eval",
        "status": "done" if success else "failed",
        "success": success, "attempt": attempt, "accepted": accepted, "error": error,
    }


def _happy_tree(tmp_path: pathlib.Path):
    plan = _run_plan(tmp_path)
    root = tmp_path / "build_l1s1"
    journal = []
    for uid in plan["uids"]:
        journal.append(_journal_row(uid))
        task = plan["prefixes"][uid].split("/")[1]
        _write_h5(root, plan["prefixes"][uid], 1, task=task, success=True)
    return plan, root, journal


# ------------------------------------------------------------------
# Admission + journal-edge semantics (frozen §4.3.6-(5))
# ------------------------------------------------------------------


def test_admission_rule_literal():
    assert is_admissible(_journal_row("u"))
    assert not is_admissible(_journal_row("u", accepted=False))  # stale late result
    assert not is_admissible(_journal_row("u", success=False))
    assert not is_admissible(_journal_row("u", error="boom"))


def test_happy_path_audit_ok(tmp_path):
    plan, root, journal = _happy_tree(tmp_path)
    report = audit(root=root, journal_records=journal, plans=[plan], target=1)
    assert report["ok"], report
    assert not report["missing_terminal"] and not report["missing_file"]


def test_retry_case_a_kill_no_first_row(tmp_path):
    # Kill/disconnect: the first attempt reports nothing; only the rerun's
    # accepted row exists. Its _a02 file is admitted; a leftover _a01 file is
    # orphan_attempt (informational, not an error).
    plan, root, journal = _happy_tree(tmp_path)
    uid = plan["uids"][0]
    prefix = plan["prefixes"][uid]
    task = prefix.split("/")[1]
    journal = [r for r in journal if r["task_uid"] != uid]
    journal.append(_journal_row(uid, attempt=2))
    _write_h5(root, prefix, 2, task=task, success=True)  # rerun's file (a01 remains on disk)
    report = audit(root=root, journal_records=journal, plans=[plan], target=1)
    assert report["ok"], report
    assert report["admitted"][uid]["attempt"] == 2
    assert f"{prefix}_a01.h5" in report["orphan_attempts"]


def test_retry_case_b_stale_late_result_single_accepted(tmp_path):
    # Stale late result: multiple journal rows for one uid, exactly one accepted.
    plan, root, journal = _happy_tree(tmp_path)
    uid = plan["uids"][0]
    journal.append(_journal_row(uid, accepted=False, attempt=1))  # the stale row
    for row in journal:
        if row["task_uid"] == uid and row["accepted"]:
            row["attempt"] = 2
    prefix = plan["prefixes"][uid]
    _write_h5(root, prefix, 2, task=prefix.split("/")[1], success=True)
    report = audit(root=root, journal_records=journal, plans=[plan], target=1)
    assert report["ok"], report
    assert report["admitted"][uid]["attempt"] == 2


def test_case_c_missing_terminal_comes_from_run_plan(tmp_path):
    # Delete a uid's journal rows AND its file: only the run-plan can surface it.
    plan, root, journal = _happy_tree(tmp_path)
    uid = plan["uids"][0]
    journal = [r for r in journal if r["task_uid"] != uid]
    (root / f"{plan['prefixes'][uid]}_a01.h5").unlink()
    report = audit(root=root, journal_records=journal, plans=[plan], target=1)
    assert not report["ok"]
    assert report["missing_terminal"] == [uid]


def test_missing_file_detected_despite_clean_journal(tmp_path):
    # The silent-h5-write-failure shape: journal says done, disk says nothing.
    plan, root, journal = _happy_tree(tmp_path)
    uid = plan["uids"][1]
    (root / f"{plan['prefixes'][uid]}_a01.h5").unlink()
    report = audit(root=root, journal_records=journal, plans=[plan], target=1)
    assert not report["ok"]
    assert report["missing_file"] == [uid]


def test_schema_negatives_fail_loud(tmp_path):
    plan, root, journal = _happy_tree(tmp_path)
    uid = plan["uids"][0]
    prefix = plan["prefixes"][uid]
    path = root / f"{prefix}_a01.h5"
    with h5py.File(path, "a") as f:
        del f.attrs["success"]
        del f["step_0000"]["clean_action"]
    report = audit(root=root, journal_records=journal, plans=[plan], target=1)
    assert not report["ok"]
    problems = report["schema_errors"][uid]
    assert any("success" in p for p in problems)
    assert any("clean_action" in p for p in problems)


def test_schema_checks_every_step_not_just_the_first(tmp_path):
    # G2R3 probe shape: first step intact, a LATER step missing a field — a
    # first-step-only checker admits a file whose write died halfway.
    plan, root, journal = _happy_tree(tmp_path)
    uid = plan["uids"][0]
    path = root / f"{plan['prefixes'][uid]}_a01.h5"
    with h5py.File(path, "a") as f:
        del f["step_0001"]["clean_action"]
    report = audit(root=root, journal_records=journal, plans=[plan], target=1)
    assert not report["ok"]
    assert any("step_0001" in p and "clean_action" in p for p in report["schema_errors"][uid])


def test_false_success_attr_rejected_despite_clean_journal(tmp_path):
    # G2R3 probe shape: journal says success, h5 attr says False — the file
    # belongs to a different outcome than the ledger claims.
    plan, root, journal = _happy_tree(tmp_path)
    uid = plan["uids"][0]
    path = root / f"{plan['prefixes'][uid]}_a01.h5"
    with h5py.File(path, "a") as f:
        f.attrs["success"] = False
    report = audit(root=root, journal_records=journal, plans=[plan], target=1)
    assert not report["ok"]
    assert any("success=False" in p for p in report["schema_errors"][uid])


def test_cli_writes_no_manifest_on_failed_audit(tmp_path):
    import argparse

    from exp.robocasa365.run_collect import write_run_plan as _write_plan  # noqa: F401 - fixture parity
    from exp.robocasa365.verify_collection_artifacts import run_cli

    plan, root, journal = _happy_tree(tmp_path)
    uid = plan["uids"][0]
    (root / f"{plan['prefixes'][uid]}_a01.h5").unlink()  # induce missing_file
    journal_path = tmp_path / "journal.jsonl"
    journal_path.write_text("\n".join(json.dumps(r) for r in journal))
    plan_path = tmp_path / "run_plan_collect_l1s1_pi05_b01.json"
    manifest_path = tmp_path / "manifest.json"
    args = argparse.Namespace(
        root=str(root), teacher="pi05", journal=str(journal_path),
        run_plan=[str(plan_path)], target=1, report_out="", manifest_out=str(manifest_path),
    )
    with pytest.raises(SystemExit):
        run_cli(args)
    assert not manifest_path.exists(), "a failed audit must not leave a manifest behind"


def test_cli_writes_manifest_on_passing_audit(tmp_path):
    import argparse

    from exp.robocasa365.verify_collection_artifacts import run_cli

    plan, root, journal = _happy_tree(tmp_path)
    journal_path = tmp_path / "journal.jsonl"
    journal_path.write_text("\n".join(json.dumps(r) for r in journal))
    plan_path = tmp_path / "run_plan_collect_l1s1_pi05_b01.json"
    manifest_path = tmp_path / "manifest.json"
    args = argparse.Namespace(
        root=str(root), teacher="pi05", journal=str(journal_path),
        run_plan=[str(plan_path)], target=1, report_out="", manifest_out=str(manifest_path),
    )
    report = run_cli(args)
    assert report["ok"] and manifest_path.exists()


def test_case_e_multi_batch_union_and_duplicate_rejection(tmp_path):
    plan1 = _run_plan(tmp_path, batch=1)
    plan2 = _run_plan(tmp_path, batch=2, episode_lo={"OpenCabinet": 2, "CloseDrawer": 1})
    uids, _prefixes, batches, hashes = merge_run_plans([plan1, plan2])
    assert len(uids) == len(set(uids)) == 6
    assert len(hashes) == 2
    assert {batches[uid] for uid in uids} == {1, 2}
    with pytest.raises(ValueError, match="more than one run-plan"):
        merge_run_plans([plan1, plan1])


def test_run_plan_params_schema_matches_frozen_text(tmp_path):
    # Frozen §4.3.6-(6): tasks carry inclusive episode_lo/episode_hi (batch
    # boundaries live in the hash), and collect_root is canonicalized.
    plan = _run_plan(tmp_path, batch=2, episode_lo={"OpenCabinet": 2, "CloseDrawer": 1})
    rows = {row["task_name"]: row for row in plan["params"]["tasks"]}
    assert set(rows["OpenCabinet"]) == {"task_name", "task_id", "episode_lo", "episode_hi"}
    assert (rows["OpenCabinet"]["episode_lo"], rows["OpenCabinet"]["episode_hi"]) == (2, 3)
    assert (rows["CloseDrawer"]["episode_lo"], rows["CloseDrawer"]["episode_hi"]) == (1, 1)
    assert not plan["params"]["collect_root"].endswith("/")


def test_collect_root_is_canonicalized():
    from exp.robocasa365.run_collect import canonical_collect_root

    assert canonical_collect_root("/data//x/./build_l1s1/") == "/data/x/build_l1s1"
    assert canonical_collect_root("/data/x/build_l1s1") == "/data/x/build_l1s1"


def test_collect_root_rejects_relative_paths():
    # G2R5: a relative root hashes identically across runs while resolving
    # against whatever cwd the server has — the hash would "verify" an
    # unstable output location.
    from exp.robocasa365.run_collect import canonical_collect_root

    with pytest.raises(ValueError, match="absolute"):
        canonical_collect_root("data/build_l1s1")
    with pytest.raises(ValueError, match="absolute"):
        canonical_collect_root("./build_l1s1")


def test_run_plan_hash_validated_on_every_read(tmp_path):
    plan = _run_plan(tmp_path)
    path = tmp_path / "tampered.json"
    plan["uids"] = plan["uids"][:-1]  # tamper without recomputing the hash
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan_hash"):
        load_run_plan(path)


# ------------------------------------------------------------------
# Canonical task / task_key three-way consistency (frozen §4.3.1a)
# ------------------------------------------------------------------


def test_task_key_three_way_consistency(tmp_path):
    plan, root, journal = _happy_tree(tmp_path)
    uid = plan["uids"][0]
    canonical = plan["prefixes"][uid].split("/")[1]
    # 1) The run-plan's canonical name is the env name, not natural language.
    assert canonical == "OpenCabinet"
    # 2) The h5 ``task`` attr equals it (the audit enforces this per file)...
    with h5py.File(root / f"{plan['prefixes'][uid]}_a01.h5", "r") as f:
        assert str(f.attrs["task"]) == canonical
        assert str(f.attrs["task"]) != "open the cabinet door"
    # 3) ...and the offline builder copies attrs["task"] verbatim into
    # task_key, so byte-equality of (episode_start.task == h5 task) — pinned in
    # the runner test — closes the chain.
    report = audit(root=root, journal_records=journal, plans=[plan], target=1)
    assert report["ok"]


# ------------------------------------------------------------------
# Manifest determinism
# ------------------------------------------------------------------


def test_manifest_deterministic_and_ordered(tmp_path):
    plan, root, journal = _happy_tree(tmp_path)
    report = audit(root=root, journal_records=journal, plans=[plan], target=1)
    m1 = json.dumps(build_manifest(report, root=root, target=1), sort_keys=True)
    m2 = json.dumps(build_manifest(report, root=root, target=1), sort_keys=True)
    assert m1 == m2
    manifest = json.loads(m1)
    assert manifest["plan_hashes"] == [plan["plan_hash"]]
    rows = manifest["tasks"]["OpenCabinet"]
    assert [r["episode_idx"] for r in rows] == sorted(r["episode_idx"] for r in rows)
    assert all(len(r["sha256"]) == 64 for r in rows)


# ------------------------------------------------------------------
# Completion rule (frozen §4.3.6-(4))
# ------------------------------------------------------------------


def test_binomial_rule_literals():
    # Locked literals: implementation drift must fail loudly.
    assert min_episodes_for_target(0.1) == 256
    assert min_episodes_for_target(0.2) == 126
    assert min_episodes_for_target(0.5) == 48


def test_binomial_rule_boundary_minimality():
    for sr in (0.1, 0.2, 0.5):
        n = min_episodes_for_target(sr)
        assert prob_at_least(n, sr, 20) >= 0.90
        assert prob_at_least(n - 1, sr, 20) < 0.90


def test_binomial_rule_rejects_degenerate_sr():
    with pytest.raises(ValueError):
        min_episodes_for_target(0.0)
