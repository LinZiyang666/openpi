"""Three-source join, admission rules, and the cross-machine package (X14 §3.6/§3.7).

Every rule tested here exists to keep a specific kind of bad episode out of an
Adam step:

  - a **stale attempt** is journaled exactly like the live one, so only the
    scheduler's ``accepted`` flag separates them;
  - an **errored** episode never executed the arm it claims;
  - a **partial shard** means the episode ended without an episode_end, so its
    trajectory is truncated at an arbitrary point;
  - a **version mismatch** is a hot-swap race — the rollout came from other
    weights than the batch is crediting;
  - a **discontinuous decision_idx** means the client and the server disagree
    about how many verdicts happened.

Plus the repair path: a slot re-run under a new ``#r1`` uid must contribute
exactly one selected episode, never two.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from exp.rl_router.batch_package import (
    COMPLETE_MARKER,
    assemble_package,
    build_batch_manifest,
    repair_round,
    slot_of,
    verify_package,
)

BATCH = "b0"
VERSION = "v3"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _journal(uid: str, *, attempt: int = 1, accepted: bool = True,
             success: bool = True, error: str | None = None) -> dict:
    return {"task_uid": uid, "yaml_id": "rlr", "phase": "eval",
            "status": "done" if success else "failed", "success": success,
            "attempt": attempt, "accepted": accepted, "error": error, "ts": 0.0}


def _shard(uid: str, *, attempt: int = 1, rows: int = 3, status: str = "complete",
           weights_version: str = VERSION) -> dict:
    return {"run_id": "run0", "batch_id": BATCH, "task_uid": uid, "attempt": attempt,
            "weights_version": weights_version, "encoder_version": "enc",
            "shard": f"{uid}.bin", "sidecar": f"{uid}.jsonl", "rows": rows, "dim": 8,
            "dtype": "float16", "sha256": "0" * 64, "status": status, "ts": 0.0}


def _client_rows(uid: str, *, attempt: int = 1, n: int = 3,
                 weights_version: str = VERSION, skip: int | None = None) -> list[dict]:
    rows = []
    for i in range(n):
        if skip is not None and i == skip:
            continue
        rows.append({
            "task_uid": uid, "attempt": attempt, "step_idx": i * 5,
            "router_outputs": {
                "decision_idx": i, "arm_sampled": "teacher", "arm_executed": "teacher",
                "probs": [1.0], "temperature": 1.0, "weights_version": weights_version,
                "seed_ep": 1, "fallback": False,
            },
        })
    return rows


def _manifest(*, journal, shards, client_rows, expected):
    return build_batch_manifest(
        batch_id=BATCH, weights_version=VERSION, expected_slots=expected,
        journal=journal, client_rows=client_rows, shards=shards,
    )


# ---------------------------------------------------------------------------
# Slot identity
# ---------------------------------------------------------------------------


def test_repair_uid_maps_back_to_its_slot() -> None:
    assert slot_of("y:eval:1:2") == "y:eval:1:2"
    assert slot_of("y:eval:1:2#r1") == "y:eval:1:2"
    assert repair_round("y:eval:1:2") == 0
    assert repair_round("y:eval:1:2#r2") == 2


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_complete_batch_selects_every_slot_once() -> None:
    uids = ["u0", "u1", "u2"]
    m = _manifest(
        journal=[_journal(u) for u in uids],
        shards=[_shard(u) for u in uids],
        client_rows=[r for u in uids for r in _client_rows(u)],
        expected=uids,
    )
    assert m.complete and not m.rejected
    assert sorted(r.task_uid for r in m.selected) == uids
    assert all(r.rows == 3 for r in m.selected)


def test_success_flag_comes_from_the_journal() -> None:
    m = _manifest(
        journal=[_journal("u0", success=True), _journal("u1", success=False)],
        shards=[_shard("u0"), _shard("u1")],
        client_rows=_client_rows("u0") + _client_rows("u1"),
        expected=["u0", "u1"],
    )
    assert {r.task_uid: r.success for r in m.selected} == {"u0": True, "u1": False}


# ---------------------------------------------------------------------------
# Rejection matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate,reason",
    [
        ("stale", "scheduler_rejected"),
        ("error", "episode_error"),
        ("partial_shard", "shard_partial"),
        ("no_shard", "shard_missing"),
        ("bad_version", "weights_version_mismatch"),
        ("zero_rows", "no_steps"),
        ("client_gap", "decision_idx_discontinuous"),
        ("no_client", "client_rows_missing"),
    ],
)
def test_inadmissible_episodes_are_rejected_with_a_reason(mutate: str, reason: str) -> None:
    journal = [_journal("u0")]
    shards = [_shard("u0")]
    client = _client_rows("u0")

    if mutate == "stale":
        journal = [_journal("u0", accepted=False)]
    elif mutate == "error":
        journal = [_journal("u0", success=False, error="SidecarError: down")]
    elif mutate == "partial_shard":
        shards = [_shard("u0", status="partial")]
    elif mutate == "no_shard":
        shards = []
    elif mutate == "bad_version":
        shards = [_shard("u0", weights_version="v2")]
    elif mutate == "zero_rows":
        shards = [_shard("u0", rows=0, status="empty")]
    elif mutate == "client_gap":
        client = _client_rows("u0", skip=1)
    elif mutate == "no_client":
        client = []

    m = _manifest(journal=journal, shards=shards, client_rows=client, expected=["u0"])
    assert not m.complete and m.missing_slots == ["u0"]
    assert not m.selected
    # "empty" reports its own status; the rest map to the parametrised reason.
    assert m.rejected[0]["reason"] in (reason, "shard_empty")


def test_a_short_batch_is_never_silently_trained_on() -> None:
    """Shrinking the batch would change the estimator's variance and the
    interaction accounting; the packager reports the gap instead."""
    m = _manifest(
        journal=[_journal("u0")], shards=[_shard("u0")],
        client_rows=_client_rows("u0"), expected=["u0", "u1", "u2"],
    )
    assert not m.complete
    assert m.missing_slots == ["u1", "u2"]
    assert len(m.selected) == 1


def test_episodes_from_another_batch_are_ignored_not_rejected() -> None:
    m = _manifest(
        journal=[_journal("u0"), _journal("other-batch-uid")],
        shards=[_shard("u0")], client_rows=_client_rows("u0"), expected=["u0"],
    )
    assert m.complete and not m.rejected


# ---------------------------------------------------------------------------
# Repair rounds
# ---------------------------------------------------------------------------


def test_repair_round_fills_a_missing_slot_exactly_once() -> None:
    """First round: shard never finalized. Repair round re-runs the init under
    a new uid and the slot closes with one selected episode."""
    first = _manifest(journal=[_journal("u0")], shards=[], client_rows=_client_rows("u0"),
                      expected=["u0"])
    assert first.missing_slots == ["u0"]

    repaired = _manifest(
        journal=[_journal("u0"), _journal("u0#r1")],
        shards=[_shard("u0#r1")],
        client_rows=_client_rows("u0") + _client_rows("u0#r1"),
        expected=["u0"],
    )
    assert repaired.complete
    assert [r.task_uid for r in repaired.selected] == ["u0#r1"]


def test_original_wins_when_both_the_original_and_a_repair_land() -> None:
    """A repair dispatched before a late original finalized: the priority is
    deterministic and reproducible from the inputs, never arrival order."""
    m = _manifest(
        journal=[_journal("u0"), _journal("u0#r1")],
        shards=[_shard("u0"), _shard("u0#r1")],
        client_rows=_client_rows("u0") + _client_rows("u0#r1"),
        expected=["u0"],
    )
    assert [r.task_uid for r in m.selected] == ["u0"]
    assert [s["task_uid"] for s in m.superseded] == ["u0#r1"]


def test_earlier_repair_round_wins_over_a_later_one() -> None:
    m = _manifest(
        journal=[_journal("u0#r2"), _journal("u0#r1")],
        shards=[_shard("u0#r1"), _shard("u0#r2")],
        client_rows=_client_rows("u0#r1") + _client_rows("u0#r2"),
        expected=["u0"],
    )
    assert [r.task_uid for r in m.selected] == ["u0#r1"]


def test_selection_is_independent_of_input_order() -> None:
    args = dict(
        journal=[_journal("u0"), _journal("u0#r1")],
        shards=[_shard("u0"), _shard("u0#r1")],
        client_rows=_client_rows("u0") + _client_rows("u0#r1"),
        expected=["u0"],
    )
    forward = _manifest(**args)
    reversed_args = {k: (list(reversed(v)) if k != "expected" else v) for k, v in args.items()}
    assert [r.task_uid for r in forward.selected] == \
           [r.task_uid for r in _manifest(**reversed_args).selected]


def test_a_higher_attempt_of_one_uid_wins() -> None:
    """Both attempts accepted (a requeue whose first result arrived late but
    was still fenced in): the live generation is the higher one."""
    m = _manifest(
        journal=[_journal("u0", attempt=1), _journal("u0", attempt=2)],
        shards=[_shard("u0", attempt=1), _shard("u0", attempt=2)],
        client_rows=_client_rows("u0", attempt=1) + _client_rows("u0", attempt=2),
        expected=["u0"],
    )
    assert [(r.task_uid, r.attempt) for r in m.selected] == [("u0", 2)]
    assert m.superseded[0]["attempt"] == 1


# ---------------------------------------------------------------------------
# Cross-machine package
# ---------------------------------------------------------------------------


def _write_package(tmp_path) -> pathlib.Path:
    return assemble_package(
        tmp_path / "pkg", batch_id=BATCH, weights_version=VERSION,
        journal_rows=[_journal("u0")], client_rows=_client_rows("u0"),
        expected_slots=["u0"],
    )


def test_assembled_package_verifies(tmp_path) -> None:
    pkg = _write_package(tmp_path)
    meta = verify_package(pkg)
    assert meta["batch_id"] == BATCH and meta["weights_version"] == VERSION
    assert set(meta["sha256"]) == {
        "journal_slice.jsonl", "accepted_manifest.json", "per_step_rows_batch.jsonl",
    }


def test_missing_completion_marker_is_a_partial_copy(tmp_path) -> None:
    """A package without its marker is mid-scp, not short — re-pushing is the
    fix, and it must be detectable rather than trained on."""
    pkg = _write_package(tmp_path)
    (pkg / COMPLETE_MARKER).unlink()
    with pytest.raises(ValueError, match=COMPLETE_MARKER):
        verify_package(pkg)


def test_truncated_payload_fails_the_digest(tmp_path) -> None:
    pkg = _write_package(tmp_path)
    (pkg / "per_step_rows_batch.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_package(pkg)


def test_missing_payload_file_is_caught(tmp_path) -> None:
    pkg = _write_package(tmp_path)
    (pkg / "journal_slice.jsonl").unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_package(pkg)


def test_repush_is_idempotent(tmp_path) -> None:
    """A retried scp overwrites in place and still verifies, so transport
    failures cost a re-copy rather than a re-run."""
    first = _write_package(tmp_path)
    digest = json.loads((first / COMPLETE_MARKER).read_text())["package_sha256"]
    second = _write_package(tmp_path)
    assert json.loads((second / COMPLETE_MARKER).read_text())["package_sha256"] == digest


def test_package_round_trips_into_a_manifest(tmp_path) -> None:
    from exp.rl_router.batch_package import read_jsonl

    pkg = _write_package(tmp_path)
    accepted = json.loads((pkg / "accepted_manifest.json").read_text())
    m = build_batch_manifest(
        batch_id=accepted["batch_id"], weights_version=accepted["weights_version"],
        expected_slots=accepted["expected_slots"],
        journal=read_jsonl(pkg / "journal_slice.jsonl"),
        client_rows=read_jsonl(pkg / "per_step_rows_batch.jsonl"),
        shards=[_shard("u0")],
    )
    assert m.complete and [r.task_uid for r in m.selected] == ["u0"]


def test_manifest_serialises_to_a_reloadable_dict(tmp_path) -> None:
    """The trainer reads the manifest back from disk on the server side."""
    from exp.rl_router.train_router import _manifest_from_dict

    m = _manifest(journal=[_journal("u0")], shards=[_shard("u0")],
                  client_rows=_client_rows("u0"), expected=["u0"])
    reloaded = _manifest_from_dict(json.loads(json.dumps(m.to_dict())))
    assert reloaded.complete
    assert [r.task_uid for r in reloaded.selected] == ["u0"]
    assert reloaded.selected[0].rows == 3


# ---------------------------------------------------------------------------
# Trainer refuses an incomplete batch
# ---------------------------------------------------------------------------


def test_trainer_refuses_to_load_a_short_batch(tmp_path) -> None:
    from exp.rl_router.train_router import load_batch

    m = _manifest(journal=[_journal("u0")], shards=[_shard("u0")],
                  client_rows=_client_rows("u0"), expected=["u0", "u1"])
    with pytest.raises(ValueError, match="repair round"):
        load_batch(m, shard_dir=tmp_path, package_dir=tmp_path, n_arms=3)


def test_trainer_detects_a_shard_that_changed_after_finalize(tmp_path) -> None:
    """The manifest's digest is what makes a finalized shard trustworthy."""
    from exp.rl_router.train_router import load_batch

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / "u0.bin").write_bytes(b"\x00" * 48)
    (shard_dir / "u0.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "per_step_rows_batch.jsonl").write_text("", encoding="utf-8")

    m = _manifest(journal=[_journal("u0")], shards=[_shard("u0")],
                  client_rows=_client_rows("u0"), expected=["u0"])
    from exp.rl_router.train_router import EpisodeAdmissionError

    # A load defect names its slot so the run loop can repair it, rather than
    # surfacing as a generic crash the bounded repair loop cannot act on.
    with pytest.raises(EpisodeAdmissionError) as excinfo:
        load_batch(m, shard_dir=shard_dir, package_dir=tmp_path, n_arms=3)
    assert excinfo.value.rejected[0]["reason"] == "shard_digest_mismatch"


def test_trainer_loads_a_verified_batch(tmp_path) -> None:
    from exp.rl_router.train_router import load_batch

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    payload = b"\x00\x3c" * (3 * 8)          # 3 rows x 8 dims of fp16 1.0
    (shard_dir / "u0.bin").write_bytes(payload)
    sidecar_bytes = "".join(
        json.dumps({"task_uid": "u0", "attempt": 1, "batch_id": BATCH,
                    "weights_version": VERSION, "decision_idx": i,
                    "arm_sampled": "teacher", "arm_mapped": "teacher",
                    "logits": [0.1, 0.2, 0.3], "logprob_sampled": -1.2}) + "\n"
        for i in range(3)
    ).encode("utf-8")
    (shard_dir / "u0.jsonl").write_bytes(sidecar_bytes)
    (tmp_path / "per_step_rows_batch.jsonl").write_text("".join(
        json.dumps(r) + "\n" for r in _client_rows("u0")
    ), encoding="utf-8")

    shard = _shard("u0")
    shard["sha256"] = hashlib.sha256(payload).hexdigest()
    shard["sidecar_sha256"] = hashlib.sha256(sidecar_bytes).hexdigest()
    m = _manifest(journal=[_journal("u0")], shards=[shard],
                  client_rows=_client_rows("u0"), expected=["u0"])
    (episode,) = load_batch(m, shard_dir=shard_dir, package_dir=tmp_path, n_arms=3)
    assert episode.features.shape == (3, 8)
    assert episode.arm_sampled == ["teacher"] * 3
    assert episode.arm_executed == ["teacher"] * 3
    assert episode.success is True


def test_trainer_rejects_a_judge_interceptor_disagreement(tmp_path) -> None:
    """If the judge mapped one arm and the interceptor executed another, the
    cost accounting and the gradient would describe different rollouts."""
    from exp.rl_router.train_router import load_batch

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    payload = b"\x00\x3c" * 8
    (shard_dir / "u0.bin").write_bytes(payload)
    (shard_dir / "u0.jsonl").write_text(json.dumps(
        {"task_uid": "u0", "attempt": 1, "batch_id": BATCH, "weights_version": VERSION,
         "decision_idx": 0, "arm_sampled": "cache", "arm_mapped": "cache",
         "logits": [0.1, 0.2, 0.3], "logprob_sampled": -0.9}) + "\n", encoding="utf-8")
    rows = _client_rows("u0", n=1)
    rows[0]["router_outputs"]["arm_executed"] = "teacher"
    (tmp_path / "per_step_rows_batch.jsonl").write_text(
        json.dumps(rows[0]) + "\n", encoding="utf-8")

    shard = _shard("u0", rows=1)
    shard["sha256"] = hashlib.sha256(payload).hexdigest()
    m = _manifest(journal=[_journal("u0")], shards=[shard],
                  client_rows=rows, expected=["u0"])
    from exp.rl_router.train_router import EpisodeAdmissionError

    with pytest.raises(EpisodeAdmissionError) as excinfo:
        load_batch(m, shard_dir=shard_dir, package_dir=tmp_path, n_arms=3)
    assert excinfo.value.rejected[0]["reason"] == "arm_mapping_disagreement"
