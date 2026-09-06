"""Object pinning end to end: the audit gate, then the library it feeds.

Two seams are covered here, in the order the pipeline runs them:

* ``verify_collection_artifacts._check_pin_provenance`` — whether an episode is
  allowed into the admitted set at all. The point of the check is that a
  *declared* identity proves nothing: a worker that accepted a pin table and
  then built an unpinned scene through any plumbing bug still stamps a
  perfectly correct ``pin_id`` / ``pin_task_id``. Only ``realized_objects``,
  read back off the built scene, can contradict it — so the negatives below
  keep the declaration honest and corrupt the evidence.
* ``build_in_memory_cache_artifact.resolve_from_manifest`` — whether the
  library is built from exactly the audited bytes, and whether the collection's
  identity survives the trip into the artifact and back out of the backend.

Fixture builders (run-plan, journal rows, h5 writer) are reused from
``test_collection_artifacts`` so the two files cannot drift into two different
ideas of what a collection tree looks like.
"""

from __future__ import annotations

import json
import pathlib
import pickle

import h5py
import numpy as np
import pytest

from exp.common.build_in_memory_cache_artifact import build_artifact
from exp.common.build_in_memory_cache_artifact import resolve_from_manifest
from exp.robocasa365.pinned_objects import compute_pin_id, compute_pin_task_id
from exp.robocasa365.verify_collection_artifacts import audit, build_manifest, merge_run_plans, run_cli
from tests.robocasa365.test_collection_artifacts import _happy_tree, _journal_row, _run_plan, _write_h5

# The table the collection is supposed to have run under.
PIN_TABLE = {
    "OpenCabinet": {
        "obj": "objects/objaverse/mug/mug_1/model.xml",
        "container": "objects/aigen_objs/plate/plate_7/model.xml",
    },
    "CloseDrawer": {"obj": "objects/objaverse/apple/apple_3/model.xml"},
}
PIN_ID = compute_pin_id(PIN_TABLE)

# A different table that agrees with PIN_TABLE on OpenCabinet. An OpenCabinet
# episode collected under it has a byte-identical slot map and a matching
# per-task identity — only the global identity separates the two runs.
OTHER_TABLE = {
    **PIN_TABLE,
    "CloseDrawer": {"obj": "objects/objaverse/banana/banana_5/model.xml"},
}
OTHER_PIN_ID = compute_pin_id(OTHER_TABLE)


# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _pin_attrs(task: str, realized: dict[str, str], *, pin_id: str = PIN_ID) -> dict[str, str]:
    """The provenance triple as ``episode_runner._realized_metadata`` writes it.

    The declared half (``pin_id`` / ``pin_task_id``) is derived from the pin
    TABLE and the realized half from whatever the caller passes, mirroring the
    real split: the driver stamps the declaration, the env supplies the
    evidence, and nothing on the worker side forces the two to agree.
    """
    return {
        "pin_id": pin_id,
        "pin_task_id": compute_pin_task_id(task, PIN_TABLE[task]),
        "realized_objects": json.dumps(realized, sort_keys=True),
    }


def _pinned_tree(tmp_path: pathlib.Path):
    """An honest collection tree: every episode realized what the table pinned."""
    plan = _run_plan(tmp_path, pin_id=PIN_ID, pinned_objects=PIN_TABLE)
    root = tmp_path / "build_l1s1"
    journal = []
    for uid in plan["uids"]:
        journal.append(_journal_row(uid))
        prefix = plan["prefixes"][uid]
        task = prefix.split("/")[1]
        _write_h5(root, prefix, 1, task=task, success=True,
                  pin_attrs=_pin_attrs(task, PIN_TABLE[task]))
    return plan, root, journal


def _restamp(root: pathlib.Path, plan: dict, uid: str, **overrides) -> None:
    """Rewrite one episode's provenance in place, leaving the others honest.

    Each negative case corrupts exactly one episode so the remaining ones stay
    available as the control: rejection has to be per episode, not per run.
    """
    prefix = plan["prefixes"][uid]
    task = prefix.split("/")[1]
    realized = overrides.pop("realized", PIN_TABLE[task])
    with h5py.File(root / f"{prefix}_a01.h5", "a") as f:
        for key, value in _pin_attrs(task, realized, **overrides).items():
            f.attrs[key] = value


def _set_attr(root: pathlib.Path, plan: dict, uid: str, key: str, value) -> None:
    """Write one raw attr, bypassing ``_pin_attrs``' well-formedness."""
    with h5py.File(root / f"{plan['prefixes'][uid]}_a01.h5", "a") as f:
        f.attrs[key] = value


def _pinned_audit(plan, root, journal):
    return audit(
        root=root, journal_records=journal, plans=[plan], target=1,
        pin_id=PIN_ID, pin_table=PIN_TABLE,
    )


def _manifest_uids(report, root) -> set[str]:
    manifest = build_manifest(report, root=root, target=1)
    return {row["task_uid"] for rows in manifest["tasks"].values() for row in rows}


def _uid_of(plan, task: str, episode_idx: int = 0) -> str:
    suffix = f"episode_{episode_idx:04d}"
    return next(uid for uid, prefix in plan["prefixes"].items()
                if prefix.split("/")[1] == task and prefix.endswith(suffix))


def _write_pin_manifest(tmp_path: pathlib.Path, table: dict, pin_id: str) -> pathlib.Path:
    path = tmp_path / "pinned_objects.json"
    path.write_text(json.dumps({"pin_id": pin_id, "pinned_objects": table}))
    return path


# ------------------------------------------------------------------
# Admission: realized objects decide, declarations do not
# ------------------------------------------------------------------


def test_pinned_happy_path_admits_and_manifest_carries_the_pin_id(tmp_path):
    plan, root, journal = _pinned_tree(tmp_path)
    report = _pinned_audit(plan, root, journal)
    assert report["ok"], report
    assert report["pin_errors"] == {}
    assert set(report["admitted"]) == set(plan["uids"])
    manifest = build_manifest(report, root=root, target=1)
    # The library builder reads the identity off the manifest; if it does not
    # travel there, the artifact gets stamped from a second source of truth.
    assert manifest["pin_id"] == PIN_ID


def test_realized_drift_rejected_although_the_declared_identity_is_perfect(tmp_path):
    # The failure this whole check exists for: the override was accepted (so the
    # declaration is right) but did not take effect (so one slot got a different
    # mesh). Nothing except realized_objects can see it.
    plan, root, journal = _pinned_tree(tmp_path)
    uid = _uid_of(plan, "OpenCabinet", 0)
    _restamp(root, plan, uid,
             realized={**PIN_TABLE["OpenCabinet"], "obj": "objects/objaverse/mug/mug_9/model.xml"})

    report = _pinned_audit(plan, root, journal)

    assert not report["ok"]
    assert uid in report["pin_errors"]
    assert any("mug_9" in problem for problem in report["pin_errors"][uid])
    assert uid not in report["admitted"]
    assert uid not in _manifest_uids(report, root)
    # The honest episodes are unaffected: rejection is per episode, not per run.
    assert set(report["admitted"]) == set(plan["uids"]) - {uid}


def test_foreign_global_pin_id_rejected_despite_a_self_consistent_task_slice(tmp_path):
    # An OpenCabinet episode from the OTHER table has the same slot map and the
    # same pin_task_id — checking only the task slice would admit an episode
    # from a different experiment.
    plan, root, journal = _pinned_tree(tmp_path)
    uid = _uid_of(plan, "OpenCabinet", 1)
    _restamp(root, plan, uid, pin_id=OTHER_PIN_ID)

    report = _pinned_audit(plan, root, journal)

    assert not report["ok"]
    problems = report["pin_errors"][uid]
    assert any("pin_id" in problem for problem in problems)
    # Only the global identity is wrong: the slot-level checks stayed silent.
    assert not any("realized" in problem and "!=" in problem for problem in problems)
    assert uid not in _manifest_uids(report, root)


def test_realized_missing_a_slot_is_rejected(tmp_path):
    # A slot that silently failed to spawn leaves a scene the library would
    # describe as containing an object it never had.
    plan, root, journal = _pinned_tree(tmp_path)
    uid = _uid_of(plan, "OpenCabinet", 0)
    _restamp(root, plan, uid, realized={"obj": PIN_TABLE["OpenCabinet"]["obj"]})

    report = _pinned_audit(plan, root, journal)

    assert not report["ok"]
    assert any("container" in problem for problem in report["pin_errors"][uid])
    assert uid not in _manifest_uids(report, root)


def test_realized_with_an_extra_slot_is_rejected(tmp_path):
    # The mirror case: an unpinned slot (a distractor the table never named)
    # made it into the scene, so the episode is not the pinned configuration.
    plan, root, journal = _pinned_tree(tmp_path)
    uid = _uid_of(plan, "CloseDrawer", 0)
    _restamp(root, plan, uid,
             realized={**PIN_TABLE["CloseDrawer"], "distractor": "objects/objaverse/can/can_2/model.xml"})

    report = _pinned_audit(plan, root, journal)

    assert not report["ok"]
    assert any("distractor" in problem for problem in report["pin_errors"][uid])
    assert uid not in _manifest_uids(report, root)


@pytest.mark.parametrize("attr", ["pin_id", "pin_task_id", "realized_objects"])
def test_missing_pin_attr_is_rejected(tmp_path, attr):
    # An episode collected before the worker stamped provenance, or by a worker
    # started without the flag, must not be quietly counted as pinned.
    plan, root, journal = _pinned_tree(tmp_path)
    uid = _uid_of(plan, "OpenCabinet", 0)
    with h5py.File(root / f"{plan['prefixes'][uid]}_a01.h5", "a") as f:
        del f.attrs[attr]

    report = _pinned_audit(plan, root, journal)

    assert not report["ok"]
    assert any(attr in problem for problem in report["pin_errors"][uid])
    assert uid not in _manifest_uids(report, root)


def test_task_absent_from_the_pin_table_is_rejected(tmp_path):
    # Auditing a run against a table that never named one of its tasks is a
    # table/run mismatch; admitting those episodes would pin nothing.
    plan, root, journal = _pinned_tree(tmp_path)
    partial_table = {"OpenCabinet": PIN_TABLE["OpenCabinet"]}
    report = audit(
        root=root, journal_records=journal, plans=[plan], target=1,
        pin_id=PIN_ID, pin_table=partial_table,
    )
    uid = _uid_of(plan, "CloseDrawer", 0)
    assert not report["ok"]
    assert any("no slot map" in problem for problem in report["pin_errors"][uid])


def test_pin_errors_are_not_reported_as_schema_errors(tmp_path):
    # The two error buckets answer different questions ("is the file whole" vs
    # "is it the right experiment"); collapsing them would hide either one.
    plan, root, journal = _pinned_tree(tmp_path)
    uid = _uid_of(plan, "OpenCabinet", 0)
    _restamp(root, plan, uid,
             realized={**PIN_TABLE["OpenCabinet"],
                       "container": "objects/aigen_objs/plate/plate_1/model.xml"})

    report = _pinned_audit(plan, root, journal)

    assert report["schema_errors"] == {}
    assert list(report["pin_errors"]) == [uid]


@pytest.mark.parametrize("bad", ["3", "[1, 2]", '"objects/objaverse/mug/mug_1/model.xml"'])
def test_realized_objects_that_is_json_but_not_an_object_is_a_problem_not_a_crash(tmp_path, bad):
    # Well-formed JSON of the wrong shape used to reach the set() comparison and
    # take the whole audit down with a TypeError — one malformed episode must
    # cost that episode, not the census.
    plan, root, journal = _pinned_tree(tmp_path)
    uid = _uid_of(plan, "OpenCabinet", 0)
    _set_attr(root, plan, uid, "realized_objects", bad)

    report = _pinned_audit(plan, root, journal)

    assert not report["ok"]
    assert uid in report["pin_errors"]
    assert any("not an object" in problem for problem in report["pin_errors"][uid])
    assert uid not in report["admitted"]
    # The other episodes were still judged: the audit ran to completion.
    assert set(report["admitted"]) == set(plan["uids"]) - {uid}


def test_unparsable_realized_objects_is_also_a_problem_not_a_crash(tmp_path):
    # The neighbouring branch: not JSON at all.
    plan, root, journal = _pinned_tree(tmp_path)
    uid = _uid_of(plan, "CloseDrawer", 0)
    _set_attr(root, plan, uid, "realized_objects", "{not json")

    report = _pinned_audit(plan, root, journal)

    assert any("not JSON" in problem for problem in report["pin_errors"][uid])


@pytest.mark.parametrize("pin_id", [None, ""])
def test_pin_table_without_its_identity_is_refused(tmp_path, pin_id):
    # Half a pin configuration is worse than none: the global check would
    # compare every episode against an empty string and fail the entire run
    # with a message that points at the episodes rather than at the caller.
    plan, root, journal = _pinned_tree(tmp_path)
    with pytest.raises(ValueError, match="pin_table given without pin_id"):
        audit(root=root, journal_records=journal, plans=[plan], target=1,
              pin_id=pin_id, pin_table=PIN_TABLE)


# ------------------------------------------------------------------
# Batch agreement: one audit, one pinning
# ------------------------------------------------------------------


_EXTENSION_LO = {"OpenCabinet": 2, "CloseDrawer": 1}


def test_pinned_and_unpinned_batches_cannot_be_audited_together(tmp_path):
    # The extension batch that forgot the flag: two scene distributions unioned
    # into one census, and the resulting library records only one of them.
    pinned = _run_plan(tmp_path, batch=1, pin_id=PIN_ID, pinned_objects=PIN_TABLE)
    unpinned = _run_plan(tmp_path, batch=2, episode_lo=_EXTENSION_LO)
    with pytest.raises(ValueError, match="disagree on pin_id"):
        merge_run_plans([pinned, unpinned])


def test_batches_under_different_pin_tables_cannot_be_audited_together(tmp_path):
    # Both pinned, but not to the same meshes — the subtler version of the same
    # mistake, and the one no per-episode check can see (each batch's episodes
    # are internally consistent with their own table).
    first = _run_plan(tmp_path, batch=1, pin_id=PIN_ID, pinned_objects=PIN_TABLE)
    second = _run_plan(tmp_path, batch=2, episode_lo=_EXTENSION_LO,
                       pin_id=OTHER_PIN_ID, pinned_objects=OTHER_TABLE)
    with pytest.raises(ValueError, match="disagree on pin_id"):
        merge_run_plans([first, second])


def test_batches_agreeing_on_the_pin_id_merge(tmp_path):
    # Both directions of agreement, including the pre-pinning one: the guard
    # must not turn "every batch is unpinned" into an error.
    pinned = [
        _run_plan(tmp_path, batch=1, pin_id=PIN_ID, pinned_objects=PIN_TABLE),
        _run_plan(tmp_path, batch=2, episode_lo=_EXTENSION_LO,
                  pin_id=PIN_ID, pinned_objects=PIN_TABLE),
    ]
    uids, _prefixes, _batches, hashes = merge_run_plans(pinned)
    assert len(uids) == len(set(uids)) == 6 and len(hashes) == 2

    # A separate tree: write_run_plan refuses to reuse a _bNN file for a plan
    # that hashes differently, which is its own (already covered) contract.
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy = [_run_plan(legacy_dir, batch=1),
              _run_plan(legacy_dir, batch=2, episode_lo=_EXTENSION_LO)]
    assert len(merge_run_plans(legacy)[0]) == 6


# ------------------------------------------------------------------
# Non-regression: the audit without --pinned-objects
# ------------------------------------------------------------------


def test_audit_without_the_pin_flag_is_unchanged(tmp_path):
    # Pre-pinning trees have none of the three attrs. Adding the check must not
    # retroactively invalidate every library built before it existed.
    plan, root, journal = _happy_tree(tmp_path)
    report = audit(root=root, journal_records=journal, plans=[plan], target=1)
    assert report["ok"], report
    assert report["pin_errors"] == {}
    assert report["pin_id"] is None
    assert len(report["admitted"]) == len(plan["uids"])
    # No pin table, no pin field: the manifest keeps its pre-pinning shape.
    assert "pin_id" not in build_manifest(report, root=root, target=1)


def test_pin_checks_run_only_when_a_table_is_supplied(tmp_path):
    # Same tree that the drift test rejects, audited with no table: admitted.
    # This pins the gating to the flag alone, so an unpinned audit can never
    # start failing on provenance it was never asked to judge.
    plan, root, journal = _pinned_tree(tmp_path)
    uid = _uid_of(plan, "OpenCabinet", 0)
    _restamp(root, plan, uid,
             realized={**PIN_TABLE["OpenCabinet"], "obj": "objects/objaverse/mug/mug_9/model.xml"})

    report = audit(root=root, journal_records=journal, plans=[plan], target=1)

    assert report["ok"], report
    assert report["pin_errors"] == {}
    assert uid in report["admitted"]


# ------------------------------------------------------------------
# CLI wiring
# ------------------------------------------------------------------


def _cli_args(tmp_path, root, journal, *, pinned_objects: str, manifest_out: pathlib.Path):
    import argparse

    journal_path = tmp_path / "journal.jsonl"
    journal_path.write_text("\n".join(json.dumps(r) for r in journal))
    return argparse.Namespace(
        root=str(root), teacher="pi05", journal=str(journal_path),
        run_plan=[str(tmp_path / "run_plan_collect_l1s1_pi05_b01.json")],
        target=1, report_out="", manifest_out=str(manifest_out),
        pinned_objects=pinned_objects,
    )


def test_cli_pinned_objects_flag_reaches_the_admission_gate(tmp_path):
    plan, root, journal = _pinned_tree(tmp_path)
    pin_path = _write_pin_manifest(tmp_path, PIN_TABLE, PIN_ID)
    manifest_out = tmp_path / "manifest.json"

    report = run_cli(_cli_args(tmp_path, root, journal, pinned_objects=str(pin_path),
                               manifest_out=manifest_out))

    assert report["ok"] and report["pin_id"] == PIN_ID
    assert json.loads(manifest_out.read_text())["pin_id"] == PIN_ID
    assert set(report["admitted"]) == set(plan["uids"])


def test_cli_writes_no_manifest_when_provenance_fails(tmp_path):
    # A manifest from a run whose objects drifted would look authoritative
    # while naming episodes from a different scene configuration.
    plan, root, journal = _pinned_tree(tmp_path)
    _restamp(root, plan, _uid_of(plan, "OpenCabinet", 0),
             realized={**PIN_TABLE["OpenCabinet"], "obj": "objects/objaverse/mug/mug_9/model.xml"})
    pin_path = _write_pin_manifest(tmp_path, PIN_TABLE, PIN_ID)
    manifest_out = tmp_path / "manifest.json"

    with pytest.raises(SystemExit):
        run_cli(_cli_args(tmp_path, root, journal, pinned_objects=str(pin_path),
                          manifest_out=manifest_out))

    assert not manifest_out.exists()


def test_cli_refuses_a_pinned_run_plan_without_its_table(tmp_path):
    # Forgetting the flag on a pinned collection is silent by construction: the
    # audit skips every provenance check and still emits a clean-looking
    # manifest. The run-plan already knows the run was pinned, so the CLI can
    # tell the two apart even though audit() cannot.
    _plan, root, journal = _pinned_tree(tmp_path)
    manifest_out = tmp_path / "manifest.json"

    with pytest.raises(SystemExit, match="refusing to audit a pinned collection without its table"):
        run_cli(_cli_args(tmp_path, root, journal, pinned_objects="", manifest_out=manifest_out))

    assert not manifest_out.exists()


def test_cli_still_audits_an_unpinned_run_plan_without_the_flag(tmp_path):
    # The refusal keys off the run-plan, not off the flag being absent: every
    # pre-pinning collection must still audit exactly as before.
    plan, root, journal = _happy_tree(tmp_path)
    manifest_out = tmp_path / "manifest.json"

    report = run_cli(_cli_args(tmp_path, root, journal, pinned_objects="", manifest_out=manifest_out))

    assert report["ok"] and report["pin_id"] is None
    assert set(report["admitted"]) == set(plan["uids"])
    assert "pin_id" not in json.loads(manifest_out.read_text())


def test_cli_without_the_pinned_objects_attribute_at_all(tmp_path):
    # Callers that build the Namespace by hand (this test file included)
    # predate the flag; the new refusal must read through the same getattr seam
    # rather than making the attribute required.
    plan, root, journal = _happy_tree(tmp_path)
    args = _cli_args(tmp_path, root, journal, pinned_objects="",
                     manifest_out=tmp_path / "manifest.json")
    del args.pinned_objects

    assert run_cli(args)["ok"]


# ------------------------------------------------------------------
# Library build from the manifest
# ------------------------------------------------------------------


def _audited_manifest(tmp_path) -> tuple[pathlib.Path, pathlib.Path, dict]:
    """Run the real audit over a pinned tree and return its manifest on disk."""
    _plan, root, journal = _pinned_tree(tmp_path)
    report = _pinned_audit(_plan, root, journal)
    assert report["ok"], report
    manifest = build_manifest(report, root=root, target=1)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=1))
    return root, path, manifest


def _rewrite_manifest(path: pathlib.Path, mutate) -> pathlib.Path:
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=1))
    return path


def test_manifest_resolves_to_exactly_the_audited_files(tmp_path):
    root, manifest_path, manifest = _audited_manifest(tmp_path)
    paths, loaded = resolve_from_manifest(root, manifest_path)
    expected = {(root / row["path"]).resolve()
                for rows in manifest["tasks"].values() for row in rows}
    assert set(paths) == expected
    assert len(paths) == len(expected), "resolve must not duplicate or drop entries"
    assert loaded["pin_id"] == PIN_ID and loaded["plan_hashes"] == manifest["plan_hashes"]


def test_file_edited_after_the_audit_is_refused_and_named(tmp_path):
    # The gap the digest closes: the audit judged bytes that are no longer on
    # disk. Silently building from the new bytes would make "the library is the
    # audited set" unverifiable after the fact.
    root, manifest_path, manifest = _audited_manifest(tmp_path)
    victim = manifest["tasks"]["OpenCabinet"][0]["path"]
    with h5py.File(root / victim, "a") as f:
        f.attrs["num_steps"] = 99

    with pytest.raises(ValueError, match="hashes to") as excinfo:
        resolve_from_manifest(root, manifest_path)
    # Which file changed, not just that one did: the operator has to be able to
    # go look at it.
    assert victim in str(excinfo.value)


def _unpinned_manifest(tmp_path) -> tuple[pathlib.Path, pathlib.Path, dict]:
    """The same shape from a pre-pinning collection: no attrs, no manifest pin_id."""
    plan, root, journal = _happy_tree(tmp_path)
    report = audit(root=root, journal_records=journal, plans=[plan], target=1)
    assert report["ok"], report
    manifest = build_manifest(report, root=root, target=1)
    path = tmp_path / "legacy_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=1))
    return root, path, manifest


def test_forged_manifest_pin_id_is_refused(tmp_path):
    # The hole the digests do not cover: editing ONE field of the manifest and
    # touching no episode at all would stamp a false identity onto the library,
    # which the serve-time binding check then accepts as gospel.
    root, manifest_path, manifest = _audited_manifest(tmp_path)
    _rewrite_manifest(manifest_path, lambda m: m.update(pin_id=OTHER_PIN_ID))

    with pytest.raises(ValueError, match="was collected under pin_id") as excinfo:
        resolve_from_manifest(root, manifest_path)

    # Named, not just counted — tasks are walked in sorted order, so the first
    # episode of the first task is the one that reports.
    assert manifest["tasks"]["CloseDrawer"][0]["path"] in str(excinfo.value)


def test_manifest_claiming_a_pin_id_over_unpinned_episodes_is_refused(tmp_path):
    # A pin_id pasted onto a legacy manifest: every digest matches, and every
    # episode is silent about which experiment produced it.
    root, manifest_path, _manifest = _unpinned_manifest(tmp_path)
    _rewrite_manifest(manifest_path, lambda m: m.update(pin_id=PIN_ID))

    with pytest.raises(ValueError, match="records no pin_id"):
        resolve_from_manifest(root, manifest_path)


def test_legacy_manifest_without_a_pin_id_skips_the_episode_check(tmp_path):
    # Non-regression: libraries built from pre-pinning manifests must keep
    # building, and must not acquire an identity they never had.
    root, manifest_path, manifest = _unpinned_manifest(tmp_path)

    paths, loaded = resolve_from_manifest(root, manifest_path)

    assert len(paths) == sum(len(rows) for rows in manifest["tasks"].values())
    assert loaded.get("pin_id") is None


def test_manifest_referencing_a_missing_file_is_refused(tmp_path):
    root, manifest_path, manifest = _audited_manifest(tmp_path)
    (root / manifest["tasks"]["CloseDrawer"][0]["path"]).unlink()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_from_manifest(root, manifest_path)


def test_manifest_path_escaping_the_data_dir_is_refused(tmp_path):
    # A manifest is an input file like any other; a '..' entry would let it
    # pull episodes from outside the tree the audit ever looked at.
    root, manifest_path, _manifest = _audited_manifest(tmp_path)
    _rewrite_manifest(
        manifest_path,
        lambda m: m["tasks"]["CloseDrawer"][0].update(path="../outside/episode_0000_a01.h5"),
    )
    with pytest.raises(ValueError, match="escapes"):
        resolve_from_manifest(root, manifest_path)


def test_duplicate_manifest_entry_is_refused(tmp_path):
    # One file listed twice would be built into two entries with the same
    # trajectory, inflating the library size the ablation treats as a variable.
    root, manifest_path, _manifest = _audited_manifest(tmp_path)
    _rewrite_manifest(
        manifest_path,
        lambda m: m["tasks"]["CloseDrawer"].append(dict(m["tasks"]["CloseDrawer"][0])),
    )
    with pytest.raises(ValueError, match="duplicate"):
        resolve_from_manifest(root, manifest_path)


def test_non_manifest_json_is_refused(tmp_path):
    root, manifest_path, _manifest = _audited_manifest(tmp_path)
    manifest_path.write_text(json.dumps({"episodes": []}))
    with pytest.raises(ValueError, match="not an audit manifest"):
        resolve_from_manifest(root, manifest_path)


def test_manifest_and_episode_list_are_mutually_exclusive(tmp_path):
    # Two different answers to "which episodes" cannot both be honoured; the
    # dangerous outcome is one silently winning.
    root, manifest_path, _manifest = _audited_manifest(tmp_path)
    listing = tmp_path / "subset.txt"
    listing.write_text("pi05/OpenCabinet/episode_0000_a01.h5\n")
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_artifact(
            str(root), "cp1_mean_pool", workers=-1,
            manifest=str(manifest_path), episode_list=str(listing),
        )


# ------------------------------------------------------------------
# Identity survives into the artifact and out of the backend
# ------------------------------------------------------------------


def _write_buildable_h5(root: pathlib.Path, prefix: str, *, task: str, pin_attrs: dict[str, str]):
    """One episode that satisfies BOTH the auditor's schema and the key builder.

    The manifest hands the builder the very files the auditor admitted, so a
    fixture that only satisfies one of the two contracts cannot exercise the
    handoff. 256 tokens per vision field is not decoration: the CP1 builders
    slice prefix_embs at fixed 256-token boundaries.
    """
    path = root / f"{prefix}_a01.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["task"] = task
        f.attrs["success"] = True
        f.attrs["num_steps"] = 1
        for key, value in pin_attrs.items():
            f.attrs[key] = value
        grp = f.create_group("step_0000")
        for j in range(3):
            grp.create_dataset(f"vision_{j}", data=np.full((256, 2048), 0.5, dtype=np.float16))
        grp.create_dataset("prompt_emb", data=np.full((10, 2048), 0.25, dtype=np.float16))
        grp.create_dataset("robot_state", data=np.arange(32, dtype=np.float32))
        grp.create_dataset("clean_action", data=np.ones((10, 32), dtype=np.float32))
    return path


def test_manifest_built_library_carries_the_collection_identity(tmp_path):
    # The whole chain in one test: audit -> manifest -> artifact -> backend.
    # If the identity drops anywhere along it, the serving-side pin check has
    # nothing to compare a config against and silently accepts any library.
    plan = _run_plan(tmp_path, pin_id=PIN_ID, pinned_objects=PIN_TABLE)
    root = tmp_path / "build_l1s1"
    journal = []
    for uid in plan["uids"]:
        journal.append(_journal_row(uid))
        prefix = plan["prefixes"][uid]
        task = prefix.split("/")[1]
        _write_buildable_h5(root, prefix, task=task, pin_attrs=_pin_attrs(task, PIN_TABLE[task]))

    report = _pinned_audit(plan, root, journal)
    assert report["ok"], report
    manifest = build_manifest(report, root=root, target=1)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    artifact = build_artifact(
        str(root), "cp1_mean_pool", workers=-1,
        manifest=str(manifest_path), trajectory_id_mode="relpath",
    )

    assert artifact["pin_id"] == PIN_ID
    assert artifact["plan_hashes"] == [plan["plan_hash"]]
    # Exactly the manifest's episodes (one step each), no more and no less —
    # the manifest is the target-capped selection, not the whole tree, so a
    # builder that fell back to the directory scan would show up here.
    listed = [row["path"] for rows in manifest["tasks"].values() for row in rows]
    assert {e.id for e in artifact["entries"]} == {f"{p[: -len('.h5')]}:0" for p in listed}

    from openpi.cache.backends.in_memory_backend import InMemoryBackend

    pkl = tmp_path / "library.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(artifact, fh)
    backend = InMemoryBackend(vector_dims=artifact["vector_dims"])
    backend.load_artifact(str(pkl))

    assert backend.artifact_meta["pin_id"] == PIN_ID


def test_library_built_without_a_manifest_reports_no_pin_identity(tmp_path):
    # Non-regression: the legacy scan path must keep producing artifacts whose
    # pin_id reads back as None (the serving check treats None as "unpinned"),
    # not as a missing key that would raise on access.
    root = tmp_path / "build_l1s1"
    _write_buildable_h5(root, "pi05/OpenCabinet/episode_0000", task="OpenCabinet", pin_attrs={})

    artifact = build_artifact(str(root), "cp1_mean_pool", workers=-1)

    assert "pin_id" not in artifact and "plan_hashes" not in artifact

    from openpi.cache.backends.in_memory_backend import InMemoryBackend

    pkl = tmp_path / "legacy.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(artifact, fh)
    backend = InMemoryBackend(vector_dims=artifact["vector_dims"])
    backend.load_artifact(str(pkl))

    assert backend.artifact_meta["pin_id"] is None


# ------------------------------------------------------------------
# CLI settles table identity once, not once per episode
# ------------------------------------------------------------------


class TestCliPlanIdentity:
    """The run-plan knows how the collection ran; the CLI should say so plainly.

    Both cases would eventually surface through the per-episode checks -- as one
    identical failure per admitted episode, naming an attr mismatch rather than
    the operator error that produced it.
    """

    def test_wrong_table_is_named_as_such_not_as_n_attr_mismatches(self, tmp_path):
        plan, root, journal = _pinned_tree(tmp_path)
        other = {**PIN_TABLE, "ExtraTask": {"obj": "objects/x/y/model.xml"}}
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        wrong = _write_pin_manifest(other_dir, other, compute_pin_id(other))
        args = _cli_args(tmp_path, root, journal, pinned_objects=str(wrong),
                         manifest_out=tmp_path / "m.json")
        with pytest.raises(SystemExit, match="not the table this collection ran under"):
            run_cli(args)
        assert not (tmp_path / "m.json").exists()

    def test_matching_table_still_passes(self, tmp_path):
        plan, root, journal = _pinned_tree(tmp_path)
        pin_path = _write_pin_manifest(tmp_path, PIN_TABLE, PIN_ID)
        report = run_cli(_cli_args(tmp_path, root, journal, pinned_objects=str(pin_path),
                                   manifest_out=tmp_path / "ok.json"))
        assert report["ok"]
