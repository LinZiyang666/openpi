import json

import numpy as np
import pytest

from exp.offline_search.debug import reader, schema
from exp.offline_search.debug.fixtures import make_synthetic_arm
from exp.offline_search.debug.validate import validate_arm, sha256


@pytest.fixture
def arm(tmp_path):
    return reader.open_arm(make_synthetic_arm(tmp_path / "run"), "synthetic")


def codes(report):
    return {m["code"] for m in report["missing"]}


def test_capture_and_augmentation_pass(arm):
    report = validate_arm(arm, expected_pairs=6)
    assert report["status"] == "PASS", report["missing"]
    assert report["n_decisions"] == 54
    assert report["n_controls"] == 282
    assert report["unaccepted_attempts"] == 0
    assert "accepted_pairs" in codes(validate_arm(arm))


@pytest.mark.parametrize("field,code", [("control_idx", "control_contiguity"), ("contact_off", "contact_offsets"), ("qpos", "control_dtype")])
def test_bad_physics_and_indices_are_exactly_reported(arm, field, code):
    path = next((arm.debug_dir / "client").glob("*/controls_0000.npz"))
    data = reader.read_npz(path)
    if field == "qpos":
        data[field] = data[field].astype(np.float32)
    else:
        data[field] = data[field].copy()
        data[field][0] += 7
    schema.write_npz_block(path, data)
    report = validate_arm(arm, 6)
    assert report["status"] == "INCOMPLETE"
    assert code in codes(report)
    assert "receipt_mismatch" in codes(report)


def test_missing_and_orphan_server_decisions(arm):
    path = arm.server_dirs[0] / "decisions.jsonl"
    rows = reader.read_jsonl(path)
    rows[1]["decision_id"] += "x"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = validate_arm(arm, 6)
    assert {"server_join", "missing_client_event"} <= codes(report)


def test_image_shapes_checked_against_wire_camera_aliases(arm):
    for path in (arm.debug_dir / "client").glob("*/episode.json"):
        ep = reader.read_json(path)
        ep["image_shapes"] = {"observation/image": [16, 16, 3], "observation/wrist_image": [16, 16, 3]}
        path.write_text(json.dumps(ep))
    arm.server_meta["camera_wire_keys"] = dict(third="observation/image", wrist="observation/wrist_image")
    report = validate_arm(arm, 6)
    assert "image_decode" not in codes(report)
    path = next((arm.server_dirs[0] / "blocks").glob("*.npz"))
    data = reader.read_npz(path)
    data["img_wrist"] = data["img_wrist"].astype(np.float32)
    schema.write_npz_block(path, data)
    assert "image_decode" in codes(validate_arm(arm, 6))


def test_receipt_complete_and_file_entries_reconcile(arm):
    for complete in (arm.debug_dir / "receipts").glob("*/complete.json"):
        obj = reader.read_json(complete)
        files = {v["name"]: dict(bytes=v["bytes"], sha256=v["sha256"]) for v in obj["files"]}
        obj["files"] = files
        obj["status"] = "complete"
        complete.write_text(json.dumps(obj))
        for name, value in files.items():
            (complete.parent / (name + ".json")).write_text(json.dumps(dict(relative_path=complete.parent.name + "/" + name, uid=obj["task_uid"], **value)))
    assert validate_arm(arm, 6)["status"] == "PASS"


def test_capture_only_vs_missing_augmentation(arm):
    next((arm.debug_dir / "aug/policy_shadow").glob("*.npz")).unlink()
    assert validate_arm(arm, 6, require_aug=False)["status"] == "PASS"
    report = validate_arm(arm, 6)
    assert report["capture_status"] == "PASS"
    assert report["augmentation_status"] == "INCOMPLETE"
    assert len(report["augmentation_missing"]) == 54


def test_snapshot_sampling_missing_receipt_and_object_pickle(arm):
    path = next((arm.debug_dir / "client").glob("*/snap_reset.npz"))
    path.unlink()
    receipt = next((arm.debug_dir / "receipts").glob("*/complete.json"))
    receipt.unlink()
    report = validate_arm(arm, 6)
    assert {"missing_snapshot", "missing_complete_receipt", "missing_receipt"} <= codes(report)
    path = next((arm.debug_dir / "client").glob("*/controls_0000.npz"))
    with path.open("wb") as f:
        np.savez_compressed(f, control_idx=np.array([{}], object))
    assert "control_decode" in codes(validate_arm(arm, 6))


def test_aug_duplicate_and_provenance_checked(arm):
    path = next((arm.debug_dir / "aug/policy_shadow").glob("*.npz"))
    data = reader.read_npz(path)
    data["decision_id"][1] = data["decision_id"][0]
    data["_meta_json"] = np.array("{}")
    schema.write_npz_block(path, data)
    report = validate_arm(arm, 6)
    assert {"duplicate_aug_id", "aug_provenance", "missing_aug_decision"} <= codes(report)


def test_corrupt_archive_and_incomplete_receipt_are_incomplete(arm):
    path = next((arm.server_dirs[0] / "blocks").glob("*.npz"))
    data = path.read_bytes()
    path.write_bytes(data[:len(data) // 2])
    receipt = next((arm.debug_dir / "receipts").glob("*/complete.json"))
    metadata = reader.read_json(receipt)
    metadata["status"] = "error"
    receipt.write_text(json.dumps(metadata))
    report = validate_arm(arm, 6)
    assert {"block_index", "decision_block_decode", "receipt_status"} <= codes(report)


def test_missing_camera_declaration_and_invalid_settle_assignment(arm):
    episode = next((arm.debug_dir / "client").glob("*/episode.json"))
    metadata = reader.read_json(episode)
    metadata["image_shapes"] = {}
    episode.write_text(json.dumps(metadata))
    block = episode.parent / "controls_0000.npz"
    data = reader.read_npz(block)
    data["decision_seq"][0] = metadata["n_decisions"]
    schema.write_npz_block(block, data)
    assert {"missing_image_shape", "settle_assignment"} <= codes(validate_arm(arm, 6))


def test_top_level_server_error_and_image_availability_are_admission_errors(arm):
    path = arm.server_dirs[0] / "decisions.jsonl"
    rows = reader.read_jsonl(path)
    rows[0]["status"] = "error"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    block = arm.server_dirs[0] / "blocks/d_1_000000.npz"
    data = reader.read_npz(block)
    data["img_wrist_available"][0] = False
    schema.write_npz_block(block, data)
    report = validate_arm(arm, 6)
    assert {"server_capture_error", "image_unavailable"} <= codes(report)
    did = rows[0]["decision_id"]
    assert arm.decision_arrays(["state_wire"], [did])["state_wire"].shape == (1, 8)
    with pytest.raises(KeyError, match="unavailable"):
        arm.images([did])


def test_validate_uses_pid_meta_and_reports_skipped_tail_without_crashing(arm):
    directory = arm.server_dirs[0]
    (directory / "meta.json").rename(directory / "meta_1.json")
    path = directory / "decisions.jsonl"
    path.rename(directory / "decisions_1.jsonl")
    path = directory / "decisions_1.jsonl"
    with path.open("a") as f:
        f.write('{"killed":')
    arm = reader.open_arm(arm.run_root, "synthetic")
    result = validate_arm(arm, 6)
    assert result["status"] == "PASS", result["missing"]
    assert result["read_issues"][0]["code"] == "truncated_jsonl_tail"
    meta = reader.read_json(directory / "meta_1.json")
    meta["H"] = 16
    (directory / "meta_1.json").write_text(json.dumps(meta))
    assert "decision_chunk" in codes(validate_arm(reader.open_arm(arm.run_root, "synthetic"), 6))


def test_new_process_cannot_borrow_an_unrelated_legacy_meta(arm):
    directory = arm.server_dirs[0]
    rows = reader.read_jsonl(directory / "decisions.jsonl")
    for row in rows:
        row["pid"] = 99
    (directory / "decisions_99.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "decisions.jsonl").unlink()
    arm = reader.open_arm(arm.run_root, "synthetic")
    assert "missing_server_meta" in codes(validate_arm(arm, 6))


def test_float64_wire_actions_retain_precision_and_float32_is_rejected(arm):
    assert validate_arm(arm, 6)["status"] == "PASS"
    block = next((arm.server_dirs[0] / "blocks").glob("*.npz"))
    data = reader.read_npz(block)
    value = .123456789012345
    assert value != float(np.float32(value))
    data["served_wire"][0, 0, 0] = value
    schema.write_npz_block(block, data)
    ek = str(data["decision_id"][0]).split(":")[0]
    control_path = arm.debug_dir / "client" / ek / "controls_0000.npz"
    control = reader.read_npz(control_path)
    seq = int(str(data["decision_id"][0]).rsplit(":", 1)[1])
    position = np.flatnonzero(control["decision_seq"] == seq)[0]
    control["action"][position, 0] = value
    schema.write_npz_block(control_path, control)
    receipt_path = arm.debug_dir / "receipts" / ek / "complete.json"
    receipt = reader.read_json(receipt_path)
    for entry in receipt["files"]:
        if entry["name"] == control_path.name:
            entry.update(bytes=control_path.stat().st_size, sha256=sha256(control_path))
    receipt_path.write_text(json.dumps(receipt))
    assert validate_arm(arm, 6)["status"] == "PASS"
    data["served_wire"] = data["served_wire"].astype(np.float32)
    schema.write_npz_block(block, data)
    assert {"wire_action_dtype", "issued_action"} <= codes(validate_arm(arm, 6))
