"""Collected-analysis rules: the manifest, the copy semantics, and the three failure kinds."""

from __future__ import annotations

import json

import pytest

from exp.data_authority.collect import collect, refresh
from exp.data_authority.registry import (
    ANALYSIS_DIR,
    MANIFEST_NAME,
    analysis_tasks,
    load_all_analysis,
    load_analysis_manifest,
    validate_analysis_manifest,
    validate_record,
)
from exp.data_authority.verify import verify_analysis_task


def _manifest(**over) -> dict:
    base = {
        "schema_version": 1,
        "task": "demo",
        "title": "demo",
        "policy": "copy",
        "collected_at": "2026-08-20",
        "files": [
            {
                "name": "a.png",
                "sha256": "a" * 64,
                "size_bytes": 5,
                "source": "exp/x/a.png",
            }
        ],
    }
    base.update(over)
    return base


def _seed(tmp_path, *, names=("a.png",), payload=b"hello"):
    """A task directory built through the real collect() path."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    specs = []
    for name in names:
        (src_dir / name).write_bytes(payload)
        specs.append(str(src_dir / name))
    collect("demo", specs, title="demo", analysis_dir=tmp_path / "analysis")
    return tmp_path / "analysis", tmp_path / "analysis" / "demo"


# ------------------------------------------------------------------
# Shipped tasks
# ------------------------------------------------------------------


def test_every_shipped_task_has_a_valid_manifest():
    tasks = analysis_tasks()
    assert tasks, "the registry ships no analysis task"
    for manifest in load_all_analysis():
        assert validate_analysis_manifest(manifest, task=manifest["task"]) == []


def test_every_shipped_task_matches_its_bytes():
    for task in analysis_tasks():
        report = verify_analysis_task(task)
        assert report["ok"], report


def test_shipped_manifests_record_a_source_for_every_file():
    for manifest in load_all_analysis():
        for entry in manifest["files"]:
            assert entry["source"], f"{manifest['task']}/{entry['name']} has no source"


def test_mandatory_task_level_is_respected():
    # Nothing may sit directly in analysis/ except task directories.
    strays = [p.name for p in ANALYSIS_DIR.iterdir() if p.is_file()]
    assert strays == []
    for task in analysis_tasks():
        assert (ANALYSIS_DIR / task / MANIFEST_NAME).is_file()


# ------------------------------------------------------------------
# Manifest validation
# ------------------------------------------------------------------


def test_valid_manifest_passes():
    assert validate_analysis_manifest(_manifest()) == []


@pytest.mark.parametrize("key", ["schema_version", "task", "policy", "files"])
def test_missing_manifest_key_is_reported(key):
    bad = _manifest()
    del bad[key]
    assert any(key in p for p in validate_analysis_manifest(bad))


def test_move_policy_is_rejected():
    assert any(
        "copy" in p for p in validate_analysis_manifest(_manifest(policy="move"))
    )


def test_task_must_match_its_directory():
    assert any(
        "does not match" in p
        for p in validate_analysis_manifest(_manifest(), task="other")
    )


def test_empty_file_list_is_rejected():
    assert any(
        "non-empty" in p for p in validate_analysis_manifest(_manifest(files=[]))
    )


def test_duplicate_names_are_rejected():
    entry = {
        "name": "a.png",
        "sha256": "a" * 64,
        "size_bytes": 5,
        "source": "exp/x/a.png",
    }
    assert any(
        "duplicated" in p
        for p in validate_analysis_manifest(_manifest(files=[entry, dict(entry)]))
    )


def test_file_without_source_is_rejected():
    entry = {"name": "a.png", "sha256": "a" * 64, "size_bytes": 5, "source": ""}
    assert any(
        "source" in p for p in validate_analysis_manifest(_manifest(files=[entry]))
    )


# ------------------------------------------------------------------
# Collection semantics
# ------------------------------------------------------------------


def test_collect_copies_and_leaves_the_source_in_place(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    original = src / "fig.png"
    original.write_bytes(b"pixels")
    collect("demo", [str(original)], title="demo", analysis_dir=tmp_path / "analysis")

    assert original.is_file(), "collection must copy, never move"
    assert (tmp_path / "analysis" / "demo" / "fig.png").read_bytes() == b"pixels"


def test_collect_honours_a_destination_subpath(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "fig.png").write_bytes(b"pixels")
    collect(
        "demo",
        [f"{src / 'fig.png'}:libero_10/fig.png"],
        title="demo",
        analysis_dir=tmp_path / "analysis",
    )
    manifest = load_analysis_manifest("demo", tmp_path / "analysis")
    assert [f["name"] for f in manifest["files"]] == ["libero_10/fig.png"]


def test_collect_refuses_to_overwrite_without_force(tmp_path):
    analysis_dir, _ = _seed(tmp_path)
    other = tmp_path / "src" / "a.png"
    with pytest.raises(FileExistsError):
        collect("demo", [str(other)], analysis_dir=analysis_dir)
    collect("demo", [str(other)], analysis_dir=analysis_dir, force=True)


def test_collect_rejects_a_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect(
            "demo", [str(tmp_path / "nope.png")], analysis_dir=tmp_path / "analysis"
        )


def test_refresh_preserves_a_hand_written_description(tmp_path):
    analysis_dir, task_dir = _seed(tmp_path)
    manifest_path = task_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["description"] = "the money plot"
    manifest_path.write_text(json.dumps(manifest))

    refreshed = refresh("demo", analysis_dir=analysis_dir)
    assert refreshed["files"][0]["description"] == "the money plot"
    assert refreshed["files"][0]["source"], "refresh must not drop the recorded source"


# ------------------------------------------------------------------
# The three failure kinds, reported apart
# ------------------------------------------------------------------


def test_deleted_file_reports_as_missing_not_mismatched(tmp_path):
    analysis_dir, task_dir = _seed(tmp_path)
    (task_dir / "a.png").unlink()
    report = verify_analysis_task("demo", analysis_dir=analysis_dir)
    assert report["ok"] is False
    assert report["missing"] == ["a.png"]
    assert report["mismatched"] == []


def test_regenerated_file_reports_as_mismatched_not_missing(tmp_path):
    analysis_dir, task_dir = _seed(tmp_path)
    (task_dir / "a.png").write_bytes(b"redrawn")
    report = verify_analysis_task("demo", analysis_dir=analysis_dir)
    assert report["ok"] is False
    assert report["missing"] == []
    assert [m["name"] for m in report["mismatched"]] == ["a.png"]


def test_unregistered_file_is_surfaced(tmp_path):
    analysis_dir, task_dir = _seed(tmp_path)
    (task_dir / "orphan.png").write_bytes(b"who put this here")
    report = verify_analysis_task("demo", analysis_dir=analysis_dir)
    assert report["ok"] is False
    assert report["unregistered"] == ["orphan.png"]


# ------------------------------------------------------------------
# Dataset -> analysis join
# ------------------------------------------------------------------


def test_dangling_analysis_task_is_rejected(tmp_path):
    rec = {
        "schema_version": 1,
        "dataset_id": "demo_exp/libero_spatial/thing",
        "kind": "cache_artifact",
        "title": "demo",
        "experiment": "exp/demo_exp",
        "status": "authoritative",
        "analysis_task": "no_such_task",
        "authority": {"node": "local", "path": "exp/demo/x.pkl", "access": "local"},
        "integrity": {"sha256": "a" * 64, "size_bytes": 10, "file_count": 1},
        "content": {},
        "provenance": {
            "produced_by": "x.py",
            "measured_at": "2026-08-20",
            "measured_by": "sha256sum",
        },
    }
    assert any("no_such_task" in p for p in validate_record(rec, analysis_dir=tmp_path))

    _seed(tmp_path)
    rec["analysis_task"] = "demo"
    assert validate_record(rec, analysis_dir=tmp_path / "analysis") == []
