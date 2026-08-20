"""Ledger format rules and the shipped records that must satisfy them."""

from __future__ import annotations

import json

import pytest

from exp.data_authority.registry import (
    RECORDS_DIR,
    SCHEMA_VERSION,
    find,
    load_all,
    load_record,
    record_path_for,
    validate_record,
)


def _minimal() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "demo_exp/libero_spatial/thing",
        "kind": "cache_artifact",
        "title": "demo",
        "experiment": "exp/demo_exp",
        "suite": "libero_spatial",
        "status": "authoritative",
        "authority": {"node": "local", "path": "exp/demo/x.pkl", "access": "local"},
        "integrity": {"sha256": "a" * 64, "size_bytes": 10, "file_count": 1},
        "content": {},
        "provenance": {
            "produced_by": "x.py",
            "measured_at": "2026-08-20",
            "measured_by": "sha256sum",
        },
    }


# ------------------------------------------------------------------
# Shipped records
# ------------------------------------------------------------------


def test_every_shipped_record_validates():
    for path in sorted(RECORDS_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        problems = validate_record(
            json.loads(path.read_text(encoding="utf-8")), filename=path.name
        )
        assert problems == [], f"{path.name}: {problems}"


def test_shipped_dataset_ids_are_unique_and_round_trip():
    recs = load_all()
    ids = [r["dataset_id"] for r in recs]
    assert len(ids) == len(set(ids))
    for rec in recs:
        assert (
            load_record(record_path_for(rec["dataset_id"]))["dataset_id"]
            == rec["dataset_id"]
        )


def test_find_filters_are_conjunctive():
    assert {
        r["dataset_id"] for r in find(experiment="weighted_sum", suite="libero_10")
    } == {"weighted_sum/libero_10/cp1_spatial_pool_16"}
    assert find(experiment="weighted_sum", suite="no_such_suite") == []


# ------------------------------------------------------------------
# Validation rules
# ------------------------------------------------------------------


def test_minimal_record_is_valid():
    assert validate_record(_minimal()) == []


@pytest.mark.parametrize(
    "key", ["schema_version", "dataset_id", "integrity", "provenance"]
)
def test_missing_required_key_is_reported(key):
    rec = _minimal()
    del rec[key]
    assert any(key in p for p in validate_record(rec))


@pytest.mark.parametrize("sha", ["", "z" * 64, "a" * 63, "A" * 64, 123])
def test_bad_sha256_is_rejected(sha):
    rec = _minimal()
    rec["integrity"]["sha256"] = sha
    assert any("sha256" in p for p in validate_record(rec))


@pytest.mark.parametrize("size", [0, -1, "10", 1.5, True])
def test_non_positive_int_sizes_are_rejected(size):
    rec = _minimal()
    rec["integrity"]["size_bytes"] = size
    assert any("size_bytes" in p for p in validate_record(rec))


def test_remote_record_requires_absolute_path():
    rec = _minimal()
    rec["authority"] = {
        "node": "weilandserver",
        "path": "relative/x.pkl",
        "access": "tether",
    }
    assert any("absolute" in p for p in validate_record(rec))


def test_filename_must_encode_dataset_id():
    rec = _minimal()
    assert validate_record(rec, filename="demo_exp__libero_spatial__thing.json") == []
    assert any("filename" in p for p in validate_record(rec, filename="wrong.json"))


def test_unknown_kind_and_status_are_rejected():
    rec = _minimal()
    rec["kind"] = "mystery"
    rec["status"] = "maybe"
    problems = validate_record(rec)
    assert any("kind" in p for p in problems)
    assert any("status" in p for p in problems)
