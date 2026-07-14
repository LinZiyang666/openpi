"""Tests for the Phase-6 weights/artifact/YAML SHA binding (plan §6.3d)."""

import pytest

from exp.zixuan_proposal.phase6_provenance import (
    assert_binding,
    assert_recorded_digest,
    assert_serve_binding,
    record_weights_digest,
    weights_sha256,
)


def test_sha_matches_and_binding_ok(tmp_path):
    w = tmp_path / "w.pt"
    w.write_bytes(b"trained-projection-weights")
    same = tmp_path / "w_copy.pt"
    same.write_bytes(b"trained-projection-weights")
    assert weights_sha256(w) == weights_sha256(same)
    assert assert_binding(w, same) == weights_sha256(w)  # same content -> bound


def test_binding_rejects_mismatch(tmp_path):
    a = tmp_path / "a.pt"
    a.write_bytes(b"lane-B-weights")
    b = tmp_path / "b.pt"
    b.write_bytes(b"lane-C-weights")  # different head -> mixed space
    with pytest.raises(ValueError, match="mismatch"):
        assert_binding(a, b)


def test_missing_weights_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        weights_sha256(tmp_path / "nope.pt")


def test_record_and_assert_immutable_digest(tmp_path):
    w = tmp_path / "laneB.pt"
    w.write_bytes(b"trained-weights-v1")
    art = {"projection_params": {"projection_weights_path": str(w)}}
    record_weights_digest(art, w)
    assert art["projection_params"]["projection_weights_sha256"] == weights_sha256(w)
    assert assert_recorded_digest(art, w) == weights_sha256(w)  # live bytes match recorded


def test_recorded_digest_detects_post_build_weight_swap(tmp_path):
    w = tmp_path / "laneB.pt"
    w.write_bytes(b"trained-weights-v1")
    art = {"projection_params": {"projection_weights_path": str(w)}}
    record_weights_digest(art, w)
    w.write_bytes(b"swapped-weights-v2")  # someone repoints the file after build
    with pytest.raises(ValueError, match="digest drift"):
        assert_recorded_digest(art, w)


def test_recorded_digest_requires_stamp(tmp_path):
    w = tmp_path / "laneB.pt"
    w.write_bytes(b"x")
    with pytest.raises(ValueError, match="immutable projection_weights_sha256"):
        assert_recorded_digest({"projection_params": {"projection_weights_path": str(w)}}, w)


def test_serve_binding_enforces_full_chain(tmp_path):
    w = tmp_path / "laneB.pt"
    w.write_bytes(b"trained")
    art = {"projection_params": {"projection_weights_path": str(w)}}
    record_weights_digest(art, w)
    yaml_cfg = {"key_builder": {"type": "projection", "projection": {"weights_path": str(w)}}}
    assert assert_serve_binding(art, yaml_cfg, w) == weights_sha256(w)
    # YAML points at a different head -> serve aborts
    other = tmp_path / "laneC.pt"
    other.write_bytes(b"different")
    with pytest.raises(ValueError, match="mismatch"):
        assert_serve_binding(art, {"key_builder": {"type": "projection", "projection": {"weights_path": str(other)}}}, w)
