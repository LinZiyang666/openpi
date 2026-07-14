"""Tests for the projected-lane emit / filter / diagnostic consumers (plan §6.3-§6.6)."""

import pathlib

import pytest

from exp.zixuan_proposal.phase6_emit import (
    emit_projection_config,
    ical_filter,
    ical_only_entries,
)
from exp.zixuan_proposal.phase6_provenance import assert_artifact_yaml_binding

_REPO = pathlib.Path(__file__).resolve().parents[2]
_TEMPLATE = _REPO / "exp/zixuan_proposal/config/dual_retrieval_projection_l10.yaml"


def test_emit_fills_all_placeholders_and_is_loadable():
    import yaml

    out = emit_projection_config(
        _TEMPLATE.read_text(),
        weights_path="w/laneB.pt",
        preload_path="art/projB_ical_dual.pkl",
        normalizers={
            "vision_0": {"mu": 0.5, "sigma": 0.1},
            "vision_1": {"mu": 0.4, "sigma": 0.2},
            "robot_state": {"mu": -1.0, "sigma": 0.9},
        },
        betas={"b0": -0.3, "b3": 0.7},
    )
    assert "__FILL_AT_EXECUTION__" not in out
    cfg = yaml.safe_load(out)  # loadable
    assert cfg["key_builder"]["projection"]["weights_path"] == "w/laneB.pt"
    assert cfg["backend"]["in_memory"]["preload_path"] == "art/projB_ical_dual.pkl"
    assert cfg["checkpoints"]["cp1"]["judge"]["gate_betas"]["b0"] == -0.3


def test_emit_raises_on_partial_fill():
    with pytest.raises((ValueError, KeyError)):
        emit_projection_config(
            _TEMPLATE.read_text(),
            weights_path="w.pt",
            preload_path="p.pkl",
            normalizers={"vision_0": {"mu": 0.5, "sigma": 0.1}},  # missing vision_1/robot_state
            betas={"b0": -0.3, "b3": 0.7},
        )


def test_ical_only_filter_zero_odd():
    entries = [("a", (0, 0)), ("b", (0, 2)), ("c", (0, 1)), ("d", (1, 4))]  # c is odd
    kept = ical_only_entries(entries, ident_fn=lambda e: e[1])
    assert {e[0] for e in kept} == {"a", "b", "d"}  # odd 'c' dropped


def test_ical_filter_manifest_and_exhaustive_partition():
    entries = [("a", (0, 0)), ("b", (0, 2)), ("c", (0, 1)), ("d", None), ("e", (1, 3))]
    kept, manifest = ical_filter(entries, ident_fn=lambda e: e[1])
    assert {e[0] for e in kept} == {"a", "b"}
    assert manifest["n_total"] == 5 and manifest["n_kept"] == 2
    assert manifest["n_odd_dropped"] == 2 and manifest["n_unresolved_dropped"] == 1
    assert manifest["odd_idents"] == [(0, 1), (1, 3)]
    # exhaustive: kept + odd + unresolved == total
    assert manifest["n_kept"] + manifest["n_odd_dropped"] + manifest["n_unresolved_dropped"] == manifest["n_total"]


def test_ical_filter_resolves_each_entry_once():
    # A non-deterministic resolver would break the OLD twice-called check; the new single-scan
    # partition must call ident_fn EXACTLY once per entry (no double eval to disagree on).
    calls = {}

    def counting_ident(e):
        calls[e] = calls.get(e, 0) + 1
        return (0, 0) if e in ("a", "b") else (0, 1)

    entries = ["a", "b", "c"]
    kept, manifest = ical_filter(entries, ident_fn=counting_ident)
    assert {e for e in kept} == {"a", "b"}
    assert all(v == 1 for v in calls.values())  # each entry resolved exactly once
    assert manifest["n_odd_dropped"] == 1


# The Retrieval@K top-1-prefix diagnostic now runs the REAL DualRetrievalKnnStrategy at K=1/K=5;
# see tests/exp/test_phase6_assemble.py::test_real_strategy_topk_prefix_holds (a re-sort of a
# caller-supplied score list would be tautological, so it is no longer implemented in emit).


def test_artifact_yaml_sha_binding(tmp_path):
    w = tmp_path / "laneB.pt"
    w.write_bytes(b"trained")
    artifact = {"projection_params": {"projection_weights_path": str(w)}}
    yaml_cfg = {"key_builder": {"type": "projection", "projection": {"weights_path": str(w)}}}
    assert assert_artifact_yaml_binding(artifact, yaml_cfg)  # same file -> bound

    other = tmp_path / "laneC.pt"
    other.write_bytes(b"different")
    yaml_bad = {"key_builder": {"type": "projection", "projection": {"weights_path": str(other)}}}
    with pytest.raises(ValueError, match="mismatch"):
        assert_artifact_yaml_binding(artifact, yaml_bad)

    with pytest.raises(ValueError, match="placeholder|unset"):
        assert_artifact_yaml_binding(
            artifact, {"key_builder": {"type": "projection", "projection": {"weights_path": "__FILL_AT_EXECUTION__"}}}
        )
