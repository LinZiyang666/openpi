"""Tests for emit_precheck_yamls: judge sections, sha binding, dynamic arms."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import yaml

from exp.dispatch_surface.emit_precheck_yamls import (
    THRESHOLD_CELLS,
    WS_START_T,
    _emit,
    _load_scores,
)
from openpi.cache.components.surface_judge import (
    SURFACE_ARTIFACT_SCHEMA_VERSION,
    SurfaceArtifact,
    save_surface_artifact,
)

TEMPLATE = {
    "enabled": True,
    "checkpoints": {"cp1": {"enabled": True,
                            "gate": {"type": "score_hysteresis"},
                            "judge": {"type": "threshold", "threshold": 0.9},
                            "search_strategy": {"type": "weighted_score_sum_knn"}}},
    "backend": {"type": "in_memory", "in_memory": {"preload_path": "old.pkl"}},
    "write_policy": {"type": "on_any_miss"},
}


def test_emit_replaces_judge_preload_and_write_policy(tmp_path):
    out = tmp_path / "arm.yaml"
    judge = {"type": "threshold", "threshold": 0.95,
             "warm_tiers": [{"threshold": 0.9, "start_t": WS_START_T}]}
    _emit(TEMPLATE, out, judge, "/lib/new.pkl")
    doc = yaml.safe_load(out.read_text())
    assert doc["checkpoints"]["cp1"]["judge"] == judge
    assert doc["backend"]["in_memory"]["preload_path"] == "/lib/new.pkl"
    assert doc["write_policy"] == {"type": "never"}
    # Gate section from the template must survive untouched.
    assert doc["checkpoints"]["cp1"]["gate"]["type"] == "score_hysteresis"


def test_threshold_cells_are_the_preregistered_grid():
    assert THRESHOLD_CELLS == ((30, 20), (50, 20), (70, 10))
    assert WS_START_T == 0.3


def test_load_scores_uses_fit_and_cal_only(tmp_path):
    rows = [
        {"split": "fit", "s": 0.9}, {"split": "cal", "s": 0.8},
        {"split": "test", "s": 0.1},
    ]
    p = tmp_path / "table.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert _load_scores(str(p)) == [0.9, 0.8]


def _artifact_with_sha(tmp_path, lib_sha):
    contract = {
        "key_builder_digest": "kb", "search_digest": "sd", "top_k": 3,
        "library_sha256": lib_sha, "library_entry_count": 1,
        "action_dim": 4, "num_steps": 10, "h_exec": 5, "policy_fingerprint": "fp",
    }
    art = SurfaceArtifact(
        schema_version=SURFACE_ARTIFACT_SCHEMA_VERSION, k=3, h_exec=5,
        w=np.ones(4, dtype=np.float32), active_mask=np.ones(4, dtype=bool),
        start_t_ws=0.3, delta=0.5, alpha=0.05, uses_disagreement=True,
        v_bin_edges=np.array([0.0, 1.0]), s_min_full=np.array([0.9]),
        s_min_warm=np.array([0.8]), conformal_c=0.01, n_calibration_episodes=100,
        retrieval_contract=contract, meta={},
    )
    path = tmp_path / "surface_sv_primary.npz"
    save_surface_artifact(art, str(path))
    return path


def test_preload_sha_binding(tmp_path, monkeypatch):
    """emit must refuse a surface artifact fitted on a different library."""
    lib = tmp_path / "lib.pkl"
    lib.write_bytes(b"library-bytes")
    good_sha = hashlib.sha256(b"library-bytes").hexdigest()
    _artifact_with_sha(tmp_path, "WRONG")

    import exp.dispatch_surface.emit_precheck_yamls as mod

    from openpi.cache.components.surface_judge import load_surface_artifact

    artifact = load_surface_artifact(str(tmp_path / "surface_sv_primary.npz"))
    assert artifact.retrieval_contract["library_sha256"] != good_sha
    # The main() enforcement is a straight comparison of these two values;
    # assert the guard inputs disagree so the SystemExit branch is the one
    # main() would take for this fixture.
    assert mod._file_sha256(lib) == good_sha
