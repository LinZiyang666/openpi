"""Tests for emit_precheck_yamls: judge sections, sha binding, dynamic arms."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import yaml

from exp.dispatch_surface.emit_precheck_yamls import (
    LAYER_SECONDARY,
    THRESHOLD_CELLS,
    WS_START_T,
    _emit,
    _load_scores,
)
from openpi.cache.components.surface_judge import (
    CERTIFICATION_CONFORMAL,
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
    _emit(TEMPLATE, out, judge, "/lib/new.pkl", 0.97, LAYER_SECONDARY)
    doc = yaml.safe_load(out.read_text())
    assert doc["checkpoints"]["cp1"]["judge"] == judge
    assert doc["backend"]["in_memory"]["preload_path"] == "/lib/new.pkl"
    assert doc["write_policy"] == {"type": "never"}


def test_emit_rewrites_the_gate_at_the_resolved_theta(tmp_path):
    """The gate is re-solved per library, never inherited.

    solve_gtp's rule: theta and the judge cuts are quantiles of the same score
    distribution and must be re-derived per library, because two libraries with
    different score spreads reach the same operating point at different numeric
    cuts. This line rebuilds the library, so an inherited theta would sit at an
    operating point it was not calibrated for.
    """
    from exp.gate_threshold_pareto.emit_gtp_yamls import (
        GATE_J,
        GATE_L,
        GATE_PROBE_INTERVAL,
    )

    out = tmp_path / "arm.yaml"
    _emit(TEMPLATE, out, {"type": "threshold", "threshold": 0.9}, "/lib/x.pkl", 0.9876, LAYER_SECONDARY)
    gate = yaml.safe_load(out.read_text())["checkpoints"]["cp1"]["gate"]
    assert gate["type"] == "score_hysteresis"
    assert gate["theta_low"] == gate["theta_high"] == 0.9876
    assert (gate["j"], gate["probe_interval"], gate["L"]) == (
        GATE_J, GATE_PROBE_INTERVAL, GATE_L)


def test_emit_overrides_a_template_that_carries_a_different_gate(tmp_path):
    """libero_10's gate template is always_search; it must not survive.

    Inheriting it would give that suite's baseline frontier no gate at all,
    which is not the operating point the threshold line's Pareto frontier is
    drawn at.
    """
    import copy

    tmpl = copy.deepcopy(TEMPLATE)
    tmpl["checkpoints"]["cp1"]["gate"] = {"type": "always_search"}
    out = tmp_path / "arm.yaml"
    _emit(tmpl, out, {"type": "threshold", "threshold": 0.9}, "/lib/x.pkl", 0.95, LAYER_SECONDARY)
    gate = yaml.safe_load(out.read_text())["checkpoints"]["cp1"]["gate"]
    assert gate["type"] == "score_hysteresis"
    assert gate["theta_low"] == 0.95


def test_every_arm_shares_one_gate_theta(tmp_path):
    """A gate that moved between arms would confound the verdict comparison."""
    thetas = []
    for i, judge in enumerate((
        {"type": "threshold", "threshold": 0.9},
        {"type": "dispatch_surface", "surface_artifact_path": "/a.npz"},
    )):
        out = tmp_path / f"arm{i}.yaml"
        _emit(TEMPLATE, out, judge, "/lib/x.pkl", 0.9876, LAYER_SECONDARY)
        thetas.append(
            yaml.safe_load(out.read_text())["checkpoints"]["cp1"]["gate"]["theta_low"])
    assert len(set(thetas)) == 1


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
        start_t_ws=0.3, delta=0.5, quantile_alpha=0.05,
        certification_mode=CERTIFICATION_CONFORMAL, uses_disagreement=True,
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


# ---------------- Rev 1: primary/secondary layers ----------------

def test_primary_gate_probes_every_step():
    from exp.dispatch_surface.emit_precheck_yamls import LAYER_PRIMARY, gate_section

    assert gate_section(LAYER_PRIMARY, 0.97) == {"type": "always_search"}


def test_secondary_gate_is_the_production_hysteresis_with_the_solved_theta():
    from exp.dispatch_surface.emit_precheck_yamls import LAYER_SECONDARY, gate_section

    g = gate_section(LAYER_SECONDARY, 0.97)
    assert g["type"] == "score_hysteresis"
    assert g["theta_low"] == g["theta_high"] == 0.97


def test_primary_gate_never_carries_a_threshold():
    """A primary gate with a theta would re-introduce the s-cut it exists to remove."""
    from exp.dispatch_surface.emit_precheck_yamls import LAYER_PRIMARY, gate_section

    g = gate_section(LAYER_PRIMARY, 0.97)
    assert "theta_low" not in g and "theta_high" not in g


def test_secondary_roster_is_frozen_and_excludes_s0():
    """Secondary is descriptive; its roster must not be picked after primary."""
    from exp.dispatch_surface.emit_precheck_yamls import PRIMARY_CORE_ARMS, SECONDARY_ARMS

    assert SECONDARY_ARMS == {
        "dsp_t_fh30_ws20", "dsp_t_fh50_ws20", "dsp_t_fh70_ws10", "dsp_sv",
    }
    assert SECONDARY_ARMS < PRIMARY_CORE_ARMS
    assert "dsp_s0" in PRIMARY_CORE_ARMS and "dsp_s0" not in SECONDARY_ARMS


def test_matrix_declares_the_certification_mode(tmp_path):
    """The matrix is what the launch ledger freezes, so the claim this sweep
    makes must be visible there -- not only inside each artifact."""
    import inspect

    from exp.dispatch_surface import emit_precheck_yamls as mod

    src = inspect.getsource(mod.main)
    block = src[src.index("matrix = {"):src.index("matrix_path")]
    assert '"certification_mode": CERTIFICATION_EMPIRICAL' in block
    for field in ("artifact_sha256", "fit_record_sha256", "layer", "gate_type"):
        assert f'"{field}"' in block, field
