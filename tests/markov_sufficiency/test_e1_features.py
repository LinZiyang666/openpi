"""Tests for E1 feature construction and episode-level aggregation.

Asserts the two invariants the design rests on: group B must degenerate to
group A at depth 1 (otherwise B's "history effect" is partly an implementation
difference), and the inference unit must be the episode, not the step.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from exp.markov_sufficiency import _library, _scoring
from exp.markov_sufficiency import e1_loeo_residual as e1

REPO = pathlib.Path(__file__).resolve().parents[2]
LIBRARY = REPO / "exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl"
YAML = REPO / (
    "exp/weighted_sum/config/trajectory/libero_spatial/"
    "cp1_spatial_pool_16__grid3_vision_0@6_vision_1@44_robot_state@50__d3.yaml"
)

needs_artifacts = pytest.mark.skipif(
    not (LIBRARY.exists() and YAML.exists()), reason="requires the local cache artifact and eval yaml"
)


# ------------------------------------------------------------------
# Feature invariants
# ------------------------------------------------------------------


@needs_artifacts
def test_group_b_degenerates_to_group_a_at_depth_one():
    lib = _library.load_library(LIBRARY)
    scorer = _scoring.build_scorer(YAML)
    entry = lib.entries[30]
    cands = [c for c in scorer.candidates(lib.entries, entry.payload.task_key) if c.trajectory_id != entry.trajectory_id]

    single = scorer.score_batch(entry.query_keys, cands, cache={})
    traj = np.array([scorer.score_trajectory([entry.query_keys], [c], [1.0]) for c in cands])
    np.testing.assert_allclose(traj, single, rtol=1e-4, atol=1e-5)


@needs_artifacts
def test_history_frames_are_newest_first_and_padded():
    lib = _library.load_library(LIBRARY)
    entry = lib.by_traj[sorted(lib.by_traj)[0]][0]  # first step of an episode
    frames = e1._history_frames(lib, entry, 3)
    assert frames[0] is entry.query_keys
    assert frames[1] is None and frames[2] is None


@needs_artifacts
def test_progress_is_normalised_within_the_episode():
    lib = _library.load_library(LIBRARY)
    traj_id = sorted(lib.by_traj)[0]
    items = lib.by_traj[traj_id]
    assert e1._progress(lib, items[0]) == pytest.approx(0.0)
    assert e1._progress(lib, items[-1]) == pytest.approx(1.0)
    # Progress is monotone, which is what makes it a usable phase oracle.
    values = [e1._progress(lib, e) for e in items]
    assert values == sorted(values)


# ------------------------------------------------------------------
# Aggregation: the episode is the inference unit
# ------------------------------------------------------------------


def _rows(n_ep=12, per_ep=8, delta=0.0, padding_every=None):
    rows = []
    rng = np.random.default_rng(0)
    for ep in range(n_ep):
        for step in range(per_ep):
            pad = padding_every is not None and step < padding_every
            base = float(rng.uniform(0.5, 1.5))
            rows.append({"trajectory_id": f"ep{ep}", "step_idx": step, "padding": pad, "k": 1, "group": "A", "residual": base})
            rows.append({"trajectory_id": f"ep{ep}", "step_idx": step, "padding": pad, "k": 1, "group": "B-d3", "residual": base - delta})
    return rows


def test_aggregate_uses_episodes_not_steps():
    rows = _rows(n_ep=12, per_ep=8)
    out = e1.aggregate(rows, "B-d3", k=1)
    # 12 episodes x 8 steps: the test must see 12 units, not 96.
    assert out["n_episodes"] == 12


def test_aggregate_detects_a_uniform_improvement():
    out = e1.aggregate(_rows(delta=0.30), "B-d3", k=1)
    assert out["hodges_lehmann"] == pytest.approx(0.30, abs=1e-6)
    assert out["relative_delta"] > 0.10
    assert out["p_value"] < 0.01


def test_aggregate_reports_no_effect_when_groups_match():
    out = e1.aggregate(_rows(delta=0.0), "B-d3", k=1)
    assert out["hodges_lehmann"] == pytest.approx(0.0, abs=1e-9)
    assert out["p_value"] > 0.05


def test_padding_steps_are_excluded_from_the_primary_analysis():
    rows = _rows(n_ep=6, per_ep=6, padding_every=2)
    kept = e1.aggregate(rows, "B-d3", k=1, include_padding=False)
    all_steps = e1.aggregate(rows, "B-d3", k=1, include_padding=True)
    assert kept["n_episodes"] == all_steps["n_episodes"] == 6
    # Different step sets feed the medians, so the baselines must differ.
    assert kept["median_residual_A"] != all_steps["median_residual_A"]


def test_aggregate_ignores_other_k_values():
    rows = _rows()
    for r in rows:
        r["k"] = 5
    assert e1.aggregate(rows, "B-d3", k=1)["n_episodes"] == 0


# ------------------------------------------------------------------
# G2 round 1: per-depth weights, per-modality C, fold-safe calibration
# ------------------------------------------------------------------

D3_YAML = REPO / (
    "exp/weighted_sum/config/trajectory/libero_spatial/"
    "cp1_spatial_pool_16__grid3_vision_0@6_vision_1@44_robot_state@50__d3.yaml"
)
D5_YAML = REPO / (
    "exp/weighted_sum/config/trajectory/libero_spatial/"
    "cp1_spatial_pool_16__grid3_vision_0@6_vision_1@44_robot_state@50__d5.yaml"
)

needs_depth_yamls = pytest.mark.skipif(
    not (D3_YAML.exists() and D5_YAML.exists() and LIBRARY.exists()),
    reason="requires the per-depth eval yamls and the cache artifact",
)


@needs_depth_yamls
def test_each_depth_carries_its_own_production_weights():
    d3 = _scoring.build_scorer(D3_YAML)
    d5 = _scoring.build_scorer(D5_YAML)
    assert d3.trajectory_weights == pytest.approx([0.5, 0.3, 0.2])
    assert d5.trajectory_weights == pytest.approx([0.35, 0.25, 0.2, 0.12, 0.08])
    # Truncating d3's vector to five slots would be a different configuration.
    assert d5.trajectory_weights[:3] != pytest.approx(d3.trajectory_weights)


@needs_depth_yamls
def test_run_suite_rejects_a_depth_with_mismatched_weights():
    """Passing the d3 scorer as the d5 entry must fail loudly, not truncate."""
    d3 = _scoring.build_scorer(D3_YAML)
    with pytest.raises(ValueError, match="production trajectory weights"):
        e1.run_suite(
            "libero_spatial",
            "cp1_spatial_pool_16",
            {5: d3},
            library_root=str(REPO / "exp/common/data/cache_artifacts"),
            max_episodes=1,
        )


@needs_depth_yamls
def test_group_c_is_per_modality_and_gamma_changes_the_ranking():
    lib = _library.load_library(LIBRARY)
    scorer = _scoring.build_scorer(D3_YAML)
    held = sorted(lib.by_traj)[0]
    entry = lib.by_traj[held][4]
    pool = [e for e in lib.entries if e.trajectory_id != held]
    cands = [e for e in pool if e.payload.task_key == entry.payload.task_key]
    frames = e1._history_frames(lib, entry, 3)
    norms = e1.calibrate_diff_normalizers(scorer, lib, pool, held, n_pairs=40)

    # Every active modality participates, not just the first one.
    assert set(norms) == {f for f, _, _ in scorer.active_fields}

    base = scorer.score_batch(entry.query_keys, cands, cache={})
    lo = e1.score_group_c(scorer, lib, frames, cands, norms, 0.5, base)
    hi = e1.score_group_c(scorer, lib, frames, cands, norms, 1.0, base)
    assert not np.allclose(lo, hi), "gamma must change the group-C scores"


@needs_depth_yamls
def test_diff_calibration_never_uses_held_out_values():
    lib = _library.load_library(LIBRARY)
    scorer = _scoring.build_scorer(D3_YAML)
    held = sorted(lib.by_traj)[0]
    pool = [e for e in lib.entries if e.trajectory_id != held]
    norms = e1.calibrate_diff_normalizers(scorer, lib, pool, held, n_pairs=40)
    for norm in norms.values():
        assert held not in norm.fit_trajectories


def test_fit_diff_normalizer_rejects_values_sourced_from_held_out():
    """Name-level checks are not enough; the *values* must be library-side."""
    with pytest.raises(ValueError, match="calibration value"):
        _scoring.fit_diff_normalizer(
            [0.1, 0.2], ["ep_a"], held_out_trajectory="ep_held",
            source_trajectories=["ep_a", "ep_held"],
        )


def test_knn_aggregation_is_shared_by_all_groups():
    scores = np.array([0.9, 0.1, 0.5])

    class _E:
        def __init__(self, v):
            self.v = v

    cands = [_E(0.0), _E(1.0), _E(2.0)]
    chain = None
    import exp.markov_sufficiency._library as lib_mod

    orig = lib_mod.executed_action
    lib_mod.executed_action = lambda e, out_chain: np.array([e.v])
    try:
        top1 = e1._knn_predict(scores, cands, chain, 1)
        top3 = e1._knn_predict(scores, cands, chain, 3)
    finally:
        lib_mod.executed_action = orig
    assert top1 == pytest.approx([0.0])
    # Similarity-weighted, not a plain mean: (0*0.9 + 1*0.1 + 2*0.5)/1.5.
    assert top3 == pytest.approx([(0 * 0.9 + 1 * 0.1 + 2 * 0.5) / 1.5])


def test_family_analysis_applies_holm_over_four_cells():
    rows = _rows(n_ep=20, per_ep=6, delta=0.25)
    family = e1.family_analysis({"libero_spatial": rows, "libero_10": rows})
    assert len(family["cells"]) == 4
    for cell in family["cells"]:
        assert "holm_rejected" in cell and "holm_ci" in cell
        assert cell["holm_level"] >= 0.95
        assert cell["verdict"] in ("history_helps", "equivalent_no_history_value", "inconclusive")


def test_family_verdict_needs_both_holm_and_the_effect_floor():
    tiny = _rows(n_ep=20, per_ep=6, delta=0.001)  # significant but negligible
    family = e1.family_analysis({"libero_spatial": tiny, "libero_10": tiny})
    assert all(c["verdict"] != "history_helps" for c in family["cells"])


def test_dose_response_uses_disjoint_folds():
    builders = {f"kb{i}": _rows(n_ep=12, per_ep=5, delta=0.02 * i) for i in range(4)}
    out = e1.dose_response(builders, "B-d3")
    assert out["n"] == 4
    assert -1.0 <= out["spearman"] <= 1.0


def test_hodges_lehmann_is_the_walsh_average_median():
    from exp.markov_sufficiency import _stats as st

    assert st.hodges_lehmann([0, 2, 10]) == pytest.approx(3.5)
    # The plain median of the same sample is a different statistic.
    assert float(np.median([0, 2, 10])) == pytest.approx(2.0)


def test_aggregate_reports_hl_and_a_permutation_interval():
    rows = _rows(n_ep=15, per_ep=5, delta=0.2)
    out = e1.aggregate(rows, "B-d3", k=1)
    assert out["hodges_lehmann"] == pytest.approx(0.2, abs=1e-6)
    lo, hi = out["hl_ci"]
    assert lo <= out["hodges_lehmann"] <= hi


def test_pilot_power_flags_an_underpowered_cell():
    noisy = []
    rng = np.random.default_rng(0)
    for ep in range(6):
        for step in range(4):
            base = float(rng.uniform(0.5, 1.5))
            noisy.append({"trajectory_id": f"ep{ep}", "step_idx": step, "padding": False, "k": 1,
                          "group": "A", "residual": base})
            noisy.append({"trajectory_id": f"ep{ep}", "step_idx": step, "padding": False, "k": 1,
                          "group": "B-d3", "residual": base + float(rng.normal(0, 0.8))})
    power = e1.pilot_power(noisy, "B-d3")
    assert power["underpowered"] is True


def test_family_downgrades_a_cell_when_the_pilot_is_underpowered():
    rows = _rows(n_ep=20, per_ep=6, delta=0.25)
    family = e1.family_analysis(
        {"libero_spatial": rows, "libero_10": rows},
        pilot_power_by_cell={("libero_spatial", "B-d3"): {"underpowered": True, "power": 0.4}},
    )
    downgraded = [c for c in family["cells"] if c["suite"] == "libero_spatial" and c["group"] == "B-d3"]
    assert downgraded and downgraded[0]["verdict"] == "estimation_only_underpowered"


# ------------------------------------------------------------------
# Registered driver (G2 round 5): pilot power + two-suite family + artifacts
# ------------------------------------------------------------------


def test_driver_requires_both_suites():
    """A single suite has no Holm family, so it may not produce a verdict."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "exp.markov_sufficiency.e1_loeo_residual",
         "--suite-yaml", "libero_spatial:3=x.yaml", "--out-dir", "/tmp/does_not_matter"],
        capture_output=True, text=True, cwd=str(REPO), env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode != 0
    assert "two-suite" in (proc.stdout + proc.stderr)


def test_run_family_writes_artifacts_and_downgrades_underpowered_cells(tmp_path, monkeypatch):
    """The driver must compute pilot power itself and gate the verdict on it."""
    rng = np.random.default_rng(0)

    def fake_run_suite(suite, key_builder, scorers, library_root=None, max_episodes=None):
        rows = []
        for ep in range(6):
            for step in range(4):
                base = float(rng.uniform(0.5, 1.5))
                for group, val in (("A", base), ("B-d3", base + float(rng.normal(0, 0.9))),
                                   ("C-g1.0", base + float(rng.normal(0, 0.9)))):
                    rows.append({"suite": suite, "key_builder": key_builder, "trajectory_id": f"{suite}-ep{ep}",
                                 "step_idx": step, "task_key": "t", "padding": False,
                                 "group": group, "k": 1, "residual": val})
        return {"rows": rows, "manifest": {"suite": suite, "n_rows": len(rows)}}

    monkeypatch.setattr(e1, "run_suite", fake_run_suite)
    monkeypatch.setattr(_scoring, "build_scorer", lambda path: None)

    out = e1.run_family(
        {"libero_spatial": {3: "a.yaml"}, "libero_10": {3: "b.yaml"}},
        tmp_path,
    )
    # Artifacts the plan registers.
    assert pathlib.Path(out["manifest_path"]).exists()
    assert pathlib.Path(out["family_path"]).exists()
    for path in out["manifest"]["row_artifacts"].values():
        assert pathlib.Path(path).exists()
    # Pilot power is computed for all four primary cells.
    assert len(out["manifest"]["pilot_power"]) == 4
    # Pure noise: every cell is underpowered, so no binary verdict may survive.
    assert all(c["verdict"] == "estimation_only_underpowered" for c in out["family"]["cells"])
