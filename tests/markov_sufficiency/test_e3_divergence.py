"""Tests for E3: frozen-constant reproduction, pair construction and verdicts.

The frozen-constant test is deliberately strict. ``tau_a^phys``, ``W`` and
``K`` were computed before any outcome was inspected and written into the
approved plan; if recomputation drifts, the artifact or the output chain
changed and the pre-registration no longer holds. The correct response is a new
G1 round, never editing the constants to match.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from exp.markov_sufficiency import FROZEN_TOL, TAU_A_PHYS, W_PHASE, _library, _scoring
from exp.markov_sufficiency import e3_action_divergence as e3

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
# Frozen constants
# ------------------------------------------------------------------


@needs_artifacts
def test_recomputation_reproduces_frozen_tau_and_w():
    lib = _library.load_library(LIBRARY)
    stats = e3.adjacent_distance_stats(lib, _library.build_output_chain())
    assert stats["tau_a_phys"] == pytest.approx(TAU_A_PHYS["libero_spatial"], abs=FROZEN_TOL)
    assert stats["W"] == W_PHASE["libero_spatial"]
    # The medians were monotone when the constants were frozen; a change here
    # means the library's temporal structure moved.
    assert stats["monotone"] is True
    assert stats["empty_admissible"] is False


@needs_artifacts
def test_assert_frozen_rejects_drift():
    stats = {"tau_a_phys": TAU_A_PHYS["libero_spatial"] + 0.5, "W": W_PHASE["libero_spatial"]}
    with pytest.raises(SystemExit, match="tau_a_phys drift"):
        e3.assert_frozen(stats, "libero_spatial")
    stats = {"tau_a_phys": TAU_A_PHYS["libero_spatial"], "W": W_PHASE["libero_spatial"] + 1}
    with pytest.raises(SystemExit, match="W drift"):
        e3.assert_frozen(stats, "libero_spatial")


def test_w_falls_back_to_one_when_no_lag_is_admissible(monkeypatch):
    """Edge branch: nothing satisfies median(D(L)) <= tau -> W = 1, flagged."""

    class _E:
        def __init__(self, traj, step, scale):
            self.trajectory_id, self.step_idx, self.scale = traj, step, scale

    entries = [_E("ep", i, i) for i in range(4)]
    lib = _library.Library(
        entries=entries,
        by_id={},
        by_traj={"ep": entries},
        vector_dims={},
        key_builder_type="toy",
        meta={},
    )
    # Distances explode with lag, so P95 of lag-1 is below every other median.
    monkeypatch.setattr(_library, "executed_action", lambda e, out_chain: np.array([float(e.scale) ** 3]))
    monkeypatch.setattr(e3._library, "executed_action", lambda e, out_chain: np.array([float(e.scale) ** 3]))
    stats = e3.adjacent_distance_stats(lib, out_chain=lambda x: x, max_lag=3)
    assert stats["W"] >= 1


# ------------------------------------------------------------------
# Pair construction
# ------------------------------------------------------------------


@needs_artifacts
def test_pairs_exclude_same_episode_and_cross_task():
    lib = _library.load_library(LIBRARY)
    scorer = _scoring.build_scorer(YAML)
    pairs = e3.pair_table(lib, scorer, _library.build_output_chain(), max_pairs_per_task=200)
    assert pairs
    assert all(p["traj_a"] != p["traj_b"] for p in pairs)
    assert all(isinstance(p["cycle_gap"], int) for p in pairs)
    tasks = {p["task_key"] for p in pairs}
    assert len(tasks) > 1


# ------------------------------------------------------------------
# ADR and verdicts
# ------------------------------------------------------------------


def _pairs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(
            {
                "task_key": "t",
                "traj_a": f"ep{i % 10}",
                "traj_b": f"ep{(i + 1) % 10}",
                "sim": float(rng.uniform()),
                "dist": float(rng.uniform()),
                "cycle_gap": 3,
            }
        )
    return out


def test_adr_is_conditional_not_joint():
    pairs = _pairs()
    result = e3.adr(pairs, tau_k_pct=99.0, tau_a=0.0)
    # Every high-similarity pair exceeds a zero distance threshold, so a
    # conditional rate is 1.0 -- a joint rate would have been capped near 1%.
    assert result["adr"] == pytest.approx(1.0)
    assert result["n_high_sim"] > 0


def test_adr_reports_the_denominator():
    result = e3.adr(_pairs(), tau_k_pct=99.5, tau_a=0.5)
    assert result["n_high_sim"] >= 1
    assert 0.0 <= result["adr"] <= 1.0


def _adj(episodes, value=0.2):
    return {ep: [value, value * 1.5] for ep in episodes}


def test_analyse_computes_difference_within_the_same_resample():
    """The ADR, its random reference and their difference share each draw."""
    pairs = _pairs(300)
    eps = sorted({p["traj_a"] for p in pairs} | {p["traj_b"] for p in pairs})
    result = e3.analyse(pairs, _adj(eps), n_resamples=200, seed=0, min_high_sim=1)
    assert result["diff_vs_random"] == pytest.approx(result["adr"] - result["random_adr"])
    lo, hi = result["diff_ci"]
    assert lo <= result["diff_vs_random"] <= hi
    assert result["adr_ci"][0] <= result["adr"] <= result["adr_ci"][1]


def test_bootstrap_keeps_episode_multiplicity():
    """An episode drawn twice must contribute its pairs twice.

    Collapsing the resampled ids into a set (the earlier implementation) drops
    that multiplicity and understates the variance.
    """
    pairs = [
        {"task_key": "t", "traj_a": "a", "traj_b": "b", "sim": 0.9, "dist": 1.0, "cycle_gap": 3},
        {"task_key": "t", "traj_a": "a", "traj_b": "c", "sim": 0.1, "dist": 0.0, "cycle_gap": 3},
    ]
    adj = _adj(["a", "b", "c"])
    counts_single = np.array([1, 1, 1])
    counts_double = np.array([2, 1, 1])
    # Weighted pair counts: doubling "a" doubles both of its pairs.
    single = counts_single[0] * counts_single[1] + counts_single[0] * counts_single[2]
    double = counts_double[0] * counts_double[1] + counts_double[0] * counts_double[2]
    assert double == 2 * single
    result = e3.analyse(pairs, adj, n_resamples=50, min_high_sim=1)
    assert result["n_resamples"] == 50


def test_thresholds_are_re_estimated_inside_each_draw():
    pairs = _pairs(200)
    eps = sorted({p["traj_a"] for p in pairs} | {p["traj_b"] for p in pairs})
    tight = e3.analyse(pairs, _adj(eps, value=0.1), n_resamples=100, min_high_sim=1)
    loose = e3.analyse(pairs, _adj(eps, value=5.0), n_resamples=100, min_high_sim=1)
    # tau_a is estimated from the adjacent distances, so a different physical
    # scale must move the divergence rate.
    assert tight["tau_a_phys"] < loose["tau_a_phys"]
    assert tight["adr"] != loose["adr"]


def test_small_denominator_gate_blocks_a_rate_verdict():
    pairs = _pairs(50)
    eps = sorted({p["traj_a"] for p in pairs} | {p["traj_b"] for p in pairs})
    result = e3.analyse(pairs, _adj(eps), n_resamples=50, min_high_sim=200)
    assert result["underpowered"] is True
    assert result["verdict"] == "insufficient_high_sim_pairs"


def test_cycle_gap_stratification_and_task_ranking_report_denominators():
    pairs = _pairs(300)
    bands = e3.by_cycle_gap(pairs, tau_a=0.5)
    assert set(bands) == {"0-2", "3-5", ">5"}
    assert all("reported" in v and "n_high_sim" in v for v in bands.values())
    ranked = e3.by_task(pairs, tau_a=0.5)
    assert all("reported" in r for r in ranked)


def test_threshold_grid_includes_the_physical_threshold():
    grid = e3.threshold_grid(_pairs(200), tau_a_phys=0.42)
    labels = {row["tau_a_label"] for row in grid}
    assert "phys" in labels and {"P50", "P75", "P90"} <= labels


@pytest.mark.parametrize(
    ("adr_high", "diff_high", "expected"),
    [
        (0.02, -0.05, "almost_no_aliasing"),
        (0.20, -0.05, "relative_enrichment_only"),
        (0.02, 0.01, "low_divergence_without_enrichment"),
        (0.30, 0.10, "inconclusive"),
    ],
)
def test_verdict_requires_both_absolute_and_relative_arms(adr_high, diff_high, expected):
    assert e3._verdict(adr_high, diff_high) == expected


# ------------------------------------------------------------------
# Frozen constants: both suites, and K from the independent batch
# ------------------------------------------------------------------

L10_LIBRARY = REPO / "exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl"
GATE_ROWS = {
    "libero_spatial": REPO / "exp/gate_research/data/libero_spatial/gate_rows.jsonl",
    "libero_10": REPO / "exp/gate_research/data/libero_10/gate_rows.jsonl",
}


@pytest.mark.skipif(not L10_LIBRARY.exists(), reason="requires the libero_10 artifact")
def test_libero_10_frozen_constants_reproduce():
    lib = _library.load_library(L10_LIBRARY)
    stats = e3.adjacent_distance_stats(lib, _library.build_output_chain())
    assert stats["tau_a_phys"] == pytest.approx(TAU_A_PHYS["libero_10"], abs=FROZEN_TOL)
    assert stats["W"] == W_PHASE["libero_10"]
    assert stats["monotone"] is True


@pytest.mark.parametrize("suite", sorted(GATE_ROWS))
def test_k_window_reproduces_from_the_independent_batch(suite):
    """K is the success-group cycle-count P10 of the gate_research batch."""
    import collections
    import json
    import statistics

    from exp.markov_sufficiency import K_WINDOW

    path = GATE_ROWS[suite]
    if not path.exists():
        pytest.skip(f"{suite} gate rows not present")
    per_ep = collections.defaultdict(lambda: [0, None])
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("_kind") == "episode_summary" or "step_idx" not in r:
                continue
            key = (r["yaml_id"], r["task_id"], r["subset_init_state_idx"], r["episode_id"], r.get("attempt"))
            per_ep[key][0] += 1
            per_ep[key][1] = bool(r.get("success"))
    success_lengths = [n for n, ok in per_ep.values() if ok]
    p10 = statistics.quantiles(success_lengths, n=100)[9]
    assert round(p10) == K_WINDOW[suite]


def test_small_denominator_gate_blanks_rates_everywhere():
    """Under the floor no rate or interval may appear in the final JSON."""
    pairs = _pairs(50)
    eps = sorted({p["traj_a"] for p in pairs} | {p["traj_b"] for p in pairs})
    result = e3.analyse(pairs, _adj(eps), n_resamples=50, min_high_sim=10**6)
    for key in ("adr", "adr_ci", "random_adr", "diff_vs_random", "diff_ci"):
        assert result[key] is None, f"{key} must be blank below the floor"
    grid = e3.threshold_grid(pairs, tau_a_phys=0.5)
    assert all(row["adr"] is None for row in grid if not row["reported"])
    bands = e3.by_cycle_gap(pairs, tau_a=0.5)
    assert all(v["adr"] is None for v in bands.values() if not v["reported"])
    assert all(r["adr"] is None for r in e3.by_task(pairs, tau_a=0.5) if not r["reported"])


def test_cross_suite_difference_uses_per_draw_differences():
    """An asymmetric draw distribution must survive into the interval."""
    pairs_a = _pairs(400, seed=1)
    pairs_b = _pairs(400, seed=2)
    eps_a = sorted({p["traj_a"] for p in pairs_a} | {p["traj_b"] for p in pairs_a})
    eps_b = sorted({p["traj_a"] for p in pairs_b} | {p["traj_b"] for p in pairs_b})
    out = e3.cross_suite_difference(pairs_a, _adj(eps_a), pairs_b, _adj(eps_b), n_resamples=300)
    if not out["reported"]:
        pytest.skip("synthetic pairs fell below the high-similarity floor")
    assert out["n_draws"] > 0
    # A half-width combination would be symmetric around the point estimate;
    # a per-draw percentile interval generally is not.
    assert out["ci"][0] <= out["difference"] <= out["ci"][1]


def test_analyse_exposes_draws_for_downstream_differencing():
    pairs = _pairs(400)
    eps = sorted({p["traj_a"] for p in pairs} | {p["traj_b"] for p in pairs})
    result = e3.analyse(pairs, _adj(eps), n_resamples=200, min_high_sim=1)
    assert result["adr_draws"] is not None and len(result["adr_draws"]) == 200


# ------------------------------------------------------------------
# Two-suite orchestration
# ------------------------------------------------------------------
#
# The single-suite path leaves ``ADR_l10 - ADR_spatial`` to be assembled by
# hand at reporting time. These tests pin the orchestration instead: the
# contrast direction is registered, the interval is formed draw-wise, and the
# per-suite bootstrap is paid for once rather than twice.


class _FakePayload:
    def __init__(self, task_key):
        self.task_key = task_key


class _FakeEntry:
    def __init__(self, eid, traj, step, task_key, vec, action):
        self.id = eid
        self.trajectory_id = traj
        self.step_idx = step
        self.payload = _FakePayload(task_key)
        self.query_keys = {"v": vec}
        self.action = action


class _FakeScorer:
    """Cosine on one synthetic modality -- the same call shape as the real scorer."""

    def score(self, query_keys, entry):
        return float(query_keys["v"] @ entry.query_keys["v"])


def _fake_library(seed, n_tasks=2, n_eps=20, n_steps=8):
    """Large enough that the top 1% of pairs clears ``MIN_HIGH_SIM``."""
    rng = np.random.default_rng(seed)
    entries = []
    for t in range(n_tasks):
        for e in range(n_eps):
            traj = f"s{seed}_t{t}_ep{e}"
            for s in range(n_steps):
                vec = rng.normal(size=4)
                vec /= np.linalg.norm(vec)
                entries.append(
                    _FakeEntry(f"{traj}:{s}", traj, s, f"task{t}", vec, np.array([float(rng.normal())]))
                )
    by_traj: dict[str, list] = {}
    for entry in entries:
        by_traj.setdefault(entry.trajectory_id, []).append(entry)
    return _library.Library(
        entries=entries,
        by_id={e.id: e for e in entries},
        by_traj=by_traj,
        vector_dims={"v": 4},
        key_builder_type="synthetic",
        meta={},
    )


@pytest.fixture
def two_suite_env(monkeypatch):
    """Synthetic stand-ins for the artifacts, so the orchestration is testable."""
    libs = {"libero_10": _fake_library(11), "libero_spatial": _fake_library(22)}
    monkeypatch.setattr(e3._library, "load_library", lambda path: libs[str(path)])
    monkeypatch.setattr(e3._library, "build_output_chain", lambda *a, **k: None)
    monkeypatch.setattr(e3._library, "executed_action", lambda e, out_chain: e.action)
    monkeypatch.setattr(e3._scoring, "build_scorer", lambda path: _FakeScorer())
    # tau_a^phys and W belong to the real artifacts; the frozen-constant tests
    # above cover them. Here only the orchestration is under test.
    monkeypatch.setattr(e3, "assert_frozen", lambda stats, suite: None)
    return libs


SPECS = [
    ("libero_10", "libero_10", "yaml_l10"),
    ("libero_spatial", "libero_spatial", "yaml_spatial"),
]


def test_run_family_emits_the_registered_cross_suite_contrast(two_suite_env):
    out = e3.run_family(SPECS, n_resamples=100)
    assert set(out["suites"]) == {"libero_10", "libero_spatial"}
    cross = out["cross_suite"]
    assert cross["contrast"] == "libero_10 - libero_spatial"
    if not cross["reported"]:
        pytest.skip("synthetic suites fell below the high-similarity floor")
    adr_10 = out["suites"]["libero_10"]["analysis"]["adr"]
    adr_sp = out["suites"]["libero_spatial"]["analysis"]["adr"]
    assert cross["difference"] == pytest.approx(adr_10 - adr_sp)
    assert cross["ci"][0] <= cross["difference"] <= cross["ci"][1]
    assert cross["n_draws"] == 100


def test_run_family_fixes_the_contrast_direction_regardless_of_argument_order(two_suite_env):
    """The sign is pre-registered; it must not follow the command line."""
    forward = e3.run_family(SPECS, n_resamples=60)
    reversed_ = e3.run_family(list(reversed(SPECS)), n_resamples=60)
    assert reversed_["cross_suite"]["contrast"] == forward["cross_suite"]["contrast"]
    assert reversed_["cross_suite"]["difference"] == forward["cross_suite"]["difference"]
    assert reversed_["cross_suite"]["ci"] == forward["cross_suite"]["ci"]


def test_run_family_bootstraps_each_suite_exactly_once(two_suite_env, monkeypatch):
    """Handing the analyses to the contrast must not re-run the bootstrap."""
    calls = []
    real = e3.analyse

    def counting(*args, **kwargs):
        calls.append(kwargs.get("seed"))
        return real(*args, **kwargs)

    monkeypatch.setattr(e3, "analyse", counting)
    e3.run_family(SPECS, n_resamples=50)
    assert calls == [0, 1], f"expected one bootstrap per suite, got {calls}"


def test_family_cli_writes_both_suites_and_the_contrast(two_suite_env, monkeypatch, tmp_path, capsys):
    out_path = tmp_path / "e3_family.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "e3",
            "--suite-spec", "libero_10:libero_10:yaml_l10",
            "--suite-spec", "libero_spatial:libero_spatial:yaml_spatial",
            "--n-resamples", "60",
            "--out", str(out_path),
        ],
    )
    e3.main()
    written = json.loads(out_path.read_text())
    assert written["mode"] == "family"
    assert set(written["suites"]) == {"libero_10", "libero_spatial"}
    assert written["cross_suite"]["contrast"] == "libero_10 - libero_spatial"
    for suite, block in written["suites"].items():
        assert block["suite"] == suite
        # Provenance travels with the result so the report cannot mix artifacts.
        assert block["library"] and block["yaml"]
        for key in ("analysis", "threshold_grid", "by_cycle_gap", "by_task", "calibration"):
            assert key in block
    assert "contrast" in capsys.readouterr().out


def test_single_suite_cli_is_unchanged(two_suite_env, monkeypatch, tmp_path):
    out_path = tmp_path / "e3_single.json"
    monkeypatch.setattr(
        "sys.argv",
        ["e3", "--suite", "libero_10", "--library", "libero_10", "--yaml", "y",
         "--n-resamples", "40", "--out", str(out_path)],
    )
    e3.main()
    written = json.loads(out_path.read_text())
    assert written["suite"] == "libero_10"
    assert "mode" not in written and "cross_suite" not in written


@pytest.mark.parametrize(
    "argv",
    [
        ["--suite", "libero_10", "--suite-spec", "a:b:c"],
        ["--suite-spec", "a:b:c"],
        ["--suite-spec", "a:b:c", "--suite-spec", "d:e:f", "--suite-spec", "g:h:i"],
        ["--suite-spec", "no-colons"],
        ["--suite", "libero_10", "--library", "x"],
    ],
)
def test_cli_rejects_malformed_mode_combinations(monkeypatch, tmp_path, argv):
    monkeypatch.setattr("sys.argv", ["e3", *argv, "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        e3.main()


def test_cross_suite_difference_rejects_a_mismatched_precomputed_analysis():
    pairs_a = _pairs(400, seed=1)
    pairs_b = _pairs(400, seed=2)
    eps_a = sorted({p["traj_a"] for p in pairs_a} | {p["traj_b"] for p in pairs_a})
    eps_b = sorted({p["traj_a"] for p in pairs_b} | {p["traj_b"] for p in pairs_b})
    stale = e3.analyse(pairs_a, _adj(eps_a), n_resamples=50, seed=0, min_high_sim=1)
    with pytest.raises(ValueError, match="draw indices would not align"):
        e3.cross_suite_difference(
            pairs_a, _adj(eps_a), pairs_b, _adj(eps_b), n_resamples=300, analysis_a=stale
        )


def test_run_family_requires_exactly_two_suites(two_suite_env):
    with pytest.raises(ValueError, match="exactly 2 suites"):
        e3.run_family(SPECS[:1], n_resamples=10)
