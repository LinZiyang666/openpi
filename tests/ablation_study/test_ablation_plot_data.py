"""Plot pipeline provenance for the executor-substitution figures.

The figures read only ``plot_data.json``; that file is built by
``emit_plot_data.py`` from the journals, per-step exports, anchor shards and the
analyzer output. These tests pin both halves: the collector may not silently
disagree with the upstream analyzer nor drop failed episodes, and the plot may
not draw what the data file cannot back.
"""

from __future__ import annotations

import json

import pytest

from exp.ablation_study.analysis.emit_plot_data import MAIN_ARMS
from exp.ablation_study.analysis.emit_plot_data import SCHEMA
from exp.ablation_study.analysis.emit_plot_data import SWEEP_TAGS
from exp.ablation_study.analysis.emit_plot_data import X_ALL_TEACHER
from exp.ablation_study.analysis.emit_plot_data import X_FROM_HIT_RATE
from exp.ablation_study.analysis.emit_plot_data import X_NO_TEACHER
from exp.ablation_study.analysis.emit_plot_data import arm_outcomes
from exp.ablation_study.analysis.emit_plot_data import collect_family
from exp.ablation_study.analysis.emit_plot_data import load_data
from exp.ablation_study.analysis.plot_ablation import by_arm
from exp.ablation_study.analysis.plot_ablation import load
from exp.ablation_study.analysis.plot_ablation import sweep_points

ALL_ARMS = list(MAIN_ARMS) + [f"kinroute_act_{t}" for t in SWEEP_TAGS]


# ------------------------------------------------------------------
# Fixtures: minimal but structurally faithful raw inputs
# ------------------------------------------------------------------
def _journal(tmp_path, name, arms, *, n=10, n_fail=2):
    """One journal with ``n`` eval episodes per arm, ``n_fail`` of them failed."""
    rows = []
    for arm in arms:
        for i in range(n):
            ok = i >= n_fail
            rows.append({
                "task_uid": f"{arm}:eval:0:{i}",
                "yaml_id": arm,
                "phase": "eval",
                "status": "done" if ok else "failed",
                "success": ok,
            })
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _per_step(tmp_path, name, hit_rates, *, steps=10):
    """One per-step export realising the requested FULL_HIT fraction per arm."""
    rows = []
    for arm, rate in hit_rates.items():
        n_hit = round(rate * steps)
        for i in range(steps):
            rows.append({
                "yaml_id": arm,
                "hit_type": "FULL_HIT" if i < n_hit else "MISS",
            })
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _anchor(tmp_path, *, n=10, n_fail=1):
    d = tmp_path / "anchor"
    d.mkdir()
    rows = [{"task_id": 0, "init_state_idx": i, "seed": 7, "success": i >= n_fail}
            for i in range(n)]
    (d / "results_tasks0.json").write_text(json.dumps(rows))
    return d


def _inputs(tmp_path, *, hit_rate=0.6, n=10, n_fail=2):
    rates = {a: hit_rate for a in ALL_ARMS}
    rates["pure_act"] = 0.0
    rates["pure_smolvla"] = 0.0
    return {
        "main_journal": _journal(tmp_path, "main.jsonl", MAIN_ARMS, n=n, n_fail=n_fail),
        "main_per_step": _per_step(tmp_path, "main_steps.jsonl",
                                   {a: rates[a] for a in MAIN_ARMS}),
        "sweep_journal": _journal(tmp_path, "sweep.jsonl",
                                  [f"kinroute_act_{t}" for t in SWEEP_TAGS],
                                  n=n, n_fail=n_fail),
        "sweep_per_step": _per_step(tmp_path, "sweep_steps.jsonl",
                                    {f"kinroute_act_{t}": hit_rate for t in SWEEP_TAGS}),
        "anchor_dir": _anchor(tmp_path),
    }


def _analysis(tmp_path, arms_block):
    p = tmp_path / "analysis.json"
    p.write_text(json.dumps({"arms": arms_block}))
    return p


# ------------------------------------------------------------------
# Collector
# ------------------------------------------------------------------
def test_failed_episodes_count_towards_the_denominator(tmp_path):
    """A journal row with status "failed" is an evaluated episode, not a gap.

    Filtering on status instead of phase silently drops every failure and makes
    every arm look like SR 1.0, which is the one way this collector could lie
    without raising.
    """
    j = _journal(tmp_path, "j.jsonl", ["hit_act"], n=10, n_fail=4)
    assert arm_outcomes(j)["hit_act"] == (10, 6)


def test_family_carries_every_plotted_point(tmp_path):
    fam = collect_family("libero_spatial", **_inputs(tmp_path))
    assert [p["arm"] for p in fam["points"]] == ALL_ARMS
    assert fam["matrix_arm_order"] == list(MAIN_ARMS)
    assert fam["teacher_anchor"]["success_rate"] == pytest.approx(0.9)
    for p in fam["points"]:
        assert p["success_rate"] == pytest.approx(0.8)
        assert p["n_episodes"] == 10


def test_abscissa_follows_the_routing_semantics_not_the_hit_rate(tmp_path):
    """Arms whose MISS slot runs a student never pay a teacher step."""
    fam = collect_family("libero_spatial", **_inputs(tmp_path, hit_rate=0.6))
    pts = {p["arm"]: p for p in fam["points"]}

    assert pts["hit_act"]["teacher_inference_rate_basis"] == X_FROM_HIT_RATE
    assert pts["hit_act"]["teacher_inference_rate"] == pytest.approx(0.4)
    assert pts["cache_baseline"]["teacher_inference_rate"] == pytest.approx(0.4)
    assert pts["kinroute_act_fh40"]["teacher_inference_rate"] == pytest.approx(0.4)

    for arm in ("miss_act", "miss_smolvla", "pure_act", "pure_smolvla"):
        assert pts[arm]["teacher_inference_rate_basis"] == X_NO_TEACHER
        assert pts[arm]["teacher_inference_rate"] == 0.0
    # miss arms do hit the cache; the abscissa must ignore that, not follow it.
    assert pts["miss_act"]["full_hit_rate"] == pytest.approx(0.6)

    anchor = fam["teacher_anchor"]
    assert anchor["teacher_inference_rate_basis"] == X_ALL_TEACHER
    assert anchor["teacher_inference_rate"] == 1.0


def test_main_matrix_sr_is_copied_verbatim_when_the_analyzer_is_supplied(tmp_path):
    inputs = _inputs(tmp_path)
    block = {a: {"sr": 0.8, "n": 10, "wilson_ci95": [0.5, 0.94]} for a in MAIN_ARMS}
    fam = collect_family("libero_spatial",
                         paired_analysis=_analysis(tmp_path, block), **inputs)
    pts = {p["arm"]: p for p in fam["points"]}
    assert pts["hit_act"]["success_rate_source"] == "verbatim from analyze_ablation.py"
    assert pts["hit_act"]["success_rate_ci95"] == [0.5, 0.94]
    # 4b has no upstream analyzer, so those points are honestly labelled derived.
    assert "aggregated from journal" in pts["kinroute_act_fh40"]["success_rate_source"]
    assert "paired_analysis" in fam["sources"]


def test_analyzer_disagreement_is_an_error_not_an_overwrite(tmp_path):
    inputs = _inputs(tmp_path)
    block = {a: {"sr": 0.8, "n": 10, "wilson_ci95": [0.5, 0.94]} for a in MAIN_ARMS}
    block["hit_act"]["sr"] = 0.99  # stale analysis against a fresher journal
    with pytest.raises(SystemExit, match="out of sync"):
        collect_family("libero_spatial",
                       paired_analysis=_analysis(tmp_path, block), **inputs)


def test_missing_arm_is_an_error(tmp_path):
    inputs = _inputs(tmp_path)
    inputs["main_journal"] = _journal(tmp_path, "short.jsonl",
                                      [a for a in MAIN_ARMS if a != "pure_smolvla"])
    with pytest.raises(SystemExit, match="pure_smolvla"):
        collect_family("libero_spatial", **inputs)


def test_absent_hit_rate_fails_instead_of_guessing_the_abscissa(tmp_path):
    inputs = _inputs(tmp_path)
    inputs["main_per_step"] = _per_step(
        tmp_path, "gap.jsonl",
        {a: 0.6 for a in MAIN_ARMS if a != "hit_act"})
    with pytest.raises(SystemExit, match="hit_act"):
        collect_family("libero_spatial", **inputs)


def test_sources_record_sha256_of_every_input(tmp_path):
    fam = collect_family("libero_spatial", **_inputs(tmp_path))
    for key in ("main_matrix_journal", "main_matrix_per_step",
                "kinematic_sweep_journal", "kinematic_sweep_per_step"):
        assert len(fam["sources"][key]["sha256"]) == 64
    assert fam["sources"]["teacher_anchor_shards"]


def test_schema_mismatch_refuses_to_merge(tmp_path):
    p = tmp_path / "plot_data.json"
    p.write_text(json.dumps({"schema": SCHEMA + 1, "families": {}}))
    with pytest.raises(SystemExit, match="schema"):
        load_data(p)


def test_missing_file_starts_an_empty_document(tmp_path):
    assert load_data(tmp_path / "absent.json") == {"schema": SCHEMA, "families": {}}


# ------------------------------------------------------------------
# Plot side
# ------------------------------------------------------------------
def _data_file(tmp_path, suites=("libero_spatial", "libero_10")):
    families = {}
    for i, suite in enumerate(suites):
        raw = tmp_path / f"raw{i}"
        raw.mkdir()
        families[suite] = collect_family(suite, **_inputs(raw))
    p = tmp_path / "plot_data.json"
    p.write_text(json.dumps({"schema": SCHEMA, "families": families}))
    return p


def test_plot_refuses_a_data_file_missing_a_suite(tmp_path):
    p = _data_file(tmp_path, suites=("libero_spatial",))
    with pytest.raises(SystemExit, match="libero_10"):
        load(p)


def test_plot_reads_both_suites_and_orders_the_sweep_by_threshold(tmp_path):
    data = load(_data_file(tmp_path))
    fam = data["families"]["libero_spatial"]
    assert [p["threshold_tag"] for p in sweep_points(fam)] == list(SWEEP_TAGS)
    assert set(by_arm(fam)) == set(ALL_ARMS)
