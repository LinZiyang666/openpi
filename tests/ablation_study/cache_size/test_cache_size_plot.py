"""Plot pipeline provenance (plan §8.5).

The figure reads only ``plot_data.json``; that file is built by
``emit_plot_data.py``, which copies the analyzer output verbatim and records its
sha256. These tests pin both halves: the collector may not invent or recompute
numbers, and the plot may not draw what the data file cannot back.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from exp.ablation_study.cache_size.analysis.emit_plot_data import (
    attach_latency,
    collect_family,
    family_key,
    load_data,
)
from exp.ablation_study.cache_size.analysis.plot_size import (
    TIERS,
    authoritative_family,
)


def _result(**over):
    r = {
        "suite": "libero_spatial",
        "outcome_filter": "all",
        "family_role": "primary",
        "tier_sr": {t: 0.5 + i * 0.05 for i, t in enumerate(TIERS)},
        "tier_ci": {t: [0.4 + i * 0.05, 0.6 + i * 0.05] for i, t in enumerate(TIERS)},
        "teacher_sr": 0.95,
        "verdict": {"branch": "G", "q": "Q-fail", "d": "D-teacher", "p": "P-inconc"},
    }
    r.update(over)
    return r


def _collect(tmp_path, result=None, **kw):
    rp = tmp_path / "result.json"
    rp.write_text(json.dumps(result or _result()))
    return collect_family(json.loads(rp.read_text()), result_path=rp,
                          grid=kw.get("grid"), entries=kw.get("entries"),
                          bucket_sizes=kw.get("bucket_sizes"))


def _data(tmp_path, fam=None):
    fam = fam or _collect(tmp_path)
    return {"schema": 1,
            "families": {family_key(fam["suite"], fam["outcome_filter"]): fam}}


# --- collector -----------------------------------------------------------------

def test_collector_copies_verbatim_and_records_sha(tmp_path):
    fam = _collect(tmp_path)
    assert fam["teacher_success_rate"] == 0.95
    assert [p["success_rate"] for p in fam["points"]] == [0.5 + i * 0.05 for i in range(6)]
    assert len(fam["source"]["sha256"]) == 64
    assert fam["source"]["result_json"].endswith("result.json")


def test_collector_missing_field_is_fatal(tmp_path):
    r = _result()
    del r["tier_sr"]
    with pytest.raises(SystemExit, match="tier_sr"):
        _collect(tmp_path, result=r)


def test_collector_partial_tiers_are_fatal(tmp_path):
    r = _result()
    del r["tier_sr"]["S4"]
    with pytest.raises(SystemExit, match="missing tiers"):
        _collect(tmp_path, result=r)


def test_collector_prefers_realized_x(tmp_path):
    grid = {"mean_realized": {"S6": 43.9}}
    fam = _collect(tmp_path, grid=grid)
    by = {p["source_arm"].rsplit("_", 1)[-1]: p for p in fam["points"]}
    assert by["S6"]["trajectories_per_task"] == 43.9
    assert by["S6"]["trajectories_per_task_is_realized"]
    assert by["S6"]["episodes_in_library"] == 439
    assert by["S1"]["trajectories_per_task"] == 1
    assert not by["S1"]["trajectories_per_task_is_realized"]


def test_points_carry_no_bare_tier_codes(tmp_path):
    """Every descriptive field is self-explaining; tier ids survive only inside
    the source_arm artifact name."""
    fam = _collect(tmp_path)
    for pt in fam["points"]:
        assert "tier" not in pt
        for k in pt:
            assert k == "source_arm" or not any(s in k for s in ("S1", "tier_id")), k
    assert fam["points"][0]["source_arm"] == "cache_size_libero_spatial_all_S1"


def test_recollect_replaces_only_its_family(tmp_path):
    a = _collect(tmp_path)
    b = _collect(tmp_path, result=_result(suite="libero_10", teacher_sr=0.868))
    data = {"schema": 1, "families": {}}
    for fam in (a, b):
        data["families"][family_key(fam["suite"], fam["outcome_filter"])] = fam
    a2 = _collect(tmp_path, result=_result(teacher_sr=0.974))
    data["families"][family_key(a2["suite"], a2["outcome_filter"])] = a2
    assert data["families"]["libero_spatial/all"]["teacher_success_rate"] == 0.974
    assert data["families"]["libero_10/all"]["teacher_success_rate"] == 0.868  # untouched


def test_schema_mismatch_refuses_to_merge(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"schema": 99, "families": {}}))
    with pytest.raises(SystemExit, match="schema"):
        load_data(p)


# --- latency attachment ----------------------------------------------------------

def test_latency_attaches_under_label_without_touching_sr(tmp_path):
    data = _data(tmp_path)
    a1 = "cache_size_libero_spatial_all_S1"
    a6 = "cache_size_libero_spatial_all_S6"
    attach_latency(data, key="libero_spatial/all", label="wsl_optimized",
                   latency={a1: 1.6, a6: 9.9})
    by = {p["source_arm"]: p for p in data["families"]["libero_spatial/all"]["points"]}
    assert by[a1]["retrieval_latency_ms"]["wsl_optimized"] == 1.6
    assert by[a1]["success_rate"] == 0.5
    attach_latency(data, key="libero_spatial/all", label="srv_optimized",
                   latency={a1: 1.1})
    assert set(by[a1]["retrieval_latency_ms"]) == {"wsl_optimized", "srv_optimized"}


def test_latency_cannot_invent_points(tmp_path):
    data = _data(tmp_path)
    with pytest.raises(SystemExit, match="unknown arms"):
        attach_latency(data, key="libero_spatial/all", label="x",
                       latency={"cache_size_libero_spatial_all_S9": 1.0})
    with pytest.raises(SystemExit, match="not in the data file"):
        attach_latency(data, key="nope/all", label="x",
                       latency={"cache_size_libero_spatial_all_S1": 1.0})


# --- plot side -------------------------------------------------------------------

def test_plot_missing_family_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="not in plot data"):
        authoritative_family(_data(tmp_path), "libero_10/success")


def test_plot_lone_point_is_fatal(tmp_path):
    fam = _collect(tmp_path)
    fam["points"] = fam["points"][:1]
    with pytest.raises(SystemExit, match="full collected ladder"):
        authoritative_family({"schema": 2, "families": {"libero_spatial/all": fam}},
                             "libero_spatial/all")


def test_plot_point_without_sr_is_fatal(tmp_path):
    fam = _collect(tmp_path)
    del fam["points"][0]["success_rate"]
    with pytest.raises(SystemExit, match="lacks 'success_rate'"):
        authoritative_family({"schema": 2, "families": {"libero_spatial/all": fam}},
                             "libero_spatial/all")


def test_numbers_cannot_be_overridden_by_the_caller():
    """plot() takes no SR/CI/teacher parameters -- provenance is not optional."""
    import inspect

    from exp.ablation_study.cache_size.analysis.plot_size import plot

    params = set(inspect.signature(plot).parameters)
    forbidden = {"tier_sr", "tier_ci", "teacher_sr", "entries", "latency_ms",
                 "success_rate", "teacher_success_rate"}
    assert not (params & forbidden), (
        "the figure must not be able to disagree with the collected data file"
    )


# --- x-axis wording follows the library family (plan §3.1b ruling 1) ----------

def _axis_label(tmp_path, monkeypatch, result_over):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    from exp.ablation_study.cache_size.analysis.plot_size import plot

    seen: list[str] = []
    original = Axes.set_xlabel
    monkeypatch.setattr(
        Axes, "set_xlabel",
        lambda self, label, *a, **k: (seen.append(label), original(self, label, *a, **k))[1],
    )
    fam = _collect(tmp_path, result=_result(**result_over))
    key = family_key(fam["suite"], fam["outcome_filter"])
    plot({"schema": 2, "families": {key: fam}}, families=[key],
         out_path=tmp_path / "p.png")
    assert seen, "plot() set no x label"
    return seen[0]


def test_all_family_axis_says_collected_not_successful(tmp_path, monkeypatch):
    label = _axis_label(tmp_path, monkeypatch, {"outcome_filter": "all"})
    assert "collected trajectories" in label
    assert "successful" not in label


def test_success_family_axis_still_says_successful(tmp_path, monkeypatch):
    label = _axis_label(tmp_path, monkeypatch, {"outcome_filter": "success"})
    assert "successful trajectories" in label


def test_unlabelled_result_keeps_the_pre_ruling_wording(tmp_path, monkeypatch):
    label = _axis_label(tmp_path, monkeypatch, {"outcome_filter": None})
    assert "successful trajectories" in label


# --- latency panel: one curve per label, gaps stay gaps -------------------------

def test_latency_panel_draws_one_curve_per_label(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    from exp.ablation_study.cache_size.analysis.plot_size import plot

    data = _data(tmp_path)
    arms = [f"cache_size_libero_spatial_all_{t}" for t in TIERS]
    attach_latency(data, key="libero_spatial/all", label="host_a",
                   latency={a: 1.0 + i for i, a in enumerate(arms)})
    # host_b legitimately lacks the top tier (e.g. OOM) -- its curve is shorter,
    # the missing point must NOT be invented or borrowed from host_a.
    attach_latency(data, key="libero_spatial/all", label="host_b",
                   latency={a: 2.0 + i for i, a in enumerate(arms[:-1])})

    lines = []
    original = Axes.plot
    monkeypatch.setattr(Axes, "plot",
                        lambda self, *a, **k: (lines.append((a, k)), original(self, *a, **k))[1])
    plot(data, families=["libero_spatial/all"], out_path=tmp_path / "p.png",
         latency_labels=["host_a", "host_b"])
    lat_lines = [(a, k) for a, k in lines if k.get("label") in ("host_a", "host_b")]
    assert len(lat_lines) == 2
    by = {k["label"]: a for a, k in lat_lines}
    assert len(by["host_a"][0]) == 6
    assert len(by["host_b"][0]) == 5   # gap preserved, not filled


def test_latency_unknown_label_is_fatal(tmp_path):
    from exp.ablation_study.cache_size.analysis.plot_size import plot

    with pytest.raises(SystemExit, match="no point carries latency"):
        plot(_data(tmp_path), families=["libero_spatial/all"],
             out_path=tmp_path / "p.png", latency_labels=["typo"])


def test_plot_records_figure_name_into_family_block(tmp_path):
    from exp.ablation_study.cache_size.analysis.plot_size import plot

    data = _data(tmp_path)
    plot(data, families=["libero_spatial/all"], out_path=tmp_path / "curve_a.png")
    plot(data, families=["libero_spatial/all"], out_path=tmp_path / "curve_a.png")  # idempotent
    plot(data, families=["libero_spatial/all"], out_path=tmp_path / "curve_b.png")
    assert data["families"]["libero_spatial/all"]["figures"] == ["curve_a.png", "curve_b.png"]


# --- overlay: both filters of one suite share a figure --------------------------

def _two_family_data(tmp_path):
    a = _collect(tmp_path)                                    # spatial/all
    b = _collect(tmp_path, result=_result(outcome_filter="success",
                                          family_role="secondary-descriptive",
                                          tier_sr={t: 0.52 + i * 0.05
                                                   for i, t in enumerate(TIERS)}))
    data = {"schema": 2, "families": {}}
    for fam in (a, b):
        data["families"][family_key(fam["suite"], fam["outcome_filter"])] = fam
    return data


def test_overlay_draws_both_families_and_one_teacher_line(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    from exp.ablation_study.cache_size.analysis.plot_size import plot

    lines, hlines = [], []
    orig_plot, orig_axh = Axes.plot, Axes.axhline
    monkeypatch.setattr(Axes, "plot",
                        lambda self, *a, **k: (lines.append(k.get("label")), orig_plot(self, *a, **k))[1])
    monkeypatch.setattr(Axes, "axhline",
                        lambda self, *a, **k: (hlines.append(k.get("label")), orig_axh(self, *a, **k))[1])
    data = _two_family_data(tmp_path)
    plot(data, families=["libero_spatial/all", "libero_spatial/success"],
         out_path=tmp_path / "overlay.png")
    assert "all collected trajectories (primary)" in lines
    assert "successful trajectories only (secondary)" in lines
    assert len(hlines) == 1  # one shared teacher line
    for key in ("libero_spatial/all", "libero_spatial/success"):
        assert data["families"][key]["figures"] == ["overlay.png"]


def test_overlay_refuses_mixed_suites_or_teachers(tmp_path):
    from exp.ablation_study.cache_size.analysis.plot_size import plot

    data = _two_family_data(tmp_path)
    other = _collect(tmp_path, result=_result(suite="libero_10", teacher_sr=0.868))
    data["families"]["libero_10/all"] = other
    with pytest.raises(SystemExit, match="one suite"):
        plot(data, families=["libero_spatial/all", "libero_10/all"],
             out_path=tmp_path / "x.png")
    data["families"]["libero_spatial/success"]["teacher_success_rate"] = 0.9
    with pytest.raises(SystemExit, match="teacher anchor"):
        plot(data, families=["libero_spatial/all", "libero_spatial/success"],
             out_path=tmp_path / "x.png")


def test_no_delta_margin_band(tmp_path, monkeypatch):
    """Owner 08-19: the shaded delta band under the teacher line is gone."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    from exp.ablation_study.cache_size.analysis.plot_size import plot

    spans = []
    orig = Axes.axhspan
    monkeypatch.setattr(Axes, "axhspan",
                        lambda self, *a, **k: (spans.append(a), orig(self, *a, **k))[1])
    plot(_data(tmp_path), families=["libero_spatial/all"], out_path=tmp_path / "p.png")
    assert spans == []
