"""Unit tests for Stage 2b fair Pareto overlay assembly (all non-manual).

Covers the plan's locked contracts:
- RPG rows parse spatial-only + carry inference_ratio_source verbatim + are
  reference-layer;
- overlay layering is asymmetric: libero_spatial includes the RPG reference
  layer, libero_10 has NO RPG; a non-reference RPG point is rejected;
- frontier interpolation / gain / Pareto-dominance math.
"""

from __future__ import annotations

import math

import pytest

from exp.gate_research import stage2b_pareto_overlay as B

RPG_CSV = (
    "cfg,gate_type,param_slug,p_inference,seed,cache_len,inference_len,episodes,"
    "success_rate,mean_inference_ratio,inference_ratio_source\n"
    "clip_w7_d4,periodic,periodic_k10_n1,,,10,1,500,0.716,0.0738,derived\n"
    "clip_w7_d4,random,random_p0p05,0.05,\"0,1,2\",,,1500,0.7327,0.05,expected\n"
)


def _anchor(label, inf, sr):
    return {"label": label, "inf": inf, "sr": sr, "layer": "loadbearing", "kind": "d1_anchor"}


def _live(label, inf, sr, suite, gate):
    return {"label": label, "inf": inf, "sr": sr, "suite": suite, "gate_type": gate,
            "layer": "loadbearing", "kind": gate}


# ---------------------------------------------------------------- RPG parse
def test_parse_rpg_anchors(tmp_path):
    p = tmp_path / "aggregate.csv"
    p.write_text(RPG_CSV)
    rows = B.parse_rpg_anchors(p)
    assert len(rows) == 2
    assert all(r["layer"] == "reference" and r["kind"] == "rpg" for r in rows)
    per = next(r for r in rows if r["gate_type"] == "periodic")
    assert math.isclose(per["sr"], 71.6) and math.isclose(per["inf"], 0.0738)
    assert per["inf_source"] == "derived"
    rnd = next(r for r in rows if r["gate_type"] == "random")
    assert rnd["inf_source"] == "expected"


# ---------------------------------------------------------------- asymmetric layering
def test_overlay_spatial_has_rpg_reference():
    anchors = [_anchor("fh75_ws10", 0.287, 83.0)]
    live = [_live("spatial_A_periodic", 0.280, 90.4, "libero_spatial", "periodic"),
            _live("l10_A_periodic", 0.70, 82.2, "libero_10", "periodic")]
    rpg = [{"label": "x", "inf": 0.07, "sr": 71.6, "layer": "reference", "kind": "rpg"}]
    ov = B.build_overlay_layers("libero_spatial", anchors, live, rpg)
    assert len(ov["reference"]) == 1
    lb_labels = {p["label"] for p in ov["loadbearing"]}
    assert "spatial_A_periodic" in lb_labels and "l10_A_periodic" not in lb_labels


def test_overlay_l10_has_no_rpg():
    anchors = [_anchor("fh5_ws40", 0.636, 78.0)]
    live = [_live("l10_A_periodic", 0.70, 82.2, "libero_10", "periodic")]
    rpg = [{"label": "x", "inf": 0.07, "sr": 71.6, "layer": "reference", "kind": "rpg"}]
    ov = B.build_overlay_layers("libero_10", anchors, live, rpg)
    assert ov["reference"] == []
    assert {p["label"] for p in ov["loadbearing"]} == {"fh5_ws40", "l10_A_periodic"}


def test_overlay_rejects_loadbearing_rpg():
    bad_rpg = [{"label": "x", "inf": 0.07, "sr": 71.6, "layer": "loadbearing", "kind": "rpg"}]
    with pytest.raises(AssertionError):
        B.build_overlay_layers("libero_spatial", [], [], bad_rpg)


# ---------------------------------------------------------------- frontier math
def test_frontier_interp_and_out_of_range():
    anchors = [_anchor("a", 0.27, 85.0), _anchor("b", 0.29, 83.0), _anchor("c", 0.35, 91.0)]
    assert math.isclose(B.frontier_interp(anchors, 0.30), 83.0 + 8.0 * 0.01 / 0.06)
    assert B.frontier_interp(anchors, 0.20) is None   # below range -> no extrapolation
    assert B.frontier_interp(anchors, 0.40) is None   # above range


def test_frontier_gain():
    anchors = [_anchor("a", 0.27, 85.0), _anchor("b", 0.29, 83.0), _anchor("c", 0.35, 91.0)]
    g = B.frontier_gain({"label": "p", "inf": 0.30, "sr": 90.0}, anchors)
    assert math.isclose(g["gain_pp"], 90.0 - (83.0 + 8.0 * 0.01 / 0.06))
    oor = B.frontier_gain({"label": "q", "inf": 0.10, "sr": 90.0}, anchors)
    assert oor["gain_pp"] is None and oor["frontier_sr"] is None


def test_pareto_dominates():
    p = {"sr": 90.0, "inf": 0.28}
    q = {"sr": 85.0, "inf": 0.30}
    assert B.pareto_dominates(p, q)
    assert not B.pareto_dominates(q, p)
    assert not B.pareto_dominates(p, p)  # equal on both -> not strict domination


# ---------------------------------------------------------------- optional pure-inf anchor
def test_parse_extra_anchors_and_frontier_extension():
    extra = B.parse_extra_anchors(["libero_10:1.0:83:pure_inf", "libero_spatial:1.0:94:pure_inf"])
    assert len(extra["libero_10"]) == 1 and extra["libero_10"][0]["kind"] == "pure_inf"
    assert extra["libero_10"][0]["inf"] == 1.0 and extra["libero_10"][0]["sr"] == 83.0
    # a live point beyond the d1 range becomes in-range once the pure-inf anchor is added
    d1 = [_anchor("fh5_ws40", 0.636, 78.0)]
    frontier = d1 + extra["libero_10"]
    assert B.frontier_interp(d1, 0.70) is None            # OOR without pure-inf
    assert B.frontier_interp(frontier, 0.70) is not None  # in-range with pure-inf
    assert {p["kind"] for p in B._frontier_anchors(frontier)} == {"d1_anchor", "pure_inf"}
