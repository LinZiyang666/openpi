"""RIT-PL: knot-value quantile LP, exact crossings, IR inversion, exporter,
emit integration, IR calibration, plot compatibility, LaTeX build and the
owner-authorized freeze-record retirement (plan logs/rit_pl_ir_ladder_plan.log.md section 7)."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

from exp.dispatch_surface import export_rit_pl, rit_pl, sgrid_sweep
from exp.dispatch_surface import rev1_package as pkgmod
from exp.dispatch_surface.analysis import analytic_cost as ac
from exp.dispatch_surface.analysis import ir_calibration
from exp.dispatch_surface.emit_precheck_yamls import _export_record_arms
from exp.dispatch_surface.fit_surface import load_table
from exp.dispatch_surface.phase0_roster import FAMILY_S0_PL, PROTOCOL_PHASE0
from exp.dispatch_surface.template_parity import (
    RIT_PL_EXPORT_ARTIFACT_KEYS,
    RIT_PL_EXPORT_RECORD_KEYS,
    RIT_PL_FIT_RECORD_KEYS,
    assert_template_parity,
)
from openpi.cache.components.surface_judge import load_surface_artifact, save_surface_artifact, surface_verdict
from tests.dispatch_surface.test_rev2_phase0 import TEMPLATE, build_package, build_world, run_exporter

REPO = pathlib.Path(__file__).resolve().parents[2]
FULL_MS, WARM_MS, MISS_MS = ac.unit_cost("FULL_HIT", None), ac.unit_cost("WARM_START", 0.3), ac.unit_cost("MISS", None)


def _sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _synthetic(n=400, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, 1.0, n)
    if ties:
        s = np.round(s, 1)
    y7 = 0.5 * (1 - s) + rng.exponential(0.05, n) + 0.2
    y10 = y7 + 0.2 + rng.exponential(0.05, n)
    return s, y7, y10


def _fit(n=400, seed=0, eps=rit_pl.EPS_TOTAL, ties=False):
    s, y7, y10 = _synthetic(n, seed, ties)
    knots, n_req = rit_pl.choose_knots(s)
    return s, y7, y10, rit_pl.fit_pl_quantile(s, y7, y10, knots, n_seg_req=n_req, alpha=0.05, eps_total=eps)


def _pinball(fit, s, y7, y10, alpha=0.05):
    loss = 0.0
    for layer, y in (("warm", y7), ("full", y10)):
        r = y - rit_pl.predict(fit, s, layer)
        loss += float(np.sum(np.where(r >= 0, (1 - alpha) * r, -alpha * r)))
    return loss


# ------------------------------------------------------------------
# 1-2. LP properties, exact crossings
# ------------------------------------------------------------------


def test_lp_is_strictly_monotone_nested_and_nonnegative():
    s, y7, y10, fit = _fit()
    eps = fit.eps_total / (fit.knots[-1] - fit.knots[0])
    for q in (fit.q_warm, fit.q_full):
        assert (np.diff(q) <= -eps * np.diff(fit.knots) + 1e-9).all()
        assert (q >= -1e-12).all()
    assert (fit.q_warm <= fit.q_full + 1e-9).all()
    assert fit.n_seg == len(fit.knots) - 1 and fit.n_seg_req == 24
    _, _, _, flat = _fit(eps=0.0)
    for q in (flat.q_warm, flat.q_full):
        assert (np.diff(q) <= 1e-9).all()
    assert _pinball(flat, s, y7, y10) <= _pinball(fit, s, y7, y10) + 1e-9


def test_fit_input_contracts():
    s, y7, y10 = _synthetic()
    knots, n_req = rit_pl.choose_knots(s)
    with pytest.raises(ValueError):
        rit_pl.fit_pl_quantile(s[:-1], y7, y10, knots, n_seg_req=n_req, alpha=0.05, eps_total=0.02)
    with pytest.raises(ValueError):
        rit_pl.fit_pl_quantile(s, y7, y10, knots[::-1], n_seg_req=n_req, alpha=0.05, eps_total=0.02)
    with pytest.raises(ValueError):
        rit_pl.fit_pl_quantile(s, y7, y10, knots, n_seg_req=n_req - 30, alpha=0.05, eps_total=0.02)
    with pytest.raises(ValueError):
        rit_pl.choose_knots(np.array([[0.1, 0.2]]))


def test_cut_at_is_the_exact_crossing_with_endpoint_rules():
    s, _, _, fit = _fit()
    for layer in rit_pl.LAYERS:
        q = getattr(fit, f"q_{layer}")
        for delta in np.linspace(q[-1] + 1e-6, q[0] - 1e-6, 25):
            theta = rit_pl.cut_at(fit, layer, float(delta))
            assert rit_pl.predict(fit, np.array([theta]), layer)[0] == pytest.approx(delta, abs=1e-9)
        assert rit_pl.cut_at(fit, layer, float(q[0]) + 1.0) == fit.knots[0]
        assert rit_pl.cut_at(fit, layer, float(q[-1]) - 1e-6) == np.inf
        grid = np.linspace(q[-1] - 0.1, q[0] + 0.1, 3000)
        thetas = np.array([rit_pl.cut_at(fit, layer, float(d)) for d in grid])
        finite = np.isfinite(thetas)
        assert (np.diff(thetas[finite]) <= 1e-12).all()                    # nonincreasing in delta
        drops = -np.diff(q) / np.diff(fit.knots)
        slope_bound = (grid[1] - grid[0]) / drops.min()
        assert np.abs(np.diff(thetas[finite])).max() <= slope_bound + 1e-9  # continuous
    full, warm = rit_pl.cuts(fit, float(np.median(fit.q_full)))
    assert warm <= full


def test_cut_at_refuses_flat_fits():
    _, _, _, flat = _fit(eps=0.0)
    with pytest.raises(ValueError):
        rit_pl.cut_at(flat, "full", 0.5)
    rit_pl.predict(flat, np.array([0.5]), "full")   # prediction stays allowed


# ------------------------------------------------------------------
# 3. deployed-verdict cost parity
# ------------------------------------------------------------------


def test_predicted_ir_matches_surface_verdict_row_by_row():
    s, _, _ = _synthetic(300, seed=3)
    s[0], s[1] = 0.5, 0.25                                    # rows exactly on the cuts
    edges = np.array([-np.inf, np.inf])
    for theta_full, theta_warm in ((0.5, 0.25), (np.inf, 0.25), (np.inf, np.inf), (0.0, 0.0), (0.9, 0.9)):
        total = 0.0
        for x in s:
            v = surface_verdict(float(x), None, edges, np.array([theta_full]), np.array([theta_warm]),
                                uses_disagreement=False)
            total += {"full": FULL_MS, "warm": WARM_MS, "miss": MISS_MS}[v]
        assert rit_pl.predicted_ir(s, theta_full, theta_warm) == pytest.approx(100 * total / (len(s) * MISS_MS), abs=1e-12)


# ------------------------------------------------------------------
# 4. inverse IR contract
# ------------------------------------------------------------------


def test_delta_for_ir_hits_targets_on_a_dense_table():
    s, _, _, fit = _fit(n=4000, seed=1)
    ir_lo, ir_hi = rit_pl.attainable_range(fit, s)
    assert ir_hi == 100.0 and ir_lo == pytest.approx(100 * FULL_MS / MISS_MS, abs=1e-9)
    last = None
    for target in range(20, 100, 5):
        res = rit_pl.delta_for_ir(fit, s, float(target))
        assert abs(res["ir_gap"]) <= rit_pl.IR_TOL and res["delta"] > 0
        assert res["predicted_ir"] == pytest.approx(rit_pl.predicted_ir(s, res["theta_full"], res["theta_warm"]))
        if last is not None:
            assert res["delta"] <= last                         # more budget -> larger tolerance? no: lower cost <-> larger delta
        last = res["delta"]
    deltas = np.linspace(*rit_pl._endpoints(fit), 400)
    irs = [rit_pl._ir_at(fit, s, float(d)) for d in deltas]
    assert (np.diff(irs) <= 1e-12).all()                         # IR nonincreasing in delta


def test_delta_for_ir_on_a_coarse_table_reports_the_nearest_attainable_point():
    s, _, _, fit = _fit(n=80, seed=2)                             # 80 rows: ~0.4 pt per row, ladder lands on 6 segments
    gaps = []
    for target in range(20, 100, 5):
        res = rit_pl.delta_for_ir(fit, s, float(target))
        br = res["bracket"]
        assert res["predicted_ir"] in (br["ir_lo"], br["ir_hi"])
        assert abs(res["ir_gap"]) <= min(abs(br["ir_lo"] - target), abs(br["ir_hi"] - target)) + 1e-12
        if abs(br["ir_lo"] - target) == abs(br["ir_hi"] - target):
            assert res["delta"] == br["delta_lo"]               # tie -> conservative end
        gaps.append(abs(res["ir_gap"]))
    assert max(gaps) > rit_pl.IR_TOL                            # the coarse table cannot hit everything
    with pytest.raises(ValueError):
        rit_pl.delta_for_ir(fit, s, 100.0)
    with pytest.raises(ValueError):
        rit_pl.delta_for_ir(fit, s, 5.0)


def test_q_min_zero_fit_stays_in_the_positive_delta_domain():
    fit = rit_pl.PLFit(knots=np.array([0.0, 1.0, 2.0]), q_warm=np.array([0.5, 0.2, 0.0]),
                       q_full=np.array([0.6, 0.3, 0.0]), eps_total=0.02, n_seg_req=6, n_seg=2, alpha=0.05)
    s = np.concatenate([np.linspace(0.0, 1.9, 50), [2.0, 2.0]])
    lo, hi = rit_pl._endpoints(fit)
    assert lo > 0 and lo == np.nextafter(0.0, 1.0)
    ir_lo, ir_hi = rit_pl.attainable_range(fit, s)
    assert ir_hi < 100.0                                        # all-MISS is not reachable with delta > 0
    res = rit_pl.delta_for_ir(fit, s, ir_hi)
    assert res["delta"] > 0 and res["predicted_ir"] == ir_hi
    with pytest.raises(ValueError):
        rit_pl.delta_for_ir(fit, s, 0.5 * (ir_hi + 100.0))


def test_floor_info_flags_data_plateaus_and_is_reproducible():
    rng = np.random.default_rng(5)
    s = rng.uniform(0, 1, 600)
    y10 = 5.0 + rng.normal(0, 0.01, 600)                       # risk does not depend on s at all
    y7 = y10 - 1.0
    knots, n_req = rit_pl.choose_knots(s)
    fit = rit_pl.fit_pl_quantile(s, y7, y10, knots, n_seg_req=n_req, alpha=0.05, eps_total=0.02)
    info = rit_pl.floor_info(fit, s, float(np.median(fit.q_full)))
    assert info["full"]["on_eps_floor"] and info["full"]["segment_share"] > 0
    rec = rit_pl.fit_record_fields(fit)
    again = rit_pl.fit_from_record(json.loads(json.dumps(rec)))
    assert rit_pl.floor_info(again, s, float(np.median(fit.q_full))) == info
    assert rit_pl.pl_fit_digests(again) == rit_pl.pl_fit_digests(fit)
    assert rit_pl.floor_info(fit, s, float(fit.q_full[-1]) - 1.0)["full"] is None


def test_ecdf_quantile_is_right_continuous():
    y = np.array([1.0, 2.0, 2.0, 3.0])
    assert rit_pl.ecdf_quantile(y, 2.0) == 0.75 and rit_pl.ecdf_quantile(y, 1.999) == 0.25


# ------------------------------------------------------------------
# 5. knot ladder
# ------------------------------------------------------------------


def test_choose_knots_descends_the_ladder_and_keeps_the_requested_rung():
    rng = np.random.default_rng(7)
    knots, n_req = rit_pl.choose_knots(rng.uniform(0, 1, 100))
    assert n_req == 12                                          # 100 / 24 < 8 per segment, 100 / 12 >= 8
    assert rit_pl.choose_knots(rng.uniform(0, 1, 10)) is None
    tied = np.repeat(np.linspace(0, 1, 5), 100)
    knots, n_req = rit_pl.choose_knots(tied)
    assert n_req == 24 and len(knots) - 1 < 24                  # de-duplicated knots, rung remembered
    fit = rit_pl.fit_pl_quantile(tied, tied + 1, tied + 2, knots, n_seg_req=n_req, alpha=0.05, eps_total=0.02)
    assert fit.n_seg == len(knots) - 1 and fit.n_seg_req == 24


# ------------------------------------------------------------------
# 6. exporter on the synthetic Rev 1 world
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ritpl")
    w = build_world(tmp)
    manifest_path, manifest = build_package(w, tmp)
    return types.SimpleNamespace(tmp=tmp, world=w, manifest_path=manifest_path, manifest=manifest,
                                 manifest_sha=pkgmod.load_manifest(manifest_path)[2])


def _export(world, out_dir, **kw):
    args = types.SimpleNamespace(rev1_package_manifest=str(world.manifest_path), table=str(world.world.table),
                                 target_ir="", quantiles="", out_dir=str(out_dir))
    for k, v in kw.items():
        setattr(args, k, v)
    return export_rit_pl.export(args)


@pytest.fixture(scope="module")
def ir_export(world):
    return _export(world, world.tmp / "pl_ir", target_ir="40,60,82.5")


@pytest.fixture(scope="module")
def q_export(world):
    return _export(world, world.tmp / "pl_q", quantiles="0.6,0.9")


def test_exporter_target_ir_mode_binds_everything(world, ir_export):
    rec = ir_export
    assert set(rec) - {"export_record_sha256", "attainable_ir_range"} == RIT_PL_EXPORT_RECORD_KEYS
    assert rec["addressing"] == "target_ir" and sorted(rec["artifacts"]) == ["ir40", "ir60", "ir82p5"]
    out = world.tmp / "pl_ir"
    assert _sha(out / export_rit_pl.FIT_RECORD_NAME) == rec["pl_fit_record_sha256"]
    fit_rec = json.loads((out / export_rit_pl.FIT_RECORD_NAME).read_text())
    assert set(fit_rec) == RIT_PL_FIT_RECORD_KEYS and fit_rec["n_seg_req"] == 24
    source = load_surface_artifact(str(pkgmod.verify_member(world.manifest, out.parent / "package", "artifact.dsp_s0")))
    table = load_table(str(world.world.table))
    for name, art_rec in rec["artifacts"].items():
        assert set(art_rec) == RIT_PL_EXPORT_ARTIFACT_KEYS
        art = load_surface_artifact(art_rec["path"])
        assert_template_parity(art, source, what=name)
        assert art.meta["pl_fit_record_sha256"] == rec["pl_fit_record_sha256"]
        assert art.meta["estimator"] == rit_pl.ESTIMATOR and art.meta["family"] == FAMILY_S0_PL
        assert art.meta["floor_info"] == art_rec["floor_info"] and art.meta["addressing"] == "target_ir"
        assert abs(art_rec["ir_gap"]) <= rit_pl.IR_MAX_GAP and art_rec["predicted_ir"] - art_rec["target_ir"] == art_rec["ir_gap"]
        assert art_rec["quantile"] == rit_pl.ecdf_quantile(table.y10, art_rec["delta"])
        assert art.s_min_warm[0] <= art.s_min_full[0] and art.delta == art_rec["delta"] > 0
    assert rec["artifacts"]["ir82p5"]["target_ir"] == 82.5


def test_exporter_quantile_mode_uses_the_phase0_delta_call(world, q_export):
    rec = q_export
    assert rec["addressing"] == "quantile" and sorted(rec["artifacts"]) == ["p60", "p90"]
    table = load_table(str(world.world.table))
    for name, art_rec in rec["artifacts"].items():
        assert art_rec["target_ir"] is None and art_rec["ir_gap"] is None
        q = float(name[1:]) / 100
        assert art_rec["quantile"] == q
        assert art_rec["delta"] == float(np.percentile(table.y10, 100 * q, method="linear"))


def test_exporter_refusals(world, tmp_path):
    with pytest.raises(SystemExit, match="exactly one"):
        _export(world, tmp_path / "a", target_ir="50", quantiles="0.5")
    with pytest.raises(SystemExit, match="exactly one"):
        _export(world, tmp_path / "b")
    with pytest.raises(SystemExit, match="strictly increasing"):
        _export(world, tmp_path / "c", target_ir="50,50")
    with pytest.raises(SystemExit, match="strictly increasing"):
        _export(world, tmp_path / "d", quantiles="0.9,0.6")
    with pytest.raises(SystemExit, match="exclusive"):
        _export(world, tmp_path / "e", target_ir="0,50")
    with pytest.raises(SystemExit, match="all-MISS"):
        _export(world, tmp_path / "f", quantiles="0.001")
    with pytest.raises(SystemExit, match="attainable"):
        _export(world, tmp_path / "g", target_ir="10")            # below the all-FULL_HIT floor
    (tmp_path / "h").mkdir()
    (tmp_path / "h" / "stale").write_text("x")
    with pytest.raises(SystemExit, match="must be empty"):
        _export(world, tmp_path / "h", target_ir="60")
    other = tmp_path / "table.jsonl"
    other.write_text(world.world.table.read_text() + "\n")
    with pytest.raises(SystemExit):
        _export(world, tmp_path / "i", target_ir="60", table=str(other))


def test_exporter_refuses_a_tampered_package_member(world, tmp_path):
    pkg_src = world.manifest_path.parent
    pkg = tmp_path / "package"
    shutil.copytree(pkg_src, pkg)
    target = pkgmod.member_path(world.manifest, pkg, "fit.s0")
    rec = json.loads(target.read_text())
    rec["dev_membership"] = rec["dev_membership"][:-1]
    target.write_text(json.dumps(rec))
    args = types.SimpleNamespace(rev1_package_manifest=str(pkg / world.manifest_path.name), table=str(world.world.table),
                                 target_ir="60", quantiles="", out_dir=str(tmp_path / "out"))
    with pytest.raises(SystemExit):
        export_rit_pl.export(args)


# ------------------------------------------------------------------
# 7. emit integration
# ------------------------------------------------------------------


def _phase0_record(world, name, role, quantiles):
    out = world.tmp / name
    if not (out / "export_record.json").is_file():
        run_exporter(world.manifest_path, world.world, out, role, quantiles)
    return out / "export_record.json"


def test_export_record_arms_default_behaviour_is_unchanged(world, ir_export):
    p0 = json.loads(_phase0_record(world, "p0_sv", "artifact.dsp_sv", "0.85").read_text())
    arms = _export_record_arms([p0])
    assert set(arms) == {"dsp_sv_p85"}
    assert set(arms["dsp_sv_p85"]) == {"family", "quantile", "artifact", "delta", "output_sha256", "export_record_index"}
    pl = json.loads((world.tmp / "pl_ir" / "export_record.json").read_text())
    with pytest.raises(SystemExit, match="not an exploratory record"):
        _export_record_arms([pl])
    both = _export_record_arms([p0, pl], protocols=(PROTOCOL_PHASE0, rit_pl.PROTOCOL_RIT_PL),
                               families=("sv", "s0", FAMILY_S0_PL))
    assert {"dsp_sv_p85", "dsp_s0_pl_ir40", "dsp_s0_pl_ir60", "dsp_s0_pl_ir82p5"} == set(both)
    assert both["dsp_s0_pl_ir60"]["extra"]["target_ir"] == 60.0 and "extra" not in both["dsp_sv_p85"]
    with pytest.raises(SystemExit, match="has family"):
        _export_record_arms([p0, pl], protocols=(PROTOCOL_PHASE0, rit_pl.PROTOCOL_RIT_PL))


def _emit(world, out_dir, records, gate_layer="primary"):
    args = types.SimpleNamespace(rev1_package_manifest=str(world.manifest_path), export_records=",".join(map(str, records)),
                                 template=str(TEMPLATE), library_pkl=str(world.world.lib), out_dir=str(out_dir),
                                 gate_layer=gate_layer)
    sgrid_sweep.emit(args)
    return json.loads((out_dir / "arm_matrix_sgrid.json").read_text())


@pytest.fixture(scope="module")
def mixed_matrix(world, ir_export, q_export):
    records = [_phase0_record(world, "p0_sv", "artifact.dsp_sv", "0.85"),
               _phase0_record(world, "p0_s0", "artifact.dsp_s0", "0.7"),
               world.tmp / "pl_ir" / "export_record.json", world.tmp / "pl_q" / "export_record.json"]
    matrix = _emit(world, world.tmp / "cfg_mixed", records)
    gated = _emit(world, world.tmp / "cfg_mixed_gated", records, gate_layer="secondary")
    return matrix, gated


def test_sgrid_emit_accepts_mixed_phase0_and_pl_records(world, mixed_matrix, ir_export):
    matrix, gated = mixed_matrix
    arms = set(matrix["arms"])
    assert {"dsp_sv_p85", "dsp_s0_p70", "dsp_s0_pl_ir40", "dsp_s0_pl_ir60", "dsp_s0_pl_ir82p5", "dsp_s0_pl_p60", "dsp_s0_pl_p90"} <= arms
    assert matrix["families"]["dsp_s0_pl_ir60"] == FAMILY_S0_PL and matrix["families"]["dsp_s0_p70"] == "s0"
    assert matrix["target_ir"]["dsp_s0_pl_ir60"] == 60.0 and matrix["target_ir"]["dsp_s0_pl_p60"] is None
    assert matrix["target_ir"]["dsp_sv_p85"] is None and matrix["estimator"]["dsp_sv_p85"] is None
    assert matrix["estimator"]["dsp_s0_pl_p90"] == rit_pl.ESTIMATOR
    assert matrix["pl_fit_record_sha256"]["dsp_s0_pl_ir40"] == ir_export["pl_fit_record_sha256"]
    assert matrix["predicted_ir"]["dsp_s0_pl_ir40"] == ir_export["artifacts"]["ir40"]["predicted_ir"]
    assert matrix["contract_arm"] == "dsp_sv_p85" and matrix["gate_type"] == "always_search"
    assert gated["gate_type"] == "score_hysteresis" and set(gated["arms"]) == arms
    import yaml as yamllib
    doc = yamllib.safe_load(open(matrix["arms"]["dsp_s0_pl_ir60"]))
    assert doc["checkpoints"]["cp1"]["judge"] == {"type": "dispatch_surface",
                                                   "surface_artifact_path": matrix["artifact_paths"]["dsp_s0_pl_ir60"]}


def test_pl_summary_fields_are_passed_through(mixed_matrix):
    matrix, _ = mixed_matrix
    assert sgrid_sweep.pl_arm_fields(matrix, "dsp_s0_pl_ir60") == {
        "target_ir": 60.0, "predicted_ir": matrix["predicted_ir"]["dsp_s0_pl_ir60"], "estimator": rit_pl.ESTIMATOR}
    assert sgrid_sweep.pl_arm_fields({"arms": {}}, "dsp_s0_p70") == {}


def _pl_copy(world, tmp_path, name):
    """Fresh export whose files may be tampered in place."""
    out = tmp_path / name
    _export(world, out, target_ir="60")
    return out


def test_emit_refuses_missing_drifted_and_semantically_tampered_pl_fit_records(world, tmp_path):
    sv = _phase0_record(world, "p0_sv", "artifact.dsp_sv", "0.85")
    # (i) missing
    out = _pl_copy(world, tmp_path, "missing")
    (out / export_rit_pl.FIT_RECORD_NAME).unlink()
    with pytest.raises(SystemExit, match="PL fit record missing"):
        _emit(world, tmp_path / "cfg_missing", [sv, out / "export_record.json"])
    # (ii) byte drift with the recorded SHA untouched
    out = _pl_copy(world, tmp_path, "drift")
    p = out / export_rit_pl.FIT_RECORD_NAME
    p.write_text(p.read_text() + " ")
    with pytest.raises(SystemExit, match="drifted from the export record SHA"):
        _emit(world, tmp_path / "cfg_drift", [sv, out / "export_record.json"])
    # (iii) semantic tamper: knots changed, every SHA updated so the byte leg passes
    out = _pl_copy(world, tmp_path, "semantic")
    fit_rec = json.loads(p.read_text()) if False else json.loads((out / export_rit_pl.FIT_RECORD_NAME).read_text())
    fit_rec["knots"][1] = 0.5 * (fit_rec["knots"][0] + fit_rec["knots"][1])
    (out / export_rit_pl.FIT_RECORD_NAME).write_text(json.dumps(fit_rec, indent=2, sort_keys=True))
    new_sha = _sha(out / export_rit_pl.FIT_RECORD_NAME)
    rec_path = out / "export_record.json"
    rec = json.loads(rec_path.read_text())
    rec["pl_fit_record_sha256"] = new_sha
    for name, art_rec in rec["artifacts"].items():
        art = load_surface_artifact(art_rec["path"])
        art.meta["pl_fit_record_sha256"] = new_sha
        save_surface_artifact(art, art_rec["path"])
        art_rec["output_sha256"] = _sha(art_rec["path"])
    rec_path.write_text(json.dumps(rec, indent=2, sort_keys=True))
    with pytest.raises(SystemExit, match="pl_fit_digests do not match"):
        _emit(world, tmp_path / "cfg_semantic", [sv, rec_path])


def test_emit_refuses_unknown_protocols(world, tmp_path):
    sv = _phase0_record(world, "p0_sv", "artifact.dsp_sv", "0.85")
    bad = tmp_path / "bad.json"
    rec = json.loads(sv.read_text())
    rec["protocol"] = "dispatch_surface_unknown"
    bad.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="not accepted by the sweep"):
        _emit(world, tmp_path / "cfg_bad", [sv, bad])


# ------------------------------------------------------------------
# 8. source lock (outcome-blind)
# ------------------------------------------------------------------


@pytest.mark.parametrize("mod", ["rit_pl", "export_rit_pl", "analysis/ir_calibration"])
def test_source_lock_outcome_blind(mod):
    tree = ast.parse((REPO / f"exp/dispatch_surface/{mod}.py").read_text())
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            names.add(n.module)
        if isinstance(n, ast.Import):
            names.update(a.name for a in n.names)
    for banned in ("frontier_hull", "phase0_outcome_design", "analyze_precheck"):
        assert not any(banned in nm for nm in names), (mod, banned)
    consts = {c.value for c in ast.walk(tree) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    assert not ({"success", "status"} & consts), mod


# ------------------------------------------------------------------
# 9. IR calibration
# ------------------------------------------------------------------


def _summary_for(matrix_path: pathlib.Path, matrix: dict, world, arms=None, subset=False, **overrides) -> dict:
    arms = arms or list(matrix["arms"])
    rows = {}
    for i, arm in enumerate(arms):
        cost = 30.0 + 3.0 * i
        rows[arm] = {"family": matrix["families"][arm], "cost": cost, "sr": 0.8, "t": cost * 100, "d": 100.0}
    summary = {"suite": matrix["suite"], "protocol": matrix["protocol"], "arms": rows, "arms_subset": subset,
               "input_sha256": {"arm_matrix": _sha(matrix_path)}, "rev1_package_manifest_sha256": world.manifest_sha,
               "cost_model_digest": ac.cost_model_digest(), "posthoc_exploratory": True}
    summary.update(overrides)
    return summary


def _calibrate(world, sources, table=None):
    args = types.SimpleNamespace(rev1_package_manifest=str(world.manifest_path), table=table or str(world.world.table),
                                 source=sources)
    return ir_calibration.calibrate(args)


def _write(p: pathlib.Path, obj) -> pathlib.Path:
    p.write_text(json.dumps(obj, indent=1, sort_keys=True))
    return p


@pytest.fixture(scope="module")
def calib_inputs(world, mixed_matrix):
    matrix, gated = mixed_matrix
    m_path, g_path = world.tmp / "cfg_mixed" / "arm_matrix_sgrid.json", world.tmp / "cfg_mixed_gated" / "arm_matrix_sgrid.json"
    s_path = _write(world.tmp / "summary_mixed.json", _summary_for(m_path, matrix, world))
    g_summary = _write(world.tmp / "summary_gated.json", _summary_for(g_path, gated, world, arms=["dsp_s0_pl_ir60"], subset=True))
    return types.SimpleNamespace(matrix=matrix, m_path=m_path, s_path=s_path, gated=gated, g_path=g_path, g_summary=g_summary)


def test_ir_calibration_predicts_row_by_row_and_reports_gaps(world, calib_inputs):
    ci = calib_inputs
    res = _calibrate(world, [f"plain:{ci.m_path}:{ci.s_path}", f"gated:{ci.g_path}:{ci.g_summary}"])
    assert set(res) == ir_calibration.OUTPUT_KEYS and set(res["sources"]) == {"plain", "gated"}
    plain = res["sources"]["plain"]
    assert set(plain) == ir_calibration.SOURCE_KEYS and plain["gate_type"] == "always_search"
    assert set(plain["arms"]) == set(ci.matrix["arms"])
    assert set(res["sources"]["gated"]["arms"]) == {"dsp_s0_pl_ir60"}                    # subset summary respected
    table = load_table(str(world.world.table))
    fit_rec = json.loads(pkgmod.verify_member(world.manifest, world.manifest_path.parent, "fit.s0").read_text())
    from exp.dispatch_surface.export_exploratory_surface import dev_mask_from_membership
    dev = dev_mask_from_membership(table, fit_rec["dev_membership"])
    for arm, a in plain["arms"].items():
        assert set(a) == ir_calibration.ARM_KEYS
        art = load_surface_artifact(ci.matrix["artifact_paths"][arm])
        total = sum({"full": FULL_MS, "warm": WARM_MS, "miss": MISS_MS}[surface_verdict(
            float(table.s[i]), float(table.v[i]), art.v_bin_edges, art.s_min_full, art.s_min_warm,
            uses_disagreement=art.uses_disagreement)] for i in np.where(dev)[0])
        assert a["predicted_ir"] == pytest.approx(100 * total / (dev.sum() * MISS_MS))
        assert a["measured_ir"] == pytest.approx(100 * 30.0 / MISS_MS, abs=100 * 3.0 * len(plain["arms"]) / MISS_MS)
        assert a["gap"] == pytest.approx(a["measured_ir"] - a["predicted_ir"])
    pl_arm = plain["arms"]["dsp_s0_pl_ir60"]
    assert pl_arm["estimator"] == rit_pl.ESTIMATOR and pl_arm["family"] == FAMILY_S0_PL
    assert pl_arm["predicted_ir"] == pytest.approx(ci.matrix["predicted_ir"]["dsp_s0_pl_ir60"], abs=1e-9)


def test_ir_calibration_refusals(world, calib_inputs, tmp_path):
    ci = calib_inputs
    good = f"plain:{ci.m_path}:{ci.s_path}"
    with pytest.raises(SystemExit, match="duplicate --source tag"):
        _calibrate(world, [good, good])
    with pytest.raises(SystemExit, match="non-empty"):
        _calibrate(world, [f":{ci.m_path}:{ci.s_path}"])
    other = tmp_path / "table.jsonl"
    other.write_text(world.world.table.read_text() + "\n")
    with pytest.raises(SystemExit, match="not the shadow table"):
        _calibrate(world, [good], table=str(other))

    def variant(name, mutate_matrix=None, mutate_summary=None, arms=None, subset=False):
        m = copy.deepcopy(ci.matrix)
        if mutate_matrix:
            mutate_matrix(m)
        mp = _write(tmp_path / f"{name}_matrix.json", m)
        s = _summary_for(mp, m, world, arms=arms, subset=subset)
        if mutate_summary:
            mutate_summary(s)
        sp = _write(tmp_path / f"{name}_summary.json", s)
        return f"{name}:{mp}:{sp}"

    def _set(key, value):
        def f(d):
            d[key] = value
        return f

    with pytest.raises(SystemExit, match="different Rev 1 package"):
        _calibrate(world, [variant("foreign", _set("rev1_package_manifest_sha256", "0" * 64))])
    with pytest.raises(SystemExit, match="package suite"):
        _calibrate(world, [variant("suite", _set("suite", "libero_spatial"), _set("suite", "libero_spatial"))])
    with pytest.raises(SystemExit, match="not an sgrid sweep"):
        _calibrate(world, [variant("tgrid", _set("layer", "exploratory_tgrid"))])
    with pytest.raises(SystemExit, match="frozen library"):
        _calibrate(world, [variant("lib", _set("library_sha256", "1" * 64))])
    with pytest.raises(SystemExit, match="cost model digest"):
        _calibrate(world, [variant("cost", mutate_summary=_set("cost_model_digest", "2" * 64))])
    with pytest.raises(SystemExit, match="does not bind this arm matrix"):
        _calibrate(world, [variant("bind", mutate_summary=lambda s: s["input_sha256"].update(arm_matrix="3" * 64))])
    with pytest.raises(SystemExit, match="subset of the matrix arms"):
        _calibrate(world, [variant("extra", mutate_summary=lambda s: s["arms"].update(dsp_ghost=dict(s["arms"]["dsp_sv_p85"])))])

    def drop_artifact(m):
        m["artifact_paths"].pop("dsp_s0_pl_ir60")
    with pytest.raises(SystemExit, match="no surface artifact"):
        _calibrate(world, [variant("noart", drop_artifact)])
    with pytest.raises(SystemExit, match="family differs"):
        _calibrate(world, [variant("fam", mutate_summary=lambda s: s["arms"]["dsp_s0_pl_ir60"].update(family="s0"))])
    # a foreign package: a second synthetic world whose manifest SHA differs
    (tmp_path / "w2").mkdir()
    w2 = build_world(tmp_path / "w2")
    manifest2, _ = build_package(w2, tmp_path / "w2")
    args = types.SimpleNamespace(rev1_package_manifest=str(manifest2), table=str(w2.table), source=[good])
    with pytest.raises(SystemExit, match="different Rev 1 package"):
        ir_calibration.calibrate(args)
    with pytest.raises(SystemExit, match="drifted since emit"):
        def touch(m):
            m["artifact_sha256"]["dsp_s0_pl_ir60"] = "4" * 64
        _calibrate(world, [variant("art", touch)])


# ------------------------------------------------------------------
# 10. plot compatibility
# ------------------------------------------------------------------


def _design(with_pl: bool) -> dict:
    fams = {}
    for i, fam in enumerate(("threshold", "s0", "sv")):
        arms = {f"dsp_{fam}_a": {"cost": 38.0 + i, "sr": 0.6 + 0.05 * i, "t": 3800.0 + 100 * i, "d": 100.0},
                f"dsp_{fam}_b": {"cost": 55.0 + i, "sr": 0.8 + 0.02 * i, "t": 5500.0 + 100 * i, "d": 100.0}}
        fams[fam] = {"measured_policies": arms, "active": list(arms)}
    return {"suite": "libero_10", "interval": [41.7, 47.7], "B_1": 43.7, "B_2": 45.7, "families": fams}


def _sgrid_summary(with_pl: bool) -> dict:
    arms = {"dsp_s0_p60": {"family": "s0", "cost": 47.0, "sr": 0.72, "t": 4700.0, "d": 100.0},
            "dsp_sv_p60": {"family": "sv", "cost": 46.0, "sr": 0.74, "t": 4600.0, "d": 100.0}}
    if with_pl:
        arms["dsp_s0_pl_ir82p5"] = {"family": "s0_pl", "cost": 55.7, "sr": 0.83, "t": 5570.0, "d": 100.0}
        arms["dsp_s0_pl_p925"] = {"family": "s0_pl", "cost": 39.0, "sr": 0.6, "t": 3900.0, "d": 100.0}
    return {"suite": "libero_10", "arms": arms}


@pytest.mark.parametrize("with_pl", [False, True])
def test_budget_amendment_frontier_figures_accept_optional_s0_pl(tmp_path, with_pl):
    from exp.dispatch_surface.analysis import plot_budget_amendment as pb

    dense = pb.merge_sgrid(_design(with_pl), _sgrid_summary(with_pl))
    assert ("s0_pl" in dense["families"]) is with_pl
    assert pb._present_families(dense["families"]) == (["threshold", "s0", "s0_pl", "sv"] if with_pl else ["threshold", "s0", "sv"])
    pb.fig_family_frontiers(dense, tmp_path, suffix="_dense")
    pb.fig_pareto_hull_percent(dense, {"realized_cost_ms": 67.518595, "sr": 0.85}, tmp_path, suffix="_dense")
    assert (tmp_path / "family_frontiers_dense.png").is_file() and (tmp_path / "pareto_hull_percent_dense.png").is_file()
    assert pb._short("dsp_s0_pl_ir82p5") == "IR82.5" and pb._short("dsp_s0_pl_p925") == "q.925"
    with pytest.raises(SystemExit, match="unknown to the outcome design"):
        pb.merge_sgrid(_design(False), {"suite": "libero_10", "arms": {"x": {"family": "crd", "cost": 1, "sr": 1, "t": 1, "d": 1}}})


@pytest.mark.parametrize("with_pl", [False, True])
def test_suite_frontier_figure_accepts_optional_s0_pl(tmp_path, monkeypatch, with_pl):
    from exp.dispatch_surface.analysis import plot_suite_frontiers as ps

    summary = tmp_path / "sgrid.json"
    summary.write_text(json.dumps(_sgrid_summary(with_pl)))
    monkeypatch.setattr(sys, "argv", ["plot_suite_frontiers", "--sgrid-summary", str(summary), "--anchor", "67.518595", "0.85",
                                      "--out-dir", str(tmp_path), "--tag", "t"])
    ps.main()
    assert (tmp_path / "pareto_frontiers_t.png").is_file()
    assert ps._label("dsp_s0_pl_ir82p5") == "IR82.5" and ps._label("dsp_s0_pl_p925") == "q.925"
    assert ps._label("dsp_s0_p9362wp9542") == "q.9362→.9542"


# ------------------------------------------------------------------
# 11. LaTeX note compiles
# ------------------------------------------------------------------


def test_sonly_note_compiles(tmp_path):
    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex not installed")
    latex = REPO / "docs/iclr/latex"
    proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tmp_path}",
                           "sonly_note.tex"], cwd=latex, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert (tmp_path / "sonly_note.pdf").is_file()


# ------------------------------------------------------------------
# 12. freeze-record retirement is exactly the authorized transformation
# ------------------------------------------------------------------


def test_freeze_record_retirement_is_recorded():
    from exp.dispatch_surface import freeze_record as fr

    path = REPO / fr.RECORD_PATH
    rec = json.loads(path.read_text())
    retired = rec["retired_documents"]
    assert len(retired) == 1
    (doc, entry), = retired.items()
    assert doc == "docs/iclr/ICLR_PAPER_BLOCKING_TODO.md" and not (REPO / doc).exists()
    assert set(entry) == {"sha256", "retired_in_commit", "reason", "record_sha256_before_retirement"}
    assert doc not in rec["documents_sha256"]
    assert subprocess.run(["git", "cat-file", "-e", f"{entry['retired_in_commit']}^{{commit}}"], cwd=REPO).returncode == 0
    reconstructed = copy.deepcopy(rec)
    reconstructed.pop("retired_documents")
    reconstructed["documents_sha256"][doc] = entry["sha256"]
    old_bytes = (json.dumps(reconstructed, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    assert hashlib.sha256(old_bytes).hexdigest() == entry["record_sha256_before_retirement"]
    assert hashlib.sha256(old_bytes).hexdigest() == "9e28b6a3564add5ea3252856e51f1782d1cad0790df7e73c809a66e6c0f36fcc"
    fr.verify(fr.load_record(path), REPO)            # the amended record verifies on the working tree


# ------------------------------------------------------------------
# G2 Round 1: provenance coherence, refusal branches, end-to-end passes
# ------------------------------------------------------------------


def _resync_pl_shas(out: pathlib.Path) -> pathlib.Path:
    """After editing fit_record_pl.json in place, propagate its new SHA into the
    export record and every artifact meta so the byte leg passes and the
    semantic leg is what decides."""
    new_sha = _sha(out / export_rit_pl.FIT_RECORD_NAME)
    rec_path = out / "export_record.json"
    rec = json.loads(rec_path.read_text())
    rec["pl_fit_record_sha256"] = new_sha
    for art_rec in rec["artifacts"].values():
        art = load_surface_artifact(art_rec["path"])
        art.meta["pl_fit_record_sha256"] = new_sha
        save_surface_artifact(art, art_rec["path"])
        art_rec["output_sha256"] = _sha(art_rec["path"])
    rec_path.write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec_path


def test_emit_refuses_record_cut_tampering(world, tmp_path):
    sv = _phase0_record(world, "p0_sv", "artifact.dsp_sv", "0.85")
    out = _pl_copy(world, tmp_path, "cut")
    rec_path = out / "export_record.json"
    rec = json.loads(rec_path.read_text())
    rec["artifacts"]["ir60"]["theta_full"] += 1e-6                 # record says one cut, artifact deploys another
    rec_path.write_text(json.dumps(rec, indent=2, sort_keys=True))
    with pytest.raises(SystemExit, match="theta_full/theta_warm differ from the deployed artifact"):
        _emit(world, tmp_path / "cfg_cut", [sv, rec_path])


def test_emit_refuses_artifact_meta_addressing_tampering(world, tmp_path):
    sv = _phase0_record(world, "p0_sv", "artifact.dsp_sv", "0.85")
    out = _pl_copy(world, tmp_path, "meta")
    rec_path = out / "export_record.json"
    rec = json.loads(rec_path.read_text())
    art_rec = rec["artifacts"]["ir60"]
    art = load_surface_artifact(art_rec["path"])
    art.meta["target_ir"] = 61.0                                     # meta diverges from the record entry
    save_surface_artifact(art, art_rec["path"])
    art_rec["output_sha256"] = _sha(art_rec["path"])
    rec_path.write_text(json.dumps(rec, indent=2, sort_keys=True))
    with pytest.raises(SystemExit, match="meta target_ir differs from the export record entry"):
        _emit(world, tmp_path / "cfg_meta", [sv, rec_path])


def test_exporter_refuses_gap_above_max_and_warns_above_tol(world, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_rit_pl, "IR_MAX_GAP", 1e-9)
    with pytest.raises(SystemExit, match="not attainable within"):
        _export(world, tmp_path / "gap", target_ir="60")
    monkeypatch.setattr(export_rit_pl, "IR_MAX_GAP", rit_pl.IR_MAX_GAP)
    monkeypatch.setattr(export_rit_pl, "IR_TOL", 1e-12)
    rec = _export(world, tmp_path / "tol", target_ir="60")
    assert "warning: target IR 60" in capsys.readouterr().out and abs(rec["artifacts"]["ir60"]["ir_gap"]) > 1e-12


def test_emit_semantic_leg_rejects_forged_n_seg_req(world, tmp_path):
    sv = _phase0_record(world, "p0_sv", "artifact.dsp_sv", "0.85")
    for forged, message in ((7, "not a ladder rung"), (12, "inconsistent")):
        out = _pl_copy(world, tmp_path, f"nseg{forged}")
        fit_path = out / export_rit_pl.FIT_RECORD_NAME
        fit_rec = json.loads(fit_path.read_text())
        assert fit_rec["n_seg_req"] == 24 and fit_rec["n_seg"] == 24
        fit_rec["n_seg_req"] = forged
        fit_path.write_text(json.dumps(fit_rec, indent=2, sort_keys=True))
        rec_path = _resync_pl_shas(out)
        rec = json.loads(rec_path.read_text())                     # keep meta n_seg_req coherent with the fit record
        for art_rec in rec["artifacts"].values():
            art = load_surface_artifact(art_rec["path"])
            art.meta["n_seg_req"] = forged
            save_surface_artifact(art, art_rec["path"])
            art_rec["output_sha256"] = _sha(art_rec["path"])
        rec_path.write_text(json.dumps(rec, indent=2, sort_keys=True))
        with pytest.raises(SystemExit, match=message):
            _emit(world, tmp_path / f"cfg_nseg{forged}", [sv, rec_path])


def test_exporter_reaches_the_membership_digest_branch(world, tmp_path):
    """Tamper the s-only fit record's membership while keeping every package
    SHA consistent, so the exporter passes verify_package and fails on the
    membership digest itself."""
    pkg = tmp_path / "package"
    shutil.copytree(world.manifest_path.parent, pkg)
    manifest_path = pkg / world.manifest_path.name
    manifest = json.loads(manifest_path.read_text())

    def member(role):
        return pkg / manifest["members"][role]["member"]

    fit = json.loads(member("fit.s0").read_text())
    fit["dev_membership"] = fit["dev_membership"][:-1]              # digest field left untouched
    member("fit.s0").write_text(json.dumps(fit))
    manifest["members"]["fit.s0"]["sha256"] = _sha(member("fit.s0"))
    matrix = json.loads(member("matrix").read_text())
    matrix["fit_record_sha256"]["s0"] = manifest["members"]["fit.s0"]["sha256"]
    member("matrix").write_text(json.dumps(matrix))
    manifest["members"]["matrix"]["sha256"] = _sha(member("matrix"))
    verdict = json.loads(member("verdict").read_text())
    verdict["discipline"]["fit_record_sha256"]["s0"] = manifest["members"]["fit.s0"]["sha256"]
    verdict["discipline"]["arm_matrix_sha256"] = manifest["members"]["matrix"]["sha256"]
    member("verdict").write_text(json.dumps(verdict))
    manifest["members"]["verdict"]["sha256"] = _sha(member("verdict"))
    manifest_path.write_text(json.dumps(manifest))
    pkgmod.verify_package(manifest_path)                             # the package itself is consistent
    args = types.SimpleNamespace(rev1_package_manifest=str(manifest_path), table=str(world.world.table),
                                 target_ir="60", quantiles="", out_dir=str(tmp_path / "out"))
    with pytest.raises(SystemExit, match="membership is missing or its digest drifted"):
        export_rit_pl.export(args)


def test_summarize_passes_pl_fields_end_to_end(world, mixed_matrix, tmp_path):
    from tests.dispatch_surface.test_rev2_phase0 import FULL, MISS, WARM, _officials, _rows_for

    matrix, _ = mixed_matrix
    matrix_path = world.tmp / "cfg_mixed" / "arm_matrix_sgrid.json"
    rng = np.random.default_rng(11)
    per_step, journal = [], []
    for arm in matrix["arms"]:
        for t, offs in _officials().items():
            for subset, official in enumerate(offs):
                verdicts = [FULL if u < 0.4 else (WARM if u < 0.6 else MISS) for u in rng.random(int(rng.integers(6, 12)))]
                rows, j = _rows_for(arm, t, subset, official, verdicts, bool(rng.random() < 0.6), run_id="runsummarize")
                per_step += rows
                journal.append(j)
    run = tmp_path / "run"
    run.mkdir()
    (run / "per_step.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_step))
    (run / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in journal))
    _write(run / "per_step.jsonl.launch.json", {"schema_version": 2, "launches": [{"run_id": "runsummarize"}]})
    out = tmp_path / "summary.json"
    sgrid_sweep.summarize(types.SimpleNamespace(arm_matrix=str(matrix_path), journal=str(run / "journal.jsonl"),
                                                per_step=str(run / "per_step.jsonl"),
                                                launch_manifest=str(run / "per_step.jsonl.launch.json"),
                                                split_manifest=str(world.world.split), trials=30, arms="", out=str(out)))
    summary = json.loads(out.read_text())
    pl = summary["arms"]["dsp_s0_pl_ir60"]
    assert pl["family"] == FAMILY_S0_PL and pl["target_ir"] == 60.0 and pl["estimator"] == rit_pl.ESTIMATOR
    assert pl["predicted_ir"] == matrix["predicted_ir"]["dsp_s0_pl_ir60"]
    assert summary["arms"]["dsp_s0_pl_p60"]["target_ir"] is None and summary["arms"]["dsp_s0_pl_p60"]["estimator"] == rit_pl.ESTIMATOR
    assert summary["arms"]["dsp_sv_p85"]["target_ir"] is None and summary["arms"]["dsp_sv_p85"]["estimator"] is None
    assert summary["run_ids"] == ["runsummarize"] and pl["episodes"] == 300


def _legend_texts(captured) -> list[str]:
    return [t.get_text() for fig in captured for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()]


@pytest.mark.parametrize("with_pl", [False, True])
def test_frontier_legends_name_rit_pl_only_when_present(tmp_path, monkeypatch, with_pl):
    from matplotlib.figure import Figure

    from exp.dispatch_surface.analysis import plot_budget_amendment as pb
    from exp.dispatch_surface.analysis import plot_suite_frontiers as ps

    captured = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: captured.append(self))
    dense = pb.merge_sgrid(_design(with_pl), _sgrid_summary(with_pl))
    pb.fig_family_frontiers(dense, tmp_path, suffix="_dense")
    pb.fig_pareto_hull_percent(dense, {"realized_cost_ms": 67.518595, "sr": 0.85}, tmp_path, suffix="_dense")
    summary = tmp_path / "sgrid.json"
    summary.write_text(json.dumps(_sgrid_summary(with_pl)))
    monkeypatch.setattr(sys, "argv", ["plot_suite_frontiers", "--sgrid-summary", str(summary), "--anchor", "67.518595", "0.85",
                                      "--out-dir", str(tmp_path), "--tag", "t"])
    ps.main()
    assert len(captured) >= 3
    texts = _legend_texts(captured)
    assert any("RIT-PL" in t for t in texts) is with_pl
    assert all("RIT-PL" in t for t in texts if "IR-addressed" in t)
