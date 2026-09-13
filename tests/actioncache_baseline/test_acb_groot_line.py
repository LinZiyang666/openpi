"""The GR00T arm of the ActionCache baseline in ``exp/actioncache_baseline``:
teacher profiles, the encoded-key cost formula (E in the numerator, teacher
forward M in the denominator), cost-record binding, IR addressing with the
preferred / fallback selection rule, the preflight gate, the ten-arm export,
GR00T-priced aggregation and the verifier's schedule negatives -- and that the
Pi0.5 line's numbers are untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import pickle

import numpy as np
import pytest
import torch
import yaml

from exp.actioncache_baseline import export_arms as ex
from exp.actioncache_baseline import libs
from exp.actioncache_baseline.aggregate import aggregate, pricing_for
from exp.actioncache_baseline.verify_cp2_artifact import VerificationError, verify
from openpi.cache.groot.cp2_key_builder import GrootCP2TernaryKeyBuilder
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

GROOT = libs.GROOT_LIBERO
K8 = "groot_n15_k8_v1"
LAYOUT = {"kind": "groot_encoded_v1", "token_len": 5, "feature_dim": 4, "state_feat_dim": 2}
PROJ_META = GrootCP2TernaryKeyBuilder(seed=7, d=8, p=0.25, token_len=5, feature_dim=4, state_feat_dim=2).projection_meta()
S1, S2, L, E = 6.145862978883088, 7.191964512458071, 28.104, 2.5
P, M = S1 + S2, S1 + S2 + L
MODEL_DIGEST = "c" * 64          # the library's / the E record's model
CKPT_SHA = "d" * 64              # the shared teacher table's calibration checkpoint
GPU_UUID, GPU_NAME = "GPU-98d36ed2-test", "NVIDIA GeForce RTX 4090"
ACC_SHA = "1" * 64               # accepted cohort manifest digest shared by shadow table and preflight
PROVENANCE = {"suite": "libero_spatial", "cost_record_sha256": "a" * 64, "encoder_cost_record_sha256": "b" * 64,
              "mode": "reduce-overhead", "layout": LAYOUT, "model": {"weights_digest": MODEL_DIGEST},
              "ckpt_sha256": CKPT_SHA, "gpu_uuid": GPU_UUID, "gpu_name": GPU_NAME, "n_tokens": 566,
              "warmup": 30, "iters": 200, "teacher_ckpt_sha256": CKPT_SHA}


def _record(e=E, **prov):
    return libs.CostRecord(teacher=GROOT.name, table="measured", stage1_ms=S1, stage2_ms=S2,
                           stage3_full_loop_ms=L, encoder_ms=e, schedule_id=K8,
                           provenance={"cost_formula_version": libs.COST_FORMULA_GROOT, **PROVENANCE, **prov})


# ------------------------------------------------------------------
# profiles and the cost formula
# ------------------------------------------------------------------


def test_profiles_are_keyed_by_builder_and_carry_the_frozen_constants():
    assert libs.profile_for_builder("cp2_groot_ternary") is GROOT
    assert libs.profile_for_builder("cp2_vlm_ternary") is libs.PI05
    assert libs.profile_for_builder("cp1_groot_libero_spatial_pool_16") is None
    assert GROOT.tiers == {"n0": ("FULL_HIT", None), "n1": ("WARM_START", 0.875)}
    assert libs.PI05.tiers == {"n0": ("FULL_HIT", None), "n1": ("WARM_START", 0.1)}
    assert GROOT.denoise_schedule == K8 and GROOT.action_chunk_shape == (16, 32)
    assert GROOT.cost_tables == ("measured",) and GROOT.reference_theta_raw == 0.65
    assert GROOT.stage1_paths == ("groot_reconstructed_template",)
    assert libs.PI05.cost_tables == ("cuda_graph", "eager") and libs.PI05.reference_theta_raw == 0.85
    with pytest.raises(KeyError):
        libs.profile("nope")


def test_groot_verdict_costs_put_E_in_the_numerator_only():
    rec = _record()
    assert libs.teacher_forward_cost(rec) == pytest.approx(M)  # denominator has no E
    assert libs.cp2_verdict_cost(rec, "FULL_HIT", None) == pytest.approx(P + E)
    assert libs.cp2_verdict_cost(rec, "WARM_START", 0.875) == pytest.approx(P + E + L / 8)
    assert libs.cp2_verdict_cost(rec, "WARM_START", 0.5) == pytest.approx(P + E + L / 2)
    assert libs.cp2_verdict_cost(rec, "MISS", None) == pytest.approx(M + E)
    assert rec.stage3_fraction(0.875) == pytest.approx(1 / 8) and rec.stage3_fraction(0.125) == pytest.approx(7 / 8)
    with pytest.raises(ValueError):
        libs.cp2_verdict_cost(rec, "WARM_START", None)
    with pytest.raises(ValueError):
        libs.cp2_verdict_cost(rec, "PARTIAL", None)
    # Owner's numbers with E = 2: 15.338 / 18.851 / 43.442 over M = 41.442.
    r2 = _record(2.0)
    assert (round(libs.cp2_verdict_cost(r2, "FULL_HIT", None), 3), round(libs.cp2_verdict_cost(r2, "WARM_START", 0.875), 3),
            round(libs.cp2_verdict_cost(r2, "MISS", None), 3), round(libs.teacher_forward_cost(r2), 3)) == (15.338, 18.851, 43.442, 41.442)


def test_all_miss_prices_above_100_percent_and_the_range_follows():
    s = np.array([0.1, 0.5, 0.9])
    rec = _record()
    assert ex.ir_percent(s, np.inf, "n0", rec, profile=GROOT) == pytest.approx(100 * (M + E) / M)
    assert ex.ir_percent(s, np.inf, "n0", rec, profile=GROOT) > 100
    assert ex.ir_percent(s, -np.inf, "n0", rec, profile=GROOT) == pytest.approx(100 * (P + E) / M)
    assert ex.ir_percent(s, -np.inf, "n1", rec, profile=GROOT) == pytest.approx(100 * (P + E + L / 8) / M)
    lo, hi = ex.attainable_range(s, "n1", rec, profile=GROOT)
    assert lo < 100 < hi
    with pytest.raises(ValueError, match="needs a CostRecord"):
        ex.ir_percent(s, 0.5, "n0", "cuda_graph", profile=GROOT)


def test_pi05_pricing_is_unchanged():
    for table in ("cuda_graph", "eager"):
        rec = libs.pi05_cost_record(table)
        assert rec.encoder_ms == 0.0 and rec.schedule_id is None
        assert libs.teacher_forward_cost(rec) == pytest.approx(libs.miss_cost(table))
        for ht, st in (("FULL_HIT", None), ("WARM_START", 0.1), ("WARM_START", 0.5), ("MISS", None)):
            assert libs.cp2_verdict_cost(rec, ht, st) == pytest.approx(libs.cp2_tier_cost(ht, st, table))
        s = np.array([0.2, 0.4, 0.6, 0.8])
        assert ex.ir_percent(s, 0.5, "n0", table) == ex.ir_percent(s, 0.5, "n0", rec, profile=libs.PI05)
    assert ex.ir_percent(np.array([0.5]), np.inf, "n0") == pytest.approx(100.0)
    summary = libs.cost_record_summary(libs.pi05_cost_record(), libs.PI05)
    assert summary["encoder_ms"] == 0.0 and summary["cost_formula_version"] == libs.COST_FORMULA_PI05
    assert libs.cost_record_from_summary(summary).teacher_forward_ms == pytest.approx(libs.miss_cost())


def test_cost_summary_round_trips_for_groot_with_its_binding():
    rec = _record()
    summary = libs.cost_record_summary(rec, GROOT)
    assert summary["verdict_unit_ms"] == {"FULL_HIT": pytest.approx(P + E), "WARM_START@0.875": pytest.approx(P + E + L / 8),
                                          "MISS": pytest.approx(M + E)}
    assert summary["teacher_forward_ms"] == pytest.approx(M) and summary["cost_formula_version"] == libs.COST_FORMULA_GROOT
    back = libs.cost_record_from_summary(json.loads(json.dumps(summary)))
    assert (back.encoder_ms, back.schedule_id, back.table, back.teacher) == (E, K8, "measured", GROOT.name)
    # Everything a consumer re-binds survives the round trip (model, hardware, sampling, suite).
    assert {k: back.provenance[k] for k in libs.GROOT_COST_PROVENANCE_KEYS} == PROVENANCE
    for key in libs.GROOT_COST_PROVENANCE_KEYS:
        with pytest.raises(SystemExit, match="lacks provenance"):
            libs.cost_record_from_summary({k: v for k, v in summary.items() if k != key})
    with pytest.raises(SystemExit, match="unknown GR00T cost formula"):
        libs.cost_record_from_summary(dict(summary, cost_formula_version="other"))
    with pytest.raises(SystemExit, match="unknown teacher"):
        libs.cost_record_from_summary(dict(summary, teacher="x"))
    with pytest.raises(SystemExit, match="not a finite positive E"):
        libs.cost_record_from_summary(dict(summary, encoder_ms=0.0))


# ------------------------------------------------------------------
# cost-record binding (E never defaulted)
# ------------------------------------------------------------------


def _cost_files(tmp_path, *, enc_over=None, cost_over=None, suite="libero_spatial", prefix=""):
    """A teacher cost table and a matching, fully bound encoder-cost (E) record."""
    cost = {"teacher": GROOT.name, "schedule_id": K8, "mode": "reduce-overhead", "certified": True,
            "stage1_ms": S1, "stage2_ms": S2, "stage3_full_loop_ms": L, "ckpt_sha256": CKPT_SHA,
            "gpu_uuid": GPU_UUID, "gpu_name": GPU_NAME, "prompt_shape_n": 566, "warmup": 30, "iters": 200}
    cost.update(cost_over or {})
    cp = tmp_path / f"{prefix}cost.json"
    cp.write_text(json.dumps(cost))
    enc = {"suite": suite, "teacher": GROOT.name, "cp2_key_encoder_ms": E, "teacher_cost_record_sha256": libs.sha256_file(cp),
           "schedule_id": K8, "mode": "reduce-overhead", "certified": True, "valid": True, "layout": LAYOUT,
           "model": {"weights_digest": MODEL_DIGEST, "checkpoint_dir": "/ckpt"}, "ckpt_weights_digest": MODEL_DIGEST,
           "ckpt_sha256": CKPT_SHA, "n_tokens": 566, "warmup": 30, "iters": 200, "gpu_uuid": GPU_UUID,
           "hardware": {"gpu": GPU_NAME, "torch": "2.5.1"}, "cudagraph_launch_count": 600,
           "expected_cudagraph_launch_count": 600}
    enc.update(enc_over or {})
    ep = tmp_path / f"{prefix}enc.json"
    ep.write_text(json.dumps(enc))
    return cp, ep


def test_groot_cost_record_binds_the_two_files(tmp_path):
    cp, ep = _cost_files(tmp_path)
    rec = libs.groot_cost_record(cp, ep, suite="libero_spatial")
    assert rec.encoder_ms == E and rec.schedule_id == K8 and rec.table == "measured"
    assert rec.provenance["cost_record_sha256"] == libs.sha256_file(cp)
    assert rec.provenance["encoder_cost_record_sha256"] == libs.sha256_file(ep)
    assert rec.provenance["layout"] == LAYOUT and rec.provenance["mode"] == "reduce-overhead"
    assert rec.provenance["model"]["weights_digest"] == MODEL_DIGEST and rec.provenance["ckpt_sha256"] == CKPT_SHA
    assert (rec.provenance["gpu_uuid"], rec.provenance["gpu_name"]) == (GPU_UUID, GPU_NAME)
    assert (rec.provenance["n_tokens"], rec.provenance["warmup"], rec.provenance["iters"]) == (566, 30, 200)


@pytest.mark.parametrize(
    "enc_over, cost_over, fragment",
    [
        ({"cp2_key_encoder_ms": 0.0}, None, "must be a finite number > 0"),
        ({"cp2_key_encoder_ms": -1.0}, None, "must be a finite number > 0"),
        ({"cp2_key_encoder_ms": True}, None, "must be a finite number > 0"),
        ({"cp2_key_encoder_ms": "2.5"}, None, "must be a finite number > 0"),
        ({"cp2_key_encoder_ms": float("nan")}, None, "must be a finite number > 0"),
        ({"teacher_cost_record_sha256": "0" * 64}, None, "does not match"),
        ({"valid": False}, None, "certified and valid"),
        ({"cudagraph_launch_count": 599}, None, "cudagraph_launch_count"),
        ({"cudagraph_launch_count": 0, "expected_cudagraph_launch_count": 0}, None, "cudagraph_launch_count"),
        ({"n_tokens": 565}, None, "encoder sampling"),
        ({"warmup": 5}, None, "encoder sampling"),
        ({"iters": 20}, None, "encoder sampling"),
        (None, {"prompt_shape_n": 512}, "teacher table sampling"),
        ({"ckpt_sha256": "short"}, None, "checkpoint by sha256"),
        (None, {"ckpt_sha256": "short"}, "checkpoint by sha256"),
        ({"gpu_uuid": "GPU-other"}, None, "gpu_uuid"),
        ({"hardware": {"gpu": "NVIDIA A100"}}, None, "hardware.gpu"),
        ({"model": {"weights_digest": "short"}, "ckpt_weights_digest": "short"}, None, "weights_digest"),
        ({"model": {"weights_digest": "e" * 64}}, None, "weights_digest"),
        ({"layout": {**LAYOUT, "kind": "other"}}, None, "layout"),
        ({"suite": "libero_10"}, None, "for suite 'libero_10'"),
        ({"certified": False}, None, "certified and valid under the same mode"),
        ({"mode": "eager"}, None, "certified and valid under the same mode"),
        (None, {"certified": False}, "certified and valid under the same mode"),
        ({"schedule_id": "groot_n15_k4_v1"}, None, "must be stamped"),
        (None, {"schedule_id": "groot_n15_k4_v1"}, "must be stamped"),
        ({"teacher": "pi05"}, None, "must be for teacher"),
        (None, {"teacher": "pi05"}, "must be for teacher"),
        (None, {"stage3_full_loop_ms": 0}, "missing or non-positive"),
    ],
)
def test_groot_cost_record_rejections(tmp_path, enc_over, cost_over, fragment):
    cp, ep = _cost_files(tmp_path, enc_over=enc_over, cost_over=cost_over)
    with pytest.raises(SystemExit, match=fragment):
        libs.groot_cost_record(cp, ep, suite="libero_spatial")


def test_groot_cost_record_requires_every_field(tmp_path):
    cp, ep = _cost_files(tmp_path)
    enc = json.loads(ep.read_text())
    del enc["layout"]
    ep.write_text(json.dumps(enc))
    with pytest.raises(SystemExit, match=r"lacks \['layout'\]"):
        libs.groot_cost_record(cp, ep, suite="libero_spatial")


def test_suites_share_teacher_table_but_keep_separate_encoder_checkpoints(tmp_path):
    cp, ep = _cost_files(tmp_path)
    spatial = libs.groot_cost_record(cp, ep, suite="libero_spatial")
    l10_raw = json.loads(ep.read_text())
    l10_raw.update(suite="libero_10", ckpt_sha256="e" * 64,
                   model={"weights_digest": "f" * 64}, ckpt_weights_digest="f" * 64,
                   cp2_key_encoder_ms=3.0)
    l10_path = tmp_path / "l10_encoder.json"
    l10_path.write_text(json.dumps(l10_raw))
    l10 = libs.groot_cost_record(cp, l10_path, suite="libero_10")
    assert spatial.teacher_forward_ms == l10.teacher_forward_ms
    assert spatial.provenance["cost_record_sha256"] == l10.provenance["cost_record_sha256"]
    assert spatial.provenance["teacher_ckpt_sha256"] == l10.provenance["teacher_ckpt_sha256"] == CKPT_SHA
    assert l10.provenance["ckpt_sha256"] == "e" * 64
    assert l10.provenance["model"]["weights_digest"] == "f" * 64
    assert libs.cost_record_from_summary(libs.cost_record_summary(l10, GROOT)).encoder_ms == 3.0


@pytest.mark.parametrize("change", [
    {"cost_record_sha256": ""}, {"encoder_cost_record_sha256": None}, {"teacher_ckpt_sha256": "short"},
    {"model": {}}, {"mode": "eager"}, {"n_tokens": 512}, {"gpu_name": "NVIDIA A100"},
    {"stage_ms": {"stage1": float("nan"), "stage2": S2, "stage3_full_loop": L}},
    {"stage_ms": {"stage1": S1, "stage2": -1, "stage3_full_loop": L}},
    {"layout": {**LAYOUT, "feature_dim": True}}, {"teacher_forward_ms": M + 1},
    {"verdict_unit_ms": {"FULL_HIT": P + E, "WARM_START@0.875": P + E, "MISS": M + E}},
])
def test_summary_rejects_invalid_provenance_values_and_inconsistent_prices(change):
    summary = libs.cost_record_summary(_record(), GROOT)
    with pytest.raises(SystemExit):
        libs.cost_record_from_summary({**summary, **change})


# ------------------------------------------------------------------
# IR addressing: preferred targets, fallback quartiles, failure
# ------------------------------------------------------------------


def test_inverse_addressing_uses_the_groot_units():
    rec = _record()
    s = np.linspace(0.0, 1.0, 2001)
    sol = ex.invert_ir(s, "n0", 60.0, table=rec, max_gap=1.0, profile=GROOT)
    assert sol is not None and abs(sol["predicted_ir"] - 60.0) <= 1.0
    admit = sol["admit_frac"]
    assert sol["predicted_ir"] == pytest.approx(100 * (admit * (P + E) + (1 - admit) * (M + E)) / M)
    assert ex.invert_ir(s, "n1", 200.0, table=rec, max_gap=1.0, profile=GROOT) is None


def test_plan_groot_tier_prefers_the_four_targets_when_reachable():
    rec = _record(0.5)  # small E: both tier floors sit below 45 %
    s = np.linspace(0.0, 1.0, 4001)
    for tier in ("n0", "n1"):
        assert ex.attainable_range(s, tier, rec, profile=GROOT)[0] < 45.0
        plan, sel = ex.plan_groot_tier(s, tier, table=rec, max_gap=1.0)
        assert sel["selection_rule"] == "preferred_targets" and sel["dropped"] == []
        assert [lbl for lbl, _ in plan] == ["t01", "t02", "t03", "t04"]
        assert [sol["target_ir"] for _, sol in plan] == [45.0, 60.0, 75.0, 90.0]
        assert len({sol["theta_raw"] for _, sol in plan}) == 4
        assert all(abs(sol["ir_gap"]) <= 1.0 for _, sol in plan)


def test_a_tier_floor_above_a_preferred_target_sends_the_whole_tier_to_the_fallback():
    """E raises the N_hit=1 floor (P + E + L/8 over M): with E = 2.5 ms it is
    46.7 %, so 45 % is unreachable and the tier as a whole uses the fallback
    rule -- never three preferred arms plus one filler."""
    rec = _record()
    s = np.linspace(0.0, 1.0, 4001)
    lo, _ = ex.attainable_range(s, "n1", rec, profile=GROOT)
    assert lo > 45.0
    plan, sel = ex.plan_groot_tier(s, "n1", table=rec, max_gap=1.0)
    assert sel["selection_rule"] == "fallback_quartiles"
    assert [d["reason"] for d in sel["dropped"]] == ["below_tier_floor"]
    assert [r["label"] for r in sel["preferred_resolved"]] == ["ir60", "ir75", "ir90"]
    assert [lbl for lbl, _ in plan] == ["t01", "t02", "t03", "t04"]


def test_plan_groot_tier_falls_back_to_the_frozen_quartile_rule():
    rec = _record()
    # Only six distinct scores: 45/60/75/90 cannot all be hit within 1 pt.
    s = np.repeat([0.10, 0.30, 0.50, 0.70, 0.80, 0.95], 10)
    plan, sel = ex.plan_groot_tier(s, "n0", table=rec, max_gap=1.0)
    assert sel["selection_rule"] == "fallback_quartiles" and sel["dropped"]
    assert [lbl for lbl, _ in plan] == ["t01", "t02", "t03", "t04"]
    cands = ex.candidate_cuts(s, "n0", rec, profile=GROOT)
    assert sel["candidate_range"]["n_candidates"] == 6
    irs = [c["predicted_ir"] for c in cands]
    lo, hi = irs[0], irs[-1]
    chosen = [sol["theta_raw"] for _, sol in plan]
    assert chosen[0] == cands[0]["theta_raw"] and chosen[-1] == cands[-1]["theta_raw"]
    for frac, theta in ((1 / 3, chosen[1]), (2 / 3, chosen[2])):
        goal = lo + frac * (hi - lo)
        pool = [c for c in cands if c["theta_raw"] not in (chosen[0], chosen[-1])]
        best = min(pool, key=lambda c: (abs(c["predicted_ir"] - goal), -c["theta_raw"]))
        assert theta == best["theta_raw"]
    assert all(sol["target_ir"] == sol["predicted_ir"] and sol["ir_gap"] == 0.0 for _, sol in plan)
    assert len(set(chosen)) == 4 and chosen == sorted(chosen)


def test_fallback_tie_prefers_the_higher_cut_and_too_few_cuts_fail():
    rec = _record()
    # Five distinct equally-spaced admit fractions -> the 1/3 goal sits exactly
    # between two candidates: the higher cut wins.
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    plan = ex.fallback_tier_targets(s, "n0", table=rec, profile=GROOT)
    thetas = [sol["theta_raw"] for _, sol in plan]
    assert thetas[0] == 0.1 and thetas[-1] == 0.6
    cands = ex.candidate_cuts(s, "n0", rec, profile=GROOT)
    irs = [c["predicted_ir"] for c in cands]
    goal = irs[0] + (irs[-1] - irs[0]) / 3  # between cands[1] (0.2) and cands[2] (0.3)? both at distance 1/6 of span
    d1, d2 = abs(irs[1] - goal), abs(irs[2] - goal)
    if abs(d1 - d2) < 1e-9:
        assert thetas[1] == 0.3
    with pytest.raises(SystemExit, match="only 3 distinct finite cuts"):
        ex.fallback_tier_targets(np.array([0.1, 0.1, 0.5, 0.9]), "n0", table=rec, profile=GROOT)


# ------------------------------------------------------------------
# preflight gate
# ------------------------------------------------------------------


def _preflight(tmp_path, name="overhead.json", **over):
    """A passing decision-overhead record bound to the test library / cohort / model."""
    rec = {"record_kind": "cp2_decision_overhead", "teacher": GROOT.name, "suite": "libero_spatial",
           "library_sha256": "d" * 64, "accepted_manifest_sha256": ACC_SHA, "projection": PROJ_META,
           "model": {"bound": {"weights_digest": MODEL_DIGEST}}, "schedule_id": K8, "stage1_path": GROOT.stage1_paths[0],
           "verdict": "ok_report", "timer_enabled": True, "n_decisions": 60, "cold_decisions": 50,
           "cold": {"p95": 9.0, "median": 6.0, "count": 50}, "warm": {"p95": 4.2, "median": 3.0, "count": 10},
           "per_segment": {s: {"count": 60, "median": 1.0, "p95": 1.5}
                           for s in ("cp2_encode", "cp2_collect", "cp2_build", "cp2_search", "cp2_judge", "cp2_fetch")}}
    rec.update(over)
    p = tmp_path / name
    p.write_text(json.dumps(rec))
    return p


@pytest.mark.parametrize("over, ok", [
    ({}, True),
    ({"verdict": "report_with_caption", "warm": {"p95": 25.0, "count": 10}}, True),
    # The label is recomputed from the measurement, never trusted.
    ({"verdict": "ok_report", "warm": {"p95": 41.0, "count": 10}}, False),
    ({"verdict": "ok_report", "warm": {"p95": 25.0, "count": 10}}, False),
    ({"verdict": "ok_report", "warm": {"p95": None, "count": 0}}, False),
    ({"verdict": "ok_report", "warm": {"p95": float("nan"), "count": 10}}, False),
    ({"verdict": "ok_report", "warm": {"p95": -1.0, "count": 10}}, False),
    ({"stage1_path": "unverified_template"}, False),
    ({"verdict": "halt_profile_segments", "warm": {"p95": 41.0, "count": 10}}, False),
    ({"verdict": "insufficient_decisions"}, False),
    ({"suite": "libero_10"}, False), ({"library_sha256": "e" * 64}, False),
    ({"accepted_manifest_sha256": "2" * 64}, False), ({"projection": {**PROJ_META, "seed": 8}}, False),
    ({"model": {"bound": {"weights_digest": "e" * 64}}}, False), ({"schedule_id": "groot_n15_k4_v1"}, False),
    ({"record_kind": "cp2_encoder_cost"}, False), ({"teacher": "pi05"}, False),
    ({"timer_enabled": False}, False), ({"n_decisions": 50, "warm": {"p95": 4.2, "count": 0}}, False),
    ({"cold_decisions": 40, "warm": {"p95": 4.2, "count": 20}}, False),
    ({"warm": {"p95": 4.2, "count": 9}}, False),  # count != n_decisions - cold
    ({"per_segment": {"cp2_collect": {"count": 60}, "cp2_build": {"count": 60}, "cp2_search": {"count": 60}, "cp2_judge": {"count": 60}}}, False),
    ({"per_segment": {s: {"count": 59} for s in ("cp2_encode", "cp2_collect", "cp2_build", "cp2_search", "cp2_judge")}}, False),
])
def test_check_preflight_record(tmp_path, over, ok):
    p = _preflight(tmp_path, **over)
    kw = dict(suite="libero_spatial", library_sha256="d" * 64, accepted_manifest_sha256=ACC_SHA, projection=PROJ_META,
              library_model_digest=MODEL_DIGEST)
    if ok:
        out = ex.check_preflight_record(p, **kw)
        assert out["sha256"] == libs.sha256_file(p) and out["warm_total_p95_ms"] == over.get("warm", {}).get("p95", 4.2)
        assert out["verdict"] == libs.preflight_verdict(out["warm_total_p95_ms"]) and out["n_decisions"] == 60
    else:
        with pytest.raises(SystemExit, match="preflight record rejected"):
            ex.check_preflight_record(p, **kw)


@pytest.mark.parametrize("stat,value", [("median", -0.1), ("p95", float("nan")), ("p95", None)])
def test_preflight_refuses_invalid_segment_measurements(tmp_path, stat, value):
    p = _preflight(tmp_path)
    rec = json.loads(p.read_text())
    rec["per_segment"]["cp2_encode"][stat] = value
    p.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="cp2_encode"):
        ex.check_preflight_record(p, suite="libero_spatial", library_sha256="d" * 64)


# ------------------------------------------------------------------
# export: ten arms, gated on the records
# ------------------------------------------------------------------


def _library(tmp_path, name="cp2.pkl", **over):
    entries = [CacheEntry(id=f"e{i}", checkpoint_id=CheckpointID.CP2, query_keys={libs.FIELD: torch.randn(8)},
                          payload=CachePayload(action_chunk=torch.zeros(16, 32), intermediates={t / 8: torch.zeros(16, 32) for t in range(1, 8)},
                                               denoising_num_steps=8, task_key="t", schedule_id=K8),
                          step_idx=i, trajectory_id="traj", prev_ids=[], next_ids=[]) for i in range(4)]
    art = {"key_builder_type": "cp2_groot_ternary", "checkpoint_id": "CP2", "vector_dims": {libs.FIELD: 8},
           "entries": entries, "projection": PROJ_META, "id_policy": libs.ID_POLICY, "schedule_id": K8,
           "teacher": GROOT.name, "stage1_path": "groot_reconstructed_template",
           "model": {"weights_digest": "c" * 64, "checkpoint_dir": "/ckpt"}, "library_stats": {"n": 4}}
    art.update(over)
    p = tmp_path / name
    with open(p, "wb") as f:
        pickle.dump(art, f)
    return p


def _shadow(tmp_path, s, *, teacher=GROOT.name, library_sha=None, name="shadow.jsonl", record=True, **over):
    """A shadow table with the complete sidecar ``build_shadow_table_groot.py`` writes."""
    p = tmp_path / name
    # Preserve the score distribution while covering the real 10 x 15 cohort.
    if 0 < len(s) < 150:
        s = np.tile(s, int(np.ceil(150 / len(s))))
    p.write_text("\n".join(json.dumps({
        "episode": f"e_{(i % 150) // 15}_{i % 15}", "task": f"instruction {(i % 150) // 15}",
        "task_id": (i % 150) // 15, "subset": i % 15, "orig": 2 * (i % 15) + 1,
        "step_idx": i // 150, "s_raw": float(v), "success": True, "winner_id": "e0",
    }) for i, v in enumerate(s)) + "\n")
    rec = {"teacher": teacher, "suite": "libero_spatial", "library_sha256": library_sha or "d" * 64,
           "accepted_manifest_sha256": ACC_SHA, "cohort_manifest_sha256": "3" * 64, "task_map_sha256": "4" * 64,
           "cohort_episodes": 150, "cohort_expected": 150, "complete": True, "limited": False,
           "n_rows": int(len(s)), "out_jsonl_sha256": libs.sha256_file(p), "projection": PROJ_META, "schedule_id": K8,
           "stage1_path": GROOT.stage1_paths[0],
           "model": {"weights_digest": MODEL_DIGEST, "bound_to_library": True, "library_model": {"weights_digest": MODEL_DIGEST}}}
    rec.update(over)
    if record:
        p.with_suffix(".record.json").write_text(json.dumps(rec))
    return p


def _export_args(tmp_path, lib, shadow, cost, enc, pre, **over):
    d = dict(teacher=GROOT.name, suite="libero_spatial", lib_tag="w13s3", shadow_table=str(shadow), library_pkl=str(lib),
             deploy_library_path="/srv/libs/cp2.pkl", out_dir=str(tmp_path / "out"), tiers="n0,n1", targets="",
             cost_table="measured", max_gap=1.0, ref_theta=None, cost_record=str(cost) if cost else "",
             encoder_cost_record=str(enc) if enc else "", preflight_record=str(pre) if pre else "")
    d.update(over)
    return argparse.Namespace(**d)


@pytest.mark.parametrize("change", [
    {"tiers": "n0"}, {"tiers": "n1"}, {"tiers": "n0,n0"}, {"tiers": "n0,n1,n1"},
    {"tiers": "n0,n2"}, {"ref_theta": 0.7}, {"max_gap": float("nan")}, {"max_gap": 2.0},
])
def test_groot_export_rejects_nonfrozen_matrix_before_writing_any_arm(tmp_path, change):
    lib = _library(tmp_path)
    sha = libs.sha256_file(lib)
    shadow = _shadow(tmp_path, np.linspace(0, 1, 4001), library_sha=sha)
    cost, enc = _cost_files(tmp_path)
    pre = _preflight(tmp_path, library_sha256=sha)
    args = _export_args(tmp_path, lib, shadow, cost, enc, pre, **change)
    with pytest.raises(SystemExit):
        ex.export(args)
    assert not pathlib.Path(args.out_dir).exists()


def test_fractional_ir_targets_emit_ten_unique_configs(tmp_path):
    lib = _library(tmp_path)
    sha = libs.sha256_file(lib)
    shadow = _shadow(tmp_path, np.linspace(0, 1, 4001), library_sha=sha)
    cost, enc = _cost_files(tmp_path, enc_over={"cp2_key_encoder_ms": 0.5})
    pre = _preflight(tmp_path, library_sha256=sha)
    args = _export_args(tmp_path, lib, shadow, cost, enc, pre, targets="45.1,45.2,75.1,90.1")
    rec = ex.export(args)
    assert len(rec["arms"]) == len(list((pathlib.Path(args.out_dir) / "arms").glob("*.yaml"))) == 10
    for tier in ("n0", "n1"):
        targets = [a for a in rec["arms"].values() if a["tier"] == tier and a["target_ir"] is not None]
        assert len({a["theta_raw"] for a in targets}) == 4
        assert {libs.parse_arm(a["arm_id"])["target"] for a in targets} == {"t01", "t02", "t03", "t04"}


@pytest.mark.parametrize("mutation", ["one_episode", "duplicate_decision", "step_gap", "invalid_cosine", "orig_conflict"])
def test_shadow_table_rows_must_prove_complete_consistent_cohort(tmp_path, mutation):
    path = _shadow(tmp_path, np.linspace(0, 1, 301))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "one_episode":
        for i, row in enumerate(rows):
            row.update(episode="one", task="one instruction", task_id=0, subset=0, orig=1, step_idx=i)
    elif mutation == "duplicate_decision":
        rows[-1] = rows[0]
    elif mutation == "step_gap":
        rows[-1]["step_idx"] += 1
    elif mutation == "invalid_cosine":
        rows[-1]["s_raw"] = float("nan")
    else:
        rows[-1]["orig"] = 4
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    sidecar = json.loads(path.with_suffix(".record.json").read_text())
    sidecar["out_jsonl_sha256"] = libs.sha256_file(path)
    with pytest.raises(SystemExit, match="shadow record rejected"):
        ex.check_shadow_record(sidecar, shadow_path=path, n_rows=len(rows), suite="libero_spatial",
                               library_sha256="d" * 64, projection=PROJ_META, library_model_digest=MODEL_DIGEST)


def test_groot_export_emits_exactly_ten_arms_with_the_bound_pricing(tmp_path):
    lib = _library(tmp_path)
    lib_sha = libs.sha256_file(lib)
    shadow = _shadow(tmp_path, np.linspace(0.0, 1.0, 4001), library_sha=lib_sha)
    cost, enc = _cost_files(tmp_path)
    pre = _preflight(tmp_path, library_sha256=lib_sha)
    rec = ex.export(_export_args(tmp_path, lib, shadow, cost, enc, pre))
    arms = rec["arms"]
    assert len(arms) == 10 and rec["budget"]["arms_per_tier"] == {"n0": 5, "n1": 5}
    assert sorted(a for a in arms if a.endswith("ref650")) == ["acb_sp_w13s3_n0_ref650", "acb_sp_w13s3_n1_ref650"]
    assert all(arms[a]["theta_raw"] == 0.65 for a in arms if a.endswith("ref650"))
    assert {a["hit_type"] for a in arms.values() if a["tier"] == "n1"} == {"WARM_START"}
    assert {a["start_t"] for a in arms.values() if a["tier"] == "n1"} == {0.875}
    assert rec["teacher"] == GROOT.name and rec["cost"]["encoder_ms"] == E and rec["cost"]["schedule_id"] == K8
    assert rec["cost"]["model"]["weights_digest"] == MODEL_DIGEST and rec["cost"]["ckpt_sha256"] == CKPT_SHA
    assert rec["preflight"]["verdict"] == "ok_report" and rec["schedule_id"] == K8
    assert rec["shadow_binding"]["accepted_manifest_sha256"] == ACC_SHA and rec["shadow_binding"]["n_rows"] == 4001
    assert rec["selection"]["n0"]["selection_rule"] == "preferred_targets"
    assert rec["library_model"]["weights_digest"] == "c" * 64
    matrix = yaml.safe_load((tmp_path / "out" / "arm_matrix.yaml").read_text())
    assert matrix["teacher"] == GROOT.name and matrix["checkpoint"] == "cp2" and len(matrix["arms"]) == 10
    doc = yaml.safe_load(pathlib.Path(arms["acb_sp_w13s3_n1_ref650"]["yaml"]).read_text())
    assert doc["denoise_schedule"] == K8 and doc["key_builder"]["type"] == "cp2_groot_ternary"
    assert doc["key_builder"]["cp2_groot"]["token_len"] == 5 and doc["backend"]["in_memory"]["preload_path"] == "/srv/libs/cp2.pkl"
    assert doc["checkpoints"]["cp2"]["judge"]["warm_tiers"] == [{"threshold": libs.theta_norm(0.65), "start_t": 0.875}]
    # The record prices with E and the same IR the aggregate will recompute.
    a = arms["acb_sp_w13s3_n0_t02"]
    assert a["predicted_ir"] == pytest.approx(100 * (a["admit_frac"] * (P + E) + (1 - a["admit_frac"]) * (M + E)) / M)


def test_groot_export_falls_back_and_refuses_without_records(tmp_path):
    lib = _library(tmp_path)
    lib_sha = libs.sha256_file(lib)
    sparse = _shadow(tmp_path, np.repeat([0.1, 0.3, 0.5, 0.7, 0.8, 0.95], 10), library_sha=lib_sha)
    cost, enc = _cost_files(tmp_path)
    pre = _preflight(tmp_path, library_sha256=lib_sha)
    rec = ex.export(_export_args(tmp_path, lib, sparse, cost, enc, pre))
    assert rec["selection"]["n0"]["selection_rule"] == "fallback_quartiles"
    assert sorted(a for a in rec["arms"] if "_n0_" in a) == ["acb_sp_w13s3_n0_ref650"] + [f"acb_sp_w13s3_n0_t0{i}" for i in range(1, 5)]
    for missing in ("cost_record", "encoder_cost_record", "preflight_record"):
        with pytest.raises(SystemExit, match="needs --cost-record"):
            ex.export(_export_args(tmp_path, lib, sparse, cost, enc, pre, **{missing: ""}))
    halted = _preflight(tmp_path, "halted.json", library_sha256=lib_sha, verdict="halt_profile_segments")
    with pytest.raises(SystemExit, match="preflight record rejected"):
        ex.export(_export_args(tmp_path, lib, sparse, cost, enc, halted))
    cost_other, enc_other = _cost_files(tmp_path, enc_over={"layout": {**LAYOUT, "token_len": 6}}, prefix="other_")
    with pytest.raises(SystemExit, match="layout"):
        ex.export(_export_args(tmp_path, lib, sparse, cost_other, enc_other, pre))
    too_few = _shadow(tmp_path, np.repeat([0.1, 0.5, 0.9], 10), library_sha=lib_sha, name="few.jsonl")
    with pytest.raises(SystemExit, match="only 3 distinct finite cuts"):
        ex.export(_export_args(tmp_path, lib, too_few, cost, enc, pre))
    pi05_shadow = _shadow(tmp_path, np.linspace(0, 1, 101), teacher="pi05", library_sha=lib_sha, name="pi.jsonl")
    with pytest.raises(SystemExit, match="teacher 'pi05'"):
        ex.export(_export_args(tmp_path, lib, pi05_shadow, cost, enc, pre))
    # Encoder cost measured on another model than the library's.
    cost_m, enc_m = _cost_files(tmp_path, enc_over={"model": {"weights_digest": "e" * 64}, "ckpt_weights_digest": "e" * 64},
                                prefix="model_")
    with pytest.raises(SystemExit, match="measured on model"):
        ex.export(_export_args(tmp_path, lib, sparse, cost_m, enc_m, pre))
    # Preflight bound to another cohort / model than the shadow table's.
    other_cohort = _preflight(tmp_path, "cohort.json", library_sha256=lib_sha, accepted_manifest_sha256="9" * 64)
    with pytest.raises(SystemExit, match="accepted_manifest_sha256 differs"):
        ex.export(_export_args(tmp_path, lib, sparse, cost, enc, other_cohort))


def test_groot_export_refuses_shadow_tables_without_frozen_provenance(tmp_path):
    lib = _library(tmp_path)
    lib_sha = libs.sha256_file(lib)
    cost, enc = _cost_files(tmp_path)
    pre = _preflight(tmp_path, library_sha256=lib_sha)
    s = np.linspace(0.0, 1.0, 4001)
    cases = [
        (dict(record=False), "no .record.json sidecar"),
        (dict(limited=True), "not complete"),
        (dict(complete=False), "not complete"),
        (dict(cohort_episodes=149), "not complete"),
        (dict(cohort_expected=149, cohort_episodes=149), "cohort_expected 149"),
        (dict(n_rows=4000), "n_rows 4000 != 4001"),
        (dict(suite="libero_10"), "suite 'libero_10'"),
        (dict(projection={**PROJ_META, "seed": 8}), "projection differs"),
        (dict(schedule_id="groot_n15_k4_v1"), "schedule_id / stage1_path"),
        (dict(model={"weights_digest": "e" * 64, "bound_to_library": True, "library_model": {"weights_digest": "e" * 64}}),
         "weights_digest"),
        (dict(model={"weights_digest": MODEL_DIGEST, "bound_to_library": False, "library_model": {"weights_digest": MODEL_DIGEST}}),
         "weights_digest"),
    ]
    for i, (over, fragment) in enumerate(cases):
        shadow = _shadow(tmp_path, s, library_sha=lib_sha, name=f"s{i}.jsonl", **over)
        with pytest.raises(SystemExit, match=fragment):
            ex.export(_export_args(tmp_path, lib, shadow, cost, enc, pre, out_dir=str(tmp_path / f"out{i}")))
        assert not (tmp_path / f"out{i}" / "arms").exists() or not list((tmp_path / f"out{i}" / "arms").glob("*.yaml"))
    # Truncated after recording: the sidecar's content hash no longer matches the table.
    shadow = _shadow(tmp_path, s, library_sha=lib_sha, name="trunc.jsonl")
    shadow.write_text("\n".join(shadow.read_text().splitlines()[::2]) + "\n")
    with pytest.raises(SystemExit, match="out_jsonl_sha256 differs"):
        ex.export(_export_args(tmp_path, lib, shadow, cost, enc, pre))
    # A missing field is a rejection, not a KeyError.
    shadow = _shadow(tmp_path, s, library_sha=lib_sha, name="nofield.jsonl")
    rec = json.loads(shadow.with_suffix(".record.json").read_text())
    del rec["accepted_manifest_sha256"]
    shadow.with_suffix(".record.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="missing field 'accepted_manifest_sha256'"):
        ex.export(_export_args(tmp_path, lib, shadow, cost, enc, pre))


# ------------------------------------------------------------------
# aggregate under the GR00T pricing
# ------------------------------------------------------------------


def _write_run(tmp_path, arm, tier_rows, *, lib_sha, n_ep=4, n_rows=42):
    run = tmp_path / arm
    run.mkdir()
    journal, per_step = [], []
    for i in range(n_ep):
        uid = f"{arm}:eval:{i % 2}:{i}"
        journal.append({"task_uid": uid, "yaml_id": arm, "phase": "eval", "status": "done" if i % 3 else "failed",
                        "success": bool(i % 3), "attempt": 1, "accepted": True})
        for step in range(n_rows):
            ht, st = tier_rows[step % len(tier_rows)]
            per_step.append({"yaml_id": arm, "task_id": i % 2, "task_uid": uid, "step_idx": step, "hit_type": ht,
                             "start_t": st, "attempt": 1, "checkpoint": "CP2", "score": 0.9, "library_sha256": lib_sha})
        per_step.append({"_kind": "client_timing", "task_uid": uid, "yaml_id": arm, "task_id": i % 2, "attempt": 1,
                         "steps": libs.STEP_CAP["libero_spatial"]})
    (run / "journal.jsonl").write_text("\n".join(json.dumps(r) for r in journal) + "\n")
    (run / "per_step.jsonl").write_text("\n".join(json.dumps(r) for r in per_step) + "\n")
    return run


def _groot_export_record(*arms, lib_sha, suite="libero_spatial"):
    rec = _record(suite=suite)
    return {"teacher": GROOT.name, "suite": suite, "schedule_id": K8, "library_sha256": lib_sha,
            "library_model": {"weights_digest": MODEL_DIGEST}, "projection": PROJ_META,
            "cost": libs.cost_record_summary(rec, GROOT),
            "arms": {a: {"target_ir": 60.0, "predicted_ir": 61.0, "theta_raw": 0.7, "theta_norm": 0.85} for a in arms}}


def test_aggregate_prices_groot_arms_with_E_and_refuses_unbound_records(tmp_path):
    lib_sha = "f" * 64
    arm = "acb_sp_w13s3_n1_ir60"
    run = _write_run(tmp_path, arm, [("WARM_START", 0.875), ("MISS", None)], lib_sha=lib_sha)
    res = aggregate(run, expect_episodes=4, export_record=_groot_export_record(arm, lib_sha=lib_sha))
    a = res["arms"][arm]
    assert res["teacher"] == GROOT.name and set(res["cost"]) == {"measured"}
    assert a["ir_percent"] == pytest.approx(100 * ((P + E + L / 8) + (M + E)) / (2 * M))
    assert a["counts"]["WARM_START@0.875"] == 84 and "ir_percent_eager" not in a
    all_miss = _write_run(tmp_path, "acb_sp_w13s3_n0_ir90", [("MISS", None)], lib_sha=lib_sha)
    res = aggregate(all_miss, expect_episodes=4, export_record=_groot_export_record("acb_sp_w13s3_n0_ir90", lib_sha=lib_sha))
    assert res["arms"]["acb_sp_w13s3_n0_ir90"]["ir_percent"] == pytest.approx(100 * (M + E) / M)
    assert res["arms"]["acb_sp_w13s3_n0_ir90"]["ir_percent"] > 100
    wrong_t = _write_run(tmp_path, "acb_sp_w13s3_n1_ir70", [("WARM_START", 0.1)], lib_sha=lib_sha)
    with pytest.raises(SystemExit, match="unpriceable verdict WARM_START@0.1 under groot_libero"):
        aggregate(wrong_t, expect_episodes=4, export_record=_groot_export_record("acb_sp_w13s3_n1_ir70", lib_sha=lib_sha))
    # A resumable-but-wrong tier (0.5 is a k=8 snapshot) is priced, then caught by the purity gate.
    wrong_tier = _write_run(tmp_path, "acb_sp_w13s3_n1_ir80", [("WARM_START", 0.5)], lib_sha=lib_sha)
    with pytest.raises(SystemExit, match="N_hit=1 is WARM_START@0.875"):
        aggregate(wrong_tier, expect_episodes=4, export_record=_groot_export_record("acb_sp_w13s3_n1_ir80", lib_sha=lib_sha))
    with pytest.raises(SystemExit, match="must carry its bound cost summary"):
        pricing_for({"teacher": GROOT.name, "library_sha256": lib_sha, "arms": {}})
    unbound = _groot_export_record(arm, lib_sha=lib_sha)
    unbound["cost"]["encoder_ms"] = 0.0
    with pytest.raises(SystemExit, match="not a finite positive E"):
        pricing_for(unbound)
    # E belongs to the other suite: the arms / library are right, the pricing is not.
    cross = _groot_export_record(arm, lib_sha=lib_sha)
    cross["cost"]["suite"] = "libero_10"
    with pytest.raises(SystemExit, match="export record suite differs from the bound encoder cost"):
        aggregate(run, expect_episodes=4, export_record=cross)
    wrong_record_suite = _groot_export_record(arm, lib_sha=lib_sha, suite="libero_10")
    with pytest.raises(SystemExit, match="export record suite 'libero_10'"):
        aggregate(run, expect_episodes=4, export_record=wrong_record_suite)
    prof, pricing = pricing_for(None)
    assert prof is libs.PI05 and set(pricing) == {"cuda_graph", "eager"}


def test_aggregate_explicit_suite_must_match_arm_ids(tmp_path):
    arm = "acb_sp_w13s3_n0_t01"
    run = _write_run(tmp_path, arm, [("MISS", None)], lib_sha="f" * 64)
    rec = _groot_export_record(arm, lib_sha="f" * 64, suite="libero_10")
    with pytest.raises(SystemExit, match="differs from arm suite"):
        aggregate(run, expect_episodes=4, export_record=rec, suite="libero_10")


@pytest.mark.parametrize("change", [{"library_model": {"weights_digest": "e" * 64}},
                                    {"projection": {**PROJ_META, "layout": {**LAYOUT, "token_len": 6}}}])
def test_pricing_rebinds_encoder_model_and_layout_to_export_library(change):
    rec = _groot_export_record("acb_sp_w13s3_n0_t01", lib_sha="f" * 64)
    with pytest.raises(SystemExit, match="differs from the export library"):
        pricing_for({**rec, **change}, suite="libero_spatial")


# ------------------------------------------------------------------
# verifier negatives on the schedule fields (GR00T profile)
# ------------------------------------------------------------------


def _source(tmp_path, entries_from, schedule=K8):
    src_entries = [CacheEntry(id=e.id, checkpoint_id=CheckpointID.CP1, query_keys={"vision_0": torch.zeros(2)},
                              payload=e.payload, step_idx=e.step_idx, trajectory_id=e.trajectory_id,
                              prev_ids=e.prev_ids, next_ids=e.next_ids) for e in entries_from]
    p = tmp_path / "src.pkl"
    with open(p, "wb") as f:
        pickle.dump({"key_builder_type": "cp1_groot_libero_spatial_pool_16", "checkpoint_id": "CP1",
                     "vector_dims": {"vision_0": 2}, "entries": src_entries, "schedule_id": schedule}, f)
    return p


def _verifiable(tmp_path, **over):
    lib = _library(tmp_path)
    art = libs.load_pickle(lib)
    src = _source(tmp_path, art["entries"])
    art.update({"source_pkl_sha256": libs.sha256_file(src), "h5_manifest": {"files": []}, "tokenizer": {"n": 1},
                "build_git_commit": "abc"})
    art.update(over)
    with open(lib, "wb") as f:
        pickle.dump(art, f)
    return lib, src


def test_verifier_accepts_a_groot_artifact_and_rejects_schedule_drift(tmp_path):
    lib, src = _verifiable(tmp_path)
    assert verify(str(lib), str(src), teacher=GROOT.name, search_samples=4)["ok"]
    cases = [
        ({"schedule_id": "groot_n15_k4_v1"}, "top-level schedule_id"),
        ({"teacher": "pi05"}, "teacher 'pi05'"),
        ({"stage1_path": "online"}, "stage1_path"),
        ({"key_builder_type": "cp2_vlm_ternary"}, "key_builder_type"),
        ({"projection": {k: v for k, v in PROJ_META.items() if k != "layout"}}, "projection metadata"),
        ({"model": {"weights_digest": "short"}}, "weights_digest"),
    ]
    for over, fragment in cases:
        bad, _ = _verifiable(tmp_path, **over)
        with pytest.raises(VerificationError, match=fragment):
            verify(str(bad), str(src), teacher=GROOT.name, search_samples=4)
    # Payload stamps: the same k=4 payload on both sides (so the copy check passes)
    # inside an otherwise k=8-stamped artifact.
    lib, src = _verifiable(tmp_path)
    art = libs.load_pickle(lib)
    art["entries"][0].payload = copy.deepcopy(art["entries"][0].payload)
    art["entries"][0].payload.schedule_id = "groot_n15_k4_v1"
    stale_src = _source(tmp_path, art["entries"])
    art["source_pkl_sha256"] = libs.sha256_file(stale_src)
    with open(lib, "wb") as f:
        pickle.dump(art, f)
    with pytest.raises(VerificationError, match="payload of e0 is stamped"):
        verify(str(lib), str(stale_src), teacher=GROOT.name, search_samples=4)
    # Source without the schedule is refused too.
    lib, src = _verifiable(tmp_path)
    bad_src = _source(tmp_path, libs.load_pickle(lib)["entries"], schedule="pi05_v1")
    art = libs.load_pickle(lib)
    art["source_pkl_sha256"] = libs.sha256_file(bad_src)
    with open(lib, "wb") as f:
        pickle.dump(art, f)
    with pytest.raises(VerificationError, match="source schedule_id"):
        verify(str(lib), str(bad_src), teacher=GROOT.name, search_samples=4)


# ------------------------------------------------------------------
# cross-line reference pricing (not executed this round; the units are frozen)
# ------------------------------------------------------------------


def test_reference_line_is_priced_without_the_encoder_over_the_same_denominator():
    from exp.actioncache_baseline.compare_to_reference import cp1_reference_cost

    rec = _record()
    cost = cp1_reference_cost(rec)
    assert cost("FULL_HIT", None) == pytest.approx(S1)
    assert cost("WARM_START", 0.75) == pytest.approx(P + L * 2 / 8)
    assert cost("WARM_START", 0.5) == pytest.approx(P + L / 2)
    assert cost("MISS", None) == pytest.approx(M)
    assert libs.teacher_forward_cost(rec) == pytest.approx(M)
    with pytest.raises(ValueError):
        cost("WARM_START", 0.1)  # not a k=8 snapshot
    pi = cp1_reference_cost(libs.pi05_cost_record())
    assert pi("WARM_START", 0.5) == pytest.approx(libs.cp2_tier_cost("WARM_START", 0.5))
    assert pi("FULL_HIT", None) == pytest.approx(libs.COST_TABLES["cuda_graph"]["stage1"])


def _ref_run(tmp_path, arm, *, start_t=0.75, n_ep=4):
    """A CP1-line (RIT) reference run directory: journal + per_step without CP2 fields."""
    run = tmp_path / f"ref_{arm}"
    run.mkdir()
    journal, per_step = [], []
    for i in range(n_ep):
        uid = f"{arm}:eval:{i % 2}:{i}"
        journal.append({"task_uid": uid, "yaml_id": arm, "phase": "eval", "status": "done", "success": bool(i % 2),
                        "attempt": 1, "accepted": True})
        for step in range(10):
            ht, st = (("WARM_START", start_t), ("MISS", None), ("FULL_HIT", None))[step % 3]
            per_step.append({"yaml_id": arm, "task_id": i % 2, "task_uid": uid, "step_idx": step, "hit_type": ht,
                             "start_t": st, "attempt": 1})
    (run / "journal.jsonl").write_text("\n".join(json.dumps(r) for r in journal) + "\n")
    (run / "per_step.jsonl").write_text("\n".join(json.dumps(r) for r in per_step) + "\n")
    return run


def test_compare_binds_the_groot_reference_record_and_reprices_the_reference(tmp_path):
    from exp.actioncache_baseline.compare_to_reference import check_groot_reference, compare

    lib_sha = "f" * 64
    arm = "acb_sp_w13s3_n0_ir60"
    run = _write_run(tmp_path, arm, [("FULL_HIT", None), ("MISS", None)], lib_sha=lib_sha)
    export_record = _groot_export_record(arm, lib_sha=lib_sha)
    ref = _ref_run(tmp_path, "sp_rit_k2_ir60")
    ref_record = {"protocol": "libero_groot_rit_arms_v1", "suite": "libero_spatial", "denoising_steps": 8,
                  "arms": {"sp_rit_k2_ir60": {}}, "cost": {"miss_ms": 47.3, "provenance": "INTERIM"}}
    with pytest.raises(SystemExit, match="needs --ref-record"):
        compare(run, ref, export_record=export_record, expect_episodes=4, allow_partial=False, B=20, seed=0)
    res = compare(run, ref, export_record=export_record, expect_episodes=4, allow_partial=False, B=20, seed=0,
                  ref_record=ref_record)
    assert res["suite"] == "libero_spatial" and res["reference_binding"]["interim_cost_not_used"]["miss_ms"] == 47.3
    assert res["reference_binding"]["n_arms"] == 1 and res["reference_arms"] == ["sp_rit_k2_ir60"]
    assert res["cost"]["encoder_ms"] == E  # the CP2 line's record prices both sides
    assert arm in res["comparisons"]
    for bad, fragment in [
        ({**ref_record, "suite": "libero_10"}, "suite 'libero_10'"),
        ({**ref_record, "denoising_steps": 4}, "denoising_steps 4"),
        ({**ref_record, "protocol": "other"}, "protocol 'other'"),
        ({**ref_record, "arms": {}}, "not in the arm record"),
    ]:
        with pytest.raises(SystemExit, match=fragment):
            check_groot_reference(bad, suite="libero_spatial", ref_arms=["sp_rit_k2_ir60"])
    # A reference verdict the k=8 loop cannot price (Pi0.5-style start_t) is refused, not mis-priced.
    ref_bad = _ref_run(tmp_path, "sp_rit_k2_ir65", start_t=0.1)
    with pytest.raises(SystemExit, match="not priceable under groot_n15_k8_v1"):
        compare(run, ref_bad, export_record=export_record, expect_episodes=4, allow_partial=False, B=20, seed=0,
                ref_record={**ref_record, "arms": {"sp_rit_k2_ir65": {}}})
