"""Experiment tooling of exp/actioncache_baseline: IR inversion, artifact verifier, stats, aggregate."""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.actioncache_baseline import libs, parity_check, stats
from exp.actioncache_baseline.aggregate import aggregate
from exp.actioncache_baseline.bench_cp2_overhead import verdict_for
from exp.actioncache_baseline.build_cp2_artifact import _entry_copy_with_key
from exp.actioncache_baseline.compare_to_reference import compare
from exp.actioncache_baseline.export_arms import (
    DEFAULT_TARGETS,
    GROUP_ARM_CAP,
    TIER_TARGET_CAP,
    attainable_range,
    invert_ir,
    ir_percent,
    plan_tier_targets,
)
from exp.actioncache_baseline.verify_cp2_artifact import VerificationError, verify
from exp.dispatch_surface.analysis.analytic_cost import STAGE1_MS, STAGE2_MS, STAGE3_MS
from openpi.cache.components.cp2_vlm_key_builder import get_projection_spec, project
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


# ------------------------------------------------------------------
# libs
# ------------------------------------------------------------------


def test_cost_tables_and_theta_mapping():
    assert libs.cp2_tier_cost("FULL_HIT", None) == pytest.approx(STAGE1_MS + STAGE2_MS)
    assert libs.cp2_tier_cost("WARM_START", 0.1) == pytest.approx(STAGE1_MS + STAGE2_MS + 0.1 * STAGE3_MS)
    assert libs.miss_cost() == pytest.approx(STAGE1_MS + STAGE2_MS + STAGE3_MS)
    assert libs.cp2_tier_cost("FULL_HIT", None, "eager") == pytest.approx(63.06 + 35.27)
    with pytest.raises(ValueError):
        libs.cp2_tier_cost("WARM_START", None)
    assert libs.theta_norm(0.85) == pytest.approx(0.925)
    assert libs.theta_raw(libs.theta_norm(0.3)) == pytest.approx(0.3)
    arm = libs.arm_name("libero_spatial", "lib50", "n1", "ir60")
    assert arm == "acb_sp_lib50_n1_ir60"
    assert libs.parse_arm(arm) == {"suite_tag": "sp", "lib": "lib50", "tier": "n1", "target": "ir60"}
    assert libs.parse_arm("rit_sp_ng_ir20") is None


# ------------------------------------------------------------------
# IR inversion (GST K=1)
# ------------------------------------------------------------------


def test_invert_ir_monotone_gap_and_reference():
    s = np.linspace(0.5, 0.99, 1000)
    lo, hi = attainable_range(s, "n0")
    assert lo == pytest.approx(100 * (STAGE1_MS + STAGE2_MS) / libs.miss_cost()) and hi == 100.0
    assert invert_ir(s, "n0", 10.0) is None  # below the attainable floor
    prev = -np.inf
    for target in (60, 70, 80, 90):
        sol = invert_ir(s, "n0", target)
        assert sol is not None and abs(sol["ir_gap"]) <= 1.0
        assert sol["theta_raw"] > prev  # higher IR target -> higher cut (fewer admits)
        prev = sol["theta_raw"]
        assert ir_percent(s, sol["theta_raw"], "n0") == pytest.approx(sol["predicted_ir"])
    top = invert_ir(s, "n0", 100.0)
    assert top is not None and not np.isfinite(top["theta_raw"]) and top["predicted_ir"] == 100.0
    # n1 admits cost more than n0 at the same cut: same theta -> higher IR.
    assert ir_percent(s, 0.8, "n1") > ir_percent(s, 0.8, "n0")


def test_tier_budget_is_mechanical_on_the_default_ladder():
    """G2 R1 counterexample: 10,000 uniform shadow scores + defaults gave 18 arms."""
    s = np.random.default_rng(0).uniform(0.0, 1.0, 10_000)
    skipped: list[dict] = []
    plans = {t: plan_tier_targets(s, t, DEFAULT_TARGETS, table="cuda_graph", max_gap=1.0,
                                  skipped=skipped, cap=TIER_TARGET_CAP[t]) for t in ("n0", "n1")}
    assert len(plans["n0"]) == 8 and len(plans["n1"]) == 7
    assert len(plans["n0"]) + len(plans["n1"]) + 2 == GROUP_ARM_CAP
    # 60 % lies below the N_hit=1 floor (60.6 %): omitted with its reason, not
    # rescued by the 1-point max_gap tolerance.
    assert [(r["tier"], r["target_ir"], r["reason"]) for r in skipped] == [("n1", 60.0, "below_tier_floor")]
    assert all(p[1]["target_ir"] >= attainable_range(s, "n1")[0] for p in plans["n1"])


def test_tier_budget_drops_lowest_targets_with_reason():
    s = np.random.default_rng(1).uniform(0.0, 1.0, 5_000)
    skipped: list[dict] = []
    dense = [60 + 2.5 * i for i in range(15)]  # 60 .. 95 in 2.5 steps: 15 targets
    plan = plan_tier_targets(s, "n0", dense, table="cuda_graph", max_gap=1.0, skipped=skipped, cap=8)
    assert len(plan) == 8
    reasons = {r["reason"] for r in skipped}
    assert "tier_budget" in reasons
    dropped = sorted(r["target_ir"] for r in skipped if r["reason"] == "tier_budget")
    kept = sorted(p[1]["target_ir"] for p in plan)
    assert max(dropped) < min(kept)  # the poorest-resolution (lowest IR) end is dropped first
    # a duplicate cut (two targets resolving to one theta) is recorded, never emitted twice
    tiny = np.asarray([0.2, 0.9])
    sk: list[dict] = []
    plan2 = plan_tier_targets(tiny, "n0", [70, 72, 100], table="cuda_graph", max_gap=100.0, skipped=sk, cap=8)
    thetas = [p[1]["theta_raw"] for p in plan2]
    assert len(thetas) == len(set(thetas)) and any(r["reason"] == "duplicate_cut" for r in sk)


# ------------------------------------------------------------------
# artifact verifier
# ------------------------------------------------------------------


def _source_entries(n=4):
    out = []
    for i in range(n):
        out.append(CacheEntry(
            id=f"traj:{i}", checkpoint_id=CheckpointID.CP1,
            query_keys={"robot_state": torch.randn(32)},
            payload=CachePayload(action_chunk=torch.randn(10, 32),
                                 intermediates={round(1 - k / 10, 4): torch.randn(10, 32) for k in range(1, 10)},
                                 denoising_num_steps=10, task_key="t"),
            step_idx=i, trajectory_id="traj",
            prev_ids=[f"traj:{i-1}"] if i else [], next_ids=[f"traj:{i+1}"] if i < n - 1 else [],
            outcome=1,
        ))
    return out


def _write_pair(tmp_path, spec, *, tamper=None):
    src_entries = _source_entries()
    src = {"key_builder_type": "cp1_spatial_pool_16", "checkpoint_id": "CP1",
           "vector_dims": {"robot_state": 32}, "entries": src_entries}
    src_path = tmp_path / "src.pkl"
    with open(src_path, "wb") as f:
        pickle.dump(src, f)
    entries = [_entry_copy_with_key(e, project(torch.randn(spec.input_dim), spec)) for e in src_entries]
    if tamper:
        tamper(entries)
    art = {"key_builder_type": libs.KEY_BUILDER_TYPE, "checkpoint_id": "CP2",
           "vector_dims": {libs.FIELD: spec.d}, "entries": entries, "projection": spec.meta(),
           "id_policy": libs.ID_POLICY, "source_pkl_sha256": libs.sha256_file(src_path),
           "h5_manifest": {"files": [], "digest": "x"},
           "model": {"checkpoint_dir": "c", "weights_digest": "0" * 64}, "stage1_path": "online",
           "tokenizer": {"source": "s"}, "build_git_commit": "g"}
    cp2_path = tmp_path / "cp2.pkl"
    with open(cp2_path, "wb") as f:
        pickle.dump(art, f)
    return cp2_path, src_path


def test_verify_cp2_artifact_accepts_faithful_copy_and_rejects_tampering(tmp_path):
    spec = get_projection_spec(3, 8, 0.25, 64)
    cp2, src = _write_pair(tmp_path, spec)
    rec = verify(cp2, src, search_samples=4)
    assert rec["ok"] and rec["n_entries"] == 4 and rec["action_chunk_shape"] == (10, 32)

    def cp1_tag(entries):
        entries[1].checkpoint_id = CheckpointID.CP1
    cp2, src = _write_pair(tmp_path / "a", spec, tamper=cp1_tag) if (tmp_path / "a").mkdir() or True else None
    with pytest.raises(VerificationError, match=r"\(a'\)"):
        verify(cp2, src, search_samples=2)

    def drop(entries):
        entries.pop()
    (tmp_path / "b").mkdir()
    cp2, src = _write_pair(tmp_path / "b", spec, tamper=drop)
    with pytest.raises(VerificationError, match=r"\(a\)"):
        verify(cp2, src, search_samples=2)

    def wrong_key_dim(entries):
        entries[0].query_keys[libs.FIELD] = torch.randn(7)
    (tmp_path / "c").mkdir()
    cp2, src = _write_pair(tmp_path / "c", spec, tamper=wrong_key_dim)
    with pytest.raises(VerificationError, match=r"\(d\)"):
        verify(cp2, src, search_samples=2)


def test_verify_rejects_uniformly_wrong_chunk_shape_and_partial_model_digest(tmp_path):
    spec = get_projection_spec(3, 8, 0.25, 64)
    (tmp_path / "shape").mkdir()
    cp2, src = _write_pair(tmp_path / "shape", spec)
    for pkl in (cp2, src):  # source AND copy consistently (5, 32): (b) equality and consensus alone would pass
        art = pickle.load(open(pkl, "rb"))
        for e in art["entries"]:
            e.payload.action_chunk = torch.zeros(5, 32)
        if pkl is cp2:
            art["source_pkl_sha256"] = None
        pickle.dump(art, open(pkl, "wb"))
    art = pickle.load(open(cp2, "rb"))
    art["source_pkl_sha256"] = libs.sha256_file(src)
    pickle.dump(art, open(cp2, "wb"))
    with pytest.raises(VerificationError, match=r"\(e\) action_chunk .* expected \(10, 32\)"):
        verify(cp2, src, search_samples=2)
    (tmp_path / "digest").mkdir()
    cp2, src = _write_pair(tmp_path / "digest", spec)
    art = pickle.load(open(cp2, "rb"))
    art["model"] = {"checkpoint_dir": "c", "sha256_head1mib": "ab" * 32}  # the pre-R1 partial identity
    pickle.dump(art, open(cp2, "wb"))
    with pytest.raises(VerificationError, match=r"\(g\) model.weights_digest"):
        verify(cp2, src, search_samples=2)
    (tmp_path / "s1").mkdir()
    cp2, src = _write_pair(tmp_path / "s1", spec)
    art = pickle.load(open(cp2, "rb"))
    art.pop("stage1_path")  # a library that does not say how its keys were made
    pickle.dump(art, open(cp2, "wb"))
    with pytest.raises(VerificationError, match=r"\(g\) stage1_path"):
        verify(cp2, src, search_samples=2)


def test_weights_digest_covers_every_byte_and_binds(tmp_path):
    ck = tmp_path / "ckpt"
    (ck / "params").mkdir(parents=True)
    blob = bytearray(np.random.default_rng(0).integers(0, 256, 2 * 1024 * 1024, dtype=np.uint8).tobytes())
    (ck / "params" / "w.bin").write_bytes(blob)
    (ck / "config.json").write_text("{}")
    a = libs.weights_digest(ck)
    assert a["files"] == 2 and a["bytes"] == len(blob) + 2 and len(a["weights_digest"]) == 64
    assert libs.weights_digest(ck)["weights_digest"] == a["weights_digest"]  # deterministic
    blob[-1] ^= 0x01  # G2 R1 counterexample: flip the LAST byte of a 2 MiB weight file
    (ck / "params" / "w.bin").write_bytes(blob)
    b = libs.weights_digest(ck)
    assert b["weights_digest"] != a["weights_digest"] and b["bytes"] == a["bytes"]
    assert libs.assert_model_binding({"weights_digest": b["weights_digest"]}, ck)["weights_digest"] == b["weights_digest"]
    with pytest.raises(SystemExit, match="model binding failed"):
        libs.assert_model_binding({"weights_digest": a["weights_digest"]}, ck)
    with pytest.raises(SystemExit, match="no model.weights_digest"):
        libs.assert_model_binding({"checkpoint_dir": "c"}, ck)


def test_parity_run_requires_artifact_model_binding_before_work(tmp_path):
    """Direct callers cannot bypass the CLI's required digest."""
    args = SimpleNamespace(expect_weights_digest="", checkpoint_dir=str(tmp_path))
    with pytest.raises(SystemExit, match="expect-weights-digest is required"):
        parity_check.run(args)


# ------------------------------------------------------------------
# stats
# ------------------------------------------------------------------


def test_wilson_frontier_interp():
    lo, hi = stats.wilson(0, 10)
    assert lo == 0.0 and 0 < hi < 0.35
    lo, hi = stats.wilson(10, 10)
    assert hi == pytest.approx(1.0) and lo > 0.65
    # (55, .85) is non-dominated but sags below the (40,.8)-(60,.9) chord: not on the upper hull.
    front = stats.reference_hull([(40, 0.8), (50, 0.7), (60, 0.9), (55, 0.85), (70, 0.9)])
    assert front == [(40, 0.8), (60, 0.9)]
    assert stats.interp_frontier(front, 47.5) == pytest.approx(0.8375)
    assert stats.interp_frontier(front, 30) is None and stats.interp_frontier(front, 65) is None
    # G2 R1 counterexample: the middle point must go, x=1 interpolates to .70 not .51.
    hull = stats.reference_hull([(0, 0.5), (1, 0.51), (2, 0.9)])
    assert hull == [(0.0, 0.5), (2.0, 0.9)] and stats.interp_frontier(hull, 1) == pytest.approx(0.70)
    # a point above the chord stays; same-cost duplicates keep the best SR
    assert stats.reference_hull([(0, 0.5), (1, 0.8), (2, 0.9), (2, 0.85)]) == [(0.0, 0.5), (1.0, 0.8), (2.0, 0.9)]
    assert stats.reference_hull([(0, 0.5), (1, 0.7), (2, 0.9)]) == [(0.0, 0.5), (2.0, 0.9)]  # collinear -> dropped


def _ledger(rng, n_task=3, per_task=40, p=0.8, cost=30.0, prefix="a"):
    eps = []
    for t in range(n_task):
        for i in range(per_task):
            eps.append({"uid": f"{prefix}:eval:{t}:{i}", "init": f"eval:{t}:{i}", "task": t,
                        "success": bool(rng.random() < p), "cost_ms": cost * 20, "n_dec": 20})
    return eps


def test_bootstrap_frontier_delta_decisions():
    rng = np.random.default_rng(0)
    miss = libs.miss_cost()
    ref = {"r_lo": _ledger(rng, p=0.6, cost=0.4 * miss, prefix="r_lo"),
           "r_hi": _ledger(rng, p=0.9, cost=0.9 * miss, prefix="r_hi")}
    same = _ledger(rng, p=0.75, cost=0.65 * miss, prefix="c")
    out = stats.bootstrap_frontier_delta(same, ref, miss_ms=miss, B=200, seed=1)
    assert out["decision"] in ("indistinguishable", "cp2_higher", "reference_higher")
    assert out["support_miss_frac"] <= 0.01 and out["valid_replicates"] > 0
    better = _ledger(rng, p=0.99, cost=0.65 * miss, prefix="c2")
    assert stats.bootstrap_frontier_delta(better, ref, miss_ms=miss, B=200, seed=1)["decision"] == "cp2_higher"
    worse = _ledger(rng, p=0.3, cost=0.65 * miss, prefix="c3")
    assert stats.bootstrap_frontier_delta(worse, ref, miss_ms=miss, B=200, seed=1)["decision"] == "reference_higher"
    outside = _ledger(rng, p=0.5, cost=0.2 * miss, prefix="c4")
    assert stats.bootstrap_frontier_delta(outside, ref, miss_ms=miss, B=50, seed=1)["decision"] == "outside_reference_support"
    bad_ref = {"r_lo": ref["r_lo"], "r_hi": ref["r_hi"][:-1]}
    with pytest.raises(ValueError, match="same episode identity"):
        stats.bootstrap_frontier_delta(same, bad_ref, miss_ms=miss, B=10)


# ------------------------------------------------------------------
# aggregate (journal + per_step)
# ------------------------------------------------------------------


LIB_SHA = "ab" * 32
STEPS_FULL = libs.STEP_CAP["libero_spatial"]


def _write_run(tmp_path, arm, tier_rows, n_ep=4, checkpoint="CP2", *, library_sha256=LIB_SHA,
               n_rows=None, edit=None):
    """A minimal but complete run: one accepted terminal row per uid, per_step
    verdict rows (``n_rows`` per episode, cycling ``tier_rows``) and a
    ``client_timing`` row at the suite's full step count. ``edit(journal,
    per_step)`` applies a corruption before writing."""
    run = tmp_path / arm
    run.mkdir()
    journal, per_step = [], []
    n_rows = n_rows or libs.MIN_HIT_ROWS["libero_spatial"]
    for i in range(n_ep):
        uid = f"{arm}:eval:{i % 2}:{i}"
        journal.append({"task_uid": uid, "yaml_id": arm, "phase": "eval", "status": "done" if i % 3 else "failed",
                        "success": bool(i % 3), "attempt": 1, "accepted": True})
        for step in range(n_rows):
            ht, st = tier_rows[step % len(tier_rows)]
            per_step.append({"yaml_id": arm, "task_id": i % 2, "task_uid": uid, "step_idx": step, "hit_type": ht,
                             "start_t": st, "attempt": 1, "checkpoint": checkpoint, "score": 0.9,
                             "library_sha256": library_sha256})
        per_step.append({"_kind": "client_timing", "task_uid": uid, "yaml_id": arm, "task_id": i % 2,
                         "attempt": 1, "steps": STEPS_FULL})
    if edit:
        edit(journal, per_step)
    (run / "journal.jsonl").write_text("\n".join(json.dumps(r) for r in journal) + "\n")
    (run / "per_step.jsonl").write_text("\n".join(json.dumps(r) for r in per_step) + "\n")
    return run


def _record(*arms, sha=LIB_SHA):
    return {"library_sha256": sha, "arms": {a: {"target_ir": 60.0, "predicted_ir": 61.0, "theta_raw": 0.8,
                                                "theta_norm": 0.9} for a in arms}}


def test_aggregate_prices_and_gates(tmp_path):
    arm = "acb_sp_lib50_n0_ir60"
    run = _write_run(tmp_path, arm, [("FULL_HIT", None), ("MISS", None)], n_rows=42)
    res = aggregate(run, expect_episodes=4, export_record=_record(arm))
    a = res["arms"][arm]
    expected_ir = 100 * (libs.cp2_tier_cost("FULL_HIT", None) + libs.miss_cost()) / (2 * libs.miss_cost())
    assert a["ir_percent"] == pytest.approx(expected_ir) and a["n_ep"] == 4 and a["tier"] == "n0"
    assert a["counts"] == {"FULL_HIT": 84, "WARM_START": 0, "MISS": 84}
    assert 0 < a["ir_percent_eager"] < 100
    assert res["suite"] == "libero_spatial" and res["audit"]["problems"] == []
    assert a["ir_gap_realized"] == pytest.approx(a["ir_percent"] - 61.0)

    impure = _write_run(tmp_path, "acb_sp_lib50_n0_ir70", [("WARM_START", 0.1)])
    with pytest.raises(SystemExit, match="tier purity"):
        aggregate(impure, expect_episodes=4)

    short = _write_run(tmp_path, "acb_sp_lib50_n1_ir60", [("WARM_START", 0.1)], n_ep=3)
    with pytest.raises(SystemExit, match="not at 4 accepted"):
        aggregate(short, expect_episodes=4)
    assert aggregate(short, expect_episodes=4, allow_partial=True)["arms"]["acb_sp_lib50_n1_ir60"]["n_ep"] == 3

    legacy = _write_run(tmp_path, "acb_sp_lib50_n0_ir80", [("MISS", None)], checkpoint=None)
    with pytest.raises(SystemExit, match="checkpoint="):
        aggregate(legacy, expect_episodes=4)


def _dup_terminal(journal, per_step):
    journal.append(dict(journal[0]))


def _drop_per_step_pair(journal, per_step):
    uid = journal[1]["task_uid"]
    per_step[:] = [r for r in per_step if r["task_uid"] != uid]


def _extra_per_step_pair(journal, per_step):
    per_step.append({**per_step[0], "attempt": 2})


def _truncate_failed(journal, per_step):
    failed = next(r["task_uid"] for r in journal if r["status"] == "failed")
    for r in per_step:
        if r.get("_kind") == "client_timing" and r["task_uid"] == failed:
            r["steps"] = STEPS_FULL - 1


def _short_failed(journal, per_step):
    failed = next(r["task_uid"] for r in journal if r["status"] == "failed")
    kept = 0
    out = []
    for r in per_step:
        if r["task_uid"] == failed and r.get("hit_type") is not None:
            kept += 1
            if kept > 3:
                continue
        out.append(r)
    per_step[:] = out


def _no_digest(journal, per_step):
    for r in per_step:
        r.pop("library_sha256", None)


@pytest.mark.parametrize(
    "edit, fragment",
    [
        (_dup_terminal, "duplicate terminal journal row"),
        (_drop_per_step_pair, "without per_step rows"),
        (_extra_per_step_pair, "without a terminal journal row"),
        (_truncate_failed, "truncated below 200 steps"),
        (_short_failed, "< 42 verdict rows"),
        (_no_digest, "carry no library_sha256"),
    ],
)
def test_aggregate_completeness_gate_fails_closed(tmp_path, edit, fragment):
    arm = "acb_sp_lib50_n0_ir60"
    run = _write_run(tmp_path, arm, [("MISS", None)], edit=edit)
    with pytest.raises(SystemExit, match=fragment):
        aggregate(run, expect_episodes=4, export_record=_record(arm))
    # --allow-partial never relaxes an identity / provenance failure
    with pytest.raises(SystemExit, match=fragment):
        aggregate(run, expect_episodes=4, allow_partial=True, export_record=_record(arm))


def test_aggregate_arm_set_and_library_digest_against_export_record(tmp_path):
    """G2 R1 counterexample: a wrong export digest and a missing arm passed silently."""
    arm = "acb_sp_lib50_n0_ir60"
    run = _write_run(tmp_path, arm, [("MISS", None)])
    with pytest.raises(SystemExit, match="library_sha256 != export record"):
        aggregate(run, expect_episodes=4, export_record=_record(arm, sha="cd" * 32))
    with pytest.raises(SystemExit, match="in export record but not in run"):
        aggregate(run, expect_episodes=4, export_record=_record(arm, "acb_sp_lib50_n0_ir70"))
    with pytest.raises(SystemExit, match="in run but not in export record"):
        aggregate(run, expect_episodes=4, export_record=_record("acb_sp_lib50_n0_ir70"))
    # without a record: rows from two different libraries in one run are rejected
    mixed = _write_run(tmp_path, "acb_sp_lib50_n0_ir65", [("MISS", None)],
                       edit=lambda j, p: p[0].update({"library_sha256": "ef" * 32}))
    with pytest.raises(SystemExit, match="different libraries"):
        aggregate(mixed, expect_episodes=4)


def test_compare_cannot_bypass_cp2_aggregate_gates(tmp_path):
    arm = "acb_sp_lib50_n0_ir60"
    run = _write_run(tmp_path, arm, [("MISS", None)])
    with pytest.raises(SystemExit, match="library_sha256 != export record"):
        compare(
            run,
            tmp_path / "reference-not-read-before-gate",
            export_record=_record(arm, sha="cd" * 32),
            expect_episodes=4,
            allow_partial=False,
            B=10,
            seed=0,
        )


def test_bench_verdict_thresholds():
    assert verdict_for(None) == "insufficient_decisions"
    assert verdict_for(9.9) == "ok_report"
    assert verdict_for(25.0) == "report_with_caption"
    assert verdict_for(41.0) == "halt_profile_segments"


# ------------------------------------------------------------------
# figure-spec export (points only)
# ------------------------------------------------------------------


def test_figure_point_labels_and_series_split():
    from exp.actioncache_baseline.export_figure_points import point_label, series_arms

    assert point_label("acb_sp_lib50_n0_ir65") == "N0 IR65"
    assert point_label("acb_l10_s6_n1_ref850") == "N1 θ=.85"
    assert point_label("rit_sp_ng_ir20") == "rit_sp_ng_ir20"
    arms = {a: {"ir_percent": 60.0 + i, "success_rate": 0.9, "n_ep": 500}
            for i, a in enumerate(("acb_sp_lib50_n0_ir65", "acb_sp_lib50_n1_ir65", "acb_sp_s6_n0_ir65"))}
    s = series_arms(arms, "lib50", "n0")
    assert list(s) == ["acb_sp_lib50_n0_ir65"] and s["acb_sp_lib50_n0_ir65"]["label"] == "N0 IR65"
    assert set(s["acb_sp_lib50_n0_ir65"]) == {"ir_percent", "success_rate", "label", "n_ep"}
    assert list(series_arms(arms, "s6", "n0")) == ["acb_sp_s6_n0_ir65"] and series_arms(arms, "s6", "n1") == {}
