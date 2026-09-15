"""CPU tests for the online RIT experiment package (plan §8 items 7-8; G2 R1 probes)."""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest
import torch
import yaml

from exp.online_rit import ladder3
from exp.online_rit.aggregate_online import aggregate, paired_bootstrap
from exp.online_rit.common import SCHEDULE, CostLedger, load_ledger, write_jsonl
from exp.online_rit.emit_online_arms import MATRIX_COHORT, build_arm, online_judge, rprime_judge, structured_diff, write_arm
from exp.online_rit.fit_init_curves import build_init_state, fit_rprime, knots_from_scores, require_gates, rprime_fit_from_record
from exp.online_rit.ir_replay import (
    Episode,
    GateParams,
    ReplayResult,
    cuts_from_rprime,
    cuts_from_state,
    episodes_from_table,
    replay_ir,
    search_delta,
)
from exp.online_rit.library_prep import compute_update_scales, library_check, sample_s3b, split_indices
from exp.online_rit.pick_terminal_state import frozen_arm_from, pick_terminal
from exp.online_rit.replay_sim import candidate_rows, cold_start, coverage, fit_curves, release_gate, score_bands, walk
from openpi.cache.components.online_rit import OnlineRiskCurves
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

REPO = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "exp/libero_groot/config/rit/libero_10/template.yaml"
KNOTS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
TIERS = ladder3.warm_tiers()
GATE = GateParams(theta=0.5)


# ------------------------------------------------------------------
# synthetic data
# ------------------------------------------------------------------


def synthetic_table(n_traj: int = 40, steps: int = 30, seed: int = 0, *, n_library: int = 0, no_candidate_every: int = 0) -> list[dict]:
    """d and D both decrease with s plus noise; d carries extra information about D."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_traj):
        traj = f"episode_{k:04d}"
        success = k % 5 != 0
        split = "fit" if k % 2 == 0 else "test"
        in_library = k < n_library
        for step in range(steps):
            s = float(np.clip(rng.beta(5, 2), 0, 1))
            base = (1 - s) * 0.6
            row = {
                "trajectory_id": traj, "episode_id": k, "decision_id": step, "task": f"t{k % 10}", "task_id": k % 10,
                "orig_init_state_idx": k, "in_library": in_library, "episode_success": success, "split": split,
            }
            if no_candidate_every and step % no_candidate_every == 0:
                rows.append({**row, "s": None, "candidate_id": None, "no_candidate": "no_hit"})
                continue
            row["s"] = s
            row["candidate_id"] = f"c{k}:{step}"
            for t in TIERS:
                latent = abs(rng.standard_normal()) * 0.2
                row[t.d_column] = base * (1.0 + 0.3 * t.remaining_steps / 4) + latent
                row[t.y_column] = base + 0.8 * latent + 0.05 * abs(rng.standard_normal())
                if in_library:
                    row[f"d_self_{t.index}"] = 0.001 * abs(rng.standard_normal())
            row["y_full"] = base + 0.2
            rows.append(row)
    return rows


def synthetic_ledger() -> CostLedger:
    return CostLedger(stage1_ms=6.146, stage2_ms=7.192, stage3_head_ms=0.0, stage3_step_ms=3.513, fb_batch_ms={2: 4.0, 3: 4.5}, source="synthetic")


def complete_cost_record(ledger=None):
    """Synthetic complete v2 ledger; zero host prices isolate tier accounting tests."""
    ledger = ledger or synthetic_ledger()
    return {"protocol": "online_rit_cost_v2", "bench": {"mode": "eager"},
            "stage1_ms": ledger.stage1_ms, "stage2_ms": ledger.stage2_ms,
            "stage3_head_ms": ledger.stage3_head_ms, "stage3_step_ms": ledger.stage3_step_ms,
            "num_steps": 8, "stage3_ladder_ms": {str(i): ledger.stage3_ms(i) for i in (1, 2, 4, 8)},
            "captured_stage3_ms": {str(i): ledger.stage3_ms(i) for i in (1, 2, 4)},
            "executed_feedback_ms": {str(i): 0.0 for i in (1, 2, 4)},
            "fb_batch_ms": {"1": 1.0, **{str(k): v for k, v in ledger.fb_batch_ms.items()}},
            "dispatch_ms": {"online": 0.0, "threshold": 0.0},
            "commit_ms": {f"{m}:{n}": 0.0 for m in ("frozen", "learning") for n in (0, 1, 3)},
            "snapshot_ms": {"frozen": 0.0, "learning": 0.0}}


def parity_fixture():
    """A complete numeric parity fixture with explicit independent input identities."""
    from exp.online_rit.build_disagreement_table import parity_gate
    from exp.online_rit.provenance import GATE_IDENTITY_KEYS
    identity = {k: k + "-test" for k in GATE_IDENTITY_KEYS}
    identity.update(suite="libero_10", h_exec=5, schedule_id=SCHEDULE.schedule_id)
    rows = [{"task_id": i % 10, "trajectory_id": str(i), "decision_id": 0,
             "floor_D_ref1_ref2": 1.0, **{f"parity_D_{t}": 0.01 for t in ("full", "warm875", "warm750", "warm500")}} for i in range(200)]
    return parity_gate(rows, identity)


# ------------------------------------------------------------------
# ladder3
# ------------------------------------------------------------------


def test_warm_tiers_are_three_warm_rungs_without_full():
    assert [t.name for t in TIERS] == ["warm875", "warm750", "warm500"]
    assert [t.index for t in TIERS] == [7, 6, 4]
    assert [t.remaining_steps for t in TIERS] == [1, 2, 4]
    with pytest.raises(ValueError):
        ladder3.warm_tiers((0.5, 0.75))


def test_fits_nested_and_independent_and_dispatch():
    rows = synthetic_table()
    cand = candidate_rows([r for r in rows if r["split"] == "fit"])
    s = np.array([r["s"] for r in cand])
    ys = {t.name: np.array([r[t.y_column] for r in cand]) for t in TIERS}
    nested = ladder3.fit_nested(s, ys, np.array(KNOTS), alpha=0.05, tiers=TIERS)
    q = nested.q
    assert (np.asarray(q["warm875"]) >= np.asarray(q["warm500"]) - 1e-9).all()
    ind = ladder3.fit_independent(s, {t.name: np.array([r[t.d_column] for r in cand]) for t in TIERS}, np.array(KNOTS), alpha=0.05, tiers=TIERS)
    assert set(ind) == {t.name for t in TIERS}
    cuts = ladder3.rprime_cuts(nested, 0.3)
    order = [cuts[t.name] for t in TIERS]
    assert all(b <= a + 1e-9 for a, b in zip(order, order[1:]))
    assert ladder3.dispatch(0.99, [0.9, 0.5, 0.1]) == 0
    assert ladder3.dispatch(0.3, [0.9, 0.5, 0.1]) == 2
    assert ladder3.dispatch(0.05, [0.9, 0.5, 0.1]) is None
    assert ladder3.dispatch(0.99, [math.inf, math.inf, math.inf]) is None


# ------------------------------------------------------------------
# library_prep
# ------------------------------------------------------------------


def _entries(n: int = 12, dim: int = 32, seed: int = 0) -> list[CacheEntry]:
    gen = torch.Generator().manual_seed(seed)
    out = []
    for k in range(n):
        inter = {SCHEDULE.snapshot_t(i): torch.randn(16, dim, generator=gen) for i in range(1, 8)}
        chunk = torch.randn(16, dim, generator=gen)
        chunk[:, 7:] = 0.0
        for x in inter.values():
            x[:, 7:] = 0.0
        payload = CachePayload(action_chunk=chunk, intermediates=inter, denoising_num_steps=8, task_key=f"t{k % 3}", schedule_id=SCHEDULE.schedule_id)
        out.append(CacheEntry(id=f"e{k}", checkpoint_id=CheckpointID.CP1, query_keys={"robot_state": torch.zeros(8)}, payload=payload, step_idx=k, trajectory_id=f"traj{k // 2}"))
    return out


def test_scales_use_executed_dims_only_and_check_snapshots():
    ents = _entries()
    chk = library_check(ents, SCHEDULE)
    assert chk["ok"] and chk["n_entries"] == 12
    arrays = compute_update_scales(ents, SCHEDULE)
    assert arrays["exec_mask"].sum() == 7
    for i in (7, 6, 4):
        assert arrays[f"mask_d_{i}"].sum() == 7 and (arrays[f"scale_d_{i}"][7:] == 0).all() and (arrays[f"scale_d_{i}"][:7] > 0).all()
    assert arrays["mask_D"].sum() == 7
    del ents[0].payload.intermediates[0.875]
    assert not library_check(ents, SCHEDULE)["ok"]


def test_knots_s3b_and_pool_split():
    knots = knots_from_scores(np.random.default_rng(0).uniform(0.9, 1.0, 500))
    assert knots[0] == 0.0 and knots[-1] == 1.0 and len(knots) == 9
    with pytest.raises(SystemExit):
        knots_from_scores(np.full(100, 0.0))
    ents = _entries(n=30)
    s3 = {"traj0", "traj1", "traj2"}
    picked = sample_s3b(ents, s3, per_task=1, seed=1)
    assert set(picked) == {"t0", "t1", "t2"}
    assert not {t for ts in picked.values() for t in ts} & s3
    assert sample_s3b(ents, s3, per_task=1, seed=1) == picked
    a, b = split_indices(50, 25, 20260914, 3)
    assert len(a) == len(b) == 25 and not set(a) & set(b) and set(a) | set(b) == set(range(50))


# ------------------------------------------------------------------
# population (B6)
# ------------------------------------------------------------------


def test_calibration_population_excludes_library_rows_everywhere():
    rows = synthetic_table(n_traj=20, n_library=4)
    cand = candidate_rows(rows)
    assert all(not r["in_library"] for r in cand) and len(cand) == 16 * 30
    assert len(candidate_rows(rows, include_library=True)) == 20 * 30
    eps = episodes_from_table(rows)
    assert len(eps) == 16 and all(not e.trajectory_id.endswith(("0000", "0001", "0002", "0003")) for e in eps)
    c = fit_curves(rows, KNOTS)
    assert c.n_updates == 16 * 30
    state = build_init_state(rows, KNOTS)
    assert state["source"]["population"] == "calibration" and state["n_updates"] == 16 * 30
    bands = score_bands([r for r in rows if r["split"] == "fit"])
    assert len(bands) == 3 and bands == sorted(bands)


# ------------------------------------------------------------------
# replay_sim / fit_init_curves
# ------------------------------------------------------------------


def test_replay_coverage_gate_walks_and_cold_start_with_the_real_gate():
    rows = synthetic_table(no_candidate_every=7)
    fit = [r for r in rows if r["split"] == "fit"]
    test = [r for r in rows if r["split"] == "test"]
    curves = fit_curves(fit, KNOTS)
    cov = coverage(curves, test, bands=score_bands(fit))
    for name, t in cov["tiers"].items():
        assert t["n_finite"] > 0 and 0.0 <= t["E"] <= 0.15
        for cell in t["cells"].values():
            assert cell["E"] is None or cell["n"] >= 30
    assert release_gate(cov)["status"] == "PASS"
    w0 = walk(curves, test, delta=0.25, mode="fm0", gate=GATE)
    w1 = walk(curves, test, delta=0.25, mode="fm1", gate=GATE)
    assert w0["n_decisions"] == len(test) == w1["n_decisions"]  # no-candidate rows are kept as decisions
    assert w0["counts"]["no_candidate"] > 0 and w0["counts"]["no_candidate"] == w1["counts"]["no_candidate"]
    assert w1["n_updates"] >= w0["n_updates"]
    assert w0["n_updates"] <= w0["counts"]["warm875"] + w0["counts"]["warm750"] + w0["counts"]["warm500"]
    cs = cold_start(rows, KNOTS, delta=0.25, gate=GATE)
    assert cs["n_decisions"] == len([r for r in rows if not r["in_library"]])
    assert all(v is None or v > 0 for v in cs["first_finite_cut_after_updates"].values())


def test_walk_survives_an_all_no_candidate_episode_and_gate_skips():
    rows = synthetic_table(n_traj=4, steps=12)
    empty_ep = [{**r, "s": None, "candidate_id": None} for r in rows if r["trajectory_id"] == "episode_0001"]
    for r in empty_ep:
        for t in TIERS:
            r.pop(t.d_column, None)
    rows = [r for r in rows if r["trajectory_id"] != "episode_0001"] + empty_ep
    curves = fit_curves(rows, KNOTS)
    out = walk(curves, rows, delta=0.25, mode="fm1", gate=GateParams(theta=0.99))  # high theta -> skips
    # the gate decides before retrieval: a no-candidate row inside a skip run is a skip, not a search
    assert out["counts"]["no_candidate"] > 0 and out["counts"]["skip"] > 0
    assert out["counts"]["no_candidate"] + out["counts"]["skip"] + out["counts"]["miss"] + sum(out["counts"][t.name] for t in TIERS) == 4 * 12
    assert out["n_decisions"] == 4 * 12


def test_init_state_and_rprime_round_trip_and_gate_binding():
    rows = synthetic_table()
    state = build_init_state(rows, KNOTS, fixed_params={"scales_sha256": "x" * 64, "schedule_id": "groot_n15_k8_v1", "h_exec": 5})
    c = OnlineRiskCurves.from_snapshot(state, update_enabled=False)
    assert c.n_updates == 0 and state["n_updates"] == len(candidate_rows(rows))
    assert c.fixed_params["scales_sha256"] == "x" * 64
    rec = fit_rprime(rows, KNOTS)
    fit = rprime_fit_from_record(rec)
    assert [float(x) for x in fit.knots] == KNOTS
    ok = {"release": {"status": "PASS"}, "table_sha256": "T"}
    par = parity_fixture()
    record = {"out_sha256": "T", "parity_gate_sha256": "P", "identity": par["identity"]}
    rep = {"release_gate": {"status": "PASS"}, "table_sha256": "T", "knots_sha256": "K"}
    require_gates("T", signal=ok, parity=par, replay=rep, knots_sha="K", record=record, parity_sha="P")
    with pytest.raises(SystemExit, match="Q1 signal gate"):
        require_gates("T", signal={"release": {"status": "FAIL"}, "table_sha256": "T"}, parity=par, replay=rep, knots_sha="K")
    with pytest.raises(SystemExit, match="different table"):
        require_gates("T", signal={"release": {"status": "PASS"}, "table_sha256": "OTHER"}, parity=par, replay=rep, knots_sha="K")
    with pytest.raises(SystemExit, match="parity gate"):
        require_gates("T", signal=ok, parity={"status": "FAIL", "reasons": ["warm875"], "identity": {"table_sha256": "T"}}, replay=rep, knots_sha="K")
    with pytest.raises(SystemExit, match="different knots"):
        require_gates("T", signal=ok, parity=par, replay={**rep, "knots_sha256": "K2"}, knots_sha="K")


def test_parity_gate_fails_when_only_warm875_is_broken_or_identity_differs(tmp_path):
    from exp.online_rit.build_disagreement_table import parity_gate, require_parity_pass

    rng = np.random.default_rng(0)
    rows = []
    for k in range(200):
        rows.append({"task_id": k % 10, "trajectory_id": f"e{k}", "decision_id": k, "floor_D_ref1_ref2": 1.0 + 0.1 * rng.standard_normal(),
                     "parity_D_full": 0.01, "parity_D_warm875": 0.01, "parity_D_warm750": 0.01, "parity_D_warm500": 0.01})
    ident = {"library_sha256": "L", "scales_sha256": "S", "checkpoint_identity_sha256": "C", "h_exec": 5, "schedule_id": "groot_n15_k8_v1", "template_sha256": "T", "suite": "libero_10", "corpus_manifest_sha256": "M", "label_code_sha256": "CODE"}
    assert parity_gate(rows, ident)["status"] == "PASS"
    broken = [dict(r, parity_D_warm875=0.5) for r in rows]
    g = parity_gate(broken, ident)
    assert g["status"] == "FAIL" and any("warm875" in r for r in g["reasons"]) and not any("warm750" in r for r in g["reasons"])
    assert parity_gate(rows[:150], ident)["status"] == "FAIL"
    assert parity_gate([dict(r, floor_D_ref1_ref2=float("nan")) for r in rows], ident)["status"] == "FAIL"
    p = tmp_path / "gate.json"
    p.write_text(json.dumps(parity_gate(rows, ident)))
    require_parity_pass(str(p), ident)
    with pytest.raises(SystemExit, match="different input"):
        require_parity_pass(str(p), {**ident, "scales_sha256": "S2"})


# ------------------------------------------------------------------
# signal_check (B1)
# ------------------------------------------------------------------


def test_auroc_hand_values_and_fit_half_residual_model():
    from exp.online_rit.analysis.signal_check import RankResidualModel, auroc, spearman

    pos = np.array([True] * 5 + [False] * 5)
    assert auroc(np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], float), pos) == 1.0
    assert auroc(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], float), pos) == 0.0
    assert auroc(np.full(10, 3.0), pos) == 0.5
    assert auroc(np.array([1.0, 2.0, 2.0, 3.0]), np.array([False, True, False, True])) == pytest.approx(0.875)
    assert auroc(np.array([1.0, 2.0]), np.array([True, True])) is None
    assert auroc(np.array([1.0, float("nan")]), np.array([True, False])) is None
    rng = np.random.default_rng(1)
    s = rng.uniform(0, 1, 500)
    d = 1 - s + 0.1 * rng.standard_normal(500)
    y = 1 - s + 0.1 * rng.standard_normal(500)
    model = RankResidualModel(d, y, s)
    r_indep = model.residual_spearman(d, y, s)
    assert abs(r_indep) < 0.2  # no shared information beyond s
    latent = rng.standard_normal(500)
    r_dep = model.residual_spearman(d + latent, y + latent, s)
    assert r_dep > 0.6
    assert spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) is None


def test_signal_check_on_synthetic_table():
    from exp.online_rit.analysis.signal_check import episode_success_auroc, floor_stats, release, tier_stats

    rows = synthetic_table(n_traj=64, steps=40, n_library=4)
    fit = [r for r in rows if r["split"] == "fit" and not r["in_library"]]
    test = [r for r in rows if r["split"] == "test" and not r["in_library"]]
    result = {"tiers": {}, "floor": {}}
    for t in TIERS:
        result["tiers"][t.name] = tier_stats(fit, test, t)
        result["floor"][t.name] = floor_stats(rows, t)
        assert episode_success_auroc(test, t)["n"] > 0
    st = result["tiers"]["warm875"]
    assert st["partial_spearman_given_s"] > 0.2 and st["auroc_weighted"] > 0.65
    assert release(result, TIERS)["status"] == "PASS"


# ------------------------------------------------------------------
# ir_replay (B9 / B10)
# ------------------------------------------------------------------


def test_replay_prices_gate_and_feedback_like_the_real_machine():
    ledger = synthetic_ledger()
    ep = [Episode("a", tuple([0.9] * 14))]
    res = replay_ir(ep, [0.0, math.inf, math.inf], ledger, GATE)
    assert res.counts["WARM@0.875"] == 12 and res.counts["SKIP"] == 2
    per_warm = ledger.warm_ms(0.875) + ledger.fb_ms(2)
    expected = (12 * per_warm + 2 * ledger.miss_ms) / (14 * ledger.miss_ms) * 100
    assert res.ir_percent == pytest.approx(expected)
    res0 = replay_ir([Episode("b", (None, 0.9, 0.2))], [0.5, math.inf, math.inf], ledger, GATE, feedback_mode="fm0")
    assert res0.counts["MISS_no_candidate"] == 1 and res0.fb_batches == {"0": 3, "2": 0, "3": 0}
    res1 = replay_ir([Episode("b", (None, 0.9, 0.2))], [0.5, math.inf, math.inf], ledger, GATE, feedback_mode="fm1")
    assert res1.fb_batches["3"] == 1 and res1.fb_batches["2"] == 1
    none = replay_ir([Episode("b", (None, 0.9, 0.2))], [0.5, math.inf, math.inf], ledger, GATE, feedback_mode="none")
    assert none.fb_batches == {"0": 3, "2": 0, "3": 0} and none.ir_percent == none.ir_percent_no_fb
    with pytest.raises(ValueError):
        replay_ir(ep, [0.0, math.inf, math.inf], ledger, GATE, feedback_mode="fm2")


def test_rprime_cli_path_prices_without_feedback(tmp_path, monkeypatch):
    """The --rprime branch must never charge side steps (B9)."""
    from exp.online_rit import ir_replay as mod

    rows = synthetic_table(n_traj=12)
    rec = fit_rprime(rows, KNOTS)
    (tmp_path / "rprime_fit.json").write_text(json.dumps(rec))
    write_jsonl(tmp_path / "table.jsonl", rows)
    led = synthetic_ledger()
    (tmp_path / "cost.json").write_text(json.dumps(complete_cost_record(led)))
    argv = ["ir_replay", "--table", str(tmp_path / "table.jsonl"), "--state", str(tmp_path / "rprime_fit.json"), "--rprime", "--ledger", str(tmp_path / "cost.json"), "--gate-theta", "0.5", "--targets", "70", "--out", str(tmp_path / "out.json")]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["feedback_mode"] == "none" and out["rule"] == "rprime"
    for ev in out["targets"].values():
        assert ev["fb_batches"]["2"] == 0 and ev["fb_batches"]["3"] == 0
        assert ev["ir_percent"] == pytest.approx(ev["ir_percent_no_fb"])


def _fake_result(ir: float) -> ReplayResult:
    return ReplayResult(ir_percent=ir, ir_percent_no_fb=ir, n_decisions=1, counts={}, fb_batches={}, trajectory_sha256=f"tr{ir:.6f}")


def test_search_handles_a_non_monotone_staircase_and_dedupes_only_found_targets():
    """IR(delta): a 60->80 jump in one interval, a 70 plateau elsewhere (B10 probe)."""

    def ir_of(delta: float) -> float:
        if delta < 0.30:
            return 40.0 + 20.0 * delta / 0.30      # 40 -> 60 on [0, 0.3)
        if delta < 0.31:
            return 60.0                            # jump interval: 60 then 80
        if delta < 0.60:
            return 80.0 + 10.0 * (delta - 0.31)    # 80 -> ~83
        if delta < 0.62:
            return 70.0                            # narrow 70 plateau, off the coarse grid
        return 90.0

    out = search_delta([], lambda d: [], synthetic_ledger(), GATE, delta_lo=0.0, delta_hi=1.0, targets=(50.0, 70.0, 75.0, 90.0), evaluate=lambda d: _fake_result(ir_of(d)))
    t = out["targets"]
    assert t["50"]["found"] and abs(t["50"]["ir_percent"] - 50) <= 1
    assert t["70"]["found"], "the narrow plateau must be reached by refinement"
    assert t["90"]["found"]
    assert not t["75"]["found"] and t["75"]["duplicate_of"] is None
    assert out["n_evaluated"] <= 4097
    # duplicates are decided only among found targets
    out2 = search_delta([], lambda d: [], synthetic_ledger(), GATE, delta_lo=0.0, delta_hi=1.0, targets=(85.0, 90.0), evaluate=lambda d: _fake_result(90.0 if d > 0.5 else 40.0))
    assert not out2["targets"]["85"]["found"] and out2["targets"]["90"]["found"] and out2["targets"]["90"]["duplicate_of"] is None


def test_search_on_real_replay_finds_reachable_targets():
    rows = synthetic_table(n_traj=30)
    episodes = episodes_from_table(rows)
    curves = fit_curves(rows, KNOTS)
    provider = cuts_from_state(curves, [t.index for t in TIERS])
    out = search_delta(episodes, provider, synthetic_ledger(), GATE, delta_lo=0.0, delta_hi=2.0, targets=(50.0, 70.0, 90.0, 99.5))
    assert any(v["found"] for v in out["targets"].values())
    for v in out["targets"].values():
        if v["found"]:
            assert abs(v["ir_percent"] - v["target"]) <= 1.0
    rec = fit_rprime(rows, KNOTS)
    rp = cuts_from_rprime(rprime_fit_from_record(rec), [t.name for t in TIERS])
    out2 = search_delta(episodes, rp, synthetic_ledger(), GATE, delta_lo=0.0, delta_hi=2.0, targets=(70.0,), feedback_mode="none")
    assert "70" in out2["targets"]


# ------------------------------------------------------------------
# emit / pick / aggregate
# ------------------------------------------------------------------


def test_rprime_judge_drops_dead_rungs_and_disables_full():
    judge, dropped = rprime_judge({"warm875": 0.99, "warm750": 0.99, "warm500": 0.5}, TIERS)
    assert judge["threshold"] == 2.0 and dropped == ["warm750"]
    assert judge["warm_tiers"] == [{"threshold": 0.99, "start_t": 0.875}, {"threshold": 0.5, "start_t": 0.5}]
    judge, dropped = rprime_judge({"warm875": math.inf, "warm750": math.inf, "warm500": math.inf}, TIERS)
    assert "warm_tiers" not in judge and len(dropped) == 3


def test_emitted_arms_touch_only_the_allowed_sections(tmp_path):
    from tests.cache.test_config_online_rit import fixed_for, write_scales

    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    scales = write_scales(tmp_path / "scales.npz")
    state = build_init_state(synthetic_table(), KNOTS, fixed_params=fixed_for(scales))
    init_path = tmp_path / "init_state.json"
    init_path.write_text(json.dumps(state))
    gate = {"type": "score_hysteresis", "theta_low": 0.99, "theta_high": 0.99, "j": 3, "probe_interval": 3, "L": 6, "include_ws": True}
    judge = online_judge(delta=0.3, knots=KNOTS, scales_path=scales, init_state_path=str(init_path), update_enabled=False, feedback_mode="fm1", state_log_dir=str(tmp_path / "state"))
    doc = build_arm(template, judge=judge, gate=gate, preload_path="/srv/lib.pkl")
    assert structured_diff(template, doc) == []
    sha = write_arm(doc, tmp_path / "l10_test_f_ir70.yaml", template)
    assert len(sha) == 64
    doc["keys"]["vision_0"]["weight"] = 0.1
    assert structured_diff(template, doc) == ["keys.vision_0.weight"]
    with pytest.raises(SystemExit, match="outside the allowed"):
        write_arm(doc, tmp_path / "bad.yaml", template)


def test_matrix_cohorts_map_to_distinct_pools_and_episode_counts():
    from exp.gate_threshold_pareto.run_gtp import SweepStrategy
    from openpi.conductor.task import ServerEndpoint

    assert MATRIX_COHORT["online_adapt"]["pool"] == "adapt" and MATRIX_COHORT["online_adapt"]["trials"] == 25
    assert MATRIX_COHORT["online_a500"]["single_process"] and not MATRIX_COHORT["frozen"]["single_process"]
    assert {m["judge_type"] for m in MATRIX_COHORT.values()} == {"threshold", "online_rit"}
    idx = {t: list(range(t, t + 25)) for t in range(10)}
    strat = SweepStrategy("libero_10", {"arm": "/x.yaml"}, 25, init_index_map=idx)
    eps = strat._episodes("arm", ServerEndpoint("h", 1))  # noqa: SLF001
    assert len(eps) == 250 and eps[0].orig_init_state_idx == 0 and eps[25].orig_init_state_idx == 1
    plain = SweepStrategy("libero_10", {"arm": "/x.yaml"}, 50)
    assert len(plain._episodes("arm", ServerEndpoint("h", 1))) == 500  # noqa: SLF001
    with pytest.raises(SystemExit):
        SweepStrategy("libero_10", {"arm": "/x.yaml"}, 25, init_index_map={0: list(range(10))})


def test_pick_terminal_requires_counts_to_agree_and_a_valid_stream(tmp_path):
    d = tmp_path / "arm__abc" / "proc"
    d.mkdir(parents=True)
    from openpi.cache.online_state import CurveRegistry
    from openpi.cache.components.online_rit import ContinuationFeedback
    reg = CurveRegistry(state_log_root=str(d))
    key = reg.attach(yaml_id="arm", library_sha256="abc", fingerprint="fp", factory=lambda: OnlineRiskCurves(knots=KNOTS, tier_indices=[7, 6, 4], alpha=.05, window=128, n_min=20))
    d = reg.log_dir(key)
    for n in range(7):
        snap = reg.decision_snapshot(key, ("arm", "e", 0, n), .8, .3)
        reg.record_batch(key, snap, [ContinuationFeedback(7, .1, "executed")])
    reg.flush(key, "task_end")
    best, st = pick_terminal(d)
    assert best.name.startswith("state_00000007") and st["n_updates"] == 7
    original = (d / "feedback.jsonl").read_text()
    (d / "feedback.jsonl").write_text("\n".join(original.splitlines()[:-1]) + "\n")
    with pytest.raises(SystemExit, match="did not end cleanly"):
        pick_terminal(d)
    (d / "feedback.jsonl").write_text(original)
    reg.mark_invalid(key, ["bad feedback"])
    with pytest.raises(SystemExit, match="flow_invalid"):
        pick_terminal(d)
    src = tmp_path / "ocold.yaml"
    src.write_text(yaml.safe_dump({"checkpoints": {"cp1": {"judge": {"type": "online_rit", "update_enabled": True}}}}))
    frozen = frozen_arm_from(src, best, new_stem="frozen")
    j = frozen["checkpoints"]["cp1"]["judge"]
    assert j["update_enabled"] is False and j["init_state_path"] == str(best)


def _per_step(uid, attempt, step, hit, start_t, diag, **extra):
    return {"yaml_id": "arm", "task_uid": uid, "attempt": attempt, "step_idx": step, "hit_type": hit, "start_t": start_t, "task_id": int(uid.split(":")[2]), "orig_init_state_idx": int(uid.split(":")[3]), "cp1_score": 0.8, "online_rit": diag, **extra}


def _diag(i, d7):
    return {"fb_batch_size": 2, "delta": 0.3, "learned": True, "update_revision_after": i + 1, "q_pre": {"7": 0.2, "6": 0.3, "4": 0.4},
            "support_kind": {"7": "supported", "6": "supported", "4": "unavailable"}, "shadowed": [], "rejected": 0, "flow_invalid": False,
            "fb": [{"tier": 7, "d": d7, "source": "executed"}, {"tier": 6, "d": 0.1, "source": "shadow"}]}


def test_aggregate_prices_feedback_dedupes_and_flags_incomplete(tmp_path):
    ledger = synthetic_ledger()
    journal = [
        {"status": "done", "accepted": True, "task_uid": "arm:eval:0:0", "attempt": 0, "yaml_id": "arm", "run_id": "r1", "error": None},
        {"status": "failed", "accepted": True, "task_uid": "arm:eval:0:1", "attempt": 1, "yaml_id": "arm", "run_id": "r1", "error": None},
        {"status": "failed", "accepted": False, "task_uid": "arm:eval:0:1", "attempt": 0, "yaml_id": "arm", "run_id": "r1", "error": None},
        {"status": "failed", "accepted": True, "task_uid": "arm:eval:1:0", "attempt": 0, "yaml_id": "arm", "run_id": "r1", "error": "worker crashed"},
    ]
    write_jsonl(tmp_path / "journal.jsonl", journal)
    rows = [_per_step("arm:eval:0:0", 0, i * 5, "WARM_START", 0.875, _diag(i, 0.5 if i == 0 else 0.1), accepted=True, run_id="r1") for i in range(6)]
    rows.append(_per_step("arm:eval:0:0", 0, 30, "MISS", None, {"fb_batch_size": 3, "delta": 0.3, "learned": True, "update_revision_after": 7, "q_pre": {}, "support_kind": {}, "shadowed": [4], "rejected": 1, "fb": []}, accepted=True, run_id="r1"))
    rows.append(dict(rows[0]))  # exact duplicate report -> counted once
    rows.append(_per_step("arm:eval:0:0", 0, 5, "MISS", None, {"fb_batch_size": 3}, accepted=False, run_id="r1"))  # fenced duplicate step
    rows.append(_per_step("arm:eval:0:1", 0, 0, "MISS", None, {"fb_batch_size": 3}, accepted=True, run_id="r1"))  # stale attempt
    rows.append(_per_step("arm:eval:0:1", 1, 0, "MISS", None, {"fb_batch_size": 0}, accepted=True, run_id="r1"))
    rows.append(_per_step("arm:eval:1:0", 0, 0, "MISS", None, {"fb_batch_size": 0}, accepted=True, run_id="r1"))  # infra-failed episode
    write_jsonl(tmp_path / "per_step.jsonl", rows)
    out = aggregate(tmp_path, ledger, score_bands=[0.5, 0.7, 0.9])["arm"]
    assert out["n_ep"] == 2 and out["success_rate"] == 0.5 and out["n_infra_failed"] == 1 and out["incomplete"]
    assert out["decisions"] == 8
    assert out["counts"] == {"FULL_HIT": 0, "WARM_START": 6, "MISS": 2, "WARM_START@0.875": 6}
    spend = 6 * (ledger.warm_ms(0.875) + ledger.fb_ms(2)) + ledger.miss_ms + ledger.fb_ms(3) + ledger.miss_ms
    assert out["ir_percent"] == pytest.approx(100 * spend / (8 * ledger.miss_ms))
    assert out["violation"]["7"]["n"] == 6 and out["violation"]["7"]["rate"] == pytest.approx(1 / 6)
    assert out["shadowed"] == {"4": 1} and out["rejected"] == 1
    cell = out["tail_rate"]["7|executed|q2|supported"]
    assert cell["n"] == 6 and cell["E"] is None  # below the 30 / 10-episode floor -> not estimable
    assert out["warm_run_len"]["max"] == 6 and out["fb_batches"] == {"2": 6, "3": 1, "0": 1}
    bad = rows + [dict(rows[1], hit_type="MISS", start_t=None)]
    write_jsonl(tmp_path / "per_step.jsonl", bad)
    with pytest.raises(SystemExit, match="conflicting"):
        aggregate(tmp_path, ledger)
    write_jsonl(tmp_path / "per_step.jsonl", rows[:1] + [dict(rows[1], run_id="r9")])
    with pytest.raises(SystemExit, match="run_id"):
        aggregate(tmp_path, ledger)


def test_aggregate_flags_flow_invalid_and_pool_violations(tmp_path):
    write_jsonl(tmp_path / "journal.jsonl", [{"status": "done", "accepted": True, "task_uid": "arm:eval:0:0", "attempt": 0, "yaml_id": "arm", "error": None}])
    rows = [_per_step("arm:eval:0:0", 0, 0, "MISS", None, {"fb_batch_size": 3, "learned": True, "flow_invalid": True, "invalid_reasons": ["0:executed:warm875:not finite"], "fb": []}, accepted=True)]
    write_jsonl(tmp_path / "per_step.jsonl", rows)
    out = aggregate(tmp_path, None)["arm"]
    assert out["incomplete"] and any("flow_invalid" in r for r in out["incomplete_reasons"])
    out2 = aggregate(tmp_path, None, pool_manifest={"adapt": {"0": [5, 6, 7]}}, pool_key="adapt")["arm"]
    assert any("init pool" in r for r in out2["incomplete_reasons"])
    out3 = aggregate(tmp_path, None, pool_manifest={"adapt": {"0": [0, 6, 7]}}, pool_key="adapt")["arm"]
    assert not any("init pool" in r for r in out3["incomplete_reasons"])


def test_paired_bootstrap_pairs_on_task_and_init_position():
    a = {f"armA:eval:{t}:{e}": (e % 2 == 0) for t in range(10) for e in range(20)}
    b = {f"armB:eval:{t}:{e}": (e % 4 != 0) for t in range(10) for e in range(20)}
    def identified(values):
        return {uid: {"suite": "s", "parent_pool_sha256": "p", "task_id": int(uid.split(":")[-2]),
                      "orig_init_state_idx": int(uid.split(":")[-1]), "success": ok} for uid, ok in values.items()}
    out = paired_bootstrap(identified(a), identified(b), n_boot=200, seed=1)
    assert out["n_shared"] == 200 and out["n_tasks"] == 10
    assert out["diff"] == pytest.approx(0.75 - 0.5)
    assert out["ci95"][0] <= out["diff"] <= out["ci95"][1]
    assert paired_bootstrap(identified({"x:eval:0:0": True}), identified({"y:eval:1:0": True}))["n_shared"] == 0


def test_load_ledger_requires_feedback_costs(tmp_path):
    p = tmp_path / "cost.json"
    p.write_text(json.dumps({"stage1_ms": 6.0, "stage2_ms": 7.0, "stage3_head_ms": 0.0, "stage3_step_ms": 3.5, "num_steps": 8}))
    with pytest.raises(SystemExit):
        load_ledger(p)
    led = load_ledger(p, require_fb=False)
    assert led.miss_ms == pytest.approx(6 + 7 + 8 * 3.5)
    with pytest.raises(KeyError):
        led.fb_ms(2)


def test_bench_ladder_fit_and_mode_guard():
    from exp.online_rit.bench_fb_cost import SUPPORTED_MODES, fit_step_ladder

    head, step = fit_step_ladder({1: 5.0, 2: 8.5, 4: 15.5, 8: 29.5})
    assert step == pytest.approx(3.5) and head == pytest.approx(1.5)
    assert SUPPORTED_MODES == ("eager",)
