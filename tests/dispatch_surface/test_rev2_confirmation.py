"""Rev 2 confirmation-plan tests (logs/dispatch_surface_rev2_confirmation_plan.log.md section 6,
G2 Round 1 regressions B1-B9).

Reuses the synthetic-but-schema-faithful Rev 1 / Phase 0 world of
``test_rev2_phase0`` and extends it with a dense threshold-grid rollout, the
budget amendment (cost map -> outcome design -> C roster), the power Monte
Carlo (formal pipeline under test-sized frozen constants) and its replay, the
fresh pools with their validation artifact, the P pilot chain, the task plan,
the seal, the outcome-blind confirmation discipline, the re-certifying unseal
and the confirmation analyzer.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import random
import shutil
import types

import numpy as np
import pytest

from exp.dispatch_surface import action_cache_decision as acd
from exp.dispatch_surface import build_confirmation_task_plan as tplan
from exp.dispatch_surface import finalize_tgrid_package as finalize
from exp.dispatch_surface import freeze_record as fr
from exp.dispatch_surface import generate_fresh_inits as gen
from exp.dispatch_surface import pilot as pilot_mod
from exp.dispatch_surface import seal_confirmation as sealmod
from exp.dispatch_surface import tgrid_package as tpkg
from exp.dispatch_surface.analysis import budget_cost_map as bcm
from exp.dispatch_surface.analysis import budget_mixture as bm
from exp.dispatch_surface.analysis import budget_outcome_design as bod
from exp.dispatch_surface.analysis import confirmation_analyzer as cana
from exp.dispatch_surface.analysis import confirmation_discipline as cdisc
from exp.dispatch_surface.analysis import confirmation_power_mc as pmc
from exp.dispatch_surface.analysis import estimator_version as ev
from exp.dispatch_surface.analysis import h1_verdict as hv
from exp.dispatch_surface.analysis import phase0_discipline
from exp.dispatch_surface.analysis.analytic_cost import cost_model_digest
from exp.dispatch_surface.emit_precheck_yamls import LAYER_TGRID, emit_tgrid
from exp.dispatch_surface.phase0_roster import (
    ANCHOR_ARM,
    F_MIN,
    M_MAX,
    THRESHOLD_GRID_FH,
    THRESHOLD_GRID_WS,
    tgrid_arm_id,
    tgrid_cells,
    tgrid_roster_spec_digest,
)
from exp.dispatch_surface.run_precheck import (
    CONFIRMATION_FROZEN_LAUNCH_KEYS,
    LAYER_PILOT,
    PROTOCOL_PILOT,
    TGRID_FROZEN_LAUNCH_KEYS,
    ConfirmationStrategy,
    validate_confirmation_arms,
    validate_fresh_pool,
    validate_pool_files,
    validate_tgrid_arms,
    validate_tgrid_matrix_artifacts,
)
from tests.dispatch_surface import test_rev2_phase0 as p0mod
from tests.dispatch_surface.test_rev2_phase0 import (
    SUITE,
    TEMPLATE,
    TRIALS,
    _officials,
    _rows_for,
    _sha,
    _write_json,
)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """A fresh synthetic Rev 1 / Phase 0 world (same builders as test_rev2_phase0)."""
    tmp = tmp_path_factory.mktemp("confirmation")
    world = p0mod.build_world(tmp)
    manifest_path, manifest = p0mod.build_package(world, tmp)
    p0 = p0mod.build_phase0(world, manifest_path, tmp)
    return types.SimpleNamespace(tmp=tmp, world=world, manifest_path=manifest_path, manifest=manifest, p0=p0)

HERE = pathlib.Path(__file__).parent
REPO = HERE.parents[1]
FIXTURE = json.loads((HERE / "fixtures" / "budget_mixture_dev_stats.json").read_text())
FULL, WARM, MISS = "FULL_HIT", "WARM_START", "MISS"
RUN_TG = "runtgrid00001"
RUN_C = "runconfirm001"
RUN_P = "runpilot00001"
# the formal power constants, captured before any test-sized patch of the module
FROZEN_PMC = {"N_CANDIDATES": pmc.N_CANDIDATES, "R_OUTER": pmc.R_OUTER, "R_INNER": pmc.R_INNER,
              "OUTER_SEED": pmc.OUTER_SEED, "INNER_SEED": pmc.INNER_SEED, "POWER_TARGET": pmc.POWER_TARGET,
              "LCB_ALPHA": pmc.LCB_ALPHA}
FROZEN_INNER_30_0_SHA = "2be4c6792d86b187216f9eca4905c310e2aa4b149410529e2da0ea0cebf71a4e"
# Test-sized power design: the formal 4 x 200 x 10000 is hours of CPU, and the synthetic world is genuinely
# underpowered at 16 outer replicates (16/16 passes needed for LCB >= 0.80; joint-miss granularity 1/64), which the
# mechanical rule correctly reports as underpowered_stop. The pipeline is therefore exercised end to end under a
# patched (candidates, R_OUTER, R_INNER, POWER_TARGET); every validator reads the module constants, so a formal
# record under the formal constants is validated by exactly the same code. The formal values are pinned by
# FROZEN_PMC / the freeze record test.
TEST_N_CANDIDATES = (16, 20, 24, 28)
TEST_R_OUTER = 16
TEST_R_INNER = 64
TEST_POWER_TARGET = 0.05


def _stats(suite):
    f = FIXTURE[suite]
    return f["interval"], f["selected"], {a: bm.ArmStats(**v) for a, v in f["stats"].items()}


# ------------------------------------------------------------------
# 1. budget_mixture (plan 6-1)
# ------------------------------------------------------------------

def test_frozen_l10_fixture_reproduces_plan_values():
    (BL, BH), sel, st = _stats("libero_10")
    h1, miss = bm.auc_with_support(sel["sv"], st, sel["threshold"], st, BL, BH)
    assert not miss and abs(h1 - 0.0816441) < 1e-6
    assert abs(bm.auc_with_support(sel["sv"], st, sel["s0"], st, BL, BH)[0] - 0.0447783) < 1e-6
    assert abs(bm.auc_with_support(sel["s0"], st, sel["threshold"], st, BL, BH)[0] - 0.0368658) < 1e-6
    # the old straight-hull estimator is kept untouched for Phase 0 (0.0722133)
    from exp.dispatch_surface.analysis import frontier_hull as fh
    old, _ = fh.auc_with_support([(st[a].c, st[a].s) for a in sel["sv"]], [(st[a].c, st[a].s) for a in sel["threshold"]], BL, BH)
    assert abs(old - 0.0722133) < 1e-6
    (BL, BH), sel, st = _stats("libero_spatial")
    assert bm.auc_with_support(sel["sv"], st, sel["threshold"], st, BL, BH)[0] < 0  # direction preserved


def test_pareto_pruning_would_be_wrong_r2b1():
    st = {"A": bm.ArmStats(10, 1, 0.5, 1), "B": bm.ArmStats(2000, 100, 0.4, 1), "C": bm.ArmStats(100, 1, 1.0, 1)}
    v_all, basis = bm.value_at(["A", "B", "C"], st, 30.0)
    assert abs(v_all - 0.9607) < 1e-4 and basis == (1, 2)
    assert abs(bm.value_at(["A", "C"], st, 30.0)[0] - 0.6111) < 1e-4
    assert bm.standalone_dominance(["A", "B", "C"], st)["B"] is True   # descriptive flag only


def _random_family(rng, n):
    arms = [f"a{i}" for i in range(n)]
    st = {}
    for a in arms:
        E = 30
        D = rng.randint(40, 100) * E
        T = D * rng.uniform(10, 68)
        S = rng.randint(0, E)
        st[a] = bm.ArmStats(T, D, S, E)
    cs = sorted(st[a].c for a in arms)
    return arms, st, cs[0] + 0.5, cs[-1] + 5.0


def test_lp_equals_hull_at_zero_and_sweep_equals_full_enumeration():
    rng = random.Random(0)
    worst = 0.0
    for _ in range(1000):
        arms, st, BL, BH = _random_family(rng, rng.choice([2, 3, 4, 6, 8]))
        for __ in range(3):
            B = rng.uniform(BL, BH)
            assert abs(bm.value_at(arms, st, B)[0] - bm.hull_at_zero(arms, st, B)) < 1e-9
        a1 = bm.auc_norm(arms, st, BL, BH)
        t, d, s = bm._arrays(arms, st)
        a2 = sum(bm._integrate_piece(p, t, d, s) for p in bm.pieces(arms, st, BL, BH, full_enumeration=True)) / (BH - BL)
        worst = max(worst, abs(a1 - a2), bm.audit_family(arms, st, BL, BH)["abs_diff"])
    assert worst < 1e-8


def test_equal_decisions_reduce_to_straight_hull():
    from exp.dispatch_surface.analysis import frontier_hull as fh
    st = {"a": bm.ArmStats(30 * 40 * 20, 30 * 40, 10, 30), "b": bm.ArmStats(30 * 40 * 40, 30 * 40, 20, 30), "c": bm.ArmStats(30 * 40 * 60, 30 * 40, 24, 30)}
    arms = ["a", "b", "c"]
    hull = fh.upper_concave_hull([(st[a].c, st[a].s) for a in arms])
    for B in (25.0, 33.0, 47.0, 55.0):
        assert abs(bm.value_at(arms, st, B)[0] - fh.sr_at(hull, B)) < 1e-9


def test_canonical_ties_and_order_independence():
    # exact tie: two arms with identical stats -> smaller roster index wins
    st = {"x": bm.ArmStats(30 * 50 * 30, 30 * 50, 15, 30), "y": bm.ArmStats(30 * 50 * 30, 30 * 50, 15, 30),
          "z": bm.ArmStats(30 * 40 * 50, 30 * 40, 27, 30)}
    v1, b1 = bm.value_at(["x", "y", "z"], st, 35.0)
    v2, b2 = bm.value_at(["y", "x", "z"], st, 35.0)
    assert abs(v1 - v2) < 1e-12 and b1 == b2 == (0, 2)
    # same cost, different d / s: both enter the LP, no de-duplication
    st2 = {"p": bm.ArmStats(30 * 60 * 30, 30 * 60, 12, 30), "q": bm.ArmStats(30 * 90 * 30, 30 * 90, 12, 30), "r": bm.ArmStats(30 * 40 * 60, 30 * 40, 28, 30)}
    assert bm.value_at(["p", "q", "r"], st2, 40.0)[1] in ((0, 2), (1, 2))
    # order independence of the active union
    arms = ["a0", "a1", "a2", "a3"]
    st3 = {a: bm.ArmStats(30 * (50 + i * 7) * (40 - i * 3), 30 * (50 + i * 7), 10 + i * 4, 30) for i, a in enumerate(arms)}
    u1 = sorted(bm.active_basis_union(arms, st3, 32.0, 39.0)["active"])
    u2 = sorted(bm.active_basis_union(list(reversed(arms)), st3, 32.0, 39.0)["active"])
    assert u1 == u2
    assert set(bm.active_basis_union(arms, st3, 32.0, 39.0)["active"]) <= set(arms)


def test_numeric_mismatch_and_domain_are_fail_closed(monkeypatch):
    arms, st, BL, BH = _random_family(random.Random(3), 5)
    with pytest.raises(ValueError):
        bm.ArmStats(1.0, 0.0, 1.0, 30)
    with pytest.raises(ValueError):
        bm.ArmStats(1.0, 10.0, 31.0, 30)
    monkeypatch.setattr(bm, "simpson_auc_norm", lambda *a, **k: 123.0)
    with pytest.raises(bm.NumericMismatch):
        bm.audit_family(arms, st, BL, BH)


def test_left_infeasible_right_constant_symmetry_and_step_envelope():
    arms, st, BL, BH = _random_family(random.Random(5), 4)
    cmin = min(st[a].c for a in arms)
    assert bm.auc_with_support(arms, st, arms, st, cmin - 1.0, BH) == (bm.SUPPORT_MISS_AUC, True)
    assert abs(bm.auc_with_support(arms, st, arms, st, BL, BH)[0]) < 1e-12
    # antisymmetry on an interval where BOTH sub-families are feasible
    BL2 = max(min(st[a].c for a in arms[:2]), min(st[a].c for a in arms[2:])) + 0.5
    a = bm.auc_with_support(arms[:2], st, arms[2:], st, BL2, BH)[0]
    b = bm.auc_with_support(arms[2:], st, arms[:2], st, BL2, BH)[0]
    assert abs(a + b) < 1e-12
    cmax = max(st[a].c for a in arms)
    assert abs(bm.value_at(arms, st, cmax + 100)[0] - max(st[a].s for a in arms)) < 1e-12
    assert abs(bm.step_value(arms, st, cmax + 100) - max(st[a].s for a in arms)) < 1e-12
    assert bm.step_value(arms, st, cmin - 1.0) is None
    # the step envelope never exceeds the mixture envelope
    for B in np.linspace(BL, BH, 20):
        assert bm.step_value(arms, st, B) <= bm.value_at(arms, st, B)[0] + 1e-12


def test_extrema_bitset_and_estimator_digest():
    (BL, BH), sel, st = _stats("libero_10")
    ex = bm.difference_extrema(sel["sv"], st, sel["threshold"], st, BL, BH)
    assert ex["max"] > 0 and ex["min"] > 0 and not ex["a_dominated"]
    assert bm.bitset_bytes(["a", "b", "c"], ["a", "c"]) == bytes([0b10100000])
    assert bm.bitset_rollup_sha256([b"\x01", b"\x02"]) == hashlib.sha256(b"\x01\x02").hexdigest()
    assert ev.budget_mixture_digest() == hashlib.sha256(ev.canonical(ev.BUDGET_MIXTURE_V1).encode()).hexdigest()
    assert ev.BUDGET_MIXTURE_V1["pruning"] == "none"


def test_freeze_record_matches_code_constants():
    rec = fr.load_record(REPO / fr.RECORD_PATH)
    c = rec["constants"]
    assert tuple(c["threshold_grid_fh"]) == THRESHOLD_GRID_FH and tuple(c["threshold_grid_ws"]) == THRESHOLD_GRID_WS
    assert c["F_MIN"] == F_MIN and c["M_MAX"] == M_MAX and c["threshold_grid_new_arms"] == len(tgrid_cells())
    assert tuple(c["N_CANDIDATES"]) == FROZEN_PMC["N_CANDIDATES"] and c["R_OUTER"] == FROZEN_PMC["R_OUTER"] and c["R_INNER"] == FROZEN_PMC["R_INNER"]
    assert c["OUTER_SEED"] == FROZEN_PMC["OUTER_SEED"] and c["INNER_SEED"] == FROZEN_PMC["INNER_SEED"]
    assert c["POWER_TARGET"] == FROZEN_PMC["POWER_TARGET"] == 0.80 and c["LCB_ALPHA"] == FROZEN_PMC["LCB_ALPHA"] == 0.05
    assert c["MAX_RETRIES"] == gen.MAX_RETRIES and c["POOL_QUOTA"] == gen.POOL_QUOTA
    assert c["PILOT_TOLERANCE_PT"] == pilot_mod.PILOT_TOLERANCE_PT
    assert c["AUDIT_TOL"] == bm.AUDIT_TOL and c["VALUE_TOL"] == bm.VALUE_TOL and c["BREAKPOINT_TOL_MS"] == bm.BREAKPOINT_TOL_MS
    # G1 binding on the working tree: whole-file digests + the plan's frozen G1 prefix (G2R1-B1)
    got = fr.verify(rec, REPO)
    plan_rel = "logs/dispatch_surface_rev2_confirmation_plan.log.md"
    assert plan_rel in rec["frozen_prefix"] and got[plan_rel] == rec["frozen_prefix"][plan_rel]["sha256"]


def test_freeze_prefix_survives_appends_but_not_edits(tmp_path):
    plan_rel = "logs/dispatch_surface_rev2_confirmation_plan.log.md"
    rec = fr.load_record(REPO / fr.RECORD_PATH)
    spec = rec["frozen_prefix"][plan_rel]
    data = (REPO / plan_rel).read_bytes()
    root = tmp_path / "repo"
    (root / "logs").mkdir(parents=True)
    small = {"documents_sha256": {}, "frozen_prefix": {plan_rel: spec}, "constants": {}}
    # appending a Code / G2 record after the boundary keeps the freeze intact
    (root / plan_rel).write_bytes(data + b"\n### G2 Round 9 \xe2\x80\x94 appended later\n\nmore text\n")
    assert fr.verify(small, root)[plan_rel] == spec["sha256"]
    # any byte inside the frozen G1 body breaks it
    body = bytearray(data)
    body[200] ^= 0x01
    (root / plan_rel).write_bytes(bytes(body))
    with pytest.raises(SystemExit):
        fr.verify(small, root)
    # a duplicated boundary marker is ambiguous and refused
    (root / plan_rel).write_bytes(data + b"\n" + spec["end_marker"].encode() + b" again\n")
    with pytest.raises(SystemExit):
        fr.verify(small, root)
    # the whole-file digest of the appended plan differs, i.e. the old freeze test would have failed
    assert hashlib.sha256(data + b"x").hexdigest() != spec["sha256"]


# ------------------------------------------------------------------
# 2. threshold grid: emit / runner / discipline / package (plan 6-2..4)
# ------------------------------------------------------------------

def test_grid_constants():
    assert len(tgrid_cells(include_rev1=True)) == 32 and len(tgrid_cells()) == 29
    assert all(fh + ws <= 100 for fh, ws in tgrid_cells(include_rev1=True))
    with pytest.raises(SystemExit):
        tgrid_arm_id(80, 30)
    assert tgrid_roster_spec_digest(SUITE) == tgrid_roster_spec_digest(SUITE)


def _tgrid_rollout(matrix, world, run_dir, rng):
    per_step, journal = [], []
    for arm in matrix["arms"]:
        fh, ws = matrix["nominal"][arm]["fh"], matrix["nominal"][arm]["ws"]
        for t, offs in _officials().items():
            for subset, official in enumerate(offs):
                n = int(rng.integers(6, 12))
                u = rng.random(n)
                verdicts = [FULL if x < fh / 100 else (WARM if x < (fh + ws) / 100 else MISS) for x in u]
                # a deliberately weaker dense baseline than SV so the synthetic development H1 proceeds
                success = bool(rng.random() < 0.2 + 0.4 * (1 - fh / 100))
                rows, j = _rows_for(arm, t, subset, official, verdicts, success, run_id=RUN_TG)
                per_step += rows
                journal.append(j)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "per_step.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_step))
    (run_dir / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in journal))


@pytest.fixture(scope="module")
def tgrid(chain):
    tmp = chain.tmp / "tgrid"
    out_dir = tmp / "cfg"
    args = types.SimpleNamespace(suite=SUITE, rev1_package_manifest=str(chain.manifest_path), table=str(chain.world.table),
                                 template=str(TEMPLATE), library_pkl=str(chain.world.lib), out_dir=str(out_dir), layer=LAYER_TGRID)
    emit_tgrid(args)
    matrix_path = out_dir / f"arm_matrix_{LAYER_TGRID}.json"
    matrix = json.loads(matrix_path.read_text())
    run_dir = tmp / "run"
    _tgrid_rollout(matrix, chain.world, run_dir, np.random.default_rng(11))
    entry = {"protocol": matrix["protocol"], "layer": LAYER_TGRID, "suite": SUITE, "run_id": RUN_TG,
             "executed_arms": sorted(matrix["arms"]), "core_arms": [], "descriptive_arms": sorted(matrix["arms"]),
             "trials_per_task": TRIALS, "replan_steps": 5, "env_seed": 7, "policy_fingerprint": chain.world.contract["policy_fingerprint"],
             "library_sha256": _sha(chain.world.lib), "aprime_content_sha256": chain.world.pool["rollup_sha256"],
             "split_manifest_sha256": _sha(chain.world.split), "arm_matrix_sha256": _sha(matrix_path),
             "contract_binding": {"h_exec": 5, "policy_fingerprint": chain.world.contract["policy_fingerprint"], "servers": {}},
             "pool": chain.world.pool, "frozen_yaml_sha256": matrix["arm_yaml_sha256"], "artifact_sha256": {},
             "fit_record_sha256": None, "executed_yaml_sha256": matrix["arm_yaml_sha256"], "posthoc_exploratory": True,
             "tgrid_roster_spec_sha256": matrix["tgrid_roster_spec_sha256"],
             "threshold_pair_rollup_sha256": matrix["threshold_pair_rollup_sha256"], "contract_source": matrix["contract_source"],
             "estimator_version": matrix["estimator_version"], "rev1_package_manifest_sha256": matrix["rev1_package_manifest_sha256"],
             "cost_model_digest": matrix["cost_model_digest"]}
    ledger = _write_json(run_dir / "per_step.jsonl.launch.json", {"schema_version": 2, "launches": [entry]})
    pkg_manifest = finalize.finalize(str(matrix_path), str(run_dir / "journal.jsonl"), str(run_dir / "per_step.jsonl"),
                                     str(ledger), str(chain.world.split), str(tmp / "pkg"))
    return types.SimpleNamespace(tmp=tmp, cfg_dir=out_dir, run_dir=run_dir, matrix_path=matrix_path, matrix=matrix,
                                 journal=run_dir / "journal.jsonl", per_step=run_dir / "per_step.jsonl", ledger=ledger,
                                 entry=entry, pkg_manifest=pkg_manifest)


def test_tgrid_emit_matrix_and_yamls(tgrid):
    m = tgrid.matrix
    assert len(m["arms"]) == 29 and m["layer"] == LAYER_TGRID and m["posthoc_exploratory"] is True
    assert m["estimator_version"] == ev.budget_mixture_digest() and m["cost_model_digest"] == cost_model_digest()
    for arm, (fh, ws) in ((a, (n["fh"], n["ws"])) for a, n in m["nominal"].items()):
        y = pathlib.Path(m["arms"][arm]).read_text()
        assert ("warm_tiers" in y) == (ws > 0)
        assert abs(m["nominal_cost_ms"][arm] - ((fh / 100) * 10.260266 + (ws / 100) * 46.818293 + (1 - (fh + ws) / 100) * 67.518595)) < 1e-9
    assert len({tuple(p) for p in m["threshold_pairs"].values()}) == 29
    assert set(m["rev1_reference_cells"]) == {"dsp_t_fh30_ws20", "dsp_t_fh50_ws20", "dsp_t_fh70_ws10"}
    validate_tgrid_matrix_artifacts(m)
    validate_tgrid_arms(m["arms"], m)


def test_tgrid_emit_refuses_wrong_table_and_nonempty_dir(chain, tmp_path):
    bad_table = tmp_path / "t.jsonl"
    bad_table.write_text(chain.world.table.read_text() + "\n")
    args = types.SimpleNamespace(suite=SUITE, rev1_package_manifest=str(chain.manifest_path), table=str(bad_table),
                                 template=str(TEMPLATE), library_pkl=str(chain.world.lib), out_dir=str(tmp_path / "o"), layer=LAYER_TGRID)
    with pytest.raises(SystemExit):
        emit_tgrid(args)


def test_tgrid_runner_rejects_bad_yamls(tgrid, tmp_path):
    import yaml
    m = tgrid.matrix
    arm_ws0 = next(a for a, n in m["nominal"].items() if n["ws"] == 0)
    arm_ws = next(a for a, n in m["nominal"].items() if n["ws"] > 0)
    doc = yaml.safe_load(pathlib.Path(m["arms"][arm_ws0]).read_text())
    doc["checkpoints"]["cp1"]["judge"]["warm_tiers"] = [{"threshold": 0.5, "start_t": 0.3}]
    p = tmp_path / "ws0_with_tier.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(SystemExit):
        validate_tgrid_arms({arm_ws0: str(p)}, m)
    doc = yaml.safe_load(pathlib.Path(m["arms"][arm_ws]).read_text())
    doc["checkpoints"]["cp1"]["judge"]["warm_tiers"][0]["start_t"] = 0.5
    p = tmp_path / "bad_start_t.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(SystemExit):
        validate_tgrid_arms({arm_ws: str(p)}, m)
    doc = yaml.safe_load(pathlib.Path(m["arms"][arm_ws]).read_text())
    doc["checkpoints"]["cp1"]["judge"]["threshold"] = doc["checkpoints"]["cp1"]["judge"]["threshold"] + 1e-6
    p = tmp_path / "pair_drift.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(SystemExit):
        validate_tgrid_arms({arm_ws: str(p)}, m)
    bad = dict(m)
    bad["threshold_pair_rollup_sha256"] = "0" * 64
    with pytest.raises(SystemExit):
        validate_tgrid_matrix_artifacts(bad)


def test_tgrid_discipline_and_ledger_keys(tgrid, chain, tmp_path):
    ctx = phase0_discipline.validate_tgrid(str(tgrid.matrix_path), str(tgrid.ledger), str(chain.world.split), trials=TRIALS)
    assert ctx["roster_complete"] and set(ctx["arms"]) == set(tgrid.matrix["arms"])
    for key in TGRID_FROZEN_LAUNCH_KEYS:
        assert key in tgrid.entry
    bad = json.loads(tgrid.ledger.read_text())
    bad["launches"][0]["threshold_pair_rollup_sha256"] = "1" * 64
    p = _write_json(tmp_path / "bad_ledger.json", bad)
    with pytest.raises(SystemExit):
        phase0_discipline.validate_tgrid(str(tgrid.matrix_path), str(p), str(chain.world.split), trials=TRIALS)
    bad = json.loads(tgrid.ledger.read_text())
    dropped = bad["launches"][0]["executed_arms"].pop()
    bad["launches"][0]["executed_yaml_sha256"].pop(dropped)
    p = _write_json(tmp_path / "partial_ledger.json", bad)
    ctx2 = phase0_discipline.validate_tgrid(str(tgrid.matrix_path), str(p), str(chain.world.split), trials=TRIALS)
    assert not ctx2["roster_complete"]


def test_tgrid_package_verify_and_negatives(tgrid, chain, tmp_path):
    manifest = tpkg.verify_package(tgrid.pkg_manifest)
    assert len(manifest["members"]) == len(tpkg.required_roles())
    # finalize refuses an incomplete grid and a non-empty target
    with pytest.raises(SystemExit):
        finalize.finalize(str(tgrid.matrix_path), str(tgrid.journal), str(tgrid.per_step), str(tgrid.ledger),
                          str(chain.world.split), str(tgrid.pkg_manifest.parent))
    lines = tgrid.journal.read_text().splitlines()
    short = tmp_path / "short.jsonl"
    short.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(SystemExit):
        finalize.finalize(str(tgrid.matrix_path), str(short), str(tgrid.per_step), str(tgrid.ledger), str(chain.world.split), str(tmp_path / "pkg2"))
    # drift after finalisation is refused
    dup = tmp_path / "pkgdup"
    shutil.copytree(tgrid.pkg_manifest.parent, dup)
    (dup / "journal.jsonl").write_text((dup / "journal.jsonl").read_text() + "\n")
    with pytest.raises(SystemExit):
        tpkg.verify_package(dup / tpkg.MANIFEST_NAME)
    # a package-internal yaml member drift (one byte) is refused
    dup2 = tmp_path / "pkgdup2"
    shutil.copytree(tgrid.pkg_manifest.parent, dup2)
    arm = tpkg.grid_arms()[0]
    y = dup2 / "yaml" / f"{arm}.yaml"
    y.write_text(y.read_text() + "\n# drift\n")
    with pytest.raises(SystemExit):
        tpkg.verify_package(dup2 / tpkg.MANIFEST_NAME)


def test_tgrid_package_is_self_contained_after_execution_dir_is_gone(tgrid, chain):
    """G2R1-B8: the finalised package must be consumable when the emit / run
    directories the matrix still names no longer exist."""
    m = json.loads((tgrid.pkg_manifest.parent / "arm_matrix_exploratory_tgrid.json").read_text())
    assert all(str(tgrid.cfg_dir) in p for p in m["arms"].values())   # historical execution paths
    moved_cfg, moved_run = tgrid.cfg_dir.with_name("cfg_moved"), tgrid.run_dir.with_name("run_moved")
    tgrid.cfg_dir.rename(moved_cfg)
    tgrid.run_dir.rename(moved_run)
    try:
        assert not any(pathlib.Path(p).exists() for p in m["arms"].values())
        tpkg.verify_package(tgrid.pkg_manifest)
        tg = bcm._load_tgrid(str(tgrid.pkg_manifest), TRIALS, rev1_manifest_path=str(chain.manifest_path))
        assert len(tg["cells"]) == 29 and all(len(c) == 10 * TRIALS for c in tg["cells"].values())
        rev1 = bcm._load_rev1(str(chain.manifest_path), TRIALS)
        p0 = bcm._load_phase0(str(chain.p0.matrix_path), str(chain.p0.ledger), str(chain.world.split), str(chain.p0.journal),
                              str(chain.p0.per_step), TRIALS)
        built = bcm.build(rev1, p0, tg)
        assert len(built["family_points"]["threshold"]["arms"]) == 32
        # the old behaviour (matrix paths dereferenced) cannot even validate the matrix now
        with pytest.raises(SystemExit):
            phase0_discipline.validate_tgrid(str(tgrid.pkg_manifest.parent / "arm_matrix_exploratory_tgrid.json"),
                                             str(tgrid.pkg_manifest.parent / "per_step.jsonl.launch.json"),
                                             str(tgrid.pkg_manifest.parent / "split_manifest.json"), trials=TRIALS)
    finally:
        moved_cfg.rename(tgrid.cfg_dir)
        moved_run.rename(tgrid.run_dir)


# ------------------------------------------------------------------
# 3. budget cost map (cost-only) and outcome design (plan 6-5, 6-6)
# ------------------------------------------------------------------

def _cost_map_args(chain, tgrid, out, tgrid_manifest=None):
    return types.SimpleNamespace(rev1_package_manifest=str(chain.manifest_path), phase0_arm_matrix=str(chain.p0.matrix_path),
                                 phase0_launch_manifest=str(chain.p0.ledger), phase0_journal=str(chain.p0.journal),
                                 phase0_per_step=str(chain.p0.per_step), split_manifest=str(chain.world.split),
                                 tgrid_package_manifest=str(tgrid_manifest if tgrid_manifest is not None else tgrid.pkg_manifest),
                                 trials=TRIALS, out=str(out))


def _build_cost_map(chain, tgrid, out, **kw):
    rev1 = bcm._load_rev1(str(chain.manifest_path), TRIALS)
    p0 = bcm._load_phase0(str(chain.p0.matrix_path), str(chain.p0.ledger), str(chain.world.split), str(chain.p0.journal), str(chain.p0.per_step), TRIALS)
    tg = bcm._load_tgrid(str(tgrid.pkg_manifest), TRIALS)
    m = bcm.build(rev1, p0, tg, **kw)
    out.write_text(json.dumps(m, indent=2, sort_keys=True))
    return out, _sha(out), m


@pytest.fixture(scope="module")
def budget(chain, tgrid):
    path, sha, m = _build_cost_map(chain, tgrid, tgrid.tmp / "budget_cost_map_frozen.json")
    args = _cost_map_args(chain, tgrid, tgrid.tmp / "budget_outcome_design.json")
    args.budget_cost_map = str(path)
    args.budget_cost_map_sha256 = sha
    args.out_roster = str(tgrid.tmp / "c_roster.json")
    design = bod.run(args)
    return types.SimpleNamespace(path=path, sha=sha, map=m, design=design, design_path=pathlib.Path(args.out),
                                 roster_path=pathlib.Path(args.out_roster))


def test_cost_map_is_outcome_blind_and_locked(chain, tgrid, tmp_path):
    src = (REPO / "exp/dispatch_surface/analysis/budget_cost_map.py").read_text()
    tree = ast.parse(src)
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    for banned in ("budget_mixture", "budget_outcome_design", "h1_verdict", "analyze_precheck", "frontier_hull"):
        assert not any(banned in (m or "") for m in imported)
    consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "success" not in consts and "status" not in consts
    base, base_sha, _ = _build_cost_map(chain, tgrid, tmp_path / "base.json")
    again, again_sha, _ = _build_cost_map(chain, tgrid, tmp_path / "again.json")
    assert base_sha == again_sha
    m = json.loads(base.read_text())
    assert m["outcome_blind"] and m["replicates"] == 10000 and m["a3_pass"], m["a3_problems"]
    assert len(m["family_points"]["threshold"]["arms"]) == 32   # 3 Rev 1 + 29 grid
    assert m["estimator_version"] == ev.budget_mixture_digest()


def test_outcome_design_roster_and_verdict(budget):
    d = budget.design
    assert d["verdict"] in ("proceed_to_power", "stop_before_C", "roster_overflow")
    for fam, sel in d["c_roster_selection"].items():
        assert len(sel["arms"]) <= M_MAX
        for a in sel["arms"]:
            assert sel["reasons"][a]
    assert "passed" in d["hypotheses"]["H1"] and "passed" not in d["hypotheses"]["H2"]
    assert d["hypotheses"]["H1"]["step_envelope"]["effect_plugin"] is not None
    roster = json.loads(budget.roster_path.read_text())
    assert roster["outcome_design_sha256"] == _sha(budget.design_path)
    assert any(e["arm"] == ANCHOR_ARM for e in roster["arms"])
    assert d["audit_replicates"] and len(d["audit_replicates"]) <= 100
    assert d["hypotheses"]["H1"]["audit"]["full_sample"] is not None
    # the roster validator rebuilds this roster from the design (G2R1-B3)
    design = json.loads(budget.design_path.read_text())
    pmc.validate_c_roster(roster, design, budget.design_path, budget.map, budget.path)
    bad = json.loads(json.dumps(roster))
    bad["arms"][0]["reasons"] = ["hand_picked"]
    with pytest.raises(SystemExit):
        pmc.validate_c_roster(bad, design, budget.design_path, budget.map, budget.path)
    bad = json.loads(json.dumps(roster))
    bad["arms"].pop(0)
    with pytest.raises(SystemExit):
        pmc.validate_c_roster(bad, design, budget.design_path, budget.map, budget.path)


def test_outcome_design_refuses_drift_nonformal_and_trials(chain, tgrid, budget, tmp_path):
    args = _cost_map_args(chain, tgrid, tmp_path / "x.json")
    args.budget_cost_map = str(budget.path)
    args.budget_cost_map_sha256 = "0" * 64
    args.out_roster = ""
    with pytest.raises(SystemExit):
        bod.run(args)
    small, small_sha, _ = _build_cost_map(chain, tgrid, tmp_path / "small.json", reps=50)
    args.budget_cost_map = str(small)
    args.budget_cost_map_sha256 = small_sha
    with pytest.raises(SystemExit):
        bod.run(args)
    # --trials is pinned at the formal 30 for every formal entry point (G2R1-B3)
    args.budget_cost_map = str(budget.path)
    args.budget_cost_map_sha256 = budget.sha
    args.trials = 10
    with pytest.raises(SystemExit, match="frozen at --trials 30"):
        bod.run(args)


def _design_cells(chain, tgrid, budget):
    args = _cost_map_args(chain, tgrid, budget.path.parent / "unused.json")
    src = bod.load_sources(args, budget.map, TRIALS)
    from exp.dispatch_surface.cost_map_api import shared_index_from_map
    picks = shared_index_from_map(budget.map, src["grid"])
    return src, picks


def _power_args(chain, tgrid, budget, out, **extra):
    args = _cost_map_args(chain, tgrid, out)
    args.outcome_design = str(budget.design_path)
    args.c_roster = str(budget.roster_path)
    args.budget_cost_map = str(budget.path)
    args.workers = 1
    args.smoke = False
    for k, v in extra.items():
        setattr(args, k, v)
    return args


def test_roster_overflow_and_stop_loss_fail_closed(budget, monkeypatch, chain, tgrid, tmp_path):
    src, picks = _design_cells(chain, tgrid, budget)
    small = picks[:200]   # design() itself is R-agnostic; the formal R gate lives in run()
    monkeypatch.setattr(bod, "M_MAX", 0)
    out = bod.design(budget.map, src["cells"], small, suite=SUITE, audit_input_digest="0" * 64)
    assert out["verdict"] == "roster_overflow" and out["roster_overflow"]
    monkeypatch.setattr(bod, "M_MAX", M_MAX)
    # stop-loss: SV successes zeroed -> development H1 q05 <= 0
    cells = {a: dict(v) for a, v in src["cells"].items()}
    for a in budget.map["family_points"]["sv"]["arms"]:
        cells[a] = {k: (c, n, 0.0) for k, (c, n, _s) in cells[a].items()}
    out2 = bod.design(budget.map, cells, small, suite=SUITE, audit_input_digest="0" * 64)
    assert out2["verdict"] == "stop_before_C"
    # a design whose verdict is not proceed_to_power is refused by the power MC
    bad = json.loads(budget.design_path.read_text())
    bad["verdict"] = "roster_overflow"
    bad_path = _write_json(tmp_path / "bad_design.json", bad)
    roster = json.loads(budget.roster_path.read_text())
    roster["outcome_design_sha256"] = _sha(bad_path)
    roster["verdict"] = "roster_overflow"
    roster_path = _write_json(tmp_path / "bad_roster.json", roster)
    args = _power_args(chain, tgrid, budget, tmp_path / "p.json", outcome_design=str(bad_path), c_roster=str(roster_path), smoke=True)
    with pytest.raises(SystemExit):
        pmc.run(args)


# ------------------------------------------------------------------
# 4. power MC (plan 6-7, G2R1-B2 / B3)
# ------------------------------------------------------------------

def test_clopper_pearson_and_selection_rule():
    assert pmc.clopper_pearson_lower(0, 200) == 0.0
    assert abs(pmc.clopper_pearson_lower(200, 200) - 0.05 ** (1 / 200)) < 1e-12
    assert 0.80 < pmc.clopper_pearson_lower(171, 200) < 0.86
    c, tgt = FROZEN_PMC["N_CANDIDATES"], FROZEN_PMC["POWER_TARGET"]
    assert pmc.select_n({30: 0.82, 40: 0.78, 50: 0.85, 60: 0.86}, c, tgt)["selected_N"] == 50
    assert pmc.select_n({30: 0.85, 40: 0.86, 50: 0.79, 60: 0.81}, c, tgt)["selected_N"] == 60
    assert pmc.select_n({30: 0.9, 40: 0.9, 50: 0.9, 60: 0.9}, c, tgt)["selected_N"] == 30
    assert pmc.select_n({30: 0.5, 40: 0.6, 50: 0.7, 60: 0.79}, c, tgt)["verdict"] == "underpowered_stop"


def test_inner_stream_is_the_frozen_pcg64_seedsequence():
    """G2R1-B2: Generator(PCG64(SeedSequence(20260829, spawn_key=(N, r)))) verbatim."""
    frozen = np.random.Generator(np.random.PCG64(np.random.SeedSequence(FROZEN_PMC["INNER_SEED"], spawn_key=(30, 0))))
    assert frozen.integers(0, 30, 6).tolist() == [18, 20, 13, 29, 16, 27]
    idx1, _ = pmc.inner_index(30, 0, r_inner=1)
    assert idx1[0][:6].tolist() == [18, 20, 13, 29, 16, 27]
    # the old derivation (re-seeding PCG64 with generate_state(uint64)) is a different stream
    seed = int(np.random.SeedSequence(FROZEN_PMC["INNER_SEED"], spawn_key=(30, 0)).generate_state(1, dtype=np.uint64)[0])
    assert np.random.Generator(np.random.PCG64(seed)).integers(0, 30, 6).tolist() == [21, 15, 6, 0, 25, 19]
    # canonical index digest of the formal (N=30, r=0) inner design, pinned
    idx, sha = pmc.inner_index(30, 0, r_inner=FROZEN_PMC["R_INNER"])
    assert idx.shape == (FROZEN_PMC["R_INNER"], 300) and sha == FROZEN_INNER_30_0_SHA


def test_power_replicate_is_deterministic_and_shares_the_verdict_function():
    assert cana.evaluate_h1_verdict is hv.evaluate_h1_verdict and pmc.evaluate_h1_verdict is hv.evaluate_h1_verdict
    (BL, BH), sel, st = _stats("libero_10")
    rng = np.random.default_rng(0)
    cells = {}
    for a, s in FIXTURE["libero_10"]["stats"].items():
        cells[a] = {(t, i): (s["T"] / 300 * rng.uniform(0.8, 1.2), max(1, int(s["D"] / 300 * rng.uniform(0.8, 1.2))), float(rng.random() < s["S"] / s["E"]))
                    for t in range(10) for i in range(30)}
    roster = {"sv": sel["sv"], "threshold": sel["threshold"]}
    payload = {"cells": cells, "arms": sel["sv"] + sel["threshold"], "grid": sorted(cells["dsp_sv"]), "N": 40, "r": 3,
               "roster": roster, "B_L": BL, "B_H": BH, "nonformal_r_inner": 300}
    a = pmc.one_replicate(payload)
    b = pmc.one_replicate(dict(payload))
    assert a == b and a["formal"] is False and a["audit_inner0"] is not None
    assert set(a) == set(pmc.ROW_KEYS) and a["row_sha256"] == pmc.row_digest(a)
    other = pmc.one_replicate({**payload, "r": 4})
    assert other["outer_index_sha256"] != a["outer_index_sha256"] and other["inner_index_sha256"] != a["inner_index_sha256"]
    # the row digest covers the adjudication values, not only the index digests
    tampered = dict(a)
    tampered["q05"] = (a["q05"] or 0.0) + 1e-3
    assert pmc.row_digest(tampered) != a["row_sha256"]


@pytest.fixture(scope="module")
def power(chain, tgrid, budget):
    """The formal power pipeline (run -> record -> replay) under test-sized frozen constants."""
    if budget.design["verdict"] != "proceed_to_power":
        pytest.skip(f"synthetic development verdict {budget.design['verdict']}")
    mp = pytest.MonkeyPatch()
    mp.setattr(pmc, "N_CANDIDATES", TEST_N_CANDIDATES)
    mp.setattr(pmc, "R_OUTER", TEST_R_OUTER)
    mp.setattr(pmc, "R_INNER", TEST_R_INNER)
    mp.setattr(pmc, "POWER_TARGET", TEST_POWER_TARGET)
    tmp = budget.path.parent / "power"
    tmp.mkdir(exist_ok=True)
    args = _power_args(chain, tgrid, budget, tmp / "power_record.json")
    record = pmc.run(args)
    replay_args = _power_args(chain, tgrid, budget, tmp / "power_replay.json", power_record=str(tmp / "power_record.json"))
    replay = pmc.replay(replay_args)
    yield types.SimpleNamespace(tmp=tmp, record=record, record_path=tmp / "power_record.json", replay=replay,
                                replay_path=tmp / "power_replay.json", args=args, patch=mp)
    mp.undo()


def _validate_power(power, budget, record):
    return pmc.validate_power_record(record, outcome_design_path=budget.design_path, c_roster_path=budget.roster_path,
                                     budget_cost_map_path=budget.path)


def test_power_record_validates_and_forgeries_are_refused(power, budget, chain, tgrid, tmp_path):
    rec = json.loads(power.record_path.read_text())
    assert rec["smoke"] is False and len(rec["replicates"]) == len(TEST_N_CANDIDATES) * TEST_R_OUTER
    assert all(r["formal"] for r in rec["replicates"])
    assert rec["constants"]["R_INNER"] == TEST_R_INNER and rec["verdict"] in ("n_selected", "underpowered_stop")
    _validate_power(power, budget, rec)
    # G2R1-B3: the minimal record the old seal accepted is refused
    forged = {"protocol": pmc.PROTOCOL, "smoke": False, "verdict": "n_selected", "selected_N": 4, "c_roster_sha256": _sha(budget.roster_path)}
    with pytest.raises(SystemExit):
        _validate_power(power, budget, forged)
    # a tampered adjudication value breaks its row digest ...
    bad = json.loads(json.dumps(rec))
    bad["replicates"][0]["effect"] = (bad["replicates"][0]["effect"] or 0.0) + 0.01
    with pytest.raises(SystemExit, match="digest does not match"):
        _validate_power(power, budget, bad)
    # ... and re-digesting the row breaks the aggregate
    bad["replicates"][0]["row_sha256"] = pmc.row_digest(bad["replicates"][0])
    with pytest.raises(SystemExit, match="aggregate"):
        _validate_power(power, budget, bad)
    # selected N outside the candidates / not the mechanical rule
    bad = json.loads(json.dumps(rec))
    bad["selected_N"] = 4
    with pytest.raises(SystemExit):
        _validate_power(power, budget, bad)
    bad = json.loads(json.dumps(rec))
    bad["replicates"][0]["formal"] = False
    bad["replicates"][0]["row_sha256"] = pmc.row_digest(bad["replicates"][0])
    with pytest.raises(SystemExit, match="not formal"):
        _validate_power(power, budget, bad)
    # a smoke record is refused wherever a formal one is required
    smoke_args = _power_args(chain, tgrid, budget, tmp_path / "smoke.json", smoke=True)
    smoke = pmc.run(smoke_args)
    assert smoke["smoke"] is True and smoke["constants"]["R_OUTER"] == pmc.R_OUTER_SMOKE
    with pytest.raises(SystemExit):
        _validate_power(power, budget, json.loads((tmp_path / "smoke.json").read_text()))
    # --trials is pinned
    with pytest.raises(SystemExit, match="frozen at --trials 30"):
        pmc.run(_power_args(chain, tgrid, budget, tmp_path / "t.json", trials=10))


def test_power_replay_recomputes_a_digest_derived_subset(power, budget, tmp_path):
    rec = json.loads(power.record_path.read_text())
    rep = json.loads(power.replay_path.read_text())
    assert rep["passed"] and len(rep["replayed"]) == len(TEST_N_CANDIDATES) * pmc.REPLAY_PER_N
    assert [(x["N"], x["r"]) for x in rep["replayed"]] == pmc.replay_indices(rec)
    pmc.validate_power_replay(rep, rec, power.record_path)
    bad = json.loads(json.dumps(rep))
    bad["replayed"][0]["match"] = False
    with pytest.raises(SystemExit):
        pmc.validate_power_replay(bad, rec, power.record_path)
    bad = json.loads(json.dumps(rep))
    bad["replayed"] = bad["replayed"][1:]
    with pytest.raises(SystemExit):
        pmc.validate_power_replay(bad, rec, power.record_path)
    other = json.loads(json.dumps(rec))
    other["wall_seconds"] = 0.0
    other_path = _write_json(tmp_path / "other.json", other)
    with pytest.raises(SystemExit):
        pmc.validate_power_replay(rep, other, other_path)


# ------------------------------------------------------------------
# 5. fresh-init generator (plan 6-10)
# ------------------------------------------------------------------

class _StubEnv:
    calls = 0

    def __init__(self, bddl, fail_first=False):
        self.bddl = bddl
        self._fail = fail_first

    def seed(self, s):
        self._seed = s

    def reset(self):
        _StubEnv.calls += 1
        if self._fail and _StubEnv.calls % 2 == 1:
            raise RuntimeError("placement failed")
        self._state = np.random.uniform(size=3)

    def get_sim_state(self):
        return self._state

    def close(self):
        pass


def test_seed_domain_and_attempt_semantics():
    seeds = gen.derive_seeds(SUITE, "task_0", "C", 0, 0)
    assert int.from_bytes(bytes.fromhex(seeds["authority_sha256"]), "big") > 2 ** 32 - 1
    assert 0 <= seeds["seed32"] <= 2 ** 32 - 1 and seeds["py_seed"] == seeds["np_seed"] == seeds["env_seed"] == seeds["seed32"]
    np.random.seed(seeds["np_seed"])  # must not raise
    assert gen.derive_seeds(SUITE, "task_0", "C", 0, 0) == seeds
    assert gen.derive_seeds(SUITE, "task_0", "C", 0, 1)["seed32"] != seeds["seed32"]
    assert gen.MAX_RETRIES == 4


def test_generator_state_machine_and_quotas(chain, tmp_path):
    tasks = [{"task_id": t, "task_name": f"task_{t}", "bddl_file": str(tmp_path / f"t{t}.bddl")} for t in range(10)]
    block, states = gen.generate_pool(SUITE, "P", tasks, apool_dir=chain.world.pool_dir, state_dim=3,
                                      env_factory=lambda b: _StubEnv(b))
    gen.assert_pool_complete(block, "P")
    assert all(len(i["entries"]) == 10 and all(e["status"] == "ok" and e["attempts"][0]["a"] == 0 for e in i["entries"]) for i in block["tasks"].values())
    gen.materialize(block, states, tmp_path / "P")
    arr = gen.load_init_states(tmp_path / "P" / "task_0.init")
    assert len(arr) == 10 and gen.state_sha256(arr[0]) == block["tasks"]["task_0"]["entries"][0]["state_sha256"]
    assert gen.official_state_dim(chain.world.pool_dir, "task_0") == 3
    # retries: first reset raises, second succeeds -> two attempts recorded
    _StubEnv.calls = 0
    e = gen.sample_one(SUITE, "task_0", "C", 5, bddl_file="x", state_dim=3, env_factory=lambda b: _StubEnv(b, fail_first=True))
    assert e["status"] == "ok" and len(e["attempts"]) == 2 and e["attempts"][0]["outcome"].startswith("exception")
    # exhausted retries -> failed occupies k
    class Bad(_StubEnv):
        def reset(self):
            raise RuntimeError("never")
    e = gen.sample_one(SUITE, "task_0", "C", 6, bddl_file="x", state_dim=3, env_factory=lambda b: Bad(b))
    assert e["status"] == "failed" and len(e["attempts"]) == gen.MAX_RETRIES + 1
    # collision with an official state occupies k and fails the quota
    class Collide(_StubEnv):
        def reset(self):
            self._state = np.array([0.0, 0.0, 0.5])
    block2, _ = gen.generate_pool(SUITE, "P", tasks[:1], apool_dir=chain.world.pool_dir, state_dim=3, env_factory=lambda b: Collide(b))
    assert block2["tasks"]["task_0"]["entries"][0]["status"] == "collision"
    with pytest.raises(SystemExit):
        gen.assert_pool_complete(block2, "P")
    # shape check
    class Wide(_StubEnv):
        def reset(self):
            self._state = np.zeros(4)
    e = gen.sample_one(SUITE, "task_0", "C", 7, bddl_file="x", state_dim=3, env_factory=lambda b: Wide(b))
    assert e["status"] == "failed" and e["attempts"][0]["outcome"].startswith("bad_shape")


# ------------------------------------------------------------------
# 6. fresh pools / pilot / task plan / seal / discipline / analyzer (plan 6-9, 6-11, 6-12; G2R1-B4..B7, B9)
# ------------------------------------------------------------------

def _pool_manifest(block, pool, tm_sha, rollup, host="local-host"):
    return {"schema": 1, "protocol": gen.PROTOCOL, "suite": SUITE, "seed_namespace": gen.SEED_NAMESPACE, "max_retries": gen.MAX_RETRIES,
            "state_dim": 3, "environment": {"host": host, "numpy": np.__version__, "assets_rollup": rollup},
            "task_manifest_sha256": tm_sha, "pools": {pool: block}}


@pytest.fixture(scope="module")
def pools(chain, budget):
    """P / C pools with task manifest, assets rollup, peer manifests, cross-machine records and the validation artifact."""
    tmp = budget.path.parent / "pools_world"
    tmp.mkdir(exist_ok=True)
    (tmp / "bddl").mkdir(exist_ok=True)
    (tmp / "assets").mkdir(exist_ok=True)
    (tmp / "assets" / "mesh.bin").write_bytes(b"asset-bytes")
    tasks, tm_tasks = [], []
    for t in range(10):
        b = tmp / "bddl" / f"t{t}.bddl"
        b.write_text(f"(define (problem t{t}))\n")
        tasks.append({"task_id": t, "task_name": f"task_{t}", "bddl_file": str(b), "bddl_sha256": _sha(b)})
        tm_tasks.append({"task_id": t, "task_name": f"task_{t}", "bddl_file": b.name, "bddl_sha256": _sha(b)})
    tm_path = _write_json(tmp / "task_manifest.json", {"schema": 1, "suite": SUITE, "tasks": tm_tasks})
    rollup = gen.assets_rollup(tmp / "assets")
    manifests, peers, cross, dirs = {}, {}, {}, {}
    for pool in ("P", "C"):
        block, states = gen.generate_pool(SUITE, pool, tasks, apool_dir=chain.world.pool_dir, state_dim=3, env_factory=lambda b: _StubEnv(b))
        gen.assert_pool_complete(block, pool)
        dirs[pool] = tmp / "pools" / pool
        block["init_file_sha256"] = gen.materialize(block, states, dirs[pool])
        manifests[pool] = _write_json(tmp / f"pool_manifest_{pool}.json", _pool_manifest(block, pool, _sha(tm_path), rollup))
        peers[pool] = _write_json(tmp / f"pool_manifest_{pool}.peer.json", _pool_manifest(block, pool, _sha(tm_path), rollup, host="peer-host"))
        cross[pool] = _write_json(tmp / f"cross_machine_{pool}.json", gen.build_cross_machine_record(manifests[pool], peers[pool], pool))
    kwargs = dict(p_manifest_path=str(manifests["P"]), c_manifest_path=str(manifests["C"]), p_dir=str(dirs["P"]), c_dir=str(dirs["C"]),
                  apool_dir=str(chain.world.pool_dir), task_manifest_path=str(tm_path),
                  cross_p_record=str(cross["P"]), cross_p_peer=str(peers["P"]), cross_c_record=str(cross["C"]), cross_c_peer=str(peers["C"]),
                  assets_dir=str(tmp / "assets"))
    validation = _write_json(tmp / "fresh_pool_validation.json", gen.validate_pools(**kwargs))
    return types.SimpleNamespace(tmp=tmp, tasks=tasks, tm_path=tm_path, rollup=rollup, manifests=manifests, peers=peers, cross=cross,
                                 dirs=dirs, kwargs=kwargs, validation=validation)


def test_fresh_pool_validation_artifact_and_negatives(pools, chain, tmp_path):
    art = json.loads(pools.validation.read_text())
    assert art["passed"] and art["state_dim"] == 3 and art["exclusivity"] == {"official_vs_P": 0, "official_vs_C": 0, "P_vs_C": 0}
    assert art["pools"]["P"]["cross_machine"]["local_host"] == "local-host" and art["pools"]["P"]["cross_machine"]["peer_host"] == "peer-host"
    gen.validate_pool_validation(pools.validation, p_manifest_path=pools.manifests["P"], c_manifest_path=pools.manifests["C"])
    kw = dict(pools.kwargs)
    # G2R1-B5: a bare {"verified": true} is nothing; the peer must be a different host and recompute equal
    same_host = _write_json(tmp_path / "peer_same_host.json", json.loads(pools.manifests["P"].read_text()))
    with pytest.raises(SystemExit):
        gen.validate_pools(**{**kw, "cross_p_peer": str(same_host)})
    peer = json.loads(pools.peers["P"].read_text())
    peer["pools"]["P"]["tasks"]["task_0"]["entries"][0]["state_sha256"] = "0" * 64
    with pytest.raises(SystemExit):
        gen.validate_pools(**{**kw, "cross_p_peer": str(_write_json(tmp_path / "peer_drift.json", peer))})
    fake_record = _write_json(tmp_path / "cross_fake.json", {**json.loads(pools.cross["P"].read_text()), "verified": True, "problems": []})
    with pytest.raises(SystemExit):
        gen.validate_pools(**{**kw, "cross_p_record": str(fake_record), "cross_p_peer": str(same_host)})
    # self-reported state width, missing asset rollup, tampered init file, exclusivity collision
    m = json.loads(pools.manifests["C"].read_text())
    m["state_dim"] = 4
    with pytest.raises(SystemExit, match="state_dim"):
        gen.validate_pools(**{**kw, "c_manifest_path": str(_write_json(tmp_path / "c_dim.json", m))})
    m = json.loads(pools.manifests["C"].read_text())
    m["environment"]["assets_rollup"] = None
    with pytest.raises(SystemExit, match="asset rollup"):
        gen.validate_pools(**{**kw, "c_manifest_path": str(_write_json(tmp_path / "c_assets.json", m))})
    bad_dir = tmp_path / "C_tampered"
    shutil.copytree(pools.dirs["C"], bad_dir)
    import torch
    arr = np.asarray(gen.load_init_states(bad_dir / "task_0.init")).copy()
    arr[0, 0] += 1.0
    torch.save(arr, bad_dir / "task_0.init")
    with pytest.raises(SystemExit):
        gen.validate_pools(**{**kw, "c_dir": str(bad_dir)})
    m = json.loads(pools.manifests["C"].read_text())
    official = sorted(gen.official_state_digests(chain.world.pool_dir, "task_0"))[0]
    m["pools"]["C"]["tasks"]["task_0"]["entries"][0]["state_sha256"] = official
    with pytest.raises(SystemExit):
        gen.validate_pools(**{**kw, "c_manifest_path": str(_write_json(tmp_path / "c_collide.json", m))})
    # the artifact itself cannot be edited
    tampered = dict(art)
    tampered["exclusivity"] = {"official_vs_P": 0, "official_vs_C": 0, "P_vs_C": 1}
    with pytest.raises(SystemExit):
        gen.validate_pool_validation(_write_json(tmp_path / "art.json", tampered), p_manifest_path=pools.manifests["P"], c_manifest_path=pools.manifests["C"])


def _c_rows(arm, family, t, prefix, verdicts, success, run_id=RUN_C):
    uid = f"{arm}:eval:{t}:{prefix}"
    rows = [{"yaml_id": arm, "task_uid": uid, "task_id": t, "orig_init_state_idx": None, "subset_init_state_idx": prefix,
             "episode_id": t * 100 + prefix, "step_idx": i * 5, "phase": "eval", "hit_type": h, "start_t": 0.3 if h == WARM else None,
             "attempt": 1, "accepted": True, "run_id": run_id, "success": success} for i, h in enumerate(verdicts)]
    rows.append({"_kind": "client_timing", "task_uid": uid, "yaml_id": arm, "task_id": t, "subset_init_state_idx": prefix,
                 "infer_ms": 100.0 * len(verdicts), "infers": len(verdicts), "steps": 5 * len(verdicts), "attempt": 1, "accepted": True,
                 "run_id": run_id, "success": success})
    journal = {"yaml_id": arm, "task_uid": uid, "phase": "eval", "status": "done" if success else "failed", "success": success,
               "accepted": True, "attempt": 1, "run_id": run_id}
    return rows, journal


def _write_rows(run_dir, per_step, journal):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "per_step.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_step))
    (run_dir / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in journal))
    return run_dir / "journal.jsonl", run_dir / "per_step.jsonl"


@pytest.fixture(scope="module")
def pilot(chain, budget, pools):
    """The P pilot chain: P task plan -> synthetic 100-episode anchor run -> one-launch ledger -> finalize."""
    tmp = pools.tmp / "pilot"
    tmp.mkdir(exist_ok=True)
    roster = json.loads(budget.roster_path.read_text())
    anchor_yaml = pathlib.Path(next(e["yaml_path"] for e in roster["arms"] if e["arm"] == ANCHOR_ARM))
    plan = tplan.build_task_plan(SUITE, [ANCHOR_ARM], pilot_mod.PILOT_TRIALS, str(pools.manifests["P"]), pool_id="P")
    plan_path = _write_json(tmp / "pilot_task_plan.json", plan)
    rng = np.random.default_rng(21)
    per_step, journal = [], []
    for t in range(10):
        for prefix in range(pilot_mod.PILOT_TRIALS):
            rows, j = _c_rows(ANCHOR_ARM, "anchor", t, prefix, [MISS] * int(rng.integers(6, 12)), bool(rng.random() < 0.85), run_id=RUN_P)
            per_step += rows
            journal.append(j)
    journal_path, per_step_path = _write_rows(tmp / "run", per_step, journal)
    pool = validate_pool_files(str(pools.manifests["P"]), _sha(pools.manifests["P"]), "P", str(pools.dirs["P"]), pilot_mod.PILOT_TRIALS)
    entry = {"protocol": PROTOCOL_PILOT, "layer": LAYER_PILOT, "suite": SUITE, "run_id": RUN_P,
             "executed_arms": [ANCHOR_ARM], "core_arms": [ANCHOR_ARM], "descriptive_arms": [],
             "trials_per_task": pilot_mod.PILOT_TRIALS, "replan_steps": 5, "env_seed": 7,
             "policy_fingerprint": chain.world.contract["policy_fingerprint"],
             "contract_binding": {"h_exec": 5, "policy_fingerprint": chain.world.contract["policy_fingerprint"], "servers": {}},
             "library_sha256": _sha(chain.world.lib), "aprime_content_sha256": pool["rollup_sha256"], "split_manifest": "",
             "split_manifest_sha256": None, "arm_matrix_sha256": _sha(plan_path), "frozen_yaml_sha256": {ANCHOR_ARM: _sha(anchor_yaml)},
             "artifact_sha256": {}, "fit_record_sha256": None, "executed_yaml_sha256": {ANCHOR_ARM: _sha(anchor_yaml)}, "pool": pool,
             "task_plan_sha256": _sha(plan_path), "pool_digest": pool["rollup_sha256"], "N": pilot_mod.PILOT_TRIALS,
             "cost_model_digest": cost_model_digest(), "rev1_package_manifest_sha256": _sha(chain.manifest_path)}
    ledger = _write_json(tmp / "run" / "per_step.jsonl.launch.json", {"schema_version": 2, "launches": [entry]})
    record = pilot_mod.finalize(str(plan_path), str(pools.manifests["P"]), str(ledger), str(journal_path), str(per_step_path),
                                str(anchor_yaml), str(tmp / "pilot_record.json"))
    return types.SimpleNamespace(tmp=tmp, plan=plan, plan_path=plan_path, anchor_yaml=anchor_yaml, journal=journal_path,
                                 per_step=per_step_path, ledger=ledger, entry=entry, record=record, record_path=tmp / "pilot_record.json")


def test_pilot_chain_and_negatives(pilot, pools, tmp_path):
    rec = pilot.record
    assert rec["passed"] and rec["n_episodes"] == 100 and rec["attempt"] == 1 and rec["sr"] == rec["successes"] / 100
    assert abs(rec["sr"] - pilot_mod.PHASE0_ANCHOR_SR[SUITE]) * 100 <= pilot_mod.PILOT_TOLERANCE_PT
    anchor_sha = _sha(pilot.anchor_yaml)
    pilot_mod.validate_pilot(str(pilot.record_path), suite=SUITE, pool_manifest_p_path=str(pools.manifests["P"]), anchor_yaml_sha256=anchor_sha)

    def fin(journal=None, per_step=None, ledger=None, plan=None, manifest=None, out="x.json"):
        return pilot_mod.finalize(str(plan or pilot.plan_path), str(manifest or pools.manifests["P"]), str(ledger or pilot.ledger),
                                  str(journal or pilot.journal), str(per_step or pilot.per_step), str(pilot.anchor_yaml), str(tmp_path / out))

    lines = pilot.journal.read_text().splitlines()
    # one state missing
    p = tmp_path / "j_short.jsonl"
    p.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(SystemExit):
        fin(journal=p)
    # duplicate accepted
    p = tmp_path / "j_dup.jsonl"
    p.write_text("\n".join(lines + [lines[0]]) + "\n")
    with pytest.raises(SystemExit):
        fin(journal=p)
    # accepted under another run
    row = json.loads(lines[0])
    row["run_id"] = "runother"
    p = tmp_path / "j_run.jsonl"
    p.write_text("\n".join(lines[1:] + [json.dumps(row)]) + "\n")
    with pytest.raises(SystemExit):
        fin(journal=p)
    # two launches = a retried pilot
    bad = json.loads(pilot.ledger.read_text())
    bad["launches"].append(dict(bad["launches"][0], run_id="runpilot00002"))
    with pytest.raises(SystemExit, match="one-shot"):
        fin(ledger=_write_json(tmp_path / "l2.json", bad))
    # wrong pool manifest (C instead of P)
    with pytest.raises(SystemExit):
        fin(manifest=pools.manifests["C"])
    # a non-anchor plan
    other = tplan.build_task_plan(SUITE, ["dsp_sv"], pilot_mod.PILOT_TRIALS, str(pools.manifests["P"]), pool_id="P")
    with pytest.raises(SystemExit):
        fin(plan=_write_json(tmp_path / "plan_other.json", other))
    # hand-edited SR in the record is refused by the validator (G2R1-B4)
    edited = dict(rec)
    edited["sr"] = 0.85
    edited["successes"] = 85
    with pytest.raises(SystemExit):
        pilot_mod.validate_pilot(str(_write_json(tmp_path / "edited.json", edited)), suite=SUITE,
                                 pool_manifest_p_path=str(pools.manifests["P"]), anchor_yaml_sha256=anchor_sha)
    with pytest.raises(SystemExit):
        pilot_mod.validate_pilot(str(pilot.record_path), suite=SUITE, pool_manifest_p_path=str(pools.manifests["C"]), anchor_yaml_sha256=anchor_sha)
    # the old three-field record
    old = _write_json(tmp_path / "old.json", {"arm": ANCHOR_ARM, "pool_id": "P", "sr": 0.85, "attempt": 1})
    with pytest.raises(SystemExit):
        pilot_mod.validate_pilot(str(old), suite=SUITE, pool_manifest_p_path=str(pools.manifests["P"]), anchor_yaml_sha256=anchor_sha)
    # SR outside tolerance -> generator_validation_failed, record written with passed=False
    bad_j = [json.loads(line) for line in lines]
    for r in bad_j:
        r["status"], r["success"] = "failed", False
    p = tmp_path / "j_fail.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in bad_j))
    with pytest.raises(SystemExit, match="generator_validation_failed"):
        fin(journal=p, out="failed_record.json")
    assert json.loads((tmp_path / "failed_record.json").read_text())["passed"] is False


def _action_cache_no(tmp):
    return _write_json(tmp / "action_cache.json", {"inclusion": "no", "reason_code": "not_in_confirmation_scope", "statistical_status": None,
                                                   "development_selection_protocol": None, "config_digest": None, "code_digest": None,
                                                   "cost_mapping": acd.COST_MAPPING_FROZEN, "c_pool_binding": None,
                                                   "claim_restriction": {"forbid": ["superiority over Action Cache"]}})


def _seal_args(budget, power, pools, pilot, tmp, **extra):
    args = types.SimpleNamespace(outcome_design=str(budget.design_path), c_roster=str(budget.roster_path), budget_cost_map=str(budget.path),
                                 power_record=str(power.record_path), power_replay=str(power.replay_path),
                                 pool_manifest_c=str(pools.manifests["C"]), pool_manifest_p=str(pools.manifests["P"]),
                                 pool_validation=str(pools.validation), pilot_record=str(pilot.record_path),
                                 task_plan=str(tmp / "confirmation_task_plan.json"), action_cache_record=str(_action_cache_no(tmp)),
                                 action_cache_package="", protocol=str(REPO / "logs/dispatch_surface_rev2_protocol_draft.md"))
    for k, v in extra.items():
        setattr(args, k, v)
    return args


@pytest.fixture(scope="module")
def confirmation(chain, budget, power, pools, pilot):
    assert power.record["verdict"] == "n_selected", f"synthetic power verdict {power.record['verdict']}: the confirmation chain would be untested"
    tmp = budget.path.parent / "confirm"
    tmp.mkdir(exist_ok=True)
    roster = json.loads(budget.roster_path.read_text())
    arms = [e["arm"] for e in roster["arms"]]
    N = int(power.record["selected_N"])
    plan = tplan.build_task_plan(SUITE, arms, N, str(pools.manifests["C"]))
    plan_path = _write_json(tmp / "confirmation_task_plan.json", plan)
    seal_args = _seal_args(budget, power, pools, pilot, tmp)
    seal = sealmod.build_seal(seal_args)
    seal_path = _write_json(tmp / sealmod.SEAL_NAME, seal)
    seal, seal_sha = sealmod.load_seal(seal_path)
    # synthetic C rollout
    rng = np.random.default_rng(3)
    per_step, journal = [], []
    fams = seal["roster"]["families"]
    for arm in seal["roster"]["arms"]:
        for t in range(10):
            for prefix in range(N):
                n = int(rng.integers(6, 12))
                if fams[arm] == "anchor":
                    verdicts = [MISS] * n
                else:
                    p_full = {"sv": 0.5, "s0": 0.4, "threshold": 0.45}[fams[arm]]
                    u = rng.random(n)
                    verdicts = [FULL if x < p_full else (WARM if x < p_full + 0.2 else MISS) for x in u]
                success = bool(rng.random() < {"sv": 0.7, "s0": 0.6, "threshold": 0.55, "anchor": 0.85}[fams[arm]])
                rows, j = _c_rows(arm, fams[arm], t, prefix, verdicts, success)
                per_step += rows
                journal.append(j)
    journal_path, per_step_path = _write_rows(tmp / "run", per_step, journal)
    pool = validate_fresh_pool(seal, str(pools.dirs["C"]), N)
    entry = {"protocol": sealmod.PROTOCOL, "layer": "confirmation", "suite": SUITE, "run_id": RUN_C,
             "executed_arms": sorted(seal["roster"]["arms"]), "core_arms": sorted(seal["roster"]["arms"]), "descriptive_arms": [],
             "trials_per_task": N, "replan_steps": 5, "env_seed": 7, "policy_fingerprint": chain.world.contract["policy_fingerprint"],
             "contract_binding": {"h_exec": 5, "policy_fingerprint": chain.world.contract["policy_fingerprint"], "servers": {}},
             "library_sha256": seal["library_sha256"], "aprime_content_sha256": pool["rollup_sha256"], "split_manifest": "",
             "split_manifest_sha256": None, "arm_matrix_sha256": seal_sha, "frozen_yaml_sha256": seal["roster"]["yaml_sha256"],
             "artifact_sha256": seal["roster"]["artifact_sha256"], "fit_record_sha256": None,
             "executed_yaml_sha256": seal["roster"]["yaml_sha256"], "pool": pool, "seal_sha256": seal_sha,
             "confirmation_task_plan_sha256": _sha(plan_path), "pool_digest": pool["rollup_sha256"], "N": N,
             "estimator_version": seal["estimator_digest"], "cost_model_digest": seal["cost_model_digest"]}
    ledger = _write_json(tmp / "run" / "per_step.jsonl.launch.json", {"schema_version": 2, "launches": [entry]})
    disc = cdisc.certify(str(seal_path), str(plan_path), str(ledger), str(journal_path), str(per_step_path))
    disc_path = _write_json(tmp / "confirmation_discipline.json", disc)
    unseal = sealmod.write_unseal(str(seal_path), str(disc_path), str(ledger), str(tmp / sealmod.UNSEAL_NAME),
                                  task_plan_path=str(plan_path), journal_path=str(journal_path), per_step_path=str(per_step_path))
    return types.SimpleNamespace(tmp=tmp, seal=seal, seal_path=seal_path, seal_sha=seal_sha, plan=plan, plan_path=plan_path, N=N,
                                 manifests=pools.manifests, ledger=ledger, entry=entry, journal=journal_path, per_step=per_step_path,
                                 disc=disc, disc_path=disc_path, unseal=unseal, pool=pool, seal_args=seal_args)


def test_task_plan_has_no_seal_cycle_and_seal_binds_it(confirmation):
    assert "seal_sha256" not in confirmation.plan and "seal" not in confirmation.plan
    assert confirmation.seal["confirmation_task_plan_sha256"] == _sha(confirmation.plan_path)
    with pytest.raises(SystemExit):
        tplan.assert_no_cycle({**confirmation.plan, "seal_sha256": "x"})
    with pytest.raises(SystemExit):
        tplan.assert_no_cycle({**confirmation.plan, "entries": {"a": {"seal": "x"}}})
    assert confirmation.entry["seal_sha256"] == confirmation.seal_sha and confirmation.entry["confirmation_task_plan_sha256"] == _sha(confirmation.plan_path)
    for key in CONFIRMATION_FROZEN_LAUNCH_KEYS:
        assert key in confirmation.entry
    assert confirmation.seal["power_replay_sha256"] and confirmation.seal["pool"]["validation_sha256"] and confirmation.seal["pilot"]["n_episodes"] == 100


def test_task_plan_loader_requires_the_exact_cartesian_roster(confirmation, pools, tmp_path):
    """G2R1-B6: the ghost-arm plan the old loader accepted (count and per-entry self-consistency only)."""
    from openpi.conductor.task import make_task_uid
    good = confirmation.plan
    ghost = {**good, "roster_arms": ["a"], "N": 1, "entries": {}}
    for i in range(10):
        arm = f"ghost{i}"
        uid = make_task_uid(arm, "eval", 0, 0)
        ghost["entries"][uid] = {"arm": arm, "task_id": 0, "task_name": "task_0", "prefix_idx": 0, "pool_id": "C", "fresh_state_sha256": "a" * 64}
    assert len(ghost["entries"]) == len(ghost["roster_arms"]) * 10 * ghost["N"]   # the old count check passed
    with pytest.raises(SystemExit):
        tplan.validate_task_plan(ghost)
    with pytest.raises(SystemExit):
        tplan.load_task_plan(_write_json(tmp_path / "ghost.json", ghost))
    # an arm missing one cell, an extra arm, a wrong task name, an out-of-order pool entry, N beyond the quota
    missing = json.loads(json.dumps(good))
    missing["entries"].pop(next(iter(missing["entries"])))
    with pytest.raises(SystemExit):
        tplan.validate_task_plan(missing)
    extra = json.loads(json.dumps(good))
    extra["roster_arms"] = sorted(extra["roster_arms"] + ["zzz"])
    with pytest.raises(SystemExit):
        tplan.validate_task_plan(extra)
    wrong_name = json.loads(json.dumps(good))
    k = next(iter(wrong_name["entries"]))
    wrong_name["entries"][k]["task_name"] = "task_9" if wrong_name["entries"][k]["task_name"] != "task_9" else "task_8"
    with pytest.raises(SystemExit):
        tplan.validate_task_plan(wrong_name)
    m = json.loads(pools.manifests["C"].read_text())
    ents = m["pools"]["C"]["tasks"]["task_0"]["entries"]
    ents[0], ents[1] = ents[1], ents[0]
    with pytest.raises(SystemExit):
        tplan.verify_task_plan_against_pool(good, str(_write_json(tmp_path / "c_reordered.json", m)))
    with pytest.raises(SystemExit):
        tplan.build_task_plan(SUITE, ["a"], 61, str(pools.manifests["C"]))


def test_confirmation_discipline_negatives(confirmation, tmp_path):
    c = confirmation
    def run(journal=None, per_step=None, ledger=None, plan=None):
        return cdisc.certify(str(c.seal_path), str(plan or c.plan_path), str(ledger or c.ledger), str(journal or c.journal), str(per_step or c.per_step))
    assert run()["passed"]
    lines = c.journal.read_text().splitlines()
    # 1 partial ledger (an arm missing from executed_arms)
    bad = json.loads(c.ledger.read_text())
    bad["launches"][0]["executed_arms"] = bad["launches"][0]["executed_arms"][1:]
    with pytest.raises(SystemExit):
        run(ledger=_write_json(tmp_path / "l1.json", bad))
    # 2 duplicate accepted
    p = tmp_path / "j2.jsonl"
    p.write_text("\n".join(lines + [lines[0]]) + "\n")
    with pytest.raises(SystemExit):
        run(journal=p)
    # 3 stale attempt (attempt 2 accepted under an unknown run)
    row = json.loads(lines[0])
    row["attempt"] = 2
    row["run_id"] = "runstale"
    p = tmp_path / "j3.jsonl"
    p.write_text("\n".join(lines[1:] + [json.dumps(row)]) + "\n")
    with pytest.raises(SystemExit):
        run(journal=p)
    # 4 wrong pool state (plan digest tampered)
    plan = json.loads(c.plan_path.read_text())
    k = next(iter(plan["entries"]))
    plan["entries"][k]["fresh_state_sha256"] = "0" * 64
    with pytest.raises(SystemExit):
        run(plan=_write_json(tmp_path / "plan4.json", plan))
    # 5 an arm missing one cell
    p = tmp_path / "j5.jsonl"
    p.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(SystemExit):
        run(journal=p)
    # 6 per-step short by one decision row
    ps = c.per_step.read_text().splitlines()
    first_dec = next(i for i, line in enumerate(ps) if "client_timing" not in line)
    p = tmp_path / "ps6.jsonl"
    p.write_text("\n".join(ps[:first_dec] + ps[first_dec + 1:]) + "\n")
    with pytest.raises(SystemExit):
        run(per_step=p)
    # 7 off-grid cell
    row = json.loads(lines[0])
    arm = row["yaml_id"]
    row["task_uid"] = f"{arm}:eval:0:{c.N + 5}"
    p = tmp_path / "j7.jsonl"
    p.write_text("\n".join(lines + [json.dumps(row)]) + "\n")
    with pytest.raises(SystemExit):
        run(journal=p)
    # 8 orig_init_state_idx not null
    rows = [json.loads(line) for line in ps]
    for r in rows:
        if r.get("_kind") != "client_timing":
            r["orig_init_state_idx"] = 3
            break
    p = tmp_path / "ps8.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit):
        run(per_step=p)
    # 9 ledger bound to another seal digest
    bad = json.loads(c.ledger.read_text())
    bad["launches"][0]["seal_sha256"] = "1" * 64
    with pytest.raises(SystemExit):
        run(ledger=_write_json(tmp_path / "l9.json", bad))


def test_discipline_and_cost_io_never_read_outcomes():
    for rel in ("exp/dispatch_surface/analysis/confirmation_discipline.py", "exp/dispatch_surface/analysis/confirmation_io.py"):
        tree = ast.parse((REPO / rel).read_text())
        consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert "success" not in consts and "status" not in consts, rel


def test_real_conductor_path_round_trip(confirmation):
    from examples.libero.episode_runner import _hit_row
    from openpi.conductor import protocol, task as _task
    from openpi.conductor.task import make_task_uid

    seal = confirmation.seal
    arm = seal["roster"]["arms"][0]
    uid = make_task_uid(arm, "eval", 2, 1)
    t = _task.EpisodeTask(task_uid=uid, yaml_id=arm, phase="eval", experiment=SUITE, task_id=2, episode_idx=1,
                          orig_init_state_idx=None, server_host="h", server_port=1, bundle_id=arm, extra={"num_trials_per_task": confirmation.N})
    wire = json.loads(json.dumps(protocol.task_to_wire(t)))
    back = protocol.task_from_wire(wire)
    assert back.orig_init_state_idx is None and back.task_uid == uid
    row = _hit_row(back, 0, {"hit_type": MISS, "start_t": None}, confirmation.N)
    assert row["orig_init_state_idx"] is None and row["subset_init_state_idx"] == 1 and row["task_uid"] == uid
    assert uid in confirmation.plan["entries"]
    # the strategy builds exactly the plan's episodes
    strat = ConfirmationStrategy(SUITE, {arm: seal["roster"]["yaml_paths"][arm]}, confirmation.N, confirmation.plan)
    srv = types.SimpleNamespace(host="h", port=1, key="h:1")
    graph = strat.plan([arm], {arm: srv})
    stages = graph.stages.values() if isinstance(getattr(graph, "stages", None), dict) else list(getattr(graph, "stages", []))
    eps = [e for st in stages for e in getattr(st, "episodes", [])]
    assert eps and all(e.orig_init_state_idx is None for e in eps) and len(eps) == 10 * confirmation.N


def test_fresh_pool_validation_and_confirmation_arms(confirmation, pools, tmp_path):
    c = confirmation
    pool = validate_fresh_pool(c.seal, str(pools.dirs["C"]), c.N)
    assert pool["pool_id"] == "C" and pool["total_inits"] == 10 * c.N
    with pytest.raises(SystemExit):
        validate_fresh_pool(c.seal, str(pools.dirs["P"]), c.N)   # wrong pool bytes
    with pytest.raises(SystemExit):
        validate_fresh_pool(c.seal, str(pools.dirs["C"]), 999)
    validate_confirmation_arms(dict(c.seal["roster"]["yaml_paths"]), c.seal)
    with pytest.raises(SystemExit):
        validate_confirmation_arms({"ghost": next(iter(c.seal["roster"]["yaml_paths"].values()))}, c.seal)


def test_seal_refuses_bad_inputs(confirmation, budget, power, pools, pilot, tmp_path):
    c = confirmation
    # G2R1-B9: inclusion=yes fails closed even with a full digest object
    ac_yes = _write_json(tmp_path / "ac_yes.json", {"inclusion": "yes", "reason_code": "x", "statistical_status": "secondary",
                                                    "development_selection_protocol": {"a": 1}, "config_digest": "a" * 64, "code_digest": "b" * 64,
                                                    "cost_mapping": acd.COST_MAPPING_FROZEN, "c_pool_binding": {"pool": "C"},
                                                    "claim_restriction": {"x": 1}})
    full_pkg = {k: hashlib.sha256(k.encode()).hexdigest() for k in acd.YES_PACKAGE_FIELDS}
    assert acd.seal_branch(acd.load_record(ac_yes), None)["ok"] is False
    assert acd.seal_branch(acd.load_record(ac_yes), full_pkg)["ok"] is False
    with pytest.raises(SystemExit):
        sealmod.build_seal(_seal_args(budget, power, pools, pilot, c.tmp, action_cache_record=str(ac_yes),
                                      action_cache_package=str(_write_json(tmp_path / "pkg.json", full_pkg))))
    with pytest.raises(SystemExit):
        acd.validate_record({"inclusion": "no", "reason_code": "x", "statistical_status": None, "development_selection_protocol": "not-null",
                             "config_digest": None, "code_digest": None, "cost_mapping": acd.COST_MAPPING_FROZEN, "c_pool_binding": None,
                             "claim_restriction": {"x": 1}})
    with pytest.raises(SystemExit):
        acd.assert_no_action_cache_fields({"vs_action_cache": 1}, what="t")
    # G2R1-B3: the minimal forged power record and a smoke record are refused
    forged = _write_json(tmp_path / "forged_power.json", {"protocol": pmc.PROTOCOL, "smoke": False, "verdict": "n_selected", "selected_N": 4,
                                                          "c_roster_sha256": _sha(budget.roster_path)})
    with pytest.raises(SystemExit):
        sealmod.build_seal(_seal_args(budget, power, pools, pilot, c.tmp, power_record=str(forged)))
    smoke = json.loads(power.record_path.read_text())
    smoke["smoke"] = True
    with pytest.raises(SystemExit):
        sealmod.build_seal(_seal_args(budget, power, pools, pilot, c.tmp, power_record=str(_write_json(tmp_path / "smoke.json", smoke))))
    # a replay that does not bind this record
    rep = json.loads(power.replay_path.read_text())
    rep["power_record_sha256"] = "0" * 64
    with pytest.raises(SystemExit):
        sealmod.build_seal(_seal_args(budget, power, pools, pilot, c.tmp, power_replay=str(_write_json(tmp_path / "rep.json", rep))))
    # G2R1-B4: the old hand-written pilot record
    bad_pilot = _write_json(tmp_path / "pilot_bad.json", {"arm": ANCHOR_ARM, "pool_id": "P", "sr": 0.85, "attempt": 1})
    with pytest.raises(SystemExit):
        sealmod.build_seal(_seal_args(budget, power, pools, pilot, c.tmp, pilot_record=str(bad_pilot)))
    # G2R1-B5: a manifest with a hand-written cross_machine block but no validation artifact
    m = json.loads(pools.manifests["C"].read_text())
    m["cross_machine"] = {"verified": True, "problems": []}
    with pytest.raises(SystemExit):
        sealmod.build_seal(_seal_args(budget, power, pools, pilot, c.tmp, pool_manifest_c=str(_write_json(tmp_path / "c_hand.json", m))))
    art = json.loads(pools.validation.read_text())
    art["state_dim"] = 47
    with pytest.raises(SystemExit):
        sealmod.build_seal(_seal_args(budget, power, pools, pilot, c.tmp, pool_validation=str(_write_json(tmp_path / "art.json", art))))
    # G2R1-B6: a ghost-arm plan
    from openpi.conductor.task import make_task_uid
    ghost = {**c.plan, "roster_arms": ["a"], "N": 1, "entries": {}}
    for i in range(10):
        uid = make_task_uid(f"ghost{i}", "eval", 0, 0)
        ghost["entries"][uid] = {"arm": f"ghost{i}", "task_id": 0, "task_name": "task_0", "prefix_idx": 0, "pool_id": "C", "fresh_state_sha256": "a" * 64}
    with pytest.raises(SystemExit):
        sealmod.build_seal(_seal_args(budget, power, pools, pilot, c.tmp, task_plan=str(_write_json(tmp_path / "ghost.json", ghost))))


def test_unseal_recertifies_and_refuses_a_foreign_ledger(confirmation, tmp_path):
    """G2R1-B7: the unseal re-runs certify; a passing discipline of another ledger is not reusable."""
    c = confirmation
    assert c.unseal["seal_sha256"] == c.seal_sha and c.unseal["discipline_sha256"] == _sha(c.disc_path)
    assert c.unseal["ledger_sha256"] == _sha(c.ledger) and c.unseal["task_plan_sha256"] == _sha(c.plan_path)

    def unseal(disc=None, ledger=None, out="u.json"):
        return sealmod.write_unseal(str(c.seal_path), str(disc or c.disc_path), str(ledger or c.ledger), str(tmp_path / out),
                                    task_plan_path=str(c.plan_path), journal_path=str(c.journal), per_step_path=str(c.per_step))

    unseal(out="ok.json")
    bad = json.loads(c.disc_path.read_text())
    bad["passed"] = False
    with pytest.raises(SystemExit):
        unseal(disc=_write_json(tmp_path / "d.json", bad))
    bad = json.loads(c.disc_path.read_text())
    bad["ledger_sha256"] = "WRONG"
    with pytest.raises(SystemExit):
        unseal(disc=_write_json(tmp_path / "d_wrong.json", bad))
    # same seal, journal / per-step unchanged, ledger swapped for a benignly-different one: the OLD discipline is refused
    other = json.loads(c.ledger.read_text())
    other["launches"][0]["note"] = "re-launched"
    other_path = _write_json(tmp_path / "ledger_other.json", other)
    with pytest.raises(SystemExit):
        unseal(ledger=other_path)
    # a fresh certification of that (still complete) ledger is what unseal accepts: re-certify, never reuse
    fresh = cdisc.certify(str(c.seal_path), str(c.plan_path), str(other_path), str(c.journal), str(c.per_step))
    assert fresh["ledger_sha256"] == _sha(other_path)
    unseal(disc=_write_json(tmp_path / "d_fresh.json", fresh), ledger=other_path, out="ok2.json")
    # a forged discipline that claims a ledger certify would refuse (one arm never executed) is refused
    partial = json.loads(c.ledger.read_text())
    partial["launches"][0]["executed_arms"] = partial["launches"][0]["executed_arms"][1:]
    partial_path = _write_json(tmp_path / "ledger_partial.json", partial)
    forged = json.loads(c.disc_path.read_text())
    forged["ledger_sha256"] = _sha(partial_path)
    with pytest.raises(SystemExit):
        unseal(disc=_write_json(tmp_path / "d_forged.json", forged), ledger=partial_path)


def test_analyzer_verdict_and_cross_checks(confirmation, tmp_path):
    c = confirmation
    view = cana.cost_only_view(str(c.seal_path), str(c.journal), str(c.per_step))
    assert view["cost_only"] and set(view["arms"]) == set(c.seal["roster"]["arms"])
    out = cana.analyze(str(c.seal_path), str(c.tmp / sealmod.UNSEAL_NAME), str(c.disc_path), str(c.ledger), str(c.journal), str(c.per_step))
    assert out["verdict"] in ("h1_pass", "h1_fail", "support_miss")
    assert "passed" in out["hypotheses"]["H1"] and "passed" not in out["hypotheses"].get("H2", {})
    assert out["A4_anchor"]["passed"] and out["secondary_band"]["exploratory"]
    assert out["estimator_version"] == ev.budget_mixture_digest()
    acd.assert_no_action_cache_fields(out, what="analyzer")
    bad_unseal = dict(c.unseal)
    bad_unseal["ledger_sha256"] = "0" * 64
    with pytest.raises(SystemExit):
        cana.analyze(str(c.seal_path), str(_write_json(tmp_path / "u2.json", bad_unseal)), str(c.disc_path), str(c.ledger), str(c.journal), str(c.per_step))
    # G2R1-B7: a discipline of another ledger cannot feed the analyzer even with a matching unseal digest
    other = json.loads(c.ledger.read_text())
    other["launches"][0]["note"] = "x"
    other_path = _write_json(tmp_path / "ledger_other.json", other)
    u = dict(c.unseal)
    u["ledger_sha256"] = _sha(other_path)
    with pytest.raises(SystemExit):
        cana.analyze(str(c.seal_path), str(_write_json(tmp_path / "u3.json", u)), str(c.disc_path), str(other_path), str(c.journal), str(c.per_step))


# ------------------------------------------------------------------
# 7. frozen Phase 0 code is untouched (plan 6-0 snapshot)
# ------------------------------------------------------------------

def test_frontier_hull_is_byte_frozen():
    assert _sha(REPO / "exp/dispatch_surface/analysis/frontier_hull.py") == \
        hashlib.sha256((REPO / "exp/dispatch_surface/analysis/frontier_hull.py").read_bytes()).hexdigest()
    src = (REPO / "exp/dispatch_surface/analysis/frontier_hull.py").read_text()
    assert "budget" not in src.lower()   # no confirmation-plan code leaked into the frozen module
