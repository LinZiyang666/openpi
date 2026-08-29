"""Rev 2 Phase 0 tool-chain tests (plan logs/dispatch_surface_rev2_phase0_plan.log.md section 6).

One synthetic-but-schema-faithful Rev 1 discipline package drives the whole
chain: archive -> exploratory export -> emit -> runner validation -> Phase 0
discipline/summary -> outcome-blind cost map -> outcome design. The D0 fixture
is a minimal copy of a REAL D0 record (exact key set), rebound to the
synthetic inputs by digest only.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import pathlib
import types

import numpy as np
import pytest
import yaml

from exp.dispatch_surface import archive_rev1_discipline as archive
from exp.dispatch_surface import export_exploratory_surface as exporter
from exp.dispatch_surface import rev1_package as pkgmod
from exp.dispatch_surface.analysis import analytic_cost as ac
from exp.dispatch_surface.analysis import cost_map as cm
from exp.dispatch_surface.analysis import frontier_hull as fh
from exp.dispatch_surface.analysis import phase0_discipline, phase0_outcome_design, phase0_summary
from exp.dispatch_surface.analysis import precheck_io
from exp.dispatch_surface.d0_check import _canonical_digest
from exp.dispatch_surface.emit_precheck_yamls import LAYER_EXPLORATORY, emit_exploratory
from exp.dispatch_surface.fit_surface import (
    GRID_LADDER_S_ONLY,
    GRID_LADDER_SV,
    _digest_obj,
    export_boundaries,
    final_fit,
    load_table,
)
from exp.dispatch_surface.phase0_roster import (
    ANCHOR_ARM,
    CONTRACT_ANCHOR_ARM,
    ROSTERS,
    roster_spec_digest,
)
from exp.dispatch_surface.run_precheck import (
    EXPLORATORY_FROZEN_LAUNCH_KEYS,
    FROZEN_LAUNCH_KEYS,
    validate_exploratory_matrix_artifacts,
    validate_existing_launch_ledger,
    validate_precheck_arms,
)
from openpi.cache.components.surface_judge import (
    CERTIFICATION_EMPIRICAL,
    SURFACE_ARTIFACT_SCHEMA_VERSION,
    SurfaceArtifact,
    load_surface_artifact,
    save_surface_artifact,
)

HERE = pathlib.Path(__file__).parent
REPO = HERE.parents[1]
TEMPLATE = REPO / "exp/gate_research/config/libero_10/eval/cp1_spatial_pool_16__grid3_vision_0@56_vision_1@25_robot_state@18__d1__fh40_ws40_quantile.yaml"
SUITE = "libero_10"
TRIALS = 30    # the formal A' quota; the runner freezes --trials at this value
RUN = "run0123456789"
RUN_P0 = "runphase0abcd"
FULL, WARM, MISS = "FULL_HIT", "WARM_START", "MISS"


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_json(p: pathlib.Path, obj) -> pathlib.Path:
    p.write_text(json.dumps(obj, indent=1, sort_keys=True))
    return p


# ------------------------------------------------------------------
# synthetic Rev 1 world
# ------------------------------------------------------------------

def _officials():
    # LIBERO holds 50 official inits per task (0..49); A' uses the first 30, fit/cal the rest
    return {t: list(range(TRIALS)) for t in range(10)}


def _split_manifest(path: pathlib.Path, pool_dir: pathlib.Path):
    """Split manifest + a real materialised A' pool (torch-saved states), like Rev 1."""
    import torch
    pool_dir.mkdir(parents=True, exist_ok=True)
    assignment, digests = {}, {}
    for t in range(10):
        name = f"task_{t}"
        states = np.array([[t, float(i), 0.5] for i in _officials()[t]], dtype=np.float64)
        torch.save(states, pool_dir / f"{name}.init")
        assignment[str(t)] = {"task_name": name, "test": _officials()[t], "fit": [30, 31], "cal": [32, 33]}
        digests[name] = {"count": TRIALS, "indices": _officials()[t],
                         "sha256": hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest()}
    return _write_json(path, {"suite": SUITE, "quota": {"test": TRIALS, "fit": 1, "cal": 1, "dlib": 1},
                              "assignment": assignment, "seed": 20260827,
                              "pool_digests": {"test_aprime": digests}})


def _pool_attestation(split_path: pathlib.Path, pool_dir: pathlib.Path) -> dict:
    from exp.dispatch_surface.run_precheck import validate_aprime_pool
    return validate_aprime_pool(str(split_path), pool_dir, TRIALS)


def _table(path: pathlib.Path, rng):
    rows = []
    for t in range(10):
        for k in range(4):
            ep = f"ep_{t}_{k}"
            for _ in range(12):
                s = float(rng.uniform(0.90, 1.0))
                v = float(rng.uniform(0.0, 1.0))
                y10 = float(2.0 + 8.0 * (1.0 - s) * 10 + 2.0 * v + rng.normal(0, 0.3))
                y7 = float(y10 - 1.0 + rng.normal(0, 0.2))
                rows.append({"ref_mode": "fresh", "s": s, "v": v, "y_tau7": y7, "y_tau10": y10,
                             "episode_id": ep, "task_id": t, "init_idx": 30 + k,
                             "split": "fit" if k < 2 else "cal"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _tier_plan(arm: str, rng):
    """Per-arm verdict mix giving well separated costs."""
    p_full = {"dsp_t_fh70_ws10": 0.85, "dsp_t_fh50_ws20": 0.50, "dsp_t_fh30_ws20": 0.10,
              "dsp_sv_minus": 0.15, "dsp_sv": 0.75, "dsp_s0": 0.40,
              "dsp_sv_p85": 0.45, "dsp_s0_p80": 0.15, "dsp_s0_p95": 0.75}[arm]
    n = int(rng.integers(6, 12))
    out = []
    for _ in range(n):
        u = rng.random()
        out.append(FULL if u < p_full else (WARM if u < p_full + 0.15 else MISS))
    return out


def _rows_for(arm, task, subset, official, verdicts, success, run_id=RUN):
    uid = f"{arm}:eval:{task}:{subset}"
    rows = [{"yaml_id": arm, "task_uid": uid, "task_id": task, "orig_init_state_idx": official,
             "subset_init_state_idx": subset, "episode_id": task * TRIALS + subset, "step_idx": i * 5,
             "phase": "eval", "hit_type": h, "start_t": 0.3 if h == WARM else None,
             "attempt": 1, "accepted": True, "run_id": run_id, "success": success}
            for i, h in enumerate(verdicts)]
    rows.append({"_kind": "client_timing", "task_uid": uid, "yaml_id": arm, "task_id": task,
                 "subset_init_state_idx": subset, "infer_ms": 100.0 * len(verdicts), "infers": len(verdicts),
                 "steps": 5 * len(verdicts), "attempt": 1, "accepted": True, "run_id": run_id, "success": success})
    journal = {"yaml_id": arm, "task_uid": uid, "phase": "eval", "status": "done" if success else "failed",
               "success": success, "accepted": True, "attempt": 1, "run_id": run_id}
    return rows, journal


P_SUCCESS = {"dsp_t_fh30_ws20": 0.90, "dsp_t_fh50_ws20": 0.60, "dsp_t_fh70_ws10": 0.30,
             "dsp_sv_minus": 0.92, "dsp_sv_p85": 0.70, "dsp_sv": 0.45,
             "dsp_s0_p80": 0.85, "dsp_s0": 0.60, "dsp_s0_p95": 0.35, ANCHOR_ARM: 0.9}


def _runs(dirpath: pathlib.Path, arms, rng, p_success=None):
    per_step, journal = [], []
    for arm in arms:
        ps = (p_success or P_SUCCESS).get(arm, 0.6)
        for t, offs in _officials().items():
            for subset, official in enumerate(offs):
                rows, j = _rows_for(arm, t, subset, official, _tier_plan(arm, rng), bool(rng.random() < ps))
                per_step += rows
                journal.append(j)
    (dirpath / "per_step.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_step))
    (dirpath / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in journal))
    return dirpath / "journal.jsonl", dirpath / "per_step.jsonl"


def _artifact(path, *, k, uses_v, w, edges, q, delta, contract, meta):
    full, warm = export_boundaries(q, edges[0], delta)
    art = SurfaceArtifact(
        schema_version=SURFACE_ARTIFACT_SCHEMA_VERSION, k=k, h_exec=5, w=w,
        active_mask=np.ones_like(w, dtype=bool), start_t_ws=0.3, delta=float(delta),
        quantile_alpha=0.05, certification_mode=CERTIFICATION_EMPIRICAL, uses_disagreement=uses_v,
        v_bin_edges=np.asarray(edges[1], dtype=np.float64), s_min_full=full, s_min_warm=warm,
        conformal_c=0.0, n_calibration_episodes=0, retrieval_contract=dict(contract), meta=dict(meta),
    )
    save_surface_artifact(art, str(path))
    return path


def build_world(tmp_path: pathlib.Path):
    """Return a dict with every path of the synthetic Rev 1 world."""
    rng = np.random.default_rng(20260829)
    root = tmp_path / "rev1"
    root.mkdir()
    shared = root / "dsp_shared"
    shared.mkdir()
    pool_dir = root / "test_aprime"
    split = _split_manifest(root / "split_manifest.json", pool_dir)
    pool = _pool_attestation(split, pool_dir)
    table = _table(root / "table.jsonl", rng)
    lib = root / "lib.pkl"
    lib.write_bytes(b"library-bytes")
    noise = root / "noise.npz"
    noise.write_bytes(b"noise-bytes")
    weights = root / "weights.npz"
    np.savez(weights, w=np.ones(4, dtype=np.float32), active_mask=np.ones(4, dtype=bool))
    cache_yaml = root / "cache.yaml"
    cache_yaml.write_text("cache: true\n")
    cohort = _write_json(root / "cohort.json", {"n": 40})
    rebuild = _write_json(root / "rebuild_record.json", {
        "library_sha256": _sha(lib), "noise_sidecar_sha256": _sha(noise), "split_manifest_sha256": _sha(split),
        "entry_count": 1, "checkpoint_dir": "/ckpt", "config_name": "pi05_libero", "dlib_content_digests": {},
    })
    # The yaml-side half of the retrieval contract must be the REAL digests of
    # the template's retrieval configuration, or load_cache_config refuses the
    # artifact (the certificate would be for another retrieval space).
    from exp.dispatch_surface.emit_precheck_yamls import LAYER_PRIMARY, _emit
    from openpi.cache.config import compute_surface_retrieval_contract, load_cache_config
    probe = root / "probe.yaml"
    _emit(yaml.safe_load(TEMPLATE.read_text()), probe,
          {"type": "threshold", "threshold": 0.99, "warm_tiers": [{"threshold": 0.98, "start_t": 0.3}]},
          str(lib), 0.9928, LAYER_PRIMARY)
    contract = compute_surface_retrieval_contract(load_cache_config(str(probe)))
    contract.update({"top_k": 5, "library_sha256": _sha(lib), "library_entry_count": 1, "action_dim": 4,
                     "num_steps": 10, "h_exec": 5, "policy_fingerprint": "fp-synthetic"})
    # D0: real minimal record, rebound by digest only
    d0 = json.loads((HERE / "fixtures/d0_libero_10.min.json").read_text())
    for name, p in (("table", table), ("library_pkl", lib), ("noise_sidecar", noise),
                    ("cache_yaml", cache_yaml), ("weights_npz", weights)):
        d0["inputs"]["files"][name]["sha256"] = _sha(p)
    d0["inputs"]["policy"]["policy_fingerprint"] = contract["policy_fingerprint"]
    body = {k: v for k, v in d0["inputs"].items() if k != "rollup_sha256"}
    d0["inputs"]["rollup_sha256"] = _canonical_digest(body)
    d0["suite"] = SUITE
    d0_path = _write_json(root / "d0_record.json", d0)

    tbl = load_table(str(table))
    dev_mask = np.ones(len(tbl.s), dtype=bool)
    membership = sorted({(str(e), int(t), int(i)) for e, t, i in zip(tbl.episode, tbl.task, tbl.init_idx)})
    membership = [list(m) for m in membership]
    y10 = tbl.y10[dev_mask]
    grid = np.unique(np.percentile(y10, np.arange(10, 100, 10)))
    delta_star, minus = float(grid[-1]), float(grid[-2])
    w = np.ones(4, dtype=np.float32)
    fits, artifacts, records = {}, {}, {}
    for tag, s_only in (("sv", False), ("s0", True)):
        ff = final_fit(tbl, dev_mask, alpha=0.05, ladder=GRID_LADDER_S_ONLY if s_only else GRID_LADDER_SV)
        assert ff is not None
        fits[tag] = ff
        digests = {"s_edges": _digest_obj(np.asarray(ff.s_edges).tolist()),
                   "v_edges": _digest_obj(np.asarray(ff.v_edges).tolist()),
                   "q_deploy": _digest_obj(np.asarray(ff.q_hat).tolist()), "n_dev_rows": int(dev_mask.sum())}
        c = dict(contract, top_k=1 if s_only else 5)
        rec = {"s_only": s_only, "quantile_alpha": 0.05, "certification_mode": CERTIFICATION_EMPIRICAL,
               "d0_record_sha256": _sha(d0_path), "d0_suite": SUITE, "cohort_manifest": str(cohort),
               "input_digests": {"table": _sha(table), "cohort_manifest": _sha(cohort), "weights_npz": _sha(weights),
                                 "rebuild_record": _sha(rebuild), "split_manifest": _sha(split),
                                 "cache_yaml": _sha(cache_yaml), "d0_record": _sha(d0_path)},
               "d0_binding": {"record_sha256": _sha(d0_path), "suite": SUITE},
               "n_dev_episodes": 40, "delta_star": delta_star, "delta_grid": [float(x) for x in grid],
               "delta_neighbours": {"minus": minus, "plus": None}, "delta_selection_reason": "qualified",
               "dev_membership": membership, "dev_membership_sha256": _digest_obj(membership),
               "fold_map": [[m[0], i % 5] for i, m in enumerate(membership)],
               "final_fit_digests": digests, "n_calibration_episodes": 0, "conformal_c": 0.0}
        rec["fold_map_sha256"] = _digest_obj(rec["fold_map"])
        meta = {"ref_mode": "fresh", "delta_name": "primary", "input_digests": rec["input_digests"],
                "d0_binding": rec["d0_binding"], "dev_membership_sha256": rec["dev_membership_sha256"],
                "fold_map_sha256": rec["fold_map_sha256"], "final_fit_digests": digests}
        names = {"primary": delta_star} if s_only else {"primary": delta_star, "minus": minus}
        rec["artifacts"] = {}
        for name, delta in names.items():
            p = shared / f"surface_{'s_only' if s_only else 'sv'}_{name}.npz"
            _artifact(p, k=1 if s_only else 5, uses_v=not s_only, w=w, edges=(ff.s_edges, ff.v_edges),
                      q=ff.q_hat, delta=delta, contract=c, meta=dict(meta, delta_name=name))
            rec["artifacts"][name] = str(p)
            artifacts[f"dsp_{'s0' if s_only else 'sv'}" + ("" if name == "primary" else "_minus")] = p
        records[tag] = _write_json(shared / ("fit_record_s_only.json" if s_only else "fit_record.json"), rec)

    template = yaml.safe_load(TEMPLATE.read_text())
    cfg = root / "config"
    cfg.mkdir()
    arms = {}
    theta = 0.9928
    for arm, judge in (("dsp_t_fh30_ws20", {"type": "threshold", "threshold": 0.998, "warm_tiers": [{"threshold": 0.9976, "start_t": 0.3}]}),
                       ("dsp_t_fh50_ws20", {"type": "threshold", "threshold": 0.997, "warm_tiers": [{"threshold": 0.996, "start_t": 0.3}]}),
                       ("dsp_t_fh70_ws10", {"type": "threshold", "threshold": 0.996, "warm_tiers": [{"threshold": 0.995, "start_t": 0.3}]}),
                       ("dsp_s0", {"type": "dispatch_surface", "surface_artifact_path": str(artifacts["dsp_s0"])}),
                       ("dsp_sv", {"type": "dispatch_surface", "surface_artifact_path": str(artifacts["dsp_sv"])}),
                       ("dsp_sv_minus", {"type": "dispatch_surface", "surface_artifact_path": str(artifacts["dsp_sv_minus"])})):
        p = cfg / f"{arm}.yaml"
        _emit(template, p, judge, str(lib), theta, LAYER_PRIMARY)
        arms[arm] = str(p)
    matrix = {"protocol": "dispatch_surface_rev1", "layer": "primary", "suite": SUITE, "arms": arms,
              "gate_type": "always_search", "gate_theta": theta, "gate_theta_top_fraction": 0.85,
              "gate_params": {"L": 6, "j": 3, "probe_interval": 3}, "gate_theta_scores_n": 480,
              "arm_yaml_sha256": {a: _sha(pathlib.Path(p)) for a, p in arms.items()},
              "artifact_paths": {a: str(p) for a, p in artifacts.items()},
              "artifact_sha256": {a: _sha(p) for a, p in artifacts.items()},
              "certification_mode": CERTIFICATION_EMPIRICAL,
              "fit_record_paths": {"sv": str(records["sv"]), "s0": str(records["s0"])},
              "fit_record_sha256": {"sv": _sha(records["sv"]), "s0": _sha(records["s0"])},
              "core_arms": sorted(["dsp_t_fh30_ws20", "dsp_t_fh50_ws20", "dsp_t_fh70_ws10", "dsp_s0", "dsp_sv"]),
              "descriptive_arms": ["dsp_sv_minus"], "library_pkl": str(lib), "library_sha256": _sha(lib),
              "template": str(TEMPLATE)}
    matrix_path = _write_json(cfg / "arm_matrix_primary.json", matrix)
    out = root / "precheck"
    out.mkdir()
    journal, per_step = _runs(out, list(arms), rng)
    ledger = _write_json(out / "per_step.jsonl.launch.json", {"schema_version": 2, "launches": [{
        "protocol": "dispatch_surface_rev1", "layer": "primary", "suite": SUITE, "run_id": RUN,
        "executed_arms": sorted(arms), "core_arms": matrix["core_arms"], "descriptive_arms": ["dsp_sv_minus"],
        "trials_per_task": TRIALS, "replan_steps": 5, "env_seed": 7, "policy_fingerprint": contract["policy_fingerprint"],
        "library_sha256": _sha(lib), "aprime_content_sha256": pool["rollup_sha256"], "split_manifest_sha256": _sha(split),
        "arm_matrix_sha256": _sha(matrix_path), "frozen_yaml_sha256": matrix["arm_yaml_sha256"],
        "artifact_sha256": matrix["artifact_sha256"], "fit_record_sha256": matrix["fit_record_sha256"],
        "executed_yaml_sha256": matrix["arm_yaml_sha256"],
        "contract_binding": {"h_exec": 5, "policy_fingerprint": contract["policy_fingerprint"], "servers": {}},
        "pool": pool}]})
    verdict = _write_json(out / "verdict.json", {"verdict": "line_demoted", "suite": SUITE, "discipline": {
        "suite": SUITE, "protocol": "dispatch_surface_rev1", "layer": "primary", "delta_star": delta_star,
        "policy_fingerprint": contract["policy_fingerprint"], "library_sha256": _sha(lib),
        "aprime_content_sha256": pool["rollup_sha256"], "arm_matrix_sha256": _sha(matrix_path),
        "split_manifest_sha256": _sha(split), "arm_yaml_sha256": matrix["arm_yaml_sha256"],
        "artifact_sha256": matrix["artifact_sha256"], "fit_record_sha256": matrix["fit_record_sha256"],
        "cost_inputs": {"per_step_sha256": _sha(per_step), "unit_cost_ms": ac.unit_cost_table()}}})
    return types.SimpleNamespace(
        root=root, split=split, pool_dir=pool_dir, pool=pool, table=table, lib=lib, weights=weights,
        cache_yaml=cache_yaml, rebuild=rebuild, d0=d0_path, records=records, artifacts=artifacts,
        matrix=matrix_path, journal=journal, per_step=per_step, ledger=ledger, verdict=verdict,
        contract=contract, rng=rng, theta=theta,
    )


def build_package(world, tmp_path):
    sources = {"matrix": world.matrix, "fit.sv": world.records["sv"], "fit.s0": world.records["s0"],
               "artifact.dsp_sv": world.artifacts["dsp_sv"], "artifact.dsp_s0": world.artifacts["dsp_s0"],
               "artifact.dsp_sv_minus": world.artifacts["dsp_sv_minus"], "d0": world.d0, "rebuild": world.rebuild,
               "split_manifest": world.split, "ledger": world.ledger, "verdict": world.verdict,
               "journal": world.journal, "per_step": world.per_step}
    out = tmp_path / "package"
    manifest = archive.build_package(SUITE, sources, out)
    return out / pkgmod.MANIFEST_NAME, manifest


def run_exporter(manifest_path, world, out_dir, role, quantiles):
    args = types.SimpleNamespace(rev1_package_manifest=str(manifest_path), source_role=role,
                                 table=str(world.table), quantiles=quantiles, out_dir=str(out_dir))
    return exporter.export(args)


def build_phase0(world, manifest_path, tmp_path):
    ex_sv = run_exporter(manifest_path, world, tmp_path / "ex_sv", "artifact.dsp_sv", "0.85")
    ex_s0 = run_exporter(manifest_path, world, tmp_path / "ex_s0", "artifact.dsp_s0", "0.80,0.95")
    out_dir = tmp_path / "phase0_cfg"
    args = types.SimpleNamespace(
        suite=SUITE, export_records=f"{tmp_path / 'ex_sv' / 'export_record.json'},{tmp_path / 'ex_s0' / 'export_record.json'}",
        rev1_package_manifest=str(manifest_path), template=str(TEMPLATE), library_pkl=str(world.lib),
        out_dir=str(out_dir), layer=LAYER_EXPLORATORY)
    emit_exploratory(args)
    matrix_path = out_dir / f"arm_matrix_{LAYER_EXPLORATORY}.json"
    matrix = json.loads(matrix_path.read_text())
    run_dir = tmp_path / "phase0_run"
    run_dir.mkdir()
    rng = np.random.default_rng(7)
    per_step, journal = [], []
    for arm in matrix["arms"]:
        for t, offs in _officials().items():
            for subset, official in enumerate(offs):
                verdicts = [MISS] * int(rng.integers(6, 12)) if arm == ANCHOR_ARM else _tier_plan(arm, rng)
                rows, j = _rows_for(arm, t, subset, official, verdicts, bool(rng.random() < P_SUCCESS[arm]),
                                    run_id=RUN_P0)
                per_step += rows
                journal.append(j)
    (run_dir / "per_step.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_step))
    (run_dir / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in journal))
    entry = {"protocol": matrix["protocol"], "layer": LAYER_EXPLORATORY, "suite": SUITE, "run_id": RUN_P0,
             "executed_arms": sorted(matrix["arms"]), "core_arms": [], "descriptive_arms": sorted(matrix["arms"]),
             "trials_per_task": TRIALS, "replan_steps": 5, "env_seed": 7, "policy_fingerprint": world.contract["policy_fingerprint"],
             "library_sha256": _sha(world.lib), "aprime_content_sha256": world.pool["rollup_sha256"],
             "split_manifest_sha256": _sha(world.split), "arm_matrix_sha256": _sha(matrix_path),
             "contract_binding": {"h_exec": 5, "policy_fingerprint": world.contract["policy_fingerprint"], "servers": {}},
             "pool": world.pool,
             "frozen_yaml_sha256": matrix["arm_yaml_sha256"], "artifact_sha256": matrix["artifact_sha256"],
             "fit_record_sha256": None, "executed_yaml_sha256": matrix["arm_yaml_sha256"],
             "posthoc_exploratory": True, "roster_spec_sha256": matrix["roster_spec_sha256"],
             "rev1_package_manifest_sha256": matrix["rev1_package_manifest_sha256"],
             "export_record_sha256": matrix["export_record_sha256"], "cost_model_digest": matrix["cost_model_digest"],
             "contract_anchor_arm": matrix["contract_anchor_arm"]}
    ledger = _write_json(run_dir / "per_step.jsonl.launch.json", {"schema_version": 2, "launches": [entry]})
    return types.SimpleNamespace(matrix_path=matrix_path, matrix=matrix, journal=run_dir / "journal.jsonl",
                                 per_step=run_dir / "per_step.jsonl", ledger=ledger, entry=entry,
                                 ex_sv=ex_sv, ex_s0=ex_s0)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("phase0")
    world = build_world(tmp)
    manifest_path, manifest = build_package(world, tmp)
    p0 = build_phase0(world, manifest_path, tmp)
    return types.SimpleNamespace(tmp=tmp, world=world, manifest_path=manifest_path, manifest=manifest, p0=p0)


# ------------------------------------------------------------------
# 1. cost authority (B2 / B4)
# ------------------------------------------------------------------

def test_cost_authority_full_precision():
    assert ac.unit_cost("MISS", None) == 67.518595
    assert ac.cost_matches(67.518595, ac.unit_cost("MISS", None))
    assert not ac.cost_matches(67.5186, ac.unit_cost("MISS", None))     # the v2 literal must fail
    with pytest.raises(SystemExit):
        ac.assert_unit_costs_match({**ac.unit_cost_table(), "MISS": 67.5186}, what="x")


def test_analyzer_reexports_the_cost_authority():
    from exp.dispatch_surface.analysis import analyze_precheck as az
    assert az.unit_cost is ac.unit_cost and az.STAGE3_MS == ac.STAGE3_MS


def test_real_rev1_verdicts_carry_the_authority_costs():
    for suite in ("libero_10", "libero_spatial"):
        p = REPO / f"exp/dispatch_surface/data/aprime_rev1/{suite}_primary/verdict.json"
        if not p.is_file():
            pytest.skip("archived Rev 1 verdicts not present on this machine")
        v = json.loads(p.read_text())
        ac.assert_unit_costs_match(v["discipline"]["cost_inputs"]["unit_cost_ms"], what=suite)


# ------------------------------------------------------------------
# 2. loaders: outcome-blind cost-only path (B4 / R3-B2)
# ------------------------------------------------------------------

def _costonly_funcs():
    tree = ast.parse((REPO / "exp/dispatch_surface/analysis/precheck_io.py").read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.endswith("_costonly")}


def _string_constants(node):
    return {c.value for c in ast.walk(node) if isinstance(c, ast.Constant) and isinstance(c.value, str)}


def test_costonly_functions_never_name_success_or_status():
    funcs = _costonly_funcs()
    assert set(funcs) == {"load_accepted_cells_costonly", "load_cost_cells_costonly"}
    for fn in funcs.values():
        assert not ({"success", "status"} & _string_constants(fn)), fn.name
    tree = ast.parse((REPO / "exp/dispatch_surface/analysis/cost_map.py").read_text())
    assert not ({"success", "status"} & _string_constants(tree))


def test_source_lock_cost_only_import_graph():
    for mod in ("cost_map", "precheck_io"):
        tree = ast.parse((REPO / f"exp/dispatch_surface/analysis/{mod}.py").read_text())
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                names.add(n.module)
            if isinstance(n, ast.Import):
                names.update(a.name for a in n.names)
        for banned in ("frontier_hull", "phase0_outcome_design", "analyze_precheck"):
            assert not any(banned in nm for nm in names), (mod, banned)


def test_costmap_bytes_identical_when_outcomes_are_deleted_or_replaced(chain, tmp_path):
    def run(journal, per_step):
        rev1 = cm._load_rev1_source(str(chain.manifest_path), TRIALS)
        p0 = cm._load_phase0_source(str(chain.p0.matrix_path), str(chain.p0.ledger), str(chain.world.split),
                                    str(journal), str(per_step), TRIALS)
        out = cm.build_cost_map(rev1, p0, reps=150)
        # the frozen INPUT digests legitimately change when bytes change; every
        # mechanical output must not
        out.pop("phase0_discipline")
        out.pop("input_sha256")
        return json.dumps(out, sort_keys=True)
    base = run(chain.p0.journal, chain.p0.per_step)
    rng = np.random.default_rng(1)
    for mode in ("delete", "replace"):
        j2, p2 = tmp_path / f"j_{mode}.jsonl", tmp_path / f"p_{mode}.jsonl"
        for src, dst in ((chain.p0.journal, j2), (chain.p0.per_step, p2)):
            rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
            for r in rows:
                for key in ("success", "status"):
                    if key in r:
                        if mode == "delete":
                            del r[key]
                        else:
                            r[key] = bool(rng.random() < 0.5) if key == "success" else rng.choice(["done", "failed", "weird"])
            dst.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert run(j2, p2) == base, mode


def test_outcome_loader_still_validates_status(chain, tmp_path):
    rows = [json.loads(line) for line in chain.p0.journal.read_text().splitlines()]
    rows[0]["status"] = "weird"
    p = tmp_path / "bad.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    grid = {(t, i) for t in range(10) for i in range(TRIALS)}
    with pytest.raises(SystemExit):
        precheck_io.load_accepted_episodes(str(p), [rows[0]["yaml_id"]], grid)
    precheck_io.load_accepted_cells_costonly(str(p), [rows[0]["yaml_id"]], grid)   # blind to it


def test_real_journals_reproduce_archived_costs_with_moved_loaders():
    from exp.dispatch_surface.analysis.analyze_precheck import EXPECTED_TRIALS, arm_cost, official_by_task
    for suite, sm in (("libero_10", "exp/dispatch_surface/data/libero_10/init_pools/split_manifest.json"),
                      ("libero_spatial", "exp/dispatch_surface/data/init_pools/split_manifest.json")):
        d = REPO / f"exp/dispatch_surface/data/aprime_rev1/{suite}_primary"
        if not (d / "verdict.json").is_file():
            pytest.skip("archived Rev 1 data not present")
        v = json.loads((d / "verdict.json").read_text())
        core = sorted(v["point_estimates"])
        officials = official_by_task(str(REPO / sm), EXPECTED_TRIALS)
        grid = {(t, i) for t in officials for i in range(len(officials[t]))}
        acc = precheck_io.load_accepted_episodes(str(d / "journal.jsonl"), core, grid)
        cells, summary = precheck_io.load_analytic_cost(str(d / "per_step.jsonl"), core, acc, officials)
        for a in core:
            assert arm_cost(cells[a], sorted(grid)) == pytest.approx(v["point_estimates"][a]["cost_ms_per_decision"], abs=1e-9)
        assert summary["verdict_counts"] == v["discipline"]["cost_inputs"]["verdict_counts"]
        assert summary["decisions"] == v["discipline"]["cost_inputs"]["decisions"]
        acc2 = precheck_io.load_accepted_cells_costonly(str(d / "journal.jsonl"), core, grid)
        cells2, _ = precheck_io.load_cost_cells_costonly(str(d / "per_step.jsonl"), core, acc2, officials)
        assert cells2 == cells


# ------------------------------------------------------------------
# 3. package / exporter (B2, B3, R3-B1, R4-B1)
# ------------------------------------------------------------------

def test_package_manifest_maps_roles_and_verifies(chain):
    manifest = pkgmod.verify_package(chain.manifest_path)
    assert set(pkgmod.REQUIRED_ROLES) <= set(manifest["members"])
    for role, entry in manifest["members"].items():
        assert not entry["member"].startswith("/")


def test_package_refuses_matrix_declared_sha_drift(chain, tmp_path):
    pkg = tmp_path / "pkg2"
    import shutil
    shutil.copytree(chain.manifest_path.parent, pkg)
    m = json.loads((pkg / pkgmod.MANIFEST_NAME).read_text())
    m["members"]["artifact.dsp_sv"]["sha256"] = "0" * 64
    (pkg / pkgmod.MANIFEST_NAME).write_text(json.dumps(m))
    with pytest.raises(SystemExit):
        pkgmod.verify_package(pkg / pkgmod.MANIFEST_NAME)


def test_exporter_reproduces_grid_quantiles_and_binds_everything(chain):
    rec = chain.p0.ex_sv
    fit = json.loads(chain.world.records["sv"].read_text())
    y10 = load_table(str(chain.world.table)).y10
    assert abs(exporter.delta_at_quantile(y10, 0.8) - fit["delta_grid"][-2]) < 1e-9
    assert abs(exporter.delta_at_quantile(y10, 0.9) - fit["delta_grid"][-1]) < 1e-9
    assert rec["source_artifact_sha256"] == pkgmod.member_sha(chain.manifest, "artifact.dsp_sv")
    assert rec["final_fit_digests"] == fit["final_fit_digests"]
    assert "p85" in rec["artifacts"]
    art = load_surface_artifact(rec["artifacts"]["p85"]["path"])
    src = load_surface_artifact(str(chain.world.artifacts["dsp_sv"]))
    assert art.meta["posthoc_exploratory"] is True
    for f in ("k", "h_exec", "quantile_alpha", "certification_mode", "uses_disagreement"):
        assert getattr(art, f) == getattr(src, f)
    assert art.retrieval_contract == src.retrieval_contract
    assert np.array_equal(art.w, src.w) and np.array_equal(art.v_bin_edges, src.v_bin_edges)
    assert art.delta != src.delta
    assert not any("pending" in str(v).lower() for v in art.meta.values())


def test_exporter_import_graph_has_no_live_d0_validator():
    tree = ast.parse((REPO / "exp/dispatch_surface/export_exploratory_surface.py").read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and "d0_check" in n.module:
            assert "validate_input_attestation" not in {a.name for a in n.names}


def test_d0_binding_uses_real_key_set_and_refuses_tampering(chain, tmp_path):
    manifest, pkg, _ = pkgmod.load_manifest(chain.manifest_path)
    d0 = pkgmod.load_json_member(manifest, pkg, "d0")
    fit = json.loads(chain.world.records["sv"].read_text())
    src = load_surface_artifact(str(chain.world.artifacts["dsp_sv"]))
    assert set(d0["inputs"]["files"]) == exporter.D0_FILE_KEYS
    exporter.validate_export_d0_binding(d0, fit, manifest, pkg, _sha(chain.world.table), src)
    for mutate in (lambda d: d["inputs"]["files"]["weights_npz"].__setitem__("sha256", "1" * 64),
                   lambda d: d["inputs"]["policy"].__setitem__("policy_fingerprint", "other"),
                   lambda d: d["inputs"]["files"].__setitem__("rebuild_record", {"sha256": "x"}),
                   lambda d: d.__setitem__("D0", "FAIL")):
        bad = copy.deepcopy(d0)
        mutate(bad)
        with pytest.raises(SystemExit):
            exporter.validate_export_d0_binding(bad, fit, manifest, pkg, _sha(chain.world.table), src)


def test_exporter_refuses_nonempty_out_dir_and_wrong_role(chain, tmp_path):
    with pytest.raises(SystemExit):
        run_exporter(chain.manifest_path, chain.world, chain.tmp / "ex_sv", "artifact.dsp_sv", "0.85")
    with pytest.raises(SystemExit):
        run_exporter(chain.manifest_path, chain.world, tmp_path / "x", "artifact.dsp_sv_minus", "0.85")


# ------------------------------------------------------------------
# 4. emitter / runner (B1, B6)
# ------------------------------------------------------------------

def test_emitted_matrix_is_the_frozen_roster_with_anchor(chain):
    m = chain.p0.matrix
    assert set(m["arms"]) == set(ROSTERS[SUITE])
    assert m["core_arms"] == [] and sorted(m["descriptive_arms"]) == sorted(m["arms"])
    assert m["contract_anchor_arm"] == CONTRACT_ANCHOR_ARM[SUITE]
    assert m["roster_spec_sha256"] == roster_spec_digest(SUITE)
    assert m["cost_model_digest"] == ac.cost_model_digest()
    anchor = yaml.safe_load(pathlib.Path(m["arms"][ANCHOR_ARM]).read_text())["checkpoints"]["cp1"]
    assert anchor["judge"] == {"type": "threshold", "threshold": 1.5}
    assert anchor["gate"]["type"] == "always_search"


def test_runner_validates_exploratory_matrix_and_anchor(chain):
    validate_exploratory_matrix_artifacts(chain.p0.matrix)
    validate_precheck_arms(dict(chain.p0.matrix["arms"]), LAYER_EXPLORATORY, frozenset({ANCHOR_ARM}))
    with pytest.raises(SystemExit):     # the Rev 1 rule refuses the anchor shape
        validate_precheck_arms({ANCHOR_ARM: chain.p0.matrix["arms"][ANCHOR_ARM]}, "primary")


def test_runner_refuses_broken_chain_and_wrong_contract_arm(chain):
    m = copy.deepcopy(chain.p0.matrix)
    m["contract_anchor_arm"] = "dsp_s0_p80"
    with pytest.raises(SystemExit):
        validate_exploratory_matrix_artifacts(m)
    m = copy.deepcopy(chain.p0.matrix)
    m["export_record_sha256"][0] = "0" * 64
    with pytest.raises(SystemExit):
        validate_exploratory_matrix_artifacts(m)
    m = copy.deepcopy(chain.p0.matrix)
    m["families"]["dsp_s0_p80"] = "sv"
    with pytest.raises(SystemExit):
        validate_exploratory_matrix_artifacts(m)


def test_analyzer_refuses_exploratory_matrix(chain):
    from exp.dispatch_surface.analysis.analyze_precheck import check_discipline
    with pytest.raises(SystemExit):
        check_discipline(types.SimpleNamespace(fit_record="x"), chain.p0.matrix)


def _runner_argv(chain, matrix_path, layer, out_dir, extra=()):
    return ["run_precheck", "--arm-matrix", str(matrix_path), "--layer", layer, "--task-suite", SUITE,
            "--servers", "localhost:1", "--trials", str(TRIALS), "--replan-steps", "5", "--seed", "7",
            "--journal", str(out_dir / "journal.jsonl"), "--per-step-out", str(out_dir / "per_step.jsonl"),
            "--split-manifest", str(chain.world.split), "--pool-dir", str(chain.world.pool_dir),
            "--workers", "1", "--dry-validate", *extra]


class _StubPolicy:
    fingerprint = None

    def __init__(self, host, port):
        pass

    def get_server_metadata(self):
        return {"policy_fingerprint": _StubPolicy.fingerprint, "monitor_level": "test"}


def _run_main(monkeypatch, argv):
    import sys
    from exp.dispatch_surface import run_precheck as rp
    import openpi_client.websocket_client_policy as wcp
    monkeypatch.setattr(wcp, "WebsocketClientPolicy", _StubPolicy)
    monkeypatch.setattr(sys, "argv", argv)
    rp.main()


def test_runner_main_dry_validates_the_exploratory_chain(chain, tmp_path, monkeypatch, capsys):
    _StubPolicy.fingerprint = chain.world.contract["policy_fingerprint"]
    out = tmp_path / "dry"
    out.mkdir()
    _run_main(monkeypatch, _runner_argv(chain, chain.p0.matrix_path, LAYER_EXPLORATORY, out))
    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_validate"] is True and summary["layer"] == LAYER_EXPLORATORY
    assert summary["would_append_launch"]["contract_anchor_arm"] == CONTRACT_ANCHOR_ARM[SUITE]
    assert summary["would_append_launch"]["posthoc_exploratory"] is True
    assert not (out / "per_step.jsonl.launch.json").exists()      # dry run appends nothing
    # subset launch still validates against the same frozen matrix
    _run_main(monkeypatch, _runner_argv(chain, chain.p0.matrix_path, LAYER_EXPLORATORY, out,
                                        extra=("--arms", ANCHOR_ARM)))
    assert json.loads(capsys.readouterr().out)["arms"] == [ANCHOR_ARM]


def test_runner_main_refuses_wrong_layer_flag_and_server(chain, tmp_path, monkeypatch, capsys):
    _StubPolicy.fingerprint = chain.world.contract["policy_fingerprint"]
    out = tmp_path / "dry2"
    out.mkdir()
    with pytest.raises(SystemExit):     # --layer disagrees with the matrix
        _run_main(monkeypatch, _runner_argv(chain, chain.p0.matrix_path, "primary", out))
    flagged = tmp_path / "flagged_primary.json"
    m = json.loads(chain.world.matrix.read_text())
    m["posthoc_exploratory"] = True
    flagged.write_text(json.dumps(m))
    with pytest.raises(SystemExit):     # a Rev 1 matrix carrying the flag is refused by main()
        _run_main(monkeypatch, _runner_argv(chain, flagged, "primary", out))
    _StubPolicy.fingerprint = "someone-else"
    with pytest.raises(SystemExit):     # server attests another policy
        _run_main(monkeypatch, _runner_argv(chain, chain.p0.matrix_path, LAYER_EXPLORATORY, out))


def test_ledger_freezes_exploratory_keys(chain):
    entry = copy.deepcopy(chain.p0.entry)
    ledger = {"schema_version": 2, "launches": [entry]}
    new = copy.deepcopy(entry)
    new["run_id"] = "runanother0001"
    validate_existing_launch_ledger(ledger, new)
    for key in EXPLORATORY_FROZEN_LAUNCH_KEYS:
        drift = copy.deepcopy(new)
        drift[key] = "drift" if not isinstance(drift[key], list) else ["drift"]
        with pytest.raises(SystemExit):
            validate_existing_launch_ledger(ledger, drift)
    assert set(EXPLORATORY_FROZEN_LAUNCH_KEYS).isdisjoint(FROZEN_LAUNCH_KEYS)


# ------------------------------------------------------------------
# 5. discipline / summary (B4, B8)
# ------------------------------------------------------------------

def test_phase0_discipline_and_summary(chain):
    ctx = phase0_discipline.validate(str(chain.p0.matrix_path), str(chain.p0.ledger), str(chain.world.split),
                                     trials=TRIALS)
    assert ctx["executed_arms_by_run"] == {RUN_P0: sorted(chain.p0.matrix["arms"])} and ctx["roster_complete"]
    assert ctx["posthoc_exploratory"] and ctx["cost_model_digest"] == ac.cost_model_digest()
    args = types.SimpleNamespace(arm_matrix=str(chain.p0.matrix_path), launch_manifest=str(chain.p0.ledger),
                                 split_manifest=str(chain.world.split), journal=str(chain.p0.journal),
                                 per_step=str(chain.p0.per_step), trials=TRIALS, executed_only=False)
    out = phase0_summary.summarize(args)
    anchor = out["arms"][ANCHOR_ARM]
    assert anchor["a4"]["passed"] and anchor["a4"]["expected_cost_ms"] == 67.518595
    assert "sr_recorded_not_judged" in anchor
    for arm, e in out["arms"].items():
        if arm != ANCHOR_ARM:
            assert "sr" not in e and "success" not in json.dumps(e)


def test_anchor_gate_refuses_one_full_hit():
    cells = {(0, 0): (ac.unit_cost("MISS", None) * 2, 2), (0, 1): (ac.unit_cost("MISS", None) + ac.STAGE1_MS, 2)}
    g = phase0_summary.anchor_gate(cells, {"MISS": 3, "FULL_HIT": 1}, 2)
    assert not g["passed"]
    ok = phase0_summary.anchor_gate({(0, 0): (ac.unit_cost("MISS", None) * 3, 3)}, {"MISS": 3}, 1)
    assert ok["passed"]


# ------------------------------------------------------------------
# 6. cost map (B3, B6, B8)
# ------------------------------------------------------------------

def test_quantile_indices_are_frozen():
    rng = np.random.default_rng(0)
    L = rng.normal(size=10000)
    H = rng.normal(size=10000)
    assert np.quantile(L, 0.995, method="higher") == np.sort(L)[9950]
    assert np.quantile(H, 0.005, method="lower") == np.sort(H)[49]


def test_isotonic_is_decreasing_and_ties_by_delta():
    iso = cm.decreasing_isotonic([0.8, 0.85, 0.9], [50.0, 55.0, 20.0], [100, 100, 100])
    assert iso[0] >= iso[1] >= iso[2] and iso[0] == iso[1] == pytest.approx(52.5)
    with pytest.raises(ValueError):
        cm.decreasing_isotonic([0.8, 0.8], [1, 2], [1, 1])


def test_middle_pick_prefers_smaller_delta_on_ties():
    assert cm.pick_middle([(0.90, 40.0), (0.95, 40.0)], 40.0) == 0.90
    assert cm.pick_middle([(0.90, 45.0), (0.95, 41.0)], 40.0) == 0.95


def test_cost_map_is_mechanical_and_reproducible(chain):
    rev1 = cm._load_rev1_source(str(chain.manifest_path), TRIALS)
    p0 = cm._load_phase0_source(str(chain.p0.matrix_path), str(chain.p0.ledger), str(chain.world.split),
                                str(chain.p0.journal), str(chain.p0.per_step), TRIALS)
    a = cm.build_cost_map(rev1, p0, reps=200)
    b = cm.build_cost_map(rev1, p0, reps=200)
    assert a["bootstrap_index_sha256"] == b["bootstrap_index_sha256"] and a["interval"] == b["interval"]
    assert a["selected"]["threshold"] == ["dsp_t_fh70_ws10", "dsp_t_fh50_ws20", "dsp_t_fh30_ws20"]
    assert a["families"]["threshold"]["isotonic_cost"] is None
    assert set(a["selected"]["sv"]) == {"dsp_sv_minus", "dsp_sv_p85", "dsp_sv"}
    assert set(a["selected"]["s0"]) == {"dsp_s0_p80", "dsp_s0", "dsp_s0_p95"}
    assert a["a3_pass"], a["a3_problems"]
    assert a["interval"]["c_H"] - a["interval"]["c_L"] >= 4.0


def test_cost_map_a3_fails_closed_on_narrow_support():
    L = np.full(100, 40.0)
    H = np.full(100, 42.0)
    iv = cm.interval_from_endpoints(L, H)
    assert iv["c_H"] - iv["c_L"] < 4.0


# ------------------------------------------------------------------
# 7. hull / outcome design (B5, R3-B3)
# ------------------------------------------------------------------

def test_hull_frozen_dominance_drops_equal_sr_higher_cost():
    """G2R2-B3: the G1-frozen Pareto rule -- an equal-SR point at higher cost is
    dominated and must NOT extend the family's support."""
    h = fh.upper_concave_hull([(0, 0), (0.3, 1), (1, 1)])
    assert h == [(0.0, 0.0), (0.3, 1.0)] and fh.hull_support(h) == (0.0, 0.3)
    assert not fh.covers(h, 0, 1)
    with pytest.raises(ValueError):
        fh.auc_norm(h, 0, 1)


def test_hull_three_cases():
    on = fh.upper_concave_hull([(1, 0.5), (2, 0.8), (3, 0.9)])
    assert len(on) == 3
    dominated = fh.upper_concave_hull([(1, 0.5), (2, 0.55), (3, 0.9)])
    assert len(dominated) == 2 and (2, 0.55) not in dominated
    same_cost = fh.upper_concave_hull([(1, 0.5), (1, 0.7), (3, 0.9)])
    assert same_cost[0] == (1.0, 0.7)


def test_support_miss_is_minus_one_and_kept():
    v, miss = fh.auc_with_support([(1, 0.5), (2, 0.6)], [(3, 0.4), (4, 0.5)], 2.5, 3.5)
    assert miss and v == -1.0


def test_effect_is_plugin_not_bootstrap_mean_and_sd_feels_minus_one(chain):
    rev1 = cm._load_rev1_source(str(chain.manifest_path), TRIALS)
    p0 = cm._load_phase0_source(str(chain.p0.matrix_path), str(chain.p0.ledger), str(chain.world.split),
                                str(chain.p0.journal), str(chain.p0.per_step), TRIALS)
    cost_map = cm.build_cost_map(rev1, p0, reps=120)
    from exp.dispatch_surface.analysis.analyze_precheck import official_by_task
    officials = official_by_task(str(chain.world.split), TRIALS)
    grid = {(t, i) for t in officials for i in range(len(officials[t]))}
    arms = sorted({a for arms in cost_map["selected"].values() for a in arms})
    cells, sr = {}, {}
    for journal, per_step, sub in ((chain.world.journal, chain.world.per_step, [a for a in arms if a in rev1["cells"]]),
                                   (chain.p0.journal, chain.p0.per_step, [a for a in arms if a in p0["cells"]])):
        acc = precheck_io.load_accepted_episodes(str(journal), sub, grid)
        c, _ = precheck_io.load_analytic_cost(str(per_step), sub, acc, officials)
        cells.update(c)
        sr.update({a: {k: float(v["success"]) for k, v in acc[a].items()} for a in sub})
    from exp.dispatch_surface.cost_map_api import shared_index_from_map
    picks = shared_index_from_map(cost_map, grid)
    out = phase0_outcome_design.design(cost_map, cells, sr, picks, suite=SUITE)
    h1 = out["hypotheses"]["H1"]
    assert h1["effect_plugin"] is not None and h1["effect_plugin"] != h1["bootstrap_mean"]
    # push c_H upward until only SOME replicates lose support: the -1 scores are
    # kept, so sd30 must grow relative to the covered case
    partial = None
    for step in np.arange(0.2, 30.0, 0.2):
        trial = copy.deepcopy(cost_map)
        trial["interval"]["c_H"] = cost_map["interval"]["c_H"] + float(step)
        res = phase0_outcome_design.design(trial, cells, sr, picks, suite=SUITE)["hypotheses"]["H1"]
        if 0.05 < res["support_miss_rate"] < 0.95:
            partial = res
            break
    assert partial is not None, "could not construct a partial support-miss case"
    assert partial["sd30_including_support_miss"] > h1["sd30_including_support_miss"]
    if partial["support_miss_rate"] > 0.01:
        assert partial["verdict"] == "support_miss" and partial["power_n40"] is None
    # every replicate missing: scored -1 throughout, gate closed
    allmiss = copy.deepcopy(cost_map)
    allmiss["interval"]["c_H"] = cost_map["interval"]["c_H"] + 60.0
    res = phase0_outcome_design.design(allmiss, cells, sr, picks, suite=SUITE)["hypotheses"]["H1"]
    assert res["support_miss_rate"] == 1.0 and res["bootstrap_mean"] == -1.0 and res["verdict"] == "support_miss"


def test_power_formula():
    assert phase0_outcome_design.power_n40(0.075, 0.027) == pytest.approx(
        phase0_outcome_design.norm_cdf(0.075 / (0.027 * np.sqrt(30 / 40)) - 1.6448536269514722))
    assert phase0_outcome_design.power_n40(-0.01, 0.02) is None
    assert phase0_outcome_design.power_n40(0.05, 0.0) is None


# ------------------------------------------------------------------
# 8. migration: no /tmp path is ever opened (B3, R3-B1)
# ------------------------------------------------------------------

def test_migration_chain_works_when_historical_paths_are_gone(chain, tmp_path, monkeypatch):
    # every artifact/fit record inside the frozen matrix points at chain.world.root; make those unreadable
    import shutil
    moved = tmp_path / "moved"
    shutil.copytree(chain.manifest_path.parent, moved / "package")
    manifest_path = moved / "package" / pkgmod.MANIFEST_NAME
    # copy the table (an explicit input) and hide the original world
    table = moved / "table.jsonl"
    shutil.copyfile(chain.world.table, table)
    hidden = tmp_path / "hidden"
    shutil.move(str(chain.world.root), str(hidden))
    try:
        pkgmod.verify_package(manifest_path)
        world = types.SimpleNamespace(table=table)
        rec = run_exporter(manifest_path, world, moved / "ex", "artifact.dsp_sv", "0.85")
        assert rec["artifacts"]["p85"]["output_sha256"]
        assert not any(str(chain.world.root) in json.dumps(rec) for _ in [0])
    finally:
        shutil.move(str(hidden), str(chain.world.root))


# ------------------------------------------------------------------
# 9. task manifest / data authority (B7)
# ------------------------------------------------------------------

def test_task_manifests_match_split_assignments_and_known_atoms():
    for suite, sm in (("libero_10", "exp/dispatch_surface/data/libero_10/init_pools/split_manifest.json"),
                      ("libero_spatial", "exp/dispatch_surface/data/init_pools/split_manifest.json")):
        m = json.loads((REPO / f"exp/dispatch_surface/config/task_manifest_{suite}.json").read_text())
        assignment = json.loads((REPO / sm).read_text())["assignment"]
        for t in m["tasks"]:
            assert assignment[str(t["task_id"])]["task_name"] == t["task_name"]
        if suite == "libero_10":
            by = {t["task_id"]: t for t in m["tasks"]}
            assert "STUDY_SCENE1" in by[5]["task_name"] and by[5]["n_goal_atoms"] == 1
            assert "microwave" in by[9]["task_name"] and by[9]["n_goal_atoms"] == 2
            assert by[8]["n_goal_atoms"] == 3


def test_data_authority_new_kinds_validate():
    from exp.data_authority.registry import KNOWN_KINDS, load_record
    assert {"external_asset", "task_manifest"} <= set(KNOWN_KINDS)
    for name in ("dispatch_surface__libero_spatial__libero_assets_hf.json",
                 "dispatch_surface__libero_10__task_manifest.json"):
        load_record(REPO / "exp/data_authority/records" / name)


# ------------------------------------------------------------------
# 10. G2 R1 adversarial regressions
# ------------------------------------------------------------------

def _rewrite_rows(src, dst, mutate):
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    for r in rows:
        mutate(r)
    dst.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return dst


def test_b1_unregistered_run_id_is_refused_everywhere(chain, tmp_path):
    def foreign(r):
        # one anchor cell (seen by the summary) and one surface cell (seen by the cost map)
        if r.get("task_uid") in (f"{ANCHOR_ARM}:eval:0:0", "dsp_sv_p85:eval:0:0"):
            r["run_id"] = "runnotinledger"
    j = _rewrite_rows(chain.p0.journal, tmp_path / "j.jsonl", foreign)
    p = _rewrite_rows(chain.p0.per_step, tmp_path / "p.jsonl", foreign)
    args = types.SimpleNamespace(arm_matrix=str(chain.p0.matrix_path), launch_manifest=str(chain.p0.ledger),
                                 split_manifest=str(chain.world.split), journal=str(j), per_step=str(p),
                                 trials=TRIALS, executed_only=False)
    with pytest.raises(SystemExit, match="no launch in the ledger executed"):
        phase0_summary.summarize(args)
    with pytest.raises(SystemExit, match="no launch in the ledger executed"):
        cm._load_phase0_source(str(chain.p0.matrix_path), str(chain.p0.ledger), str(chain.world.split),
                               str(j), str(p), TRIALS)


def test_b1_registered_run_that_did_not_execute_the_arm_is_refused(chain, tmp_path):
    ledger = json.loads(chain.p0.ledger.read_text())
    ledger["launches"][0]["executed_arms"] = [a for a in ledger["launches"][0]["executed_arms"] if a != "dsp_s0_p80"]
    ledger["launches"][0]["executed_yaml_sha256"].pop("dsp_s0_p80")
    lp = tmp_path / "ledger.json"
    lp.write_text(json.dumps(ledger))
    ctx = phase0_discipline.validate(str(chain.p0.matrix_path), str(lp), str(chain.world.split), trials=TRIALS)
    assert not ctx["roster_complete"]
    with pytest.raises(SystemExit):
        cm._load_phase0_source(str(chain.p0.matrix_path), str(lp), str(chain.world.split),
                               str(chain.p0.journal), str(chain.p0.per_step), TRIALS)


def test_b1_ledger_duplicate_run_ids_and_pool_drift_refused(chain, tmp_path):
    ledger = json.loads(chain.p0.ledger.read_text())
    ledger["launches"].append(copy.deepcopy(ledger["launches"][0]))
    lp = tmp_path / "dup.json"
    lp.write_text(json.dumps(ledger))
    with pytest.raises(SystemExit, match="duplicated run ids"):
        phase0_discipline.validate(str(chain.p0.matrix_path), str(lp), str(chain.world.split), trials=TRIALS)
    ledger = json.loads(chain.p0.ledger.read_text())
    ledger["launches"][0]["pool"]["rollup_sha256"] = "0" * 64
    lp.write_text(json.dumps(ledger))
    with pytest.raises(SystemExit):
        phase0_discipline.validate(str(chain.p0.matrix_path), str(lp), str(chain.world.split), trials=TRIALS)


def _frozen_cost_map(chain, tmp_path, reps=cm.REPLICATES):
    """A cost map as the formal entry writes it. Default R is the frozen 10000;
    the small-R variant exists ONLY for the pure design() path."""
    rev1 = cm._load_rev1_source(str(chain.manifest_path), TRIALS)
    p0 = cm._load_phase0_source(str(chain.p0.matrix_path), str(chain.p0.ledger), str(chain.world.split),
                                str(chain.p0.journal), str(chain.p0.per_step), TRIALS)
    out = cm.build_cost_map(rev1, p0, reps=reps)
    path = tmp_path / f"cost_map_frozen_R{reps}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    return path, _sha(path)


@pytest.fixture(scope="module")
def frozen(chain):
    """The formal R=10000 cost map, built once per module."""
    path, sha = _frozen_cost_map(chain, chain.tmp)
    return types.SimpleNamespace(path=path, sha=sha, map=json.loads(path.read_text()))


def _outcome_args(chain, cm_path, cm_sha, out, **over):
    kw = dict(cost_map=str(cm_path), cost_map_sha256=cm_sha, rev1_package_manifest=str(chain.manifest_path),
              phase0_arm_matrix=str(chain.p0.matrix_path), phase0_launch_manifest=str(chain.p0.ledger),
              phase0_journal=str(chain.p0.journal), phase0_per_step=str(chain.p0.per_step),
              split_manifest=str(chain.world.split), trials=TRIALS, out=str(out))
    kw.update(over)
    return types.SimpleNamespace(**kw)


def test_b2_outcome_stage_refuses_any_frozen_input_drift(chain, frozen, tmp_path):
    cm_path, cm_sha = frozen.path, frozen.sha
    out = tmp_path / "design.json"
    res = phase0_outcome_design.run(_outcome_args(chain, cm_path, cm_sha, out))
    assert res["decision_gates"].keys() == {"A2", "A5", "A6"} and out.is_file()
    assert _sha(cm_path) == cm_sha
    # (a) flip one success only
    j = _rewrite_rows(chain.p0.journal, tmp_path / "j.jsonl", lambda r: r.__setitem__("success", not r["success"]))
    with pytest.raises(SystemExit, match="journal differs"):
        phase0_outcome_design.run(_outcome_args(chain, cm_path, cm_sha, out, phase0_journal=str(j)))
    # (b) change one verdict tier only
    def tier(r):
        if r.get("hit_type") == MISS and r.get("step_idx") == 0 and r.get("yaml_id") == "dsp_sv_p85":
            r["hit_type"] = FULL
    p = _rewrite_rows(chain.p0.per_step, tmp_path / "p.jsonl", tier)
    with pytest.raises(SystemExit, match="per_step differs"):
        phase0_outcome_design.run(_outcome_args(chain, cm_path, cm_sha, out, phase0_per_step=str(p)))
    # (c) wrong frozen SHA
    with pytest.raises(SystemExit, match="cost map SHA"):
        phase0_outcome_design.run(_outcome_args(chain, cm_path, "0" * 64, out))
    # (d) another (legal) Rev 1 package
    import shutil
    other = tmp_path / "other_pkg"
    shutil.copytree(chain.manifest_path.parent, other)
    m = json.loads((other / pkgmod.MANIFEST_NAME).read_text())
    m["note"] = "same members, different manifest bytes"
    (other / pkgmod.MANIFEST_NAME).write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="not the package the cost map froze"):
        phase0_outcome_design.run(_outcome_args(chain, cm_path, cm_sha, out,
                                                rev1_package_manifest=str(other / pkgmod.MANIFEST_NAME)))
    # (e) seed / R are part of the frozen bytes
    rev1 = cm._load_rev1_source(str(chain.manifest_path), TRIALS)
    p0 = cm._load_phase0_source(str(chain.p0.matrix_path), str(chain.p0.ledger), str(chain.world.split),
                                str(chain.p0.journal), str(chain.p0.per_step), TRIALS)
    alt = cm.build_cost_map(rev1, p0, seed=1, reps=120)
    assert alt["bootstrap_index_sha256"] != json.loads(cm_path.read_text())["bootstrap_index_sha256"]


def test_b3_exact_integral_cases():
    h = fh.upper_concave_hull([(0, 0), (0.3, 0.9), (1, 1)])
    assert h == [(0.0, 0.0), (0.3, 0.9), (1.0, 1.0)]
    assert fh.auc_norm(h, 0, 1) == pytest.approx(0.3 * 0.45 + 0.7 * 0.95, abs=1e-12)   # breakpoint off any grid
    assert fh.auc_norm(h, 0.1, 0.2) == pytest.approx(0.45, abs=1e-12)
    a, b = [(0, 0), (0.3, 0.9), (1, 1)], [(0, 0.2), (1, 0.6)]
    assert fh.auc_with_support(a, b, 0, 1)[0] == pytest.approx(-fh.auc_with_support(b, a, 0, 1)[0])
    assert fh.auc_with_support(a, a, 0, 1)[0] == 0.0


def test_b4_parity_tampering_is_refused_by_runner_and_discipline(chain, tmp_path):
    from exp.dispatch_surface.template_parity import assert_no_placeholders, assert_template_parity
    art = load_surface_artifact(chain.p0.matrix["artifact_paths"]["dsp_sv_p85"])
    src = load_surface_artifact(str(chain.world.artifacts["dsp_sv"]))
    assert_template_parity(art, src, what="ok")
    bad = copy.deepcopy(art)
    bad.w = np.asarray(bad.w) * 2
    with pytest.raises(SystemExit, match="array w"):
        assert_template_parity(bad, src, what="w")
    bad = copy.deepcopy(art)
    bad.quantile_alpha = 0.1
    with pytest.raises(SystemExit, match="quantile_alpha"):
        assert_template_parity(bad, src, what="alpha")
    with pytest.raises(SystemExit, match="placeholder"):
        assert_no_placeholders({"nested": {"source": "pending"}}, what="meta")
    with pytest.raises(SystemExit, match="placeholder"):
        assert_no_placeholders({"list": ["x", {"a": ["TBD"]}]}, what="meta")
    # export record with an extra key / duplicate mapping
    rec = json.loads((chain.tmp / "ex_sv" / "export_record.json").read_text())
    rec["extra"] = 1
    from exp.dispatch_surface.template_parity import assert_export_record_schema
    with pytest.raises(SystemExit, match="frozen schema"):
        assert_export_record_schema(rec, what="x", cost_model_digest=ac.cost_model_digest(),
                                    protocol="dispatch_surface_rev2_phase0")
    # discipline recomputes YAML bytes and re-validates the anchor shape
    m = copy.deepcopy(chain.p0.matrix)
    ypath = tmp_path / "anchor_tampered.yaml"
    doc = yaml.safe_load(pathlib.Path(m["arms"][ANCHOR_ARM]).read_text())
    doc["checkpoints"]["cp1"]["judge"]["warm_tiers"] = [{"threshold": 0.9, "start_t": 0.3}]
    ypath.write_text(yaml.safe_dump(doc, sort_keys=False))
    m["arms"][ANCHOR_ARM] = str(ypath)
    m["arm_yaml_sha256"][ANCHOR_ARM] = _sha(ypath)
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(m))
    with pytest.raises(SystemExit):
        phase0_discipline.validate(str(mp), str(chain.p0.ledger), str(chain.world.split), trials=TRIALS)


def test_b5_endpoint_ties_and_quantile_indices():
    # tied minimum cost -> the LARGEST delta is the low endpoint; tied maximum -> smallest delta
    lo, hi = cm.family_endpoints([0.80, 0.90, 0.95, 0.975], [50.0, 30.0, 20.0, 20.0],
                                 ["p80", "p90", "p95", "p975"])
    assert (lo, hi) == ("p975", "p80")
    lo, hi = cm.family_endpoints([0.80, 0.90, 0.95], [50.0, 50.0, 20.0], ["p80", "p90", "p95"])
    assert (lo, hi) == ("p95", "p80")
    ones = np.ones(10000)
    iv = cm.interval_from_endpoints(ones, ones)
    assert (iv["qL_zero_based_index"], iv["qH_zero_based_index"]) == (9950, 49)
    assert cm.quantile_index(0.995, 10000, "higher") == 9950 and cm.quantile_index(0.005, 10000, "lower") == 49
    with pytest.raises(SystemExit):
        cm.interval_from_endpoints(np.ones(5), np.ones(4))


def test_b5_cost_map_orders_by_real_delta(chain, frozen):
    out = frozen.map
    for fam in ("sv", "s0"):
        ds = out["families"][fam]["deltas"]
        assert ds == sorted(ds) and len(set(ds)) == len(ds)
        assert all(out["deltas"][a] == d for a, d in zip(out["families"][fam]["arms"], ds))
    assert out["endpoints"]["sv"] == {"low": "dsp_sv", "high": "dsp_sv_minus"}
    assert out["endpoints"]["s0"] == {"low": "dsp_s0_p95", "high": "dsp_s0_p80"}
    assert set(out["input_sha256"]["phase0"]) == {"matrix", "ledger", "split_manifest", "journal", "per_step", "export_records"}


def test_b6_per_task_blocks_and_separate_gates(chain, frozen, tmp_path):
    out = tmp_path / "d.json"
    res = phase0_outcome_design.run(_outcome_args(chain, frozen.path, frozen.sha, out))
    assert set(res["per_task_descriptive"]) == {str(t) for t in range(10)}
    block = res["per_task_descriptive"]["0"]
    assert {"H1_descriptive_auc", "H2_descriptive_auc", "H1_middle_discordance", "ceiling", "floor"} <= set(block)
    assert "a6_pass" not in res["hypotheses"]["H1"] and "a6_pass" not in res["hypotheses"]["H2"]
    assert res["decision_gates"]["A6"]["power_n40"] == res["hypotheses"]["H2"]["power_n40"]
    assert res["decision_gates"]["A2"]["applies"] is False       # libero_10 fixture


def test_b7_full_migration_chain_without_historical_paths(chain, tmp_path, monkeypatch, capsys):
    import shutil
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copytree(chain.manifest_path.parent, moved / "package")
    manifest_path = moved / "package" / pkgmod.MANIFEST_NAME
    for name in ("table.jsonl", "lib.pkl", "split_manifest.json"):
        shutil.copyfile(chain.world.root / name, moved / name)
    shutil.copytree(chain.world.pool_dir, moved / "test_aprime")
    hidden = tmp_path / "hidden"
    shutil.move(str(chain.world.root), str(hidden))
    try:
        world = types.SimpleNamespace(table=moved / "table.jsonl", lib=moved / "lib.pkl",
                                      split=moved / "split_manifest.json", pool_dir=moved / "test_aprime",
                                      contract=chain.world.contract)
        run_exporter(manifest_path, world, moved / "ex_sv", "artifact.dsp_sv", "0.85")
        run_exporter(manifest_path, world, moved / "ex_s0", "artifact.dsp_s0", "0.80,0.95")
        out_dir = moved / "cfg"
        emit_exploratory(types.SimpleNamespace(
            suite=SUITE, export_records=f"{moved / 'ex_sv' / 'export_record.json'},{moved / 'ex_s0' / 'export_record.json'}",
            rev1_package_manifest=str(manifest_path), template=str(TEMPLATE), library_pkl=str(world.lib),
            out_dir=str(out_dir), layer=LAYER_EXPLORATORY))
        matrix_path = out_dir / f"arm_matrix_{LAYER_EXPLORATORY}.json"
        matrix = json.loads(matrix_path.read_text())
        validate_exploratory_matrix_artifacts(matrix)
        _StubPolicy.fingerprint = chain.world.contract["policy_fingerprint"]
        dry = moved / "dry"
        dry.mkdir()
        argv = _runner_argv(chain, matrix_path, LAYER_EXPLORATORY, dry)
        argv[argv.index("--split-manifest") + 1] = str(world.split)
        argv[argv.index("--pool-dir") + 1] = str(world.pool_dir)
        capsys.readouterr()          # drain the emitter's own print line
        _run_main(monkeypatch, argv)
        assert json.loads(capsys.readouterr().out)["dry_validate"] is True
        # a real (synthetic) run on the migrated matrix, then discipline -> cost map
        pool = _pool_attestation(world.split, world.pool_dir)
        run_dir = moved / "run"
        run_dir.mkdir()
        rng = np.random.default_rng(11)
        per_step, journal = [], []
        for arm in matrix["arms"]:
            for t, offs in _officials().items():
                for subset, official in enumerate(offs):
                    verdicts = [MISS] * 8 if arm == ANCHOR_ARM else _tier_plan(arm, rng)
                    rows, j = _rows_for(arm, t, subset, official, verdicts, True, run_id="runmigrated01")
                    per_step += rows
                    journal.append(j)
        (run_dir / "per_step.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_step))
        (run_dir / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in journal))
        entry = dict(chain.p0.entry, run_id="runmigrated01", executed_arms=sorted(matrix["arms"]),
                     descriptive_arms=sorted(matrix["arms"]), arm_matrix_sha256=_sha(matrix_path),
                     frozen_yaml_sha256=matrix["arm_yaml_sha256"], executed_yaml_sha256=matrix["arm_yaml_sha256"],
                     artifact_sha256=matrix["artifact_sha256"], export_record_sha256=matrix["export_record_sha256"],
                     rev1_package_manifest_sha256=matrix["rev1_package_manifest_sha256"],
                     split_manifest_sha256=_sha(world.split), aprime_content_sha256=pool["rollup_sha256"], pool=pool)
        ledger = _write_json(run_dir / "per_step.jsonl.launch.json", {"schema_version": 2, "launches": [entry]})
        phase0_discipline.validate(str(matrix_path), str(ledger), str(world.split), trials=TRIALS)
        rev1 = cm._load_rev1_source(str(manifest_path), TRIALS)
        p0 = cm._load_phase0_source(str(matrix_path), str(ledger), str(world.split),
                                    str(run_dir / "journal.jsonl"), str(run_dir / "per_step.jsonl"), TRIALS)
        out = cm.build_cost_map(rev1, p0, reps=60)
        assert str(hidden) not in json.dumps(out) and str(chain.world.root) not in json.dumps(out)
    finally:
        shutil.move(str(hidden), str(chain.world.root))


# ------------------------------------------------------------------
# 11. G2 R2 adversarial regressions
# ------------------------------------------------------------------

@pytest.mark.parametrize("reps", [None, 1, 120, 9999, 10001, True])
def test_g2r2_b1_non_frozen_replicates_are_refused(frozen, reps):
    m = copy.deepcopy(frozen.map)
    m["replicates"] = reps
    with pytest.raises(SystemExit, match="replicates"):
        phase0_outcome_design.validate_cost_map_header(m)


def test_g2r2_b1_wrong_seed_and_happy_path(frozen):
    m = copy.deepcopy(frozen.map)
    m["seed"] = 1
    with pytest.raises(SystemExit, match="seed"):
        phase0_outcome_design.validate_cost_map_header(m)
    phase0_outcome_design.validate_cost_map_header(frozen.map)
    assert frozen.map["replicates"] == 10000 and frozen.map["seed"] == 20260829


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda m: m.__setitem__("protocol", "dispatch_surface_rev3"), "protocol"),
        (lambda m: m.__setitem__("cost_model_digest", "0" * 64), "cost model"),
        (lambda m: m["roster_spec"].__setitem__("suite", "libero_spatial"), "roster"),
        (lambda m: m["input_sha256"]["phase0"].pop("journal"), "wrong schema"),
        (lambda m: m["selected"]["sv"].__setitem__(0, "dsp_s0"), "another family"),
        (lambda m: m["selected"]["sv"].reverse(), "low/middle/high"),
        (lambda m: m["interval"].__setitem__("c_2", m["interval"]["c_1"]), "strictly increasing"),
        (lambda m: m["interval"].update({"c_1": m["interval"]["c_L"] + 1.3,
                                          "c_2": m["interval"]["c_L"] + 2.6,
                                          "c_H": m["interval"]["c_L"] + 3.9}), "width"),
        (lambda m: m["point_cost"].__setitem__(m["selected"]["sv"][1], m["point_cost"][m["selected"]["sv"][0]]),
         "mechanical A-3"),
    ],
)
def test_reviewer_cost_map_header_is_fail_closed(frozen, mutation, message):
    m = copy.deepcopy(frozen.map)
    mutation(m)
    with pytest.raises(SystemExit, match=message):
        phase0_outcome_design.validate_cost_map_header(m)


def test_reviewer_bootstrap_index_has_one_shared_implementation_and_rejects_bad_shape():
    from exp.dispatch_surface import cost_map_api

    assert cm.shared_bootstrap_index is cost_map_api.shared_index
    grid = {(0, 0), (0, 1)}
    with pytest.raises(SystemExit, match="full resampled grid"):
        cost_map_api.index_arrays([[(0, 0)]], grid)
    with pytest.raises(SystemExit, match="outside the frozen grid"):
        cost_map_api.index_arrays([[(0, 0), (9, 9)]], grid)


def test_g2r2_b1_entry_refuses_small_r_cost_map(chain, tmp_path):
    cm_path, cm_sha = _frozen_cost_map(chain, tmp_path, reps=120)
    with pytest.raises(SystemExit, match="replicates"):
        phase0_outcome_design.run(_outcome_args(chain, cm_path, cm_sha, tmp_path / "x.json"))


def test_g2r2_b2_a2_extrema_are_exact_at_breakpoints():
    """Reviewer's counterexample: a 101-point grid says 'dominated', the breakpoint at c=0.024 says not."""
    sv = [(0.0, 0.4998), (0.024, 0.50196), (4.0, 0.50196)]
    t = [(0.0, 0.5), (0.016, 0.5016), (4.0, 0.66096)]
    grid = np.linspace(0, 4, 101)
    grid_max = max(fh.sr_at(sv, c) - fh.sr_at(t, c) for c in grid)
    assert grid_max < 0                                   # the old grid verdict
    ext = fh.frontier_difference_extrema(sv, t, 0.0, 4.0)
    assert ext["max"] == pytest.approx(0.00004, abs=1e-9) and ext["argmax_cost"] == 0.024
    assert not ext["a_dominated"]
    assert set(ext["breakpoints"]) == {0.0, 0.016, 0.024, 4.0}


def test_g2r2_b2_design_a2_uses_breakpoints(chain, frozen, tmp_path):
    res = phase0_outcome_design.run(_outcome_args(chain, frozen.path, frozen.sha, tmp_path / "a2.json"))
    a2 = res["decision_gates"]["A2"]
    if "checked_breakpoints" in a2:
        assert a2["checked_breakpoints"][0] == res["interval"][0] and a2["checked_breakpoints"][-1] == res["interval"][1]
        assert a2["argmax_cost"] in a2["checked_breakpoints"]


def test_g2r2_nonblocking_outcome_write_is_atomic_after_final_check(chain, frozen, tmp_path):
    out = tmp_path / "atomic.json"
    phase0_outcome_design.run(_outcome_args(chain, frozen.path, frozen.sha, out))
    assert out.is_file() and not out.with_name(out.name + ".tmp").exists()
