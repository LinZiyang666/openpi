"""Tests for the cost-bench power freeze (G2-B5 / G2R2-B4): gate-identical
simulation, tamper-evident record validation, and the fixed-seed per-task
block permutation with content-covering digests (G2R2-B5)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import exp.dispatch_surface.power_sim_cost_blocks as power_mod
from exp.dispatch_surface.analysis.analyze_precheck import GATE1_UPPER_Q, GATE2_UPPER_Q
from exp.dispatch_surface.power_sim_cost_blocks import (
    GATES,
    POWER_TARGET,
    R_CANDIDATES,
    derive_chosen_r,
    gate_power,
    record_digest,
    simulate,
    validate_power_record,
)
from exp.dispatch_surface.run_cost_bench import (
    assert_compute_pass_metadata,
    materialize_block_pools,
)


@pytest.fixture(autouse=True)
def _small_but_frozen_power_replay(monkeypatch):
    """Keep unit replay cheap while preserving the production freeze checks."""
    monkeypatch.setattr(power_mod, "POWER_SEED", 1)
    monkeypatch.setattr(power_mod, "POWER_N_SIM", 50)
    monkeypatch.setattr(power_mod, "POWER_N_BOOT", 50)


def test_gate_quantiles_match_the_formal_adjudicator():
    """Regression for G2R2-B4: Gate 2 was simulated at p95 while the formal
    gate adjudicates at p97.5. The frozen gate table must carry the SAME
    quantile constants the analyzer imports."""
    by_name = {g[0]: g for g in GATES}
    assert by_name["gate1_compute"][3] == GATE1_UPPER_Q == 0.95
    assert by_name["gate1_latency"][3] == GATE1_UPPER_Q
    assert by_name["gate2_compute"][3] == GATE2_UPPER_Q == 0.975
    assert by_name["gate2_latency"][3] == GATE2_UPPER_Q
    # Per-axis sigmas are separate inputs, not one shared scalar.
    assert {g[4] for g in GATES} == {"compute", "latency"}


def test_gate_power_direction():
    rng = np.random.default_rng(0)
    hi = gate_power(5, -0.5, -0.05, 0.95, 0.01, rng, n_sim=100, n_boot=100)
    lo = gate_power(5, -0.05, -0.05, 0.95, 0.05, rng, n_sim=100, n_boot=100)
    assert hi > 0.95 and lo < POWER_TARGET


def _record_with_source(tmp_path, **sim_kwargs):
    import hashlib

    src = tmp_path / "variance.json"
    source = {
        "schema_version": 1,
        "sigma_compute": sim_kwargs["sigma_compute"],
        "sigma_latency": sim_kwargs["sigma_latency"],
    }
    src.write_text(json.dumps(source, sort_keys=True))
    rec = simulate(**sim_kwargs)
    rec["variance_source"] = str(src)
    rec["variance_source_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
    rec["record_digest"] = record_digest(rec)
    return rec, src


def test_simulate_chooses_smallest_sufficient_r_and_is_tamper_evident(tmp_path):
    rec, src = _record_with_source(
        tmp_path, sigma_compute=0.01, sigma_latency=0.01, seed=1, n_sim=50, n_boot=50,
    )
    assert rec["chosen_r"] == R_CANDIDATES[0]
    assert validate_power_record(rec) == R_CANDIDATES[0]

    # Hand-written record: chosen_r not derivable from per_r_power -> refused.
    fake = dict(rec)
    fake["chosen_r"] = 1
    fake["record_digest"] = record_digest(fake)
    with pytest.raises(SystemExit):
        validate_power_record(fake)

    # Digest tamper: edited powers without recomputing the digest -> refused.
    edited = dict(rec)
    edited["per_r_power"] = {k: dict(v) for k, v in rec["per_r_power"].items()}
    edited["per_r_power"]["5"]["gate1_compute"] = 0.123
    with pytest.raises(SystemExit):
        validate_power_record(edited)

    # Gate-constant drift (e.g. quantile downgraded to p95) -> refused.
    drifted = dict(rec)
    drifted["gates"] = [dict(g, quantile=0.95) for g in rec["gates"]]
    drifted["record_digest"] = record_digest(drifted)
    with pytest.raises(SystemExit):
        validate_power_record(drifted)


def test_forged_powers_with_recomputed_digest_fail_replay(tmp_path):
    """G2R3-B2 adversarial reproduction: forge every gate power to 1.0,
    keep chosen_r derivable, RECOMPUTE the self-digest — the self-hash check
    passes, but deterministic replay of the recorded parameters refuses."""
    rec, _ = _record_with_source(
        tmp_path, sigma_compute=0.3, sigma_latency=0.3, seed=1, n_sim=50, n_boot=50,
    )
    forged = dict(rec)
    forged["per_r_power"] = {
        k: {n: 1.0 for n in v} for k, v in rec["per_r_power"].items()
    }
    forged["chosen_r"] = R_CANDIDATES[0]  # derivable from the forged powers
    forged["record_digest"] = record_digest(forged)
    with pytest.raises(SystemExit):
        validate_power_record(forged)


def test_variance_source_byte_drift_refused(tmp_path):
    rec, src = _record_with_source(
        tmp_path, sigma_compute=0.01, sigma_latency=0.01, seed=1, n_sim=50, n_boot=50,
    )
    src.write_text('{"drifted": true}')
    with pytest.raises(SystemExit):
        validate_power_record(rec)


def test_variance_values_and_simulation_budget_cannot_self_authorize(tmp_path):
    rec, _ = _record_with_source(
        tmp_path, sigma_compute=0.01, sigma_latency=0.01, seed=1, n_sim=50, n_boot=50,
    )
    # A record that cites the same source but replays a more favourable sigma
    # is self-consistent; it must still fail the content-authority comparison.
    forged_sigma = simulate(0.001, 0.001, seed=1, n_sim=50, n_boot=50)
    forged_sigma["variance_source"] = rec["variance_source"]
    forged_sigma["variance_source_sha256"] = rec["variance_source_sha256"]
    forged_sigma["record_digest"] = record_digest(forged_sigma)
    with pytest.raises(SystemExit):
        validate_power_record(forged_sigma)

    # Deterministic replay of an operator-selected 1x1 simulation is not the
    # preregistered power calculation.
    tiny = simulate(0.01, 0.01, seed=1, n_sim=1, n_boot=1)
    tiny["variance_source"] = rec["variance_source"]
    tiny["variance_source_sha256"] = rec["variance_source_sha256"]
    tiny["record_digest"] = record_digest(tiny)
    with pytest.raises(SystemExit):
        validate_power_record(tiny)

    boolean_power = dict(rec)
    boolean_power["per_r_power"] = {k: dict(v) for k, v in rec["per_r_power"].items()}
    boolean_power["per_r_power"]["5"]["gate1_compute"] = True
    boolean_power["record_digest"] = record_digest(boolean_power)
    with pytest.raises(SystemExit):
        validate_power_record(boolean_power)


def test_underpowered_record_is_refused_even_if_consistent(tmp_path):
    rec, _ = _record_with_source(
        tmp_path, sigma_compute=5.0, sigma_latency=5.0, seed=1, n_sim=50, n_boot=50,
    )
    assert rec["chosen_r"] is None
    with pytest.raises(SystemExit):
        validate_power_record(rec)


def test_derive_chosen_r_requires_all_four_gates():
    per_r = {"5": {"a": 0.9, "b": 0.7}, "10": {"a": 0.9, "b": 0.9}, "15": {"a": 1.0, "b": 1.0}}
    assert derive_chosen_r(per_r) == 10


# ------------------------------------------------------------------
# Compute-pass probe-backend attestation (G2R3-B3)
# ------------------------------------------------------------------


def _cuda_meta(backends=None):
    return {
        "cuda_available": True, "gpu_name": "RTX",
        "stage_devices": {"stage1": "cuda:0", "stage2": "cuda:0", "stage3": "cuda:0"},
        "stage_probe_backends": backends if backends is not None else
        {"stage1": "cuda", "stage2": "cuda", "stage3": "cuda"},
    }


def test_compute_pass_accepts_all_cuda_probes():
    assert_compute_pass_metadata(_cuda_meta())


def test_compute_pass_rejects_cpu_stage_on_cuda_host():
    """cuda_available=True with an explicit CPU stage2 used to pass; the
    per-stage probe-backend map must refuse it."""
    with pytest.raises(SystemExit):
        assert_compute_pass_metadata(_cuda_meta({"stage1": "cuda", "stage2": "cpu",
                                                 "stage3": "cuda"}))


def test_compute_pass_rejects_missing_attestation_or_no_cuda():
    with pytest.raises(SystemExit):
        assert_compute_pass_metadata({"cuda_available": True})  # no backend map
    with pytest.raises(SystemExit):
        assert_compute_pass_metadata({"cuda_available": False,
                                      "stage_devices": {"stage1": "cuda:0",
                                                        "stage2": "cuda:0",
                                                        "stage3": "cuda:0"},
                                      "stage_probe_backends": {"stage1": "cuda",
                                                               "stage2": "cuda",
                                                               "stage3": "cuda"}})


@pytest.mark.parametrize("backends", [
    {},
    {"stage1": "cuda", "stage2": "cuda"},
    {"stage1": "cuda", "stage2": "cuda", "stage3": "cuda", "fake": "cuda"},
])
def test_compute_pass_rejects_incomplete_or_extra_backend_key_domain(backends):
    with pytest.raises(SystemExit):
        assert_compute_pass_metadata(_cuda_meta(backends))


# ------------------------------------------------------------------
# Block permutation pools
# ------------------------------------------------------------------


def _aprime(tmp_path, name="aprime", n_tasks=2, n_inits=30, offset=0.0):
    d = tmp_path / name
    d.mkdir()
    for t in range(n_tasks):
        states = (torch.arange(n_inits * 2, dtype=torch.float32).reshape(n_inits, 2)
                  + t * 100 + offset)
        torch.save(states, d / f"task_{t}.init")
    return d


def test_block_pools_use_seeded_permutation(tmp_path):
    aprime = _aprime(tmp_path)
    pools, mapping, digest = materialize_block_pools(aprime, tmp_path / "o1", 5, seed=7)
    _, mapping2, digest2 = materialize_block_pools(aprime, tmp_path / "o2", 5, seed=7)
    assert mapping == mapping2 and digest == digest2
    _, _, digest3 = materialize_block_pools(aprime, tmp_path / "o3", 5, seed=8)
    assert digest3 != digest
    for _task_file, block_map in mapping.items():
        positions = list(block_map.values())
        assert len(set(positions)) == len(positions)
        assert all(0 <= p < 29 for p in positions)
    states = torch.load(aprime / "task_0.init", weights_only=False)
    b0 = torch.load(pools[0] / "task_0.init", weights_only=False)
    assert torch.equal(b0[0], states[mapping["task_0.init"]["0"]])


def test_block_pool_digest_covers_state_bytes(tmp_path):
    """Regression for G2R2-B5: two A' trees with identical filenames and
    lengths but different state BYTES must not share a digest."""
    a1 = _aprime(tmp_path, "a1")
    a2 = _aprime(tmp_path, "a2", offset=0.5)
    _, _, d1 = materialize_block_pools(a1, tmp_path / "o1", 3, seed=7)
    _, _, d2 = materialize_block_pools(a2, tmp_path / "o2", 3, seed=7)
    assert d1 != d2


def test_block_pools_reject_r_beyond_usable(tmp_path):
    aprime = _aprime(tmp_path, n_inits=4)
    with pytest.raises(SystemExit):
        materialize_block_pools(aprime, tmp_path / "o", 4, seed=7)
