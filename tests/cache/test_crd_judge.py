"""State-machine and artifact fail-fast tests for the H-CRD judge (exploratory prototype).

Synthetic s-only artifact with 4 s-bins x 1 v-bin. Grids (layer 0 = warm,
layer 1 = full; q decreasing in s): with delta = 1 the top two bins are locally
admissible for both tiers, the second bin for neither, and s below the second
edge is +inf (never admissible) -- mirroring ``export_boundaries``.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from openpi.cache.components import crd_judge as crd
from openpi.cache.components.judge import HitType
from openpi.cache.components.surface_judge import SURFACE_ARTIFACT_SCHEMA_VERSION
from openpi.cache.types import CheckpointID

U = np.array([[[9.0], [2.0], [0.8], [0.6]],      # warm
              [[9.0], [3.0], [0.9], [0.7]]])      # full
D = np.array([[[4.0], [1.0], [0.4], [0.3]],
              [[4.0], [1.5], [0.5], [0.35]]])
S_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


class _Res:
    def __init__(self, score, rid="e0"):
        self.score = score
        self.id = rid


def write_crd(tmp_path, *, gamma=1.0, beta=5.0, j_bad=2, l_max=3, delta=1.0, delta_reopen=None,
              min_recovery_misses=0, u=U, d=D, s_edges=S_EDGES, task_scale=None, meta_extra=None,
              contract=None, w=None, k=1, name="crd.npz"):
    u = np.asarray(u, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    w = np.ones(2, np.float32) if w is None else np.asarray(w, dtype=np.float32)
    scalars = {"schema_version": SURFACE_ARTIFACT_SCHEMA_VERSION, "k": k, "h_exec": 5, "start_t_ws": 0.3,
               "delta": delta, "quantile_alpha": 0.05, "certification_mode": "empirical_no_certificate",
               "uses_disagreement": False, "conformal_c": 0.0, "n_calibration_episodes": 0}
    crd_block = {"gamma": gamma, "beta": None if beta == math.inf else beta,
                 "j_bad": j_bad, "l_max": l_max, "min_recovery_misses": min_recovery_misses,
                 "delta_reopen": delta if delta_reopen is None else delta_reopen,
                 "task_scale": {"0": 1.0, "1": 2.0} if task_scale is None else task_scale,
                 "upper_grid_sha256": crd.grid_sha256(u), "central_grid_sha256": crd.grid_sha256(d)}
    crd_block.update(meta_extra or {})
    meta = {"judge_variant": crd.JUDGE_VARIANT, "crd": crd_block}
    contract = contract or {"h_exec": 5, "top_k": k, "action_dim": len(w), "library_sha256": "x"}
    path = tmp_path / name
    np.savez(path, w=w, active_mask=np.ones(len(w), bool), v_bin_edges=np.array([-np.inf, np.inf]),
             s_min_full=np.array([0.75]), s_min_warm=np.array([0.5]), q_hat=u, q_hat_central=d, s_edges=np.asarray(s_edges),
             scalars_json=np.frombuffer(json.dumps(scalars).encode(), dtype=np.uint8),
             contract_json=np.frombuffer(json.dumps(contract, sort_keys=True).encode(), dtype=np.uint8),
             meta_json=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))
    return str(path)


def _step(judge, s, executed=None):
    res = judge([_Res(s)], CheckpointID.CP1, {})
    hit = res.hit_type if executed is None else executed
    out = judge.commit_verdict(CheckpointID.CP1, hit_type=hit)
    return res.hit_type, out


def _run(judge, scores):
    return [_step(judge, s)[0] for s in scores]


# ------------------------------------------------------------------
# core rule
# ------------------------------------------------------------------

def test_static_equivalence_when_memory_off(tmp_path):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path, gamma=0.0, beta=math.inf, j_bad=None, l_max=None))
    j.on_episode_start({"task_id": 0})
    assert _step(j, 0.9)[0] == HitType.FULL_HIT      # bin 3 -> lookup bin 2 (u_full 0.9 <= 1)
    assert _step(j, 0.6)[0] == HitType.MISS          # lookup bin 1 (3 / 2 > 1) -> region MISS
    assert _step(j, 0.3)[0] == HitType.MISS          # lookup bin 0 -> MISS
    assert _step(j, 0.1)[0] == HitType.MISS          # below the second edge -> +inf


def test_debt_budget_forces_reanchor_and_resets(tmp_path):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path, gamma=1.0, beta=1.2, j_bad=None, l_max=None))
    j.on_episode_start({"task_id": 0})
    # d_full 0.5 / step: FULL, FULL (D = 1.0), 1.5 > beta -> debt MISS (D = 0), FULL again
    assert _run(j, [0.9] * 4) == [HitType.FULL_HIT, HitType.FULL_HIT, HitType.MISS, HitType.FULL_HIT]
    assert j.state["mode"] == crd.ACTIVE and j.state["bad_run"] == 0


def test_task_scale_divides_debt(tmp_path):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path, gamma=1.0, beta=1.2, j_bad=None, l_max=None))
    j.on_episode_start({"task_id": 1})   # scale 2 -> d_full 0.25, d_warm 0.15 per step
    # four FULLs (D = 1.0); the fifth FULL would reach 1.25 > beta but WARM (1.15) still fits.
    assert _run(j, [0.9] * 5) == [HitType.FULL_HIT] * 4 + [HitType.WARM_START]


def test_fh_run_fuse(tmp_path):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path, gamma=0.0, beta=math.inf, j_bad=None, l_max=3))
    j.on_episode_start({"task_id": 0})
    assert _run(j, [0.9] * 5) == [HitType.FULL_HIT] * 3 + [HitType.MISS, HitType.FULL_HIT]


def test_non_cp1_checkpoint_leaves_no_proposal(tmp_path):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path))
    j.on_episode_start({"task_id": 0})
    res = j([_Res(0.9)], CheckpointID.CP3, {})
    assert res.hit_type == HitType.MISS and j.state["pending"] is False
    assert _step(j, 0.9)[0] == HitType.FULL_HIT       # CP1 still works afterwards


# ------------------------------------------------------------------
# B1: real hysteresis -- min_recovery_misses changes the verdict sequence
# ------------------------------------------------------------------

def test_recovery_hysteresis_changes_verdict_sequence(tmp_path):
    scores = [0.6, 0.6, 0.9, 0.9, 0.9, 0.9]     # two region MISSes, then admissible steps
    pure = crd.CumulativeRiskJudge(write_crd(tmp_path, j_bad=None, l_max=None, name="pure.npz"))
    pure.on_episode_start({"task_id": 0})
    hys = crd.CumulativeRiskJudge(write_crd(tmp_path, j_bad=2, l_max=None, min_recovery_misses=2, name="hys.npz"))
    hys.on_episode_start({"task_id": 0})
    seq_pure = _run(pure, scores)
    seq_hys = _run(hys, scores)
    assert seq_pure == [HitType.MISS, HitType.MISS] + [HitType.FULL_HIT] * 4
    # The second region MISS causes entry; it does not count as a MISS already
    # executed inside RECOVERY.  Exactly two additional MISSes precede reopen.
    assert seq_hys == [HitType.MISS] * 4 + [HitType.FULL_HIT] * 2
    assert seq_hys != seq_pure
    assert hys.state["mode"] == crd.ACTIVE


def test_recovery_with_zero_min_misses_is_inert(tmp_path):
    """The v1 configuration (delta_reopen == delta, immediate reopen) cannot change any verdict."""
    scores = [0.6, 0.6, 0.6, 0.9, 0.6, 0.9]
    pure = crd.CumulativeRiskJudge(write_crd(tmp_path, j_bad=None, l_max=None, name="p.npz"))
    pure.on_episode_start({"task_id": 0})
    hys = crd.CumulativeRiskJudge(write_crd(tmp_path, j_bad=2, l_max=None, min_recovery_misses=0, name="h.npz"))
    hys.on_episode_start({"task_id": 0})
    assert _run(pure, scores) == _run(hys, scores)


# ------------------------------------------------------------------
# B5: RECOVERY reopen must respect the debt budget
# ------------------------------------------------------------------

def test_recovery_reopen_respects_beta(tmp_path):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path, gamma=1.0, beta=0.1, j_bad=1, l_max=None))
    j.on_episode_start({"task_id": 0})
    assert _step(j, 0.6)[0] == HitType.MISS and j.state["mode"] == crd.RECOVERY
    # locally admissible (u_full 0.9 <= 1) but d_full 0.5 > beta 0.1 -> must stay MISS
    hit, out = _step(j, 0.9)
    assert hit == HitType.MISS and out["D_after"] == 0.0 and j.state["mode"] == crd.RECOVERY


# ------------------------------------------------------------------
# B4: propose / commit contract
# ------------------------------------------------------------------

def test_downgraded_warm_is_booked_as_miss_and_fails_closed(tmp_path):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path, gamma=1.0, beta=1.2, j_bad=None, l_max=None))
    j.on_episode_start({"task_id": 1})
    _run(j, [0.9] * 4)                                       # D = 1.0 -> next FULL infeasible, WARM feasible
    res = j([_Res(0.9)], CheckpointID.CP1, {})
    assert res.hit_type == HitType.WARM_START
    out = j.commit_verdict(CheckpointID.CP1, hit_type=HitType.MISS)   # orchestrator downgraded the payload
    assert out["reason"] == "downgrade" and j.state["D"] == 0.0 and j.state["mode"] == crd.RECOVERY


@pytest.mark.parametrize("proposed_score, executed", [
    (0.6, HitType.WARM_START),    # proposed MISS, executed WARM
    (0.6, HitType.FULL_HIT),      # proposed MISS, executed FULL
    (0.9, HitType.WARM_START),    # proposed FULL, executed WARM
    (0.9, HitType.MISS),          # proposed FULL, executed MISS (not a legal downgrade)
])
def test_illegal_transitions_fail_loud_and_keep_the_proposal(tmp_path, proposed_score, executed):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path, j_bad=None, l_max=None))
    j.on_episode_start({"task_id": 0})
    res = j([_Res(proposed_score)], CheckpointID.CP1, {})
    assert res.hit_type != executed
    with pytest.raises(RuntimeError, match="illegal transition"):
        j.commit_verdict(CheckpointID.CP1, hit_type=executed)
    assert j.state["pending"] is True                        # proposal not consumed by the rejected commit
    j.commit_verdict(CheckpointID.CP1, hit_type=res.hit_type)  # the legal commit still works
    assert j.state["pending"] is False


def test_propose_commit_discipline(tmp_path):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path))
    with pytest.raises(RuntimeError):
        j([_Res(0.9)], CheckpointID.CP1, {})                           # before on_episode_start
    j.on_episode_start({"task_id": 0})
    with pytest.raises(RuntimeError):
        j.commit_verdict(CheckpointID.CP1, hit_type=HitType.MISS)      # nothing proposed
    j([_Res(0.9)], CheckpointID.CP1, {})
    with pytest.raises(RuntimeError):
        j([_Res(0.9)], CheckpointID.CP1, {})                           # proposal still pending
    j.commit_verdict(CheckpointID.CP1, hit_type=HitType.FULL_HIT)
    with pytest.raises(RuntimeError):
        j.commit_verdict(CheckpointID.CP1, hit_type=HitType.FULL_HIT)  # double commit
    j.on_episode_start({"task_id": 0})
    assert j.state["D"] == 0.0 and not j.state["pending"]


def test_commit_returns_ledger_diagnostics(tmp_path):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path, j_bad=None, l_max=None))
    j.on_episode_start({"task_id": 0})
    _, out = _step(j, 0.9)
    for key in ("token", "proposed", "executed", "src", "reason", "D_before", "D_after", "mode_before", "mode_after",
                "u_full", "u_warm", "d_full", "d_warm", "D_full", "D_warm", "bad_run", "fh_run", "recovery_misses"):
        assert key in out, key
    assert out["reason"] == "full" and out["D_after"] == pytest.approx(0.5)


# ------------------------------------------------------------------
# B3: artifact fail-fast
# ------------------------------------------------------------------

def _neg(d):
    d = np.array(d, dtype=np.float64)
    d[1, 2, 0] = -100.0
    return d


def _nan(d):
    d = np.array(d, dtype=np.float64)
    d[1, 2, 0] = np.nan
    return d


@pytest.mark.parametrize("label, kwargs", [
    ("negative debt", {"d": _neg(D)}),
    ("nan debt", {"d": _nan(D)}),
    ("nan admission", {"u": _nan(U)}),
    ("central above upper", {"d": U * 1.5}),
    ("non-monotone edges", {"s_edges": np.array([0.0, 0.5, 0.25, 0.75, 1.0])}),
    ("nonpositive task scale", {"task_scale": {"0": 0.0}}),
    ("nan task scale", {"task_scale": {"0": float("nan")}}),
    ("negative task id", {"task_scale": {"-1": 1.0}}),
    ("noncanonical task id", {"task_scale": {"00": 1.0}}),
    ("incomplete task ids", {"task_scale": {"0": 1.0, "2": 1.0}}),
    ("wrong central digest", {"meta_extra": {"central_grid_sha256": "0" * 64}}),
    ("wrong upper digest", {"meta_extra": {"upper_grid_sha256": "0" * 64}}),
    ("beta zero", {"beta": 0.0}),
    ("beta bool", {"beta": True}),
    ("gamma out of range", {"gamma": 1.5}),
    ("gamma nan", {"gamma": float("nan")}),
    ("fractional j_bad", {"j_bad": 1.5}),
    ("boolean l_max", {"l_max": True}),
    ("fractional recovery dwell", {"min_recovery_misses": 1.5}),
])
def test_artifact_fail_fast(tmp_path, label, kwargs):
    path = write_crd(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        crd.CumulativeRiskJudge(path)


def test_unknown_task_id_fails_loud(tmp_path):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path))
    with pytest.raises(ValueError, match="task scale"):
        j.on_episode_start({"task_id": 7})
    with pytest.raises(ValueError, match="task scale"):
        j.on_episode_start({})


def test_rejected_episode_identity_cannot_reuse_prior_scale(tmp_path):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path))
    j.on_episode_start({"task_id": 1})
    assert j.state["task_scale"] == 2.0
    with pytest.raises(ValueError, match="task scale"):
        j.on_episode_start({"task_id": 7})
    assert j.state["task_scale"] is None
    with pytest.raises(RuntimeError, match="before on_episode_start"):
        j([_Res(0.9)], CheckpointID.CP1, {})


@pytest.mark.parametrize("task_id", [True, 1.0, "1"])
def test_episode_task_id_must_be_an_exact_integer(tmp_path, task_id):
    j = crd.CumulativeRiskJudge(write_crd(tmp_path))
    with pytest.raises(ValueError, match="task scale"):
        j.on_episode_start({"task_id": task_id})


def test_declared_crd_missing_arrays_is_not_downgraded_to_static(tmp_path):
    path = write_crd(tmp_path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "q_hat_central"}
    broken = tmp_path / "missing-central.npz"
    np.savez(broken, **arrays)
    assert crd.is_crd_artifact(str(broken))
    with pytest.raises(ValueError, match="lacks q_hat"):
        crd.CumulativeRiskJudge(str(broken))


def test_is_crd_artifact(tmp_path):
    assert crd.is_crd_artifact(write_crd(tmp_path))
