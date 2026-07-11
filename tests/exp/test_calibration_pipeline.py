"""Tests for the TRACER Phase 5 calibration pipeline (offline, no GPU).

Covers the four exp scripts of the light-calibration flow with synthetic
fixtures:

  * ``build_calibration_table``  — LOEO temp-pool exclusion (fail-loud on
    self-entries) and the chain-aware per-episode replay (row schema + the
    "state before search / action after" history timing that makes step k's
    u_t depend on prior steps).
  * ``solve_calibration.solve``  — signal table -> calibrated params doc.
  * ``emit_calibrated_yaml.apply_params`` — params -> calibrated YAML dict that
    round-trips through ``load_cache_config`` + ``validate_cache_config``, with
    ``u_t_factor`` attached iff b2 != 0.
  * ``analyze_phase5_rollout.compute_metrics`` / ``sr_from_episodes`` — exact
    BHR/FFR/IR numerators + denominators and the incomplete-episode handling.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.components.judge import FailureAwareGateJudge
from openpi.cache.config import (
    DepthPolicyConfig,
    FieldSimilarityConfig,
    SearchStrategyConfig,
    load_cache_config,
    validate_cache_config,
)
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

from exp.zixuan_proposal.analyze_phase5_rollout import (
    check_paired_provenance,
    compute_metrics,
    sr_from_episodes,
)
from exp.zixuan_proposal.build_calibration_table import (
    _assert_no_self_entries,
    _entry_identity,
    _parse_global_episode_id,
    _temp_storage_excluding,
    _winner_has_warm_snapshot,
    build_rows_for_episode,
)
from exp.zixuan_proposal.emit_calibrated_yaml import apply_params
from exp.zixuan_proposal.solve_calibration import solve

_ACTIVE_YAML = (
    Path(__file__).resolve().parents[2]
    / "exp/zixuan_proposal/config/dual_retrieval_active.yaml"
)

# Small synthetic feature dims (keep the fixtures light — real dims are 2048/32).
_EMB = 64
_STATE = 8
_ACT = 8
_CHUNK = 6
_WARM_T = 0.5


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _mk_payload(action: torch.Tensor, *, task: str, warm: bool) -> CachePayload:
    inter = {_WARM_T: action.clone()} if warm else None
    return CachePayload(
        action_chunk=action,
        task_key=task,
        intermediates=inter,
        denoising_num_steps=10 if warm else None,
    )


def _mk_entry(
    *,
    entry_id: str,
    trajectory_id: str,
    task: str,
    outcome: int,
    step_idx: int,
    vision: torch.Tensor,
    state: torch.Tensor,
    action: torch.Tensor,
    warm: bool = True,
) -> CacheEntry:
    e = CacheEntry(
        id=entry_id,
        checkpoint_id=CheckpointID.CP1,
        query_keys={"vision_0": vision, "robot_state": state},
        payload=_mk_payload(action, task=task, warm=warm),
        step_idx=step_idx,
        trajectory_id=trajectory_id,
    )
    e.outcome = outcome
    return e


def _linked_trajectory(task: str, traj: str, outcome: int, n: int, rng) -> list[CacheEntry]:
    """A prev/next-linked chain of ``n`` entries (so walk_next / trajectory search work)."""
    ents = []
    for i in range(n):
        ents.append(_mk_entry(
            # Task-qualified id: real merged artifacts guarantee globally-unique
            # ids (build_dual_artifact asserts it), so stem collisions across
            # tasks never collide at the id level — only the LOEO identity does.
            entry_id=f"{task}:{traj}:{i}",
            trajectory_id=traj,
            task=task,
            outcome=outcome,
            step_idx=i,
            vision=torch.from_numpy(rng.standard_normal(_EMB).astype(np.float32)),
            state=torch.from_numpy(rng.standard_normal(_STATE).astype(np.float32)),
            action=torch.from_numpy(rng.standard_normal((_CHUNK, _ACT)).astype(np.float32)),
        ))
    for i in range(n):
        if i > 0:
            ents[i].prev_ids = [ents[i - 1].id]
        if i < n - 1:
            ents[i].next_ids = [ents[i + 1].id]
    return ents


def _write_probe_h5(path: Path, *, task: str, n_steps: int, rng) -> None:
    """Write a per-step embedding h5 the real cp1_mean_pool builder can replay."""
    with h5py.File(path, "w") as f:
        f.attrs["task"] = task
        f.attrs["success"] = True
        for s in range(n_steps):
            g = f.create_group(f"step_{s:04d}")
            g.create_dataset("vision_0", data=rng.standard_normal((256, _EMB)).astype(np.float32))
            g.create_dataset("prompt_emb", data=rng.standard_normal((4, _EMB)).astype(np.float32))
            g.create_dataset("robot_state", data=rng.standard_normal(_STATE).astype(np.float32))
            g.create_dataset("clean_action", data=rng.standard_normal((_CHUNK, _ACT)).astype(np.float32))


def _dual_strategy_cfg() -> SearchStrategyConfig:
    """Mirror the active dual_retrieval_active.yaml search_strategy block."""
    return SearchStrategyConfig(
        type="dual_retrieval_knn",
        base_fusion="weighted_rrf",
        top_k=1,
        rrf_k=60,
        trajectory_depth=5,
        trajectory_weights=[0.4, 0.3, 0.15, 0.1, 0.05],
        allowed_depths=[5],
        depth_policy=DepthPolicyConfig(type="constant", depth=5),
        margin_lambda=0.5,
        enable_dual=True,
        field_similarity={
            "vision_0": FieldSimilarityConfig(type="cosine"),
            "robot_state": FieldSimilarityConfig(
                type="l2", to_similarity={"type": "exp", "tau": 1.0}
            ),
        },
    )


# ---------------------------------------------------------------------------
# build_calibration_table — LOEO exclusion
# ---------------------------------------------------------------------------


def test_loeo_excludes_self_entries_by_task_and_trajectory():
    rng = np.random.default_rng(0)
    # Two task dirs share the "episode_0" stem -> identity MUST use both fields.
    self_pos = _linked_trajectory("task_A", "episode_0", +1, 2, rng)          # excluded
    other_pos = _linked_trajectory("task_A", "episode_1", +1, 1, rng)          # kept
    collide = _linked_trajectory("task_B", "episode_0", +1, 1, rng)           # kept (diff task)
    self_neg = _mk_entry(                                                       # excluded (D-)
        entry_id="task_A:episode_0:neg", trajectory_id="episode_0", task="task_A",
        outcome=-1, step_idx=0,
        vision=torch.zeros(_EMB), state=torch.zeros(_STATE),
        action=torch.zeros((_CHUNK, _ACT)), warm=False,
    )
    all_entries = self_pos + other_pos + collide + [self_neg]
    identity = ("task_A", 0)  # (task_key, global_episode_id parsed from "episode_0")

    storage = _temp_storage_excluding(
        all_entries, vector_dims={"vision_0": _EMB, "robot_state": _STATE},
        library_stats=None, identity=identity,
    )

    # Exactly the two non-self entries survive.
    assert storage.count() == 2
    # The stem-colliding but different-task entry is retained (BOTH-field filter).
    assert storage.fetch_entry("task_A:episode_1:0").trajectory_id == "episode_1"
    assert storage.fetch_entry(collide[0].id).payload.task_key == "task_B"
    # No surviving entry carries the excluded identity (both D+ steps + the D-).
    for eid in ("task_A:episode_0:0", "task_A:episode_0:1", "task_A:episode_0:neg"):
        with pytest.raises(KeyError):
            storage.fetch_entry(eid)


def test_loeo_failloud_assert_catches_missing_exclusion():
    rng = np.random.default_rng(1)
    self_pos = _linked_trajectory("task_A", "episode_0", +1, 2, rng)
    other = _linked_trajectory("task_A", "episode_1", +1, 1, rng)
    all_entries = self_pos + other
    identity = ("task_A", 0)  # (task_key, global_episode_id parsed from "episode_0")

    # Post-exclusion pool is clean.
    kept = [e for e in all_entries if _entry_identity(e) != identity]
    _assert_no_self_entries(kept, identity)  # no raise

    # Forgetting to exclude is caught fail-loud.
    with pytest.raises(AssertionError, match="LOEO leakage"):
        _assert_no_self_entries(all_entries, identity)


# ---------------------------------------------------------------------------
# build_calibration_table — chain-aware per-episode replay
# ---------------------------------------------------------------------------

_ROW_KEYS = {
    "suite", "task_id", "init_state_idx", "episode_id", "step_idx",
    "s_pos", "s_neg", "delta_pos", "u_t", "winner_id",
    "winner_has_warm_snapshot", "episode_success",
}


def _build_library(rng):
    pos = (
        _linked_trajectory("lib", "traj_p0", +1, 3, rng)
        + _linked_trajectory("lib", "traj_p1", +1, 3, rng)
    )
    neg = (
        _linked_trajectory("lib", "traj_n0", -1, 2, rng)
        + _linked_trajectory("lib", "traj_n1", -1, 2, rng)
    )
    # library_stats is D+-only (Phase 4 contract): compute over the positive pool.
    ls = LibraryStats.compute_from_entries(pos)
    return pos, neg, ls


def _import_build_calibration_table():
    from exp.zixuan_proposal import build_calibration_table as m
    from exp.common.build_in_memory_cache_artifact import _create_builder

    return m, _create_builder


def test_build_rows_schema_and_chain_aware_timing(tmp_path):
    rng = np.random.default_rng(7)
    pos, neg, ls = _build_library(rng)
    all_entries = pos + neg

    _m, _create_builder = _import_build_calibration_table()
    builder = _create_builder("cp1_mean_pool")

    # u_t factor with past=2: undefined at step 0 (only 1 state buffered before
    # the search) and defined from step 1 -> proves state is appended BEFORE the
    # search and accrues across steps. b2 != 0 so the gate actually computes u_t.
    gate = FailureAwareGateJudge(
        gate_betas={"b0": -0.9, "b1": 1.0, "b2": 0.5, "b3": 0.0},
        threshold=0.5,
        u_t_factor={"descriptor": "direction", "channel": "state", "past": 2, "future": 0},
        library_stats=ls,
    )

    # Probe episode NOT present in the library (nothing excluded; full pool
    # searched). Its task attr matches the library task_key so the task-scoped
    # retrieval filter keeps the candidates (production replays a same-task query).
    h5_path = tmp_path / "episode_999.h5"
    _write_probe_h5(h5_path, task="lib", n_steps=3, rng=rng)

    rows = build_rows_for_episode(
        h5_path,
        builder=builder,
        all_entries=all_entries,
        vector_dims={"vision_0": _EMB, "robot_state": _STATE},
        library_stats=ls,
        strategy_cfg=_dual_strategy_cfg(),
        fusion_weights={"vision_0": 1.0, "robot_state": 1.0},
        gate=gate,
        suite="libero_spatial",
        episode_success=True,
        ids={"task_id": 3, "init_state_idx": 7, "episode_id": 157},
    )

    assert len(rows) == 3
    # Row schema: every required key present, types sane.
    for r in rows:
        assert set(r) == _ROW_KEYS
        assert r["suite"] == "libero_spatial"
        assert isinstance(r["episode_success"], bool) and r["episode_success"] is True
        assert isinstance(r["winner_has_warm_snapshot"], bool)
        assert r["winner_id"] is not None
        assert r["u_t"] is None or isinstance(r["u_t"], float)
    # Ids threaded through for the success join.
    assert rows[0]["task_id"] == 3 and rows[0]["init_state_idx"] == 7
    assert rows[0]["episode_id"] == 157 and rows[0]["step_idx"] == 0

    # Dual retrieval active -> the negative pool contributes a non-zero s_neg.
    assert any(r["s_neg"] > 0.0 for r in rows)

    # Chain-aware history timing: u_t undefined at step 0 (state buffer len 1 <
    # past=2), defined once >= 2 states have accrued (state appended before each
    # search) -> step k's signal depends on prior steps.
    assert rows[0]["u_t"] is None
    assert any(r["u_t"] is not None for r in rows[1:])

    # All positive library entries carry a warm snapshot at start_t=0.5, and the
    # winner is drawn from the positive pool -> flag is True.
    assert rows[0]["winner_has_warm_snapshot"] is True


def test_winner_has_warm_snapshot_helper():
    rng = np.random.default_rng(2)
    warm_entry = _mk_entry(
        entry_id="w:0", trajectory_id="w", task="lib", outcome=+1, step_idx=0,
        vision=torch.from_numpy(rng.standard_normal(_EMB).astype(np.float32)),
        state=torch.from_numpy(rng.standard_normal(_STATE).astype(np.float32)),
        action=torch.from_numpy(rng.standard_normal((_CHUNK, _ACT)).astype(np.float32)),
        warm=True,
    )
    cold_entry = _mk_entry(
        entry_id="c:0", trajectory_id="c", task="lib", outcome=+1, step_idx=0,
        vision=torch.from_numpy(rng.standard_normal(_EMB).astype(np.float32)),
        state=torch.from_numpy(rng.standard_normal(_STATE).astype(np.float32)),
        action=torch.from_numpy(rng.standard_normal((_CHUNK, _ACT)).astype(np.float32)),
        warm=False,
    )
    storage = _temp_storage_excluding(
        [warm_entry, cold_entry], vector_dims={"vision_0": _EMB, "robot_state": _STATE},
        library_stats=None, identity=("none", "none"),
    )
    from openpi.cache.components.payload_view import StoragePayloadView

    view = StoragePayloadView(storage)
    assert _winner_has_warm_snapshot(view, "w:0") is True
    assert _winner_has_warm_snapshot(view, "c:0") is False


# ---------------------------------------------------------------------------
# solve_calibration.solve
# ---------------------------------------------------------------------------


def _synthetic_signal_rows():
    return [
        {"s_pos": 0.95, "s_neg": 0.10, "delta_pos": 0.30, "u_t": 0.4,
         "winner_has_warm_snapshot": True, "episode_success": True},
        {"s_pos": 0.30, "s_neg": 0.70, "delta_pos": 0.02, "u_t": None,
         "winner_has_warm_snapshot": False, "episode_success": False},
        {"s_pos": 0.80, "s_neg": 0.20, "delta_pos": 0.25, "u_t": 1.1,
         "winner_has_warm_snapshot": True, "episode_success": True},
        {"s_pos": 0.55, "s_neg": 0.50, "delta_pos": 0.05, "u_t": -0.3,
         "winner_has_warm_snapshot": True, "episode_success": False},
        {"s_pos": 0.90, "s_neg": 0.05, "delta_pos": 0.40, "u_t": 0.2,
         "winner_has_warm_snapshot": False, "episode_success": True},
    ]


def test_solve_returns_params_and_metrics():
    doc = solve(_synthetic_signal_rows(), suite="libero_spatial", c_miss=1.0, c_warm=0.75)

    assert doc["suite"] == "libero_spatial"
    assert doc["n_rows"] == 5
    params = doc["params"]
    assert {"margin_lambda", "gate_betas", "threshold", "warm_tiers"} <= set(params)
    assert {"b0", "b1", "b2", "b3"} <= set(params["gate_betas"])
    assert isinstance(params["warm_tiers"], list)
    assert "metrics" in doc and "L_cal" in doc["metrics"]
    assert 0.0 <= doc["metrics"]["BadHitRate"] <= 1.0


# ---------------------------------------------------------------------------
# emit_calibrated_yaml.apply_params
# ---------------------------------------------------------------------------


def _base_cfg_dict() -> dict:
    return yaml.safe_load(_ACTIVE_YAML.read_text(encoding="utf-8"))


def _roundtrip_ok(cfg_dict: dict, tmp_path: Path, name: str) -> None:
    out = tmp_path / name
    out.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")
    cfg = load_cache_config(out)     # parses + validates
    validate_cache_config(cfg)       # explicit re-validate (idempotent)


def test_apply_params_writes_calibrated_gate_and_lambda(tmp_path):
    params = {
        "margin_lambda": 0.75,
        "gate_betas": {"b0": -1.2, "b1": 1.0, "b2": 0.0, "b3": 0.25},
        "threshold": 0.5,
        "warm_tiers": [{"threshold": 0.3, "start_t": 0.5}],
    }
    out = apply_params(_base_cfg_dict(), params, u_t_factor=None)
    judge = out["checkpoints"]["cp1"]["judge"]
    ss = out["checkpoints"]["cp1"]["search_strategy"]

    assert judge["gate_betas"] == params["gate_betas"]
    assert judge["threshold"] == 0.5
    assert judge["warm_tiers"] == params["warm_tiers"]
    assert ss["margin_lambda"] == 0.75
    # b2 == 0 -> no u_t_factor attached.
    assert "u_t_factor" not in judge
    _roundtrip_ok(out, tmp_path, "calibrated_b2zero.yaml")


def test_apply_params_attaches_u_t_factor_when_b2_nonzero(tmp_path):
    params = {
        "margin_lambda": 0.5,
        "gate_betas": {"b0": -0.8, "b1": 1.0, "b2": 0.6, "b3": 0.0},
        "threshold": 0.5,
        "warm_tiers": [],
    }
    u_t_factor = {"descriptor": "direction", "channel": "state", "past": 2, "future": 1}
    out = apply_params(_base_cfg_dict(), params, u_t_factor=u_t_factor)
    judge = out["checkpoints"]["cp1"]["judge"]

    assert judge["u_t_factor"] == u_t_factor
    assert judge["gate_betas"]["b2"] == 0.6
    _roundtrip_ok(out, tmp_path, "calibrated_b2nonzero.yaml")


def test_apply_params_b2_nonzero_without_u_t_factor_raises():
    params = {
        "margin_lambda": 0.5,
        "gate_betas": {"b0": -0.8, "b1": 1.0, "b2": 0.6, "b3": 0.0},
        "threshold": 0.5,
        "warm_tiers": [],
    }
    with pytest.raises(SystemExit):
        apply_params(_base_cfg_dict(), params, u_t_factor=None)


# ---------------------------------------------------------------------------
# analyze_phase5_rollout.compute_metrics / sr_from_episodes
# ---------------------------------------------------------------------------


def test_compute_metrics_exact_numerators_and_denominators():
    rows = [
        {"phase": "eval", "hit_type": "FULL_HIT", "success": True},    # safe, labeled ok
        {"phase": "eval", "hit_type": "FULL_HIT", "success": False},   # bad hit
        {"phase": "eval", "hit_type": "MISS", "success": True},        # safe + miss_safe (FFR num)
        {"phase": "eval", "hit_type": "WARM_START", "success": True},  # safe, warm
        {"phase": "eval", "hit_type": "MISS", "success": False},       # not safe
        {"phase": "eval", "hit_type": "FULL_HIT", "success": None},    # incomplete
        {"phase": "warmup", "hit_type": "FULL_HIT", "success": True},  # non-eval -> ignored
    ]
    m = compute_metrics(rows, c_warm=0.75)

    assert m["n_verdicts"] == 6
    assert m["n_full_hit"] == 3 and m["n_warm_start"] == 1 and m["n_miss"] == 2
    assert m["n_incomplete"] == 1                       # success None excluded from BHR/FFR
    # BadHitRate = fh_bad / fh_labeled = 1 / 2 (None FULL_HIT not labeled).
    assert m["BHR_num"] == 1 and m["BHR_den"] == 2
    assert m["BHR"] == pytest.approx(0.5)
    # FFR = miss_safe / safe = 1 / 3.
    assert m["FFR_num"] == 1 and m["FFR_den"] == 3
    assert m["FFR"] == pytest.approx(1.0 / 3.0)
    # IR = (n_miss + c_warm * n_ws) / n = (2 + 0.75) / 6.
    assert m["IR"] == pytest.approx((2 + 0.75) / 6)
    assert m["zero_full_hit"] is False


def test_compute_metrics_zero_full_hit_flag():
    rows = [
        {"phase": "eval", "hit_type": "MISS", "success": True},
        {"phase": "eval", "hit_type": "WARM_START", "success": True},
    ]
    m = compute_metrics(rows, c_warm=0.75)
    assert m["BHR_den"] == 0
    assert m["BHR"] == 0.0
    assert m["zero_full_hit"] is True


def test_sr_from_episodes_excludes_unlabeled(tmp_path):
    import json

    p = tmp_path / "episodes.json"
    p.write_text(json.dumps([
        {"success": True}, {"success": False}, {"success": None}, {"success": True},
    ]), encoding="utf-8")
    sr = sr_from_episodes(str(p))
    assert sr["n_episodes"] == 3      # None excluded
    assert sr["n_success"] == 2
    assert sr["SR"] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# G2 R1 finding 2 — LOEO identity survives a re-collection timestamp
# ---------------------------------------------------------------------------


def test_parse_global_episode_id():
    assert _parse_global_episode_id("episode_0275_20260710_042540_118699") == 275
    assert _parse_global_episode_id("episode_1") == 1
    assert _parse_global_episode_id("traj_p0") is None
    assert _parse_global_episode_id(None) is None


def test_loeo_identity_survives_recollection_timestamp():
    """A D- entry with a timestamped trajectory_id and a re-collected episode with a
    DIFFERENT timestamp share the same global id -> the self entry IS excluded."""
    rng = np.random.default_rng(3)
    # Phase-4 D- entry: trajectory_id carries the ORIGINAL collection timestamp.
    self_entry = _mk_entry(
        entry_id="task_A:ep275:neg", trajectory_id="episode_0275_20260710_042540",
        task="task_A", outcome=-1, step_idx=0,
        vision=torch.from_numpy(rng.standard_normal(_EMB).astype(np.float32)),
        state=torch.from_numpy(rng.standard_normal(_STATE).astype(np.float32)),
        action=torch.from_numpy(rng.standard_normal((_CHUNK, _ACT)).astype(np.float32)),
        warm=False,
    )
    other = _linked_trajectory("task_A", "episode_0300_20260710_050000", +1, 1, rng)
    all_entries = [self_entry] + other

    # timestamp-independent identity: (task_key, global episode id 275).
    assert _entry_identity(self_entry) == ("task_A", 275)
    # A Phase-5 re-collection of the same init gets a NEW timestamp -> still id 275.
    identity = ("task_A", 275)

    storage = _temp_storage_excluding(
        all_entries, vector_dims={"vision_0": _EMB, "robot_state": _STATE},
        library_stats=None, identity=identity,
    )
    # The timestamped self entry is gone (would leak under the old stem identity).
    with pytest.raises(KeyError):
        storage.fetch_entry("task_A:ep275:neg")
    assert storage.count() == 1  # only the unrelated episode_0300 survives


# ---------------------------------------------------------------------------
# G2 R1 finding 1 — reference config accepts an injected u_t factor
# ---------------------------------------------------------------------------


def test_reference_config_accepts_injected_u_t_factor():
    """Pass 2 injects a u_t_factor into the reference active YAML; it must validate.

    The reference config alone carries no u_t_factor (b2=0, Phase 3 shape), so
    without the injection every replayed row's u_t is None and the calibrated b2
    is a no-op (the failure this guards)."""
    cfg = load_cache_config(_ACTIVE_YAML)
    assert cfg.checkpoints["cp1"].judge.u_t_factor is None  # reference has none
    cfg.checkpoints["cp1"].judge.u_t_factor = {
        "descriptor": "direction", "channel": "state", "past": 2, "future": 1,
    }
    validate_cache_config(cfg)  # injected factor valid on the in_memory dual config


# ---------------------------------------------------------------------------
# G2 R1 finding 3 — analyzer enforces paired I_val / same-seed provenance
# ---------------------------------------------------------------------------


def _ep(task, init, *, success=True, seed=7):
    """Episode-results row shape (main.py --save-episode-results): init_state_idx + seed."""
    return {"task_id": task, "init_state_idx": init, "success": success, "seed": seed}


def _step(task, init, ht, success):
    """Per-step recorder row shape (main.py _rec): subset_init_state_idx, NO seed."""
    return {"phase": "eval", "task_id": task, "subset_init_state_idx": init,
            "hit_type": ht, "success": success}


def test_analyzer_accepts_paired_i_val():
    # Same odd-init I_val set + same seed across all three runs -> OK.
    eps = [_ep(0, 1), _ep(0, 3)]
    steps = [_step(0, 1, "FULL_HIT", True), _step(0, 3, "MISS", True)]
    keys = check_paired_provenance(eps, eps, eps, illus_steps=steps, calib_steps=steps)
    assert keys == {(0, 1), (0, 3)}


def test_analyzer_accepts_production_per_step_schema():
    """Per-step JSONL uses subset_init_state_idx (not init_state_idx); analyzer must
    consume it via the canonical key rather than exiting 'missing init' (G2 R2 #1)."""
    eps = [_ep(0, 1), _ep(0, 3)]
    steps = [_step(0, 1, "FULL_HIT", True), _step(0, 3, "MISS", True)]
    assert "init_state_idx" not in steps[0] and "subset_init_state_idx" in steps[0]
    keys = check_paired_provenance(eps, eps, eps, illus_steps=steps, calib_steps=steps)
    assert keys == {(0, 1), (0, 3)}


def test_analyzer_rejects_unpaired_provenance():
    base = [_ep(0, 1), _ep(0, 3)]
    other = [_ep(0, 1), _ep(0, 5)]  # different init set
    steps = [_step(0, 1, "MISS", True), _step(0, 3, "MISS", True)]
    with pytest.raises(SystemExit, match="paired-provenance mismatch"):
        check_paired_provenance(base, other, base, illus_steps=steps, calib_steps=steps)


def test_analyzer_rejects_missing_per_step_episode():
    """A truncated per-step log covering a strict subset of I_val must fail loud
    (subset != full set), not silently bias BHR/FFR/IR (G2 R2 #2)."""
    eps = [_ep(0, 1), _ep(0, 3)]
    full = [_step(0, 1, "MISS", True), _step(0, 3, "MISS", True)]
    truncated = [_step(0, 1, "MISS", True)]  # missing episode (0, 3)
    with pytest.raises(SystemExit, match="full paired I_val set"):
        check_paired_provenance(eps, eps, eps, illus_steps=full, calib_steps=truncated)


def test_analyzer_rejects_even_init_non_i_val():
    eps = [_ep(0, 1), _ep(0, 2)]  # init 2 is even -> not I_val
    steps = [_step(0, 1, "MISS", True), _step(0, 2, "MISS", True)]
    with pytest.raises(SystemExit, match="I_val"):
        check_paired_provenance(eps, eps, eps, illus_steps=steps, calib_steps=steps)


def test_analyzer_rejects_seed_mismatch():
    base = [_ep(0, 1, seed=7)]
    calib = [_ep(0, 1, seed=9)]  # different seed
    steps = [_step(0, 1, "MISS", True)]
    with pytest.raises(SystemExit, match="seed mismatch"):
        check_paired_provenance(base, base, calib, illus_steps=steps, calib_steps=steps)


def test_analyzer_rejects_unverifiable_missing_seed():
    """All three episode-results lacking a seed must fail loud, not default-true
    the same-seed claim (G2 R2 #3)."""
    eps = [{"task_id": 0, "init_state_idx": 1, "success": True}]  # no seed
    steps = [_step(0, 1, "MISS", True)]
    with pytest.raises(SystemExit, match="missing 'seed'"):
        check_paired_provenance(eps, eps, eps, illus_steps=steps, calib_steps=steps)


@pytest.mark.parametrize("dup_success", [True, False])
def test_analyzer_rejects_duplicate_episode_identity(dup_success):
    """Two episode-results rows for the SAME (task_id, init_idx) — same OR
    contradictory success — must fail loud; set-ification would otherwise hide the
    duplicate while SR still counts it, breaking the 1-to-1 paired sample (G2 R3)."""
    base = [_ep(0, 1), _ep(0, 3)]
    # illustrative repeats identity (0, 1) with the same or opposite success.
    illus = [_ep(0, 1, success=True), _ep(0, 1, success=dup_success), _ep(0, 3)]
    steps = [_step(0, 1, "MISS", True), _step(0, 3, "MISS", True)]
    with pytest.raises(SystemExit, match="duplicate episode identity"):
        check_paired_provenance(base, illus, base, illus_steps=steps, calib_steps=steps)


def test_analyzer_allows_per_step_multiple_rows_per_episode():
    """Per-step logs legitimately have many rows per episode (one per step) — the
    uniqueness rule applies to episode-results only, not per-step."""
    eps = [_ep(0, 1)]
    # 3 steps for the same episode (0, 1) — must NOT trip the uniqueness guard.
    steps = [_step(0, 1, "FULL_HIT", True), _step(0, 1, "MISS", True), _step(0, 1, "WARM_START", True)]
    keys = check_paired_provenance(eps, eps, eps, illus_steps=steps, calib_steps=steps)
    assert keys == {(0, 1)}
