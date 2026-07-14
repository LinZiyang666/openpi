"""Integration + adversarial coverage for the Phase-6 build/serve chain (plan §6.3).

Exercises the WHOLE enforceable path end to end -- build -> filtered artifact + D- manifest +
frozen-total validation + immutable digest -> serve-time binding + odd-init rejection + a REAL
``DualRetrievalKnnStrategy`` top-K prefix -- plus the rejection probes the reviewer required
(non-finite gate, D+-only/zero-D- artifact, wrong frozen step total, post-build weight swap,
served odd-init row).
"""

import pathlib

import pytest
import torch
import yaml

from exp.zixuan_proposal.phase6_assemble import (
    DMINUS_TOTALS,
    assemble_projected_artifact,
    build_dminus_manifest,
    resolve_dminus_ident,
    serve_init,
    strategy_topk_prefix_ok,
    validate_dminus_manifest,
)
from exp.zixuan_proposal.phase6_emit import emit_projection_config

_REPO = pathlib.Path(__file__).resolve().parents[2]
_TEMPLATE = _REPO / "exp/zixuan_proposal/config/dual_retrieval_projection_l10.yaml"

_SUITE = "libero_spatial"  # frozen totals: 18 episodes / 792 steps
_STEPS_EACH = 44  # 18 * 44 == 792


# ------------------------------------------------------------------
# Frozen 18-episode / 792-step spatial D- fixture
# ------------------------------------------------------------------
def _dminus_gids():
    # 9 tasks x 2 inits (one even=2, one odd=3) -> 18 distinct (task, init) cells.
    return [t * 50 + init for t in range(9) for init in (2, 3)]


def _prov_rows():
    return [{"h5_basename": f"fail_{g}.h5", "gid": g} for g in _dminus_gids()]


def _prov_table():
    return {(r["h5_basename"], *resolve_dminus_ident(r["gid"])) for r in _prov_rows()}


def _good_stat(base):
    return {"exists": True, "success": False, "n_steps": _STEPS_EACH, "complete": True}


def _manifest():
    return build_dminus_manifest(_prov_rows(), _good_stat)


# ------------------------------------------------------------------
# D- manifest unit + rejection (frozen totals)
# ------------------------------------------------------------------
def test_dminus_manifest_validates_against_frozen_totals():
    m = _manifest()
    assert validate_dminus_manifest(m, _prov_table(), suite=_SUITE) is m
    assert len(m) == DMINUS_TOTALS[_SUITE]["episodes"] == 18
    assert sum(x["n_steps"] for x in m) == DMINUS_TOTALS[_SUITE]["steps"] == 792


def test_dminus_manifest_rejects_wrong_episode_count():
    m = _manifest()[:-1]  # 17 episodes
    with pytest.raises(ValueError, match="frozen libero_spatial total is 18"):
        validate_dminus_manifest(m, _prov_table(), suite=_SUITE)


def test_dminus_manifest_rejects_wrong_step_total():
    m = _manifest()
    m[0] = {**m[0], "n_steps": _STEPS_EACH + 10}  # 802 total, not 792
    with pytest.raises(ValueError, match="step total 802 != frozen"):
        validate_dminus_manifest(m, _prov_table(), suite=_SUITE)


def test_dminus_manifest_rejects_unknown_suite():
    with pytest.raises(ValueError, match="unknown suite"):
        validate_dminus_manifest(_manifest(), _prov_table(), suite="not_a_suite")


def test_dminus_manifest_rejects_incomplete_h5():
    def bad_stat(base):
        return {"exists": True, "success": False, "n_steps": _STEPS_EACH, "complete": base != "fail_2.h5"}

    with pytest.raises(ValueError, match="incomplete"):
        validate_dminus_manifest(build_dminus_manifest(_prov_rows(), bad_stat), _prov_table(), suite=_SUITE)


def test_dminus_manifest_rejects_non_failure():
    def success_stat(base):
        return {"exists": True, "success": base == "fail_2.h5", "n_steps": _STEPS_EACH, "complete": True}

    with pytest.raises(ValueError, match="non-failure"):
        validate_dminus_manifest(build_dminus_manifest(_prov_rows(), success_stat), _prov_table(), suite=_SUITE)


# ------------------------------------------------------------------
# Full build -> emit -> serve integrated path (with REAL D- rows)
# ------------------------------------------------------------------
def _e(ident, outcome, *, h5=None, step=None):
    return {"ident": ident, "outcome": outcome, "h5": h5, "step": step}


def _ident(e):
    return e["ident"]


def _outcome(e):
    return e["outcome"]


def _source(e):
    return (e["h5"], e["step"])


def _dminus_rows(manifest, *, rows_per=_STEPS_EACH, step_fn=None):
    """Per-step D- rows in the REAL builder shape: source recorded as the trajectory_id STEM
    (``Path(h5_basename).stem``, no ``.h5``) as ``build_in_memory_cache_artifact.py`` does, while
    the manifest keeps the ``.h5`` basename -- so the positive path exercises the production
    stem/basename incompatibility rather than copying the manifest token."""
    rows = []
    for m in manifest:
        ident = (m["task_id"], m["init_state_idx"])
        stem = pathlib.Path(m["h5_basename"]).stem
        for s in range(rows_per):
            step = step_fn(s) if step_fn else s
            rows.append(_e(ident, -1, h5=stem, step=step))
    return rows


def _projected_artifact(weights_path, *, dminus_rows):
    entries = [
        _e((0, 0), 1),  # D+ even -> kept
        _e((0, 2), 1),  # D+ even -> kept
        _e((0, 1), 1),  # D+ odd  -> dropped by I_cal filter
        _e(None, 1),    # D+ unresolved -> counted, dropped
    ]
    entries += dminus_rows
    return {"entries": entries, "projection_params": {"projection_weights_path": str(weights_path)}}


def test_build_emit_serve_integrated_path(tmp_path):
    w = tmp_path / "laneB.pt"
    w.write_bytes(b"trained-projection-laneB")
    manifest = _manifest()
    art = _projected_artifact(w, dminus_rows=_dminus_rows(manifest))  # full 792 rows w/ provenance

    assembled = assemble_projected_artifact(
        art, _ident, w, dminus_manifest=manifest, provenance_table=_prov_table(),
        suite=_SUITE, outcome_fn=_outcome, source_fn=_source,
    )
    # D+ odd + unresolved gone; even D+ and even D- (init 2) kept, odd D- (init 3) dropped.
    kept_idents = {e["ident"] for e in assembled["entries"]}
    assert (0, 0) in kept_idents and (0, 2) in kept_idents
    assert (0, 1) not in kept_idents and None not in kept_idents
    assert all(i[1] % 2 == 0 for i in kept_idents)  # served library is pure I_cal
    assert assembled["projection_params"]["projection_weights_sha256"]
    assert assembled["dminus_manifest"] is manifest
    assert "projection_weights_sha256" not in art["projection_params"]  # caller dict untouched

    out = emit_projection_config(
        _TEMPLATE.read_text(),
        weights_path=str(w),
        preload_path=str(tmp_path / "projB_ical_dual.pkl"),
        normalizers={
            "vision_0": {"mu": 0.5, "sigma": 0.1},
            "vision_1": {"mu": 0.4, "sigma": 0.2},
            "robot_state": {"mu": -1.0, "sigma": 0.9},
        },
        betas={"b0": -0.3, "b3": 0.7},
    )
    yaml_cfg = yaml.safe_load(out)
    assert "__FILL_AT_EXECUTION__" not in out

    digest = serve_init(assembled, yaml_cfg, w, _ident)
    assert digest == assembled["projection_params"]["projection_weights_sha256"]


def _assemble(art, w, manifest):
    return assemble_projected_artifact(
        art, _ident, w, dminus_manifest=manifest, provenance_table=_prov_table(),
        suite=_SUITE, outcome_fn=_outcome, source_fn=_source,
    )


def test_assemble_rejects_artifact_with_no_dminus_rows(tmp_path):
    # A D+-only artifact (zero D- rows) must NOT be certified by a well-formed 18-episode manifest.
    w = tmp_path / "laneB.pt"
    w.write_bytes(b"trained")
    art = _projected_artifact(w, dminus_rows=[])  # no D- rows
    with pytest.raises(ValueError, match="source H5 basenames != manifest"):
        _assemble(art, w, _manifest())


def test_assemble_rejects_truncated_dminus_rows(tmp_path):
    # 18 D- rows (ONE per identity) must NOT be stamped as a complete 792-step product when the
    # manifest claims 44 steps each -- the step indices are bound to the contiguous 0..43 set.
    w = tmp_path / "laneB.pt"
    w.write_bytes(b"trained")
    manifest = _manifest()
    art = _projected_artifact(w, dminus_rows=_dminus_rows(manifest, rows_per=1))  # 18 rows, step_idx {0}
    with pytest.raises(ValueError, match="step indices are not the contiguous"):
        _assemble(art, w, manifest)


def test_assemble_rejects_wrong_source_dump(tmp_path):
    # Same identities + full 792 rows, but rows sourced from a DIFFERENT dump (trajectory_id /
    # h5_basename disagrees) must be rejected -- the exact §6.3b "equal totals can't distinguish
    # datasets" contamination guard. Stem normalization must NOT mask a genuinely different stem.
    w = tmp_path / "laneB.pt"
    w.write_bytes(b"trained")
    manifest = _manifest()
    rows = _dminus_rows(manifest)
    for r in rows:
        r["h5"] = "otherdump_" + r["h5"]  # right identity + count, wrong source dump
    art = _projected_artifact(w, dminus_rows=rows)
    with pytest.raises(ValueError, match="source H5 basenames != manifest"):
        _assemble(art, w, manifest)


def test_assemble_rejects_duplicate_step_idx(tmp_path):
    # Correct total row count but duplicate/non-contiguous step indices (all 0) -> rejected.
    w = tmp_path / "laneB.pt"
    w.write_bytes(b"trained")
    manifest = _manifest()
    art = _projected_artifact(w, dminus_rows=_dminus_rows(manifest, step_fn=lambda s: 0))
    with pytest.raises(ValueError, match="step indices are not the contiguous"):
        _assemble(art, w, manifest)


def test_serve_rejects_served_odd_init_row(tmp_path):
    # An assembled artifact whose served set somehow contains an odd-init row must ABORT serve
    # (the row is not silently filtered).
    w = tmp_path / "laneB.pt"
    w.write_bytes(b"trained")
    manifest = _manifest()
    assembled = _assemble(_projected_artifact(w, dminus_rows=_dminus_rows(manifest)), w, manifest)
    assembled["entries"].append(_e((3, 7), 1))  # inject an odd-init row post-assembly
    yaml_cfg = {"key_builder": {"type": "projection", "projection": {"weights_path": str(w)}}}
    with pytest.raises(ValueError, match="not pure I_cal"):
        serve_init(assembled, yaml_cfg, w, _ident)


def test_serve_aborts_on_post_build_weight_swap(tmp_path):
    w = tmp_path / "laneB.pt"
    w.write_bytes(b"trained-v1")
    manifest = _manifest()
    assembled = _assemble(_projected_artifact(w, dminus_rows=_dminus_rows(manifest)), w, manifest)
    yaml_cfg = {"key_builder": {"type": "projection", "projection": {"weights_path": str(w)}}}
    w.write_bytes(b"swapped-v2")
    with pytest.raises(ValueError, match="digest drift"):
        serve_init(assembled, yaml_cfg, w, _ident)


# ------------------------------------------------------------------
# REAL-strategy Retrieval@K top-1-prefix (§C) -- not a re-sort of a score list
# ------------------------------------------------------------------
def _frozen_strategy_kwargs():
    """ONE frozen production-lane config (fusion/normalizers/filters/LOEO/depth); the diagnostic
    clones it changing only top_k, so any `top_k` here is ignored."""
    from openpi.cache.components.search_strategy import ConstantDepthPolicy

    return {
        "base_fusion": "weighted_score_sum",
        "depth_policy": ConstantDepthPolicy(1),
        "allowed_depths": [1],
        "fusion_weights": {"robot_state": 1.0},
    }


def _populated_storage(n=6):
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID

    storage = CacheStorage(InMemoryBackend({"robot_state": 8}))
    for i in range(n):
        v = torch.zeros(8)
        v[i] = 1.0
        storage.insert(
            CacheEntry(
                id=f"e{i}",
                checkpoint_id=CheckpointID.CP1,
                query_keys={"robot_state": v},
                payload=CachePayload(action_chunk=torch.randn(50, 32)),
            )
        )
    return storage


def test_real_strategy_topk_prefix_holds():
    from openpi.cache.components.search_strategy import SearchContext
    from openpi.cache.types import CheckpointID

    storage = _populated_storage()
    q = torch.zeros(8)
    q[0] = 0.9  # closest to e0
    ctx = SearchContext(query_keys={"robot_state": q}, checkpoint_id=CheckpointID.CP1)
    # ONE frozen config cloned with only top_k changed (proves K=1/K=5 identical elsewhere).
    assert strategy_topk_prefix_ok(storage, _frozen_strategy_kwargs(), ctx, production_k=1, eval_k=5)


def test_real_strategy_topk_prefix_false_on_empty_backend():
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.components.search_strategy import SearchContext
    from openpi.cache.types import CheckpointID

    storage = CacheStorage(InMemoryBackend({"robot_state": 8}))  # empty
    ctx = SearchContext(query_keys={"robot_state": torch.ones(8)}, checkpoint_id=CheckpointID.CP1)
    assert not strategy_topk_prefix_ok(storage, _frozen_strategy_kwargs(), ctx)  # no results -> False
