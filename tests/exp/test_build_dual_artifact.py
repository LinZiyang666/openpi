"""Tests for TRACER Phase 4 D-/D+ merge build path.

Covers the shared builder ``--outcome-filter`` extension, the
``cp1_llm_layer_extract`` fail-loud guard, the HDF5 completeness gate, and the
``build_dual_artifact`` merge (outcome tagging + coverage + id uniqueness +
default-path non-regression). Everything is CPU-only over synthetic HDF5.

The enable_dual retrieval mechanics themselves (margin = s_pos - lambda*s_neg,
three-state gate, degenerate parity) are covered by the Phase 3 suite in
tests/cache/test_search_strategy.py; the end-to-end non-trivial margin on a real
merged artifact is covered by exp/zixuan_proposal/validate_dual_artifact.py.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest

from exp.common.build_in_memory_cache_artifact import build_artifact

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_episode(path, *, success, task="t", fill=1.0, n_steps=3, drop_step=None,
                   missing_dataset=None, missing_dataset_at_step=0, state_offset=0.0):
    """Write a multi-step synthetic collection HDF5.

    ``drop_step`` omits that step index (creates a hole). ``missing_dataset``
    omits that dataset from step ``missing_dataset_at_step`` (default 0) to model
    a corrupt step -- which may be AFTER step 0. ``state_offset`` shifts
    ``robot_state`` (default 0.0 == legacy constant) so a caller can give D+ vs D-
    distinguishable state distributions (used by the D+-only library_stats
    regression).
    """
    with h5py.File(path, "w") as f:
        f.attrs["success"] = success
        f.attrs["task"] = task
        for s in range(n_steps):
            if drop_step is not None and s == drop_step:
                continue
            g = f.create_group(f"step_{s:03d}")
            for name, shape, val in (
                ("vision_0", (256, 2048), fill),
                ("vision_1", (256, 2048), fill + 1),
                ("vision_2", (256, 2048), fill + 2),
                ("prompt_emb", (10, 2048), fill + 3),
            ):
                if missing_dataset == name and s == missing_dataset_at_step:
                    continue
                g.create_dataset(name, data=np.full(shape, val, dtype=np.float32))
            if not (missing_dataset == "robot_state" and s == missing_dataset_at_step):
                g.create_dataset("robot_state", data=(np.arange(32, dtype=np.float32) + state_offset))
            if not (missing_dataset == "clean_action" and s == missing_dataset_at_step):
                g.create_dataset("clean_action", data=np.ones((10, 32), dtype=np.float32))


# ---------------------------------------------------------------------------
# --outcome-filter on the pool builder
# ---------------------------------------------------------------------------


def test_outcome_filter_selects_right_episodes(tmp_path):
    _write_episode(tmp_path / "s1.h5", success=True, task="s1")
    _write_episode(tmp_path / "s2.h5", success=True, task="s2")
    _write_episode(tmp_path / "f1.h5", success=False, task="f1")

    succ = build_artifact(str(tmp_path), "cp1_mean_pool", workers=-1, outcome_filter="success")
    fail = build_artifact(str(tmp_path), "cp1_mean_pool", workers=-1, outcome_filter="failure")
    allb = build_artifact(str(tmp_path), "cp1_mean_pool", workers=-1, outcome_filter="all")

    succ_tasks = {e.payload.task_key for e in succ["entries"]}
    fail_tasks = {e.payload.task_key for e in fail["entries"]}
    all_tasks = {e.payload.task_key for e in allb["entries"]}

    assert succ_tasks == {"s1", "s2"}
    assert fail_tasks == {"f1"}
    assert all_tasks == {"s1", "s2", "f1"}
    # builder stays outcome-agnostic: it never sets .outcome (merge does)
    assert all(getattr(e, "outcome", None) is None for e in allb["entries"])


def test_outcome_filter_threads_through_pool_dispatch(tmp_path, monkeypatch):
    """Both the serial (workers=-1) and ProcessPool (workers>0) dispatch paths
    call ``_process_episode(str(p), *_ep_args)`` with the SAME ``_ep_args`` tuple,
    so outcome_filter reaches the worker on both. We assert the arg is threaded
    via a spy on the serial path -- a real forked pool deadlocks under pytest
    (torch-import + test-runner threads make fork() unsafe, which is exactly why
    build_artifact runs tests with workers=-1)."""
    import exp.common.build_in_memory_cache_artifact as bm

    _write_episode(tmp_path / "s.h5", success=True)
    seen = {}
    orig = bm._process_episode

    def spy(h5_path_str, *args, **kw):
        seen["args"] = args
        return orig(h5_path_str, *args, **kw)

    monkeypatch.setattr(bm, "_process_episode", spy)
    bm.build_artifact(str(tmp_path), "cp1_mean_pool", workers=-1, outcome_filter="all")

    # Assert by meaning rather than by index: the dispatch tuple grew a
    # trailing per-file trajectory_id, and pinning outcome_filter to args[-1]
    # would break on any future append.
    import inspect

    params = list(inspect.signature(orig).parameters)[1:]  # drop h5_path_str
    bound = dict(zip(params, seen["args"]))
    assert bound["outcome_filter"] == "all"
    # The per-file id rides the same dispatch; None keeps the historical stems.
    assert bound["trajectory_id"] is None


def test_outcome_filter_invalid_raises(tmp_path):
    _write_episode(tmp_path / "s.h5", success=True)
    with pytest.raises(ValueError, match="outcome_filter must be one of"):
        build_artifact(str(tmp_path), "cp1_mean_pool", workers=-1, outcome_filter="bogus")


def test_llm_layer_extract_nondefault_filter_fails_loud_cpu(tmp_path):
    """cp1_llm_layer_extract rejects a non-default filter BEFORE any model load,
    so the failure is CPU-only (no --checkpoint-dir needed to hit it)."""
    _write_episode(tmp_path / "s.h5", success=True)
    with pytest.raises(ValueError, match="only supported for pool builders"):
        build_artifact(str(tmp_path), "cp1_llm_layer_extract", workers=-1, outcome_filter="failure")


# ---------------------------------------------------------------------------
# default-path non-regression (field/value equivalence, NOT pickle bytes)
# ---------------------------------------------------------------------------


def test_default_success_filter_matches_legacy_default(tmp_path):
    """--outcome-filter default 'success' is behaviorally identical to the legacy
    call (no outcome_filter kwarg). Assert stable fields, not pickle bytes."""
    _write_episode(tmp_path / "ok.h5", success=True, task="ok")
    _write_episode(tmp_path / "bad.h5", success=False, task="bad")

    legacy = build_artifact(str(tmp_path), "cp1_mean_pool", workers=-1)
    explicit = build_artifact(str(tmp_path), "cp1_mean_pool", workers=-1, outcome_filter="success")

    assert [e.id for e in legacy["entries"]] == [e.id for e in explicit["entries"]]
    assert legacy["vector_dims"] == explicit["vector_dims"]
    assert legacy["key_builder_type"] == explicit["key_builder_type"]
    for a, b in zip(legacy["entries"], explicit["entries"]):
        assert a.payload.task_key == b.payload.task_key
        assert getattr(a, "outcome", None) is None and getattr(b, "outcome", None) is None
        for k in a.query_keys:
            assert np.array_equal(a.query_keys[k], b.query_keys[k]), k


# ---------------------------------------------------------------------------
# D- HDF5 completeness gate
# ---------------------------------------------------------------------------


def test_completeness_gate_accepts_clean(tmp_path):
    from exp.zixuan_proposal.build_dual_artifact import check_failure_h5_completeness

    _write_episode(tmp_path / "f1.h5", success=False, n_steps=4)
    _write_episode(tmp_path / "f2.h5", success=False, n_steps=3)
    paths = check_failure_h5_completeness(str(tmp_path), min_steps=2)
    assert len(paths) == 2


def test_completeness_gate_rejects_hole(tmp_path):
    from exp.zixuan_proposal.build_dual_artifact import check_failure_h5_completeness

    _write_episode(tmp_path / "hole.h5", success=False, n_steps=4, drop_step=2)
    with pytest.raises(SystemExit, match="completeness check FAILED"):
        check_failure_h5_completeness(str(tmp_path), min_steps=2)


def test_completeness_gate_rejects_stub(tmp_path):
    from exp.zixuan_proposal.build_dual_artifact import check_failure_h5_completeness

    _write_episode(tmp_path / "stub.h5", success=False, n_steps=1)
    with pytest.raises(SystemExit, match="completeness check FAILED"):
        check_failure_h5_completeness(str(tmp_path), min_steps=2)


def test_completeness_gate_rejects_missing_dataset(tmp_path):
    from exp.zixuan_proposal.build_dual_artifact import check_failure_h5_completeness

    # G2 R1 regression: corruption AFTER step 0 (step 1), which a step-0-only
    # check would falsely accept into D-.
    _write_episode(tmp_path / "corrupt.h5", success=False, n_steps=3,
                   missing_dataset="prompt_emb", missing_dataset_at_step=1)
    with pytest.raises(SystemExit, match="completeness check FAILED"):
        check_failure_h5_completeness(str(tmp_path), min_steps=2)


# ---------------------------------------------------------------------------
# build_dual_artifact merge
# ---------------------------------------------------------------------------


def _make_pos_artifact(pos_dir, out_pkl, library_stats=None):
    """Build a tiny success-only D+ artifact and pickle it (mimics a shipped pkl)."""
    art = build_artifact(str(pos_dir), "cp1_mean_pool", workers=-1, outcome_filter="success")
    # A shipped D+ carries a library_stats; the merge must preserve it (D+-only).
    art["library_stats"] = library_stats
    with open(out_pkl, "wb") as fh:
        pickle.dump(art, fh)
    return art


def test_build_dual_artifact_merge_coverage_uniqueness(tmp_path):
    from exp.zixuan_proposal.build_dual_artifact import build_dual_artifact

    # D+ source: 2 success episodes (distinct vision fill)
    pos_dir = tmp_path / "pos"
    pos_dir.mkdir()
    _write_episode(pos_dir / "s1.h5", success=True, task="s1", fill=1.0, n_steps=3)
    _write_episode(pos_dir / "s2.h5", success=True, task="s2", fill=1.5, n_steps=3)
    pos_pkl = tmp_path / "pos.pkl"
    pos_art = _make_pos_artifact(pos_dir, pos_pkl)

    # D- source: 2 failure episodes (distinct vision fill)
    neg_dir = tmp_path / "neg"
    neg_dir.mkdir()
    _write_episode(neg_dir / "f1.h5", success=False, task="f1", fill=9.0, n_steps=3)
    _write_episode(neg_dir / "f2.h5", success=False, task="f2", fill=9.5, n_steps=4)

    out_pkl = tmp_path / "dual.pkl"
    merged = build_dual_artifact(
        str(pos_pkl), str(neg_dir), "cp1_mean_pool", str(out_pkl), workers=-1, min_steps=2,
    )

    n_pos = sum(1 for e in merged["entries"] if e.outcome == 1)
    n_neg = sum(1 for e in merged["entries"] if e.outcome == -1)
    n_none = sum(1 for e in merged["entries"] if e.outcome is None)

    assert n_pos == len(pos_art["entries"]) > 0
    assert n_neg > 0
    assert n_none == 0  # every D+ entry MUST be tagged +1 (else vanishes under enable_dual)
    assert merged["vector_dims"] == pos_art["vector_dims"]
    ids = [e.id for e in merged["entries"]]
    assert len(ids) == len(set(ids))  # global id uniqueness

    # persisted artifact round-trips with the same coverage
    with open(out_pkl, "rb") as fh:
        reloaded = pickle.load(fh)
    assert sum(1 for e in reloaded["entries"] if e.outcome == 1) == n_pos
    assert sum(1 for e in reloaded["entries"] if e.outcome == -1) == n_neg


def test_build_dual_artifact_computes_dplus_only_stats_when_source_lacks_them(tmp_path):
    """Regression (G2 R2): when the D+ source artifact carries NO library_stats
    (the real libero_10/cp1_mean_pool.pkl case), the merge MUST compute D+-only
    stats and never leave None. A None library_stats makes the in_memory backend
    fallback-recompute over ALL loaded entries (D+ ∪ D-), folding D- failure steps
    into Phase 5 u_t normalization -- the exact D+-only contract this merge protects."""
    import torch

    from exp.zixuan_proposal.build_dual_artifact import build_dual_artifact
    from openpi.cache.components.factors.base import LibraryStats

    # D+ source: 2 success episodes, DEFAULT robot_state (offset 0)
    pos_dir = tmp_path / "pos"
    pos_dir.mkdir()
    _write_episode(pos_dir / "s1.h5", success=True, task="s1", fill=1.0, n_steps=3)
    _write_episode(pos_dir / "s2.h5", success=True, task="s2", fill=1.5, n_steps=3)
    pos_pkl = tmp_path / "pos.pkl"
    # library_stats=None mimics the libero_10 D+ source (key absent / None both
    # take the same `is None` branch the fix guards).
    _make_pos_artifact(pos_dir, pos_pkl, library_stats=None)

    # D- source: 2 failure episodes with a DIFFERENT robot_state distribution
    # (offset 100) so D+-only vs D+∪D- stats are provably distinguishable.
    neg_dir = tmp_path / "neg"
    neg_dir.mkdir()
    _write_episode(neg_dir / "f1.h5", success=False, task="f1", fill=9.0, n_steps=3, state_offset=100.0)
    _write_episode(neg_dir / "f2.h5", success=False, task="f2", fill=9.5, n_steps=4, state_offset=100.0)

    out_pkl = tmp_path / "dual.pkl"
    merged = build_dual_artifact(
        str(pos_pkl), str(neg_dir), "cp1_mean_pool", str(out_pkl), workers=-1, min_steps=2,
    )

    ls = merged["library_stats"]
    assert ls is not None, "merged library_stats must not be None (backend would recompute over D+ ∪ D-)"
    assert isinstance(ls, LibraryStats)

    pos_entries = [e for e in merged["entries"] if e.outcome == 1]
    all_entries = merged["entries"]
    d_plus_only = LibraryStats.compute_from_entries(pos_entries)
    over_all = LibraryStats.compute_from_entries(all_entries)

    # merged stats are computed from the D+ entries ONLY ...
    assert torch.allclose(ls.state_sigma, d_plus_only.state_sigma)
    # ... and provably NOT the contaminated D+∪D- stats (D- has offset-100 state).
    assert not torch.allclose(ls.state_sigma, over_all.state_sigma), \
        "library_stats must be D+-only, not folded with D- failure steps"

    # persisted artifact keeps the D+-only stats (round-trips non-None)
    with open(out_pkl, "rb") as fh:
        reloaded = pickle.load(fh)
    assert reloaded["library_stats"] is not None


def test_build_dual_artifact_empty_failure_pool_raises(tmp_path):
    from exp.zixuan_proposal.build_dual_artifact import build_dual_artifact

    pos_dir = tmp_path / "pos"
    pos_dir.mkdir()
    _write_episode(pos_dir / "s1.h5", success=True, task="s1")
    pos_pkl = tmp_path / "pos.pkl"
    _make_pos_artifact(pos_dir, pos_pkl)

    # neg dir has only a SUCCESS episode -> outcome_filter=failure yields 0 -> raise
    neg_dir = tmp_path / "neg"
    neg_dir.mkdir()
    _write_episode(neg_dir / "s.h5", success=True, task="s", n_steps=3)

    with pytest.raises(SystemExit):
        build_dual_artifact(str(pos_pkl), str(neg_dir), "cp1_mean_pool", str(tmp_path / "d.pkl"),
                            workers=-1, min_steps=2)


def test_outcome_filter_real_processpool_via_cli(tmp_path):
    """Real workers>0 ProcessPool dispatch, exercised through the CLI in a FRESH
    process. pytest's own threads make an in-process fork() pool deadlock, so a
    clean subprocess (mirroring the production build) proves outcome_filter
    actually reaches the pool workers -- not only the serial spy path."""
    import subprocess
    import sys

    _write_episode(tmp_path / "s1.h5", success=True, task="s1")
    _write_episode(tmp_path / "f1.h5", success=False, task="f1")
    out = tmp_path / "art.pkl"
    r = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "exp/common/build_in_memory_cache_artifact.py"),
         "--data-dir", str(tmp_path), "--builder-type", "cp1_mean_pool",
         "--output", str(out), "--workers", "2", "--outcome-filter", "failure"],
        cwd=str(_REPO_ROOT), env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    with open(out, "rb") as fh:
        art = pickle.load(fh)
    assert {e.payload.task_key for e in art["entries"]} == {"f1"}  # only failure, via the real pool


def test_dual_strategy_searches_merged_artifact(tmp_path):
    """A tiny merged D+/D- artifact loads through the production config path and
    DualRetrievalKnnStrategy(enable_dual=True) returns a non-zero s_neg for a query
    near a D- entry -- proving D- is actually searched (non-degenerate)."""
    import torch
    import yaml

    from exp.zixuan_proposal.build_dual_artifact import build_dual_artifact
    from openpi.cache.components.search_strategy import SearchContext
    from openpi.cache.config import build_cache_components, load_cache_config
    from openpi.cache.types import CheckpointID

    pos_dir = tmp_path / "pos"
    pos_dir.mkdir()
    _write_episode(pos_dir / "s1.h5", success=True, task="s1", fill=1.0, n_steps=3)
    pos_pkl = tmp_path / "pos.pkl"
    _make_pos_artifact(pos_dir, pos_pkl)
    neg_dir = tmp_path / "neg"
    neg_dir.mkdir()
    _write_episode(neg_dir / "f1.h5", success=False, task="f1", fill=9.0, n_steps=3)
    merged_pkl = tmp_path / "dual.pkl"
    merged = build_dual_artifact(str(pos_pkl), str(neg_dir), "cp1_mean_pool",
                                 str(merged_pkl), workers=-1, min_steps=2)

    # reuse the shipped active config, repoint preload at the tiny merged artifact
    base = yaml.safe_load((_REPO_ROOT / "exp/zixuan_proposal/config/dual_retrieval_active.yaml").read_text())
    base["backend"]["in_memory"]["preload_path"] = str(merged_pkl)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(base))

    comps = build_cache_components(load_cache_config(str(cfg_path)))
    strat = comps["search_strategies"][CheckpointID.CP1]

    neg_entry = next(e for e in merged["entries"] if e.outcome == -1)
    qk = {k: (torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v)
          for k, v in neg_entry.query_keys.items()}
    strat.search(SearchContext(query_keys=qk, checkpoint_id=CheckpointID.CP1, current_step=0))
    sig = strat.last_retrieval_signals()
    assert sig.s_neg > 0.0                 # D- pool actively searched
    assert sig.margin <= sig.s_pos + 1e-9  # margin = s_pos - lambda*s_neg


def test_build_dual_artifact_preserves_dplus_library_stats(tmp_path):
    """The merge reuses the D+ library_stats verbatim (D+-only) -- it never Nones
    it out nor recomputes over D- (which would contaminate the Phase-5 u_t norm)."""
    from exp.zixuan_proposal.build_dual_artifact import build_dual_artifact

    pos_dir = tmp_path / "pos"
    pos_dir.mkdir()
    _write_episode(pos_dir / "s1.h5", success=True, task="s1")
    pos_pkl = tmp_path / "pos.pkl"
    _make_pos_artifact(pos_dir, pos_pkl, library_stats="D+ONLY_SENTINEL")
    neg_dir = tmp_path / "neg"
    neg_dir.mkdir()
    _write_episode(neg_dir / "f1.h5", success=False, task="f1", n_steps=3)

    merged = build_dual_artifact(str(pos_pkl), str(neg_dir), "cp1_mean_pool",
                                 str(tmp_path / "d.pkl"), workers=-1, min_steps=2)
    assert merged["library_stats"] == "D+ONLY_SENTINEL"
