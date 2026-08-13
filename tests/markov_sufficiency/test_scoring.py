"""Tests for production-parity scoring and the group-C normalizer contract.

The parity test is the gate that makes E1/E3 meaningful: if the offline scorer
ranks candidates differently from the server, nothing measured offline
transfers to the real retrieval path. It is stratified exactly as the plan
requires -- with the task filter on, across history depths, and with episode
prefix steps (where ancestors are missing) as their own layer.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

from exp.markov_sufficiency import _library, _scoring

REPO = pathlib.Path(__file__).resolve().parents[2]
LIBRARY = REPO / "exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl"
YAML = REPO / (
    "exp/weighted_sum/config/trajectory/libero_spatial/"
    "cp1_spatial_pool_16__grid3_vision_0@6_vision_1@44_robot_state@50__d3.yaml"
)

needs_artifacts = pytest.mark.skipif(
    not (LIBRARY.exists() and YAML.exists()),
    reason="requires the local cache artifact and eval yaml",
)


@pytest.fixture(scope="module")
def scorer():
    return _scoring.build_scorer(YAML)


@pytest.fixture(scope="module")
def library():
    return _library.load_library(LIBRARY)


# ------------------------------------------------------------------
# Scorer construction
# ------------------------------------------------------------------


@needs_artifacts
def test_scorer_reads_weights_from_the_yaml(scorer):
    weights = {f: w for f, w, _ in scorer.active_fields}
    assert weights == pytest.approx({"vision_0": 0.06, "vision_1": 0.44, "robot_state": 0.5})
    sim_types = {f: cfg.get("type") for f, _, cfg in scorer.active_fields}
    assert sim_types["robot_state"] == "l2"
    assert scorer.trajectory_depth == 3


@needs_artifacts
def test_score_batch_matches_scalar_path(scorer, library):
    entry = library.entries[10]
    cands = scorer.candidates(library.entries, entry.payload.task_key)[:40]
    batch = scorer.score_batch(entry.query_keys, cands, cache={})
    scalar = np.array([scorer.score(entry.query_keys, c) for c in cands])
    np.testing.assert_allclose(batch, scalar, rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------
# Production parity (plan §8)
# ------------------------------------------------------------------


def _production_top1(library_path, yaml_path, entry, candidates):
    """Top-1 id as the real InMemoryBackend would rank it."""
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.config import load_cache_config
    from openpi.cache.storage_types import QuerySpec

    import dataclasses

    cfg = load_cache_config(yaml_path)
    ss = cfg.checkpoints["cp1"].search_strategy
    backend = InMemoryBackend(cfg.backend.vector_dims)
    for cand in candidates:
        # Offline artifacts store numpy vectors; the backend's batched path
        # expects the torch tensors that load_artifact would have produced.
        tensorised = dataclasses.replace(
            cand,
            query_keys={k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in cand.query_keys.items()},
        )
        backend._entries[cand.id] = tensorised  # noqa: SLF001 - test-only direct load

    import dataclasses as _dc

    field_sim = {
        k: (v if isinstance(v, dict) else _dc.asdict(v)) for k, v in (ss.field_similarity or {}).items()
    }
    sn = ss.score_normalization
    # Production's KeyBuilder only emits the fields the yaml enables, so the
    # spec must carry exactly those -- handing over every stored field would
    # make the backend look for a normalizer that the yaml never defines.
    fusion_weights = {
        name: field_cfg.weight
        for name, field_cfg in (
            ("vision_0", cfg.keys.vision_0),
            ("vision_1", cfg.keys.vision_1),
            ("vision_2", cfg.keys.vision_2),
            ("prompt_emb", cfg.keys.prompt_emb),
            ("robot_state", cfg.keys.robot_state),
        )
        if field_cfg.enabled and field_cfg.weight > 0
    }
    spec = QuerySpec(
        query_keys={
            k: torch.as_tensor(np.asarray(v)) for k, v in entry.query_keys.items() if k in fusion_weights
        },
        top_k=1,
        fusion_method="weighted_score_sum",
        fusion_weights=fusion_weights,
        field_similarity=field_sim,
        score_normalization=None if sn is None else (sn if isinstance(sn, dict) else _dc.asdict(sn)),
    )
    results = backend.search(spec)
    return results[0].id if results else None


SUITES = {
    "libero_spatial": (
        REPO / "exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl",
        REPO / ("exp/weighted_sum/config/trajectory/libero_spatial/"
                "cp1_spatial_pool_16__grid3_vision_0@6_vision_1@44_robot_state@50__d3.yaml"),
    ),
    "libero_10": (
        REPO / "exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl",
        REPO / ("exp/weighted_sum/config/trajectory_weight_alloc/libero_10/"
                "cp1_spatial_pool_16__grid3_vision_0@56_vision_1@25_robot_state@18__d1.yaml"),
    ),
}

#: Plan §8 fixes the gate's sampling: 200 queries per suite per depth, with
#: episode-prefix steps as their own layer of at least 30.
PARITY_QUERIES = 200
PARITY_PREFIX_QUERIES = 30


@pytest.fixture(scope="module")
def parity_backends():
    """One loaded backend per suite -- load_artifact also does the tensor conversion."""
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.config import load_cache_config

    out = {}
    for suite, (lib_path, yaml_path) in SUITES.items():
        if not (lib_path.exists() and yaml_path.exists()):
            continue
        cfg = load_cache_config(yaml_path)
        backend = InMemoryBackend(cfg.backend.vector_dims)
        backend.load_artifact(str(lib_path))
        out[suite] = (backend, cfg, _library.load_library(lib_path), _scoring.build_scorer(yaml_path))
    return out


def _spec(cfg, entry, task_key, top_k):
    import dataclasses as _dc

    from openpi.cache.storage_types import QueryFilter, QuerySpec

    ss = cfg.checkpoints["cp1"].search_strategy
    field_sim = {k: (v if isinstance(v, dict) else _dc.asdict(v)) for k, v in (ss.field_similarity or {}).items()}
    sn = ss.score_normalization
    fusion_weights = {
        name: fc.weight
        for name, fc in (
            ("vision_0", cfg.keys.vision_0), ("vision_1", cfg.keys.vision_1), ("vision_2", cfg.keys.vision_2),
            ("prompt_emb", cfg.keys.prompt_emb), ("robot_state", cfg.keys.robot_state),
        )
        if fc.enabled and fc.weight > 0
    }
    return QuerySpec(
        query_keys={k: torch.as_tensor(np.asarray(v)) for k, v in entry.query_keys.items() if k in fusion_weights},
        top_k=top_k,
        filters=QueryFilter(task_key=task_key),
        fusion_method="weighted_score_sum",
        fusion_weights=fusion_weights,
        field_similarity=field_sim,
        score_normalization=None if sn is None else (sn if isinstance(sn, dict) else _dc.asdict(sn)),
    )


def _parity_over(entries, backend, cfg, lib, scorer, seed):
    """Compare the runner-up (rank 2) so the query's own entry cannot mask a mismatch."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(entries), size=min(PARITY_QUERIES, len(entries)), replace=False)
    agree = checked = 0
    for i in picks:
        entry = entries[int(i)]
        task_key = entry.payload.task_key
        results = backend.search(_spec(cfg, entry, task_key, top_k=2))
        if len(results) < 2 or results[0].id != entry.id:
            continue  # self-hit missing: not a comparable sample
        cands = [c for c in lib.entries if c.payload.task_key == task_key and c.id != entry.id]
        mine = cands[int(np.argmax(scorer.score_batch(entry.query_keys, cands, cache={})))].id
        checked += 1
        agree += int(mine == results[1].id)
    return agree, checked


@needs_artifacts
@pytest.mark.parametrize("suite", sorted(SUITES))
@pytest.mark.parametrize("depth", [1, 3, 5])
def test_top1_parity_at_plan_scale(parity_backends, suite, depth):
    """Offline ranking must equal production ranking, 200 queries per suite/depth."""
    if suite not in parity_backends:
        pytest.skip(f"{suite} artifacts not present")
    backend, cfg, lib, scorer = parity_backends[suite]
    entries = [e for e in lib.entries if e.step_idx is not None and e.step_idx >= depth]
    agree, checked = _parity_over(entries, backend, cfg, lib, scorer, seed=depth)
    assert checked >= 50, f"only {checked} comparable queries for {suite} d{depth}"
    assert agree == checked, f"parity {agree}/{checked} for {suite} at depth {depth}"


@needs_artifacts
@pytest.mark.parametrize("suite", sorted(SUITES))
def test_parity_on_episode_prefix_steps(parity_backends, suite):
    """Prefix steps (missing ancestors) form their own stratum, >= 30 samples."""
    if suite not in parity_backends:
        pytest.skip(f"{suite} artifacts not present")
    backend, cfg, lib, scorer = parity_backends[suite]
    prefix = [e for e in lib.entries if e.step_idx in (0, 1)]
    assert len(prefix) >= PARITY_PREFIX_QUERIES
    agree, checked = _parity_over(prefix, backend, cfg, lib, scorer, seed=99)
    assert checked >= PARITY_PREFIX_QUERIES
    assert agree == checked


@needs_artifacts
def test_task_filter_is_enforced(scorer, library):
    entry = library.entries[0]
    cands = scorer.candidates(library.entries, entry.payload.task_key)
    assert cands, "task filter dropped every candidate"
    assert all(c.payload.task_key == entry.payload.task_key for c in cands)
    assert len(cands) < len(library.entries)


# ------------------------------------------------------------------
# Trajectory scoring semantics
# ------------------------------------------------------------------


@needs_artifacts
def test_trajectory_scoring_degenerates_to_single_frame_at_depth_one(scorer, library):
    entry = library.entries[20]
    cand = library.entries[21]
    single = scorer.score(entry.query_keys, cand)
    traj = scorer.score_trajectory([entry.query_keys], [cand], [1.0])
    assert traj == pytest.approx(single)


@needs_artifacts
def test_missing_ancestor_scores_zero_not_skipped(scorer, library):
    """A missing ancestor contributes 0.0, so the weights are not renormalised."""
    entry = library.entries[20]
    cand = library.entries[21]
    with_missing = scorer.score_trajectory([entry.query_keys, entry.query_keys], [cand, None], [0.5, 0.5])
    assert with_missing == pytest.approx(0.5 * scorer.score(entry.query_keys, cand))


def test_score_trajectory_rejects_misaligned_inputs():
    s = _scoring.Scorer(active_fields=[], normalizers={}, yaml_path="", trajectory_depth=1, trajectory_weights=None)
    with pytest.raises(ValueError):
        s.score_trajectory([{}], [None, None], [1.0, 1.0])


# ------------------------------------------------------------------
# Group C: fold-safe difference normalizer
# ------------------------------------------------------------------


def test_fit_diff_normalizer_rejects_fold_leakage():
    with pytest.raises(ValueError, match="fold leakage"):
        _scoring.fit_diff_normalizer([0.1, 0.2], ["ep_a", "ep_b"], held_out_trajectory="ep_a")


def test_fit_diff_normalizer_is_fitted_on_library_side_only():
    norm = _scoring.fit_diff_normalizer([0.0, 1.0], ["ep_a"], held_out_trajectory="ep_held")
    assert norm.n_fit == 2
    assert norm.fit_trajectories == ("ep_a",)
    assert 0.0 <= norm(0.5) <= 1.0


def test_diff_features_flags_padding_at_episode_start():
    frames = [{"f": np.ones(4)}, None, None]
    delta, delta2, padding = _scoring.diff_features(frames, "f")
    assert padding is True
    assert not delta.any() and not delta2.any()

    frames = [{"f": np.array([2.0, 2.0])}, {"f": np.array([1.0, 1.0])}, {"f": np.array([0.0, 0.0])}]
    delta, delta2, padding = _scoring.diff_features(frames, "f")
    assert padding is False
    np.testing.assert_allclose(delta, [1.0, 1.0])
    np.testing.assert_allclose(delta2, [0.0, 0.0])
