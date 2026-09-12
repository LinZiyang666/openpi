"""Probe realistic CP1 dimensions and narrow normalizers against frozen scores.

These tests keep production float32 arithmetic and serialization. Deliberately
ill-scaled vectors expose float64 disagreement; corrupted runtime ranking and
scores must still fail the equivalence gates.
"""

import copy
import pickle

import numpy as np
import pytest

from openpi.cache.components.factors.base import LibraryStats
from exp.ablation_study.cache_prune import verify_prune as verification
from exp.ablation_study.cache_prune.prune_library import (
    audit_source,
    compute_task_scores,
    export_pruned,
    load_raw,
    load_score_blocks,
    select_retained,
)
from .test_cache_prune_offline import tiny_source as tiny_source


def test_real_dimension_narrow_sigma_and_diagnostic_only_oracle(
    tiny_source, tmp_path, monkeypatch
):
    """A 32768-dimensional reduction disagreement cannot invalidate faithful reloads."""
    path, expected, cfg, _ = tiny_source
    raw = load_raw(path)
    for field in ("vision_0", "vision_1"):
        cfg["backend"]["vector_dims"][field] = 32768
        cfg["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"][
            field
        ]["params"].update(mu=0.978, sigma=0.007)
    for index, entry in enumerate(raw["entries"]):
        for field in ("vision_0", "vision_1"):
            vector = np.full(32768, 1e-3, dtype=np.float32)
            vector[0], vector[1] = 1, 0.001 if index % 4 == 0 else 0.22 + index * 0.0001
            entry.query_keys[field] = vector
    raw["vector_dims"] = cfg["backend"]["vector_dims"]
    raw["library_stats"] = LibraryStats.compute_from_entries(raw["entries"])
    with path.open("wb") as handle:
        pickle.dump(raw, handle)
    source = audit_source(path, expected)
    scores = compute_task_scores(
        raw["entries"], cfg, tmp_path / "scores", source_manifest=source
    )
    blocks = load_score_blocks(scores, source["rows"])
    output = export_pruned(
        path,
        select_retained(source["rows"], blocks, 0.0),
        tmp_path / "pruned",
        source_manifest=source,
    )
    report = verification.verify_pruned(source, output, cfg, score_manifest=scores)
    assert report["score_reference"] == "frozen_float32_v1"
    assert report["float64_diagnostic"]["max_score_error"] > 5e-5
    assert report["max_score_error"] < report["max_float32_bound"]
    assert report["max_top1_regret"] <= 2 * report["max_score_error"]
    original = verification.oracle_scores
    monkeypatch.setattr(
        verification,
        "oracle_scores",
        lambda *args, **kwargs: original(*args, **kwargs) + 0.1,
    )
    diagnostic = verification.verify_pruned(source, output, cfg, score_manifest=scores)
    assert (
        diagnostic["passed"]
        and diagnostic["max_score_error"] == report["max_score_error"]
    )
    assert diagnostic["float64_diagnostic"]["max_score_error"] > 0.09


@pytest.mark.parametrize("corruption", ["score", "winner"])
def test_numeric_verifier_rejects_corrupted_float32_retrieval(
    tiny_source, tmp_path, monkeypatch, corruption
):
    """Changing raw returned scores or selecting a nonmaximum remains a hard failure."""
    path, expected, cfg, entries = tiny_source
    source = audit_source(path, expected)
    scores = compute_task_scores(
        entries, cfg, tmp_path / "scores", source_manifest=source
    )
    output = export_pruned(
        path,
        select_retained(source["rows"], {}, None),
        tmp_path / "baseline",
        source_manifest=source,
    )
    original = verification.strategy_for_storage

    def factory(storage, config, top_k):
        strategy = original(
            storage,
            config,
            len(entries) if corruption == "winner" and top_k == 1 else top_k,
        )
        original_search = strategy.search

        def search(context):
            hits = original_search(context)
            if corruption == "winner" and top_k == 1:
                return hits[-1:]
            if corruption == "score":
                hits = copy.deepcopy(hits)
                for hit in hits:
                    hit.score += 0.01
            return hits

        strategy.search = search
        return strategy

    monkeypatch.setattr(verification, "strategy_for_storage", factory)
    with pytest.raises(
        ValueError, match="float32 propagation bound|actual score maximum"
    ):
        verification.verify_pruned(source, output, cfg, score_manifest=scores)
