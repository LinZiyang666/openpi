"""G2 R2 Item 2 — `warm_fallback_start_t` is CP1-only.

Plan §3.6 / validator §13.3 rule 5c: any WARM_START emission path
requires CP1 (CP3 has no `intermediates` payload to resume from).
``weighted_sum_with_warm_fallback`` introduces a second emission path
(the all-NaN fallback) that must obey the same constraint.

These tests exercise:

  - cp1 + warm_fallback_start_t  → load passes
  - cp3 + warm_fallback_start_t  → load fails with explicit CP1-only message
  - cp1 + warm_start_t           → load passes (regression on the existing rule)
  - cp3 + warm_start_t           → load fails (regression on the existing rule)

Doc-style yaml shape regression: the example shipped in
``docs/cache/verdict_factor_judge.md`` § 3 / §7.4 places composer
fields directly under ``composer:`` (no ``params:`` wrapper). Verify
that the documented shape loads cleanly so the published example does
not silently produce a rejected config (G2 R2 Item 1).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
import torch

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.config import ConfigValidationError, load_cache_config
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


@pytest.fixture
def library_pkl(tmp_path: Path) -> Path:
    a = torch.ones(2, dtype=torch.float32)
    s = torch.ones(2, dtype=torch.float32)
    library_stats = LibraryStats(
        action_sigma=a, action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones(2, dtype=torch.bool),
    )
    entries = [
        CacheEntry(
            id=f"e{i}",
            checkpoint_id=CheckpointID.CP1,
            query_keys={"robot_state": torch.tensor([float(i), 0.0], dtype=torch.float32)},
            payload=CachePayload(
                action_chunk=torch.tensor([[float(i), 0.0]], dtype=torch.float32),
            ),
            trajectory_id="traj",
            step_idx=i,
        )
        for i in range(4)
    ]
    p = tmp_path / "fix.pkl"
    with open(p, "wb") as fh:
        pickle.dump({"entries": entries, "library_stats": library_stats}, fh)
    return p


@pytest.fixture
def calib_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "calib.jsonl"
    with p.open("w") as fh:
        for i in range(60):
            fh.write(json.dumps({"factor_raw": {
                "jerk_online_action__p1_f1": float(i) / 60.0,
            }}) + "\n")
    return p


def _yaml(
    library_pkl: Path,
    calib_path: Path,
    *,
    cp_name: str,
    composer_yaml: str,
) -> str:
    """Composite yaml on the chosen checkpoint (cp1 / cp3) with the
    composer block plugged in verbatim. ``composer_yaml`` is indented at
    8 spaces from column 0 (the body of ``composer:`` — see indents below).
    """
    return f"""\
enabled: true
keys:
  robot_state: {{enabled: true, weight: 1.0}}
key_builder: {{type: placeholder}}
backend:
  type: in_memory
  vector_dims: {{robot_state: 2}}
  in_memory:
    preload_path: {library_pkl}
    index_type: brute_force
checkpoints:
  {cp_name}:
    gate: {{type: always_search}}
    search_strategy: {{type: weighted_rrf_knn, rrf_k: 60, top_k: 1}}
    judge:
      type: composite
      normalization:
        type: zscore
        params: {{}}
        stats_source: {{type: offline}}
      factors:
        - type: jerk_online_action
          params: {{windows: [{{past: 1, future: 1}}]}}
      calibration:
        type: percentile_rolling
        params: {{window_size: 50}}
        samples_source:
          type: offline
          offline:
            path: {calib_path}
            format: jsonl
      composer:
{composer_yaml}
"""


# Body indented 8 spaces, fields directly under ``composer:`` (flat shape).
_FALLBACK_COMPOSER_FLAT = (
    "        type: weighted_sum_with_warm_fallback\n"
    "        weights: {jerk_online_action__p1_f1: 1.0}\n"
    "        tier_thresholds: {full_hit: 0.5}\n"
    "        warm_fallback_start_t: 0.7"
)

_PLAIN_WEIGHTED_SUM = (
    "        type: weighted_sum\n"
    "        weights: {jerk_online_action__p1_f1: 1.0}\n"
    "        tier_thresholds: {full_hit: 0.5, warm_start: 0.2}\n"
    "        warm_start_t: 0.7"
)


# ----------------------------------------------------------------------
# G2 R2 Item 2 — CP1 vs CP3 guard for warm_fallback_start_t
# ----------------------------------------------------------------------


def test_warm_fallback_start_t_accepted_on_cp1(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    p = tmp_path / "ok.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp1",
                       composer_yaml=_FALLBACK_COMPOSER_FLAT))
    cfg = load_cache_config(p)
    assert cfg.checkpoints["cp1"].judge.composer.type == "weighted_sum_with_warm_fallback"
    assert cfg.checkpoints["cp1"].judge.composer.warm_fallback_start_t == 0.7


def test_warm_fallback_start_t_rejected_on_cp3(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp3",
                       composer_yaml=_FALLBACK_COMPOSER_FLAT))
    with pytest.raises(ConfigValidationError, match="warm_fallback_start_t is only supported on CP1"):
        load_cache_config(p)


# ----------------------------------------------------------------------
# Regression on the existing rule (warm_start_t)
# ----------------------------------------------------------------------


def test_warm_start_t_accepted_on_cp1(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    p = tmp_path / "ok2.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp1",
                       composer_yaml=_PLAIN_WEIGHTED_SUM))
    cfg = load_cache_config(p)
    assert cfg.checkpoints["cp1"].judge.composer.warm_start_t == 0.7


def test_warm_start_t_rejected_on_cp3(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    p = tmp_path / "bad2.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp3",
                       composer_yaml=_PLAIN_WEIGHTED_SUM))
    with pytest.raises(ConfigValidationError, match="warm_start_t is only supported on CP1"):
        load_cache_config(p)


# ----------------------------------------------------------------------
# G2 R2 Item 1 — published doc example shape regression
# ----------------------------------------------------------------------


def test_doc_style_flat_composer_shape_loads(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    """The exact ``composer:`` shape shipped in
    ``docs/cache/verdict_factor_judge.md`` §7.4 must round-trip through
    ``load_cache_config`` without "Unknown config key" warnings or a
    missing-tier_thresholds reject.
    """
    composer_yaml = (
        "        type: weighted_sum_with_warm_fallback\n"
        "        weights:\n"
        "          jerk_online_action__p1_f1: 1.0\n"
        "        tier_thresholds:\n"
        "          full_hit: 0.30\n"
        "          warm_start: 0.10\n"
        "        warm_start_t: 0.7\n"
        "        warm_fallback_start_t: 0.7"
    )
    p = tmp_path / "doc_style.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp1",
                       composer_yaml=composer_yaml))
    cfg = load_cache_config(p)
    composer = cfg.checkpoints["cp1"].judge.composer
    # Field-by-field: every doc-promised field is parsed (flat shape).
    assert composer.type == "weighted_sum_with_warm_fallback"
    assert composer.weights == {"jerk_online_action__p1_f1": 1.0}
    assert composer.tier_thresholds == {"full_hit": 0.30, "warm_start": 0.10}
    assert composer.warm_start_t == 0.7
    assert composer.warm_fallback_start_t == 0.7


def test_legacy_params_wrapper_is_ignored_so_doc_must_not_use_it(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    """The pre-G2-R2 doc example put composer fields under ``params:``;
    the loader treats ``params`` as an unknown ComposerConfig field and
    emits a warning, then validation fails because ``tier_thresholds``
    never made it into the dataclass. This regression locks the doc to
    the flat shape — if a future doc edit reintroduces ``composer.params``
    this test catches it.
    """
    composer_yaml = (
        "        type: weighted_sum_with_warm_fallback\n"
        "        params:\n"
        "          weights: {jerk_online_action__p1_f1: 1.0}\n"
        "          tier_thresholds: {full_hit: 0.5}\n"
        "          warm_fallback_start_t: 0.7"
    )
    p = tmp_path / "wrong.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp1",
                       composer_yaml=composer_yaml))
    with pytest.raises(ConfigValidationError):
        load_cache_config(p)
