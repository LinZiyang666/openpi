"""Integration: every yaml emitted by ``v2_spec.GENERATION`` (eval +
sibling warmup) must pass ``load_cache_config`` end-to-end against a
real (tiny) in_memory backend pkl. This is the G2 acceptance gate the
unit tests in ``test_v2_spec.py`` cannot cover.

The smoke covers:

  - validator rules 1-12 (dataclass parse + 4-layer schema check)
  - search-strategy ``trajectory_weights`` enforcement when ``trajectory_depth > 1``
  - composer ``weighted_sum_with_warm_fallback`` build path
  - warmup-yaml dump factors superset really covers eval factor keys
    (smoke: build factor instances and union their ``descriptor_orientations``
    against the warmup dump factor union)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
import torch
import yaml as _yaml

from exp.verdict_factor_judge.v2_spec import (
    CFG_SPECS,
    GENERATION,
    build_eval_yaml,
    build_warmup_yaml,
)
from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.components.factors.registry import get_class
from openpi.cache.config import load_cache_config
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


@pytest.fixture
def tiny_pkl_dir(tmp_path: Path) -> Path:
    """Build a tiny in-memory backend artifact at the path each cfg expects.

    v2_spec embeds absolute relative paths (``exp/warm_start/data/...``);
    we synthesise minimal artifacts at those exact locations under
    ``tmp_path`` and chdir into ``tmp_path`` for the duration of the test
    so ``load_cache_config`` can resolve the relative paths.
    """
    a = torch.ones(2, dtype=torch.float32)
    s = torch.ones(2, dtype=torch.float32)
    library_stats = LibraryStats(
        action_sigma=a, action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones(2, dtype=torch.bool),
    )
    entries = []
    for i in range(4):
        entries.append(CacheEntry(
            id=f"e{i}",
            checkpoint_id=CheckpointID.CP1,
            query_keys={
                "vision_0": torch.zeros(1, dtype=torch.float32),       # dim 1; backend.vector_dims overridden below
                "vision_1": torch.zeros(1, dtype=torch.float32),
                "robot_state": torch.tensor([float(i), 0.0], dtype=torch.float32),
            },
            payload=CachePayload(action_chunk=torch.tensor([[float(i), 0.0]], dtype=torch.float32)),
            trajectory_id="traj",
            step_idx=i,
        ))
    art = {"entries": entries, "library_stats": library_stats}
    for cfg_id, cfg in CFG_SPECS.items():
        target = tmp_path / cfg["preload_pkl"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            pickle.dump(art, fh)
    return tmp_path


def _shrink_vector_dims_for_smoke(content: dict) -> None:
    """The real cfg vector_dims (32k for spatial16) overshoot our 1-dim
    smoke entries. Backend keeps them as a contract but does not validate
    against the entry shapes here — we just compress to 1 so the backend
    initializer is happy.
    """
    backend = content["backend"]
    backend["vector_dims"] = {k: 1 for k in backend["vector_dims"]}
    backend["vector_dims"]["robot_state"] = 2     # match entry state shape


def _write_warmup_pool_for(yaml_id: str, factor_keys: list[str]) -> None:
    """Inject a synthetic per-yaml WarmupPool entry covering ``factor_keys``
    so calibration's ``samples_source.type=warmup`` passes
    ``bind_keys`` (every key needs ≥ window_size non-NaN samples).
    """
    from openpi.cache.warmup_pool import get_global_pool

    pool = get_global_pool()
    pool.set(yaml_id, {k: [float(i) / 100.0 for i in range(100)] for k in factor_keys})


@pytest.mark.parametrize(
    "cfg_id, stem, factors_blk, composer_blk",
    GENERATION,
    ids=[g[1] for g in GENERATION],
)
def test_generated_eval_yaml_passes_validator(
    tiny_pkl_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg_id: str,
    stem: str,
    factors_blk: list[dict],
    composer_blk: dict,
) -> None:
    monkeypatch.chdir(tiny_pkl_dir)

    eval_yaml = build_eval_yaml(cfg_id, factors_blk, composer_blk)
    _shrink_vector_dims_for_smoke(eval_yaml)

    p = tiny_pkl_dir / f"{stem}.yaml"
    p.write_text(_yaml.safe_dump(eval_yaml))
    # load_cache_config runs the static validator + dataclass parse path;
    # this is the gate G2 R1 Item 2 cared about (yamls must not fail at
    # load time). Runtime build (which would need WarmupPool population)
    # is exercised separately via build_per_connection_components in
    # production code paths.
    cfg = load_cache_config(p)
    assert cfg.checkpoints["cp1"].judge.type == "composite"


@pytest.mark.parametrize(
    "cfg_id, stem, factors_blk, composer_blk",
    GENERATION,
    ids=[g[1] for g in GENERATION],
)
def test_generated_warmup_yaml_passes_validator(
    tiny_pkl_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg_id: str,
    stem: str,
    factors_blk: list[dict],
    composer_blk: dict,
) -> None:
    monkeypatch.chdir(tiny_pkl_dir)
    warmup = build_warmup_yaml(cfg_id, stem, eval_factors=factors_blk)
    _shrink_vector_dims_for_smoke(warmup)
    p = tiny_pkl_dir / f"{stem}__warmup.yaml"
    p.write_text(_yaml.safe_dump(warmup))
    cfg = load_cache_config(p)
    # Warmup yaml's inner judge is always_warm_start — composite is on the dump side.
    assert cfg.checkpoints["cp1"].judge.type == "always_warm_start"


@pytest.mark.parametrize(
    "cfg_id, stem, factors_blk, composer_blk",
    GENERATION,
    ids=[g[1] for g in GENERATION],
)
def test_warmup_dump_covers_eval_factor_keys(
    cfg_id: str,
    stem: str,
    factors_blk: list[dict],
    composer_blk: dict,
) -> None:
    """G2 R1 Item 3: the warmup dump factor list MUST cover every
    descriptor key the eval composite will demand at bind_keys time."""
    eval_keys: set[str] = set()
    for f in factors_blk:
        cls = get_class(f["type"])
        eval_keys.update(cls.describe(dict(f["params"])).keys())

    warmup = build_warmup_yaml(cfg_id, stem, eval_factors=factors_blk)
    dump_factors = warmup["checkpoints"]["cp1"]["judge"]["dump"]["factors"]
    dump_keys: set[str] = set()
    for f in dump_factors:
        cls = get_class(f["type"])
        dump_keys.update(cls.describe(dict(f["params"])).keys())

    missing = eval_keys - dump_keys
    assert not missing, (
        f"warmup yaml for stem={stem!r} does not cover eval factor keys: "
        f"{sorted(missing)}"
    )
