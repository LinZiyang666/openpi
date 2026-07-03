"""Tests for CacheConfig.collection validation (C5 always-search gate + frame cap).

``validate_cache_config`` accumulates all errors and raises them joined, so
these tests match on the collection-specific message even if the minimal base
config trips other (unrelated) checks.
"""

import pytest

from openpi.cache.config import BackendConfig
from openpi.cache.config import CacheConfig
from openpi.cache.config import CheckpointConfig
from openpi.cache.config import CollectionConfig
from openpi.cache.config import ConfigValidationError
from openpi.cache.config import GateConfig
from openpi.cache.config import JudgeConfig
from openpi.cache.config import validate_cache_config
from openpi.cache.config import validate_effective_collection


def test_collection_requires_always_search_gate():
    """export_collect_meta with a non-always_search CP1 gate must fail (C5)."""
    config = CacheConfig(
        backend=BackendConfig(vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="always_skip"),
                judge=JudgeConfig(threshold=0.98),
            )
        },
        collection=CollectionConfig(export_collect_meta=True),
    )
    with pytest.raises(ConfigValidationError, match="always_search"):
        validate_cache_config(config)


def test_collection_frame_byte_cap_rejects_vision():
    """A vision field (32768 dims) blows past the default 32 KiB per-step cap."""
    config = CacheConfig(
        backend=BackendConfig(vector_dims={"robot_state": 32, "vision_0": 32768}),
        checkpoints={"cp1": CheckpointConfig(judge=JudgeConfig(threshold=0.98))},
        collection=CollectionConfig(
            export_collect_meta=True,
            collect_fields=["vision_0"],
            wire_frame_cap_kib=32,
        ),
    )
    with pytest.raises(ConfigValidationError, match="wire_frame_cap_kib"):
        validate_cache_config(config)


def test_collection_unknown_field_rejected():
    """A collect field absent from vector_dims is reported."""
    config = CacheConfig(
        backend=BackendConfig(vector_dims={"robot_state": 32}),
        checkpoints={"cp1": CheckpointConfig(judge=JudgeConfig(threshold=0.98))},
        collection=CollectionConfig(
            export_collect_meta=True,
            collect_fields=["not_a_field"],
        ),
    )
    with pytest.raises(ConfigValidationError, match="not in\\s+backend.vector_dims"):
        validate_cache_config(config)


def test_effective_collection_catches_cli_bypass():
    """CLI --export-collect-meta on a YAML-off config must NOT bypass the C5 gate."""
    config = CacheConfig(
        backend=BackendConfig(vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="always_skip"),
                judge=JudgeConfig(threshold=0.98),
            )
        },
        # YAML collection OFF -> load-time validate_cache_config wouldn't check it.
    )
    with pytest.raises(ConfigValidationError, match="always_search"):
        validate_effective_collection(
            config, export_collect_meta=True, collect_fields=["robot_state"]
        )


def test_effective_collection_requires_cp1_present():
    config = CacheConfig(
        backend=BackendConfig(vector_dims={"robot_state": 32}),
        checkpoints={"cp3": CheckpointConfig(judge=JudgeConfig(threshold=0.95))},
    )
    with pytest.raises(ConfigValidationError, match="cp1"):
        validate_effective_collection(
            config, export_collect_meta=True, collect_fields=["robot_state"]
        )


def test_effective_collection_off_is_noop():
    config = CacheConfig(backend=BackendConfig(vector_dims={"robot_state": 32}))
    validate_effective_collection(config, export_collect_meta=False, collect_fields=[])

