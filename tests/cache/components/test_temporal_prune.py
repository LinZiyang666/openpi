"""Tests for CP1 temporal pruning key builder and token reducers."""

import pytest
import torch

from openpi.cache.components.key_builder import (
    CP1TemporalPruneKeyBuilder,
    _VisionHistoryBuffer,
)
from openpi.cache.components.token_reducer import (
    MaxPoolReducer,
    MeanPoolReducer,
    PruneResult,
    SpatialPoolReducer,
    TaskScoringReducer,
)
from openpi.cache.types import (
    PROMPT_EMB,
    ROBOT_STATE,
    VISION_0,
    VISION_1,
    VISION_2,
    CheckpointID,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMB_DIM = 2048
STATE_DIM = 32
NUM_PROMPT_TOKENS = 20
TOKENS_PER_IMAGE = 256


class _FakeStage1:
    def __init__(self, prefix_embs, state):
        self.prefix_embs = prefix_embs
        self.state = state


def _make_fake_stage1(
    emb_dim: int = EMB_DIM,
    state_dim: int = STATE_DIM,
    num_prompt_tokens: int = NUM_PROMPT_TOKENS,
) -> _FakeStage1:
    prefix_len = TOKENS_PER_IMAGE * 3 + num_prompt_tokens
    prefix_embs = torch.randn(1, prefix_len, emb_dim)
    state = torch.randn(1, state_dim)
    return _FakeStage1(prefix_embs, state)


def _make_builder(
    reducer=None,
    window_size: int = 4,
    keep_ratio: float = 0.5,
    enabled_fields=None,
) -> CP1TemporalPruneKeyBuilder:
    if reducer is None:
        reducer = MeanPoolReducer()
    return CP1TemporalPruneKeyBuilder(
        reducer=reducer,
        enabled_fields=enabled_fields,
        prune_window_size=window_size,
        temporal_keep_ratio=keep_ratio,
    )


def _make_prune_result(
    W: int = 4,
    K: int = 128,
    emb_dim: int = EMB_DIM,
    pruned: bool = True,
) -> PruneResult:
    tokens = torch.randn(W, K, emb_dim)
    indices = torch.randperm(TOKENS_PER_IMAGE)[:K].sort().values
    scores = torch.rand(K) if pruned else None
    return PruneResult(
        tokens=tokens,
        token_indices=indices,
        pruned=pruned,
        temporal_scores=scores,
    )


# ---------------------------------------------------------------------------
# _VisionHistoryBuffer tests
# ---------------------------------------------------------------------------


class TestVisionHistoryBuffer:
    def test_push_and_get_window(self):
        buf = _VisionHistoryBuffer(window_size=3)
        for _ in range(3):
            buf.push(torch.randn(TOKENS_PER_IMAGE, EMB_DIM))
        window = buf.get_window()
        assert window.shape == (3, TOKENS_PER_IMAGE, EMB_DIM)

    def test_fifo_eviction(self):
        """Oldest frame evicted when buffer exceeds window_size."""
        buf = _VisionHistoryBuffer(window_size=3)
        frames = [torch.randn(TOKENS_PER_IMAGE, EMB_DIM) for _ in range(5)]
        for f in frames:
            buf.push(f)
        window = buf.get_window()
        assert window.shape[0] == 3
        # Last 3 frames should be kept
        assert torch.equal(window[0], frames[2])
        assert torch.equal(window[2], frames[4])

    def test_ready_property(self):
        buf = _VisionHistoryBuffer(window_size=3)
        assert not buf.ready
        buf.push(torch.randn(TOKENS_PER_IMAGE, EMB_DIM))
        assert not buf.ready
        buf.push(torch.randn(TOKENS_PER_IMAGE, EMB_DIM))
        assert not buf.ready
        buf.push(torch.randn(TOKENS_PER_IMAGE, EMB_DIM))
        assert buf.ready

    def test_reset_clears_buffer(self):
        buf = _VisionHistoryBuffer(window_size=3)
        for _ in range(3):
            buf.push(torch.randn(TOKENS_PER_IMAGE, EMB_DIM))
        assert buf.ready
        buf.reset()
        assert not buf.ready


# ---------------------------------------------------------------------------
# Temporal scoring tests
# ---------------------------------------------------------------------------


class TestTemporalPruning:
    def test_static_tokens_removed(self):
        """Static tokens should get low temporal scores and be pruned."""
        W, N, D = 4, 256, EMB_DIM
        # Create window where first 200 tokens are static (identical across frames)
        # and last 56 tokens change significantly
        static_tokens = torch.randn(1, 200, D).expand(W, -1, -1)
        dynamic_tokens = torch.randn(W, 56, D)  # different each frame
        window = torch.cat([static_tokens, dynamic_tokens], dim=1)

        builder = _make_builder(window_size=W, keep_ratio=0.5)
        result = builder._temporal_prune(window)

        assert result.pruned is True
        assert result.tokens.shape[0] == W
        assert result.tokens.shape[1] == 128  # 256 * 0.5
        # Most of the kept indices should be from the dynamic region [200:256]
        dynamic_count = (result.token_indices >= 200).sum().item()
        assert dynamic_count >= 50  # at least ~90% of the 56 dynamic tokens kept

    def test_all_dynamic_keeps_top_ratio(self):
        """When all tokens change, still keeps exactly keep_ratio fraction."""
        W, N, D = 4, 256, EMB_DIM
        window = torch.randn(W, N, D)  # all tokens differ across frames
        builder = _make_builder(window_size=W, keep_ratio=0.5)
        result = builder._temporal_prune(window)

        assert result.pruned is True
        assert result.tokens.shape == (W, 128, D)
        assert result.token_indices.shape == (128,)
        assert result.temporal_scores.shape == (128,)

    def test_prune_result_indices_correct(self):
        """token_indices should correctly index back into original tokens."""
        W, N, D = 4, 256, 64  # smaller D for speed
        window = torch.randn(W, N, D)
        builder = _make_builder(window_size=W, keep_ratio=0.5)
        result = builder._temporal_prune(window)

        # Verify that selected tokens match original at those indices
        for w in range(W):
            expected = window[w, result.token_indices, :]
            assert torch.equal(result.tokens[w], expected)


# ---------------------------------------------------------------------------
# Degradation mode tests
# ---------------------------------------------------------------------------


class TestDegradedMode:
    def test_single_frame_no_prune(self):
        """With only 1 frame, pruning should be skipped."""
        builder = _make_builder(window_size=4)
        stage1 = _make_fake_stage1()
        builder.collect(CheckpointID.CP1, stage1=stage1)
        keys = builder.build(CheckpointID.CP1)

        assert VISION_0 in keys
        assert keys[VISION_0].shape == (EMB_DIM,)  # mean pool output
        assert keys[VISION_0].device == torch.device("cpu")
        assert keys[VISION_0].dtype == torch.float32

    def test_partial_window_no_prune(self):
        """With frames < window_size, pruning should be skipped."""
        builder = _make_builder(window_size=4)
        for _ in range(3):  # 3 < 4 = window_size
            stage1 = _make_fake_stage1()
            builder.collect(CheckpointID.CP1, stage1=stage1)
            keys = builder.build(CheckpointID.CP1)
            builder.clear()

        assert VISION_0 in keys
        assert keys[VISION_0].shape == (EMB_DIM,)

    def test_full_window_prunes(self):
        """With frames >= window_size, pruning should activate."""
        builder = _make_builder(window_size=4, keep_ratio=0.5)
        for _ in range(4):
            stage1 = _make_fake_stage1()
            builder.collect(CheckpointID.CP1, stage1=stage1)
            keys = builder.build(CheckpointID.CP1)
            builder.clear()

        # Output shape is still [EMB_DIM] because MeanPoolReducer
        assert VISION_0 in keys
        assert keys[VISION_0].shape == (EMB_DIM,)

    def test_output_dim_consistent_across_pruned_states(self):
        """Output dimension must be the same whether pruned or not."""
        builder = _make_builder(window_size=4, keep_ratio=0.5)
        dims_by_step: list[int] = []
        for _ in range(6):
            stage1 = _make_fake_stage1()
            builder.collect(CheckpointID.CP1, stage1=stage1)
            keys = builder.build(CheckpointID.CP1)
            dims_by_step.append(keys[VISION_0].shape[0])
            builder.clear()

        assert all(d == dims_by_step[0] for d in dims_by_step)


# ---------------------------------------------------------------------------
# CP1/CP3 lifecycle tests
# ---------------------------------------------------------------------------


class TestCP1CP3Lifecycle:
    def test_cp1_cp3_same_cycle_single_push(self):
        """CP1 + CP3 in the same cycle should only push history once."""
        builder = _make_builder(window_size=4)
        stage1 = _make_fake_stage1()

        # CP1 collect -> should push
        builder.collect(CheckpointID.CP1, stage1=stage1)
        builder.build(CheckpointID.CP1)
        # CP3 collect -> should NOT push
        builder.collect(CheckpointID.CP3, stage1=stage1)
        builder.build(CheckpointID.CP3)
        builder.clear()

        # History should have exactly 1 frame
        for buf in builder._history.values():
            assert len(buf._buffer) == 1

    def test_episode_start_resets_buffer(self):
        """on_episode_start() should clear all history buffers."""
        builder = _make_builder(window_size=4)
        for _ in range(3):
            stage1 = _make_fake_stage1()
            builder.collect(CheckpointID.CP1, stage1=stage1)
            builder.build(CheckpointID.CP1)
            builder.clear()

        # All buffers should have 3 frames
        for buf in builder._history.values():
            assert len(buf._buffer) == 3

        builder.on_episode_start()

        # All buffers should be empty
        for buf in builder._history.values():
            assert len(buf._buffer) == 0

    def test_clear_preserves_history(self):
        """clear() should only release per-cycle cache, not history."""
        builder = _make_builder(window_size=4)
        stage1 = _make_fake_stage1()
        builder.collect(CheckpointID.CP1, stage1=stage1)
        builder.build(CheckpointID.CP1)
        builder.clear()

        # Per-cycle cache should be empty
        assert len(builder._cache) == 0
        # History should still have 1 frame
        for buf in builder._history.values():
            assert len(buf._buffer) == 1


# ---------------------------------------------------------------------------
# Reducer tests
# ---------------------------------------------------------------------------


class TestMeanPoolReducer:
    def test_output_shape(self):
        reducer = MeanPoolReducer()
        result = _make_prune_result(W=4, K=128)
        out = reducer.reduce(result)
        assert out.shape == (EMB_DIM,)
        assert reducer.output_dim == EMB_DIM

    def test_pruned_false_same_shape(self):
        reducer = MeanPoolReducer()
        result = _make_prune_result(W=2, K=256, pruned=False)
        out = reducer.reduce(result)
        assert out.shape == (EMB_DIM,)


class TestMaxPoolReducer:
    def test_output_shape(self):
        reducer = MaxPoolReducer()
        result = _make_prune_result(W=4, K=128)
        out = reducer.reduce(result)
        assert out.shape == (EMB_DIM,)
        assert reducer.output_dim == EMB_DIM


class TestSpatialPoolReducer:
    def test_full_grid(self):
        """All 256 tokens present, no missing positions."""
        reducer = SpatialPoolReducer(output_tokens=16)
        result = _make_prune_result(W=4, K=256, pruned=False)
        result.token_indices = torch.arange(256)
        out = reducer.reduce(result)
        assert out.shape == (16 * EMB_DIM,)
        assert reducer.output_dim == 16 * EMB_DIM

    def test_pruned_partial_grid(self):
        """128 tokens with indices, missing positions zero-filled."""
        reducer = SpatialPoolReducer(output_tokens=16)
        result = _make_prune_result(W=4, K=128, pruned=True)
        out = reducer.reduce(result)
        assert out.shape == (16 * EMB_DIM,)

    def test_output_tokens_4(self):
        reducer = SpatialPoolReducer(output_tokens=4)
        result = _make_prune_result(W=4, K=128)
        out = reducer.reduce(result)
        assert out.shape == (4 * EMB_DIM,)
        assert reducer.output_dim == 4 * EMB_DIM

    def test_invalid_output_tokens(self):
        with pytest.raises(ValueError, match="perfect square"):
            SpatialPoolReducer(output_tokens=15)

    def test_varying_missing_ratio(self):
        """Different missing ratios (25%/50%/75%) should produce different outputs."""
        reducer = SpatialPoolReducer(output_tokens=16)
        outputs = []
        for k in [192, 128, 64]:  # 25%, 50%, 75% missing
            result = _make_prune_result(W=4, K=k, pruned=True)
            outputs.append(reducer.reduce(result))

        # All should have same shape
        for out in outputs:
            assert out.shape == (16 * EMB_DIM,)

        # Outputs should differ (statistical: extremely unlikely to be equal)
        assert not torch.equal(outputs[0], outputs[1])
        assert not torch.equal(outputs[1], outputs[2])


class TestTaskScoringReducer:
    def test_with_prompt(self):
        reducer = TaskScoringReducer(select_k=32)
        result = _make_prune_result(W=4, K=128)
        prompt_emb = torch.randn(NUM_PROMPT_TOKENS, EMB_DIM)
        out = reducer.reduce(result, prompt_emb=prompt_emb)
        assert out.shape == (EMB_DIM,)
        assert reducer.output_dim == EMB_DIM

    def test_no_prompt_fallback(self):
        """Without prompt_emb, should fall back to mean pool."""
        reducer = TaskScoringReducer(select_k=32)
        result = _make_prune_result(W=4, K=128)
        out = reducer.reduce(result, prompt_emb=None)
        assert out.shape == (EMB_DIM,)

    def test_empty_prompt_fallback(self):
        reducer = TaskScoringReducer(select_k=32)
        result = _make_prune_result(W=4, K=128)
        out = reducer.reduce(result, prompt_emb=torch.empty(0, EMB_DIM))
        assert out.shape == (EMB_DIM,)

    def test_select_k_larger_than_tokens(self):
        """select_k > available tokens should not crash."""
        reducer = TaskScoringReducer(select_k=256)
        result = _make_prune_result(W=4, K=64)
        prompt_emb = torch.randn(NUM_PROMPT_TOKENS, EMB_DIM)
        out = reducer.reduce(result, prompt_emb=prompt_emb)
        assert out.shape == (EMB_DIM,)


# ---------------------------------------------------------------------------
# Full pipeline integration tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_collect_build_multi_step(self):
        """End-to-end: multiple collect -> build cycles produce valid keys."""
        builder = _make_builder(window_size=3, keep_ratio=0.5)

        for step in range(5):
            stage1 = _make_fake_stage1()
            builder.collect(CheckpointID.CP1, stage1=stage1)
            keys = builder.build(CheckpointID.CP1)
            builder.clear()

            assert VISION_0 in keys
            assert ROBOT_STATE in keys
            assert PROMPT_EMB in keys
            assert keys[VISION_0].shape == (EMB_DIM,)
            assert keys[ROBOT_STATE].shape == (STATE_DIM,)
            assert keys[PROMPT_EMB].shape == (EMB_DIM,)
            assert keys[VISION_0].dtype == torch.float32
            assert keys[VISION_0].device == torch.device("cpu")

    def test_spatial_pool_reducer_pipeline(self):
        """Full pipeline with SpatialPoolReducer."""
        reducer = SpatialPoolReducer(output_tokens=16)
        builder = _make_builder(reducer=reducer, window_size=3)

        for _ in range(4):
            stage1 = _make_fake_stage1()
            builder.collect(CheckpointID.CP1, stage1=stage1)
            keys = builder.build(CheckpointID.CP1)
            builder.clear()

        assert keys[VISION_0].shape == (16 * EMB_DIM,)

    def test_task_scoring_reducer_pipeline(self):
        """Full pipeline with TaskScoringReducer."""
        reducer = TaskScoringReducer(select_k=32)
        builder = _make_builder(reducer=reducer, window_size=3)

        for _ in range(4):
            stage1 = _make_fake_stage1()
            builder.collect(CheckpointID.CP1, stage1=stage1)
            keys = builder.build(CheckpointID.CP1)
            builder.clear()

        assert keys[VISION_0].shape == (EMB_DIM,)

    def test_enabled_fields_subset(self):
        """Only enabled vision fields should appear in output."""
        builder = _make_builder(
            enabled_fields=[VISION_0, ROBOT_STATE],
            window_size=2,
        )
        for _ in range(3):
            stage1 = _make_fake_stage1()
            builder.collect(CheckpointID.CP1, stage1=stage1)
            keys = builder.build(CheckpointID.CP1)
            builder.clear()

        assert VISION_0 in keys
        assert ROBOT_STATE in keys
        assert VISION_1 not in keys
        assert VISION_2 not in keys
        assert PROMPT_EMB not in keys

    def test_two_builders_match_same_sequence(self):
        """Two independent builders given the same episode produce identical keys."""
        reducer_a = MeanPoolReducer()
        reducer_b = MeanPoolReducer()
        builder_a = _make_builder(reducer=reducer_a, window_size=3)
        builder_b = _make_builder(reducer=reducer_b, window_size=3)

        # Use identical stage1 inputs for both builders
        stages = [_make_fake_stage1() for _ in range(5)]
        builder_a.on_episode_start()
        builder_b.on_episode_start()

        for stage1 in stages:
            builder_a.collect(CheckpointID.CP1, stage1=stage1)
            keys_a = builder_a.build(CheckpointID.CP1)
            builder_a.clear()

            builder_b.collect(CheckpointID.CP1, stage1=stage1)
            keys_b = builder_b.build(CheckpointID.CP1)
            builder_b.clear()

            for field in keys_a:
                assert torch.equal(keys_a[field], keys_b[field]), (
                    f"Step mismatch on field {field}"
                )


# ---------------------------------------------------------------------------
# Constructor validation tests
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_prune_window_size_1_rejected(self):
        """prune_window_size=1 should be rejected at construction."""
        with pytest.raises(ValueError, match="prune_window_size must be >= 2"):
            _make_builder(window_size=1)

    def test_task_scoring_select_k_0_rejected(self):
        with pytest.raises(ValueError, match="select_k must be >= 1"):
            TaskScoringReducer(select_k=0)

    def test_task_scoring_temperature_0_rejected(self):
        with pytest.raises(ValueError, match="temperature must be > 0"):
            TaskScoringReducer(temperature=0.0)

    def test_task_scoring_negative_temperature_rejected(self):
        with pytest.raises(ValueError, match="temperature must be > 0"):
            TaskScoringReducer(temperature=-1.0)

    def test_spatial_pool_output_tokens_0_rejected(self):
        with pytest.raises(ValueError, match="output_tokens must be >= 1"):
            SpatialPoolReducer(output_tokens=0)

    def test_spatial_pool_output_tokens_negative_rejected(self):
        with pytest.raises(ValueError, match="output_tokens must be >= 1"):
            SpatialPoolReducer(output_tokens=-1)

    def test_temporal_keep_ratio_0_rejected(self):
        with pytest.raises(ValueError, match="temporal_keep_ratio must be in"):
            _make_builder(keep_ratio=0.0)

    def test_temporal_keep_ratio_above_1_rejected(self):
        with pytest.raises(ValueError, match="temporal_keep_ratio must be in"):
            _make_builder(keep_ratio=1.5)


# ---------------------------------------------------------------------------
# Config integration tests (quick sanity, full config tests in test_config.py)
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_config_builds_temporal_prune_builder(self):
        """config._build_key_builder should produce CP1TemporalPruneKeyBuilder."""
        from openpi.cache.config import KeyBuilderConfig, ReducerConfig, _build_key_builder

        cfg = KeyBuilderConfig(
            type="cp1_temporal_prune",
            prune_window_size=4,
            temporal_keep_ratio=0.5,
            reducer=ReducerConfig(type="mean_pool"),
        )
        builder = _build_key_builder(
            cfg,
            enabled_fields=[VISION_0, VISION_1, PROMPT_EMB, ROBOT_STATE],
            vector_dims={VISION_0: 2048, VISION_1: 2048, PROMPT_EMB: 2048, ROBOT_STATE: 32},
        )
        assert isinstance(builder, CP1TemporalPruneKeyBuilder)

    def test_config_builds_spatial_reducer(self):
        """config._build_key_builder with spatial_pool reducer."""
        from openpi.cache.config import KeyBuilderConfig, ReducerConfig, _build_key_builder

        cfg = KeyBuilderConfig(
            type="cp1_temporal_prune",
            reducer=ReducerConfig(type="spatial_pool", output_tokens=16),
        )
        builder = _build_key_builder(
            cfg,
            enabled_fields=[VISION_0, ROBOT_STATE],
            vector_dims={VISION_0: 16 * 2048, ROBOT_STATE: 32},
        )
        assert isinstance(builder, CP1TemporalPruneKeyBuilder)

    def test_config_rejects_prune_window_1(self):
        """Config validation should reject prune_window_size=1."""
        from openpi.cache.config import (
            CacheConfig,
            CheckpointConfig,
            ConfigValidationError,
            KeyBuilderConfig,
            KeysConfig,
            KeyFieldConfig,
            BackendConfig,
            ReducerConfig,
            validate_cache_config,
        )

        config = CacheConfig(
            enabled=True,
            key_builder=KeyBuilderConfig(
                type="cp1_temporal_prune",
                prune_window_size=1,
                reducer=ReducerConfig(type="mean_pool"),
            ),
            keys=KeysConfig(
                vision_0=KeyFieldConfig(enabled=True),
                robot_state=KeyFieldConfig(enabled=True),
            ),
            backend=BackendConfig(
                vector_dims={"vision_0": 2048, "robot_state": 32},
            ),
            checkpoints={"cp1": CheckpointConfig()},
        )
        with pytest.raises(ConfigValidationError, match="prune_window_size must be >= 2"):
            validate_cache_config(config)

    def test_config_rejects_temperature_0(self):
        """Config validation should reject temperature=0."""
        from openpi.cache.config import (
            CacheConfig,
            CheckpointConfig,
            ConfigValidationError,
            KeyBuilderConfig,
            KeysConfig,
            KeyFieldConfig,
            BackendConfig,
            ReducerConfig,
            validate_cache_config,
        )

        config = CacheConfig(
            enabled=True,
            key_builder=KeyBuilderConfig(
                type="cp1_temporal_prune",
                reducer=ReducerConfig(type="task_scoring", temperature=0.0),
            ),
            keys=KeysConfig(
                vision_0=KeyFieldConfig(enabled=True),
                robot_state=KeyFieldConfig(enabled=True),
            ),
            backend=BackendConfig(
                vector_dims={"vision_0": 2048, "robot_state": 32},
            ),
            checkpoints={"cp1": CheckpointConfig()},
        )
        with pytest.raises(ConfigValidationError, match="temperature must be > 0"):
            validate_cache_config(config)

    def test_config_rejects_unknown_reducer_type(self):
        """Config validation should reject unknown reducer.type."""
        from openpi.cache.config import (
            CacheConfig,
            CheckpointConfig,
            ConfigValidationError,
            KeyBuilderConfig,
            KeysConfig,
            KeyFieldConfig,
            BackendConfig,
            ReducerConfig,
            validate_cache_config,
        )

        config = CacheConfig(
            enabled=True,
            key_builder=KeyBuilderConfig(
                type="cp1_temporal_prune",
                reducer=ReducerConfig(type="nonexistent"),
            ),
            keys=KeysConfig(
                vision_0=KeyFieldConfig(enabled=True),
                robot_state=KeyFieldConfig(enabled=True),
            ),
            backend=BackendConfig(
                vector_dims={"vision_0": 2048, "robot_state": 32},
            ),
            checkpoints={"cp1": CheckpointConfig()},
        )
        with pytest.raises(ConfigValidationError, match="reducer.type.*unknown"):
            validate_cache_config(config)

    def test_config_rejects_dim_mismatch(self):
        """Config should reject when vector_dims doesn't match reducer output."""
        from openpi.cache.config import (
            CacheConfig,
            CheckpointConfig,
            ConfigValidationError,
            KeyBuilderConfig,
            KeysConfig,
            KeyFieldConfig,
            BackendConfig,
            ReducerConfig,
            validate_cache_config,
        )

        config = CacheConfig(
            enabled=True,
            key_builder=KeyBuilderConfig(
                type="cp1_temporal_prune",
                reducer=ReducerConfig(type="spatial_pool", output_tokens=16),
            ),
            keys=KeysConfig(
                vision_0=KeyFieldConfig(enabled=True),
                robot_state=KeyFieldConfig(enabled=True),
            ),
            backend=BackendConfig(
                vector_dims={"vision_0": 2048, "robot_state": 32},  # wrong: should be 32768
            ),
            checkpoints={"cp1": CheckpointConfig()},
        )
        with pytest.raises(ConfigValidationError, match="does not match reducer output dim"):
            validate_cache_config(config)


# ---------------------------------------------------------------------------
# Orchestrator integration test
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    def test_orchestrator_broadcasts_to_key_builder(self):
        """Orchestrator.on_episode_start() should call key_builder.on_episode_start()."""
        from unittest.mock import MagicMock

        from openpi.cache.orchestrator import CacheOrchestrator

        builder = _make_builder(window_size=3)
        # Feed some history
        for _ in range(2):
            stage1 = _make_fake_stage1()
            builder.collect(CheckpointID.CP1, stage1=stage1)
            builder.build(CheckpointID.CP1)
            builder.clear()

        # Verify history exists
        for buf in builder._history.values():
            assert len(buf._buffer) == 2

        storage = MagicMock()
        orchestrator = CacheOrchestrator(
            storage=storage,
            key_builder=builder,
            gates={},
            judges={},
            search_strategies={},
        )
        orchestrator.on_episode_start()

        # History should be cleared
        for buf in builder._history.values():
            assert len(buf._buffer) == 0
