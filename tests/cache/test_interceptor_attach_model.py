"""Tests for InferenceInterceptor's attach_model auto-hook (Phase 5).

Covers the wiring added so cp1_llm_layer_extract gets its model layer
references without any caller intervention:

  Interceptor.__init__:
      kb = orchestrator.key_builder
      if hasattr(kb, "attach_model"):
          kb.attach_model(self._model)

Two cases:
  - Positive: KeyBuilder implements attach_model -> ref is set.
  - Negative: KeyBuilder lacks attach_model -> Interceptor still constructs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from openpi.cache.cache_storage import CacheStorage
from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import ThresholdJudge
from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.components.llm_layer_key_builder import (
    CP1LLMLayerExtractKeyBuilder,
)
from openpi.cache.components.prefix_reducer import PrefixMeanPoolReducer
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.types import CheckpointID

from tests.cache.conftest import TestStorageSearchStrategy


# ----------------------------------------------------------------------
# Mocks: minimum policy + model surface area for InferenceInterceptor
# ----------------------------------------------------------------------


class _MockLayer(nn.Module):
    def __init__(self, hidden_size: int, layer_idx: int):
        super().__init__()
        self.self_attn = SimpleNamespace(q_proj=nn.Linear(1, 1, bias=False))
        self._proj = nn.Linear(hidden_size, hidden_size, bias=False)


class _MockRotary(nn.Module):
    def forward(self, hidden, pos_ids):
        bs, length = pos_ids.shape
        half = hidden.shape[-1] // 2
        return torch.zeros(bs, length, half), torch.zeros(bs, length, half)


def _make_mock_model(hidden_size: int = 16, depth: int = 4):
    config = SimpleNamespace(_attn_implementation="sdpa")
    layers = nn.ModuleList([_MockLayer(hidden_size, i) for i in range(depth)])
    language_model = SimpleNamespace(
        layers=layers,
        rotary_emb=_MockRotary(),
        config=config,
    )
    model = SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(
            paligemma=SimpleNamespace(language_model=language_model)
        ),
        # Stage methods exist as no-ops; eager mode just borrows the refs.
        run_stage1=lambda obs: None,
        run_stage2=lambda s1: None,
        run_stage3=lambda s2, **kw: None,
    )
    return model


class _MockPolicy:
    """Minimum surface area for InferenceInterceptor.__init__ in eager mode."""
    def __init__(self):
        self._is_pytorch_model = True
        self._model = _make_mock_model()
        self._input_transform = lambda x: x
        self._output_transform = lambda x: x
        self._pytorch_device = "cpu"


def _make_orchestrator(key_builder):
    backend = InMemoryBackend({"vision_0": 2048, "robot_state": 32})
    storage = CacheStorage(backend)
    strategy = TestStorageSearchStrategy(storage, top_k=1)
    return CacheOrchestrator(
        storage=storage,
        key_builder=key_builder,
        gates={CheckpointID.CP1: AlwaysSearchGate()},
        judges={CheckpointID.CP1: ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95)},
        search_strategies={CheckpointID.CP1: strategy},
    )


# ----------------------------------------------------------------------
# Positive: cp1_llm_layer_extract gets attached
# ----------------------------------------------------------------------


def test_interceptor_attaches_model_to_llm_layer_builder():
    builder = CP1LLMLayerExtractKeyBuilder(
        reducer=PrefixMeanPoolReducer(), extract_layer=0,
    )
    assert builder._layers is None  # not attached yet
    orch = _make_orchestrator(builder)
    policy = _MockPolicy()

    InferenceInterceptor(policy=policy, orchestrator=orch, eager=True)

    expected = policy._model.paligemma_with_expert.paligemma.language_model.layers
    assert builder._layers is expected


# ----------------------------------------------------------------------
# Negative: builders without attach_model are unaffected
# ----------------------------------------------------------------------


def test_interceptor_skips_attach_for_placeholder_builder():
    """PlaceholderKeyBuilder has no `attach_model`; Interceptor must not
    raise and must not invent the attribute on it."""
    builder = PlaceholderKeyBuilder()
    orch = _make_orchestrator(builder)
    policy = _MockPolicy()

    # Construction must succeed.
    InferenceInterceptor(policy=policy, orchestrator=orch, eager=True)

    assert not hasattr(builder, "attach_model")


def test_interceptor_with_no_orchestrator_does_not_attach():
    """orchestrator=None bypasses the entire cache path including attach_model."""
    policy = _MockPolicy()
    InferenceInterceptor(policy=policy, orchestrator=None, eager=True)
    # No assertion needed beyond "did not raise" — the absence of an
    # orchestrator means the attach_model branch is never entered.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
