"""Unit tests for `_self_check_tokenizer_consistency` (the offline guard).

The guard is the main defense against silently building incompatible
artifacts: it re-tokenizes the first step's task+state, embeds via the
loaded model, and compares to the HDF5 `prompt_emb` at non-padding
positions. The Verify-stage parity test exercises this on a real model;
these unit tests prove the *pass / fail decision logic* is correct
without needing GPU or a checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from exp.common.build_in_memory_cache_artifact import (
    _self_check_tokenizer_consistency,
)


# ----------------------------------------------------------------------
# Mocks
# ----------------------------------------------------------------------


class _MockTokenizer:
    """Returns deterministic (tokens, mask) for a fixed prompt+state."""

    def __init__(self, max_len: int = 200, real_len: int = 30):
        self._max_len = max_len
        self._real_len = real_len

    def tokenize(self, task: str, state):
        # Deterministic non-zero token ids for the first `real_len` slots,
        # zero (pad) for the rest. Mask True = real, False = padding.
        tokens = np.zeros(self._max_len, dtype=np.int64)
        tokens[:self._real_len] = 7   # arbitrary non-pad id
        mask = np.zeros(self._max_len, dtype=bool)
        mask[:self._real_len] = True
        return tokens, mask


class _MockEmbedder:
    """Embeds tokens deterministically: id 7 -> ones vector, id 0 -> zeros."""

    def __init__(self, hidden_size: int = 2048):
        self._h = hidden_size

    def __call__(self, lang_tokens: torch.Tensor) -> torch.Tensor:
        # `lang_tokens` is [B, L] int64; produce [B, L, hidden].
        bs, length = lang_tokens.shape
        out = torch.zeros(bs, length, self._h, dtype=torch.float32)
        out[lang_tokens == 7] = 1.0
        return out


def _make_mock_model(hidden_size: int = 2048, discrete_state_input: bool = True):
    return SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(
            embed_language_tokens=_MockEmbedder(hidden_size),
        ),
        config=SimpleNamespace(discrete_state_input=discrete_state_input),
    )


def _write_hdf5(
    path: Path,
    *,
    task: str = "test task",
    state: np.ndarray | None = None,
    prompt_emb: np.ndarray | None = None,
    hidden_size: int = 2048,
    real_len: int = 30,
    max_len: int = 200,
):
    """Build a single-step HDF5 file matching EpisodeDataCollector schema."""
    if state is None:
        state = np.zeros(32, dtype=np.float32)
    if prompt_emb is None:
        # Match what _MockEmbedder produces for the mock tokenizer's tokens,
        # scaled by sqrt(hidden_size) — matches how `collection_policy.py`
        # captures prompt_emb (hook body applies the scale before store).
        prompt_emb = np.zeros((max_len, hidden_size), dtype=np.float16)
        scale = math.sqrt(hidden_size)
        prompt_emb[:real_len] = scale  # tokens id=7 -> ones * sqrt(D)
    with h5py.File(path, "w") as f:
        f.attrs["task"] = task
        grp = f.create_group("step_0000")
        grp.create_dataset("robot_state", data=state)
        grp.create_dataset("prompt_emb", data=prompt_emb)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_self_check_passes_on_matching_prompt(tmp_path):
    """Aligned prompt_emb -> no exception."""
    h5 = tmp_path / "step.h5"
    _write_hdf5(h5)
    model = _make_mock_model()
    tokenizer = _MockTokenizer()
    _self_check_tokenizer_consistency(
        h5, model, tokenizer, torch.device("cpu"),
    )  # must not raise


def test_self_check_raises_on_mismatched_prompt(tmp_path):
    """prompt_emb deliberately wrong at real positions -> RuntimeError."""
    h5 = tmp_path / "step.h5"
    bad_prompt = np.zeros((200, 2048), dtype=np.float16)
    bad_prompt[:30] = 99.0  # mock embed would produce sqrt(2048) ≈ 45.25, not 99
    _write_hdf5(h5, prompt_emb=bad_prompt)

    model = _make_mock_model()
    tokenizer = _MockTokenizer()
    with pytest.raises(RuntimeError, match="self-check failed"):
        _self_check_tokenizer_consistency(
            h5, model, tokenizer, torch.device("cpu"),
        )


def test_self_check_ignores_padding_positions(tmp_path):
    """Drift at padding positions must NOT trip the check (only mask=True positions matter)."""
    h5 = tmp_path / "step.h5"
    prompt = np.zeros((200, 2048), dtype=np.float16)
    scale = math.sqrt(2048)
    prompt[:30] = scale           # real positions: aligned (post-scale)
    prompt[30:] = 1234.0           # padding positions: garbage
    _write_hdf5(h5, prompt_emb=prompt)

    model = _make_mock_model()
    tokenizer = _MockTokenizer()
    # Must not raise — padding drift is ignored by design.
    _self_check_tokenizer_consistency(
        h5, model, tokenizer, torch.device("cpu"),
    )


def test_self_check_skips_when_prompt_is_empty(tmp_path, caplog):
    """A prompt that tokenizes to zero real tokens should skip with a warning, not raise."""
    h5 = tmp_path / "step.h5"
    _write_hdf5(h5, real_len=0)

    model = _make_mock_model()
    tokenizer = _MockTokenizer(real_len=0)
    import logging
    with caplog.at_level(logging.WARNING):
        _self_check_tokenizer_consistency(
            h5, model, tokenizer, torch.device("cpu"),
        )
    assert any("empty prompt" in rec.message for rec in caplog.records)


def test_self_check_invoked_in_artifact_build(tmp_path, monkeypatch):
    """End-to-end: `build_artifact` for cp1_llm_layer_extract calls the self-check
    helper before processing any step. Stub the helper and assert it's hit."""
    from exp.common import build_in_memory_cache_artifact as bm

    # Write a minimal valid HDF5 so build_artifact reaches the self-check call.
    h5 = tmp_path / "ep_0.h5"
    _write_hdf5(h5)
    # build_artifact requires success=True attr to count the episode.
    with h5py.File(h5, "a") as f:
        f.attrs["success"] = True

    called = {"count": 0}

    def fake_self_check(*args, **kwargs):
        called["count"] += 1

    def fake_loader(checkpoint_dir, config_name, device):
        return _make_mock_model(), _MockTokenizer()

    class _NoopBuilder:
        def attach_model(self, model):
            pass

    def fake_create_builder(*args, **kwargs):
        return _NoopBuilder()

    # Stop the function before running model forward (which the mock cannot do).
    def fake_process(*args, **kwargs):
        return []

    monkeypatch.setattr(bm, "_self_check_tokenizer_consistency", fake_self_check)
    monkeypatch.setattr(bm, "_load_pi05_for_llm_extract", fake_loader)
    monkeypatch.setattr(bm, "_create_builder", fake_create_builder)
    monkeypatch.setattr(bm, "_process_episode_with_model", fake_process)

    bm.build_artifact(
        data_dir=str(tmp_path),
        builder_type="cp1_llm_layer_extract",
        workers=-1,
        extract_layer=0,
        prefix_reducer_type="prefix_mean_pool",
        checkpoint_dir="dummy/path",
        config_name="dummy_config",
        device="cpu",
    )
    assert called["count"] == 1, "self-check must be invoked exactly once"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
