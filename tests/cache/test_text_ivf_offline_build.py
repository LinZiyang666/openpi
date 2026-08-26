"""Offline artifact-builder tests for instruction-span masked prompt pooling.

Synthetic H5 episodes + a stub tokenizer injected through the builder's
per-process tokenizer cache; the real-tokenizer parity check (GCS download)
is a @pytest.mark.manual test at the bottom.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp" / "common"))

import build_in_memory_cache_artifact as bld  # noqa: E402

_PROMPT_LEN = 10
_REAL = 6
_EMB = 8


class StubTokenizer:
    """tokenize(): fixed ids/mask matching the synthetic H5 (6 real tokens);
    encode_fragment(): the marker id pair [99, 55] sits at index 3."""

    def __init__(self, with_marker: bool = True):
        ids = [2, 10, 11, 99, 55, 12] if with_marker else [2, 10, 11, 12, 13, 14]
        self._ids = np.asarray(ids + [0] * (_PROMPT_LEN - _REAL))
        self._mask = np.asarray([True] * _REAL + [False] * (_PROMPT_LEN - _REAL))

    def tokenize(self, prompt, state=None):
        return self._ids.copy(), self._mask.copy()

    def encode_fragment(self, text):
        assert text == " State:"
        return [99, 55]


@pytest.fixture()
def stub_tokenizer(monkeypatch):
    tok = StubTokenizer()
    monkeypatch.setitem(bld._TOKENIZER_CACHE, _PROMPT_LEN, tok)
    return tok


def _write_episode(path: Path, *, uniform_prompt: bool = False) -> np.ndarray:
    """Write one synthetic episode; returns the prompt_emb rows used."""
    rng = np.random.default_rng(0)
    prompt_rows = rng.normal(size=(_PROMPT_LEN, _EMB)).astype(np.float32)
    if uniform_prompt:
        prompt_rows[:] = prompt_rows[0]
    else:
        # Trailing padding rows are one shared constant vector.
        prompt_rows[_REAL:] = prompt_rows[_REAL]
    with h5py.File(path, "w") as f:
        f.attrs["task"] = "pick the bowl"
        f.attrs["prompt"] = "pick the bowl"
        f.attrs["success"] = True
        for step in range(2):
            g = f.create_group(f"step_{step:04d}")
            g.create_dataset("vision_0", data=rng.normal(size=(256, _EMB)).astype(np.float32))
            g.create_dataset("prompt_emb", data=prompt_rows)
            g.create_dataset("robot_state", data=np.zeros(2, dtype=np.float32))
            g.create_dataset("clean_action", data=np.zeros((10, 4), dtype=np.float32))
    return prompt_rows


def _build(data_dir, **kwargs):
    defaults = dict(
        builder_type="cp1_mean_pool", checkpoint_id_str="CP1", workers=-1,
    )
    defaults.update(kwargs)
    return bld.build_artifact(str(data_dir), **defaults)


def test_masked_build_matches_direct_computation(tmp_path, stub_tokenizer):
    rows = _write_episode(tmp_path / "ep0.h5")
    art = _build(tmp_path, prompt_masked_pool=True)
    assert art["prompt_pool"] == {"masked": True, "instruction_span": False}
    key = art["entries"][0].query_keys["prompt_emb"]
    key = torch.as_tensor(np.asarray(key))
    expected = torch.from_numpy(rows[:_REAL]).mean(0)
    assert torch.allclose(key, expected)


def test_legacy_build_still_full_mean_and_records_meta(tmp_path):
    rows = _write_episode(tmp_path / "ep0.h5")
    art = _build(tmp_path)
    assert art["prompt_pool"] == {"masked": False, "instruction_span": False}
    key = torch.as_tensor(np.asarray(art["entries"][0].query_keys["prompt_emb"]))
    assert torch.allclose(key, torch.from_numpy(rows).mean(0))


def test_masked_equals_online_builder(tmp_path, stub_tokenizer):
    """Offline build and the online builder classes produce identical keys."""
    from openpi.cache.components.key_builder import CP1MeanPoolKeyBuilder
    from openpi.cache.types import CheckpointID

    _write_episode(tmp_path / "ep0.h5")
    art = _build(tmp_path, prompt_masked_pool=True)
    offline_key = torch.as_tensor(np.asarray(art["entries"][0].query_keys["prompt_emb"]))

    with h5py.File(tmp_path / "ep0.h5") as f:
        group = f["step_0000"]
        lang_mask = torch.from_numpy(stub_tokenizer.tokenize("x")[1].astype(bool))
        fake = bld._build_fake_stage1(group, lang_mask=lang_mask)
    kb = CP1MeanPoolKeyBuilder(prompt_masked_pool=True)
    kb.collect(CheckpointID.CP1, stage1=fake)
    online_key = kb.build(CheckpointID.CP1)["prompt_emb"]
    assert torch.equal(offline_key, online_key)


def test_instruction_span_build_slices_at_marker(tmp_path, stub_tokenizer):
    rows = _write_episode(tmp_path / "ep0.h5")
    art = _build(tmp_path, prompt_masked_pool=True, prompt_instruction_span=True)
    assert art["prompt_pool"] == {"masked": True, "instruction_span": True}
    key = torch.as_tensor(np.asarray(art["entries"][0].query_keys["prompt_emb"]))
    # Marker [99, 55] at index 3 => instruction tokens are the first 3 rows.
    assert torch.allclose(key, torch.from_numpy(rows[:3]).mean(0))


def test_dual_source_mask_mismatch_aborts(tmp_path, stub_tokenizer):
    # All prompt rows identical => stored rows imply 0 real tokens, the stub
    # mask says 6 — the two sources disagree and the build must abort.
    _write_episode(tmp_path / "ep0.h5", uniform_prompt=True)
    with pytest.raises(ValueError, match="dual-source"):
        _build(tmp_path, prompt_masked_pool=True)


def test_span_marker_missing_aborts(tmp_path, monkeypatch):
    monkeypatch.setitem(bld._TOKENIZER_CACHE, _PROMPT_LEN, StubTokenizer(with_marker=False))
    _write_episode(tmp_path / "ep0.h5")
    with pytest.raises(ValueError, match="marker"):
        _build(tmp_path, prompt_masked_pool=True, prompt_instruction_span=True)


def test_knob_on_unsupported_builder_aborts_before_h5(tmp_path):
    # No H5 files needed: the allowlist rejects before any file is touched.
    with pytest.raises(ValueError, match="only\\s+honoured by"):
        _build(tmp_path, builder_type="cp1_temporal_prune", prompt_masked_pool=True)
    with pytest.raises(ValueError, match="only\\s+honoured by"):
        _build(tmp_path, builder_type="cp1_llm_layer_extract", prompt_masked_pool=True)


def test_span_without_masked_aborts(tmp_path):
    with pytest.raises(ValueError, match="requires --prompt-masked-pool"):
        _build(tmp_path, prompt_instruction_span=True)


@pytest.mark.manual
def test_real_tokenizer_marker_parity():
    """Real PaligemmaTokenizer (GCS download): the ' State:' marker occurs
    exactly once in the pi05 discrete-state template and never in the LIBERO
    (state-less) format, and the span cut lands right before ' State:'."""
    from openpi.cache.components.key_builder import find_instruction_span
    from openpi.models.tokenizer import PaligemmaTokenizer

    tok = PaligemmaTokenizer(max_len=200)
    marker = np.asarray(tok.encode_fragment(" State:"), dtype=np.int64)
    prompts = [
        "pick up the black bowl and place it on the plate",
        "put both moka pots on the stove",
    ]
    state = np.linspace(-1, 1, 32)

    for text in prompts:
        # LIBERO format: no state in prompt => no marker.
        ids, mask = tok.tokenize(text)
        valid = np.asarray(ids)[: int(np.asarray(mask).sum())]
        assert find_instruction_span(valid, marker) is None

        # Discrete-state format: marker exactly once, span > 0.
        ids, mask = tok.tokenize(text, state=state)
        valid = np.asarray(ids)[: int(np.asarray(mask).sum())]
        span = find_instruction_span(valid, marker)
        assert span is not None and span > 0
        assert find_instruction_span(valid[span + 1:], marker) is None
