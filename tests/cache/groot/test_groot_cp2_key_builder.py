"""The GR00T CP2 key builder: layout, projection, storability, rejections."""

from __future__ import annotations

import types

import pytest
import torch

from openpi.cache.components.cp2_vlm_key_builder import get_projection_spec, project
from openpi.cache.groot.cp2_key_builder import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_STATE_FEAT_DIM,
    DEFAULT_TOKEN_LEN,
    KEY_BUILDER_TYPE,
    LAYOUT_KIND,
    GrootCP2TernaryKeyBuilder,
    flatten_source,
    input_dim,
    projection_meta_for,
)
from openpi.cache.types import VLM_OUT, CheckpointID

TOKEN_LEN, FEAT, STATE = 5, 4, 2
D = input_dim(TOKEN_LEN, FEAT, STATE)


def _source(n_tokens=3, *, feat=FEAT, state=STATE, seed=0, inference=False):
    g = torch.Generator().manual_seed(seed)
    vl = torch.randn(n_tokens, feat, generator=g)
    st = torch.randn(state, generator=g)
    if inference:
        with torch.inference_mode():
            vl, st = vl.clone(), st.clone()
    return types.SimpleNamespace(vl_encoded=vl, state_encoded=st)


def _builder(seed=11, d=6, p=0.5):
    return GrootCP2TernaryKeyBuilder(seed=seed, d=d, p=p, token_len=TOKEN_LEN, feature_dim=FEAT, state_feat_dim=STATE)


def test_paper_dimensions():
    assert input_dim(DEFAULT_TOKEN_LEN, DEFAULT_FEATURE_DIM, DEFAULT_STATE_FEAT_DIM) == 1_312_256
    assert KEY_BUILDER_TYPE == "cp2_groot_ternary" and LAYOUT_KIND == "groot_encoded_v1"


def test_flatten_pads_tokens_then_appends_state():
    src = _source(n_tokens=3)
    h = flatten_source(src.vl_encoded, src.state_encoded, token_len=TOKEN_LEN, feature_dim=FEAT, state_feat_dim=STATE)
    assert h.shape == (D,) and h.dtype is torch.float32 and not h.is_inference()
    assert torch.equal(h[: 3 * FEAT], src.vl_encoded.reshape(-1))
    assert torch.equal(h[3 * FEAT : TOKEN_LEN * FEAT], torch.zeros((TOKEN_LEN - 3) * FEAT))
    assert torch.equal(h[TOKEN_LEN * FEAT :], src.state_encoded)


def test_flatten_accepts_a_full_length_sequence_and_bf16_inputs():
    src = _source(n_tokens=TOKEN_LEN)
    h = flatten_source(src.vl_encoded.to(torch.bfloat16), src.state_encoded.to(torch.bfloat16),
                       token_len=TOKEN_LEN, feature_dim=FEAT, state_feat_dim=STATE)
    assert torch.equal(h[: TOKEN_LEN * FEAT], src.vl_encoded.to(torch.bfloat16).float().reshape(-1))


@pytest.mark.parametrize(
    "vl, st, fragment",
    [
        (torch.zeros(TOKEN_LEN + 1, FEAT), torch.zeros(STATE), "exceed token_len"),
        (torch.zeros(3, FEAT + 1), torch.zeros(STATE), f"[N, {FEAT}]"),
        (torch.zeros(1, 3, FEAT), torch.zeros(STATE), f"[N, {FEAT}]"),
        (torch.zeros(3, FEAT), torch.zeros(STATE + 1), f"[{STATE}]"),
        (torch.zeros(3, FEAT), torch.zeros(1, STATE), f"[{STATE}]"),
        (torch.full((3, FEAT), float("nan")), torch.zeros(STATE), "non-finite"),
        (torch.zeros(3, FEAT), torch.full((STATE,), float("inf")), "non-finite"),
    ],
)
def test_flatten_rejects_bad_shapes_and_values(vl, st, fragment):
    with pytest.raises(RuntimeError, match=fragment):
        flatten_source(vl, st, token_len=TOKEN_LEN, feature_dim=FEAT, state_feat_dim=STATE)


def test_projection_meta_carries_the_layout_and_the_paper_D():
    b = _builder()
    meta = b.projection_meta()
    spec = get_projection_spec(11, 6, 0.5, D)
    assert meta["layout"] == {"kind": LAYOUT_KIND, "token_len": TOKEN_LEN, "feature_dim": FEAT, "state_feat_dim": STATE}
    assert meta["D"] == D and meta["seed"] == 11 and meta["d"] == 6
    assert {k: v for k, v in meta.items() if k != "layout"} == spec.meta()
    assert projection_meta_for(spec, token_len=TOKEN_LEN, feature_dim=FEAT, state_feat_dim=STATE) == meta


def test_build_equals_the_shared_projection_oracle_and_is_storable():
    b = _builder()
    src = _source(inference=True)
    assert src.vl_encoded.is_inference()  # the runner's session tensors look like this
    b.collect(CheckpointID.CP2, cp2_source=src, stage2=object())
    key = b.build(CheckpointID.CP2)[VLM_OUT]
    h = flatten_source(src.vl_encoded, src.state_encoded, token_len=TOKEN_LEN, feature_dim=FEAT, state_feat_dim=STATE)
    assert torch.equal(key, project(h, b.spec))
    assert key.device.type == "cpu" and key.dtype is torch.float32 and key.is_contiguous()
    assert not key.is_inference()
    key.add_(0.0)  # storable: would raise on an inference tensor
    assert set(b.cached_data) == {"vl_encoded", "state_encoded"}
    b.clear()
    assert b.cached_data == {}


def test_same_seed_same_key_other_seed_other_key():
    src = _source()
    keys = []
    for seed in (11, 11, 12):
        b = _builder(seed=seed)
        b.collect(CheckpointID.CP2, cp2_source=src)
        keys.append(b.build(CheckpointID.CP2)[VLM_OUT])
    assert torch.equal(keys[0], keys[1])
    assert not torch.equal(keys[0], keys[2])


def test_state_changes_the_key_with_identical_tokens():
    b = _builder(d=64, p=0.5)
    a, c = _source(seed=1), _source(seed=1)
    c.state_encoded = c.state_encoded + 1.0
    out = []
    for src in (a, c):
        b.collect(CheckpointID.CP2, cp2_source=src)
        out.append(b.build(CheckpointID.CP2)[VLM_OUT])
    assert not torch.equal(out[0], out[1])


def test_collect_and_build_refuse_everything_but_a_cp2_source():
    b = _builder()
    with pytest.raises(ValueError, match="CP2 only"):
        b.collect(CheckpointID.CP1, cp2_source=_source())
    with pytest.raises(RuntimeError, match="requires cp2_source"):
        b.collect(CheckpointID.CP2, stage2=object())
    with pytest.raises(RuntimeError, match="not a tensor"):
        b.collect(CheckpointID.CP2, cp2_source=types.SimpleNamespace(vl_encoded=[1.0], state_encoded=torch.zeros(STATE)))
    with pytest.raises(RuntimeError, match="before collect"):
        b.build(CheckpointID.CP2)
    b.collect(CheckpointID.CP2, cp2_source=_source())
    with pytest.raises(ValueError, match="CP2 only"):
        b.build(CheckpointID.CP1)
    # A rejected collect drops any previously held source.
    with pytest.raises(RuntimeError):
        b.collect(CheckpointID.CP2)
    assert b.cached_data == {}


def test_build_rejects_a_source_longer_than_the_layout():
    b = _builder()
    b.collect(CheckpointID.CP2, cp2_source=_source(n_tokens=TOKEN_LEN + 2))
    with pytest.raises(RuntimeError, match="exceed token_len"):
        b.build(CheckpointID.CP2)
