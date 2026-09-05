"""CP2 sparse ternary projection + key builder (plan §3.2 / §7 unit rows)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openpi.cache.components import cp2_vlm_key_builder as kb
from openpi.cache.types import VLM_OUT, CheckpointID


def test_spec_is_deterministic_shared_and_well_formed():
    a = kb.get_projection_spec(3, 8, 0.25, 64)
    b = kb.get_projection_spec(3, 8, 0.25, 64)
    assert a is b  # process-wide registry: one object per (seed, d, p, D)
    assert a.nnz_per_sign == int(np.floor(0.25 * 64 / 2)) == 8
    for row in range(8):
        pos = set(a.idx_pos[row].tolist())
        neg = set(a.idx_neg[row].tolist())
        assert len(pos) == 8 and len(neg) == 8 and not (pos & neg)
        assert max(pos | neg) < 64
    rebuilt = kb._make_spec(3, 8, 0.25, 64)
    assert rebuilt.digest == a.digest and torch.equal(rebuilt.idx_pos, a.idx_pos)
    other = kb.get_projection_spec(4, 8, 0.25, 64)
    assert other is not a and other.digest != a.digest
    meta = a.meta()
    assert meta["accumulation_dtype"] == "float32" and meta["D"] == 64 and meta["digest"] == a.digest


def test_project_matches_dense_oracle_with_float32_accumulation():
    spec = kb.get_projection_spec(11, 16, 0.2, 100)
    R = kb.dense_projection_matrix(spec)
    h = torch.randn(4, 100) * 3
    out = kb.project(h, spec)
    assert out.dtype == torch.float32 and out.shape == (4, 16)
    assert torch.allclose(out, h @ R.T, atol=1e-5)
    hb = h.to(torch.bfloat16)
    outb = kb.project(hb, spec)
    assert outb.dtype == torch.float32
    # bf16 input, but every gather is accumulated in float32: equal to the
    # float32 oracle applied to the bf16-rounded values.
    assert torch.allclose(outb, hb.float() @ R.T, atol=1e-4)
    assert torch.allclose(kb.project(h[0], spec), out[0])
    with pytest.raises(ValueError):
        kb.project(torch.randn(3, 99), spec)


def test_device_index_copies_are_shared():
    spec = kb.get_projection_spec(5, 4, 0.5, 32)
    a = kb._device_indices(spec, torch.device("cpu"))
    b = kb._device_indices(spec, torch.device("cpu"))
    assert a[0] is b[0] and a[1] is b[1]


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError):
        kb._make_spec(0, 4, 0.001, 64)  # nnz_per_sign would be 0
    with pytest.raises(ValueError):
        kb._make_spec(0, 0, 0.5, 64)  # d < 1


def test_builder_collect_build_clear():
    b = kb.CP2VlmTernaryKeyBuilder(seed=1, d=8, p=0.25, input_dim=64)
    stage2 = SimpleNamespace(prefix_out=torch.randn(1, 4, 16))
    b.collect(CheckpointID.CP2, stage2=stage2, tokenized_prompt=None)
    assert "prefix_out" in b.cached_data
    keys = b.build(CheckpointID.CP2)
    assert set(keys) == {VLM_OUT}
    k = keys[VLM_OUT]
    assert k.shape == (8,) and k.dtype == torch.float32 and k.device.type == "cpu"
    spec = kb.get_projection_spec(1, 8, 0.25, 64)
    assert torch.allclose(k, kb.project(stage2.prefix_out.reshape(1, -1), spec)[0])
    assert b.projection_meta() == spec.meta()
    b.clear()
    assert b.cached_data == {}


def test_builder_rejects_other_checkpoints_and_missing_capture():
    b = kb.CP2VlmTernaryKeyBuilder(seed=1, d=8, p=0.25, input_dim=64)
    with pytest.raises(ValueError):
        b.collect(CheckpointID.CP1, stage1=SimpleNamespace())
    with pytest.raises(ValueError):
        b.collect(CheckpointID.CP2)
    with pytest.raises(RuntimeError, match="prefix_out is None"):
        b.collect(CheckpointID.CP2, stage2=SimpleNamespace(prefix_out=None))
    b.collect(CheckpointID.CP2, stage2=SimpleNamespace(prefix_out=torch.randn(2, 4, 16)))
    with pytest.raises(ValueError, match="B=1"):
        b.build(CheckpointID.CP2)
    with pytest.raises(ValueError):
        b.build(CheckpointID.CP3)
    b.collect(CheckpointID.CP2, stage2=SimpleNamespace(prefix_out=torch.randn(1, 4, 17)))
    with pytest.raises(ValueError, match="expected input dim"):
        b.build(CheckpointID.CP2)
