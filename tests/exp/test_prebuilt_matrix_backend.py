"""Synthetic-data unit tests for PrebuiltMatrixBackend (ROUND 1 stack elimination).

Asserts the prebuilt-matrix fast path is BIT-IDENTICAL to the parent
``InMemoryBackend._compute_field_scores`` (cosine + l2), is robust to candidate
re-ordering (id->row gather), and falls back to the parent for every non-safe-window
input. CPU-only, tiny synthetic buckets, no 1.1 GB library.
"""

from types import SimpleNamespace

import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend

from exp.cache_latency_bench.opt.prebuilt_matrix_backend import PrebuiltMatrixBackend


def _entry(eid, task_key, query_keys, checkpoint_id="cp1"):
    return SimpleNamespace(
        id=eid,
        checkpoint_id=checkpoint_id,
        payload=SimpleNamespace(task_key=task_key),
        query_keys=query_keys,
    )


def _build(entries, dims):
    b = PrebuiltMatrixBackend(dims)
    b._entries = {e.id: e for e in entries}
    b._prebuild_task_matrices()
    return b


def _parent(backend, q, candidates, field, sim_cfg):
    # Call the UN-overridden parent implementation explicitly (unbound dispatch).
    return InMemoryBackend._compute_field_scores(backend, q, candidates, field, sim_cfg)


def test_cosine_fast_path_bit_identical():
    torch.manual_seed(0)
    d = 8
    entries = [_entry(f"e{i}", "taskA", {"vision_0": torch.randn(d)}) for i in range(5)]
    b = _build(entries, {"vision_0": d})
    cands = list(b._entries.values())
    q = torch.randn(d)
    fs, fm = b._compute_field_scores(q, cands, "vision_0", {"type": "cosine"})
    bs, bm = _parent(b, q, cands, "vision_0", {"type": "cosine"})
    assert torch.equal(fs, bs)  # gather+contiguous == stack, bit-for-bit
    assert torch.equal(fm, bm)


def test_l2_fast_path_bit_identical():
    torch.manual_seed(1)
    d = 6
    entries = [_entry(f"e{i}", "taskA", {"robot_state": torch.randn(d)}) for i in range(4)]
    b = _build(entries, {"robot_state": d})
    cands = list(b._entries.values())
    q = torch.randn(d)
    fs, fm = b._compute_field_scores(q, cands, "robot_state", {"type": "l2"})
    bs, bm = _parent(b, q, cands, "robot_state", {"type": "l2"})
    assert torch.equal(fs, bs)
    assert torch.equal(fm, bm)


def test_fast_path_robust_to_candidate_reordering():
    # Row map is by id; the gather must reproduce the parent stack content even when
    # candidate order differs from the prebuild (insertion) order — the R3 row-order trap.
    torch.manual_seed(2)
    d = 5
    entries = [_entry(f"e{i}", "taskA", {"vision_0": torch.randn(d)}) for i in range(6)]
    b = _build(entries, {"vision_0": d})
    cands = list(b._entries.values())[::-1]  # reversed
    q = torch.randn(d)
    fs, _ = b._compute_field_scores(q, cands, "vision_0", {"type": "cosine"})
    bs, _ = _parent(b, q, cands, "vision_0", {"type": "cosine"})
    assert torch.equal(fs, bs)


def test_fallback_cross_bucket():
    # Candidates spanning two task buckets => no single bucket => parent fallback,
    # result still equals a pure-parent compute on the mixed candidate set.
    torch.manual_seed(3)
    d = 4
    ea = [_entry(f"a{i}", "taskA", {"vision_0": torch.randn(d)}) for i in range(3)]
    eb = [_entry(f"b{i}", "taskB", {"vision_0": torch.randn(d)}) for i in range(3)]
    b = _build(ea + eb, {"vision_0": d})
    mixed = ea[:2] + eb[:2]
    q = torch.randn(d)
    fs, fm = b._compute_field_scores(q, mixed, "vision_0", {"type": "cosine"})
    bs, bm = _parent(b, q, mixed, "vision_0", {"type": "cosine"})
    assert torch.equal(fs, bs)
    assert torch.equal(fm, bm)


def test_fallback_id_absent_from_bucket():
    # A candidate whose id was never prebuilt => fallback to parent.
    torch.manual_seed(4)
    d = 4
    entries = [_entry(f"e{i}", "taskA", {"vision_0": torch.randn(d)}) for i in range(3)]
    b = _build(entries, {"vision_0": d})
    ghost = _entry("ghost", "taskA", {"vision_0": torch.randn(d)})  # not in prebuild
    cands = list(b._entries.values()) + [ghost]
    q = torch.randn(d)
    fs, fm = b._compute_field_scores(q, cands, "vision_0", {"type": "cosine"})
    bs, bm = _parent(b, q, cands, "vision_0", {"type": "cosine"})
    assert torch.equal(fs, bs)
    assert torch.equal(fm, bm)


def test_fallback_unknown_sim_type_raises_like_parent():
    d = 4
    entries = [_entry(f"e{i}", "taskA", {"vision_0": torch.randn(d)}) for i in range(3)]
    b = _build(entries, {"vision_0": d})
    cands = list(b._entries.values())
    q = torch.randn(d)
    # sim_type not in {cosine,l2} -> fast path declines -> parent raises ValueError.
    with pytest.raises(ValueError):
        b._compute_field_scores(q, cands, "vision_0", {"type": "dot"})


def test_empty_valid_returns_zeros():
    d = 4
    entries = [_entry(f"e{i}", "taskA", {"vision_0": torch.randn(d)}) for i in range(3)]
    b = _build(entries, {"vision_0": d})
    cands = list(b._entries.values())
    q = torch.randn(d)
    # No candidate carries 'robot_state' => valid empty => zeros (parent contract).
    fs, fm = b._compute_field_scores(q, cands, "robot_state", {"type": "l2"})
    assert torch.equal(fs, torch.zeros(len(cands)))
    assert torch.equal(fm, torch.zeros(len(cands)))
