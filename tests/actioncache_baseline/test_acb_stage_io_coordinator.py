"""Stage2Output.prefix_out through stack/split and request-aware coordinator capture."""

from __future__ import annotations

import threading

import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import Stage1Output, Stage2Output
from openpi.serving import stage_io
from openpi.serving.batching_coordinator import BatchingCoordinator


def _stage1(b=1, fill=0.0):
    return Stage1Output(
        state=torch.full((b, 4), fill),
        prefix_embs=torch.zeros(b, 2, 8),
        prefix_pad_masks=torch.zeros(b, 2, dtype=torch.bool),
        prefix_att_2d_masks_4d=torch.zeros(b, 1, 2, 2),
        prefix_position_ids=torch.zeros(b, 2, dtype=torch.int64),
    )


def _cache(b=1):
    from transformers import DynamicCache

    c = DynamicCache()
    c.update(torch.zeros(b, 1, 2, 4), torch.zeros(b, 1, 2, 4), layer_idx=0)
    return c


def test_stack_split_prefix_out_three_states():
    a = Stage2Output(stage1=_stage1(fill=1.0), past_key_values=_cache())
    b = Stage2Output(stage1=_stage1(fill=2.0), past_key_values=_cache())
    batched = stage_io.stack_stage2_output([a, b])
    assert batched.prefix_out is None
    shards = stage_io.split_stage2_output(batched, 2)
    assert all(s.prefix_out is None for s in shards)

    pa = torch.full((1, 2, 8), 1.0)
    pb = torch.full((1, 2, 8), 2.0)
    a2 = Stage2Output(stage1=_stage1(fill=1.0), past_key_values=_cache(), prefix_out=pa)
    b2 = Stage2Output(stage1=_stage1(fill=2.0), past_key_values=_cache(), prefix_out=pb)
    batched = stage_io.stack_stage2_output([a2, b2])
    assert batched.prefix_out.shape == (2, 2, 8)
    shards = stage_io.split_stage2_output(batched, 2)
    assert torch.equal(shards[0].prefix_out, pa) and torch.equal(shards[1].prefix_out, pb)
    assert torch.equal(shards[1].stage1.state, torch.full((1, 4), 2.0))

    with pytest.raises(ValueError, match="all-None or all-set"):
        stage_io.stack_stage2_output([a, b2])


def test_stage2output_to_keeps_prefix_out_and_legacy_ctor():
    legacy = Stage2Output(stage1=_stage1(), past_key_values=_cache())
    assert legacy.prefix_out is None and legacy.to("cpu").prefix_out is None
    withp = Stage2Output(stage1=_stage1(), past_key_values=_cache(), prefix_out=torch.ones(1, 2, 8))
    assert torch.equal(withp.to("cpu").prefix_out, torch.ones(1, 2, 8))


class _CaptureStub:
    """Stage-2 stub that records which variant ran and marks captured outputs."""

    def __init__(self):
        self.calls: list[str] = []
        self._fake_param = torch.zeros(1)

    def parameters(self):
        return iter([self._fake_param])

    def run_stage1(self, obs):
        b = obs["state"].shape[0]
        return _stage1(b)

    def run_stage2(self, stage1):
        b = stage1.state.shape[0]
        self.calls.append(f"plain:{b}")
        return Stage2Output(stage1=stage1, past_key_values=_cache(b))

    def run_stage2_capture(self, stage1):
        b = stage1.state.shape[0]
        self.calls.append(f"capture:{b}")
        return Stage2Output(stage1=stage1, past_key_values=_cache(b),
                            prefix_out=stage1.state[:, :2].reshape(b, 1, 2).clone())


def _submit_parallel(bc, flags: list[bool], bundle_ids: list[str] | None = None):
    n = len(flags)
    barrier = threading.Barrier(n)
    results: list = [None] * n
    bundle_ids = bundle_ids or ["default"] * n

    def worker(i):
        barrier.wait()
        results[i] = bc.submit_to_stage(
            2, bundle_ids[i], _stage1(fill=float(i)), request_id=f"r{i}",
            requires_stage2_capture=flags[i],
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_coordinator_request_aware_capture():
    stub = _CaptureStub()
    with BatchingCoordinator(stub, device="cpu", max_batch_size=2, max_wait_ms=50.0) as bc:
        # Legacy default: every request False -> plain run_stage2, no capture.
        outs = _submit_parallel(bc, [False, False])
        assert stub.calls == ["plain:2"] and all(o.prefix_out is None for o in outs)
        # Hot-loaded CP2 bundle later in the server's life: its request flips the batch.
        outs = _submit_parallel(bc, [True, True], ["cp2", "cp2"])
        assert stub.calls[-1] == "capture:2" and all(o.prefix_out is not None for o in outs)
        # Mixed CP1 / CP2 batch: one capture forward; the CP1 request keeps its
        # KV and just carries an unused prefix_out; the CP2 one has its own row.
        outs = _submit_parallel(bc, [False, True], ["cp1", "cp2"])
        assert stub.calls[-1] == "capture:2"
        assert outs[0].past_key_values is not None and outs[1].prefix_out is not None
        assert torch.equal(outs[1].prefix_out, torch.full((1, 1, 2), 1.0))
        assert torch.equal(outs[0].prefix_out, torch.full((1, 1, 2), 0.0))


def test_coordinator_same_bundle_id_replacement_in_flight():
    """Old wrapper (CP1, False) and new wrapper (CP2, True) share a bundle id;
    each request carries its own capability, so the in-flight batch serves both."""
    stub = _CaptureStub()
    with BatchingCoordinator(stub, device="cpu", max_batch_size=2, max_wait_ms=50.0) as bc:
        for flags in ([True, False], [False, True]):
            outs = _submit_parallel(bc, flags, ["same", "same"])
            assert stub.calls[-1] == "capture:2"
            for flag, out in zip(flags, outs):
                assert out.past_key_values is not None
                if flag:
                    assert out.prefix_out is not None
