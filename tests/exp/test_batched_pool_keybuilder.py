"""Operator-level unit tests for the ROUND 4 batched-avgpool key builder.

Asserts (1) ``avg_pool2d(k=4,s=4) == adaptive_avg_pool2d((4,4))`` bit-for-bit for the
16→4 integer-divisible case, and (2) ``_batched_spatial_pool`` is bit-identical to the src
per-field ``_spatial_pool_tokens`` for 2 and 3 cameras. The end-to-end build()+verdict
equivalence is covered by r4_equiv_batched.py on the real library. CPU-only, synthetic.
"""

import torch
import torch.nn.functional as F

from openpi.cache.components.key_builder import _spatial_pool_tokens

from exp.cache_latency_bench.opt.r4_pool_keybuilder import _batched_spatial_pool


def test_avgpool_equals_adaptive_16_to_4():
    torch.manual_seed(0)
    x = torch.randn(1, 2048, 16, 16)
    a = F.adaptive_avg_pool2d(x, (4, 4))
    b = F.avg_pool2d(x, kernel_size=4, stride=4)
    assert torch.equal(a, b)  # 16 % 4 == 0 -> non-overlapping 4x4 windows -> bit-identical


def test_batched_pool_bit_equals_src_per_field():
    torch.manual_seed(1)
    t0 = torch.randn(256, 2048)
    t1 = torch.randn(256, 2048)
    src0 = _spatial_pool_tokens(t0, 16, 4)
    src1 = _spatial_pool_tokens(t1, 16, 4)
    stack = torch.stack([t0, t1], dim=0)
    batched = _batched_spatial_pool(stack, 16, 4)
    assert torch.equal(src0, batched[0])  # bit-for-bit, per camera
    assert torch.equal(src1, batched[1])
    assert batched.shape == (2, 4 * 4 * 2048)


def test_batched_pool_three_cameras():
    torch.manual_seed(2)
    stack = torch.randn(3, 256, 2048)
    out = _batched_spatial_pool(stack, 16, 4)
    for i in range(3):
        assert torch.equal(_spatial_pool_tokens(stack[i], 16, 4), out[i])
