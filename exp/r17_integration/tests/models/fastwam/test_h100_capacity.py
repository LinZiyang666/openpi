from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from exp.r17_integration.common.serving import full_call
from exp.r17_integration.common.contracts import CaptureSpec
from exp.r17_integration.models.fastwam.batch_runtime import BatchConfig, BatchLedger, create_runtime
from exp.r17_integration.models.fastwam.ops.h100_capacity import summarize, required_memory
from exp.r17_integration.tests.models.fastwam.conftest import make_adapter, observation, request


@pytest.mark.parametrize('batch',[16,24,32])
def test_large_opt_in_native_stages_preserve_rows_and_private_noise(batch,tmp_path):
    a=make_adapter('libero',allow_unqualified_batch=False)
    obs=[observation('libero',i) for i in range(3)]
    req=[request(i) for i in range(3)]
    sampler=a.sampler_for_steps(10)
    refs=[full_call(a,o,r,sampler=sampler)[0] for o,r in zip(obs,req)]
    a.batch_config=BatchConfig(batch,5)
    a.batch_ledger=BatchLedger(tmp_path/'ledger',a.batch_config)
    prepared=[a.prepare(obs[i%3],req[i%3]) for i in range(batch)]
    stages=a.run_stage2_batch(a.run_stage1_batch(prepared))
    from exp.r17_integration.common.request_rng import RequestRNG
    noise=torch.stack([torch.randn((32,7),generator=RequestRNG(req[i%3]).torch_generator(device='cpu'))
                       for i in range(batch)])
    outputs=a._sample_batch(stages,sampler,noise,0,[CaptureSpec()]*batch,
                            [RequestRNG(req[i%3]).receipt() for i in range(batch)])
    for i,out in enumerate(outputs):
        torch.testing.assert_close(out.normalized_chunk,refs[i%3].normalized_chunk,rtol=0,atol=0)
    before=outputs[-1].normalized_chunk.clone()
    with torch.inference_mode():
        outputs[0].normalized_chunk.zero_()
    assert torch.equal(before,outputs[-1].normalized_chunk)
    with pytest.raises(ValueError,match='study limit'):
        a.run_stage1_batch(prepared+[prepared[0]])


def test_completion_window_excludes_priming_and_drain_and_counts_decisions():
    rows=[dict(finished_epoch=t,latency_ms=l,max_diff=d) for t,l,d in
          [(1,999,.01),(3,20,.02),(4,40,.03),(5,1000,.04)]]
    result=summarize(rows,3,5)
    assert result['calls']==2
    assert result['calls_per_second']==result['decisions_per_second']==1
    assert result['p50_ms']==30 and result['p90_ms']==38
    assert result['max_diff']==.04


def test_replica_admission_uses_target_b16_workspace_and_context_headroom():
    assert required_memory(2,32596)==71336
    assert required_memory(3,32596)==104956
    assert required_memory(2,32596)<81081<required_memory(3,32596)


def test_large_limits_leave_default_off_unchanged(tmp_path):
    a=make_adapter('libero',allow_unqualified_batch=False)
    old=full_call(a,observation('libero'),request())[0].normalized_chunk.numpy().tobytes()
    assert create_runtime(a,ledger_path=tmp_path/'absent') is None
    assert not list(tmp_path.iterdir()) and a.batch_config is None
    assert old==full_call(a,observation('libero'),request())[0].normalized_chunk.numpy().tobytes()
