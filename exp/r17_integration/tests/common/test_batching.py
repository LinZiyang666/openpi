import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import Stage2
from exp.r17_integration.common.fake_adapter import FakeAdapter
from openpi.serving.batching_core import BatchingCore, Stage3MissPayload


def test_real_batch_core_bucket_fault_isolation_capture_and_no_alias():
    adapter=FakeAdapter();batcher=adapter.make_stage_batcher()
    with BatchingCore(batcher,device="cpu",max_batch_size=8,max_wait_ms=5) as core:
        a=Stage3MissPayload(Stage2(np.ones((16,4),np.float32)),torch.zeros(16,4),num_steps=2,save_timesteps=(.5,))
        b=Stage3MissPayload(Stage2(np.full((16,4),2.,np.float32)),torch.zeros(16,4),num_steps=2,save_timesteps=(.5,))
        bad=Stage3MissPayload(Stage2(np.ones((16,4),np.float32)),torch.full((16,4),float("nan")),num_steps=1)
        # Sub-bucket failure never prevents other K/capture buckets replying.
        requests=core.enqueue_to_stage(3,"b",a),core.enqueue_to_stage(3,"b",bad),core.enqueue_to_stage(3,"b",b)
        r1=core.wait_for(requests[0],timeout=5)
        with pytest.raises(ValueError,match="nonfinite"):core.wait_for(requests[1],timeout=5)
        r2=core.wait_for(requests[2],timeout=5)
        assert all(r.reply_event.is_set() for r in requests)
        assert torch.equal(r1.action,torch.ones(1,16,4))
        assert torch.equal(r2.action,torch.full((1,16,4),2.))
        r1.action.zero_();r1.intermediates[.5].zero_()
        assert torch.all(r2.action==2) and torch.all(r2.intermediates[.5]==1)
        assert batcher.bucket_key(a)!=batcher.bucket_key(bad)
