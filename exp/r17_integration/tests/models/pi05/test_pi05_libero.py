"""LIBERO geometry over the same native sampler stub used by MT50."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import RequestContext
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.common.serving import full_call
from exp.r17_integration.models.pi05.adapter import Pi05Adapter, model_spec, sampler_for_steps, norm_path
from exp.r17_integration.ops.formal_eval import benchmark
from exp.r17_integration.tests.models.pi05.test_pi05 import ToyModel


class LiberoModel(ToyModel):
    def __init__(self):
        super().__init__()
        self.config = NS(action_horizon=10, action_dim=32)

    def run_stage1(self, obs):
        output = super().run_stage1(obs)
        # Keep distinguishable image slots, with the actual wrist driving slot1.
        output.prefix_embs[:, 256:512] = obs.images['left_wrist_0_rgb'].float().mean()
        return output


def fixture_adapter():
    from openpi.models.model import ModelType
    from openpi.policies.policy import Policy
    from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs
    from openpi.transforms import Normalize, Unnormalize, PadStatesAndActions
    from openpi.shared.normalize import NormStats
    stats = {name:NormStats(mean=np.zeros(width),std=np.ones(width),q01=np.full(width,-2.),q99=np.full(width,3.))
             for name,width in [('state',8),('actions',7)]}
    model = LiberoModel()
    policy = Policy(model,is_pytorch=True,pytorch_device='cpu',
        transforms=[LiberoInputs(ModelType.PI05),Normalize(stats,use_quantiles=True),PadStatesAndActions(32),
                    lambda x:{k:v for k,v in x.items() if k != 'prompt'}],
        output_transforms=[Unnormalize(stats,use_quantiles=True),LiberoOutputs()])
    return Pi05Adapter(policy,model_spec('cpu-weights','cpu-norm',benchmark='libero',benchmark_spec=benchmark(True))), model


def observation():
    return {'observation/image':np.full((256,256,3),3,np.uint8),
            'observation/wrist_image':np.full((256,256,3),11,np.uint8),
            'observation/state':np.arange(8,dtype=np.float32)/10,'prompt':'pick up the bowl'}


def request(adapter):
    b = benchmark(True)
    return RequestContext('cpu',b.identity,b.ordered_tasks[0],0,0,task_id=0,explicit_seed=123)


@pytest.mark.parametrize('k',[1,2,3,4,5,10])
def test_native_h10_sampler_and_libero_inverse(k):
    adapter,model=fixture_adapter(); obs=observation(); ctx=request(adapter)
    adapter.policy._staged_inference = k == 10
    adapter.policy._sample_kwargs = {'num_steps':k}
    legacy=adapter.policy.infer(obs,noise=RequestRNG(ctx).normal((10,32)))['actions']
    model.counts[:]=[0,0,0]
    result,wire=full_call(adapter,obs,ctx,sampler=sampler_for_steps(k))
    assert model.counts==[1,1,k] and result.actual_nfe==k
    assert wire.actions.shape==(10,7) and result.normalized_chunk.shape==(10,32)
    np.testing.assert_array_equal(wire.actions,legacy)
    np.testing.assert_allclose(wire.actions,(result.normalized_chunk[:,:7]+1)/2*(5+1e-6)-2,atol=1e-6)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_historical_two_camera_pool_and_state_only(dtype):
    from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
    from openpi.cache.types import CheckpointID
    adapter,model=fixture_adapter(); obs=observation(); ctx=request(adapter)
    s1=adapter.run_stage1(adapter.prepare(obs,ctx)); fields=adapter.retrieval_fields(s1)
    s1.data.prefix_embs=s1.data.prefix_embs.to(dtype)
    fields=adapter.retrieval_fields(s1)
    builder=CP1SpatialPool16KeyBuilder(enabled_fields=['vision_0','vision_1','robot_state'])
    builder.collect(CheckpointID.CP1,stage1=s1.data)
    historical=builder.build(CheckpointID.CP1)
    for name in ('vision_0','vision_1'): assert torch.equal(fields[name],historical[name])
    assert fields['vision_0'].shape==fields['vision_1'].shape==(32768,)
    np.testing.assert_array_equal(adapter.extract_state_only(obs).state,fields['robot_state'].numpy())
    assert fields['robot_state'].shape==(8,) and model.counts==[1,0,0]
    s1.data.prefix_embs[:,512:]=10000
    for name in fields: assert torch.equal(fields[name],adapter.retrieval_fields(s1)[name])
    adapter.spec.layout.validate_commit(10)
    assert norm_path('unused','libero').is_file()


def test_c10_free_tail_through_formal_connection():
    from exp.r17_integration.models.pi05.provider import register_provider
    from exp.r17_integration.ops.formal_eval import full_config
    from exp.r17_integration.common.config import build_shared_storage
    from exp.r17_integration.common.serving import make_connection_factory
    adapter,model=fixture_adapter(); provider=register_provider(adapter)
    config=full_config(provider,dict(domain='libero',model='pi05',commit_horizon=10))
    core=make_connection_factory(provider,config,build_shared_storage(config))()
    b=benchmark(True)
    core.on_episode_start(extra_metadata={'campaign':'cpu','task_name':b.ordered_tasks[0],'task_id':0,
        'original_init_idx':0,'seed':7,'explicit_seed':123,'ack_protocol':'r17.execution.v1'})
    first=core.infer(observation()); counts=list(model.counts)
    tail=core.infer({**observation(),'__execution__':{'decision_seq':0,'executed_count':5}})
    assert model.counts==counts and tail['__hit_meta__']['stage_calls']==(0,0,0)
    np.testing.assert_array_equal(tail['actions'],first['actions'][5:])
