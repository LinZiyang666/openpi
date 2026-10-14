"""Two-view LIBERO through the existing guarded GR00T vendor stub."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from exp.libero_groot.libero_keys import SCALAR_STATE_KEYS, ACTION_KEYS
from exp.libero_groot.policy_adapter import GrootLiberoPolicyAdapter
from exp.r17_integration.common.contracts import RequestContext
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.common.serving import full_call
from exp.r17_integration.models.groot.loader import checkpoint_paths, load_backend
from exp.r17_integration.models.groot.manifest import checkpoint_manifest
from exp.r17_integration.models.groot.provider import GrootMetaWorldAdapter, make_provider
from exp.r17_integration.ops.formal_eval import benchmark
from .conftest import Backend


class LiberoBackend(Backend):
    def apply_transforms(self, obs):
        self.apply_count += 1
        assert obs['video.image'].shape==obs['video.wrist_image'].shape==(1,1,256,256,3)
        state=torch.zeros(1,1,64)
        values=[*SCALAR_STATE_KEYS,'state.gripper']
        raw=np.concatenate([obs[key].reshape(-1) for key in values])
        state[0,0,:8]=torch.tensor(self.normalize_state_only(raw))
        mask=torch.zeros_like(state,dtype=torch.bool); mask[...,:8]=True
        # Different starts than π's fixed slots; locate both runs by mask.
        ids=torch.tensor([[1]*20+[99]*256+[2]*7+[99]*256+[3]*4])
        pixels=torch.tensor([float(obs[key].mean()) for key in ('video.image','video.wrist_image')]).reshape(2,1,1,1)
        return {'eagle_input_ids':ids,'eagle_attention_mask':torch.ones_like(ids),
                'eagle_pixel_values':pixels,'state':state,'state_mask':mask,'embodiment_id':torch.tensor([31])}

    def unapply_transforms(self, data):
        self.inverse_count+=1
        x=data['action'].detach().cpu().numpy()
        return {key:x[...,i:i+1]*.25+.5 for i,key in enumerate(ACTION_KEYS)}

    def get_action(self, obs):
        return {key:value[0] for key,value in super().get_action(obs).items()}


def observation():
    return {'observation/image':np.full((256,256,3),2,np.uint8),
            'observation/wrist_image':np.full((256,256,3),5,np.uint8),
            'observation/state':np.arange(8,dtype=np.float32)/10,'prompt':'pick up the bowl'}


def libero_manifest(manifest,k=4):
    return replace(manifest,benchmark='libero',benchmark_protocol=benchmark(True).identity,denoising_steps=k)


@pytest.mark.parametrize('k',[1,2,3,4])
def test_native_stages_inverse_gripper_and_endpoint_k(guarded_stub,manifest,k):
    guarded_stub()  # Apply the existing fixture's real source-guard pins.
    manifest=libero_manifest(manifest,k); backend=LiberoBackend(k)
    adapter=GrootMetaWorldAdapter(backend,manifest); obs=observation()
    ctx=RequestContext('cpu',manifest.benchmark_protocol,benchmark(True).ordered_tasks[0],0,0,explicit_seed=123)
    result,wire=full_call(adapter,obs,ctx)
    assert result.actual_nfe==backend.model.action_head.nfe==k
    assert result.normalized_chunk.shape==(16,32) and wire.actions.shape==(16,7)
    expected=result.normalized_chunk[:,:7].numpy()*.25+.5
    expected[:,-1]=np.sign(1-2*expected[:,-1])
    np.testing.assert_array_equal(wire.actions,expected)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(RequestRNG(ctx).seed_for())
        historical=GrootLiberoPolicyAdapter(backend).infer(obs)['actions']
    np.testing.assert_array_equal(wire.actions,historical)
    provider=make_provider(backend,manifest)
    assert provider.provider_id=='r17_groot_libero_v1'
    assert provider.sampler_for_steps(k)==manifest.model_spec().default_sampler
    with pytest.raises(ValueError,match='endpoint K'): provider.sampler_for_steps(2 if k!=2 else 4)


def test_mask_pooling_state_and_suite_norm_selection(guarded_stub,manifest):
    from openpi.cache.groot.key_builder import GrootLiberoCP1SpatialPool16KeyBuilder
    from openpi.cache.types import CheckpointID
    guarded_stub(); manifest=libero_manifest(manifest)
    sp,l10=LiberoBackend(),LiberoBackend()
    l10.normalize_state_only=lambda s:np.asarray(s,np.float32)*3-2
    adapter=GrootMetaWorldAdapter({'libero_spatial':sp,'libero_10':l10},manifest)
    obs=observation(); b=benchmark(True)
    for tid,backend in ((0,sp),(30,l10),(0,sp)):
        ctx=RequestContext('cpu',b.identity,b.ordered_tasks[tid],0,0,task_id=tid)
        s1=adapter.run_stage1(adapter.prepare(obs,ctx));fields=adapter.retrieval_fields(s1)
        builder=GrootLiberoCP1SpatialPool16KeyBuilder(['vision_0','vision_1','robot_state'])
        builder.collect(CheckpointID.CP1,stage1=s1.data)
        historical=builder.build(CheckpointID.CP1)
        assert adapter.backend is backend and fields['robot_state'].shape==(8,)
        for key in fields: assert torch.equal(fields[key],historical[key])
        np.testing.assert_array_equal(adapter.extract_state_only(obs).state,fields['robot_state'].numpy())
        assert not torch.equal(fields['vision_0'],fields['vision_1'])
    assert sp.apply_count==2 and l10.apply_count==1


def test_checkpoint_map_and_libero_metadata_loader(tmp_path,monkeypatch):
    from exp.r17_integration.models.groot import loader
    for suite in ('libero_spatial','libero_10'):
        root=tmp_path/('n15_'+suite); (root/'experiment_cfg').mkdir(parents=True)
        (root/'config.json').write_text(json.dumps({'action_head_cfg':{'action_horizon':16,'action_dim':32}}))
        (root/'model.safetensors').write_bytes(b'CPU dummy weights '+suite.encode())
        (root/'experiment_cfg/metadata.json').write_text(json.dumps({'new_embodiment':{'statistics':{'suite':suite}}}))
    token=tmp_path/'tokenizer.json';token.write_text('{}')
    code=tmp_path/'source.py';code.write_text('# CPU stub')
    paths=checkpoint_paths(tmp_path)
    manifest=checkpoint_manifest(paths,tokenizer_files=[token],code_files=[code],benchmark='libero',benchmark_protocol=benchmark(True).identity)
    assert manifest.model_spec().feature_spec.fields=={'vision_0':32768,'vision_1':32768,'robot_state':8}
    loaded=[]
    backend=load_backend(paths,device='cpu',benchmark='libero',denoising_steps=2,
        policy_loader=lambda p,**kw:loaded.append((p,kw)) or object())
    assert set(backend)=={'libero_spatial','libero_10'}
    assert all(kw=={'device':'cpu','denoising_steps':2} for _,kw in loaded)
    assert checkpoint_paths(tmp_path,suite='libero_10')=={'libero_10':paths['libero_10']}
