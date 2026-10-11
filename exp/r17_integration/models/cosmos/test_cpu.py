"""Semantic CPU gates; no policy weights, CUDA calls or simulator required."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from exp.r17_integration.models.cosmos.wire_policy import native_observation, CosmosWirePolicy
from exp.r17_integration.models.cosmos.manifest import make_manifest, make_sampler
from exp.r17_integration.models.cosmos.protocols import PROPRIO_KEY, benchmark_profile
from exp.r17_integration.models.cosmos.key_builder import CosmosKeyBuilder
from exp.r17_integration.common.contracts import RequestContext


def test_lossless_current_wire_and_official_flip():
    raw=np.arange(256*256*3,dtype=np.uint32).reshape(256,256,3).astype(np.uint8)
    state=np.array([.025,.024,.2,.3,.4,.5,.6,.7,.1],np.float64)
    wire={'observation/image':raw[::-1,::-1], 'observation/wrist_image':(255-raw)[::-1,::-1],
          PROPRIO_KEY:state,'observation/state':np.zeros(8),'prompt':'test'}
    native,_=native_observation(wire)
    assert native['primary_image'].tobytes()==raw[::-1].tobytes()
    assert native['wrist_image'].tobytes()==(255-raw)[::-1].tobytes()
    assert native['proprio'].dtype==state.dtype and native['proprio'].tobytes()==state.tobytes()
    native['proprio'][0]=99
    assert wire[PROPRIO_KEY][0] != 99
    del wire[PROPRIO_KEY]
    with pytest.raises(ValueError,match='lossless'): native_observation(wire)


def test_keys_depend_only_on_actual_current_cameras_and_state():
    manifest=make_manifest({}, {'release':'test'}, ('test_protocol',))
    builder=CosmosKeyBuilder(manifest.spec.feature_spec)
    latent=torch.zeros(1,16,9,28,28)
    latent[:,:,3]=3;latent[:,:,2]=2;latent[:,:,6:]=99
    state=np.arange(9,dtype=np.float32)
    keys=builder.fields_from_latent(latent,state)
    assert np.all(keys['vision_0']==3) and np.all(keys['vision_1']==2)
    assert np.array_equal(keys['robot_state'],state)
    latent[:,:,0:2]=500;latent[:,:,4:]=700
    assert all(np.array_equal(v,builder.fields_from_latent(latent,state)[k]) for k,v in keys.items())
    with pytest.raises(ValueError):builder.fields_from_latent(latent[:,:,:8],state)


def test_native_schedule_and_fail_closed_unsupported_options():
    from exp.r17_integration.models.cosmos.provider import validate_config
    from exp.r17_integration.common.qualification import config_for_probe
    from exp.r17_integration.common.provider import AdapterProvider
    for k in (1,2,3,5):
        schedule=make_sampler(k)
        assert len(schedule.native_timesteps)==k
        assert abs(schedule.native_timesteps[0]-80)<1e-10
    k2=make_sampler(2)
    assert k2.sampler_id=='cosmos_edm_2ab_k2_incl_clean'
    assert k2.config['expected_nfe']==1 and k2.config['arm_label']=='K2（含清理步）'
    assert k2.config['evaluated_sigma_indices']==[0]
    assert k2.identity != make_sampler(1).identity
    with pytest.raises(ValueError):make_sampler(4)
    model=make_manifest({}, {'release':'test'}, ('fake_cosmos',)).spec
    provider=AdapterProvider('cpu_cosmos',model,lambda:None,make_sampler)
    config=config_for_probe(provider,5,16)
    validate_config(config)
    for setting in ('recipe','trace'):
        getattr(config,setting).enabled=True
        with pytest.raises(ValueError,match='unsupported'):validate_config(config)
        getattr(config,setting).enabled=False
    for setting in ('native_warm_resume','cp2','recipe','split','batch_stage1','cp1_retrieval'):
        with pytest.raises(ValueError):model.capabilities.require(setting)


def test_protocol_does_not_change_existing_r17_identity():
    from exp.r17_integration.ops.formal_eval import benchmark, parse_args
    original=benchmark()
    before=original.identity
    cosmos=benchmark_profile(original)
    assert original.identity==before and original.replan_steps==5 and original.seed==7
    assert cosmos.replan_steps==16 and cosmos.seed==195 and cosmos.seed_mode=='fixed'
    assert len(cosmos.formal_indices)*len(cosmos.tasks)==400
    assert len(benchmark_profile(original,True).formal_indices)*len(cosmos.tasks)==2000
    assert cosmos.identity != before
    args=parse_args(['--model','cosmos_libero','--prepare-only'])
    assert (args.server_node,args.worker_node,args.port,args.driver_port)==('h100','timan108',23240,23249)
    assert parse_args(['--model','dp_libero','--prepare-only']).port==23160
    with pytest.raises(SystemExit):parse_args(['--model','cosmos_libero','--prepare-only','--port','23160'])
    args=parse_args(['--model','cosmos_libero','--prepare-only','--server-node','h100',
        '--replicas','2','--worker-node','timan108','--workers','12','--worker-memory-mib','6144',
        '--port','23240','--driver-port','23249','--official-50',
        '--suites','libero_spatial','libero_10','--denoising-steps','2'])
    assert args.denoising_steps==2 and args.suites==['libero_spatial','libero_10']


def test_official_cpu_sampler_counts_k2_instead_of_base_sampler_k_plus_one():
    import os
    import subprocess
    vendor=Path('/home/weiland/projects/openpi_ext/lines/cosmos-policy')
    python=vendor/'.venv/bin/python'
    if not python.exists():
        pytest.skip('official Cosmos runtime unavailable')
    code='''import json,sys,torch
sys.path.insert(0,sys.argv[1])
from cosmos_policy.modules.cosmos_sampler import CosmosPolicySampler
from cosmos_policy._src.imaginaire.modules.res_sampler import Sampler
rows=[]
for cls in (CosmosPolicySampler,Sampler):
    for k in (1,2,3,5):
        calls=[]
        def denoise(x,sigma):
            calls.append(sigma.tolist()); return x*.5
        cls()(denoise,torch.ones(1,2),num_steps=k,sigma_min=4,sigma_max=80,solver_option='2ab')
        rows.append(dict(sampler=cls.__name__,k=k,nfe=len(calls),sigmas=calls))
print(json.dumps(rows))
'''
    output=subprocess.run([str(python),'-B','-c',code,str(vendor)],check=True,text=True,
        capture_output=True,timeout=60,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
    rows=json.loads(output.stdout.splitlines()[-1])
    assert [r['nfe'] for r in rows[:4]]==[1,1,3,5]
    assert rows[1]['sigmas']==[[80.0]]
    assert [r['nfe'] for r in rows[4:]]==[2,3,4,6]


def test_controller_dtype_and_single_normalization():
    calls=[]
    def inverse(actions,stats):
        calls.append(actions.copy());return actions.astype(np.float64)*.123456789123+1
    policy=CosmosWirePolicy({},SimpleNamespace(unnormalize_actions=inverse))
    normalized=torch.arange(16*7,dtype=torch.float32).reshape(16,7)
    output=policy.to_wire(normalized,None).actions
    assert len(calls)==1 and output.dtype==np.float64
    assert output.tobytes()==(normalized.numpy().astype(np.float64)*.123456789123+1).tobytes()


def test_loaded_weight_audit_handles_native_activation_wrappers_and_rejects_bad_weights():
    from exp.r17_integration.models.cosmos.loader import audit_loaded_net
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    original=torch.nn.Sequential(torch.nn.Linear(4,3))
    state={'net.'+k:v.clone() for k,v in original.state_dict().items()}
    original[0]=checkpoint_wrapper(original[0])
    assert any('_checkpoint_wrapped_module' in k for k,_ in original.named_parameters())
    audit_loaded_net(original,state)
    state['net.0.weight'][0,0]+=1
    with pytest.raises(ValueError,match='tensor'):audit_loaded_net(original,state)


def test_current_raw_simulator_state_is_recorded_without_axis_angle_reconstruction():
    from exp.r17_integration.benchmarks.libero_bridge import _ObservedEnv
    raw={'robot0_gripper_qpos':np.array([.01,.02]),'robot0_eef_pos':np.array([.1,.2,.3]),
         'robot0_eef_quat':np.array([.4,.5,.6,.7])}
    class Environment:
        def reset(self):return raw
        def set_init_state(self,_):return dict(raw)
        def step(self,_):return dict(raw),0,False,{}
    runner=SimpleNamespace(spec=SimpleNamespace(settle_steps=10))
    env=_ObservedEnv(Environment(),runner)
    assert env.reset() is raw
    assert env.set_init_state(None)==raw
    env.step(None)
    assert env.last_raw_obs is not None
    assert np.array_equal(env.last_raw_obs['robot0_eef_quat'],raw['robot0_eef_quat'])
    expected=env.last_raw_obs['robot0_eef_quat'].copy()
    raw['robot0_eef_quat'][3]=1.
    assert np.array_equal(env.last_raw_obs['robot0_eef_quat'],expected)
