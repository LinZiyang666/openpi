"""Size-factor admission, distinct official cells, live CPU view switches and waiting."""
import asyncio
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
import uuid

import numpy as np
import pytest
import yaml

from exp.r17_integration.ops import cache_study as study, cache_lane as lane
from exp.r17_integration.common.config import build_shared_storage, load_cache_config, RecipeConfig
from exp.r17_integration.common.contracts import digest
from exp.r17_integration.common.provider import AdapterProvider, register_provider, binding_for
from exp.r17_integration.common.recipe import make_recipe_factory, bundle_identity, recipe_identity
from exp.r17_integration.common.serving import make_connection_factory, make_bundle_validator
from exp.r17_integration.common.fake_adapter import FakeAdapter
from exp.r17_integration.search.store import atomic_json

SOURCE=study.HERE/'sizes/STUDY.json'
SIZES=[50,100,200,300,400]


def provider_for(model):
    manifest=study.validate_manifest(SOURCE)
    parent=study.read(Path(manifest['models'][model][0]['library'])/'READY.json')
    _,benchmark=study.profiles(model,parent)
    spec=replace(study.model_spec(parent['binding']['model']),allowed_benchmark_protocols=(benchmark.identity,))
    provider=AdapterProvider('size_cpu_'+uuid.uuid4().hex,spec,lambda:FakeAdapter(spec),lambda k:spec.default_sampler)
    register_provider(provider)
    return manifest,parent,benchmark,provider


def config_for(row,provider):
    document=yaml.safe_load((SOURCE.parent/row['file']).read_text())
    # The actual sealed teacher fits are queried through CPU diagnostic key
    # producers. This does not claim native GPU producer qualification.
    document['adapter_binding']['provider']=provider.provider_id
    from exp.r17_integration.common.config import config_from_dict
    return config_from_dict(document)


def test_complete_size_grid_and_b_only_calibrations():
    manifest=study.validate_manifest(SOURCE)
    assert manifest['library_sizes']==SIZES and not manifest['artifacts_pending']
    assert sum(map(len,manifest['models'].values()))==240
    for rows in manifest['models'].values():
        for row in rows:
            fit=study.read(Path(row['fit'])/'READY.json')
            stall=study.read(Path(row['stall'])/'READY.json')
            assert fit['receipt']['store_digest']==stall['store_digest']==study.read(study.view_path(row)/'VIEW.json')['content_digest']
            assert fit['receipt']['config']['commit_horizon']==row['settings']['commit']
            assert stall['library_size']==row['library_size']
            assert stall['fitted_digest']==row['fit_digest']
            assert stall['calibration_partition']=='B_library_successful_LOEO'
            calibrated=study.read(Path(row['stall'])/'stall.json')['tasks']
            assert sorted(stall['unavailable_tasks'])==sorted(k for k,v in calibrated.items() if not v['available'])


@pytest.mark.parametrize('model,expected',[('dp_libero',40000),('fastwam_libero',40000),('cosmos_libero',20000)])
def test_size_cells_unique_official_a_and_cosmos_aliases(model,expected,tmp_path):
    manifest,_,benchmark,provider=provider_for(model)
    groups=[]
    for row in manifest['models'][model]:
        config=config_for(row,provider)
        target=tmp_path/row['file'];target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(yaml.safe_dump(asdict(config)))
        groups.append(dict(row,sha256=study.file_sha(target),bundle_id=study.size_label(row)+row['suite']+row['group'],
                           bundle_identity=bundle_identity(config),recipe_identity=recipe_identity(config)))
    atomic_json(tmp_path/'model_spec.json',asdict(provider.model_spec))
    atomic_json(tmp_path/'benchmark.json',benchmark.manifest())
    seal=dict(model=asdict(provider.model_spec),benchmark_protocol=benchmark.identity,groups=groups)
    seal['digest']=digest(seal);atomic_json(tmp_path/'CACHE_STUDY.json',seal)
    campaign=dict(model=model.rsplit('_',1)[0],local_assets=str(tmp_path),endpoint_host='127.0.0.1',public_port=23400,
                  workers=2,run_name='sizescpu',smoke=False)
    _,_,tasks,endpoints=study.plan(campaign)
    assert len(tasks)==len({t.task_uid for t in tasks})==expected
    assert all(t.extra['init_pool']=='A' and t.extra['partition']=='test' for t in tasks)
    assert len(endpoints)==1
    cells={}
    for task in tasks:
        cells.setdefault((task.extra['library_size'],task.experiment,task.extra['cache_group']),set()).add((task.task_id,task.orig_init_state_idx))
    assert len(cells)==(40 if model=='cosmos_libero' else 80)
    assert all(len(c)==500 for c in cells.values())
    assert [tasks[i*expected//5].extra['library_size'] for i in range(5)]==SIZES


@pytest.mark.parametrize('model',study.LIBERO_MODELS)
def test_real_library_socket_size_switch_no_resident_restart(model,monkeypatch):
    from openpi.serving import websocket_policy_server as ws
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from openpi_client import msgpack_numpy
    import websockets.asyncio.client
    manifest,parent,_,provider=provider_for(model)
    configs=[(row,config_for(row,provider)) for row in manifest['models'][model]]
    base=copy.deepcopy(configs[0][1]);base.recipe=RecipeConfig()
    factory=make_connection_factory(provider,base,build_shared_storage(base))
    for key,value in (('_bundles',{}),('_current_bundle',None),('_bundle_version',0)):monkeypatch.setattr(ws,key,value)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    async def run():
        ready=asyncio.Event()
        server=WebsocketPolicyServer(object(),host='127.0.0.1',port=port,concurrent=True,connection_policy_factory=factory,
            bundle_validator=make_bundle_validator(provider),ready_callback=ready.set)
        task=asyncio.create_task(server.run(stop_on_request=True));await asyncio.wait_for(ready.wait(),5)
        packer=msgpack_numpy.Packer();lifetimes=set();views=set()
        async def call(client,payload):
            await client.send(packer.pack(payload));reply=await asyncio.wait_for(client.recv(),30)
            assert isinstance(reply,bytes),reply
            return msgpack_numpy.unpackb(reply)
        try:
            async with websockets.asyncio.client.connect(f'ws://127.0.0.1:{port}',compression=None) as client:
                await client.recv()
                for row,config in configs:
                    label=study.size_label(row)+row['suite']+'_'+row['group']
                    ack=await call(client,dict(__ctrl__='load_cache_config',bundle_id=label,yaml_content=yaml.safe_dump(asdict(config))))
                    assert ack['__ack__']=='load_cache_config',ack
                    assert (await call(client,dict(__ctrl__='select_bundle',bundle_id=label)))['__ack__']=='select_bundle'
                    fitted=make_recipe_factory(provider)(config,None).base
                    tid=int(fitted.task[0]);i=0
                    await call(client,dict(__ctrl__='episode_start',__task__=config.recipe.ordered_tasks[tid],__extra__={'task_id':tid,'ack_protocol':'r17.execution.v1'}))
                    index=np.load(study.view_path(row)/'row_indices.npy',allow_pickle=False)[i]
                    obs={field:np.load(Path(row['library'])/parent['arrays']['key_'+field]['file'],mmap_mode='r',allow_pickle=False)[index].copy()
                         for field in provider.model_spec.feature_spec.fields if field!='robot_state'}
                    obs['state']=fitted.state[i].copy()
                    obs['__request__']=dict(campaign='cpu_size_view_switch',task_name=config.recipe.ordered_tasks[tid],
                        original_init_idx=0,seed=7,explicit_seed=18,task_id=tid,decision_seq=0,attempt=1,
                        arm=label,bundle_generation=label,benchmark_protocol=config.adapter_binding['benchmark_protocol'],
                        task_table_identity=config.recipe.task_table_identity)
                    out=await call(client,obs);hit=out['__hit_meta__']
                    assert hit['source']=='library' and hit['actual_nfe']==0
                    lifetimes.add((hit['resident_pid'],hit['resident_generation']));views.add(config.recipe.store_digest)
                    end=await call(client,dict(__ctrl__='episode_end',__execution__={'decision_seq':0,'executed_count':2,'terminal':True}))
                    assert end['execution']['terminal']
                assert len(lifetimes)==1 and len(views)==10
        finally:server.request_stop();await asyncio.wait_for(task,5)
    asyncio.run(run())


def test_predecessor_waits_for_exit_and_requires_closed_pass(tmp_path,monkeypatch):
    pred=tmp_path/'old';pred.mkdir();(pred/'.lane.lock').touch()
    atomic_json(pred/'LANE_PLAN.json',dict(lane='4090'))
    record=lane.predecessor(pred,'4090',tmp_path/'new')
    atomic_json(pred/'LANE_STATE.json',dict(status='PASS',results=[]))
    atomic_json(pred/'LANE_RECEIPT.json',dict(status='PASS',all_owned_closed=True))
    activity=iter([[dict(pid=123,start_ticks='birth')],[]]);waits=[]
    monkeypatch.setattr(lane,'lane_processes',lambda _:next(activity))
    monkeypatch.setattr(lane.time,'sleep',lambda seconds:waits.append(seconds))
    gate=lane.wait_predecessor(record,interval=1);gate.close()
    assert waits==[1]
    monkeypatch.setattr(lane,'lane_processes',lambda _:[])
    atomic_json(pred/'LANE_RECEIPT.json',dict(status='FAIL',all_owned_closed=True))
    with pytest.raises(ValueError,match='predecessor did not finish'):lane.wait_predecessor(record)


def test_size_jobs_keep_topology_and_budget_each_size():
    jobs=lane.jobs('4090',SOURCE,'csz',size_count=5)+lane.jobs('h100',SOURCE,'csz',size_count=5)
    from exp.r17_integration.ops.formal_eval import parse_args
    for job in jobs:
        args=parse_args(job['argv']+['--prepare-only'])
        assert args.seconds==432000 and args.worker_node=='timan107'
        assert args.worker_max_episodes==4 and args.worker_restart_limit==2 and args.replicas==2
        assert args.workers==lane.PROFILES[job['model']]['workers']


def test_refresh_in_fresh_cpu_process_registers_metadata_without_model_load(tmp_path):
    import os
    import subprocess
    import sys
    manifest=study.validate_manifest(SOURCE)
    target=tmp_path/'STUDY.json';target.write_bytes(SOURCE.read_bytes())
    for rows in manifest['models'].values():
        for row in rows:
            p=tmp_path/row['file'];p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes((SOURCE.parent/row['file']).read_bytes())
    code=('from pathlib import Path; from exp.r17_integration.ops import cache_study as s; '
          's.refresh(Path('+repr(str(target))+')); '
          'assert not s.validate_manifest(Path('+repr(str(target))+'))["artifacts_pending"]')
    result=subprocess.run([sys.executable,'-c',code],cwd=study.ROOT,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('name',['csl10d','csl10f','csl10c'])
def test_original_size500_episode_plan_remains_byte_equivalent(name):
    from exp.r17_integration.conductor.manifest import load_benchmark_spec
    from exp.r17_integration.common.benchmarks import register_benchmark
    from exp.r17_integration.common.provider import get_provider
    directory=lane.RUN_ROOT/name
    campaign=study.read(directory/'campaign.json')
    assets=Path(campaign['local_assets'])
    spec=study.model_spec(study.read(assets/'model_spec.json'))
    register_benchmark(load_benchmark_spec(assets/'benchmark.json'))
    seal=study.read(assets/'CACHE_STUDY.json')
    document=yaml.safe_load((assets/seal['groups'][0]['file']).read_text())
    ident=document['adapter_binding']['provider']
    provider=AdapterProvider(ident,spec,lambda:None,lambda k:spec.default_sampler)
    try:get_provider(ident)
    except ValueError:register_provider(provider)
    _,_,tasks,_=study.plan(campaign)
    original=study.read(directory/'EPISODE_PLAN.json')['tasks']
    assert [asdict(t) for t in tasks]==original
    assert all('library_size' not in t.extra for t in tasks)


def test_cpu_prepare_recovers_recorded_child_without_duplicate_dispatch(tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_prestage, formal_queue
    directory=tmp_path/'lane';directory.mkdir()
    destination=tmp_path/'campaign';destination.mkdir()
    atomic_json(destination/'PREPARED.json',dict(status='PASS',cpu_only=True))
    job=lane.jobs('4090',SOURCE,'cpuq',size_count=5)[0];job['run']=str(destination)
    atomic_json(directory/'LANE_PLAN.json',dict(lane='4090',long_dp=10,long_fast=10,study=str(SOURCE),
        study_sha256=study.file_sha(SOURCE),jobs=[job]))
    atomic_json(directory/'QUEUE_OWNED.json',[dict(phase='prepare',pid=123,start_ticks='birth',campaign=job['name'])])
    polls=[];waits=[]
    def alive(entry):
        polls.append(entry);return len(polls)==1
    monkeypatch.setattr(lane,'alive',alive)
    monkeypatch.setattr(lane.time,'sleep',lambda seconds:waits.append(seconds))
    monkeypatch.setattr(formal_prestage,'validate_prepared',lambda p:(dict(cache_study=str(SOURCE)),{}))
    def refuse(*args):raise AssertionError('duplicate CPU/GPU dispatch')
    monkeypatch.setattr(formal_queue.Runner,'invoke',refuse)
    assert lane.main(['--lane','4090','--directory',str(directory),'--prefix','cpuq','--study',str(SOURCE),'--prepare-only'])==0
    assert waits==[30]
    receipt=study.read(directory/'LANE_RECEIPT.json');assert receipt['cpu_only'] and receipt['all_owned_closed']
    assert study.read(directory/'CPU_PREPARE_RECOVERY.json')['artifacts_verified']
