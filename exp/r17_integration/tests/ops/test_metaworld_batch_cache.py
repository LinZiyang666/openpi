"""CPU MT50 residents, private request RNG, committed tails and YAML plans."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import uuid

import numpy as np
import pytest
import torch
import yaml

from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
from exp.r17_integration.common.benchmarks import register_benchmark
from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.common.contracts import RequestContext, digest
from exp.r17_integration.common.serving import full_call,make_connection_factory,make_bundle_validator
from exp.r17_integration.ops import cache_study, formal_eval


def adapter_for(model):
    if model == 'dp':
        from exp.r17_integration.tests.models.dp.conftest import make_adapter
        adapter = make_adapter.__wrapped__()('metaworld')
    else:
        from exp.r17_integration.tests.models.fastwam.conftest import make_adapter
        adapter = make_adapter('metaworld', allow_unqualified_batch=False)
        adapter.manifest = replace(adapter.manifest, spec=replace(adapter.metadata(),
            allowed_benchmark_protocols=(FORMAL_SPEC.identity,)))
    return adapter


def wire(i=0):
    return {'observation/image': np.full((480,480,3), 30+70*i, np.uint8),
            'observation/state': np.array([.1,.2,.3,.4], np.float32)+i*.03,
            'prompt': FORMAL_SPEC.tasks[i][1]}


def context(i=0):
    return RequestContext('mw_cpu', FORMAL_SPEC.identity, FORMAL_SPEC.tasks[i][0], i, 0,
                          task_id=i, explicit_seed=18+19*i)


def episode(i=0, *, capture=False):
    return dict(campaign='mw_cpu', task_uid='mw_cpu_'+str(i), task_name=FORMAL_SPEC.tasks[i][0],
        task_id=i, original_init_idx=i, orig_init_idx=i, attempt=1, seed=8 if capture else 7,
        explicit_seed=18+19*i, ack_protocol='r17.execution.v1', partition='library' if capture else 'test',
        split='library' if capture else 'test', init_pool='seed8' if capture else 'A',
        library_episode_id=i, task_data_hash='cpu_task_data_'+str(i), arm='teacher' if capture else 'full')


@pytest.mark.parametrize('model,k', [('dp',5),('dp',10),('dp',100),('fastwam',1),
    ('fastwam',2),('fastwam',10),('pi05',1),('pi05',2),('pi05',10),('groot',1),('groot',2),('groot',4)])
def test_formal_admits_mw_steps(model,k):
    args = formal_eval.parse_args(['--model', model+'_mw', '--denoising-steps',str(k),
                                  '--replicas','1','--prepare-only'])
    assert args.denoising_steps == k


@pytest.mark.parametrize('model', ['dp','fastwam'])
def test_formal_admits_mw_batch_mps_horizon(model):
    args = formal_eval.parse_args(['--model',model+'_mw','--max-batch','4','--mps',
        '--commit-horizon','10','--replicas','2','--prepare-only'])
    assert (args.max_batch,args.mps,args.commit_horizon) == (4,True,10)
    assert formal_eval.execution_horizon(dict(model=model,domain='metaworld',commit_horizon=10),
        dict(replan_steps=5,horizon=16)) == 10
    with pytest.raises(SystemExit):
        formal_eval.parse_args(['--model',model+'_mw','--commit-horizon','11','--prepare-only'])


@pytest.mark.parametrize('model', ['dp','fastwam','pi05','groot'])
def test_formal_admits_mw_cache(model):
    args = formal_eval.parse_args(['--model',model+'_mw','--cache-study','STUDY.json',
                                  '--replicas','1','--prepare-only'])
    assert args.cache_study == Path('STUDY.json') and args.driver_mode == 'group-commit'


@pytest.mark.parametrize('model', ['dp','fastwam'])
def test_fixed_mw_observations_batch_difference_information_only(model,tmp_path):
    register_benchmark(FORMAL_SPEC)
    adapter=adapter_for(model)
    k=100 if model=='dp' else 10
    from importlib import import_module
    runtime_module=import_module('exp.r17_integration.models.'+model+'.batch_runtime')
    provider_module=import_module('exp.r17_integration.models.'+model+'.provider')
    provider=provider_module.register_provider(adapter,provider_id='mw_parity_'+uuid.uuid4().hex)
    sampler=provider.sampler_for_steps(k)
    references=[full_call(adapter,wire(i),context(i),sampler=sampler)[1].actions for i in range(3)]
    runtime=runtime_module.create_runtime(adapter,3,50,tmp_path/'batches.jsonl')
    try:
        if model=='dp':
            from exp.r17_integration.models.dp.provider import CoordinatedAdapter
            clients=[CoordinatedAdapter(provider.load_adapter(),runtime.core) for _ in range(3)]
        else:
            from exp.r17_integration.models.fastwam.batcher import FastWAMBatchedAdapter
            clients=[FastWAMBatchedAdapter(provider.load_adapter(),runtime.core) for _ in range(3)]
        barrier=threading.Barrier(3)
        def infer(i):
            barrier.wait(timeout=10)
            return full_call(clients[i],wire(i),context(i),sampler=sampler)[1].actions
        with ThreadPoolExecutor(max_workers=3) as pool:
            batched=list(pool.map(infer,range(3)))
        differences=[float(np.max(np.abs(a.astype(np.float64)-b))) for a,b in zip(references,batched)]
        print('MW_CPU_PARITY '+json.dumps(dict(model=model,K=k,B=3,max_action_difference=differences)))
        assert all(x.shape==y.shape and np.isfinite(y).all() for x,y in zip(references,batched))
        rows=[json.loads(line) for line in (tmp_path/'batches.jsonl').read_text().splitlines()]
        assert any(r.get('size')==3 for r in rows)
    finally:
        runtime.close()


@pytest.mark.parametrize('model', ['dp','fastwam'])
@pytest.mark.parametrize('capture,capture_images', [(False,False),(True,False),(True,True)])
def test_batched_mw_resident_serves_fake_requests(model,capture,capture_images,tmp_path,monkeypatch):
    from importlib import import_module
    from exp.r17_integration.common import websocket_policy_server as ws
    from exp.r17_integration.qualification import model_runtime
    register_benchmark(FORMAL_SPEC)
    adapter=adapter_for(model)
    provider=import_module('exp.r17_integration.models.'+model+'.provider').register_provider(
        adapter,provider_id='mw_resident_'+uuid.uuid4().hex)
    campaign=dict(model=model,domain='metaworld',assets=str(tmp_path),device='cuda:0',
                  memory_budget_gib=8,replicas=1,server_node='local',library=capture,capture_images=capture_images)
    batch_module=import_module('exp.r17_integration.models.'+model+'.batch_runtime')
    campaign[model+'_batching']=batch_module.BatchConfig(2,50).metadata()
    cfg=formal_eval.full_config(provider,campaign)
    (tmp_path/'full.yaml').write_text(yaml.safe_dump(asdict(cfg)))
    monkeypatch.setattr(model_runtime,'load',lambda *args:(adapter,provider))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *args:SimpleNamespace(total_memory=16*2**30))
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    # Exercise graph bootstrap domain/shape selection with CPU graph stand-ins.
    if model=='dp':
        np.savez(tmp_path/'wire_observations.npz',**{k:np.stack([wire()[k]]) for k in wire()})
        from exp.r17_integration.models.dp import batch_graph
        def check_graph(a,observations,*,steps):
            assert observations[0]['observation/state'].shape==(4,)
            assert 'observation/wrist_image' not in observations[0]
            assert steps==(100,)
            return dict(graph=object(),passed=True)
        monkeypatch.setattr(batch_graph,'check_graph',check_graph)
        monkeypatch.setattr(batch_graph,'warm_batch_signatures',lambda *args:dict(batch_sizes=[1,2]))
    results=[]
    class Server:
        def __init__(self,policy,**kwargs):self.kwargs=kwargs
        def request_stop(self):pass
        def serve_forever(self,**kwargs):
            metadata=self.kwargs['metadata']
            assert metadata['debug_capture'] is capture
            assert metadata['formal_pure_inference'] is (not capture)
            clients=[self.kwargs['connection_policy_factory']() for _ in range(2)]
            for i,c in enumerate(clients):c.on_episode_start(extra_metadata=episode(i,capture=capture))
            with ThreadPoolExecutor(max_workers=2) as pool:
                results.extend(pool.map(lambda i:clients[i].infer(wire(i)),range(2)))
            for c in clients:c.on_episode_end(execution=dict(decision_seq=0,executed_count=5,terminal=True))
    monkeypatch.setattr(ws,'WebsocketPolicyServer',Server)
    module=import_module('exp.r17_integration.models.'+model+'.formal_batch_server')
    if model=='dp':module.serve(campaign,23181,tmp_path/'server',cuda_graph=True)
    else:module.serve(campaign,23174,tmp_path/'server')
    assert all(r['actions'].shape==(5,4) and r['__hit_meta__']['capture_enabled'] is capture for r in results)
    assert all(r['__hit_meta__']['batching']['stages']['stage3']['size']==2 for r in results)
    assert (tmp_path/'server/CAPTURE_CLOSED.json').exists() is capture
    if capture:
        closed=json.loads((tmp_path/'server/CAPTURE_CLOSED.json').read_text())
        assert all(row['drained'] for row in closed)
        image_fields=[]
        for block in (tmp_path/'server/capture').rglob('*.npz'):
            with np.load(block,allow_pickle=False) as data:
                image_fields.extend(k for k in data.files if 'image' in k)
        assert bool(image_fields) is capture_images


@pytest.mark.parametrize('model',['dp','fastwam'])
def test_c10_keeps_five_action_transport_and_no_compute_tail(model,tmp_path):
    from importlib import import_module
    register_benchmark(FORMAL_SPEC)
    adapter=adapter_for(model)
    provider=import_module('exp.r17_integration.models.'+model+'.provider').register_provider(
        adapter,provider_id='mw_c10_'+uuid.uuid4().hex)
    cfg=formal_eval.full_config(provider,dict(model=model,domain='metaworld',commit_horizon=10))
    from exp.r17_integration.common.serving import make_connection_factory
    core=make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    core.on_episode_start(extra_metadata=episode())
    first=core.infer(wire())
    second=core.infer(dict(wire(),__execution__=dict(decision_seq=0,executed_count=5)))
    assert first['__hit_meta__']['actual_nfe']==provider.model_spec.default_sampler.num_steps
    assert second['__hit_meta__']['source']=='policy_tail'
    assert second['__hit_meta__']['stage_calls']==(0,0,0) and second['__hit_meta__']['actual_nfe']==0
    np.testing.assert_array_equal(second['actions'],first['actions'][5:])
    core.on_episode_end(execution=dict(decision_seq=1,executed_count=5,terminal=True))


@pytest.mark.parametrize('model',['dp_mw','fastwam_mw','pi05_mw','groot_mw'])
def test_two_mt50_yaml_bundles_publish_plan_and_switch(model,tmp_path,monkeypatch):
    from exp.r17_integration.common.fake_adapter import fake_spec,fake_provider
    from exp.r17_integration.common.recipe import bundle_identity,recipe_identity
    from exp.r17_integration.conductor.strategy import BenchmarkStrategy
    from exp.r17_integration.search.store import atomic_json
    from exp.r17_integration.search.store import write_store
    from exp.r17_integration.search.fit import fit_recipe,save_fitted_recipe
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.common.config import config_from_dict,load_cache_config
    register_benchmark(FORMAL_SPEC)
    spec=replace(fake_spec(horizon=5 if model=='pi05_mw' else 16),allowed_benchmark_protocols=(FORMAL_SPEC.identity,))
    provider=fake_provider('mw_cache_'+uuid.uuid4().hex,spec)
    rng=np.random.default_rng(8)
    rows=[dict(task_name=FORMAL_SPEC.tasks[0][0],task_uid='b_'+str(i//4),attempt=1,decision_seq=i%4,
        orig_init_idx=i//4,task_data_hash='cpu_'+str(i//4),campaign='mw_b',arm='teacher',
        bundle_generation='cpu',benchmark_protocol=FORMAL_SPEC.identity,split='library',init_pool='seed8',
        seed=8,success=True,library_episode_id=i//4) for i in range(24)]
    arrays=dict(state=rng.normal(size=(24,4)).astype(np.float32),
        normalized_actions=rng.normal(size=(24,spec.layout.horizon,4)).astype(np.float32))
    arrays.update({'key_'+name:rng.normal(size=(24,w)).astype(np.float32) for name,w in spec.feature_spec.fields.items()})
    store=write_store(tmp_path/'bpool',model_binding(spec,FORMAL_SPEC.identity),rows,arrays)
    groups=[]
    for group,c in [('c',5),('ch',5 if model=='pi05_mw' else 10)]:
        fitted,receipt=fit_recipe(store,task_names=FORMAL_SPEC.ordered_tasks,corrector=False,k=16,kref=8,
            early=True,guard=False,periodic_gap=None,commit_horizon=c,compact_loeo=True)
        ready=save_fitted_recipe(fitted,receipt,tmp_path/group/'fit')
        settings=cache_study.group_settings(model,group,c)
        cfg=config_from_dict(cache_study.config_document(provider,FORMAL_SPEC,FORMAL_SPEC,store.manifest,store.manifest,
            store_path=store.directory,fit_path=tmp_path/group/'fit',fit_digest=ready['digest'],
            stall_path=None,stall_digest=None,settings=settings))
        path=tmp_path/(group+'.yaml');path.write_text(yaml.safe_dump(asdict(cfg)))
        groups.append(dict(file=path.name,sha256=cache_study.file_sha(path),suite='mt50',group=group,
            bundle_id='mt50_'+group,bundle_identity=bundle_identity(cfg),recipe_identity=recipe_identity(cfg),
            settings=settings))
    sealed=dict(model=asdict(spec),benchmark_protocol=FORMAL_SPEC.identity,groups=groups)
    sealed['digest']=digest(sealed);atomic_json(tmp_path/'CACHE_STUDY.json',sealed)
    atomic_json(tmp_path/'model_spec.json',asdict(spec));atomic_json(tmp_path/'benchmark.json',FORMAL_SPEC.manifest())
    campaign=dict(model=model.removesuffix('_mw'),domain='metaworld',local_assets=str(tmp_path),
                  run_name='mw_cache_cpu',workers=2,endpoint_host='127.0.0.1',public_port=23180,smoke=False)
    _,_,tasks,endpoints=cache_study.plan(campaign)
    assert len(tasks)==5000 and {t.bundle_id for t in tasks}=={'mt50_c','mt50_ch'}
    assert {t.orig_init_state_idx for t in tasks}==set(range(50))
    assert all(t.extra['partition']=='test' and t.extra['seed']==7 and endpoints[0].accepts(t) for t in tasks)
    _,_,smoke,_=cache_study.plan(dict(campaign,smoke=True))
    assert len(smoke)==8 and {t.orig_init_state_idx for t in smoke}=={0,1}
    calls=[]
    class Control:
        def load_cache_config(self,path,*,yaml_id,bundle_id):
            calls.append(('load',bundle_id,path));return dict(bundle_identity=next(r['bundle_identity'] for r in groups if r['bundle_id']==bundle_id))
        def select_bundle(self,bundle):calls.append(('select',bundle))
    ctl=Control();cache_study.publish(ctl,campaign)
    strategy=BenchmarkStrategy(tasks,endpoints=endpoints,bundles={r['bundle_id']:tmp_path/r['file'] for r in groups})
    strategy.bind_driver(SimpleNamespace(_ctl=lambda endpoint:ctl))
    for task in (tasks[0],tasks[2500],tasks[1]):strategy.ensure_bundle(task.server,task.bundle_id)
    assert [c[1] for c in calls if c[0]=='load']==['mt50_c','mt50_ch']
    assert [c[1] for c in calls if c[0]=='select']==['mt50_c','mt50_ch','mt50_c']
    # Use both production bundle factories and actual fitted Store recipes.
    for group in groups:
        cfg=load_cache_config(tmp_path/group['file'])
        core=make_connection_factory(provider,cfg,build_shared_storage(cfg))()
        core.on_episode_start(extra_metadata=episode())
        obs={'state':arrays['state'][2]}
        first=core.infer(obs)
        assert first['__hit_meta__']['source']=='library' and first['__hit_meta__']['actual_nfe']==0
        c=cfg.adapter_binding['commit_horizon']
        if c==10:
            tail=core.infer(dict(obs,__execution__=dict(decision_seq=0,executed_count=5)))
            assert tail['__hit_meta__']['source']=='library_tail' and tail['__hit_meta__']['stage_calls']==(0,0,0)
            np.testing.assert_array_equal(tail['actions'],first['actions'][5:])
            core.on_episode_end(execution=dict(decision_seq=1,executed_count=5,terminal=True))
        else:core.on_episode_end(execution=dict(decision_seq=0,executed_count=5,terminal=True))
    import asyncio
    import socket
    import websockets.asyncio.client
    from openpi_client import msgpack_numpy
    from openpi.serving import websocket_policy_server as stock_ws
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    for key,value in (('_bundles',{}),('_current_bundle',None),('_bundle_version',0)):
        monkeypatch.setattr(stock_ws,key,value)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    plain=formal_eval.full_config(provider,dict(model=model.removesuffix('_mw'),domain='metaworld'))
    factory=make_connection_factory(provider,plain,build_shared_storage(plain))
    async def run():
        ready=asyncio.Event()
        server=WebsocketPolicyServer(object(),host='127.0.0.1',port=port,concurrent=True,
            connection_policy_factory=factory,bundle_validator=make_bundle_validator(provider),ready_callback=ready.set)
        serving=asyncio.create_task(server.run(stop_on_request=True))
        await asyncio.wait_for(ready.wait(),5)
        packer=msgpack_numpy.Packer();lifetimes=set()
        async def call(client,payload):
            await client.send(packer.pack(payload))
            result=await asyncio.wait_for(client.recv(),10)
            assert isinstance(result,bytes),result
            return msgpack_numpy.unpackb(result)
        try:
            async with websockets.asyncio.client.connect(f'ws://127.0.0.1:{port}',compression=None) as client:
                await client.recv()
                for group in groups:
                    ack=await call(client,dict(__ctrl__='load_cache_config',bundle_id=group['bundle_id'],
                        yaml_content=(tmp_path/group['file']).read_text()))
                    assert ack['bundle_identity']==group['bundle_identity']
                for group in (groups[0],groups[1],groups[0]):
                    await call(client,dict(__ctrl__='select_bundle',bundle_id=group['bundle_id']))
                    await call(client,dict(__ctrl__='episode_start',__task__=FORMAL_SPEC.tasks[0][0],__extra__=episode()))
                    out=await call(client,{'state':arrays['state'][2]})
                    hit=out['__hit_meta__']
                    assert hit['source']=='library' and hit['actual_nfe']==0
                    lifetimes.add((hit['resident_pid'],hit['resident_generation']))
                    await call(client,dict(__ctrl__='episode_end',__execution__=dict(decision_seq=0,executed_count=5,terminal=True)))
                assert len(lifetimes)==1
        finally:
            server.request_stop();await asyncio.wait_for(serving,5)
    asyncio.run(run())
