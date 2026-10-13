from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.common.serving import full_call
from exp.r17_integration.models.fastwam.batch_runtime import BatchConfig,create_runtime
from exp.r17_integration.models.fastwam.provider import register_provider,make_connection_factory
from exp.r17_integration.models.fastwam.serve import serve_metadata
from exp.r17_integration.tests.models.fastwam.conftest import make_adapter,observation,request
from exp.r17_integration.tests.models.fastwam.test_fastwam_serving import config_for


def test_default_off_has_identical_actions_manifest_metadata_and_no_files(tmp_path):
    adapter=make_adapter('libero',allow_unqualified_batch=False)
    before=asdict(adapter.manifest),serve_metadata(adapter)
    old,_=full_call(adapter,observation('libero'),request(0))
    assert create_runtime(adapter,ledger_path=tmp_path/'must_not_exist'/'batches.jsonl') is None
    new,_=full_call(adapter,observation('libero'),request(0))
    assert old.normalized_chunk.numpy().tobytes()==new.normalized_chunk.numpy().tobytes()
    assert before==(asdict(adapter.manifest),serve_metadata(adapter))
    assert adapter.batch_config is None and not list(tmp_path.iterdir())
    with pytest.raises(ValueError,match='B>1 disabled'):
        adapter.run_stage1_batch([adapter.prepare(observation('libero'),request())]*2)


@pytest.mark.parametrize('value',[0,33,True,2.5])
def test_batch_limit_rejects_unsafe_or_noninteger_values(value):
    with pytest.raises(ValueError):
        BatchConfig(value)


@pytest.mark.parametrize('value',[-1,51,float('nan'),float('inf'),True])
def test_batch_window_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        BatchConfig(2,value)


def test_batch_scope_and_worker_count_fail_closed(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='ledger'):
        create_runtime(make_adapter('libero'),2)
    runtime=create_runtime(make_adapter('metaworld'),2,ledger_path=tmp_path/'ledger')
    runtime.close()
    monkeypatch.setenv('BATCHING_STAGE2_WORKERS','2')
    with pytest.raises(ValueError,match='one worker'):
        create_runtime(make_adapter('libero'),2,ledger_path=tmp_path/'ledger')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('k',[1,2,10])
def test_production_opt_in_real_s07_connections_and_composition(k,tmp_path):
    a=make_adapter('libero',allow_unqualified_batch=False)
    reference=[full_call(a,observation('libero',i),request(i),sampler=a.sampler_for_steps(k))[0] for i in range(2)]
    ledger=tmp_path/'batches.jsonl'
    runtime=create_runtime(a,2,50,ledger)
    try:
        p=register_provider(a,provider_id='p3_batched_'+tmp_path.name,coordinator=runtime.core)
        assert not a.allow_unqualified_batch
        clients=[p.load_adapter() for _ in range(2)]
        barrier=threading.Barrier(2)
        def one(i):
            barrier.wait(timeout=5)
            return full_call(clients[i],observation('libero',i),request(i),sampler=a.sampler_for_steps(k))[0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs=list(pool.map(one,range(2)))
        for i,result in enumerate(outputs):
            torch.testing.assert_close(result.normalized_chunk,reference[i].normalized_chunk,rtol=0,atol=0)
            receipts=result.capture['batching']
            assert set(receipts)=={'stage1','stage2','stage3'}
            for stage,receipt in receipts.items():
                assert receipt['size']==2 and receipt['stage']==stage
                assert receipt['members'][receipt['row']]['request']==asdict(request(i))
                assert len({m['request_id'] for m in receipt['members']})==2
        # Mutating one response cannot alter another response's receipt.
        other=json.dumps(outputs[1].capture['batching'],sort_keys=True)
        outputs[0].capture['batching']['stage1']['members'][0]['request']['campaign']='mutated'
        assert json.dumps(outputs[1].capture['batching'],sort_keys=True)==other
        rows=[json.loads(line) for line in ledger.read_text().splitlines()]
        assert len(rows)==3 and all(r['size']==2 for r in rows)
        assert {r['batch_id'] for r in rows}=={v['batch_id'] for v in outputs[1].capture['batching'].values()}
        metadata=serve_metadata(a)
        assert metadata['r17_batching']['max_batch']==2
        assert metadata['r17_batching']['byte_identical_to_b1'] is False
        assert metadata['batch_runtime_capabilities']['batch_stage3']=='enabled_study'
    finally:
        runtime.close()


def test_connection_receipts_preserve_tail_and_no_extra_model_calls(tmp_path):
    a=make_adapter('libero',allow_unqualified_batch=False)
    runtime=create_runtime(a,2,0,tmp_path/'ledger.jsonl')
    try:
        p=register_provider(a,provider_id='p3_tail_batch_'+tmp_path.name,coordinator=runtime.core)
        cfg=config_for(p,k=2,commit=10)
        cfg.enabled=False;cfg.checkpoints={}
        factory=runtime.wrap_connection_factory(make_connection_factory(p,cfg,build_shared_storage(cfg)))
        core=factory()
        initial=core.infer(observation('libero'))
        assert set(initial['__hit_meta__']['batching']['stages'])=={'stage1','stage2','stage3'}
        calls={k:list(v) for k,v in a.model.calls.items()}
        tail=core.infer({**observation('libero'),'__execution__':{'decision_seq':0,'executed_count':5}})
        assert tail['__hit_meta__']['source']=='policy_tail'
        assert tail['__hit_meta__']['batching']['stages']=={}
        assert tail['actions'].tobytes()==initial['actions'][5:].tobytes()
        assert a.model.calls==calls
        rows=[json.loads(line) for line in (tmp_path/'ledger.jsonl').read_text().splitlines()]
        decisions=[row for row in rows if row['schema']=='r17.fastwam.decision_batch.v1']
        assert len(decisions)==2 and decisions[-1]['stage_calls']==[0,0,0]
    finally:
        runtime.close()


def test_formal_shim_is_scoped_and_restores_launcher_on_failure(tmp_path):
    from exp.r17_integration.models.fastwam.formal_launch import launcher_hooks
    seen=[]
    def prepare(campaign,directory):
        seen.append(dict(campaign))
        raise RuntimeError('sentinel')
    host=lambda campaign:'original_python'
    launcher=SimpleNamespace(prepare=prepare,host_python=host)
    cfg=BatchConfig(4,5)
    campaign=dict(model='fastwam',domain='libero',server_node='local')
    with pytest.raises(RuntimeError,match='sentinel'):
        with launcher_hooks(launcher,cfg):
            assert launcher.host_python(campaign).endswith('formal_python.sh')
            assert launcher.host_python(dict(model='dp',domain='libero'))=='original_python'
            with pytest.raises(ValueError,match='local-only'):
                launcher.host_python({**campaign,'server_node':'h100'})
            launcher.prepare(campaign,tmp_path)
    assert launcher.prepare is prepare and launcher.host_python is host
    assert seen[0]['fastwam_batching']==cfg.metadata()


def test_numerical_summary_compares_corresponding_rows():
    from exp.r17_integration.models.fastwam.batch_profile import numerical_deviation
    refs=[SimpleNamespace(normalized_chunk=torch.zeros(1,2)),SimpleNamespace(normalized_chunk=torch.ones(1,2))]
    outs=[SimpleNamespace(normalized_chunk=torch.tensor([[.25,0.]])),refs[1]]
    row=numerical_deviation(outs,refs)
    assert row['max_abs']==.25 and row['median_abs']==0 and row['mean_abs']==.0625


def test_formal_resident_uses_sealed_provider_and_batches_real_connections(tmp_path,monkeypatch):
    from exp.r17_integration.models.fastwam import formal_batch_server
    from exp.r17_integration.qualification import model_runtime
    from exp.r17_integration.common import config as config_module
    from exp.r17_integration.common import websocket_policy_server as server_module
    from openpi.serving.batching_core import get_active_coordinator
    a=make_adapter('libero',allow_unqualified_batch=False)
    p=register_provider(a,provider_id='p3_formal_batch_'+tmp_path.name)
    cfg=config_for(p,k=10,commit=5)
    cfg.enabled=False;cfg.checkpoints={}
    monkeypatch.setattr(model_runtime,'load',lambda *args:(a,p))
    monkeypatch.setattr(config_module,'load_cache_config',lambda *args:cfg)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *args:SimpleNamespace(total_memory=48*2**30))
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    campaign=dict(model='fastwam',domain='libero',assets=str(tmp_path),device='cpu',replicas=1,memory_budget_gib=40,
                  fastwam_batching=BatchConfig(2,50).metadata())
    before=json.dumps(asdict(a.manifest),sort_keys=True)
    responses=[]
    class Server:
        def __init__(self,policy,**kwargs):
            self.factory=kwargs['connection_policy_factory']
            assert kwargs['metadata']['r17_model_spec']==asdict(p.model_spec)
            assert kwargs['metadata']['r17_batching']==campaign['fastwam_batching']
            kwargs['bundle_validator'](cfg)
            kwargs['ready_callback']()
        def request_stop(self):
            pass
        def serve_forever(self,**kwargs):
            clients=[self.factory() for _ in range(2)]
            barrier=threading.Barrier(2)
            def one(i):
                barrier.wait(timeout=5)
                return clients[i].infer(observation('libero',i))
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses.extend(pool.map(one,range(2)))
    monkeypatch.setattr(server_module,'WebsocketPolicyServer',Server)
    monkeypatch.setattr(formal_batch_server.signal,'signal',lambda *args:None)
    formal_batch_server.serve(campaign,9009,tmp_path/'replica')
    assert json.loads((tmp_path/'manifest.json').read_text())==campaign
    from exp.r17_integration.ops.sync_back import snapshot_run,install_snapshot
    archive=tmp_path.parent/(tmp_path.name+'_closed.tar')
    transfer=snapshot_run(tmp_path,archive)
    restored=tmp_path.parent/(tmp_path.name+'_restored')
    install_snapshot(archive,restored,expected_sha256=transfer['archive_sha256'])
    assert json.loads((restored/'manifest.json').read_text())==campaign
    assert cfg.adapter_binding['provider']==p.provider_id
    assert json.dumps(asdict(a.manifest),sort_keys=True)==before and a.batch_config is None
    assert get_active_coordinator() is None
    assert all(r['__hit_meta__']['batching']['stages']['stage3']['size']==2 for r in responses)
    assert len((tmp_path/'replica'/'serving.jsonl').read_text().splitlines())==2
    assert json.loads((tmp_path/'replica'/'closed.json').read_text())['status']=='CLOSED'


def test_live_websocket_connections_form_native_batches_and_tail_receipts(tmp_path,monkeypatch):
    import asyncio
    import websockets.asyncio.server as ws_server
    import websockets.asyncio.client as ws_client
    from openpi_client import msgpack_numpy
    from openpi.serving import websocket_policy_server as stock
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from exp.r17_integration.common.serving import make_bundle_validator
    a=make_adapter('libero',allow_unqualified_batch=False)
    runtime=create_runtime(a,2,50,tmp_path/'live.jsonl')
    try:
        p=register_provider(a,provider_id='p3_live_batch_'+tmp_path.name,coordinator=runtime.core)
        cfg=config_for(p,k=2,commit=10);cfg.enabled=False;cfg.checkpoints={}
        factory=runtime.wrap_connection_factory(make_connection_factory(p,cfg,build_shared_storage(cfg)))
        server=WebsocketPolicyServer(factory(),concurrent=True,connection_policy_factory=factory,
            bundle_validator=make_bundle_validator(p),metadata=serve_metadata(a))
        monkeypatch.setattr(stock,'_bundles',{})
        monkeypatch.setattr(stock,'_current_bundle',None)
        packer=msgpack_numpy.Packer()
        async def rpc(ws,msg):
            await ws.send(packer.pack(msg))
            return msgpack_numpy.unpackb(await asyncio.wait_for(ws.recv(),5))
        async def exercise():
            async with ws_server.serve(server._handler,'127.0.0.1',0) as listener:
                port=listener.sockets[0].getsockname()[1]
                async with ws_client.connect(f'ws://127.0.0.1:{port}') as c1,ws_client.connect(f'ws://127.0.0.1:{port}') as c2:
                    for c in (c1,c2):
                        assert msgpack_numpy.unpackb(await c.recv())['r17_batching']['max_batch']==2
                        await rpc(c,{'__ctrl__':'episode_start','task_id':0,'init_state_id':0,'episode_id':0})
                    results=await asyncio.gather(rpc(c1,observation('libero',0)),rpc(c2,observation('libero',1)))
                    for result in results:
                        stages=result['__hit_meta__']['batching']['stages']
                        assert set(stages)=={'stage1','stage2','stage3'}
                        assert all(r['size']==2 for r in stages.values())
                        assert len({m['connection_id'] for m in stages['stage3']['members']})==2
                    assert results[0]['actions'].tobytes()!=results[1]['actions'].tobytes()
                    calls={k:list(v) for k,v in a.model.calls.items()}
                    tails=await asyncio.gather(*[rpc(c,{**observation('libero',i),
                        '__execution__':{'decision_seq':0,'executed_count':5}}) for i,c in enumerate((c1,c2))])
                    assert a.model.calls==calls
                    for before,tail in zip(results,tails):
                        assert tail['actions'].tobytes()==before['actions'][5:].tobytes()
                        assert tail['__hit_meta__']['batching']['stages']=={}
                    for c in (c1,c2):
                        await rpc(c,{'__ctrl__':'episode_end','success':False})
        asyncio.run(exercise())
        decisions=[json.loads(line) for line in (tmp_path/'live.jsonl').read_text().splitlines()
                   if json.loads(line)['schema']=='r17.fastwam.decision_batch.v1']
        assert len(decisions)==4 and sum(bool(r['batching']['stages']) for r in decisions)==2
    finally:
        runtime.close()
