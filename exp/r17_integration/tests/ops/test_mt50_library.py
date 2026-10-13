from dataclasses import asdict, replace
import json
import uuid

import numpy as np
import pytest
import yaml

from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, decision_seed, R17MetaWorldEpisodeRunner
from exp.r17_integration.common.contracts import DecisionCapture, RequestContext
from exp.r17_integration.common.fake_adapter import fake_provider, fake_spec
from exp.r17_integration.common.qualification import config_for_probe
from exp.r17_integration.conductor.manifest import episode_tasks
from exp.r17_integration.debug.capture_bridge import CaptureBridge, DecisionSink
from exp.r17_integration.debug.library_capture import LibraryCaptureBridge, LibraryDecisionSink
from exp.r17_integration.debug.reader_profiles import CaptureReader
from exp.r17_integration.debug.validate import digest
from exp.r17_integration.ops.formal_driver import plan
from exp.r17_integration.ops.library_campaign import parse_args
from exp.r17_integration.search.teacher_library import build_teacher_parent, nested_views, fit_nested_libraries
from exp.r17_integration.tests.debug.fixtures import arrays, binding, record
from openpi.conductor.task import ServerEndpoint


def mw_assets(tmp_path, model='dp', smoke=False):
    spec = replace(fake_spec(action_dim=4,views=1),model_id=model+'_mw',allowed_benchmark_protocols=(FORMAL_SPEC.identity,))
    provider = fake_provider('mt50-'+uuid.uuid4().hex,spec)
    config = config_for_probe(provider,4,5)
    config.enabled=False; config.checkpoints={}; config.recipe.enabled=False
    config.recipe.ordered_tasks=list(FORMAL_SPEC.ordered_tasks)
    config.recipe.task_table_identity=FORMAL_SPEC.task_table_identity
    dest=tmp_path/'assets'; dest.mkdir()
    (dest/'benchmark.json').write_text(json.dumps(FORMAL_SPEC.manifest()))
    (dest/'model_spec.json').write_text(json.dumps(asdict(spec)))
    (dest/'full.yaml').write_text(yaml.safe_dump(asdict(config)))
    return dict(library=True,domain='metaworld',model=model,suite='mt50',smoke=smoke,
        local_assets=str(dest),run_name='cpu',endpoint_host='unused',public_port=23180,workers=4), provider


@pytest.mark.parametrize('model',('dp','fastwam','pi05','groot','cosmos'))
@pytest.mark.parametrize('smoke',(False,True))
def test_mt50_formal_teacher_plan(tmp_path,model,smoke):
    campaign,_=mw_assets(tmp_path,model,smoke)
    _,_,tasks,_=plan(campaign)
    coordinates={(t.extra['task_name'],t.extra['seed'],t.orig_init_state_idx) for t in tasks}
    assert len(tasks)==len(coordinates)==(4 if smoke else 2500)
    assert {t.extra['seed'] for t in tasks}=={8}
    assert {t.extra['trial'] for t in tasks}=={0}
    assert {t.orig_init_state_idx for t in tasks}==set(range(2 if smoke else 50))
    assert len({t.task_id for t in tasks})==(2 if smoke else 50)
    assert len({t.task_uid for t in tasks})==len(tasks)


def test_episode_seed_reaches_env_request_and_recorded_identity():
    from exp.r17_integration.tests.benchmarks.test_metaworld import Client, Env
    make=lambda partition: episode_tasks(campaign='cpu',arm='full',bundle_id='full',
        endpoint=ServerEndpoint('local',1),model_identity='m',norm_identity='n',layout_identity='l',
        partition=partition,names=['reach-v3'],indices=[0])[0]
    client=Client(); constructions=[]
    def environment(name,idx,seed):
        constructions.append((name,idx,seed)); return Env(success=1)
    runner=R17MetaWorldEpisodeRunner(env_factory=environment,client_factory=lambda _:client)
    for partition,seed in [('test',7),('library',8)]:
        result=runner.run(make(partition),lambda *args:None)
        request=client.requests[-1]['__request__']
        assert request['seed']==seed
        assert request['explicit_seed']==decision_seed('reach-v3',0,0,seed)
        assert result.per_step_rows[-2]['seed']==seed  # episode summary precedes ack
        assert runner._identity()['seed']==seed
        assert next(v for k,v in reversed(client.calls) if k=='start')['extra_metadata']['seed']==seed
    assert constructions==[('reach-v3',0,7),('reach-v3',0,8)]
    assert decision_seed('reach-v3',0,0,7)!=decision_seed('reach-v3',0,0,8)
    with pytest.raises(ValueError): make('dev')


def capture_cohort(path, model='dp', tasks=50, count=10, generic=False):
    b=binding(model,'mw',key_width=3)
    bridge=(CaptureBridge if generic else LibraryCaptureBridge)(path,enabled=True,binding=b,capture_images=False)
    cohort=[]
    for tid,name in enumerate(FORMAL_SPEC.ordered_tasks[:tasks]):
        for idx in range(count):
            ep=dict(record(idx,task=name),benchmark_protocol=b['benchmark_protocol'],seed=8,
                init_pool='seed8',suite_task_id=tid,task_id=tid,library_episode_id=tid*50+idx,
                error=None,closed_ok=True,success=bool(idx%2))
            cohort.append(ep)
            sink=(DecisionSink if generic else LibraryDecisionSink)(bridge,ep)
            for seq in range(2):
                if not generic: sink.bind_observation({})  # no image reads or copies
                payload,keys=arrays(b,tid*50+idx)
                request=RequestContext(ep['campaign'],b['benchmark_protocol'],name,idx,seq,
                    arm=ep['arm'],bundle_generation=ep['bundle_generation'],attempt=1,task_id=tid,seed=8)
                capture=DecisionCapture(request,'policy',0,0,(1,1,b['sampler']['num_steps']),b['sampler']['num_steps'],
                    payload['normalized_actions'],payload['wire_actions'],keys,
                    model_capture={'model_input_base_rgb':np.ones((8,8,3),np.uint8),'scalar':np.array([1.],np.float32)})
                sink(capture)
                sink.on_execution(dict(request=asdict(request),decision_seq=seq,executed_count=5,cursor=5,terminal=seq==1))
    bridge.close()
    return b,cohort


@pytest.mark.parametrize('model',('dp','fastwam','pi05','groot'))
@pytest.mark.parametrize('generic',(False,True))
def test_images_off_preserves_keys_state_actions_and_acks(tmp_path,model,generic):
    b,_=capture_cohort(tmp_path/'capture',model,tasks=1,count=1,generic=generic)
    rows=list(CaptureReader(tmp_path/'capture').decisions())
    assert len(rows)==2 and rows[-1][0]['terminal']
    for row,values,keys in rows:
        assert row['seed']==8 and row['executed_count']==5
        assert set(b['model']['feature_spec']['fields'])<=set(keys)
        assert {'state','normalized_actions'}<=set(values)
        assert not any(v.dtype==np.uint8 and v.ndim==3 for v in [*values.values(),*keys.values()])


def test_no_image_parent_nested_fits_all50_tasks(tmp_path):
    b,cohort=capture_cohort(tmp_path/'capture')
    parent=build_teacher_parent([tmp_path/'capture'],tmp_path/'parent',b,cohort,
        expected_episodes=500,capture_images=False)
    assert len(parent.rows)==1000 and not any(k.startswith('image') for k in parent.fields)
    views=nested_views(parent,sizes=(3,5,10))
    previous=set()
    for count,view in views.items():
        ids=set(view.manifest['episode_ids'])
        assert len(ids)==count*50 and previous<=ids
        assert {p.name for p in view.directory.iterdir()}=={'VIEW.json','row_indices.npy'}
        assert np.array_equal(np.asarray(view.array('state')),parent.array('state')[view.indices])
        previous=ids
    products=fit_nested_libraries(parent,None,task_names=FORMAL_SPEC.ordered_tasks,sizes=(10,))
    assert products['10']['episodes']==500
    from exp.r17_integration.search.fit import load_fitted_recipe
    fitted=load_fitted_recipe(products['10']['directory'],expected_binding=b)
    assert set(fitted.task)==set(range(50))


@pytest.mark.parametrize('sharded',(False,True))
def test_single_camera_formal_core_capture_without_images(tmp_path,monkeypatch,sharded):
    from exp.r17_integration.common.config import load_cache_config,build_shared_storage
    from exp.r17_integration.common.serving import make_connection_factory
    from exp.r17_integration.common.contracts import CaptureSpec
    from exp.r17_integration.debug.library_capture import campaign_capture_factory
    from exp.r17_integration.ops import library_staging
    campaign,provider=mw_assets(tmp_path,smoke=True)
    class Budget:
        campaign={'run_name':'cpu'}
        def check(self): pass
        def reserve(self,*args): pass
        def charge(self,*args): pass
        def closed(self,*args): pass
    monkeypatch.setattr(library_staging,'admit',lambda c:None)
    monkeypatch.setattr(library_staging,'ShardBudget',lambda c:Budget())
    campaign.update(server_node='h100' if sharded else 'local',capture_images=False)
    factory=campaign_capture_factory(campaign,tmp_path/'capture')
    config=load_cache_config(tmp_path/'assets/full.yaml')
    core=make_connection_factory(provider,config,build_shared_storage(config),
        capture_sink_factory=factory,capture_spec=CaptureSpec(enabled=True,snapshots=False))()
    tid=FORMAL_SPEC.ordered_tasks.index('reach-v3')
    ep=dict(record(0),seed=8,init_pool='seed8',split='library',library_episode_id=tid*50,suite_task_id=tid,
        benchmark_protocol=FORMAL_SPEC.identity,bundle_generation='generation1')
    core.on_episode_start(task_id=tid,episode_id=0,extra_metadata=ep)
    req=RequestContext(ep['campaign'],FORMAL_SPEC.identity,'reach-v3',0,0,seed=8,
        explicit_seed=decision_seed('reach-v3',0,0,8),arm='full',bundle_generation='generation1',task_id=tid,attempt=1)
    obs={'state':np.arange(4,dtype=np.float32),'__request__':asdict(req)}
    core.capture_sink.bind_observation(obs)
    core.infer(obs)
    core.on_episode_end(__execution__=dict(decision_seq=0,executed_count=5,cursor=5,terminal=True),__success__=False)
    assert all(row['drained'] for row in factory.close())
    captures=list((tmp_path/'capture').rglob('r17_capture.json'))
    assert len(captures)==1
    row,values,_=next(CaptureReader(captures[0].parent).decisions())
    assert row['terminal'] and row['seed']==8 and row['image_sources']=={}
    assert set(values)=={'state','normalized_actions'}


@pytest.mark.parametrize('model,port', [('dp_mw',23180),('fastwam_mw',23170),('pi05_mw',23160),('groot_mw',23160),('cosmos_mw',23240)])
def test_mt50_library_cli_direct_ports_and_worker_lease(model,port):
    a=parse_args(['--model',model,'--suite','mt50','--server-node','h100',
        '--worker-counts','timan107=12,timan1=40','--smoke','--prepare-only'])
    assert a.port==port and not a.capture_images and a.workers==52
    assert a.driver_port in range(23100,23200)
    assert a.subset_counts==[10,20,30,40,50]
    from exp.r17_integration.ops.timan1_workers import authorize,route,MW_WRAPPER
    from exp.r17_integration.ops.formal_workers import worker_command
    c=dict(domain='metaworld',worker_nodes=a.worker_counts)
    assert authorize(c)['device']==3
    worker=route(dict(c,worker_node='timan1',node_workers=40),c)
    worker.update(run_name='cpu',endpoint='unused:23180',driver_host='unused',driver_port=23189,worker_benchmark='benchmark.json',library=True)
    argv=worker_command(worker,0)[1]
    assert argv[0]==MW_WRAPPER and argv[-1]=='3'
    assert '--init-states-dir' not in argv and '--max-episodes' not in argv
    with pytest.raises(ValueError): authorize(dict(c,worker_nodes={'timan1':41}))


@pytest.mark.parametrize('model',('dp_mw','fastwam_mw'))
def test_library_cli_binds_batching_and_mps_before_prepare(tmp_path,monkeypatch,model):
    from exp.r17_integration.ops import library_campaign,formal_eval
    monkeypatch.setattr(library_campaign,'LIBRARY_ROOT',tmp_path)
    prepared=[]
    def prepare(campaign,directory):
        prepared.append(campaign)
        assert campaign['workers']==80 and campaign['replicas']==2
        assert not campaign['capture_images'] and 'alignment' not in campaign
        assert 'bpool_transfer' not in campaign and campaign['mps']
        from exp.r17_integration.models.fastwam.ops.mps import paths
        paths('/data/r17_library/'+model+'_mt50/campaigns/'+campaign['run_name'],'GPU-placeholder')
        batch=campaign['dp_batching' if model=='dp_mw' else 'fastwam_batching']
        assert batch['max_batch']==8 and batch['batch_wait_ms']==5
        return {}
    monkeypatch.setattr(formal_eval,'prepare',prepare)
    assert library_campaign.main(['--model',model,'--suite','mt50','--server-node','local',
        '--worker-counts','timan107=40,timan1=40','--replicas','2','--max-batch','8',
        '--batch-wait-ms','5','--mps','--prepare-only'])==0
    assert len(prepared)==1
