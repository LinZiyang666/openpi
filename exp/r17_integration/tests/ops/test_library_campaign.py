from dataclasses import asdict, replace
import json
from pathlib import Path
import uuid

import numpy as np
import pytest
import yaml

from exp.r17_integration.common.fake_adapter import fake_provider, fake_spec
from exp.r17_integration.benchmarks.spec import digest as task_data_digest
from exp.r17_integration.common.qualification import config_for_probe
from exp.r17_integration.debug.validate import digest, file_sha
from exp.r17_integration.ops.formal_driver import plan
from exp.r17_integration.ops.formal_eval import benchmark, aggregate
from exp.r17_integration.ops.formal_workers import worker_command
from exp.r17_integration.ops.library_alignment import load_alignment
from exp.r17_integration.ops.library_campaign import AUDIT_ROOT, disk_estimate, parse_args, require_fastwam_ab, validate_b_probe
from exp.r17_integration.ops.library_init import build_init_archive, install_init_archive


def assets(tmp_path, suite='libero_spatial', *, smoke=False, library=True):
    spec = benchmark()
    model = replace(fake_spec(action_dim=7,views=2),allowed_benchmark_protocols=(spec.identity,))
    provider = fake_provider('library-fixture-'+uuid.uuid4().hex,model)
    cfg = config_for_probe(provider,4,5)
    cfg.enabled=False; cfg.checkpoints={}; cfg.recipe.enabled=False
    cfg.recipe.ordered_tasks=list(spec.ordered_tasks); cfg.recipe.task_table_identity=spec.task_table_identity
    dest = tmp_path/'assets'; dest.mkdir()
    (dest/'benchmark.json').write_text(json.dumps(spec.manifest()))
    (dest/'model_spec.json').write_text(json.dumps(asdict(model)))
    (dest/'full.yaml').write_text(yaml.safe_dump(asdict(cfg)))
    campaign = dict(library=library,local_assets=str(dest),assets=str(dest),alignment=str(AUDIT_ROOT/(suite+'.json')),
        suite=suite,run_name='CPU',endpoint_host='unused',public_port=23320,workers=2,replicas=2,smoke=smoke,smoke_episodes=2,
        domain='libero',model='dp',worker_node='timan107',endpoint='unused:23320',driver_host='unused',driver_port=23339,
        worker_benchmark='benchmark.json',worker_egl_device=0,worker_init_states_dir='libero_spatial=/private/B/spatial,libero_10=/private/B/l10')
    return campaign,provider


@pytest.mark.parametrize('suite',('libero_spatial','libero_10'))
@pytest.mark.parametrize('smoke',(False,True))
def test_plan_exact_b_ids_full500_or_two_smoke_and_unchanged_protocol(tmp_path,suite,smoke):
    campaign,_ = assets(tmp_path,suite,smoke=smoke)
    bench,model,tasks,endpoints = plan(campaign)
    audit = load_alignment(campaign['alignment'])
    assert bench.identity==benchmark().identity and len(bench.ordered_tasks)==40
    assert len(tasks)==(2 if smoke else 500)
    rows = {(r['task_id'],r['orig_init_state_idx']):r for r in audit['rows']}
    for t in tasks:
        w = rows[t.extra['suite_task_id'],t.orig_init_state_idx]
        assert t.extra['library_episode_id']==w['pi05_episode_id']==w['groot_episode_id']
        assert t.extra['init_state_sha256']==w['init_array']['state_sha256']
        assert t.extra['task_data_hash']==task_data_digest(dict(w['init_array'],task_name=t.extra['task_name']))
        assert t.extra['init_pool']=='B' and t.extra['split']=='library' and t.extra['trial']==0
        assert t.task_id==t.extra['suite_task_id']+(30 if suite=='libero_10' else 0)
        assert t.extra['num_steps']==4 and t.extra['commit_horizon']==5
    if not smoke: assert {t.extra['library_episode_id'] for t in tasks}==set(range(500))


def test_library_absent_keeps_formal_a_plan_and_worker_command(tmp_path):
    campaign,_ = assets(tmp_path,library=False)
    _,_,tasks,_ = plan(campaign)
    assert len(tasks)==400 and all(t.orig_init_state_idx in range(10) for t in tasks)
    assert all('init_pool' not in t.extra and 'library_episode_id' not in t.extra for t in tasks)
    uid,argv = worker_command(campaign,0,python='python3')
    assert argv==['python3','-u','-m','exp.r17_integration.conductor.worker_entry','--worker-id',uid,
        '--server-key','unused:23320','--driver-host','unused','--driver-port','23339',
        '--benchmark','libero','--benchmark-spec','benchmark.json','--renderer','egl','--egl-device','0']
    campaign['library']=True
    assert worker_command(campaign,0,python='python3')[1]==argv+['--init-states-dir',campaign['worker_init_states_dir']]


def test_input_transport_byte_exact_and_no_a_fallback(tmp_path):
    audits = [load_alignment(AUDIT_ROOT/(s+'.json')) for s in ('libero_spatial','libero_10')]
    archive = tmp_path/'inputs.tar'
    receipt = build_init_archive(audits,archive)
    installed = install_init_archive(archive,tmp_path/'worker_inputs',receipt['archive_sha256'])
    assert installed['manifest']==receipt['manifest'] and len(receipt['manifest']['files'])==20
    assert install_init_archive(archive,tmp_path/'worker_inputs',receipt['archive_sha256'])==installed
    for a in audits:
        probe = {'per_task_digests':{a['suite']:{t['name']:file_sha(Path(installed['directories'][a['suite']])/Path(t['file']).name) for t in a['task_files'].values()}}}
        validate_b_probe(probe,dict(alignment=str(AUDIT_ROOT/(a['suite']+'.json')),suite=a['suite']))
        key = next(iter(probe['per_task_digests'][a['suite']]))
        probe['per_task_digests'][a['suite']][key]='A-state-files'
        with pytest.raises(ValueError,match='B pool'): validate_b_probe(probe,dict(alignment=str(AUDIT_ROOT/(a['suite']+'.json')),suite=a['suite']))
    target = next((tmp_path/'worker_inputs').rglob('*.init')); target.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='differs'): install_init_archive(archive,tmp_path/'worker_inputs',receipt['archive_sha256'])
    with pytest.raises(ValueError,match='SHA'): install_init_archive(archive,tmp_path/'bad','wrong')


@pytest.mark.parametrize('model',('dp_libero','fastwam_libero','cosmos_libero'))
def test_dedicated_ports_and_bounded_two_episode_smoke(model):
    a = parse_args(['--model',model,'--suite','libero_10','--smoke','--prepare-only'])
    assert a.seconds==900 and a.episode_seconds<=900
    assert not set([a.port,*range(a.port+1,a.port+1+a.replicas),a.driver_port])&set(range(23150,23180))
    assert a.server_node==('local' if model=='dp_libero' else 'h100')
    assert a.worker_node==('timan107' if model=='dp_libero' else 'timan108')
    with pytest.raises(SystemExit): parse_args(['--model',model,'--suite','libero_10','--prepare-only','--port','23160'])
    with pytest.raises(SystemExit): parse_args(['--model',model,'--suite','libero_10','--smoke','--run','--seconds','901'])


def test_fastwam_ab_gate_requires_gpu_result_for_identical_assets(tmp_path):
    descriptor = tmp_path/'descriptor.json'
    desc = dict(execution_protocol='released_libero',expected_digests={'weights':'one','code:loader':'two','norm':'three'})
    descriptor.write_text(json.dumps(desc))
    receipt = tmp_path/'ab.json'
    row = dict(schema='r17.fastwam.loader_parity.v1',status='passed',device='cuda:0',k=10,observation_count=2,
        execution_protocol='released_libero',source_digests={'descriptor':file_sha(descriptor)},rows=[{'field':'normalized','passed':True}])
    receipt.write_text(json.dumps(row))
    assert require_fastwam_ab(receipt,descriptor,desc)['receipt_sha256']==file_sha(receipt)
    with pytest.raises(ValueError,match='waits'): require_fastwam_ab(None,None,desc)
    with pytest.raises(ValueError,match='assets differ'): require_fastwam_ab(receipt,descriptor,dict(desc,expected_digests={'weights':'other'}))
    for mutation in ({'device':'cpu'},{'schema':'r17.fastwam.loader_header_equivalence.v1'},{'status':'failed'},
                     {'k':2},{'observation_count':1},{'source_digests':{}},{'rows':[]}):
        receipt.write_text(json.dumps(dict(row,**mutation)))
        with pytest.raises(ValueError,match='receipt'): require_fastwam_ab(receipt,descriptor,desc)


def test_disk_ceiling_includes_both_raw_cameras_and_native_replan(tmp_path):
    campaign,_ = assets(tmp_path)
    _,model,_,_ = plan(campaign)
    spatial = disk_estimate(model,'libero_spatial')
    l10 = disk_estimate(model,'libero_10')
    assert spatial['bytes_per_decision']['images']==393216
    assert spatial['max_decisions']==22000 and l10['max_decisions']==52000
    assert l10['collection_transfer_parent_ceiling_bytes']==3*l10['parent_ceiling_bytes']


def test_library_aggregate_requires_capture_and_full_native_and_keeps_failures(tmp_path):
    campaign,provider = assets(tmp_path,smoke=True)
    _,model,tasks,_ = plan(campaign)
    (tmp_path/'campaign.json').write_text(json.dumps(campaign))
    (tmp_path/'EPISODE_PLAN.json').write_text(json.dumps(dict(tasks=[asdict(t) for t in tasks],model=model,benchmark_protocol=benchmark().identity)))
    driver = tmp_path/'driver'; driver.mkdir()
    terminals = [dict(task_uid=t.task_uid,attempt=1,status='done',accepted=True,success=False,error=None) for t in tasks]
    steps = [dict(task_uid=t.task_uid,attempt=1,accepted=True,capture_enabled=True,source='policy',actual_nfe=4,stage_calls=[1,1,4]) for t in tasks]
    for name,rows in [('journal',terminals),('attempts',terminals),('per_step',steps)]:
        (driver/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (driver/'DRIVER_RECEIPT.json').write_text(json.dumps(dict(status='PASS',elapsed_seconds=1)))
    output = aggregate(tmp_path)
    assert output['status']=='PASS' and output['debug_capture'] is True and output['suites']['libero_spatial']['sr']==0
    steps[0]['capture_enabled']=False
    (driver/'per_step.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in steps))
    assert aggregate(tmp_path)['status']=='PARTIAL'


def test_cosmos_native_profile_b_plan_and_debug_capability_fails_closed(tmp_path,monkeypatch):
    """Published Cosmos bundle geometry; no resident/weights or GPU is loaded."""
    from exp.r17_integration.models.cosmos.manifest import make_manifest,make_sampler
    from exp.r17_integration.models.cosmos.protocols import benchmark_profile
    from exp.r17_integration.common.provider import AdapterProvider,register_provider
    from exp.r17_integration.qualification import assets as sealing
    from exp.r17_integration.ops.formal_eval import seal
    bench = benchmark_profile(benchmark(),False)
    model = make_manifest({'synthetic':True},{'synthetic':'CPU-fixture'},(bench.identity,)).spec
    provider = AdapterProvider('cosmos-plan-'+uuid.uuid4().hex,model,lambda:None,make_sampler)
    register_provider(provider)
    cfg = config_for_probe(provider,5,16)
    cfg.enabled=False; cfg.checkpoints={}; cfg.recipe.enabled=False
    cfg.recipe.ordered_tasks=list(bench.ordered_tasks); cfg.recipe.task_table_identity=bench.task_table_identity
    dest = tmp_path/'assets'; dest.mkdir()
    (dest/'benchmark.json').write_text(json.dumps(bench.manifest()))
    (dest/'model_spec.json').write_text(json.dumps(asdict(model)))
    (dest/'full.yaml').write_text(yaml.safe_dump(asdict(cfg)))
    campaign = dict(library=True,model='cosmos',domain='libero',suite='libero_10',official_50=False,
        alignment=str(AUDIT_ROOT/'libero_10.json'),local_assets=str(dest),assets=str(dest),run_name='CPU',
        endpoint_host='unused',public_port=23360,workers=2,smoke=False,checkpoint='synthetic')
    actual,_,tasks,_ = plan(campaign)
    assert actual.identity==bench.identity and actual.seed==195 and actual.replan_steps==16
    assert len(tasks)==500 and {t.extra['library_episode_id'] for t in tasks}==set(range(500))
    assert {t.task_id for t in tasks}==set(range(30,40))
    assert all(t.extra['num_steps']==5 and t.extra['commit_horizon']==16 for t in tasks)
    # The released bundle supports the minimal teacher observer. Retain the
    # negative seal gate with an explicitly unsupported capability fixture.
    from exp.r17_integration.common.contracts import Capabilities, Capability
    model.capabilities.require('debug_capture')
    blocked = replace(model,capabilities=Capabilities(dict(model.capabilities.features,
        debug_capture=Capability('unsupported','CPU negative admission fixture'))))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(sealing,'model_assets',lambda *a:(blocked,provider))
    with pytest.raises(ValueError,match='debug_capture'):
        seal(campaign)
