"""CPU replay of the 391/2000 interrupted baseline, with sealed mismatch fences."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import pytest
import yaml

from exp.r17_integration.ops import formal_driver, formal_eval, formal_resume as resume
from exp.r17_integration.ops.formal_process import write_json


def write_rows(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in content))


@pytest.fixture(scope='module')
def source_master(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('sealed_baseline')
    from exp.r17_integration.common.fake_adapter import fake_provider, fake_spec
    from exp.r17_integration.common.benchmarks import register_benchmark
    from exp.r17_integration.common.qualification import config_for_probe
    root = tmp_path/'baseline'
    assets = root/'assets'
    assets.mkdir(parents=True)
    benchmark = formal_eval.benchmark(True)
    register_benchmark(benchmark)
    spec = replace(fake_spec(), allowed_benchmark_protocols=(benchmark.identity,))
    provider = fake_provider('resume.fixture.'+tmp_path.name, spec)
    config = config_for_probe(provider, spec.default_sampler.num_steps, spec.layout.replan_steps)
    config.enabled = False
    config.checkpoints = {}
    config.recipe.ordered_tasks = list(benchmark.ordered_tasks)
    config.recipe.task_table_identity = benchmark.task_table_identity
    (assets/'full.yaml').write_text(yaml.safe_dump(asdict(config)))
    write_json(assets/'benchmark.json', benchmark.manifest())
    write_json(assets/'model_spec.json', asdict(spec))
    write_json(assets/'model_manifest.json', dict(schema='r17.model_manifest.v1', spec=asdict(spec),
                                                content_digests={'weights':'immutable', 'code':'pinned'}))
    campaign = dict(run_name='baseline', model='fastwam', domain='libero', checkpoint='/immutable/checkpoint',
                    server_node='h100', device='cuda:0',
                    official_50=True, smoke=False, native_parity_only=False, replicas=2, workers=24,
                    endpoint_host='old.invalid', endpoint='old.invalid:23173', public_port=23173,
                    local_assets=str(assets), episode_seconds=3600, retries=2, worker_egl_device=0)
    _, model, tasks, _ = formal_driver.plan(campaign)
    plan = dict(expected_episodes=len(tasks), benchmark_protocol=benchmark.identity, model=model,
                tasks=[asdict(t) for t in tasks], endpoint=campaign['endpoint'], cache_enabled=False, debug_capture=False)
    write_json(root/'campaign.json', campaign)
    write_json(root/'EPISODE_PLAN.json', plan)
    write_json(root/'RECEIPT.json', dict(status='FAIL', all_owned_closed=True, error='fleet OOM'))
    write_json(root/'SERVER_ATTESTATIONS.json', [dict(metadata=dict(r17_model_spec=model,
        cache_enabled=False, debug_capture=False, formal_pure_inference=True)) for _ in range(2)])
    write_json(root/'BUNDLE_ACK.json', dict(bundle_id='formal_full_v1', bundle_identity='sealed-bundle'))
    write_json(root/'WORKER_DEPLOYMENT.json', dict(timan107=dict(installed=dict(files={}))))
    (root/'runtime.tar').write_bytes(b'preserved immutable model runtime')
    probe = dict(benchmark_protocol=benchmark.identity, task_table_identity=benchmark.task_table_identity,
                 libero_code_digest='original-libero', task_data={'state_sha256':'original-init'},
                 robosuite_version='1.4.0', numpy='1.26.4', torch='2.7.1+cu126')
    write_json(root/'driver/census.json', dict(first=dict(probe=probe)))
    fill_driver(root, plan['tasks'][:391], probe=probe, status='PARTIAL')
    return root


@pytest.fixture
def source(tmp_path, source_master):
    root = tmp_path/'baseline'
    shutil.copytree(source_master, root)
    campaign = resume.read(root/'campaign.json')
    campaign['local_assets'] = str(root/'assets')
    write_json(root/'campaign.json', campaign)
    return root


def fill_driver(root, tasks, *, probe, status='PASS'):
    journal, attempts, steps = [], [], []
    for i, task in enumerate(tasks):
        uid, extra = task['task_uid'], task['extra']
        success = bool(i % 2)  # A failed episode still counts; no outcome cherry-picking.
        journal.append(dict(task_uid=uid, accepted=True, status='done' if success else 'failed',
                            success=success, error=None, attempt=1))
        attempts.append(dict(task_uid=uid, accepted=True, attempt=1, success=success,
                             n_steps=5, error=None, worker_id='worker'))
        steps.append(dict(task_uid=uid, accepted=True, attempt=1, decision_seq=0,
                          **{k:extra[k] for k in ('model_identity', 'norm_identity', 'layout_identity',
                             'benchmark_protocol', 'yaml_identity', 'num_steps', 'commit_horizon')},
                          actual_nfe=extra['num_steps'], stage_calls=[1,1,extra['num_steps']],
                          task_name=extra['task_name'], original_init_idx=task['orig_init_state_idx'],
                          bundle_id=task['bundle_id'], bundle_identity='sealed-bundle',
                          source='policy', capture_enabled=False))
    for name, content in (('journal.jsonl',journal),('attempts.jsonl',attempts),('per_step.jsonl',steps)):
        write_rows(root/'driver'/name,content)
    write_json(root/'driver/DRIVER_RECEIPT.json', dict(status=status, all_done=status=='PASS',
               elapsed_seconds=40, rejected_worker_probes=0))
    write_json(root/'driver/census.json', dict(worker=dict(probe=probe)))


def continuation(source):
    destination = source.parent/'continuation'
    destination.mkdir()
    shutil.copytree(source/'assets',destination/'assets')
    campaign = dict(resume.read(source/'campaign.json'), run_name='continuation',
        endpoint_host='new.invalid', endpoint='new.invalid:23180', public_port=23180,
        local_assets=str(destination/'assets'), worker_memory_mib=6144, worker_restart_limit=2)
    resume.attach(campaign,source)
    spec, model, tasks, _ = formal_driver.plan(campaign)
    full = dict(resume.read(source/'EPISODE_PLAN.json'), tasks=[asdict(t) for t in tasks], endpoint=campaign['endpoint'])
    write_json(destination/'campaign.json',campaign)
    return destination,campaign,full


def test_resume_391_dispatches_only_1609_and_aggregates_exactly_2000(source):
    original = resume.inspect_source(source)
    destination,campaign,full = continuation(source)
    dispatch = resume.bind(campaign,destination,full)
    write_json(destination/'EPISODE_PLAN.json',dispatch)
    write_json(destination/'campaign.json',campaign)
    actual = formal_driver.plan(campaign)[2]
    assert len(actual)==1609
    assert {t.task_uid for t in actual} == set(campaign['resume_task_uids'])
    assert not {t.task_uid for t in actual} & set(original['evidence']['accepted'])
    # Identity and original seeds survive a different endpoint and worker budget.
    from exp.r17_integration.benchmarks.libero_bridge import decision_seed
    assert {t.task_uid:decision_seed(t.extra['task_name'],t.orig_init_state_idx,0) for t in actual} == {
        t['task_uid']:decision_seed(t['extra']['task_name'],t['orig_init_state_idx'],0) for t in dispatch['tasks']}
    write_json(destination/'BUNDLE_ACK.json',resume.read(source/'BUNDLE_ACK.json'))
    fill_driver(destination,dispatch['tasks'],probe=original['probe'])
    completion = formal_eval.driver_completion(destination,0)
    proof = resume.merge(campaign,destination,completion)
    summary = formal_eval.aggregate(destination,driver_returncode=0)
    assert (proof['retained_episodes'],proof['dispatched_episodes'])==(391,1609)
    assert summary['status']=='PASS'
    assert summary['expected_episodes']==summary['completed_episodes']==2000
    assert len(summary['tasks'])==40
    assert {s['planned'] for s in summary['suites'].values()}=={500}
    assert summary['resume']['source_status']=='FAIL'
    assert len(resume.rows(destination/'resume_segment/driver/journal.jsonl'))==1609
    assert len(resume.rows(destination/'driver/journal.jsonl'))==2000
    assert resume.read(source/'RECEIPT.json')['status']=='FAIL'
    assert all(resume.sha(source/name)==pin for name,pin in original['pins'].items())


@pytest.mark.parametrize('key,value',[
    ('model','dp'),('checkpoint','/different/checkpoint'),('official_50',False),
    ('domain','metaworld'),('commit_horizon',10),('denoising_steps',2),
    ('server_node','local'),('device','cuda:1'),
    ('fastwam_batching',{'max_batch':8}),('dp_batching',{'max_batch':8}),
    ('episode_seconds',1800),('retries',0),('suites',['libero_spatial']),('worker_egl_device',1)])
def test_cli_semantic_mismatches_refuse_before_deployment(source,key,value):
    campaign = dict(resume.read(source/'campaign.json'))
    campaign[key]=value
    with pytest.raises(ValueError,match='mismatch'):resume.attach(campaign,source)


@pytest.mark.parametrize('name',resume.SEALED)
def test_fresh_checkpoint_norm_code_benchmark_and_bundle_bytes_must_match(source,name):
    destination,campaign,full = continuation(source)
    path = destination/'assets'/name
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='sealed identity mismatch'):resume.bind(campaign,destination,full)


@pytest.mark.parametrize('field',['task_uid','seed','orig_init_state_idx','bundle_id'])
def test_task_uid_seed_cell_and_bundle_mismatch_refuse(source,field):
    destination,campaign,full=continuation(source)
    full['tasks'][0][field]='changed'
    with pytest.raises(ValueError,match='cell mismatch'):resume.bind(campaign,destination,full)


@pytest.mark.parametrize('damage',['duplicate','unknown','no_attempt','unresolved_error','trace','no_trace','zero_steps'])
def test_no_unsafe_accepted_source_episode_is_retained(source,damage):
    journal=resume.rows(source/'driver/journal.jsonl')
    attempts=resume.rows(source/'driver/attempts.jsonl')
    steps=resume.rows(source/'driver/per_step.jsonl')
    if damage=='duplicate':journal.append(journal[0])
    if damage=='unknown':journal[0]['task_uid']='unknown'
    if damage=='no_attempt':attempts=attempts[1:]
    if damage=='unresolved_error':attempts.append(dict(attempts[0],attempt=2,error='EGL error'))
    if damage=='trace':steps[0]['actual_nfe']=2
    if damage=='no_trace':steps=steps[1:]
    if damage=='zero_steps':attempts[0]['n_steps']=0
    for name,content in [('journal.jsonl',journal),('attempts.jsonl',attempts),('per_step.jsonl',steps)]:
        write_rows(source/'driver'/name,content)
    with pytest.raises(ValueError):resume.inspect_source(source)


def test_live_source_and_changed_source_are_refused(source):
    destination,campaign,full=continuation(source)
    write_json(source/'RECEIPT.json',dict(status='RUNNING',all_owned_closed=False))
    with pytest.raises(ValueError,match='running'):resume.inspect_source(source)
    with pytest.raises(ValueError,match='evidence changed'):resume.bind(campaign,destination,full)


def test_new_worker_probe_cannot_change_native_runtime_or_initial_state(source):
    destination,campaign,full=continuation(source)
    probe=resume.inspect_source(source)['probe']
    resume.validate_probe(dict(probe,hostname='another-node'),campaign)
    for key in ('libero_code_digest','torch','task_data'):
        with pytest.raises(ValueError,match='simulator/init'):resume.validate_probe(dict(probe,**{key:'different'}),campaign)


def test_missing_segment_cannot_be_merged(source):
    destination,campaign,full=continuation(source)
    dispatch=resume.bind(campaign,destination,full)
    write_json(destination/'EPISODE_PLAN.json',dispatch)
    write_json(destination/'BUNDLE_ACK.json',resume.read(source/'BUNDLE_ACK.json'))
    fill_driver(destination,dispatch['tasks'][:-1],probe=resume.inspect_source(source)['probe'])
    with pytest.raises(RuntimeError,match='driver incomplete'):formal_eval.driver_completion(destination,0)
    with pytest.raises(ValueError,match='incomplete'):resume.merge(campaign,destination,{'accepted_episodes':1608})
    assert not (destination/'RESUME_MERGE.json').exists()


def test_new_port_bank_is_distinct_and_cannot_span_a_live_bank():
    args=formal_eval.parse_args(['--model','fastwam_libero','--official-50','--replicas','2',
        '--port','23180','--driver-port','23189','--resume-from','/closed/run','--prepare-only'])
    assert args.port==23180 and args.driver_port==23189
    for public,driver,replicas in [('23180','23177','2'),('23178','23189','2'),('23180','23189','9')]:
        with pytest.raises(SystemExit):formal_eval.parse_args(['--prepare-only','--replicas',replicas,
            '--port',public,'--driver-port',driver])


def test_resume_runtime_is_a_new_sha_checked_copy_and_original_is_untouched(source):
    destination,campaign,full=continuation(source)
    before=resume.sha(source/'runtime.tar')
    assert resume.copy_runtime(campaign,destination/'runtime.tar')==before
    assert (source/'runtime.tar').read_bytes()==(destination/'runtime.tar').read_bytes()
    (source/'runtime.tar').write_bytes(b'changed native model code')
    with pytest.raises(ValueError,match='evidence changed'):resume.copy_runtime(campaign,destination/'another.tar')
