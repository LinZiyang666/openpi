"""No GPU, subprocess launch, tunnel allocation, or signals: all I/O mocked."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from exp.r17_integration.ops import formal_eval as launch, formal_resume as resume
from exp.r17_integration.ops import formal_resume_waiver as waiver
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.tests.test_formal_resume import source, source_master, fill_driver

OFFLINE = 'error: exec: the agent is OFFLINE (node_offline: status=OFFLINE)'


@pytest.fixture
def closure_source(tmp_path):
    root = tmp_path/'closure_source'
    root.mkdir()
    campaign = dict(run_name='source', server_node='h100', replicas=2, worker_nodes={'timan108':24},
                    public_port=23240, backend_ports=[23241,23242], driver_port=23249,
                    ports=[23240,23241,23242,23249], endpoint='broker.invalid:14008',
                    endpoint_port=14008, driver_public_port=14009)
    local = [dict(label='driver', pid=100, pgid=100, start_ticks='1000')]
    remote = []
    for i, label in enumerate(('proxy','replica_0','replica_1','workers')):
        node = 'timan108' if label=='workers' else 'h100'
        remote.append(dict(node=node, label=label, request={'receipt':'/source/'+label+'.json'},
                           receipt={role:dict(pid=200+2*i+j,start_ticks=str(2000+2*i+j))
                                    for j,role in enumerate(('child','supervisor'))}))
    cleanup = [dict(label='driver',status='closed',closure_confirmed=True)]
    cleanup += [dict(node=r['node'],label=r['label'],closure_confirmed=r['label']!='workers') for r in remote]
    write_json(root/'campaign.json',campaign)
    write_json(root/'RECEIPT.json',dict(status='FAIL',all_owned_closed=False,
                                     owned_local=local,owned_remote=remote,cleanup=cleanup))
    return root


@pytest.fixture
def destination():
    return dict(run_name='continuation',server_node='h100',replicas=2,worker_nodes={'timan107':12},
                public_port=23243,backend_ports=[23244,23245],driver_port=23248,
                ports=[23243,23244,23245,23248],endpoint='new.invalid:23243',
                endpoint_port=23243,driver_public_port=23248)


@pytest.fixture(autouse=True)
def read_only_io(monkeypatch):
    monkeypatch.setattr(waiver,'alive',lambda row:False)
    def run(argv,**kw):
        assert argv[:2]==['tether','exec']
        node=argv[argv.index('--')-1]
        if node=='timan108':
            return subprocess.CompletedProcess(argv,64,'',OFFLINE)
        assert node=='h100'
        # Simulate exact /proc start-tick validation from the remote verifier.
        records=[]
        for item in json.loads(argv[-1]):
            records.append(dict(label=item['label'],receipt_path=item['path'],receipt_sha256='immutable',
                processes=[dict(role=role,pid=value['pid'],start_ticks=value['start_ticks'],alive=False)
                           for role,value in item['expected'].items()]))
        return subprocess.CompletedProcess(argv,0,json.dumps(records),'')
    monkeypatch.setattr(waiver.subprocess,'run',run)


def test_explicit_waiver_records_evidence_without_faking_source_closure(closure_source,destination):
    before=(closure_source/'RECEIPT.json').read_bytes()
    proof=waiver.issue(closure_source,destination,'timan108')
    waiver.validate(closure_source,proof)
    assert proof['node']=='timan108' and proof['issued_at']
    assert proof['unreachable_probe']['returncode']==64
    assert {(r['node'],r['label']) for r in proof['verified_closed']}=={
        ('local','driver'),('h100','proxy'),('h100','replica_0'),('h100','replica_1')}
    assert proof['unverified_processes']==[dict(node='timan108',label='workers',receipt_path='/source/workers.json')]
    assert (closure_source/'RECEIPT.json').read_bytes()==before
    with pytest.raises(ValueError,match='must be closed'):
        resume.inspect_source(closure_source)


@pytest.mark.parametrize('status',['PARTIAL','RUNNING','CLOSED',None])
def test_nonterminal_source_rejected(closure_source,destination,status):
    receipt=resume.read(closure_source/'RECEIPT.json')
    write_json(closure_source/'RECEIPT.json',dict(receipt,status=status))
    with pytest.raises(ValueError,match='terminal source'):
        waiver.issue(closure_source,destination,'timan108')


@pytest.mark.parametrize('role',['driver','proxy','replica_0','replica_1'])
def test_any_unverified_source_service_rejected(closure_source,destination,role):
    receipt=resume.read(closure_source/'RECEIPT.json')
    for row in receipt['cleanup']:
        if row['label']==role:
            row['closure_confirmed']=False
    write_json(closure_source/'RECEIPT.json',receipt)
    with pytest.raises(ValueError,match='only the named worker node'):
        waiver.issue(closure_source,destination,'timan108')


def test_second_unverified_worker_node_rejected(closure_source,destination):
    receipt=resume.read(closure_source/'RECEIPT.json')
    receipt['owned_remote'].append(dict(receipt['owned_remote'][-1],node='timan107'))
    receipt['cleanup'].append(dict(label='workers',node='timan107',closure_confirmed=False))
    write_json(closure_source/'RECEIPT.json',receipt)
    with pytest.raises(ValueError,match='only the named worker node'):
        waiver.issue(closure_source,destination,'timan108')


def test_missing_replica_or_cleanup_record_rejected(closure_source,destination):
    receipt=resume.read(closure_source/'RECEIPT.json')
    receipt['owned_remote']=[r for r in receipt['owned_remote'] if r['label']!='replica_1']
    write_json(closure_source/'RECEIPT.json',receipt)
    with pytest.raises(ValueError,match='every replica'):
        waiver.issue(closure_source,destination,'timan108')
    receipt['cleanup']=receipt['cleanup'][1:]
    write_json(closure_source/'RECEIPT.json',receipt)
    with pytest.raises(ValueError,match='one closure record'):
        waiver.issue(closure_source,destination,'timan108')


@pytest.mark.parametrize('rc,stderr',[(0,''),(64,'authorization error'),(1,OFFLINE)])
def test_reachable_or_unproven_offline_node_rejected(closure_source,destination,monkeypatch,rc,stderr):
    monkeypatch.setattr(waiver.subprocess,'run',lambda argv,**kw:subprocess.CompletedProcess(argv,rc,'',stderr))
    with pytest.raises(ValueError,match='node_offline evidence'):
        waiver.issue(closure_source,destination,'timan108')


def test_live_local_or_remote_process_rejected(closure_source,destination,monkeypatch):
    monkeypatch.setattr(waiver,'alive',lambda row:True)
    with pytest.raises(ValueError,match='still alive: driver'):
        waiver.issue(closure_source,destination,'timan108')
    monkeypatch.setattr(waiver,'alive',lambda row:False)
    run=waiver.subprocess.run
    def live_remote(argv,**kwargs):
        result=run(argv,**kwargs)
        if result.returncode==0:
            records=json.loads(result.stdout)
            records[0]['processes'][0]['alive']=True
            return subprocess.CompletedProcess(argv,0,json.dumps(records),'')
        return result
    monkeypatch.setattr(waiver.subprocess,'run',live_remote)
    with pytest.raises(ValueError,match='still alive on h100'):
        waiver.issue(closure_source,destination,'timan108')


@pytest.mark.parametrize('old_port',[14008,14009,23240,23241,23242,23249])
def test_old_direct_or_tunnel_port_reuse_rejected(closure_source,destination,old_port):
    destination['endpoint']='different-host.invalid:'+str(old_port)
    with pytest.raises(ValueError,match='disjoint destination'):
        waiver.issue(closure_source,destination,'timan108')


def test_run_id_and_destination_node_isolation(closure_source,destination):
    with pytest.raises(ValueError,match='different destination run id'):
        waiver.issue(closure_source,dict(destination,run_name='source'),'timan108')
    with pytest.raises(ValueError,match='cannot use the unreachable'):
        waiver.issue(closure_source,dict(destination,worker_nodes={'timan108':1}),'timan108')
    proof=waiver.issue(closure_source,destination,'timan108')
    with pytest.raises(ValueError,match='direct ports changed'):
        waiver.validate_destination(dict(destination,driver_port=23247),proof)


def test_proof_mutation_or_changed_source_rejected(closure_source,destination):
    proof=waiver.issue(closure_source,destination,'timan108')
    bad=deepcopy(proof)
    bad['verified_closed'][1]['processes'][0]['start_ticks']='different'
    with pytest.raises(ValueError,match='process closure evidence'):
        waiver.validate(closure_source,bad)
    bad=deepcopy(proof)
    bad['unreachable_probe']['argv'][5]='timan107'
    with pytest.raises(ValueError,match='unreachable-node evidence'):
        waiver.validate(closure_source,bad)
    receipt=resume.read(closure_source/'RECEIPT.json')
    write_json(closure_source/'RECEIPT.json',dict(receipt,error='modified after proof'))
    with pytest.raises(ValueError,match='source evidence changed'):
        waiver.validate(closure_source,proof)


def test_cli_exception_is_explicit_and_requires_resume():
    common=['--model','cosmos_libero','--prepare-only','--worker-node','timan107',
            '--port','23243','--driver-port','23248','--replicas','2']
    assert launch.parse_args(common).resume_waive_unreachable_source_node is None
    with pytest.raises(SystemExit):
        launch.parse_args([*common,'--resume-waive-unreachable-source-node','timan108'])
    with pytest.raises(SystemExit):
        launch.parse_args([*common,'--resume-from','/source','--resume-waive-unreachable-source-node','h100'])
    args=launch.parse_args([*common,'--resume-from','/source','--resume-waive-unreachable-source-node','timan108'])
    assert args.resume_waive_unreachable_source_node=='timan108'


def test_prepared_waiver_rechecks_reachability_before_launch(closure_source,destination,tmp_path,monkeypatch):
    proof=waiver.issue(closure_source,destination,'timan108')
    destination.update(model='cosmos',resume_source_closure_waiver=proof)
    monkeypatch.setattr(waiver.subprocess,'run',lambda argv,**kw:subprocess.CompletedProcess(argv,0,'online',''))
    with pytest.raises(ValueError,match='node_offline evidence'):
        launch.open_tunnels(destination,tmp_path)


def test_waiver_1726_retained_274_dispatched_proof_reaches_merged_summary(source,closure_source,destination):
    original=resume.read(source/'campaign.json')
    old=resume.read(closure_source/'campaign.json')
    original.update({k:v for k,v in old.items() if k!='run_name'})
    write_json(source/'campaign.json',original)
    shutil.copyfile(closure_source/'RECEIPT.json',source/'RECEIPT.json')
    before=(source/'RECEIPT.json').read_bytes()
    all_tasks=resume.read(source/'EPISODE_PLAN.json')['tasks']
    probe=resume.probe_binding(resume.read(source/'driver/census.json')['worker']['probe'])
    fill_driver(source,all_tasks[:1726],probe=probe,status='PARTIAL')
    with pytest.raises(ValueError,match='must be closed'):
        resume.inspect_source(source)
    target=source.parent/'continuation'
    target.mkdir()
    shutil.copytree(source/'assets',target/'assets')
    campaign=dict(original,**destination,local_assets=str(target/'assets'),
                  resume_waive_unreachable_source_node='timan108')
    resume.attach(campaign,source)
    from exp.r17_integration.ops.formal_driver import plan
    tasks=plan(campaign)[2]
    full=dict(resume.read(source/'EPISODE_PLAN.json'),tasks=[asdict(t) for t in tasks])
    dispatch=resume.bind(campaign,target,full)
    assert len(dispatch['tasks'])==274
    assert not set(campaign['resume_task_uids']) & {t['task_uid'] for t in all_tasks[:1726]}
    write_json(target/'EPISODE_PLAN.json',dispatch)
    write_json(target/'campaign.json',campaign)
    write_json(target/'BUNDLE_ACK.json',resume.read(source/'BUNDLE_ACK.json'))
    fill_driver(target,dispatch['tasks'],probe=probe)
    completion=launch.driver_completion(target,0)
    selection=resume.read(target/'RESUME.json')
    changed=deepcopy(selection)
    changed['source_closure_waiver']['issued_at']='changed after binding'
    write_json(target/'RESUME.json',changed)
    with pytest.raises(ValueError,match='closure waiver evidence changed'):
        resume.merge(campaign,target,completion)
    write_json(target/'RESUME.json',selection)
    proof=resume.merge(campaign,target,completion)
    summary=launch.aggregate(target,driver_returncode=0)
    assert (proof['retained_episodes'],proof['dispatched_episodes'])==(1726,274)
    assert summary['expected_episodes']==summary['completed_episodes']==2000
    assert summary['status']=='PASS'
    assert summary['resume']['source_closure_waiver']==campaign['resume_source_closure_waiver']
    assert proof['source_closure_waiver']['issued_at']
    assert (source/'RECEIPT.json').read_bytes()==before


def test_resume_receipt_keeps_waiver_even_if_launch_fails_before_spawn(closure_source,destination,tmp_path,monkeypatch):
    proof=waiver.issue(closure_source,destination,'timan108')
    campaign=dict(destination,model='cosmos',runtime='/cpu/fixture',seconds=60,server_campaign='/cpu/campaign.json',
                  resume_source_closure_waiver=proof)
    monkeypatch.setattr(launch,'host_python',lambda _: '/cpu/python')
    monkeypatch.setattr(launch,'server_env',lambda *_: {})
    def fail_before_spawn(*args,**kwargs):
        raise RuntimeError('CPU test abort before spawn')
    monkeypatch.setattr(launch,'remote',fail_before_spawn)
    assert launch.run_campaign(campaign,tmp_path,{})==2
    receipt=resume.read(tmp_path/'RECEIPT.json')
    assert receipt['status']=='FAIL'
    assert not receipt['owned_local'] and not receipt['owned_remote']
    assert receipt['source_closure_waiver']==proof
