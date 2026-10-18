"""Real prepare entries, redirected output; OFF retains native science and payloads."""
import copy
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import pytest
from exp.offline_search.rounds.r17.autoscale_1017 import defaults, switch, legacy_release
from exp.offline_search.rounds.r17.autoscale_1017.node import read, atomic
BASE=Path(__file__).resolve().parents[1]
REPO=BASE.parents[4]


def test_owner_switch_defaults_off_and_no_transport(tmp_path,monkeypatch):
    assert not switch.enabled()
    from exp.r17_integration.ops import formal_eval
    monkeypatch.setattr(formal_eval,'transfer',lambda *a:pytest.fail('OFF transfer'))
    monkeypatch.setattr(legacy_release,'build',lambda *a:pytest.fail('OFF overlay build'))
    assert legacy_release.stage(tmp_path,'timan107',pipeline='R15',maximum=39,base='/native') is None
    assert legacy_release.coordinator(tmp_path,[]) is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('name,pipeline',[('fa_pi_f_u_o','R15'),('fa_pi_r_f_l','R16')])
def test_native_queue_prepare(tmp_path,monkeypatch,name,pipeline):
    from exp.offline_search.rounds.r17.autoscale_1017.dryrun import unit
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_queue as q,vla_workers as v
    from exp.offline_search.closed_loop.ops.h100 import control,fleet
    from exp.r17_integration.ops import formal_eval
    u=unit(name);jobs=copy.deepcopy(u['jobs'])
    if not jobs:jobs=[dict(name=name,pipeline=pipeline,run=str(tmp_path/name),workers=u['workers'])]
    for j in jobs:
        j['run']=str(tmp_path/j['name']);Path(j['run']).mkdir()
        atomic(Path(j['run'])/'arms.json',[dict(arm='native_unchanged')])
    u['jobs']=jobs;u['status']='BUILT'
    monkeypatch.setattr(q,'INDEX',tmp_path/'QUEUE.json');atomic(q.INDEX,dict(units=[u]))
    monkeypatch.setattr(v,'probe',lambda h,p:dict(usable=h=='timan107',installed=True,island='/native/island'))
    monkeypatch.setattr(v,'stage_r16',lambda *a:None)
    monkeypatch.setattr(formal_eval,'transfer',lambda *a:pytest.fail('OFF overlay transfer'))
    monkeypatch.setattr(control,'remote',lambda *a,**k:pytest.fail('unexpected remote'))
    artifacts=[]
    def execute(argv,**kw):
        assert argv[0]=='bash'
        j=next(j for j in jobs if Path(argv[1]).name=='prepare_'+j['name']+'.native.sh')
        if pipeline=='R15':
            cmd,env=v.r15_command(j,'prepare','timan107')
            expected=[str(q.f.REPO/'.venv/bin/python'),'-m','exp.offline_search.rounds.r17.filter_failed.legacy','sync','--concurrent',j['run'],'native_unchanged']
            assert cmd==expected
            assert not any(k.startswith('OPENPI_AUTOSCALE') or k=='OPENPI_DESIRED_MAXIMUM' for k in env)
            assert env['WPS']==j['topology']['env']['WPS']=='13'
            artifacts.append(dict(name=j['name'],workers=39,command=cmd,overlay=False))
        else:
            v.prepare_r16_job(j)
            selection=read(Path(j['run'])/'WORKER_HOST_SELECTION.json')
            assert selection['execution_workers']=={'timan107':64}
            assert 'autoscale' not in selection
            assert not (Path(j['run'])/'LEGACY_AUTOSCALE.json').exists()
            artifacts.append(dict(name=j['name'],workers=64,overlay=False))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(q.subprocess,'run',execute)
    q.dispatch('prepare',name)
    assert len(artifacts)==len(jobs)
    atomic(tmp_path/'NATIVE_COMPARISON.json',dict(status='PASS',name=name,artifacts=artifacts))
    assert fleet.FLEETS['timan107'][2]==64


def test_r17_native_stage_matches_prepatch_same_inputs(tmp_path,monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017.tests.test_defaults import dry_stage
    from exp.offline_search.rounds.r17.filter_failed import stage
    source=Path('/data/mw_finetune/eval/fa_ls_gr_r_p_m')
    c=read(source/'campaign.json');before=(source/'campaign.json').read_bytes()
    # Test identical requested inputs; live native failover can change the source's
    # execution map after this fixture was first written.
    c.update(execution_worker_nodes={'timan1':20,'timan107':20},execution_workers=40)
    # Real stage path; frozen plan fixture isolates transport, identical on both sides.
    actual=tmp_path/'after';baseline=tmp_path/'before'
    actual.mkdir();baseline.mkdir()
    with monkeypatch.context() as m:
        a=dry_stage(actual/'fa_ls_gr_r_p_m',c,m,invoke=stage.stage)
    with monkeypatch.context() as m:
        m.setattr(defaults,'apply',lambda c,*a,**kw:c) # prepatch has no policy call
        b=dry_stage(baseline/'fa_ls_gr_r_p_m',c,m,invoke=stage.stage)
    def normalize(value,where):return json.loads(json.dumps(value).replace(str(where),'<OUTPUT>'))
    assert normalize(a,actual)==normalize(b,baseline)
    assert 'autoscale' not in a and a['worker_max_episodes']==0
    assert a['execution_worker_nodes']=={'timan1':20,'timan107':20}
    for f in ('workers.tar','runtime.tar','EPISODE_PLAN.json','PREPARED.json'):
        assert (actual/'fa_ls_gr_r_p_m'/f).read_bytes()==(baseline/'fa_ls_gr_r_p_m'/f).read_bytes(),f
    assert (source/'campaign.json').read_bytes()==before


def test_native_worker_bundle_matches_verified_prepatch_bytes(tmp_path):
    from exp.r17_integration.ops.build_client_bundle import build_client_bundle,verified_members
    pins=read(BASE/'PATCH_BASE.json')
    names=[n for n in pins if n.startswith(('src/openpi/conductor/','exp/r17_integration/ops/')) and not n.endswith('formal_eval.py')]
    build_client_bundle(REPO,names,tmp_path/'workers.tar')
    manifest,content=verified_members(tmp_path/'workers.tar','PAYLOAD.json')
    assert manifest['files']=={n:pins[n]['sha256'] for n in names}
    assert all('autoscale_1017' not in n for n in content)


def test_transport_probe_fix_preserved_and_production_bind_forbidden():
    from exp.offline_search.closed_loop.ops.h100 import control
    assert 's.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)' in Path(control.__file__).read_text()
    assert b'SO_REUSEADDR' in switch.source(Path(control.__file__))
    with socket.socket() as sock:
        with pytest.raises(AssertionError,match='production port'):sock.bind(('127.0.0.1',23197))


def test_off_routes_dead_node_without_raising_native_total():
    assert defaults.queue_workers(dict(workers={'timan1':35,'timan107':25,'timan108':25}))=={'timan1':35,'timan107':50}
    c=dict(worker_nodes={'timan108':24},workers=24)
    defaults.apply(c)
    assert c['execution_worker_nodes']=={'timan107':24} and c['workers']==24
    assert 'autoscale' not in c


def test_off_reprepare_invalidates_old_dynamic_seal_and_preserves_it(tmp_path):
    atomic(tmp_path/'PREPARED.json',dict(mode='dynamic'))
    c=dict(autoscale={'mode':'dynamic'},domain='metaworld',worker_nodes={'timan1':60},worker_max_episodes=32)
    defaults.apply(c,tmp_path)
    assert not (tmp_path/'PREPARED.json').exists()
    saved=list(tmp_path.glob('PREPARED.before_off.*.json'));assert len(saved)==1
    assert read(saved[0])==dict(mode='dynamic') and c['worker_max_episodes']==0
    assert 'autoscale' not in c


def test_off_never_reseals_a_launched_campaign(tmp_path):
    atomic(tmp_path/'PREPARED.json',{});atomic(tmp_path/'LAUNCH.json',{})
    c=dict(autoscale={'mode':'dynamic'},worker_nodes={'timan1':60})
    with pytest.raises(ValueError,match='already launched'):defaults.apply(c,tmp_path)
    assert (tmp_path/'PREPARED.json').exists() and c['autoscale']['mode']=='dynamic'
