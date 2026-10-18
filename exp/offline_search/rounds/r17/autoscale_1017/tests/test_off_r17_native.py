"""Reproduce OFF timan1 N32 injection against the actual R17 probe validator."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from exp.offline_search.rounds.r17.autoscale_1017 import switch


@pytest.fixture(autouse=True)
def off(candidate_policy,monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017 import defaults,legacy_release
    from exp.offline_search.rounds.r17.filter_failed_knob import vla_workers,formal_a_queue
    from exp.offline_search.closed_loop.ops.h100 import fleet
    for m in (switch,defaults,legacy_release,vla_workers,formal_a_queue,fleet):monkeypatch.setattr(m,'enabled',lambda:False)
    for key in ('OPENPI_WORKER_OBSERVER','OPENPI_WORKER_GOVERNOR','OPENPI_AUTOSCALE_DRIVER_MODULE'):
        monkeypatch.delenv(key,raising=False)


def native(module,monkeypatch):
    value=types.ModuleType(module.__name__+'_off_native');value.__package__=module.__package__
    value.__file__=module.__file__;monkeypatch.setitem(sys.modules,value.__name__,value)
    exec(compile(switch.source(module.__file__),module.__file__,'exec'),value.__dict__)
    return value


def test_off_gpu2_real_driver_survives_more_than_32_episodes_without_renderer_restart(tmp_path,monkeypatch):
    from exp.offline_search.rounds.r17.filter_failed import timan1_gpu2 as gpu
    from exp.r17_integration.ops import formal_workers,worker_lifetime,timan1_workers
    from exp.r17_integration.tests.conductor.test_dispatch import setup
    from exp.r17_integration.conductor.driver import MetaWorldConductorDriver
    from exp.r17_integration.benchmarks.metaworld import MetaWorldBenchmarkAdapter
    from exp.r17_integration.conductor.manifest import validate_worker_probe
    from openpi.conductor import worker,protocol
    from openpi.conductor.task import EpisodeResult
    endpoints,tasks,strategy,_,_=setup(tmp_path,names=['reach-v3','push-v3'],indices=tuple(range(30,50)))
    driver=MetaWorldConductorDriver(strategy,dispatch_path=tmp_path/'dispatch-real.jsonl',
        yaml_weights={tasks[0].yaml_id:40},servers=[endpoints[0].endpoint],journal_path=str(tmp_path/'journal.jsonl'),
        ctl_factory=lambda e:types.SimpleNamespace(close=lambda:None),worker_probe_validator=validate_worker_probe)
    for stage in driver.scheduler.pending_setups():
        driver.scheduler.mark_setup_running(stage.stage_id);driver.scheduler.mark_setup_done(stage.stage_id)
    campaign=dict(run_name='fa_ls_gr_r_p_m',domain='metaworld',worker_node='timan1',node_workers=1,
        execution_worker_nodes={'timan1':1},worker_max_episodes=0,endpoint='local:1',driver_host='localhost',driver_port=1,
        worker_benchmark='fixture.json')
    # The actual OFF coordinator route overwrote N0 with N32 before this fix.
    gpu.route(campaign,campaign)
    monkeypatch.setattr(timan1_workers,'route',gpu.route);monkeypatch.setattr(timan1_workers,'wrapper',lambda c:'/python')
    command=native(formal_workers,monkeypatch).worker_command
    Worker=native(worker,monkeypatch).WorkerLoop
    lifetime=native(worker_lifetime,monkeypatch)
    received=[];replies=[];pulls=[]
    def send(sock,kind,payload):
        if kind==protocol.MSG_PULL:
            pulls.append(copy.deepcopy(payload));driver._record_pull(payload,('127.0.0.1',1234))
            replies.append(driver.handle_pull(payload['server_host']))
        elif kind==protocol.MSG_REPORT_RESULT:
            driver.handle_result(payload);received.append(payload['task_uid'])
    monkeypatch.setattr(protocol,'send_message',send)
    monkeypatch.setattr(protocol,'recv_message',lambda sock:replies.pop(0))
    class Runner:
        def run(self,task,report):return EpisodeResult(task.task_uid,True,1,attempt=task.attempt)
        def close(self):pass
    generations=0
    try:
        while len(received)<len(tasks):
            worker_id,argv=command(campaign,18)
            probe=MetaWorldBenchmarkAdapter().worker_probe(dict(renderer='egl',GL_VENDOR='NVIDIA fixture',
                pid=921639+31138*generations,gpu_uuid=gpu.UUID),dict(task_data_identity='fixture'))
            maximum=campaign['worker_max_episodes']
            Loop=lifetime.limited_loop(Worker,maximum,tmp_path/f'recycle_{generations}.json') if maximum else Worker
            loop=Loop(worker_id,campaign['endpoint'],Runner(),connect=lambda:types.SimpleNamespace(close=lambda:None),probe=probe)
            generations+=1
            try:loop.run_forever(stop=lambda:len(received)>=len(tasks))
            except SystemExit as exc:assert exc.code==75
        assert len(received)==len(set(received))==40 and generations==1
        assert campaign['worker_max_episodes']==0 and '--max-episodes' not in argv
        assert not driver.rejected_worker_probes
        # Validator still rejects an actual PID/endpoint/probe change.
        changed=copy.deepcopy(pulls[0]);changed['probe']['renderer']['pid']+=1
        with pytest.raises(ValueError,match='probe changed'):driver._record_pull(changed,('127.0.0.1',1234))
    finally:driver.close()


def test_off_bootstrap_uses_exact_native_gpu2_source():
    import ast
    from exp.offline_search.rounds.r17.filter_failed import timan1_gpu2 as gpu
    argv=gpu.bootstrap('/python',['-m',gpu.WORKER],'fleet')
    tree=ast.parse(argv[4]);body=ast.literal_eval(tree.body[0].value)
    baseline=(Path(switch.__file__).parent/'native_sources'/str(Path(gpu.__file__).relative_to(Path(switch.__file__).parents[5])).replace('/','__')).read_text()
    assert body==baseline


def test_off_worker_and_driver_injections_use_native_functions():
    import ast
    from exp.offline_search.rounds.r17.filter_failed import worker_resilience as r
    baseline=(Path(switch.__file__).parent/'native_sources'/str(Path(r.__file__).relative_to(Path(switch.__file__).parents[5])).replace('/','__')).read_text()
    tree=ast.parse(baseline)
    expected={n.name:ast.get_source_segment(baseline,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    command=r.driver_command(['python','-m','exp.r17_integration.ops.formal_driver','--campaign','fixture'])
    assert expected['install_driver'] in command[2]
    assert expected['install_worker'] in r.worker_patch()


def test_off_allocation_preserves_requested_native_nodes():
    from exp.offline_search.rounds.r17.filter_failed.worker_resilience import allocate
    requested={'timan1':20,'timan107':20}
    measured={'timan1':{'capacity':83,'mem_available_bytes':400},'timan107':{'capacity':104,'mem_available_bytes':200},'local':{'capacity':50,'mem_available_bytes':100}}
    assert allocate(requested,measured)==requested


def test_off_measure_and_legacy_probe_inject_native_capacity(monkeypatch,tmp_path):
    from exp.offline_search.rounds.r17.filter_failed import worker_resilience as r
    from exp.offline_search.rounds.r17.filter_failed_knob import vla_workers as v
    from exp.r17_integration.ops import formal_eval as f
    expected=switch.function_source(r.capacity).strip()
    captured=[]
    def execute(argv,**kwargs):
        captured.append(argv[-1])
        return json.dumps(dict(capacity=3))
    monkeypatch.setattr(f,'execute',execute)
    campaign=dict(run_name='off_fixture',domain='metaworld',server_node='h100',
                  worker_nodes={'timan1':20},autoscale={'mode':'dynamic'},autoscale_node_maxima={'timan1':332})
    rows=r.measure(campaign,tmp_path)
    assert set(rows)==set(r.NODES) and all(row['capacity']==3 and 'autoscale' not in row for row in rows.values())
    assert len(captured)==3 and all(expected in code for code in captured)
    def run(argv,**kwargs):
        assert expected in argv[-1]
        return types.SimpleNamespace(stdout=json.dumps(dict(host='timan107',missing=[])))
    monkeypatch.setattr(v.subprocess,'run',run)
    assert v.probe('timan107','R16')['usable']


def test_off_payload_all_patched_worker_modules_match_prepatch_bytes(tmp_path):
    from exp.r17_integration.ops.build_client_bundle import build_client_bundle,verified_members
    base=Path(switch.__file__).parent;repo=base.parents[4]
    pins=json.loads((base/'PATCH_BASE.json').read_text())
    names=[n for n in pins if n.startswith(('src/openpi/conductor/', 'exp/r17_integration/ops/',
        'exp/r17_integration/conductor/')) and not n.endswith('formal_eval.py')]
    build_client_bundle(repo,names,tmp_path/'workers.tar')
    manifest,content=verified_members(tmp_path/'workers.tar','PAYLOAD.json')
    assert manifest['files']=={n:pins[n]['sha256'] for n in names}
    assert not any('autoscale_1017' in n for n in content)
    (tmp_path/'BYTE_PROOF.json').write_text(json.dumps(manifest['files'],indent=2))


def test_off_driver_release_matches_prepatch_on_same_real_frozen_archive(tmp_path,monkeypatch):
    import shutil
    from exp.r17_integration.ops import driver_release as d,immutable_release as im
    baseline=native(d,monkeypatch)
    base=Path(switch.__file__).parent;repo=base.parents[4]
    source=Path('/data/mw_finetune/eval/fa_ls_gr_r_p_m/runtime.tar')
    proof=[]
    for label,module in [('before',baseline),('after',d)]:
        root=tmp_path/label;root.mkdir();shutil.copyfile(source,root/'runtime.tar')
        monkeypatch.setattr(module,'install',lambda *a:None)
        c=dict(runtime_identity=im.identity(im.archive_manifest(root/'runtime.tar')),driver_mode='group-commit')
        module.prepare(c,root,repo)
        proof.append((im.archive_manifest(root/'driver_runtime.tar'),c['orchestration']))
    assert proof[0]==proof[1]
    assert (tmp_path/'before/driver_runtime.tar').read_bytes()==(tmp_path/'after/driver_runtime.tar').read_bytes()
    assert not any('autoscale_1017' in n for n in proof[1][0]['files'])
    (tmp_path/'DRIVER_BYTE_PROOF.json').write_text(json.dumps(dict(before=im.sha(tmp_path/'before/driver_runtime.tar'),
        after=im.sha(tmp_path/'after/driver_runtime.tar'),identical=True),indent=2))
