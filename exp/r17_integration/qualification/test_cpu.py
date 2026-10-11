"""Meaningful Q integration checks; fixtures cannot confer GPU qualification."""
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest


def test_owned_deadline_and_foreign_process(tmp_path):
    from exp.r17_integration.qualification.run_slot import Owned,Partial
    report={'owned_pids':[]}
    foreign=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    try:
        owned=Owned(tmp_path,report,time.monotonic()+.5)
        child=owned.spawn('owned',[sys.executable,'-c','import time; time.sleep(30)'])
        with pytest.raises(Partial): owned.wait(child)
        owned.close()
        assert child.poll() is not None and foreign.poll() is None
        assert report['owned_pids'][0]['exit_code'] is not None
    finally:
        foreign.terminate();foreign.wait(timeout=3)


def test_source_drift_defers_before_loading_assets(tmp_path):
    import hashlib
    from exp.r17_integration.qualification.run_slot import preflight,Partial
    snapshot=tmp_path/'exp/r17_integration/qualification/runtime_code_snapshot.json'
    snapshot.parent.mkdir(parents=True)
    source=tmp_path/'provider.py';source.write_text('owner changed source\n')
    snapshot.write_text(json.dumps(dict(schema='r17.Q.runtime_code.v1',files={
        'provider.py':hashlib.sha256(b'previous source\n').hexdigest()})))
    with pytest.raises(Partial,match='source changed.*provider.py'):
        preflight(dict(runtime=str(tmp_path)))
    assert source.read_text()=='owner changed source\n'


def test_bfloat16_observer_keeps_native_bits():
    import torch
    from exp.r17_integration.qualification.capture_portable import PortableSink
    from exp.r17_integration.common.contracts import DecisionCapture,RequestContext
    from exp.r17_integration.debug.validate import numeric_arrays
    x=torch.tensor([[1.125,-2.5]],dtype=torch.bfloat16)
    seen=[]
    sink=PortableSink(seen.append)
    cap=DecisionCapture(RequestContext('cpu','cpu','cpu',0,0),'policy',0,0,(1,1,1),1,x,x.float().numpy(),
        model_capture={'initial_noise':x})
    sink(cap)
    stored=seen[0]
    raw=stored.model_capture['q_original_bfloat16_bits']['normalized_chunk']
    assert np.array_equal(raw,x.view(torch.uint16).numpy())
    restored=torch.from_numpy(raw.copy()).view(torch.bfloat16)
    assert torch.equal(restored,x) and cap.normalized_chunk is x
    assert np.array_equal(numeric_arrays({'x':stored.normalized_chunk})['x'],x.float().numpy())
    encoding={};native=numeric_arrays({'native':x},serialization=encoding)
    assert encoding['native']['native_dtype']=='bfloat16'
    assert torch.equal(torch.from_numpy(native[encoding['native']['bits_array']]).view(torch.bfloat16),x)


def test_libero_recipe_uses_actual_task_vocabulary(tmp_path):
    from exp.r17_integration.tests.debug.fixtures import spec
    from exp.r17_integration.common.provider import AdapterProvider
    from exp.r17_integration.common.qualification import config_for_probe
    from exp.r17_integration.common.recipe import make_recipe_factory
    from exp.r17_integration.common.contracts import digest
    from exp.r17_integration.search.store import Store
    from exp.r17_integration.qualification.assets import synthetic_store
    from exp.r17_integration.qualification.benchmark_binding import benchmark_spec,recipe_settings
    benchmark=benchmark_spec(dict(domain='libero',benchmark_spec=str(Path(__file__).parent/'specs/libero_benchmark_spec.json')))
    name=benchmark.tasks[0][0]
    s=spec('pi05','libero',key_width=64)
    s=dataclasses.replace(s,allowed_benchmark_protocols=(benchmark.identity,))
    provider=AdapterProvider('q_libero_vocabulary_fixture',s,lambda:None,lambda k:s.default_sampler)
    synthetic_store(s,provider,tmp_path,task_name=name)
    store=Store(tmp_path/'synthetic_teacher_store')
    cfg=config_for_probe(provider,10,5)
    cfg.recipe.enabled=True;cfg.recipe.store_path=str(store.directory)
    cfg.recipe.store_digest=store.manifest['content_digest']
    cfg.recipe.teacher_sampler_identity=digest(store.manifest['binding']['sampler'])
    for key,value in recipe_settings(benchmark.identity).items():setattr(cfg.recipe,key,value)
    cfg.recipe.corrector=True
    assert make_recipe_factory(provider)(cfg,None) is not None
    assert all(row['task_name']==name for row in store.rows)
    cfg.recipe.ordered_tasks=['libero_mechanics']
    with pytest.raises((ValueError,KeyError)):
        make_recipe_factory(provider)(cfg,None)


def test_owner_zero_scatter_rejects_clearly():
    from exp.r17_integration.common.layers.math import fit_metric
    with pytest.raises(ValueError,match='zero-scatter'):
        fit_metric(np.ones((6,4)),np.ones((6,20)),np.repeat(np.arange(3),2))


def test_closed_mechanics_audit_and_physical_disagreement(tmp_path):
    from exp.r17_integration.debug.capture_bridge import CaptureBridge
    from exp.r17_integration.tests.debug.fixtures import binding,record,arrays
    from exp.r17_integration.debug.validate import EPISODE_KEYS
    from exp.r17_integration.qualification.validate_results import audit
    b=binding('pi05','mw');assets=tmp_path/'assets';assets.mkdir()
    (assets/'model_spec.json').write_text(json.dumps(b['model']))
    job=dict(assets=str(assets),domain='metaworld')
    run=tmp_path/'run';run.mkdir()
    bridge=CaptureBridge(run/'model_capture/binding_cpu',enabled=True,binding=b)
    journal=[];physical=[];expected=[]
    for i,arm in enumerate(('full','retrieval')):
        row=record(i,arm=arm,n_decisions=1)
        row.update(stage_calls=[1,1,10] if arm=='full' else [1,0,0],actual_nfe=10 if arm=='full' else 0,
                   source='policy' if arm=='full' else 'library',cursor_end=5,terminal=True,
                   execution_status='acknowledged')
        # Capture writers own the top-level queue_wait_ms; keep policy timings
        # in their native diagnostics ledger, as real P1/P2 captures do.
        row['diagnostics']={key:row[key] for key in
            ('cost_ms','queue_wait_ms','service_ms','pricing_time_field')}
        payload,keys=arrays(b,i);bridge.capture(row,payload,rawkeys=keys)
        journal.append(dict(row,accepted=True,status='completed',error=None))
        expected.append(row['task_uid'])
        physical.extend(dict({key:row[key] for key in EPISODE_KEYS},decision_seq=0,control_idx=k,event='control') for k in range(5))
    bridge.close()
    for name in ('WS_RECEIPT','SERVER_CLOSED'):
        (run/(name+'.json')).write_text(json.dumps(dict(status='PASS',synthetic_cpu_fixture=True)))
    result=audit(job,run,journal,physical,expected)
    assert result['status']=='PASS' and result['audit']['cost_status']=='available'
    assert all(r['server_ack']=='acknowledged_terminal' for r in result['terminal_execution'])
    assert list((run/'analysis').glob('*.png')) and list((run/'analysis').glob('*.html'))
    changed=[dict(r) for r in physical];changed[-1]['task_data_hash']='contradictory'
    with pytest.raises(AssertionError,match='identity disagreement'): audit(job,run,journal,changed,expected)
    with pytest.raises(AssertionError,match='execution ack disagrees'): audit(job,run,journal,physical[:-1],expected)
    duplicate=physical+[dict(physical[-1])]
    with pytest.raises(AssertionError,match='missing or duplicated'): audit(job,run,journal,duplicate,expected)
    with pytest.raises(AssertionError,match='actual worker/model error'):
        audit(job,run,journal+[dict(task_uid='error',accepted=False,error='actual failure')],physical,expected)


def test_h100_mechanical_start_wait_uses_owned_scope(tmp_path,monkeypatch):
    from exp.r17_integration.qualification import run_slot,libero_remote,validate_results
    stopped=[]
    class Child:
        def poll(self):
            (tmp_path/'SERVER_READY.json').write_text('{}')
            return None
    class Owner:
        def spawn(self,*args): return Child()
        def remaining(self): return 10
        def stop(self,child): stopped.append(child)
    monkeypatch.setattr(libero_remote,'run_libero',lambda *args:None)
    monkeypatch.setattr(validate_results,'validate_libero',lambda *args:dict(status='PASS',synthetic_cpu_fixture=True))
    job=dict(host='h100',domain='libero',python=sys.executable)
    assert run_slot.mechanical(job,tmp_path,Owner(),{})['status']=='PASS' and len(stopped)==1


def test_g3_full_cpu_composition(tmp_path,monkeypatch):
    from exp.r17_integration.tests.models.pi05.test_pi05 import fixture_adapter,observation
    from exp.r17_integration.models.pi05.provider import register_provider
    from exp.r17_integration.qualification import model_runtime,g3
    from exp.r17_integration.common.qualification import config_for_probe
    import yaml
    adapter,_=fixture_adapter()
    provider=register_provider(adapter,provider_id='q_cpu_fixture')
    assets=tmp_path/'assets';assets.mkdir()
    (assets/'k10.yaml').write_text(yaml.safe_dump(dataclasses.asdict(config_for_probe(provider,10,5))))
    job={'assets':str(assets),'model':'pi05','domain':'metaworld'}
    run=tmp_path/'run';run.mkdir();(run/'job.json').write_text(json.dumps(job))
    monkeypatch.setattr(model_runtime,'load',lambda job:(adapter,provider))
    def observations(job):
        result=[]
        for i in range(3):
            obs=observation();obs['observation/state']+=np.array([.01,-.02,.03,.04],np.float32)*i
            result.append(obs)
        return result
    monkeypatch.setattr(model_runtime,'wire_observations',observations)
    monkeypatch.setattr(sys,'argv',['g3','--job',str(run/'job.json'),'--run',str(run)])
    assert g3.main()==2
    receipt=json.loads((run/'G3_RECEIPT.json').read_text())
    assert receipt['status']=='PARTIAL' and receipt['subset_status']=='PASS' and not receipt['complete_G3']
    assert receipt['audit']['accepted']==3 and receipt['audit']['decisions']==9


def test_pi_native_actual_composition_on_cpu_fixture(tmp_path,monkeypatch):
    from exp.r17_integration.tests.models.pi05.test_pi05 import fixture_adapter,observation
    from exp.r17_integration.models.pi05.provider import register_provider
    from exp.r17_integration.qualification import model_runtime,pi_native
    from exp.r17_integration.models.pi05 import parity
    adapter,_=fixture_adapter();provider=register_provider(adapter,provider_id='q_pi_native_fixture')
    (tmp_path/'job.json').write_text(json.dumps(dict(assets=str(tmp_path))))
    # CPU fixture admission only; the production entry rejects an unsealed
    # archive/model and P2's profile tests exercise those fail-closed gates.
    monkeypatch.setattr(parity,'compiled_fixture_limits',lambda *args:{k:dict(
        max_abs_diff=0,mean_abs_diff=0,rtol=0,scope=parity.FIXTURE_SCOPE) for k in (2,1)})
    monkeypatch.setattr(model_runtime,'load',lambda job:(adapter,provider))
    monkeypatch.setattr(model_runtime,'wire_observations',lambda job:[observation()])
    monkeypatch.setattr(sys,'argv',['pi_native','--job',str(tmp_path/'job.json'),'--run',str(tmp_path)])
    pi_native.main()
    r=json.loads((tmp_path/'native.json').read_text());assert r['status']=='PASS'
    assert [c['k'] for c in r['checks']]==[10,2,1]


def test_native_tail_controller_on_cpu_fixture(tmp_path,monkeypatch):
    from exp.r17_integration.tests.models.pi05.test_pi05 import fixture_adapter,observation
    from exp.r17_integration.models.pi05.provider import register_provider
    from exp.r17_integration.qualification import model_runtime,tail
    adapter,_=fixture_adapter();provider=register_provider(adapter,provider_id='q_tail_fixture')
    (tmp_path/'job.json').write_text('{}')
    monkeypatch.setattr(model_runtime,'load',lambda job:(adapter,provider))
    monkeypatch.setattr(model_runtime,'wire_observations',lambda job:[observation()])
    monkeypatch.setattr(sys,'argv',['tail','--job',str(tmp_path/'job.json'),'--run',str(tmp_path)])
    assert tail.main()==2
    r=json.loads((tmp_path/'TAIL_RECEIPT.json').read_text());assert r['subset_status']=='PASS' and r['status']=='PARTIAL'
    assert all(c['actual_nfe']==0 for c in r['ledger'][1:])


def test_q_server_real_websocket_hot_bundles_on_cpu_fixture(tmp_path,monkeypatch):
    import socket
    import threading
    import yaml
    from exp.r17_integration.tests.models.pi05.test_pi05 import fixture_adapter,observation
    from exp.r17_integration.models.pi05.provider import register_provider
    from exp.r17_integration.qualification import model_runtime,driver
    from exp.r17_integration.qualification.assets import synthetic_store
    from exp.r17_integration.common.qualification import config_for_probe
    adapter,_=fixture_adapter();provider=register_provider(adapter,provider_id='q_ws_fixture')
    assets=tmp_path/'assets';assets.mkdir()
    (assets/'model_spec.json').write_text(json.dumps(dataclasses.asdict(adapter.metadata())))
    for k in (10,2,1): (assets/f'k{k}.yaml').write_text(yaml.safe_dump(dataclasses.asdict(config_for_probe(provider,k,5))))
    synthetic_store(adapter.metadata(),provider,assets)
    with socket.socket() as sock: sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    job={'assets':str(assets),'model':'pi05','domain':'metaworld','server_port':port}
    run=tmp_path/'run';run.mkdir()
    monkeypatch.setattr(model_runtime,'load',lambda job,device='cpu':(adapter,provider))
    monkeypatch.setattr(model_runtime,'wire_observations',lambda job:[observation()])
    servers=[];errors=[]
    def execute():
        try: model_runtime.serve(job,run,device='cpu',server_started=servers.append)
        except BaseException as exc: errors.append(exc)
    thread=threading.Thread(target=execute);thread.start()
    try:
        until=time.monotonic()+10
        while not (run/'SERVER_READY.json').exists():
            assert not errors,errors
            assert time.monotonic()<until
            time.sleep(.05)
        ready=json.loads((run/'SERVER_READY.json').read_text())
        assert ready['r17_samplers']['2']['num_steps']==2
        assert ready['r17_samplers']['10']['num_steps']==10
        driver.websocket_proof(job,run)
        assert json.loads((run/'WS_RECEIPT.json').read_text())['status']=='PASS'
    finally:
        if servers: servers[0].request_stop()
        thread.join(timeout=10)
    assert not thread.is_alive() and not errors,errors
    assert json.loads((run/'SERVER_CLOSED.json').read_text())['status']=='PASS'
