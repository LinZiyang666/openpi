"""CPU regressions for probe receipts and sequential-slot port cleanup."""
import json
import socket
import threading
import time
from pathlib import Path

import pytest


def listener():
    sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind(('127.0.0.1',0));sock.listen(1)
    return sock,sock.getsockname()[1]


def test_ports_wait_for_owned_listener_release_and_write_receipt(tmp_path):
    from .run_slot import wait_for_port_release
    sock,port=listener()
    release=threading.Thread(target=lambda:(time.sleep(.15),sock.close()))
    release.start();report=dict(gpu_executed=False)
    try:
        result=wait_for_port_release([port],seconds=1,poll_seconds=.01,
            report=report,receipt_path=tmp_path/'receipt.json')
        assert result['initial_busy_ports']==[port] and result['status']=='PASS'
        assert .1<result['waited_seconds']<1
        receipt=json.loads((tmp_path/'receipt.json').read_text())
        assert receipt['gpu_executed'] is False and receipt['port_wait']['busy_ports']==[]
    finally:release.join();sock.close()


def test_busy_port_timeout_preserves_listener_and_hard_deadline(tmp_path):
    from .run_slot import wait_for_port_release,Partial
    sock,port=listener();report=dict(gpu_executed=False)
    started=time.monotonic()
    try:
        with pytest.raises(Partial,match='bounded wait'):
            wait_for_port_release([port],seconds=60,deadline=started+.12,poll_seconds=.01,
                report=report,receipt_path=tmp_path/'receipt.json')
        assert .08<time.monotonic()-started<1
        with socket.create_connection(('127.0.0.1',port),timeout=.1):pass
        assert sock.fileno()!=-1
        receipt=json.loads((tmp_path/'receipt.json').read_text())
        assert receipt['port_wait']['status']=='PARTIAL' and receipt['gpu_executed'] is False
    finally:sock.close()


def test_closed_socket_time_wait_does_not_block_new_listener():
    from .run_slot import wait_for_port_release
    sock,port=listener()
    client=socket.create_connection(('127.0.0.1',port));connection,_=sock.accept()
    connection.close();client.close();sock.close()
    result=wait_for_port_release([port],seconds=.3,poll_seconds=.01)
    assert result['status']=='PASS' and result['waited_seconds']<.15


def test_port_wait_refuses_eval_or_i018_lanes():
    from .run_slot import q_ports
    assert q_ports(dict(host='local',server_port=23156,driver_port=23157))==(23156,23157)
    assert q_ports(dict(host='h100',server_port=23256,driver_port=23257,backend_ports=[23260,23261]))==(23256,23257,23260,23261)
    for host,port in [('local',23151),('local',23180),('h100',23254),('h100',23255),('h100',23258),('h100',23259)]:
        with pytest.raises(ValueError,match='reserved Q lane'):
            q_ports(dict(host=host,server_port=port,driver_port=23157 if host=='local' else 23257))


@pytest.mark.parametrize('present',[False,True])
def test_rejected_probe_receipt_preserves_missing_or_changed_wire_payload(tmp_path,present):
    from .driver import ProbeRecordingDriver
    from exp.r17_integration.benchmarks.metaworld import MetaWorldBenchmarkAdapter
    from exp.r17_integration.conductor.manifest import validate_worker_probe
    accepted=MetaWorldBenchmarkAdapter().worker_probe()
    accepted['packages'].update(metaworld='3.0.0',mujoco='3.14.0',gymnasium='1.3.0')
    driver=ProbeRecordingDriver.__new__(ProbeRecordingDriver)
    driver._census_lock=threading.RLock()
    driver._census={'worker':dict(probe=accepted,peers=['127.0.0.1:1234'],server_key='Q:1')} if present else {}
    driver._worker_probe_validator=lambda p:validate_worker_probe(p,require_renderer=False,require_task_data=False)
    driver.rejected_worker_probes=[];driver.probe_rejections=[];driver.rejection_path=tmp_path/'rejected_worker_probes.json'
    payload=dict(worker_id='worker',server_host='Q:1')
    if present:payload['probe']=dict(accepted,packages=dict(accepted['packages'],mujoco='wrong'))
    with pytest.raises(ValueError) as rejection:
        driver._record_pull(payload,('127.0.0.1',1234))
    row=json.loads(driver.rejection_path.read_text())[0]
    assert row['probe_present'] is present and row['probe']==payload.get('probe')
    assert row['accepted_probe']==(accepted if present else None) and row['pull_payload']==payload
    assert row['peer']==['127.0.0.1',1234]
    assert row['worker_id']=='worker' and row['reason']==str(rejection.value) and row['reason']
    if present:assert driver._census['worker']['probe']==accepted


@pytest.mark.parametrize('mutation',[None,'reference','alternate'])
def test_pi_native_consumes_exact_r16_contract_and_scoped_alternate(tmp_path,monkeypatch,mutation):
    """Exercise Q's entry point: a compiled delta cannot hide native defects."""
    import sys
    import numpy as np
    from . import pi_native,model_runtime
    from exp.r17_integration.common import qualification
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC,decision_seed
    from exp.r17_integration.models.pi05 import parity
    from exp.r17_integration.tests.models.pi05.test_pi05 import fixture_adapter,observation
    adapter,_=fixture_adapter();obs=observation()
    req=RequestContext('Q-pi-native',FORMAL_SPEC.identity,'reach-v3',0,0,arm='qualification',
        attempt=1,bundle_generation='native-v1',task_id=43,explicit_seed=decision_seed('reach-v3',0,0))
    reference=parity.r16_reference
    def infer(observation,noise):
        k=adapter.policy._sample_kwargs['num_steps']
        _,actions=reference(adapter,observation,req,k)
        if k!=10:actions[0,0]+=.01 if mutation!='alternate' else 1
        return dict(actions=actions)
    def native_reference(*args):
        normalized,actions=reference(*args)
        if mutation=='reference':actions[0,0]=np.nextafter(actions[0,0],np.inf)
        return normalized,actions
    archive=tmp_path/'wire_observations.npz';archive.write_bytes(b'CPU fixture')
    def limits(spec,path):
        assert spec==adapter.metadata() and path==archive
        return {k:dict(max_abs_diff=maximum,mean_abs_diff=mean,rtol=0,scope=parity.FIXTURE_SCOPE)
            for k,maximum,mean in [(2,.08,.02),(1,.13,.03)]}
    monkeypatch.setattr(adapter.policy,'infer',infer)
    monkeypatch.setattr(model_runtime,'load',lambda job:(adapter,object()))
    monkeypatch.setattr(model_runtime,'wire_observations',lambda job:[obs])
    monkeypatch.setattr(parity,'r16_reference',native_reference)
    monkeypatch.setattr(parity,'compiled_fixture_limits',limits)
    calls=[]
    monkeypatch.setattr(qualification,'run_probe',lambda *args,**kwargs:(calls.append(kwargs) or dict(status='PASS',synthetic_cpu_fixture=True)))
    job=tmp_path/'job.json';job.write_text(json.dumps(dict(assets=str(tmp_path))))
    monkeypatch.setattr(sys,'argv',['pi_native','--job',str(job),'--run',str(tmp_path)])
    if mutation is None:
        pi_native.main()
    else:
        with pytest.raises(AssertionError,match='exact R16/debug' if mutation=='reference' else 'exceeded bound'):
            pi_native.main()
    receipt=json.loads((tmp_path/'native.json').read_text())
    if mutation is not None:
        assert receipt['status']=='FAIL' and not calls
        return
    assert receipt['status']=='PASS' and calls[0]['steps']==(10,2,1)
    assert [row['k'] for row in receipt['checks']]==[10,2,1]
    for row in receipt['checks']:
        assert row['contract']==parity.CONTRACT_VERSION and row['baseline_path']==parity.REFERENCE_PATH
        assert row['exact_r16'] and row['exact_debug'] and 'exact_upstream' not in row
        assert row['alternate_path']['exact'] is (row['k']==10)
        assert row['suffixes']==row['k']-1
