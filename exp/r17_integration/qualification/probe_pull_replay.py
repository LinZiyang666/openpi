"""Replay Q's real deployed MW probes without a renderer, model or GPU."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import threading

from .assets import write_json


def replay(probe, output):
    from .driver import ProbeRecordingDriver
    from exp.r17_integration.conductor.manifest import validate_worker_probe
    output=Path(output)
    def driver(name):
        d=ProbeRecordingDriver.__new__(ProbeRecordingDriver)
        d._census_lock=threading.RLock();d._census={}
        d._worker_probe_validator=lambda p:validate_worker_probe(p,require_renderer=False,require_task_data=False)
        d.rejected_worker_probes=[];d.probe_rejections=[]
        d.rejection_path=output/(name+'_rejected_worker_probes.json')
        return d
    d=driver('accepted');peer=('127.0.0.1',1234)
    base=dict(worker_id='cpu-worker',server_host='Q:1')
    rows=[]
    for name,payload in [('first',dict(base,probe=probe)),('second',base),('terminal',base)]:
        assert d._record_pull(payload,peer)=='cpu-worker'
        rows.append(dict(pull=name,probe_present='probe' in payload,status='PASS'))
    assert d._census['cpu-worker']['pulls']==3 and d._census['cpu-worker']['probe']==probe
    assert not d.rejected_worker_probes and not d.probe_rejections
    negatives=[]
    for case in ('first_missing','new_peer','new_worker','empty','null','modified','endpoint'):
        n=driver(case)
        if case!='first_missing':n._record_pull(dict(base,probe=probe),peer)
        before=copy.deepcopy(n._census);payload=dict(base);actual_peer=peer
        if case=='new_peer':actual_peer=('127.0.0.1',4321)
        if case=='new_worker':payload['worker_id']='unknown'
        if case=='empty':payload['probe']={}
        if case=='null':payload['probe']=None
        if case=='modified':payload['probe']=dict(probe,extra_attestation='changed')
        if case=='endpoint':payload['server_host']='different:2'
        try:n._record_pull(payload,actual_peer)
        except ValueError as exc:
            assert n._census==before
            rejected=json.loads(n.rejection_path.read_text())[-1]
            assert rejected['pull_payload']==payload and rejected['peer']==list(actual_peer)
            negatives.append(dict(case=case,status='PASS',expected_rejection=str(exc)))
        else:pass
    return dict(status='PASS',cpu_only=True,gpu_executed=False,renderer_checked=False,
        task_data_checked=False,results=rows,strict_rejections=negatives,census=copy.deepcopy(d._census))


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    raw=json.loads(Path(a.probe).read_text());out=Path(a.output)
    report=replay(raw.get('probe',raw),out.parent/(out.stem+'_evidence'))
    write_json(out,report);print(json.dumps(report))


if __name__=='__main__':main()
