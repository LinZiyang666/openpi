"""Pure-inference formal driver: asynchronous evidence, durable accepted results.

Planning, probes, task/seed metadata and stock TCP framing come from the frozen
runtime. Only orchestration differs. Fixed bundles must already be acknowledged.
This is a fresh segment driver: crash recovery uses --resume-from into a NEW
campaign with the source fleet closed/isolated, never an in-place driver restart.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import argparse
import fcntl
import json
from pathlib import Path
import signal
import time

from openpi.conductor.driver import ConductorDriver, is_retriable_error
from openpi.conductor.protocol import MSG_SHUTDOWN, result_from_wire
from openpi.conductor.scheduler import StageState
from exp.r17_integration.conductor.queue import CompatibleEpisodeQueue
from exp.r17_integration.qualification.driver import ProbeRecordingDriver
from .formal_driver import plan, probe_validator
from .formal_process import write_json
from .group_commit import GroupCommit


class AsyncQueue(CompatibleEpisodeQueue):
    def __init__(self,*args,evidence,**kwargs):
        self.evidence=evidence
        self.committing={}
        super().__init__(*args,**kwargs)

    def next_task(self,server_key=None):
        # Same selection/fencing as the qualified queue; diagnostics enqueue
        # in memory. No filesystem operation occurs in this critical section.
        with self._lock:
            if server_key is not None and server_key not in self.endpoints:return None
            self._refresh_activation();busy=self.busy()
            for sid in sorted(self._rt):
                rt=self._rt[sid]
                if rt.state!=StageState.READY:continue
                for uid in rt.pending:
                    ep=self._ep_index[uid][1]
                    compatible=[e for e in self.endpoints.values() if e.accepts(ep)]
                    if not compatible:continue
                    target=min(compatible,key=lambda e:(busy[e.endpoint.key]/e.capacity,e.endpoint.key)).endpoint
                    gen=self._dispatch_gen.get(uid,0)+1
                    task=replace(ep,server_host=target.host,server_port=target.port,attempt=gen)
                    row=dict(schema='r17.dispatch.v1',task_uid=uid,attempt=gen,endpoint=target.key,
                             benchmark_protocol=ep.extra['benchmark_protocol'])
                    self.evidence.submit({'dispatch.jsonl':[row]})
                    rt.pending.remove(uid);rt.dispatched.add(uid)
                    self._dispatch_gen[uid]=gen;self._dispatch_ts[uid]=time.monotonic()
                    self._assigned_endpoint[uid]=target.key
                    return task
        return None

    def mark_result(self,uid,**kwargs):
        # Timeout/disconnect cannot steal a generation whose evidence is being
        # committed, even after its worker has moved to another episode.
        with self._lock:
            if uid in self.committing:return False
            return super().mark_result(uid,**kwargs)

    def reserve(self,result):
        with self._lock:
            entry=self._ep_index.get(result.task_uid)
            accepted=(entry is not None and result.task_uid in self._rt[entry[0]].dispatched and
                      result.attempt==self._dispatch_gen.get(result.task_uid) and result.task_uid not in self.committing)
            if accepted:self.committing[result.task_uid]=result.attempt
            if (accepted and result.error and self._ep_attempts.get(result.task_uid,0)>=self._max_ep_retries):
                result=replace(result,error='FatalEpisodeError: retry budget exhausted; '+result.error)
            return result,bool(accepted)

    def committed(self,result):
        with self._lock:
            if self.committing.get(result.task_uid)!=result.attempt:
                raise RuntimeError('durable result lost its dispatch reservation')
            self.committing.pop(result.task_uid)
            accepted=super().mark_result(result.task_uid,success=result.success,
                retriable=is_retriable_error(result.error) if not result.success else False,attempt=result.attempt)
            if not accepted:raise RuntimeError('durable result rejected by scheduler')


class AsyncCampaignDriver(ProbeRecordingDriver):
    def __init__(self,strategy,*,evidence_directory,live_fleets=False,**kwargs):
        directory=Path(evidence_directory)
        if any((directory/name).exists() and (directory/name).stat().st_size for name in
               ('journal.jsonl','dispatch.jsonl','attempts.jsonl','per_step.jsonl')):
            raise ValueError('grouped driver refuses in-place restart; use --resume-from into a fresh closed-source segment')
        super().__init__(strategy,**kwargs)
        self.evidence=(GroupCommit(directory,extra_streams=('fleet_dispatch.jsonl',)) if live_fleets
                       else GroupCommit(directory))
        self._scheduler=AsyncQueue(self._graph,endpoints=strategy.endpoints,
            dispatch_path=directory/'dispatch.jsonl',evidence=self.evidence,**(kwargs.get('scheduler_kwargs') or {}))
        if hasattr(strategy,'bind_driver'):strategy.bind_driver(self)

    def handle_pull(self,key):
        # The launcher validated every resident and broadcast this single
        # fixed bundle. Workers still select it normally on each episode.
        kind,payload=ConductorDriver.handle_pull(self,key)
        if kind==MSG_SHUTDOWN:self._shutdown_local.sent_shutdown=True
        task=payload.get('task')
        if task:
            self.evidence.submit({'worker_dispatch.jsonl':[dict(
                worker_id=getattr(self._shutdown_local,'worker_id',None),task_uid=task['task_uid'],
                attempt=task.get('attempt',1),time=time.time())]})
        return kind,payload

    def handle_result(self,payload):
        result,accepted=self._scheduler.reserve(result_from_wire(payload))
        yaml_id,phase=self._uid_meta(result.task_uid)
        rows=[]
        for original in result.per_step_rows:
            row=dict(original)
            timing=row.get('server_timing',{})
            if timing.get('clock_unit')=='segment_decision':
                row.update({k:timing[k] for k in ('decision_index','delivery_index','env_step_index','segment_head','clock_unit','decision_seed','wire_explicit_seed') if k in timing})
            row.setdefault('success',result.success);row.setdefault('task_uid',result.task_uid)
            row.setdefault('attempt',result.attempt)
            row.update(accepted=accepted,yaml_id=yaml_id,run_id=self._run_id)
            rows.append(row)
        attempt=dict({k:v for k,v in payload.items() if k!='per_step_rows'},
            worker_id=getattr(self._shutdown_local,'worker_id',None),accepted=accepted,time=time.time())
        evidence={'per_step.jsonl':rows,'attempts.jsonl':[attempt]}
        terminal=result.success or not is_retriable_error(result.error)
        if terminal:
            record=dict(task_uid=result.task_uid,yaml_id=yaml_id,phase=phase,
                status='done' if result.success else 'failed',success=result.success,ts=time.time(),
                attempt=result.attempt,accepted=accepted,run_id=self._run_id)
            if result.error is not None:record['error']=result.error
            if result.duration_s is not None:record['duration_s']=round(float(result.duration_s),3)
            evidence['journal.jsonl']=[record]
        def committed():
            self._scheduler.committed(result)
            # The central scheduler's step test counts durably accepted terminal episodes.
            if getattr(self,'_autoscaler',None) is not None:self._autoscaler.result(payload,terminal)
            if self._monitor is not None:self._monitor.on_result(result)
        self.evidence.submit(evidence,committed if accepted else None)

    def close(self):
        try:return self.evidence.close()
        finally:super().close()


def run(campaign,directory):
    if campaign.get('cache_study'):
        from .cache_study import run_driver
        return run_driver(campaign,directory)
    if campaign.get('library') or campaign.get('native_parity_only'):
        raise ValueError('grouped orchestration is pure-inference only')
    from exp.r17_integration.conductor.strategy import MetaWorldStrategy
    from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    ack=json.loads((directory.parent/'BUNDLE_ACK.json').read_text())
    if ack.get('bundle_id')!='formal_full_v1' or not ack.get('bundle_identity'):
        raise ValueError('fixed bundle was not acknowledged before dispatch')
    benchmark,model,tasks,endpoints=plan(campaign)
    strategy=MetaWorldStrategy(tasks,endpoints=endpoints,
        bundles={'formal_full_v1':Path(campaign['local_assets'])/'full.yaml'})
    lock=(directory/'.eval.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    driver=AsyncCampaignDriver(strategy,evidence_directory=directory,live_fleets=campaign.get('live_fleets',False),dispatch_path=directory/'dispatch.jsonl',
        rejection_path=directory/'rejected_worker_probes.json',expected_workers=campaign['workers'],shutdown_seconds=60,
        yaml_weights={t.yaml_id:1 for t in tasks},servers=[endpoints[0].endpoint],journal_path=str(directory/'journal.jsonl'),
        ctl_factory=lambda ep:WebsocketClientPolicy(ep.host,ep.port),bind_host='0.0.0.0',bind_port=campaign['driver_port'],
        worker_probe_validator=probe_validator(benchmark,campaign),episode_timeout_s=campaign['episode_seconds'],
        scheduler_kwargs=dict(eval_concurrency=campaign['workers'],max_episode_retries=campaign['retries']))
    if campaign.get('live_fleets'):
        from .live_fleet_driver import configure
        configure(driver,campaign,directory)
    stopped=False
    def stop(*unused):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    write_json(directory/'manifest.json',dict(campaign,benchmark=benchmark.manifest(),model=model,
        tasks=[asdict(t) for t in tasks],expected_episodes=len(tasks)))
    started=time.monotonic();error=None;commit=None
    try:
        driver.run(stop=lambda:stopped or bool(driver.probe_rejections) or driver.evidence.failure is not None or
                   time.monotonic()-started>campaign['seconds'])
    except Exception as exc:error=repr(exc)
    finally:
        try:commit=driver.close()
        except Exception as exc:error=repr(exc)
        write_json(directory/'census.json',driver.worker_census)
        write_json(directory/'rejected_worker_probes.json',driver.probe_rejections)
        report=dict(status='PASS' if not error and driver.scheduler.all_done() and not driver.probe_rejections and
                    driver.worker_shutdown['status']=='PASS' else 'PARTIAL',all_done=driver.scheduler.all_done(),
                    elapsed_seconds=time.monotonic()-started,worker_shutdown=driver.worker_shutdown,
                    rejected_worker_probes=len(driver.probe_rejections),error=error,group_commit=commit,
                    orchestration=campaign.get('orchestration'))
        write_json(directory/'DRIVER_RECEIPT.json',report);lock.close()
    raise SystemExit(0 if report['status']=='PASS' else 2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--directory',required=True)
    a=p.parse_args();run(json.loads(Path(a.campaign).read_text()),a.directory)
