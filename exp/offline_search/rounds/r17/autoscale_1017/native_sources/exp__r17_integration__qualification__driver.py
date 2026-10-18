"""Bounded Q plan/driver; reuses P2 strategy, queue and stock transport."""
from __future__ import annotations
from dataclasses import asdict
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import threading
import time

from .assets import write_json
from exp.r17_integration.conductor.driver import MetaWorldConductorDriver


class ProbeRecordingDriver(MetaWorldConductorDriver):
    """Preserve rejected wire contents without changing P2's admission policy."""
    def __init__(self,*args,rejection_path,expected_workers=2,shutdown_seconds=35,**kwargs):
        self.probe_rejections=[]
        self.rejection_path=Path(rejection_path)
        self.expected_workers=expected_workers
        self.shutdown_seconds=shutdown_seconds
        self._shutdown_lock=threading.RLock()
        self._shutdown_workers=set()
        self._shutdown_connections=set()
        self._shutdown_local=threading.local()
        self._shutdown_end=threading.Event()
        self.worker_shutdown=dict(status='PARTIAL',workers=[])
        super().__init__(*args,**kwargs)

    def _record_pull(self,payload,peer):
        # The transport peer is owned by the stock connection handler. Keep
        # per-connection identity for shutdown evidence, not admission bypass.
        local=getattr(self,'_shutdown_local',None)
        if local is not None:local.worker_id=payload.get('worker_id')
        try:
            return super()._record_pull(payload,peer)
        except ValueError as exc:
            # The accepted census probe may differ from (or be absent in) this
            # pull. Keep both, especially when a subsequent pull omits probe.
            import copy
            with self._census_lock:
                self.probe_rejections.append(dict(worker_id=payload.get('worker_id'),
                    server_key=payload.get('server_host'),reason=str(exc),time=time.time(),
                    probe_present='probe' in payload,probe=copy.deepcopy(payload.get('probe')),
                    accepted_probe=copy.deepcopy(self._census.get(payload.get('worker_id'),{}).get('probe')),
                    pull_payload=copy.deepcopy(payload),peer=list(peer) if isinstance(peer,tuple) else peer))
                write_json(self.rejection_path,self.probe_rejections)
            raise

    def handle_pull(self,server_key):
        kind,payload=super().handle_pull(server_key)
        from openpi.conductor.protocol import MSG_SHUTDOWN
        if kind==MSG_SHUTDOWN:self._shutdown_local.sent_shutdown=True
        return kind,payload

    def _handle_conn(self,conn,stop):
        # Stock run() sets its done flag immediately on scheduler completion.
        # Keep only our accepted sockets alive through the bounded drain so
        # every stock WorkerLoop can make its final pull and close normally.
        with self._shutdown_lock:self._shutdown_connections.add(conn)
        self._shutdown_local.sent_shutdown=False
        try:
            super()._handle_conn(conn,lambda:self._shutdown_end.is_set() or self._shutdown_external_stop())
        finally:
            with self._shutdown_lock:
                if getattr(self._shutdown_local,'sent_shutdown',False):
                    self._shutdown_workers.add(getattr(self._shutdown_local,'worker_id',None))
                self._shutdown_connections.discard(conn)

    def run(self,stop=None,*,poll_s=.05):
        self._shutdown_external_stop=stop or (lambda:False)
        self._shutdown_end.clear();started=time.monotonic()
        try:
            super().run(stop=self._shutdown_external_stop,poll_s=poll_s)
            until=time.monotonic()+self.shutdown_seconds
            while self.scheduler.all_done() and not self._shutdown_external_stop() and time.monotonic()<until:
                with self._shutdown_lock:
                    workers=set(self._shutdown_workers)
                    active=len(self._shutdown_connections)
                if len(workers)==self.expected_workers and not active:
                    self.worker_shutdown=dict(status='PASS',workers=sorted(workers),active_connections=0)
                    break
                time.sleep(.025)
        finally:
            self._shutdown_end.set()
            with self._shutdown_lock:
                self.worker_shutdown.update(workers=sorted(w for w in self._shutdown_workers if w is not None),
                    elapsed_seconds=time.monotonic()-started)
                connections=list(self._shutdown_connections)
            # A failed drain closes only sockets accepted by this Q driver.
            # Never classify a missing shutdown/unfinished worker as PASS.
            import socket
            for conn in connections:
                try:conn.shutdown(socket.SHUT_RDWR)
                except OSError:pass
                conn.close()


def make_plan(job, run, *, live=False):
    from exp.r17_integration.common.contracts import digest
    from exp.r17_integration.conductor.manifest import EndpointSpec, episode_tasks
    from openpi.conductor.task import ServerEndpoint
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    model = json.loads((Path(job['assets'])/'model_spec.json').read_text())
    endpoint = ServerEndpoint(job['endpoint_host'],job['server_port'])
    ks = [model['default_sampler']['num_steps'], model['default_sampler']['num_steps'] if job['model']=='groot' else 2]
    paths = [run/'full.yaml',run/'retrieval.yaml'] if live else [Path(job['assets'])/f'k{k}.yaml' for k in ks]
    tasks = []
    for arm, k, idx, path in zip(('full','retrieval'),ks,(0,1),paths):
        tasks.extend(episode_tasks(campaign='Q-'+run.name, arm=arm,bundle_id=arm+'_gen1',endpoint=endpoint,
            model_identity=model['checkpoint_identity'],norm_identity=model['norm_identity'],
            layout_identity=digest(model['layout']),num_steps=k,partition='smoke',names=['reach-v3'],indices=[idx],
            yaml_identity=hashlib.sha256(path.read_bytes()).hexdigest(),bundle_generation=arm+'_gen1'))
    endpoints=[EndpointSpec(endpoint, model['checkpoint_identity'],model['norm_identity'],
        digest(model['layout']),FORMAL_SPEC.identity,allowed_k=tuple(set(ks)),capacity=2)]
    assert all(endpoints[0].accepts(task) for task in tasks)
    write_json(run/'episode_manifest.json',dict(schema='r17.episode_plan.v1',tasks=[asdict(t) for t in tasks],
        expected_episodes=2,quality_claim=False,endpoint=asdict(endpoint),model=model))
    return tasks,endpoints,dict(zip(('full_gen1','retrieval_gen1'),paths))


def websocket_proof(job, run):
    """Two actual clients, hot bundles, invalid selection, live debug byte parity."""
    import numpy as np
    import yaml
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
    from .model_runtime import wire_observations
    obs=wire_observations(job)[0]
    spec=json.loads((Path(job['assets'])/'model_spec.json').read_text())
    protocol=spec['allowed_benchmark_protocols'][0]
    clients=[WebsocketClientPolicy('127.0.0.1',job['server_port']) for _ in range(2)]
    try:
        ctl=clients[0]
        for bundle,path in [('full_gen1',run/'full.yaml'),('retrieval_gen1',run/'retrieval.yaml'),('plain_full_gen1',run/'full.yaml')]:
            ctl.load_cache_config(str(path),yaml_id=bundle,bundle_id=bundle)
        req=RequestContext('Q-WS-fixed',protocol,'reach-v3' if job['domain']=='metaworld' else job['mechanics_task_name'],
            0,0,task_id=43 if job['domain']=='metaworld' else 0,attempt=1,arm='proof',bundle_generation='proof',explicit_seed=18)
        replies=[]
        for i,(client,bundle) in enumerate(zip(clients,('plain_full_gen1','full_gen1'))):
            client.select_bundle(bundle)
            client.episode_start('Q-WS-fixed',req.task_name,0,extra_metadata=dict(task_uid='Q-WS-proof-'+str(i),
                task_data_hash='synthetic-fixed-observation',task_name=req.task_name,campaign=req.campaign,arm=req.arm,
                original_init_idx=0,attempt=1,bundle_generation=req.bundle_generation))
            replies.append(client.infer(dict(obs,__request__=asdict(req))))
        assert replies[0]['actions'].dtype==replies[1]['actions'].dtype
        assert replies[0]['actions'].tobytes()==replies[1]['actions'].tobytes()
        fullk=spec['default_sampler']['num_steps']
        assert tuple(replies[0]['__hit_meta__']['stage_calls'])==(1,1,fullk)
        for client in clients:
            client.episode_end(False,execution=dict(decision_seq=0,executed_count=0,terminal=True))
        ctl.select_bundle('retrieval_gen1')
        ctl.episode_start('Q-WS-fixed',req.task_name,1,extra_metadata=dict(task_uid='Q-WS-hit',
            task_data_hash='synthetic-fixed-observation',task_name=req.task_name,campaign=req.campaign,arm=req.arm,
            original_init_idx=0,attempt=1,bundle_generation=req.bundle_generation))
        hit=ctl.infer(dict(obs,__request__=asdict(req)))
        assert tuple(hit['__hit_meta__']['stage_calls'])==(1,0,0)
        assert hit['__hit_meta__']['source']=='library'
        assert np.isfinite(hit['actions']).all()
        ctl.episode_end(False,execution=dict(decision_seq=0,executed_count=0,terminal=True))
        bad=yaml.safe_load((run/'full.yaml').read_text())
        bad['adapter_binding']['norm_identity']='wrong'
        rejected=0
        for call in [lambda:ctl.select_bundle('Q-unknown'),lambda:ctl.load_cache_config(yaml_content=yaml.safe_dump(bad),yaml_id='bad',bundle_id='bad')]:
            try:
                call()
            except RuntimeError:
                rejected+=1
        assert rejected==2
        for client in clients:
            client.select_bundle('full_gen1')
        write_json(run/'WS_RECEIPT.json',dict(status='PASS',actual_clients=2,debug_byte_parity=True,
            full_stage_calls=[1,1,fullk],retrieval_stage_calls=[1,0,0],invalid_rejections=2,
            synthetic_input=job['domain']!='libero',recorded_input=job['domain']=='libero',
            synthetic_library=True,quality_claim=False))
    finally:
        for client in clients:
            client.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--ws-only',action='store_true')
    p.add_argument('--seconds',type=int,default=420)
    a=p.parse_args()
    if not 1 <= a.seconds <= 480:
        p.error('driver bound <=480 seconds')
    job,run=json.loads(Path(a.job).read_text()),Path(a.run)
    run.mkdir(parents=True,exist_ok=True)
    if a.ws_only:
        websocket_proof(job,run)
        return
    tasks,endpoints,bundles=make_plan(job,run,live=not a.plan_only)
    if a.plan_only:
        write_json(run/'PLAN_RECEIPT.json',dict(status='PASS',plan_only=True,expected_episodes=2))
        return
    from exp.r17_integration.conductor.strategy import MetaWorldStrategy
    from exp.r17_integration.conductor.manifest import validate_worker_probe
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    websocket_proof(job,run)
    strategy=MetaWorldStrategy(tasks,endpoints=endpoints,bundles=bundles)
    def write_rows(stage,rows):
        with (run/'per_step.jsonl').open('a') as f:
            for row in rows:
                f.write(json.dumps(row,sort_keys=True)+'\n')
            f.flush()
            __import__('os').fsync(f.fileno())
    ctl_factory=lambda s: WebsocketClientPolicy(s.host,s.port)
    driver=ProbeRecordingDriver(strategy,dispatch_path=run/'dispatch.jsonl',rejection_path=run/'rejected_worker_probes.json',
        yaml_weights={t.yaml_id:1 for t in tasks},servers=[endpoints[0].endpoint],journal_path=str(run/'journal.jsonl'),
        ctl_factory=ctl_factory,bind_host='0.0.0.0',bind_port=job['driver_port'],per_step_writer=write_rows,
        worker_probe_validator=validate_worker_probe,episode_timeout_s=350,
        scheduler_kwargs=dict(eval_concurrency=2,max_episode_retries=1))
    stopped=False
    def stop(*args):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop)
    started=time.monotonic()
    lock=(run/'.eval.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        driver.run(stop=lambda:stopped or bool(driver.probe_rejections) or time.monotonic()-started>a.seconds)
    finally:
        write_json(run/'census.json',driver.worker_census)
        write_json(run/'rejected_worker_probes.json',driver.probe_rejections)
        write_json(run/'DRIVER_RECEIPT.json',dict(status='PASS' if driver.scheduler.all_done() and not driver.probe_rejections and driver.worker_shutdown['status']=='PASS' else 'PARTIAL',
                   rejected_worker_probes=driver.probe_rejections,worker_shutdown=driver.worker_shutdown,
                   elapsed_seconds=time.monotonic()-started))
        driver.close()
        lock.close()
    if not driver.scheduler.all_done() or driver.probe_rejections or driver.worker_shutdown['status']!='PASS':
        raise SystemExit(2)


if __name__=='__main__':
    main()
