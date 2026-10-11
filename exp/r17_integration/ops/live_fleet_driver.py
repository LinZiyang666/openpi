"""Instance-scoped driver hooks; stock transport/task/result fences are unchanged."""
from dataclasses import replace
import json
from pathlib import Path
import time


def configure(driver, campaign, directory):
    if not campaign.get('live_fleets'):
        return None
    from .live_fleets import Registry
    from .formal_process import ticks, write_json
    from openpi.conductor.protocol import MSG_SHUTDOWN
    registry = Registry(Path(directory).parent)
    last_snapshot=0.
    original_pull, original_record = driver.handle_pull, driver._record_pull
    original_run = driver.run
    identity = dict(pid=__import__('os').getpid(),start_ticks=ticks(__import__('os').getpid()))
    write_json(Path(directory)/'LIVE_FLEET_DRIVER.json',dict(schema='r17.live_fleet_driver.v1',
        campaign_sha256=registry.read()['campaign_sha256'],owner=identity))

    def update(document):
        active = [f for f in document['fleets'].values() if f['state']=='ACTIVE']
        total = sum(f['workers'] for f in active)
        driver.scheduler._eval_cc = max(1,total)
        driver.scheduler.endpoints = {key:replace(ep,capacity=max(1,total))
                                      for key,ep in driver.scheduler.endpoints.items()}
        # Retired N4 generations need not make another pull. Only a positive
        # supervisor closure permits removing their fleet from the final census.
        expected = {wid for f in document['fleets'].values() if f['state'] in ('ACTIVE','DRAINING')
                    for wid in f['worker_ids']}
        with driver._shutdown_lock:
            driver.expected_workers = len(expected | driver._shutdown_workers)
        return expected

    def identify(document, worker):
        return next((f for f in document['fleets'].values() if worker in f['worker_ids']),None)

    def record(payload, peer):
        # Reject an unregistered identity before P2 records an accepted census.
        with registry.locked() as document:
            fleet = identify(document,payload.get('worker_id'))
            if fleet is None or fleet['state'] not in ('ACTIVE','DRAINING'):
                raise ValueError('worker has no active registered fleet')
        value=original_record(payload,peer)
        with driver._census_lock:
            entry=driver._census[payload['worker_id']]
            entry.update(fleet_id=fleet['fleet_id'],worker_node=fleet['node'])
        return value

    def pull(key):
        # Registry lock also linearizes detach against assignment: after the
        # DRAINING transition no subsequent pull can receive another episode.
        with registry.locked() as document:
            with driver.scheduler._lock:
                update(document)
                fleet = identify(document,getattr(driver._shutdown_local,'worker_id',None))
                if fleet is None or fleet['state'] != 'ACTIVE':
                    driver._shutdown_local.sent_shutdown = True
                    return MSG_SHUTDOWN, {'reason':'fleet_draining'}
                kind,payload = original_pull(key)
                task = payload.get('task')
                if task:
                    row = dict(fleet_id=fleet['fleet_id'],worker_node=fleet['node'],
                        worker_id=driver._shutdown_local.worker_id,task_uid=task['task_uid'],
                        attempt=task.get('attempt',1),time=time.time())
                    if hasattr(driver,'evidence'):
                        driver.evidence.submit({'fleet_dispatch.jsonl':[row]})
                    else:
                        with (Path(directory)/'fleet_dispatch.jsonl').open('a') as stream:
                            stream.write(json.dumps(row,sort_keys=True)+'\n')
                            stream.flush();__import__('os').fsync(stream.fileno())
                return kind,payload

    def poll():
        nonlocal last_snapshot
        with registry.locked() as document:
            with driver.scheduler._lock:
                update(document)
            if time.monotonic()-last_snapshot>=5:
                write_json(Path(directory)/'LIVE_FLEET_CENSUS.json',dict(schema='r17.live_fleet_census.v1',
                    time=time.time(),worker_census=driver.worker_census,expected_workers=driver.expected_workers,
                    active_workers=sum(f['workers'] for f in document['fleets'].values() if f['state']=='ACTIVE'),
                    capacity=driver.scheduler._eval_cc))
                last_snapshot=time.monotonic()
        return False
    driver._record_pull, driver.handle_pull = record, pull
    def run(stop=None,**kwargs):
        return original_run(stop=lambda:poll() or bool(stop and stop()),**kwargs)
    driver.run=run
    return poll
