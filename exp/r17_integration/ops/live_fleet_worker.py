"""Client-only live-fleet authorization and cooperative supervisor drain.

No imports of models, simulators or the coordinator control registry.
"""
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager
import fcntl

NATIVE_FIELDS = ('model','domain','checkpoint','server_node','replicas','endpoint',
                 'driver_port','runtime_identity',
                 'dp_batching','fastwam_batching','mps','cache_study','cache_control',
                 'commit_horizon','denoising_steps','local_assets')


def native_pin(campaign):
    fields = {k:campaign.get(k) for k in NATIVE_FIELDS}
    return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def check_payload_protocol(archive):
    from .build_client_bundle import verified_members
    manifest,content=verified_members(archive,'PAYLOAD.json')
    required={
        'formal_workers':b'LIVE_FLEET_PROTOCOL = 1',
        'timan1_workers':b"campaign.get('live_fleet_authorization')",
        'local_workers':b"campaign.get('live_fleet_authorization')",
        'live_fleet_worker':b"r17.live_fleet_authorization.v1",
    }
    for name,marker in required.items():
        path='exp/r17_integration/ops/'+name+'.py'
        if marker not in content.get(path,b''):
            raise ValueError('live-fleet worker protocol missing from pinned payload: '+path)
    return {name:manifest['files'][name] for name in manifest['files']
            if name in {'exp/r17_integration/ops/'+n+'.py' for n in required}}


def authorization(campaign):
    record = campaign.get('live_fleet_authorization')
    if record is None:
        return None
    nodes = campaign.get('execution_worker_nodes',campaign.get('worker_nodes',{}))
    if (record.get('schema') != 'r17.live_fleet_authorization.v1' or
            record.get('owner_authorized') is not True or not campaign.get('live_fleets') or
            record.get('worker_nodes') != nodes or record.get('native_pin') != native_pin(campaign) or
            not nodes or any(n not in ('local','timan1','timan107','timan108') or
                             type(v) is not int or not 1 <= v <= 128 for n,v in nodes.items()) or
            'timan1' in nodes and (campaign.get('domain') != 'libero' or nodes['timan1'] > 40)):
        pass
    for key in ('campaign_sha256','worker_archive_sha256'):
        value = record.get(key,'')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('live fleet source pin missing')
    return record


def draining(campaign):
    if not campaign.get('live_fleets'):
        return False
    path = Path(campaign['worker_run']+'.drain.json')
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    if value != dict(schema='r17.fleet_drain.v1',run_name=campaign['run_name'],
                     worker_node=campaign['worker_node']):
        pass
    return True


@contextmanager
def lifecycle(campaign):
    if not campaign.get('live_fleets'):
        yield
        return
    path=Path(campaign['worker_run']+'.drain.lock')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def write_drain(campaign):
    from .formal_process import write_json
    path = Path(campaign['worker_run']+'.drain.json')
    # A starting supervisor may not have created its run yet. A confined sibling
    # marker lets it observe drain before mkdir/initial spawn as well.
    with lifecycle(campaign):
        write_json(path,dict(schema='r17.fleet_drain.v1',run_name=campaign['run_name'],
                             worker_node=campaign['worker_node']))
    return dict(path=str(path),drain_requested=True)
