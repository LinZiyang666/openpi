"""Opt-in live worker fleets. CLI queues requests; the owning launcher executes.

No driver restart, policy restart, task regeneration, or in-place resume. Frozen
campaigns without the capability refuse requests. Fleet IDs are evidence keys,
not node names. Scientific campaign.json and WORKER_DEPLOYMENT.json stay sealed.
"""
from contextlib import contextmanager
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

from .formal_process import alive, ticks, write_json
from .immutable_release import sha
from .live_fleet_worker import native_pin

VERSION = 'r17.live_fleets.v1'


def component(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',value):
        raise ValueError('fleet id must be a safe component, maximum64 characters')
    return value


class Registry:
    def __init__(self,directory):
        self.directory = Path(directory)
        self.path = self.directory/'LIVE_FLEETS.json'

    def read(self):
        document = json.loads(self.path.read_text())
        if document.get('schema') != VERSION:
            raise ValueError('unsupported live-fleet registry version')
        return document

    @contextmanager
    def locked(self):
        with (self.directory/'.live_fleets.lock').open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            yield self.read()

    def save(self,document,event=None):
        from .sync_back import atomic_bytes
        atomic_bytes(self.path,(json.dumps(document,sort_keys=True,indent=2)+'\n').encode())
        if event:
            with (self.directory/'FLEET_EVENTS.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event,time=time.time()),sort_keys=True)+'\n')
                stream.flush();os.fsync(stream.fileno())

    def initialize(self,campaign,deployment):
        if self.path.exists():
            pass
        digest = sha(self.directory/'campaign.json')
        fleets = {}
        for node,item in deployment.items():
            fleet_id = 'initial_'+node
            fleets[fleet_id] = dict(fleet_id=fleet_id,node=node,workers=item['workers'],
                evidence_key=node,state='ACTIVE',initial=True,worker_ids=[
                    f"{campaign['run_name']}_{node}_{i}" for i in range(item['workers'])],
                deployment=item,worker_manifest_sha256=sha(self.directory/(node+'_launch.json')
                    if (self.directory/(node+'_launch.json')).exists() else self.directory/(node+'.json')))
        owner = dict(pid=os.getpid(),start_ticks=ticks(os.getpid()))
        document = dict(schema=VERSION,state='OPEN',campaign_sha256=digest,
            worker_archive_sha256=sha(self.directory/'workers.tar'),native_pin=native_pin(campaign),
            launcher_owner=owner,fleets=fleets,requests={})
        self.save(document,dict(event='capability_initialized'))
        write_json(self.directory/'LIVE_FLEET_CAPABILITY.json',dict(schema=VERSION,owner=owner,
            campaign_sha256=digest,worker_archive_sha256=document['worker_archive_sha256']))
        return document

    def verify_running(self,document):
        campaign = json.loads((self.directory/'campaign.json').read_text())
        if (not campaign.get('live_fleets') or document['state']!='OPEN' or
                not alive(document['launcher_owner']) or
                sha(self.directory/'campaign.json') != document['campaign_sha256'] or
                sha(self.directory/'workers.tar') != document['worker_archive_sha256'] or
                native_pin(campaign) != document['native_pin'] or
                any((self.directory/n).exists() for n in ('GPU_RELEASED.json','SEALED_EVIDENCE_RECEIPT.json'))):
            pass
        capability = json.loads((self.directory/'driver/LIVE_FLEET_DRIVER.json').read_text())
        if capability['campaign_sha256'] != document['campaign_sha256'] or not alive(capability['owner']):
            pass
        return campaign

    def request(self,action,*,fleet_id,node=None,workers=None,execute=False):
        component(fleet_id)
        if not self.path.exists():
            pass
        with self.locked() as document:
            self.verify_running(document)
            if action=='attach':
                if node not in ('timan107','timan108','local','timan1') or type(workers) is not int or not 1 <= workers <= 128:
                    raise ValueError('allowed node and1..128 workers required')
                if node=='timan1' and workers>40:
                    pass
                if fleet_id in document['fleets'] or any(r['fleet_id']==fleet_id and r['action']=='attach'
                                                        for r in document['requests'].values()):
                    raise ValueError('fleet id already used (including failed attachment)')
            elif action=='detach':
                fleet = document['fleets'].get(fleet_id)
                if fleet is None or fleet['state'] not in ('ACTIVE','DRAINING'):
                    pass
                node,workers = fleet['node'],fleet['workers']
            else:
                raise ValueError('unknown fleet action')
            row = dict(action=action,fleet_id=fleet_id,node=node,workers=workers,status='QUEUED',
                campaign_sha256=document['campaign_sha256'],worker_archive_sha256=document['worker_archive_sha256'])
            if execute:
                key = uuid.uuid4().hex
                document['requests'][key] = row
                self.save(document,dict(event='request_queued',request_id=key,**row))
                row = dict(row,request_id=key)
            return dict(row,submitted=execute)


def child_campaign(campaign,document,row,directory):
    child = copy.deepcopy(campaign)
    name = 'lf_'+campaign['run_name'][:12]+'_'+row['fleet_id'][:10]+'_'+hashlib.sha256(row['fleet_id'].encode()).hexdigest()[:8]
    counts = {row['node']:row['workers']}
    child.update(run_name=name,worker_nodes=counts,workers=row['workers'],
                 execution_worker_nodes=counts,execution_workers=row['workers'],prepare_only=False)
    if row['node']=='timan1':
        # GPU3's existing private worker lease requires6144MiB per slot. A
        # different node's optional RAM policy must not undercut that lease.
        child.pop('worker_ram_reservation_mib',None)
    for key in ('cache_capacity_authorization','cache_size_lane_authorization','cache_worker_orchestration',
                'local_worker_exception','timan1_worker_lease'):
        child.pop(key,None)
    child['live_fleet_authorization'] = dict(schema='r17.live_fleet_authorization.v1',owner_authorized=True,
        campaign_sha256=document['campaign_sha256'],worker_archive_sha256=document['worker_archive_sha256'],
        parent_campaign=str(Path(directory).resolve()),fleet_id=row['fleet_id'],
        worker_nodes=counts,native_pin=native_pin(child))
    if native_pin(child) != document['native_pin']:
        pass
    return child


def prepare_attachment(launch,campaign,directory,document,row):
    """Use the production installer, probes, resource/disk guards and SHA stage."""
    directory = Path(directory)
    if sha(directory/'campaign.json') != document['campaign_sha256'] or sha(directory/'workers.tar') != document['worker_archive_sha256']:
        pass
    target = directory/'fleets'/component(row['fleet_id'])
    target.mkdir(parents=True,exist_ok=False)
    child = child_campaign(campaign,document,row,directory)
    write_json(target/'campaign.json',child)
    deployment = launch.prepare_workers(child,target,payload_override=directory/'workers.tar')
    if sha(target/'workers.tar') != document['worker_archive_sha256']:
        pass
    item = deployment[row['node']]
    item.update(node=row['node'],fleet_directory=str(target),fleet_campaign=child)
    worker = target/(row['node']+'.json')
    # The finalizer uses the main directory's distinct evidence key. This is
    # the exact transferred manifest, not a reconstructed worker configuration.
    key = 'fleet_'+row['fleet_id']
    shutil.copyfile(worker,directory/(key+'_launch.json'))
    return child,item,key,sha(worker)


def drain_attachment(launch,item,node):
    from .remote_json import stage_json, expression
    # File transport, never a large argv. Same /tmp stage/SHA verification as
    # repair and ordinary launcher control messages.
    config = json.loads((Path(item['fleet_directory'])/(node+'.json')).read_text()) if item.get('fleet_directory') else item['worker_campaign']
    directory = Path(item.get('fleet_directory',item['local_directory'])) if 'local_directory' in item else Path(item['fleet_directory'])
    staged = stage_json(node,config,directory,item['stage'],remote=launch.remote,
        transfer=launch.transfer,python=item['python'])
    code = ('from exp.r17_integration.ops.remote_json import load_json; '
        'from exp.r17_integration.ops.live_fleet_worker import write_drain; import json; '
        'print(json.dumps(write_drain('+expression(staged)+')))')
    return json.loads(launch.remote(node,['env','PYTHONPATH='+launch.pythonpath(item['payload']),
        item['python'],'-c',code],replay_safe=True))


def owned_spawn(launch,node,label,argv,env,root,python,stage,*,seconds,lease_seconds,ownership,record):
    """Persist ownership BEFORE remote spawn, including ambiguous tether failure."""
    request=dict(argv=argv,env=env,cwd=root,receipt=stage+'/'+label+'.process.json',
        log=stage+'/'+label+'.log',lease=stage+'/'+label+'.lease',seconds=seconds,lease_seconds=lease_seconds)
    owned=dict(node=node,label=label,request=request,root=root,python=python,receipt=None,
        transport='direct' if node=='local' else 'tether',**ownership)
    record(owned,True)
    owned['receipt']=json.loads(launch.remote(node,['env','PYTHONPATH='+launch.pythonpath(root),python,
        '-m','exp.r17_integration.ops.formal_process','spawn','--request',json.dumps(request)]))
    record(owned,False)
    return owned


class Launcher:
    """Preparation runs in one CPU thread so existing leases remain serviced."""
    def __init__(self,launch,campaign,directory,deployment,spawn):
        if campaign.get('library') or campaign.get('native_parity_only'):
            raise ValueError('live fleets support formal/cache evaluation only')
        self.launch,self.campaign,self.directory = launch,campaign,Path(directory)
        self.deployment,self.spawn = deployment,spawn
        self.registry = Registry(directory)
        self.registry.initialize(campaign,deployment)
        self.pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='fleet_prepare')
        self.pending = None

    def tick(self):
        if self.pending and self.pending[1].done():
            key,future = self.pending;self.pending = None
            with self.registry.locked() as document:
                row = document['requests'][key]
                try:
                    child,item,evidence_key,manifest_sha = future.result()
                    if document['state']!='OPEN':
                        pass
                    self.deployment[evidence_key] = item
                    fleet = dict(fleet_id=row['fleet_id'],node=row['node'],workers=row['workers'],
                        evidence_key=evidence_key,state='ACTIVE',initial=False,deployment=item,
                        worker_ids=[f"{child['run_name']}_{row['node']}_{i}" for i in range(row['workers'])],
                        worker_manifest_sha256=manifest_sha,worker_archive_sha256=document['worker_archive_sha256'],
                        native_pin=document['native_pin'],attached_at=time.time())
                    document['fleets'][row['fleet_id']] = fleet
                    # Registration precedes spawn. Worker pulls serialize against
                    # this lock; spawn failure leaves no admissible worker fleet.
                    self.registry.save(document)
                    fleet['spawn_requested']=True
                    self.registry.save(document)
                    owned = self.spawn(row['node'],'workers__'+evidence_key,
                        [item['python'],'-u','-m','exp.r17_integration.ops.formal_workers','--campaign',
                         item.get('campaign_path',item['stage']+'/campaign.json')],item['env'],item['payload'],
                        item['python'],item['stage'],ownership=dict(kind='workers',fleet_key=evidence_key,
                                                                   fleet_id=row['fleet_id']))
                    fleet['owned'] = owned
                    row['status'] = 'ATTACHED'
                except Exception as exc:
                    row.update(status='FAILED',error=repr(exc),step='prepare_or_spawn',
                        stderr=getattr(exc,'stderr',None) or getattr(exc,'output',None))
                    if row['fleet_id'] in document['fleets']:
                        document['fleets'][row['fleet_id']]['state']='FAILED'
                self.registry.save(document,dict(event='attach_result',request_id=key,**row))
                if document['fleets'].get(row['fleet_id'],{}).get('spawn_requested') and row['status']=='FAILED':
                    raise RuntimeError('attached fleet spawn failed; ownership recorded for cleanup: '+row['error'])
        with self.registry.locked() as document:
            if document['state']!='OPEN':
                return
            for key,row in document['requests'].items():
                if row['status']!='QUEUED':continue
                if row['action']=='attach':
                    if self.pending:continue
                    row['status']='PREPARING'
                    # Preserve the parent's bounded tether retry/call journal
                    # in the CPU preparation thread; ContextVars do not inherit.
                    self.pending=(key,self.pool.submit(copy_context().run,prepare_attachment,self.launch,self.campaign,
                        self.directory,copy.deepcopy(document),copy.deepcopy(row)))
                else:
                    fleet = document['fleets'][row['fleet_id']]
                    fleet['state']='DRAINING'
                    self.registry.save(document,dict(event='dispatch_drain_fenced',fleet_id=row['fleet_id']))
                    item = copy.deepcopy(self.deployment[fleet['evidence_key']])
                    if fleet['initial']:
                        path = self.directory/(fleet['node']+'_launch.json')
                        if not path.exists():path=self.directory/(fleet['node']+'.json')
                        item.update(worker_campaign=json.loads(path.read_text()),local_directory=str(self.directory))
                    try:
                        # Marker prevents new N4 generations; native transport
                        # sends SHUTDOWN at the next pull, after the current result.
                        row['drain'] = drain_attachment(self.launch,item,fleet['node'])
                        row['status']='DRAINING'
                    except Exception as exc:
                        row.update(status='DRAIN_RETRY',error=repr(exc),step='drain_marker',
                            stderr=getattr(exc,'stderr',None) or getattr(exc,'output',None))
                        # Dispatch stays fenced. Retry on the next poll; fail
                        # closed after the transport's bounded offline window.
                        self.registry.save(document,dict(event='drain_failure',request_id=key,**row))
                        raise
                self.registry.save(document,dict(event='request_processed',request_id=key,**row))

    def closed_fleet(self,owned,status):
        key = owned.get('fleet_key',owned['node'])
        if (status.get('status')!='CLOSED' or status.get('exit_code')!=0 or
                any(status.get(k) is not False for k in ('alive','child_alive','supervisor_alive')) or
                status.get('owned_worker_survivors')!=[]):
            return False
        with self.registry.locked() as document:
            fleet = next((f for f in document['fleets'].values() if f['evidence_key']==key),None)
            if fleet is None or fleet['state']!='DRAINING':return False
            fleet.update(state='DRAINED',closure=status,drained_at=time.time())
            for row in document['requests'].values():
                if row['fleet_id']==fleet['fleet_id'] and row['action']=='detach':row['status']='DRAINED'
            self.registry.save(document,dict(event='fleet_drained',fleet_id=fleet['fleet_id']))
            return True

    def freeze(self):
        with self.registry.locked() as document:
            if document['state']!='OPEN':return
            document['state']='CLOSING'
            for row in document['requests'].values():
                if row['status']=='QUEUED':row.update(status='REJECTED',error='campaign closing')
            self.registry.save(document,dict(event='attachments_fenced_for_finalization'))
        # Close GPUs/fleets before joining the CPU-only preparation thread.
        self.pool.shutdown(wait=False,cancel_futures=True)

    def seal_owners(self,report):
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.tick()  # A completed CPU preparation is recorded; never spawned now.
        with self.registry.locked() as document:
            for fleet in document['fleets'].values():
                if fleet['state']=='FAILED' and not fleet.get('spawn_requested'):continue
                rows = [r for r in report['cleanup'] if r.get('node')==fleet['node'] and
                    r.get('fleet_key',r.get('node'))==fleet['evidence_key'] and
                    r.get('kind',r.get('label'))=='workers']
                if len(rows)!=1 or rows[0].get('closure_confirmed') is not True:
                    raise ValueError('live fleet owner closure missing: '+fleet['fleet_id'])
                fleet.update(state='CLOSED',closure=rows[0])
            document['state']='CLOSED'
            self.registry.save(document,dict(event='all_fleet_owners_closed'))
            report['live_fleets']=document


def validate_seal(directory,deployment,report):
    if not report.get('live_fleets'):return
    directory=Path(directory)
    document=Registry(directory).read()
    if document!=report['live_fleets'] or document['state']!='CLOSED':
        pass
    if sha(directory/'campaign.json')!=document['campaign_sha256'] or sha(directory/'workers.tar')!=document['worker_archive_sha256']:
        pass
    from .formal_finalize import closed_proof
    proof=closed_proof(report)
    for fleet in document['fleets'].values():
        if fleet['state']=='FAILED' and not fleet.get('spawn_requested'):continue
        key=fleet['evidence_key']
        if key not in proof or key not in deployment or key not in report.get('transfers',{}):
            raise ValueError('live-fleet evidence missing: '+fleet['fleet_id'])
        archive=directory/(key+'.tar')
        if sha(archive)!=report['transfers'][key]['remote']['archive_sha256']:
            pass
    report['live_fleet_seal']=dict(schema=VERSION,registry_sha256=sha(directory/'LIVE_FLEETS.json'),
        event_log_sha256=sha(directory/'FLEET_EVENTS.jsonl'),fleet_ids=sorted(document['fleets']),
        payload_unchanged=True,scientific_inputs_unchanged=True)


def resume_pins(directory,report):
    """Extra source pins only; accepted-cell/attempt selection stays unchanged."""
    if not report.get('live_fleets'):return {}
    directory=Path(directory)
    document=Registry(directory).read()
    if document!=report['live_fleets'] or document['state']!='CLOSED' or report.get('all_owned_closed') is not True:
        pass
    for fleet in document['fleets'].values():
        if fleet['state']=='FAILED' and not fleet.get('spawn_requested'):continue
        row=fleet.get('closure',{})
        if (row.get('status')!='CLOSED' or row.get('closure_confirmed') is not True or
                any(row.get(k) is not False for k in ('alive','child_alive','supervisor_alive')) or
                row.get('owned_worker_survivors')!=[]):
            pass
    paths=[directory/'LIVE_FLEETS.json',directory/'FLEET_EVENTS.jsonl']
    paths.extend(p for p in (directory/'fleets').rglob('*') if p.is_file())
    if (directory/'driver/fleet_dispatch.jsonl').exists():paths.append(directory/'driver/fleet_dispatch.jsonl')
    return {str(p):sha(p) for p in paths}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('attach','detach','status'))
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--fleet-id')
    parser.add_argument('--node',choices=('local','timan107','timan108','timan1'))
    parser.add_argument('--workers',type=int)
    parser.add_argument('--run',action='store_true',help='submit to owning launcher; default read-only validation')
    args=parser.parse_args(argv)
    registry=Registry(args.campaign)
    if args.action=='status':
        result=registry.read()
        # Large resume cohorts stay in SHA-bound files, including status output.
        for fleet in result['fleets'].values():
            fleet.get('deployment',{}).pop('fleet_campaign',None)
        census=args.campaign/'driver/LIVE_FLEET_CENSUS.json'
        if census.exists():result['driver_observation']=json.loads(census.read_text())
    else:
        if not args.fleet_id:parser.error('--fleet-id required')
        result=registry.request(args.action,fleet_id=args.fleet_id,node=args.node,workers=args.workers,execute=args.run)
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=='__main__':main()
