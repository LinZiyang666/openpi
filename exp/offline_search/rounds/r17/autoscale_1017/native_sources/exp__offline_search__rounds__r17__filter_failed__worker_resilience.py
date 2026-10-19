"""ff/ffk worker placement and node retirement; native plans/releases are reused."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import inspect
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

from . import timan1_gpu2

NODES = ('timan107', 'timan108', 'timan1', 'local')
STATE = 'WORKER_NODE_STATE.json'


def capacity(node, run, vram, ram, growth=0, server_directory=None, timan1_cards=None):
    """Self-contained read-only probe: physical usage plus unmaterialized slots."""
    import os
    from pathlib import Path
    import json
    import subprocess
    import xml.etree.ElementTree as ET
    def alive(p):
        if not p:return False
        try:
            stat = Path('/proc/'+str(p['pid'])+'/stat').read_text().rsplit(')', 1)[1].split()
            return stat[0] != 'Z' and stat[19] == str(p['start_ticks'])
        except (OSError, KeyError):
            return False
    # timan1: PCI -> (UUID, kernel minor, NVML index), GPU2 first. The default is
    # the GPU2-only lease; new prepares/launches also pass GPU1 (owner 2026-10-18).
    timan1 = timan1_cards or {'0000:81:00.0': ('GPU-9bdcf7cc-8a05-a7f9-b1c3-4d53d0a229df', 1, 2)}
    optional = set(list(timan1)[1:]) if node == 'timan1' else set()
    cards, devices = {}, {}
    for info in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        fields = dict(line.split(':', 1) for line in info.read_text().splitlines() if ':' in line)
        minor = int(fields['Device Minor'].strip())
        if node == 'timan1' and info.parent.name not in timan1 or node == 'timan108' and minor in (1, 2):
            continue
        devices[info.parent.name] = minor
    if node == 'timan1' and next(iter(timan1)) not in devices:
        devices = {}  # the namespace never starts without GPU2; GPU1 alone is unused
    def query(pcis):
        xml = subprocess.run(['nvidia-smi', '-i', ','.join(pcis), '-q', '-x'],
                             capture_output=True, text=True, check=True, timeout=30).stdout
        return {c.findtext('pci/pci_bus_id').lower().replace('00000000:', '0000:'): c
                for c in ET.fromstring(xml).findall('gpu')}
    if devices:
        # One query for eligible PCI devices only. Cold GPUs without persistence
        # can take tens of seconds; protected cards are never queried.
        try:
            observed = query(devices)
        except (subprocess.SubprocessError, ET.ParseError):
            if not optional & set(devices):
                raise
            observed = {}  # an unhealthy GPU1 must not hide GPU2: ask each card alone
            for pci in devices:
                try:observed.update(query([pci]))
                except (subprocess.SubprocessError, ET.ParseError):pass
    for pci, minor in devices.items():
        if pci.lower() not in observed and pci in optional:
            continue  # unhealthy GPU1 contributes no slots; GPU2 replies stay mandatory
        card = observed[pci.lower()]  # Missing/partial replies are probe errors.
        if node == 'timan1' and (card.findtext('uuid') != timan1[pci][0] or minor != timan1[pci][1]):
            raise ValueError('timan1 GPU'+str(timan1[pci][2])+' identity changed')
        number = lambda s: int(s.split()[0])
        device = timan1[pci][2] if node == 'timan1' else minor
        cards[device] = dict(total=number(card.findtext('fb_memory_usage/total')),
            used=number(card.findtext('fb_memory_usage/used')), pending=0,
            processes={int(p.findtext('pid')): number(p.findtext('used_memory'))
                       for p in card.findall('processes/process_info') if p.findtext('used_memory') not in (None, 'N/A')})
    mem = {k: int(v.split()[0])//1024 for k, v in
           (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    pending_ram = 0
    for path in Path(run).parent.glob('*/manifest.json'):
        try:
            c = json.loads(path.read_text())
            receipt = json.loads((path.parent/'receipt.json').read_text())
            live = [p for p in receipt.get('pids', []) if alive(p)]
            owner = alive(receipt.get('fleet_owner', {}))
            slots = receipt.get('workers', c.get('node_workers', 0)) if owner else len(live)
            reserve = c.get('worker_ram_reservation_mib', c.get('worker_memory_mib', 4096))
            rss = []
            for p in live:
                try:
                    rss.append(next(int(s.split()[1])//1024 for s in Path('/proc/'+str(p['pid'])+'/status').read_text().splitlines() if s.startswith('VmRSS:')))
                except (OSError, StopIteration):
                    rss.append(0)
            pending_ram += sum(max(0, reserve-r) for r in rss)+max(0, slots-len(live))*reserve
            reservation = path.parent/'GPU_RESERVATION.json'
            r = json.loads(reservation.read_text()) if reservation.exists() else {}
            holding = alive(r.get('owner', {})) and not r.get('released')
            assignments = r.get('assignments', []) if holding else live
            if holding and not assignments:
                assignments = [dict(device=c['worker_egl_device'])]*slots
            for device, card in cards.items():
                selected = [a for a in assignments if a.get('device', a.get('egl_device', c.get('worker_egl_device'))) == device]
                actual = sum(card['processes'].get(p['pid'], 0) for p in live if p.get('egl_device', c.get('worker_egl_device')) == device)
                card['pending'] += max(0, len(selected)*r.get('worker_vram_mib', c.get('worker_vram_mib', 1024))-actual)
        except (OSError, ValueError, KeyError):
            # A torn/absent live reservation must not make a card look emptier.
            if path.parent.joinpath('receipt.json').exists():
                raise
    # growth is the whole admitted server budget, not a flat startup allowance.
    # Credit only live, receipt-owned replicas already included in physical
    # usage, so loading the server cannot release its reserved renderer slots.
    owners = []
    if growth and server_directory:
        receipt = Path(server_directory)/'RECEIPT.json'
        if receipt.exists():
            owners = [p['pid'] for p in json.loads(receipt.read_text()).get('owned_local', [])
                      if p.get('label', '').startswith('replica_') and alive(p)]
    for c in cards.values():
        c['server_resident_mib'] = sum(c['processes'].get(pid, 0) for pid in owners)
        c['server_pending_mib'] = max(0, growth-c['server_resident_mib'])
    gpu = sum(max(0, (c['total']-c['used']-c['pending']-(c['total']+9)//10-c['server_pending_mib'])//vram) for c in cards.values())
    memory = max(0, (mem['MemAvailable']-(mem['MemTotal']+9)//10-pending_ram)//ram)
    hard = {'libero': 83, 'metaworld': 332}
    cap = min(gpu, memory, hard['libero' if vram == 530 else 'metaworld']*len(timan1) if node == 'timan1' else gpu)
    return dict(capacity=cap, gpu_slots=gpu, ram_slots=memory, cards=cards, server_budget_mib=growth,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def measure(campaign, directory, excluded=()):
    from exp.r17_integration.ops import formal_eval as f
    def probe(node):
        if node in excluded:
            return node, dict(capacity=0, reason='retired in this launch')
        vram, ram = timan1_gpu2.SIZES[campaign['domain']]
        run = f.worker_paths(campaign, node)['run']
        # memory_budget_gib is already the total: the server divides it by R
        # before applying each replica's limit. Never multiply that total by R.
        from math import ceil
        growth = ceil(campaign['memory_budget_gib']*1024) if node == 'local' and campaign['server_node'] == 'local' else 0
        code = inspect.getsource(capacity)+'\nprint(__import__("json").dumps(capacity('+repr(node)+','+repr(run)+','+repr(vram)+','+repr(ram)+','+repr(growth)+','+repr(str(directory))+')))'
        argv = ['/usr/bin/python3', '-B', '-S', '-c', code]
        if node != 'local':
            argv = ['tether', 'exec', '--safe', '--timeout', '90s', node, '--', *argv]
        errors = []
        for attempt in (1, 2):
            try:
                row = json.loads(f.execute(argv, seconds=92))
                if type(row.get('capacity')) is not int or row['capacity'] < 0:
                    raise ValueError('invalid capacity probe reply')
                row.update(probe_status='MEASURED', probe_attempts=attempt, probe_errors=errors)
                return node, row
            except Exception as exc:
                errors.append(str(exc)[-1200:])
                print('WORKER_CAPACITY_PROBE node='+node+' attempt='+str(attempt)+
                      ' status=UNKNOWN '+('retrying' if attempt == 1 else 'excluded')+
                      ' reason='+repr(errors[-1]), file=sys.stderr, flush=True)
        return node, dict(capacity=None, probe_status='UNKNOWN', probe_attempts=2,
                          probe_errors=errors,
                          reason='capacity unknown after two failed probes; node excluded: '+errors[-1])
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(probe, NODES))


def allocate(requested, measured):
    caps = {n: r['capacity'] for n, r in measured.items()
            if r['capacity'] is not None and r['capacity'] > 0}
    if not caps:
        raise RuntimeError('no usable worker node remains')
    result = {n: min(requested.get(n, 0), cap) for n, cap in caps.items()}
    missing = min(sum(requested.values()), sum(caps.values()))-sum(result.values())
    spare = {n: caps[n]-result[n] for n in caps}
    total = sum(spare.values())
    if missing:
        shares = {n: missing*s//total for n, s in spare.items()}
        left = missing-sum(shares.values())
        order = sorted(spare, key=lambda n: (-(missing*spare[n] % total), n))
        for n in order[:left]:
            shares[n] += 1
        result = {n: result[n]+shares[n] for n in result}
    return {n: count for n, count in result.items() if count}


def record(campaign, directory, nodes, measured, phase):
    from exp.r17_integration.ops.formal_process import write_json
    from exp.r17_integration.ops.worker_topology import redistribute
    if nodes != campaign.get('execution_worker_nodes', campaign['worker_nodes']):
        redistribute(campaign, nodes, directory)
    if 'local' in nodes and not (Path(directory)/'LOCAL_WORKER_ORCHESTRATION.json').exists():
        from exp.r17_integration.ops.immutable_release import sha
        write_json(Path(directory)/'LOCAL_WORKER_ORCHESTRATION.json',dict(
            schema='r17.command_worker_orchestration.v1',archive_sha256=sha(Path(directory)/'workers.tar'),
            native_worker_bytes_unchanged=True,owner_authorized_any_workload=True))
    campaign['worker_node_failover'] = dict(schema='r17.worker_node_failover.v1', enabled=True)
    write_json(Path(directory)/STATE, dict(nodes=nodes, retired=[], measured=measured, phase=phase))
    write_json(Path(directory)/'campaign.json', campaign)


def select(campaign, directory, deployment, *, persist=False, excluded=()):
    rows = measure(campaign, directory, excluded)
    nodes = allocate(campaign.get('execution_worker_nodes', campaign['worker_nodes']), rows)
    if persist:
        record(campaign, directory, nodes, rows, 'stage')
    else:
        campaign = copy.deepcopy(campaign)
        campaign['execution_worker_nodes'] = nodes
        campaign['execution_workers'] = sum(nodes.values())
        campaign['worker_node_failover'] = dict(schema='r17.worker_node_failover.v1', enabled=True)
        campaign['_worker_node_measurements'] = rows
    return campaign, {n: v for n, v in deployment.items() if n in nodes}


def prepare(campaign, directory, native):
    """Prepare only the selected nodes; retry a failed node's share elsewhere."""
    requested = dict(campaign.get('execution_worker_nodes', campaign['worker_nodes']))
    failed = set()
    while True:
        rows = measure(campaign, directory, failed)
        nodes = allocate(requested, rows)
        record(campaign, directory, nodes, rows, 'stage')
        result = {}
        for node, number in nodes.items():
            single = copy.deepcopy(campaign)
            single['execution_worker_nodes'] = {node: number}
            single['execution_workers'] = number
            vram,ram=timan1_gpu2.SIZES[campaign['domain']]
            single.update(worker_vram_mib=vram,worker_memory_mib=ram,worker_ram_reservation_mib=ram)
            try:
                result.update(native(single, directory))
            except Exception:
                failed.add(node)
                break
        else:
            from exp.r17_integration.ops.formal_process import write_json
            if 'timan1' in nodes:
                campaign['timan1_worker_lease']=timan1_gpu2.authorize(campaign)
            write_json(Path(directory)/'WORKER_DEPLOYMENT.json',result)
            return result


def launch_workers(campaign, directory, deployment, preflight):
    """Called after the launch claim; release bytes and scientific inputs stay fixed."""
    from exp.r17_integration.ops import formal_eval as f
    rows = campaign.pop('_worker_node_measurements')
    nodes = dict(campaign['execution_worker_nodes'])
    failed = set()
    # Record with the same native topology knob as resume --worker-counts.
    original = json.loads((Path(directory)/'campaign.json').read_text())
    requested=dict(original.get('execution_worker_nodes',original['worker_nodes']))
    campaign['execution_worker_nodes'] = original.get('execution_worker_nodes', original['worker_nodes'])
    campaign['execution_workers'] = original.get('execution_workers',original['workers'])
    record(campaign, directory, nodes, rows, 'launch')
    while True:
        for node in nodes:
            if node not in deployment:
                single = copy.deepcopy(campaign)
                single['execution_worker_nodes'] = {node: nodes[node]}
                single['execution_workers'] = nodes[node]
                vram,ram=timan1_gpu2.SIZES[campaign['domain']]
                single.update(worker_vram_mib=vram,worker_memory_mib=ram,worker_ram_reservation_mib=ram)
                try:
                    source=campaign.get('cache_resume_selection',{}).get('source',campaign['filter_failed_source'])
                    added=f.prepare_workers(single, directory, payload_override=Path(source)/'workers.tar')
                    from .stage import stage_worker_inputs
                    stage_worker_inputs(f,single,Path(directory),added)
                    deployment.update(added)
                except Exception:
                    failed.add(node)
                    break
            try:
                preflight(node, deployment[node])
            except Exception:
                failed.add(node)
                break
        else:
            from exp.r17_integration.ops.formal_process import write_json
            result={n:deployment[n] for n in nodes}
            write_json(Path(directory)/'WORKER_DEPLOYMENT.json',result)
            return result
        rows=measure(campaign,directory,failed)
        for n in failed:
            rows[n] = dict(capacity=0, reason='worker launch preflight failed')
        nodes = allocate(requested, rows)
        record(campaign, directory, nodes, rows, 'launch')


class Running:
    def __init__(self, campaign, directory, deployment, report):
        self.c, self.root, self.deployment, self.report = campaign, Path(directory), deployment, report
        state = json.loads((self.root/STATE).read_text())
        self.rows = state['measured']
        self.nodes = dict(campaign['execution_worker_nodes'])
        self.retired = {}
        self.unknown_since = {}

    def retire(self, node, reason, owned=None):
        from exp.r17_integration.ops.formal_process import write_json
        from exp.r17_integration.ops.worker_topology import redistribute
        if node not in self.retired:
            self.retired[node] = dict(reason=reason, time=time.time(),
                boot_id=self.rows.get(node, {}).get('boot_id'), workers=self.nodes.pop(node, 0))
            self.report['worker_node_losses'] = self.retired
            if self.nodes:
                redistribute(self.c, self.nodes, self.root)
                write_json(self.root/'campaign.json', self.c)
            write_json(self.root/STATE, dict(nodes=self.nodes, retired=list(self.retired), measured=self.rows, phase='running'))
        if owned is not None:
            owned['retired_worker_node'] = True

    def check(self, owned, status):
        from exp.r17_integration.ops.formal_eval import worker_fleet_failed
        if owned.get('retired_worker_node'):
            return True
        node=owned['node']
        if status.get('control_probe_failed') or status.get('status') in ('UNKNOWN','OFFLINE'):
            elapsed=time.monotonic()-self.unknown_since.setdefault(node,time.monotonic())
            offline=(status.get('status')=='OFFLINE' or any(s in str(status.get('error',''))
                     for s in ('node_offline', 'status=OFFLINE', 'agent is OFFLINE')))
            if offline or elapsed>=300:
                self.retire(node,dict(status,probe_failure_seconds=elapsed),owned)
            # Handled: do not let the caller interpret transport FAIL as death.
            return True
        self.unknown_since.pop(node,None)
        if status.get('status')=='CLOSED' and status.get('exit_code')==0 and self.complete():
            return True
        if worker_fleet_failed(status, dispatch_started=False):
            self.retire(owned['node'], status, owned)
            return True
        return False

    def require_workers(self):
        if not self.nodes and not self.complete():
            raise RuntimeError('no usable worker node remains')

    def complete(self):
        path=self.root/'WORKER_DRIVER_STATE.json'
        return path.exists() and json.loads(path.read_text()).get('all_done') is True

    def cleanup(self, owned, managed):
        """A reboot proves old owners gone; unreachable never means closed."""
        from exp.r17_integration.ops import formal_eval as f
        node = owned['node']
        try:
            boot = json.loads(f.remote(node, ['/usr/bin/python3', '-B', '-S', '-c',
                'import json;from pathlib import Path;print(json.dumps(Path("/proc/sys/kernel/random/boot_id").read_text().strip()))'], seconds=5, replay_safe=True))
            if self.retired[node].get('boot_id') and boot != self.retired[node]['boot_id']:
                return dict(status='CLOSED', alive=False, child_alive=False, supervisor_alive=False,
                            owned_worker_survivors=[], exit_code=2, reboot_proven=True)
            return managed(owned, 'stop')
        except Exception as exc:
            return dict(status='UNKNOWN', alive=None, child_alive=None, supervisor_alive=None,
                        error=str(exc)[-1200:], isolation_state=str(self.root/STATE))


def install_driver():
    """Process-only hook over frozen transport, queue and accepted-result fences."""
    import json
    import socket
    import sys
    import threading
    from pathlib import Path
    from openpi.conductor.driver import ConductorDriver
    from openpi.conductor.protocol import MSG_SHUTDOWN
    from exp.r17_integration.qualification.driver import ProbeRecordingDriver as Driver
    campaign_path=Path(sys.argv[sys.argv.index('--campaign')+1])
    campaign=json.loads(campaign_path.read_text())
    state_path=campaign_path.parent/'WORKER_NODE_STATE.json'
    run_name=campaign['run_name']
    def node_of(worker):
        return next((n for n in ('timan107','timan108','timan1','local')
                     if isinstance(worker,str) and worker.startswith(run_name+'_'+n+'_')),None)
    def sync(driver):
        with driver._node_lock:
            stamp=state_path.stat().st_mtime_ns
            if stamp!=driver._node_stamp:
                state=json.loads(state_path.read_text())
                driver._node_counts=state['nodes']
                driver._node_retired=set(state['retired'])
                driver._node_stamp=stamp
            connections=[conn for conn,worker in driver._node_connections.items()
                         if node_of(worker) not in driver._node_counts]
            with driver._shutdown_lock:
                driver.expected_workers=sum(driver._node_counts.values())
                driver._shutdown_workers={w for w in driver._shutdown_workers if node_of(w) in driver._node_counts}
            if driver.scheduler.all_done() and not getattr(driver,'_node_done_published',False):
                path=state_path.with_name('WORKER_DRIVER_STATE.json')
                part=path.with_suffix('.part')
                part.write_text(json.dumps(dict(all_done=True)))
                part.replace(path)
                driver._node_done_published=True
        # Closing an owned driver socket invokes the frozen disconnect/fence
        # path. Its unaccepted in-flight episode is requeued at attempt+1.
        for conn in connections:
            try:conn.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            conn.close()
    init,connection,pull,run=Driver.__init__,Driver._handle_conn,Driver._record_pull,Driver.run
    native_pull=ConductorDriver.handle_pull
    def initialize(self,*a,**kw):
        init(self,*a,**kw)
        self._node_lock=threading.RLock()
        self._node_connections={}
        self._node_stamp=None
        sync(self)
    def handle_connection(self,conn,stop):
        self._shutdown_local.node_connection=conn
        try:return connection(self,conn,stop)
        finally:
            with self._node_lock:self._node_connections.pop(conn,None)
    def record_pull(self,payload,peer):
        sync(self)
        worker=payload.get('worker_id')
        self._shutdown_local.node_rejected=node_of(worker) not in self._node_counts
        if self._shutdown_local.node_rejected:
            self._shutdown_local.worker_id=worker
            return worker  # Shutdown, never record a scientific probe rejection.
        with self._node_lock:
            self._node_connections[self._shutdown_local.node_connection]=worker
        return pull(self,payload,peer)
    def handle_pull(self,key):
        if getattr(getattr(self,'_shutdown_local',None),'node_rejected',False):
            return MSG_SHUTDOWN,{}
        return native_pull(self,key)
    def run_driver(self,stop=None,**kw):
        def check():
            sync(self)
            return bool(stop and stop())
        return run(self,stop=check,**kw)
    Driver.__init__,Driver._handle_conn,Driver._record_pull,Driver.run=initialize,handle_connection,record_pull,run_driver
    ConductorDriver.handle_pull=handle_pull


def driver_command(argv):
    modules=('exp.r17_integration.ops.formal_driver','exp.r17_integration.ops.formal_driver_fast',
             'exp.r17_integration.ops.cache_driver_legacy')
    if not isinstance(argv,list) or '-m' not in argv:
        return argv
    i=argv.index('-m')
    if argv[i+1] not in modules:
        return argv
    code='import sys,runpy\n'+inspect.getsource(install_driver)+'\ninstall_driver()\nm=sys.argv.pop(1);runpy.run_module(m,run_name="__main__",alter_sys=True)\n'
    return [*argv[:i],'-c',code,*argv[i+1:]]


def install_worker(inventory_source):
    """Use physical PCI discovery and measured reservations in frozen workers."""
    from exp.r17_integration.ops import worker_gpu as g,worker_memory as m
    exec(inventory_source,g.__dict__)
    admission=g.admission
    # Payloads frozen before reservation_mib keep their native per-worker RAM admission.
    if hasattr(m,'reservation_mib'):
        reservation=m.reservation_mib
        m.reservation_mib=lambda c:c['worker_ram_reservation_mib'] if 'worker_ram_reservation_mib' in c else reservation(c)
    def admit(c,*a,**kw):
        result=admission(c,*a,**kw)
        slots={r['device']:max(0,(r['total_mib']-r['used_mib']-r['headroom_mib']-r['pending_mib'])//result['worker_vram_mib'])
               for r in result['cards']}
        cards={r['device']:r for r in result['cards']}
        required=result['requested_workers']
        assignments=[]
        while len(assignments)<required:
            before=len(assignments)
            for device in sorted(slots):
                if slots[device] and len(assignments)<required:
                    assignments.append(dict(index=len(assignments),device=device,uuid=cards[device]['uuid'],
                                            worker_vram_mib=result['worker_vram_mib']))
                    slots[device]-=1
            if len(assignments)==before:break
        result.update(assignments=assignments,admitted=len(assignments)==required,
                      reason=None if len(assignments)==required else 'measured worker VRAM capacity refused')
        return result
    g.admission=admit


def worker_patch():
    from exp.r17_integration.ops import worker_gpu
    return inspect.getsource(install_worker)+'\ninstall_worker('+repr(inspect.getsource(worker_gpu.inventory))+')\n'


@contextmanager
def coordinator():
    from exp.r17_integration.ops import formal_eval as f, local_workers as l,worker_memory as m
    remote, route,reservation,authorize = f.remote,l.route,m.reservation_mib,l.authorize
    def call(node, argv, seconds=60, **kw):
        # Bounded single calls on worker machines; failed/ambiguous starts are
        # retired, never replayed. Server transport retains its native policy.
        if node not in NODES or node == 'local':
            return remote(node, argv, seconds=seconds, **kw)
        if node=='timan1':argv=timan1_gpu2.auxiliary(argv)
        # Dependency hashing may legitimately take minutes. Control probes and
        # spawn are short; a read-only verification retains its caller's bound.
        bound=seconds if seconds>60 else min(seconds,10)
        return f.execute(['tether', 'exec', '--safe', '--timeout', str(bound)+'s', node, '--', *argv], seconds=bound+2)
    def local_route(worker, campaign):
        worker = route(worker, campaign)
        vram, ram = timan1_gpu2.SIZES[campaign['domain']]
        worker.update(worker_vram_mib=vram, worker_memory_mib=ram, worker_ram_reservation_mib=ram,
                      worker_server_vram_headroom_mib=campaign.get('local_worker_server_vram_headroom_mib', 16384) if campaign['server_node'] == 'local' else 0)
        return worker
    def local_authorize(campaign):
        if campaign.get('domain')!='metaworld':return authorize(campaign)
        nodes=campaign.get('execution_worker_nodes',campaign['worker_nodes'])
        return dict(schema='r17.local_worker_exception.v1',study_date=l.STUDY_DATE,workers=nodes,
                    scope='owner-authorized ff/ffk workers, any workload',node='local',host='weilandserver',
                    model_sampler_server_unchanged=True) if 'local' in nodes else None
    with patch.object(f, 'worker_resilience', sys.modules[__name__], create=True), \
            patch.object(f, 'remote', call), patch.object(l, 'route', local_route),patch.object(l,'authorize',local_authorize), \
            patch.object(m,'reservation_mib',lambda c:c['worker_ram_reservation_mib'] if 'worker_ram_reservation_mib' in c else reservation(c)):
        yield
