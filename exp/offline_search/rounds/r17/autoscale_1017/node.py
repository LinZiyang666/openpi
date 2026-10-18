"""Node-local cooperative memory ledger. No remote calls and no signals."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from .policy import Config, MemoryGuard, Peaks


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.%d.part' % os.getpid())
    part.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    os.replace(part, path)


def event(path, node, delta, reason, signals, counts):
    """Append only; lock serializes supervisor/driver writers, including threads."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as out:
        fcntl.flock(out, fcntl.LOCK_EX)
        out.write(json.dumps(dict(time=time.time(), node=node, delta=delta, reason=reason,
                                  signals=signals, counts=counts), sort_keys=True, allow_nan=False) + '\n')
        out.flush()


def identity(pid, proc=Path('/proc')):
    try:
        fields = (proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError, IndexError):
        return None


def process_tree(pid, start_ticks, proc=Path('/proc')):
    if identity(pid, proc) != str(start_ticks):
        return None
    rss = peak = 0
    todo, seen = [pid], set()
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            fields = dict(line.split(':', 1) for line in (proc / str(current) / 'status').read_text().splitlines())
            rss += int(fields.get('VmRSS', '0').split()[0]) * 1024
            peak += int(fields.get('VmHWM', '0').split()[0]) * 1024
            todo.extend(map(int, (proc / str(current) / 'task' / str(current) / 'children').read_text().split()))
        except (OSError, ValueError):
            # Reaped children retain their last observed generation high-water.
            continue
    return rss, max(rss, peak)


def process_ids(pid, start_ticks, proc=Path('/proc')):
    if identity(pid,proc)!=str(start_ticks):return []
    todo,seen=[pid],set()
    while todo:
        current=todo.pop()
        if current in seen:continue
        seen.add(current)
        try:todo.extend(map(int,(proc/str(current)/'task'/str(current)/'children').read_text().split()))
        except (OSError,ValueError):pass
    return sorted(seen)


def memory(proc=Path('/proc')):
    fields = dict(line.split(':', 1) for line in (proc / 'meminfo').read_text().splitlines())
    available = int(fields['MemAvailable'].split()[0]) * 1024
    pressure = {}
    for line in (proc / 'pressure/memory').read_text().splitlines():
        kind, *values = line.split()
        pressure[kind] = {k: float(v) for k, v in (s.split('=') for s in values)}
    # Also respect a tighter cgroup limit; /proc MemAvailable is host-wide.
    for line in (proc / 'self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if not controllers:
            base, limit, used = Path('/sys/fs/cgroup') / relative.lstrip('/'), 'memory.max', 'memory.current'
        elif 'memory' in controllers.split(','):
            base, limit, used = Path('/sys/fs/cgroup/memory') / relative.lstrip('/'), 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        else:
            continue
        try:
            available = min(available, max(0, int((base / limit).read_text()) - int((base / used).read_text())))
        except (OSError, ValueError):
            pass
    return available, pressure


def campaign_key(campaign):
    # run_name / worker placement are deliberately absent; allow recent
    # generations from a continuation with identical scientific configuration.
    keys = ('semantic_campaign', 'model', 'checkpoint', 'domain', 'benchmark_protocol', 'worker_benchmark_identity',
            'worker_max_episodes', 'worker_max_seconds', 'runtime_identity')
    return hashlib.sha256(json.dumps({k: campaign.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()


def external_legacy(covered, margin, proc=Path('/proc')):
    """Charge uncooperative LIBERO workers' observed high-water, never signal them."""
    workers = []
    for path in proc.glob('[0-9]*'):
        try:
            pid = int(path.name)
            if pid in covered or path.stat().st_uid != os.getuid():continue
            argv = (path/'cmdline').read_bytes().split(b'\0')
            if not argv or not Path(os.fsdecode(argv[0])).name.startswith('python'):continue
            index = argv.index(b'-m') if b'-m' in argv else -1
            if index < 1 or argv[index+1:index+2] != [b'examples.libero.worker_entry']:continue
            if any(not v.startswith(b'-') for v in argv[1:index]):continue
            ticks = identity(pid, proc)
            if ticks is None:continue
            sample = process_tree(pid, ticks, proc)
            if sample is None:continue
            rss, peak = sample
            workers.append(dict(pid=pid, worker_id='external_'+str(pid), rss=rss, peak=peak,
                projected_peak=int(max(rss,peak)*margin+.999), owned=False, born=0))
        except (FileNotFoundError, ProcessLookupError):continue
    return workers


class LocalGovernor:
    def __init__(self, campaign, run, *, ledger=None, clock=time.monotonic,
                 reader=memory, process_reader=process_tree):
        self.campaign, self.run = campaign, Path(run)
        self.node = campaign.get('worker_node', 'local')
        self.config = Config(**dict(campaign.get('autoscale', {}),
            **campaign.get('autoscale_node_config', {}).get(self.node, {})))
        if self.node == 'timan108':
            raise ValueError('timan108 is disabled')
        self.maximum = campaign['node_workers']
        key_config = dict(campaign)
        key_config.setdefault('semantic_campaign', campaign['run_name'])
        benchmark = campaign.get('worker_benchmark')
        if benchmark and Path(benchmark).is_file():
            key_config['worker_benchmark_identity'] = hashlib.sha256(Path(benchmark).read_bytes()).hexdigest()
        self.key = campaign_key(key_config)
        self.clock, self.reader, self.process_reader = clock, reader, process_reader
        self.guard = MemoryGuard(self.config)
        self.root = Path(ledger or os.environ.get('OPENPI_AUTOSCALE_LEDGER', '/tmp/openpi-autoscale-%d' % os.getuid()))
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.stat().st_uid != os.getuid():
            raise ValueError('autoscale ledger is not owned by current UID')
        self.boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        from .kernel_evidence import begin
        begin(self.run)
        self.started_wall = time.time()
        self.owner = dict(pid=os.getpid(), start_ticks=identity(os.getpid()))
        self.fleet = hashlib.sha256(str(self.run.resolve()).encode()).hexdigest()[:24]
        self.path = self.root / (self.fleet + '.json')
        self.history_path = self.root / ('peaks-' + self.key + '.json')
        self.snapshot_path = self.run / 'AUTOSCALE_NODE.json'
        self.control_path = self.run / 'AUTOSCALE_CONTROL.json'
        self.events = self.run / 'AUTOSCALE_DECISIONS.jsonl'
        self.entries, self.retired = {}, set()
        self.live_states=[]
        self.gpu_view=None
        self.fairness_campaign=campaign.get('autoscale_fairness_campaign',campaign['run_name'])
        self.last_reason = None
        self.last_sample_log = float('-inf')
        self.snapshot = {}
        self.minimum_available = None
        self.last_spawn = float('-inf')
        self.target = campaign.get('autoscale_initial', self.maximum)
        self.legacy = campaign.get('autoscale_legacy', False)
        if self.legacy:
            self.target = self.maximum
        self.run.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def lock(self):
        with (self.root / 'node.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def register(self, entry):
        entry = dict(entry)
        entry.setdefault('born', self.clock())
        entry.setdefault('peak', 0)
        entry.setdefault('rss', 0)
        self.entries[entry['worker_id']] = entry
        self.last_spawn = self.clock()
        self._publish()
        event(self.events, self.node, 1, 'spawn', self.snapshot,
              dict(active=len(self.entries), target=self.target, maximum=self.maximum))

    def finished(self, worker_id, *, normal=False):
        entry = self.entries.pop(worker_id)
        marker = read(Path(entry['recycle_receipt']).with_name('PEAK.json'), {}) if entry.get('recycle_receipt') else {}
        if (marker.get('pid'), marker.get('start_ticks')) == (entry['pid'], entry['start_ticks']):
            entry['peak'] = max(entry['peak'], marker.get('peak', 0))
        history = read(self.history_path, [])
        # Partial/failed generations also retain their measured high-water;
        # only a complete planned generation unlocks bootstrap calibration.
        history.append(dict(peak=entry['peak'], complete=normal, boot=self.boot))
        atomic(self.history_path, history[-128:])
        self.retired.discard(worker_id)
        self._publish()
        event(self.events, self.node, -1, 'generation_exit', dict(peak=entry['peak'], complete=normal),
              dict(active=len(self.entries), target=self.target, maximum=self.maximum))

    def _publish(self):
        atomic(self.path,self._state())

    def _state(self):
        return dict(schema=2,owner=self.owner,boot=self.boot,updated=self.clock(),campaign=self.key,
            memory_sample_valid=getattr(self,'memory_sample_valid',True),
            fairness_campaign=self.fairness_campaign,fleet=self.fleet,run=str(self.run),
            target=getattr(self,'fairness_target',self.target),requested_target=self.target,
            worker_vram_mib=self.campaign.get('worker_vram_mib',0),gpu_view=self.gpu_view,
            server_vram_headroom_mib=self.campaign.get('worker_server_vram_headroom_mib',0),
            last_spawn=self.last_spawn if self.last_spawn != float('-inf') else None,
            workers=list(self.entries.values()),projected_peak=self.snapshot.get('projected_peak',0))

    def _sample(self):
        now = self.clock()
        valid = True
        try:
            available, pressure = self.reader()
        except (OSError, ValueError, KeyError):
            available, pressure, valid = 0, {}, False
        self.memory_sample_valid = valid  # local evidence only; avoid a peer-error feedback latch
        if valid:
            self.minimum_available = available if self.minimum_available is None else min(self.minimum_available, available)
        history = read(self.history_path, [])
        peaks = Peaks(self.config.peak_margin)
        for item in history:
            peaks.add(self.key, item['peak'])
        calibrated = any(item.get('complete') and item.get('peak', 0) > 0 for item in history)
        projection = peaks.project(self.key)
        # Unknown new campaign: one canary, using half of currently free
        # headroom as a provisional envelope, never a permanent per-worker cap.
        projection = projection or max(1, (available - self.config.floor) // 2)
        own = []
        for worker_id, entry in self.entries.items():
            sample = self.process_reader(entry['pid'], entry['start_ticks'])
            if sample is None:
                continue
            entry['rss'], peak = sample
            entry['peak'] = max(entry['peak'], peak)
            entry['projected_peak'] = max(projection, entry['peak'])
            entry['draining'] = worker_id in self.retired
            entry['gpu_pids']=process_ids(entry['pid'],entry['start_ticks']) or [entry['pid']]
            own.append(dict(entry, owned=True))
        workers = list(own)
        peers=[]
        other_canary = False
        for path in self.root.glob('*.json'):
            if path == self.path or path.name.startswith('peaks-'):
                continue
            state = read(path)
            if state is None:
                valid = False
                continue
            owner = state.get('owner', {})
            if state.get('boot') != self.boot:
                continue
            if state.get('schema',0)>=2 and not state['target'] and not state['workers']:
                continue  # a finished idle supervisor holds no future growth
            owner_alive = identity(owner.get('pid', 0)) == owner.get('start_ticks')
            if not owner_alive:
                survivors = [w for w in state['workers'] if identity(w['pid']) == str(w['start_ticks'])]
                if not survivors:continue
                state = dict(state, workers=survivors)
            # A stalled owner or surviving orphan cannot release future growth.
            if not owner_alive or now - state['updated'] > 3 * self.config.sample_s:
                valid = False
            valid &= state.get('memory_sample_valid',True)
            other_canary |= state['campaign'] == self.key and bool(state['workers'])
            peers.append(state)
            for w in state['workers']:
                workers.append(dict(w, owned=False, projected_peak=max(w.get('peak', 0), w.get('projected_peak', 0), state['projected_peak'])))
        unmanaged = []
        if self.node in self.campaign.get('autoscale_external_legacy_nodes', []):
            covered = {pid for w in workers for pid in w.get('gpu_pids', [w['pid']])}
            try:unmanaged = external_legacy(covered, self.config.peak_margin)
            except (OSError, ValueError):valid = False
            # Fresh legacy generations have not reached their high-water yet.
            # Retain the node's observed envelope across PID turnover, with the
            # explicitly measured technical bootstrap floor from preparation.
            path=self.root/'peaks-external-libero.json';prior=read(path,{})
            envelope=max([self.config.external_peak_floor,prior.get('peak',0)]+
                         [w['projected_peak'] for w in unmanaged])
            for w in unmanaged:w['projected_peak']=max(w['projected_peak'],envelope)
            if envelope>prior.get('peak',0):atomic(path,dict(peak=envelope))
        external_growth = sum(max(0,w['projected_peak']-w['rss']) for w in unmanaged)
        workers.extend(unmanaged)
        control = read(self.control_path, {})
        fresh = (control.get('campaign') == self.campaign['run_name'] and
                 0 <= time.time() - control.get('time', 0) <= self.config.stale_s)
        if fresh and not self.legacy:
            self.target = min(self.maximum, max(0, int(control.get('targets', {}).get(self.node, 0))))
        if control.get('done') and fresh:
            self.target = 0
        demand = self.target
        if fresh and not self.legacy:
            demand = min(self.maximum, max(0, int(control.get('demand_targets', control.get('targets', {})).get(self.node, 0))))
        if fresh and control.get('done'):demand = 0
        self.fairness_target=demand if calibrated else min(demand,1)
        # A new lane must be able to reclaim a canary envelope even if an older
        # lane borrowed all free RAM. The pool includes returnable resident RSS.
        if not peaks.project(self.key):
            fair_peers=[s for s in peers if s.get('schema',0)>=2]
            k=len({self.fairness_campaign,*[s['fairness_campaign'] for s in fair_peers if s['target']>0]})
            pool=available-self.config.floor+sum(w.get('rss',0) for s in fair_peers for w in s['workers'])+sum(w['rss'] for w in own)
            projection=max(1,pool//(2*max(1,k)))
            for w in own:w['projected_peak']=max(projection,w['peak'])
            for w in self.entries.values():w['projected_peak']=max(projection,w['peak'])
        slots=getattr(self,'slots',None)
        self.gpu_view=slots.view() if slots is not None else None
        state=self._state()
        state.update(projected_peak=projection,workers=own)
        self.live_states=[*peers,state]
        from .fairness import plan
        sharing=plan(self.live_states,max(0,available-external_growth),self.config.floor,now=now)
        group=sharing['groups'][self.fairness_campaign]
        fair_limit=sharing['quotas'][self.fairness_campaign]
        group_live=group['workers']
        returning=sum(bool(w.get('draining')) for w in group_live)
        fair_excess=max(0,len(group_live)-returning-fair_limit)
        newest=sorted((w for w in group_live if not w.get('draining')),
                      key=lambda w:(w['born'],w['fleet'],w['worker_id']),reverse=True)
        fair_retire={w['worker_id'] for w in newest[:fair_excess] if w['fleet']==self.fleet} if valid and sharing['campaigns']>1 else set()
        decision = self.guard.evaluate(now, available, workers, projection, valid=self.memory_sample_valid,
                                       psi_full=pressure.get('full', {}).get('avg10', 0))
        if not valid:
            self.guard.blocked, self.guard.healthy_since = True, None
            decision['admit'] = False
            if decision['reason'] != 'critical':decision['reason'] = 'memory_unknown'
        new_retire = set(decision['retire'])
        excess = max(0, len(own) - len(self.retired) - self.target - len(new_retire))
        candidates = sorted(own, key=lambda w: w['born'], reverse=True)
        # Only select the requested excess; critical candidates are mandatory.
        mandatory = set(decision['retire'])
        extra = [w['worker_id'] for w in candidates if w['worker_id'] not in self.retired | mandatory][:excess]
        new_retire = mandatory | set(extra) | fair_retire
        if new_retire - self.retired:
            event(self.events, self.node, -len(new_retire - self.retired),
                  'retire_requested_' + ('critical' if mandatory else 'fair_share' if fair_retire else 'target'),
                  dict(decision,fairness={k:v for k,v in sharing.items() if k!='groups'}),
                  dict(active=len(own), draining=len(self.retired | new_retire), target=self.target))
        returning += len(new_retire - self.retired)
        self.retired |= new_retire
        for w in self.entries.values():w['draining'] = w['worker_id'] in self.retired
        for w in own:w['draining'] = w['worker_id'] in self.retired
        canary_live = bool(own) or other_canary
        # Other fleets of this campaign are charged above; provisional growth
        # reservations keep concurrent calibration admissions serialized.
        decision['admit'] &= calibrated or not canary_live
        decision['admit'] &= self.legacy or fresh
        fair_admit=len(group_live)<fair_limit
        if sharing['vram_headroom'] is not None:
            fair_admit &= sharing['vram_headroom']>=group['vram']
        if decision['admit'] and not fair_admit:decision['reason']='fair_share_wait'
        decision['admit'] &= fair_admit
        decision['fairness']={k:v for k,v in sharing.items() if k!='groups'}
        decision['fairness'].update(own_campaign=self.fairness_campaign,limit=fair_limit,
            used=len(group_live),returning=returning,entitlement=1/max(1,sharing['campaigns']))
        decision.update(external_legacy_workers=len(unmanaged), external_legacy_growth=external_growth,
            spawn_interval_s=self.config.spawn_interval_s)
        last_spawn = max([self.last_spawn] + [s['last_spawn'] for s in peers if s.get('last_spawn') is not None])
        if decision['admit'] and now-last_spawn < self.config.spawn_interval_s:
            decision.update(admit=False, reason='spawn_ramp_wait')
        self.snapshot = dict(decision, node=self.node, boot=self.boot, time=time.time(),
                             projected_peak=projection, memory_sample_valid=self.memory_sample_valid,
                             sampled_mono=now, campaign=self.campaign['run_name'],
                             owner=self.owner, workers=own, active=len(own), target=self.target,
                             maximum=self.maximum, retired=sorted(self.retired), calibrated=calibrated,
                             control_epoch=control.get('epoch'), control_sequence=control.get('sequence', 0),
                             pressure=pressure, minimum_available=self.minimum_available, done=bool(fresh and control.get('done')))
        self._publish()
        atomic(self.snapshot_path, self.snapshot)
        if now - self.last_sample_log >= 10:
            event(self.run / 'AUTOSCALE_SAMPLES.jsonl', self.node, 0, 'node_sample',
                  dict(available=available, minimum_available=self.minimum_available, pressure=pressure, remaining_growth=decision.get('remaining_growth'),
                       rss=sum(w['rss'] for w in own), peak=sum(w['peak'] for w in own)),
                  dict(active=len(own), draining=len(self.retired), target=self.target, maximum=self.maximum))
            self.last_sample_log = now
        if decision['reason'] != self.last_reason:
            event(self.events, self.node, 0, decision['reason'], decision,
                  dict(active=len(own), target=self.target, maximum=self.maximum))
            self.last_reason = decision['reason']
        return self.snapshot

    def tick(self):
        with self.lock():
            return self._sample()

    @contextmanager
    def permit(self):
        """Hold node lock through Popen + register, so pending growth is atomic."""
        with self.lock():
            sample = self._sample()
            yield bool(sample['admit'] and len(self.entries) < self.target and not self.retired)

    def worker_env(self, worker_id):
        return dict(OPENPI_WORKER_OBSERVER='exp.offline_search.rounds.r17.autoscale_1017.worker:Observer',
                    OPENPI_AUTOSCALE_NODE=str(self.snapshot_path), OPENPI_AUTOSCALE_CONTROL=str(self.control_path),
                    OPENPI_AUTOSCALE_CAMPAIGN=self.campaign['run_name'],
                    OPENPI_AUTOSCALE_MAX_EPISODES=str(self.campaign.get('worker_max_episodes', 32)),
                    OPENPI_AUTOSCALE_MAX_SECONDS=str(self.campaign.get('worker_max_seconds', 1800)))
