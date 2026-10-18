"""R16 resident groups, adaptive episode queue, and resumable conductor driver.

Only explicit driver/agent/run/serve commands operate real environments.
prepare-dev and prepare-test write definitions and audit completed B runs.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import sys
import threading
import time

from openpi.conductor.driver import ConductorDriver
from openpi.conductor.journal import Journal
from openpi.conductor.scheduler import EpisodeScheduler, StageState
from openpi.conductor.strategy import ExperimentStrategy
from openpi.conductor.task import EpisodeTask, ServerEndpoint, Stage, TaskGraph, make_task_uid

from .prepare import HERE, REPO, TRACE, CELLS, ORDER, SEED, parts, atomic, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def read_group(path, reader=None):
    read = reader.read if reader else lambda p: Path(p).read_bytes()
    group = json.loads(read(path))
    digest = group.pop('group_sha256')
    if hashlib.sha256(canonical(group)).hexdigest() != digest:
        raise ValueError('Changed conductor group')
    group['group_sha256'] = digest
    for field, name in (('design_sha256','design.json'),('protocol_sha256','PROTOCOL.md'),
                        ('amendment_sha256','CONDUCTOR_AMENDMENT.json')):
        if group[field] != hashlib.sha256(read(HERE/name)).hexdigest():
            raise ValueError(f'Changed group {field}')
    for p, h in group['definitions_sha256'].items():
        if hashlib.sha256(read(p)).hexdigest() != h:
            raise ValueError(f'Changed group definition {p}')
    for entry in group['entries']:
        source=Path(entry['source_root']);seal_path=source/'ROOT.json';raw=read(seal_path)
        if hashlib.sha256(raw).hexdigest()!=entry['source_seal_sha256']:
            raise ValueError('Changed original arm root seal')
        for name,digest in json.loads(raw)['definitions_sha256'].items():
            if hashlib.sha256(read(source/name)).hexdigest()!=digest:
                raise ValueError('Changed original arm definition')
    return group


def pack(episodes, inputs):
    return {**{f:getattr(episodes, f).tolist() for f in ('y','n','v','m','cost')},
            'audit':episodes.audit, 'inputs':inputs}


def unpack(record):
    import numpy as np
    from .analyze import Episodes
    return Episodes(**{f:np.asarray(record[f]) for f in ('y','n','v','m','cost')}, audit=record['audit'])


def verify_pack(record, reader=None):
    for path, info in record['inputs'].items():
        data = reader.read(path) if reader else Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != info['sha256']:
            raise ValueError(f'Changed completed/reference input {path}')


def completed_rows(path):
    """Error-free accepted outcomes only. A transport failure is never an SR loss."""
    rows = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # conductor's torn-last-line resume convention
            if Journal.record_counts_as_done(row) and not row.get('error'):
                if type(row.get('success')) is not bool:
                    raise ValueError('Nonbinary outcome')
                rows[row['task_uid']] = row
    return rows


def prepare_groups(phase, generation='v1', tables=None):
    from .analyze import Reader, verify, checked_root, load
    from .prepare import frozen_tables
    from .conductor_reuse import load_completed
    reader = Reader(); design = verify(reader)
    if not re.fullmatch(r'[A-Za-z0-9_]+',generation):raise ValueError('Invalid generation')
    if phase == 'test':
        frozen_tables(tables)
        sources = reader.json(HERE/'test_roots.json')
        if {r['cell'] for r in sources} != set(CELLS):
            raise ValueError('Need all 32 frozen task arms')
    else:
        sources = [v[x] for v in design['cells'].values() for x in ('dev','fallback')]
    unique = {r['root']:r for r in sources}
    for rec in unique.values():
        checked_root(rec, reader)
    targets = [TRACE/f'r16_conductor_{phase}_{m}_k{k}_{generation}' for m in ('pi05','groot') for k in (1,2)]
    if any(p.exists() for p in targets):
        raise FileExistsError('Fresh conductor generation required; existing groups resume with driver/run')
    grouped = {(m,k):[] for m in ('pi05','groot') for k in (1,2)}
    # Read only completed B arms; test preparation never opens any A outcome.
    reused, references, inventory = {}, {}, []
    for rec in unique.values():
        root = Path(rec['root'])
        for row in reader.json(root/'arms.json'):
            c = row['r16_cell']; m,s,n,k = parts(c)
            fit = row['plugin_args'][row['plugin_args'].index('--os-fit-artifact')+1]
            entry = dict(row=copy.deepcopy(row), source_root=str(root), source_seal_sha256=sha(root/'ROOT.json'),
                         fit_sha256=rec['fit_sha256'][fit], yaml_sha256=sha(row['yaml']),
                         pairs=design['cells'][c]['pairs'] if phase=='dev' else [[t,i] for t in range(10) for i in range(50)])
            if phase=='test':
                entry['freeze'] = reader.json(root/'descent_table.json')['freeze']
            grouped[m,k].append(entry)
            if phase=='dev':
                run = root/'runs'/row['arm']; summary = run/'summary.json'
                available = summary.exists() and json.loads(summary.read_bytes()).get('complete') == 300
                inventory.append(dict(arm=row['arm'], root=str(root), completed=available))
                if available:
                    rr=Reader(); ep=load_completed(rec,row['arm'],entry['pairs'],'B',k,rr); rr.stable()
                    reused[row['arm']]=pack(ep, rr.inputs)
    if phase=='dev':
        cache = {}
        for c, source in design['cells'].items():
            ref = source['full_dev']; key=(ref['root'],ref['arm'])
            if key not in cache:
                rr=Reader(); ep=load(ref,ref['arm'],source['pairs'],'B',10 if parts(c)[0]=='pi05' else 8,rr)
                rr.stable(); cache[key]=pack(ep, rr.inputs)
            references[c]=cache[key]
    reader.stable()
    groups=[]
    for (m,k),entries in grouped.items():
        root=TRACE/f'r16_conductor_{phase}_{m}_k{k}_{generation}'
        definitions={}
        def store(path, data):
            atomic(path,data); definitions[str(path)]=sha(path)
        entries.sort(key=lambda e:e['row']['arm'])
        for entry in entries:
            row=entry['row']; name=row['arm']
            # Names and all policy arguments stay original. Only definition location changes.
            cfg=root/'config'/f'{name}.yaml'; store(cfg, reader.read(row['yaml'])); row['yaml']=str(cfg)
            manifest=root/'manifests'/f'{name}.json'; store(manifest,reader.read(row['manifest'])); row['manifest']=str(manifest)
            entry['yaml_sha256']=sha(cfg)
            if phase=='dev':
                contracts=reader.json(Path(entry['source_root'])/'dev_contracts.json')
                entry['pool_contract']=contracts[name]
            if name in reused:
                path=root/'reuse'/f'{name}.json'; store(path,json.dumps(reused[name],indent=2).encode()+b'\n')
                entry['reuse']=str(path)
        if phase=='dev':
            reference_path=root/'references.json'
            store(reference_path,json.dumps({c:r for c,r in references.items() if parts(c)[0]==m and parts(c)[3]==k},indent=2).encode()+b'\n')
        group=dict(schema='r16.conductor.group.v1', root=str(root), phase=phase, model=m, k=k,
                   entries=entries, definitions_sha256=definitions,
                   design_sha256=sha(HERE/'design.json'), protocol_sha256=sha(HERE/'PROTOCOL.md'),
                   amendment_sha256=sha(HERE/'CONDUCTOR_AMENDMENT.json'),
                   references=str(root/'references.json') if phase=='dev' else None,
                   recording=False, horizon=5, simulator_seed=SEED, yaml_id_is_original_arm=True,
                   intended_server_host='local', intended_worker_host='timan107', uniform_tables=False)
        from exp.offline_search.closed_loop.devset import pool_record
        group['pool_digests']={s:pool_record(s,'B' if phase=='dev' else 'A')[1]['per_task_digests']
                               for s in ('libero_10','libero_spatial')}
        group['group_sha256']=hashlib.sha256(canonical(group)).hexdigest()
        write_json(root/'group.json',group)
        write_json(root/'ROOT.json',dict(schema=group['schema'],group_sha256=group['group_sha256'],
                   definitions_sha256={**definitions,str(root/'group.json'):sha(root/'group.json')},
                   intended_host='local', recording=False, fleet_binding=False))
        groups.append(str(root/'group.json'))
    write_json(HERE/f'conductor_{phase}_groups.json',dict(groups=groups,completed_reuse=list(reused),inventory=inventory,
               test_a_outcome_reads=0,definitions_only=True))
    atomic(HERE/f'queue_conductor_{phase}.txt',''.join(f'@local {p}\n' for p in groups).encode())
    if phase=='test':
        if (HERE/'queue_test.txt').exists():atomic(HERE/'queue_test_single_method.txt',(HERE/'queue_test.txt').read_bytes())
        atomic(HERE/'queue_test.txt',''.join(f'@local {p}\n' for p in groups).encode())
    print(f'Prepared {len(groups)} resident groups, {len(reused)} completed B arms reused; no launches')


def prepare_smoke():
    index=json.loads((HERE/'conductor_dev_groups.json').read_bytes());records=[]
    for path in index['groups']:
        original=read_group(path)
        if original['k']!=1:continue
        group=copy.deepcopy(original);root=TRACE/f'r16_conductor_smoke_{group["model"]}_k1_v1'
        if root.exists():raise FileExistsError(root)
        entries=[e for e in group['entries'] if parts(e['row']['r16_cell'])[2]==50 and e['row']['r14_descent_level'] in ('c10','ir45')]
        definitions={}
        for entry in entries:
            entry.pop('reuse',None)
            for key,directory in (('yaml','config'),('manifest','manifests')):
                old=Path(entry['row'][key]);new=root/directory/old.name
                atomic(new,old.read_bytes());entry['row'][key]=str(new);definitions[str(new)]=sha(new)
            entry['yaml_sha256']=sha(entry['row']['yaml'])
        group.update(root=str(root),entries=entries,references=None,definitions_sha256=definitions)
        group.pop('group_sha256');group['group_sha256']=hashlib.sha256(canonical(group)).hexdigest()
        write_json(root/'group.json',group)
        write_json(root/'ROOT.json',dict(schema=group['schema'],group_sha256=group['group_sha256'],
            definitions_sha256={**definitions,str(root/'group.json'):sha(root/'group.json')},
            intended_host='local',recording=False,fleet_binding=False,smoke_only=True))
        records.append(str(root/'group.json'))
    write_json(HERE/'conductor_smoke_groups.json',dict(groups=records,episodes_per_arm=2,arms_per_model=4))
    atomic(HERE/'queue_conductor_smoke.txt',''.join(f'@local {p}\n' for p in records).encode())
    print('Prepared two isolated smoke groups, four methods x two episodes each; no launches')


def refresh_reuse(index):
    """Import arms completed by the old line since preparation, before dispatch.

    Refuses to amend a group that has ever started. Resumes retain exactly the
    prior sealed definition; only as-yet-unused groups may gain reuse inputs.
    """
    from .analyze import Reader,verify,load
    from .conductor_reuse import load_completed
    rr=Reader();design=verify(rr);rr.stable()
    catalog={r['root']:r for cell in design['cells'].values() for r in (cell['dev'],cell['fallback'])}
    groups=json.loads(Path(index).read_bytes());added=[]
    for path in groups['groups']:
        group=read_group(path);root=Path(group['root'])
        if group['phase']!='dev':raise ValueError('Reuse only applies to B development')
        if any((root/name).exists() for name in ('dispatch.jsonl','journal.jsonl','launch.json','runner_state.json')):
            continue # immutable started groups retain their accepted outcomes on resume
        changes=[]
        for e in group['entries']:
            if 'reuse' in e:continue
            row=e['row'];name=row['arm'];source=Path(e['source_root']);summary=source/'runs'/name/'summary.json'
            if not summary.exists() or json.loads(summary.read_bytes()).get('complete')!=300:continue
            reader=Reader();ep=load_completed(catalog[str(source)],name,e['pairs'],'B',group['k'],reader);reader.stable()
            changes.append((e,pack(ep,reader.inputs)))
        for e,record in changes:
            path_reuse=root/'reuse'/f'{e["row"]["arm"]}.json';write_json(path_reuse,record)
            e['reuse']=str(path_reuse);group['definitions_sha256'][str(path_reuse)]=sha(path_reuse);added.append(e['row']['arm'])
        if changes:
            group.pop('group_sha256');group['group_sha256']=hashlib.sha256(canonical(group)).hexdigest()
            write_json(root/'group.json',group)
            write_json(root/'ROOT.json',dict(schema=group['schema'],group_sha256=group['group_sha256'],
                       definitions_sha256={**group['definitions_sha256'],str(root/'group.json'):sha(root/'group.json')},
                       intended_host='local',recording=False,fleet_binding=False))
    groups['completed_reuse']=sorted(set(groups.get('completed_reuse',[])+added));write_json(index,groups)
    print(f'REUSE_REFRESHED newly_completed={len(added)}; started groups unchanged')


class QueueScheduler(EpisodeScheduler):
    """One global ready queue with cross-endpoint work stealing and persisted fences.

    Protected overrides are deliberately confined to R16; tests pin the current
    conductor scheduler interface. No default scheduler behavior changes.
    """
    def __init__(self, graph, endpoints, dispatch_path, *, adaptive=True):
        super().__init__(graph,eval_concurrency=len(graph.stages)+1)
        self.endpoints=endpoints  # key -> permitted suites; workers may connect to ANY compatible endpoint
        self.dispatch_path=Path(dispatch_path)
        self.enabled_fallbacks=set(); self.skipped=set(); self.adaptive=adaptive
        self.endpoint_load={key:0 for key in endpoints}
        self.dispatch_endpoint={}
        self.cursor=0
        self.upstreams={sid:graph.upstreams(sid) for sid in graph.stages}
        if self.dispatch_path.exists():
            for line in self.dispatch_path.read_text().splitlines():
                try: rec=json.loads(line)
                except ValueError: continue
                self._dispatch_gen[rec['task_uid']]=max(self._dispatch_gen.get(rec['task_uid'],0),rec['attempt'])

    def _deps_satisfied(self, sid):
        if self._rt[sid].stage.setup.get('kind')=='fallback' and sid not in self.enabled_fallbacks:
            return False
        return all(self._rt[u].state==StageState.DONE for u in self.upstreams[sid])

    def skip(self, sid):
        with self._lock:
            rt=self._rt[sid]
            if rt.dispatched:
                raise ValueError('Cannot skip dispatched work')
            rt.pending.clear(); rt.state=StageState.DONE; self.skipped.add(sid)

    def enable_fallback(self, sid):
        with self._lock: self.enabled_fallbacks.add(sid)

    def _refresh_activation(self):
        if self.adaptive and all(rt.state==StageState.DONE for rt in self._rt.values()
                                 if rt.stage.setup.get('kind')!='fallback'):
            for sid,rt in self._rt.items():
                if rt.stage.setup.get('kind')=='fallback' and sid not in self.enabled_fallbacks and rt.state==StageState.BLOCKED:
                    self.skip(sid)
        # All fits/bundles are already resident. No per-YAML activation cap and
        # no O(stages^2) active-count scans on each worker pull.
        for sid,rt in self._rt.items():
            if rt.state==StageState.BLOCKED and not rt.fatal and self._deps_satisfied(sid):
                rt.state=StageState.SETUP_PENDING

    def next_task(self, worker_server_key):
        # Affinity is a worker connection default, NOT ownership of an arm.
        # Dispatch chooses the least-busy compatible model endpoint globally.
        with self._lock:
            if worker_server_key not in self.endpoints:
                raise ValueError('Unknown worker endpoint')
            self._refresh_activation()
            ready=[sid for sid,rt in self._rt.items() if rt.state==StageState.READY and rt.pending]
            if not ready: return None
            ready.sort(); sid=ready[self.cursor % len(ready)]; self.cursor+=1
            rt=self._rt[sid]; uid=rt.pending.pop(0); ep=self._ep_index[uid][1]
            compatible=[key for key,suites in self.endpoints.items() if ep.experiment in suites]
            if not compatible: raise ValueError('No checkpoint-compatible server')
            key=min(compatible,key=lambda key:(self.endpoint_load[key],key))
            host,port=key.rsplit(':',1); self.endpoint_load[key]+=1; self.dispatch_endpoint[uid]=key
            rt.dispatched.add(uid); self._dispatch_ts[uid]=time.monotonic()
            gen=self._dispatch_gen.get(uid,0)+1; self._dispatch_gen[uid]=gen
            task=dataclasses.replace(ep,server_host=host,server_port=int(port),attempt=gen)
            self.dispatch_path.parent.mkdir(parents=True,exist_ok=True)
            with self.dispatch_path.open('a') as f:
                f.write(json.dumps(dict(task_uid=uid,arm=task.yaml_id,attempt=gen,server=key))+'\n')
                f.flush(); os.fsync(f.fileno())  # fence survives a hard interruption BEFORE handing work out
            return task

    def mark_result(self, task_uid, **kwargs):
        with self._lock:
            accepted=super().mark_result(task_uid,**kwargs)
            if accepted and task_uid in self.dispatch_endpoint:
                key=self.dispatch_endpoint.pop(task_uid); self.endpoint_load[key]-=1
            return accepted


def stage_id(arm, task):
    return f'{arm}/task{task:02d}'


class R16Strategy(ExperimentStrategy):
    def __init__(self, group, *, smoke_episodes=None):
        self.group=group; self.smoke_episodes=smoke_episodes; self.driver=None
        self.entries={e['row']['arm']:e for e in group['entries']}
        self.references=json.loads(Path(group['references']).read_bytes()) if group['references'] else {}

    def plan(self, yamls, server_assignment):
        graph=TaskGraph(); virtual=next(iter(server_assignment.values()))
        for name,entry in self.entries.items():
            row=entry['row']; level=row['r14_descent_level']
            for task in range(10):
                pairs=[(t,i) for t,i in entry['pairs'] if t==task]
                if self.smoke_episodes is not None:
                    pairs=pairs[:self.smoke_episodes] if task==0 else []
                episodes=[EpisodeTask(make_task_uid(name,'eval',t,i),name,'eval',row['suite'],t,i,i,
                                      virtual.host,virtual.port,name,
                                      extra=dict(num_trials_per_task=50,init_pool=row['init_pool'],
                                                 manifest_sha256=sha(row['manifest']))) for t,i in pairs]
                kind='fallback' if level=='c10' and self.group['phase']=='dev' else 'level'
                graph.add_stage(Stage(stage_id(name,task),name,'eval',virtual,episodes,
                                      setup=dict(kind=kind,cell=row['r16_cell'],task=task,level=level)))
        if self.group['phase']=='dev' and self.smoke_episodes is None:
            by_cell={}
            for name,e in self.entries.items():
                row=e['row']
                if row['r14_descent_level'] in ORDER:
                    by_cell.setdefault(row['r16_cell'],{})[row['r14_descent_level']]=name
            for cell,levels in by_cell.items():
                for task in range(10):
                    for a,b in zip(ORDER,ORDER[1:]):
                        graph.add_dependency(stage_id(levels[a],task),stage_id(levels[b],task))
        return graph

    def on_stage_complete(self, stage, ctl, ctx):
        if self.smoke_episodes is not None or self.group['phase']!='dev' or stage.setup['kind']=='fallback':
            return
        task=stage.setup['task']; cell=stage.setup['cell']; level=stage.setup['level']
        outcomes=completed_rows(self.driver._journal.path)
        cohort=[outcomes.get(ep.task_uid) for ep in stage.episodes]
        if len(cohort)!=30 or any(r is None or r.get('error') for r in cohort):
            raise ValueError('FatalEpisodeError: adaptive cohort requires 30 error-free accepted episodes')
        reference=self.references[cell]['y'][task]
        successes=sum(r['success'] for r in cohort); baseline=sum(reference)
        passing=100*(successes-baseline)>=-5*30
        write_json(Path(self.group['root'])/'gates'/f'{stage.yaml_id}_task{task}.json',dict(
            stage=stage.stage_id,cell=cell,task=task,level=level,n=30,successes=successes,
            full_c10_successes=baseline,passes=passing,tolerance_pp=5,group_sha256=self.group['group_sha256']))
        if not passing:
            index=ORDER.index(level)
            for name,e in self.entries.items():
                r=e['row']
                if r['r16_cell']==cell and r['r14_descent_level'] in ORDER[index+1:]:
                    self.driver._scheduler.skip(stage_id(name,task))
            if level=='ir45':
                m,s,n,k=parts(cell)
                fallback=next(name for name,e in self.entries.items() if e['row']['r14_descent_level']=='c10'
                              and parts(e['row']['r16_cell'])[1]==s and
                              ((parts(e['row']['r16_cell'])[2]==50)==(n==50)))
                self.driver._scheduler.enable_fallback(stage_id(fallback,task))


class R16Driver(ConductorDriver):
    def __init__(self, strategy, endpoints, **kwargs):
        self.result_lock=threading.RLock()
        self.fatal_error=None
        super().__init__(strategy,**kwargs)
        self._scheduler=QueueScheduler(self._graph,endpoints,Path(strategy.group['root'])/'dispatch.jsonl',
                                       adaptive=strategy.smoke_episodes is None and strategy.group['phase']=='dev')
        self._scheduler.mark_preexisting_done(set(completed_rows(self._journal.path)))
        strategy.driver=self
        if strategy.smoke_episodes is not None or strategy.group['phase']=='test':
            self._scheduler.enabled_fallbacks=set(self._graph.stages)

    def _record_pull(self,payload,peer):
        worker=super()._record_pull(payload,peer)
        group=self._strategy.group
        if 'pool_digests' not in group:return worker  # synthetic unit fixtures
        entry=self.worker_census.get(worker,{})
        probe=entry.get('probe') or {}
        expected_size=256 if group['model']=='groot' else 224
        if (probe.get('hostname','').split('.')[0]!='timan107' or probe.get('num_steps_wait')!=10 or
            probe.get('replan_steps')!=5 or probe.get('resize_size')!=expected_size or
            probe.get('max_steps')!={'libero_10':520,'libero_spatial':220} or
            probe.get('per_task_digests')!=group['pool_digests'] or entry.get('probe_conflict')):
            self.fatal_error='worker fleet/pool/horizon attestation mismatch'
            raise ValueError('FatalEpisodeError: worker fleet/pool/horizon attestation mismatch')
        return worker

    def handle_result(self, payload):
        with self.result_lock:
            super().handle_result(payload)

    def drive_stages_once(self):
        # The base scheduler announces completion before Journal.record. Prevent
        # a gate from observing the cohort in that narrow persistence window.
        with self.result_lock:
            super().drive_stages_once()


def preload(group, endpoints):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    for key,suites in endpoints.items():
        host,port=key.rsplit(':',1); ctl=WebsocketClientPolicy(host=host,port=int(port))
        try:
            for e in group['entries']:
                r=e['row']
                if r['suite'] in suites:
                    ctl.load_cache_config(r['yaml'],yaml_id=r['arm'],bundle_id=r['arm'])
        finally:
            ctl.close()


def endpoints_from_args(spec, model):
    endpoints={}
    for item in spec.split(','):
        key,sep,suite=item.partition('=')
        if ':' not in key or key in endpoints: raise ValueError('Invalid endpoint')
        suites=['libero_10','libero_spatial'] if not sep else [suite]
        if model=='groot' and (not sep or suite not in ('libero_10','libero_spatial')):
            raise ValueError('Each GR00T endpoint must declare its suite checkpoint')
        endpoints[key]=suites
    if set(s for v in endpoints.values() for s in v)!={'libero_10','libero_spatial'}:
        raise ValueError('Both suites must have endpoints')
    return endpoints


def import_reuse(group, journal_path):
    existing=completed_rows(journal_path); journal=Journal(journal_path)
    for e in group['entries']:
        if 'reuse' not in e: continue
        record=json.loads(Path(e['reuse']).read_bytes()); verify_pack(record)
        pairs=sorted(map(tuple,e['pairs'])); by_task={t:sorted(i for tt,i in pairs if tt==t) for t in range(10)}
        for t,i in pairs:
            uid=make_task_uid(e['row']['arm'],'eval',t,i)
            if uid in existing: continue
            y=bool(record['y'][t][by_task[t].index(i)])
            journal.record(task_uid=uid,yaml_id=e['row']['arm'],phase='eval',status='done' if y else 'failed',
                           success=y,accepted=True,attempt=0,run_id='reuse_'+sha(e['reuse'])[:12])


def run_driver(args):
    from exp.offline_search.rounds.r17.autoscale_1017.switch import enabled
    if enabled():
        os.environ.update(OPENPI_WORKER_MEMORY_GUARD='1',
                          OPENPI_WORKER_GOVERNOR='exp.offline_search.rounds.r17.autoscale_1017.legacy')
    group=read_group(args.group); root=Path(group['root'])
    from .analyze import Reader,verify
    rr=Reader();verify(rr);rr.stable()
    if args.smoke_episodes is not None and 'smoke' not in root.name:
        raise ValueError('Smoke requires an isolated smoke group')
    endpoints=endpoints_from_args(args.servers,group['model'])
    lock=root/'driver.lock'; lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=root/'journal.jsonl'; import_reuse(group,journal)
        if group['references']:
            for record in json.loads(Path(group['references']).read_bytes()).values(): verify_pack(record)
        preload(group,endpoints)
        strategy=R16Strategy(group,smoke_episodes=args.smoke_episodes)
        servers=[ServerEndpoint(key.rsplit(':',1)[0],int(key.rsplit(':',1)[1])) for key in endpoints]
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        def ctl(endpoint): return WebsocketClientPolicy(host=endpoint.host,port=endpoint.port)
        def writer(arm,rows):
            path=root/'per_step'/f'{arm}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as stream:
                for r in rows: stream.write(json.dumps(r)+'\n')
                stream.flush();os.fsync(stream.fileno())
        driver=R16Driver(strategy,endpoints,yaml_weights={a:len(e['pairs']) for a,e in strategy.entries.items()},
                         servers=servers,journal_path=str(journal),ctl_factory=ctl,
                         bind_host='0.0.0.0',bind_port=args.port,per_step_writer=writer)
        started=time.monotonic()
        stop=threading.Event()
        for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stop.set())
        write_json(root/'launch.json',dict(server_host='local',worker_host='timan107',STAGE1_ONLY=0,
                   endpoints=endpoints,group_sha256=group['group_sha256'],driver_port=args.port,
                   smoke_episodes=args.smoke_episodes))
        def stopped():
            proc=getattr(args,'agent_process',None)
            if proc is not None and proc.poll() is not None:stop.set()
            if driver.fatal_error:stop.set()
            if time.monotonic()-started>120 and not driver.worker_census:
                driver.fatal_error='No attested timan107 workers connected within 120 seconds';stop.set()
            return stop.is_set()
        driver.run(stopped)
        write_json(root/'progress.json',dict(progress=driver._scheduler.progress(),
                   skipped=sorted(driver._scheduler.skipped),worker_census=driver.worker_census,
                   interrupted=stop.is_set(),group_sha256=group['group_sha256']))
        failed=[sid for sid in driver._graph.stages if driver._scheduler.stage_state(sid)==StageState.FAILED]
        if failed: raise RuntimeError(f'Failed stages; resume after resolving error: {failed}')
        if driver.fatal_error:raise RuntimeError(driver.fatal_error)
        if not stop.is_set(): write_json(root/'DONE.json',dict(group_sha256=group['group_sha256']))


def run_agent(args):
    # Runs ONLY when coordinator invokes agent on timan107. Isolated island;
    # never overwrite the checkout used by the old coordinator line.
    import socket
    if socket.gethostname().split('.')[0]!='timan107': raise ValueError('Agent requires timan107')
    from openpi.conductor.agent import WorkerAgent,WorkerSpec
    pool='B' if args.phase=='dev' else 'A'
    specs=[]
    servers=[item.partition('=')[0] for item in args.servers.split(',')]
    if args.workers is None:
        from exp.offline_search.rounds.r17.autoscale_1017.defaults import physical_maximum
        from exp.offline_search.rounds.r17.autoscale_1017.switch import enabled
        args.workers = physical_maximum('timan107') if enabled() else 64
    if args.workers < 1: raise ValueError('Positive desired worker maximum required')
    from exp.offline_search.rounds.r17.autoscale_1017.switch import enabled
    if enabled():
        os.environ.update(OPENPI_WORKER_MEMORY_GUARD='1',
                          OPENPI_WORKER_GOVERNOR='exp.offline_search.rounds.r17.autoscale_1017.legacy',
                          OPENPI_AUTOSCALE_MODEL=args.model)
        os.environ.setdefault('OPENPI_AUTOSCALE_RUN',str(Path('/scratch/zixuans8/mw_r17/autoscale_legacy')/str(os.getpid())))
    lock_path=Path('/scratch/zixuans8/openpi_trace_h100/os_cl/locks/fleet.lock')
    # Shares the OLD fleet lock read-only in meaning: fail if that line still
    # owns the workers; never terminate it or update its code/queue.
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for i in range(args.workers):
            suffix='' if pool=='B' else '_apool'
            pools=','.join(f'{s}={REPO}/exp/common/data/db_init/libero/{s}{suffix}' for s in ('libero_10','libero_spatial'))
            specs.append(WorkerSpec(f'r16_{args.model}_{i:02d}',servers[i%len(servers)],str(i%8),
                         conda_env='/scratch/zixuans8/libero_sim',init_states_dir=pools,resize_size=256 if args.model=='groot' else 224,
                         replan_steps=5,seed=SEED,probe=True,env=dict(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(i%8))))
        agent=WorkerAgent(specs,args.driver_host,args.port)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,lambda *_:agent.stop())
        if args.lease_stdin:
            def lease():
                sys.stdin.buffer.read()
                agent.stop()
            threading.Thread(target=lease,daemon=True).start()
        if args.lease_file:
            if not args.lease_token:raise ValueError('Lease token required')
            def file_lease():
                while not agent._stop.wait(2):
                    if not lease_alive(args.lease_file,args.lease_token,args.lease_timeout):
                        agent.stop();return
            threading.Thread(target=file_lease,daemon=True).start()
        try: agent.run()
        finally: agent.stop()


def lease_alive(path,token,timeout,*,now=None):
    try:
        record=json.loads(Path(path).read_bytes())
        return record.get('token')==token and record.get('active') is True and (time.time() if now is None else now)-record['ts']<=timeout
    except (OSError,ValueError,KeyError):return False


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    for phase in ('dev','test'):
        p=sub.add_parser('prepare-'+phase);p.add_argument('--generation',default='v1')
        p.add_argument('--tables',type=Path,default=HERE/'dev_results')
    sub.add_parser('prepare-smoke')
    p=sub.add_parser('refresh-reuse');p.add_argument('--groups',type=Path,default=HERE/'conductor_dev_groups.json')
    p=sub.add_parser('driver');p.add_argument('--group',type=Path,required=True);p.add_argument('--servers',required=True)
    p.add_argument('--port',type=int,default=23179);p.add_argument('--smoke-episodes',type=int)
    p=sub.add_parser('agent');p.add_argument('--model',choices=('pi05','groot'),required=True)
    p.add_argument('--phase',choices=('dev','test'),required=True);p.add_argument('--servers',required=True)
    p.add_argument('--driver-host',required=True);p.add_argument('--port',type=int,default=23179);p.add_argument('--workers',type=int,default=None,help='Desired maximum; node memory governor chooses active count')
    p.add_argument('--lease-stdin',action='store_true')
    p.add_argument('--lease-file',type=Path);p.add_argument('--lease-token');p.add_argument('--lease-timeout',type=int,default=90)
    args=ap.parse_args()
    if args.action=='prepare-smoke':prepare_smoke()
    elif args.action=='refresh-reuse':refresh_reuse(args.groups)
    elif args.action.startswith('prepare-'): prepare_groups(args.action[8:],args.generation,args.tables)
    elif args.action=='driver':run_driver(args)
    else:run_agent(args)


if __name__=='__main__':main()
