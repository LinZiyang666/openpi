"""Explicit owner-run validation lanes. Ordinary queues remain globally OFF.

build is CPU/file-only and creates a NEW fa_asv_* directory. prepare/launch
require that directory's validation receipt and enable only that process and
its newly frozen payloads. This is never imported by an ordinary prepare path.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[4]


def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n')


def planner_overlay(source):
    """Add selection to the source's own planner, retaining its exact UID formula."""
    import tarfile
    with tarfile.open(Path(source)/'runtime.tar') as archive:
        native=archive.extractfile('exp/r17_integration/ops/formal_driver.py').read()
    return native + b'\n' + b"""
_native_validation_plan = plan
def plan(campaign):
    if not campaign.get('autoscale_validation'):return _native_validation_plan(campaign)
    reference = dict(campaign, run_name=campaign['semantic_campaign'])
    benchmark, model, tasks, endpoints = _native_validation_plan(reference)
    wanted = set(campaign['resume_task_uids'])
    selected = [task for task in tasks if task.task_uid in wanted]
    if len(wanted) != len(selected) or len(selected) != len(campaign['resume_task_uids']):
        raise ValueError('validation selection is not an exact native UID subset')
    return benchmark, model, selected, endpoints
"""


def build(source,root,path,episodes=256):
    source,root=Path(source).resolve(),Path(root).resolve()
    if not root.name.startswith('fa_asv_') or root==source or root.exists():
        raise ValueError('requires a new fa_asv_* output directory')
    if not 200<=episodes<=500:raise ValueError('validation requires 200..500 episodes')
    if path not in ('r17','r15'):raise ValueError('r17 or r15 required')
    if path=='r17':
        from .defaults import desired_maxima
        maxima=desired_maxima('metaworld',{'timan1':1,'timan107':1})
        c=read(source/'campaign.json');plan=read(source/'EPISODE_PLAN.json')
        if c.get('cache_study') or c.get('resume_from') or c.get('library'):
            raise ValueError('use a native pure R17 source; no continuation or library')
        tasks=plan['tasks'][:episodes]
        if len(tasks)!=episodes:raise ValueError('source has too few episodes')
        c.update(semantic_campaign=c.get('semantic_campaign') or c['run_name'],run_name=root.name,
            resume_task_uids=[t['task_uid'] for t in tasks],local_assets=str(root/'assets'),assets=str(root/'assets'),
            server_run=str(root/'servers'),filter_failed_source=str(root),prepare_only=True,driver_mode='group-commit',
            worker_nodes=dict(maxima),execution_worker_nodes=dict(maxima),
            workers=sum(maxima.values()),execution_workers=sum(maxima.values()))
        for key in list(c):
            if key.startswith(('autoscale','runtime_source')) or key in ('runtime_from','orchestration','driver_runtime','driver_module'):
                c.pop(key)
        names=('assets','runtime.tar','workers.tar','WORKER_DEPLOYMENT.json','RUNTIME_RELEASE.json')
    else:
        from .defaults import physical_maximum
        maxima={'timan107':physical_maximum('timan107','libero')}
        arms=read(source/'arms.json')
        if len(arms)!=1:raise ValueError('validation uses one native legacy arm')
        row=dict(arms[0]);manifest=Path(row['manifest']);pairs=read(manifest)[:episodes]
        if len(pairs)!=episodes or len(set(map(tuple,pairs)))!=episodes:raise ValueError('invalid manifest subset')
        tasks=[dict(task_uid=f"{row['arm']}:eval:{t}:{i}",task_id=t,episode_idx=i,orig_init_state_idx=i) for t,i in pairs]
        names=('config','ROOT.json','arms_in.json')
    root.mkdir(parents=True)
    from .compare import wait_io
    def copy(a,b):wait_io();return shutil.copy2(a,b)
    for name in names:
        p=source/name
        if p.is_dir():shutil.copytree(p,root/name,symlinks=True,copy_function=copy)
        elif p.is_file():copy(p,root/name)
    if path=='r17':
        c['autoscale_validation']=True
        from .trial_runtime import metadata
        info=metadata(c,root)
        write(root/'RUNTIME_RELEASE.json',{k:v for k,v in info.items() if k!='document'})
        (root/'validation_driver.py').write_bytes(planner_overlay(source))
        write(root/'campaign.json',c)
    else:
        # Path rebinding only. YAML, model, arm UID namespace and seeds stay byte-identical.
        for key in ('yaml','matrix'):row[key]=str(root/'config'/Path(row[key]).name)
        row['manifest']=str(root/'validation_manifest.json');row['episodes']=episodes
        write(root/'validation_manifest.json',pairs);write(root/'arms.json',[row])
    write(root/'EPISODE_PLAN.json',dict(tasks=tasks,expected_episodes=episodes))
    record=dict(schema='autoscale.owner_validation.v1',path=path,run=str(root),source=str(source),
        episodes=episodes,maximum=sum(maxima.values()),task_uid_sha256=hashlib.sha256('\n'.join(t['task_uid'] for t in tasks).encode()).hexdigest(),
        global_switch='off',scientific_inputs_unchanged=True)
    write(root/'AUTOSCALE_VALIDATION.json',record)
    return record


@contextmanager
def validation_process(root):
    """No global flip: explicit fa_asv_* process and its own frozen release only."""
    import builtins
    root=Path(root).resolve();record=read(root/'AUTOSCALE_VALIDATION.json')
    if not root.name.startswith('fa_asv_') or record.get('run')!=str(root) or record.get('schema')!='autoscale.owner_validation.v1':
        raise ValueError('not an isolated owner validation lane')
    switch=BASE/'ENABLED';original=switch.read_bytes()
    if original.strip()!=b'off':raise ValueError('validation runner requires production OFF')
    # Length remains four bytes: tar.add stat/read sizes must agree.
    native_open,native_io=builtins.open,io.open
    def wrap(native):
        def opened(file,mode='r',*a,**kw):
            if isinstance(file,(str,bytes,os.PathLike)) and Path(os.fsdecode(file)).absolute()==switch:
                if any(k in mode for k in 'wax+'):raise PermissionError('validation may never change global switch')
                return io.BytesIO(b'on\n\n') if 'b' in mode else io.StringIO('on\n\n')
            if record['path']=='r17' and isinstance(file,(str,bytes,os.PathLike)) and Path(os.fsdecode(file)).absolute()==REPO/'exp/r17_integration/ops/formal_driver.py':
                if any(k in mode for k in 'wax+'):raise PermissionError('validation cannot edit shared planner')
                data=(root/'validation_driver.py').read_bytes()
                return io.BytesIO(data) if 'b' in mode else io.StringIO(data.decode())
            return native(file,mode,*a,**kw)
        return opened
    builtins.open,io.open=wrap(native_open),wrap(native_io)
    try:yield record
    finally:
        builtins.open,io.open=native_open,native_io
        if switch.read_bytes()!=original:raise RuntimeError('global switch changed during validation')


def legacy_processes():
    """Read-only timan107 probe; do not overlap the acceptance trials with LIBERO."""
    import os
    from pathlib import Path
    rows = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            if int(path.name) == os.getpid() or path.stat().st_uid != os.getuid():
                continue
            argv = [v.decode(errors='replace') for v in (path/'cmdline').read_bytes().split(b'\0') if v]
            if ('examples.libero.worker_entry' in argv or
                    any(Path(v).name == 'run_gtp_subset.py' for v in argv) or
                    ('agent' in argv and any('exp.offline_search.rounds.r16' in v and 'conductor' in v for v in argv))):
                rows.append(dict(pid=int(path.name), argv=argv))
        except (OSError, ValueError):
            continue
    return rows


def require_legacy_idle():
    import inspect
    import subprocess
    code = inspect.getsource(legacy_processes) + '\nimport json;print(json.dumps(legacy_processes()))\n'
    result = subprocess.run(['tether', 'exec', '--safe', '--timeout', '90s', 'timan107', '--',
                             '/usr/bin/python3', '-B', '-S', '-c', code],
                            capture_output=True, text=True, timeout=95, check=True)
    active = json.loads(result.stdout.strip().splitlines()[-1])
    if active:
        raise RuntimeError('Acceptance waits for LIBERO legacy fleets on timan107: ' + json.dumps(active))


def run(action,root):
    with validation_process(root) as record:
        from . import defaults
        # A candidate must contain fairness/maxima and both bug fixes before a live trial.
        if not hasattr(defaults,'desired_maxima'):raise ValueError('combined fixes have not been installed; no remote call made')
        import inspect
        from openpi.conductor.worker import WorkerLoop
        from .worker import Observer
        if '__generation_' not in inspect.getsource(WorkerLoop.__init__) or '_factory=factory' not in inspect.getsource(Observer.__init__):
            raise ValueError('both incident fixes required before validation')
        if action=='inspect':return record
        if action=='launch':require_legacy_idle()
        if record['path']=='r17':
            if action=='prepare':
                from .formal_trial import prepare
                from . import release
                from unittest.mock import patch
                native_files=release.driver_files
                with patch.object(release,'driver_files',lambda:native_files()+['exp/r17_integration/ops/formal_driver.py']):
                    return prepare(root)
            from exp.offline_search.rounds.r17.filter_failed.launch import main
            return main(root)
        from unittest.mock import patch
        from . import legacy_release
        from exp.offline_search.closed_loop.ops.h100 import control
        from exp.offline_search.rounds.r17.filter_failed.legacy import rsync_transport
        from exp.offline_search.rounds.r17.filter_failed_knob.vla_shared import r15
        from exp.r17_integration.ops.formal_eval import transfer
        names=[r['arm'] for r in read(root/'arms.json')]
        from .defaults import physical_maximum
        with patch.dict(os.environ,WORKER_HOST='timan107',SERVER_HOST='h100',PORTS='23230,23231,23232',WPS='3',
                        OPENPI_DESIRED_MAXIMUM=str(physical_maximum('timan107','libero')),FFK_WORKER_FLEET_ROOT=str(root)), \
                patch.object(control,'run',rsync_transport(control.run,transfer)),r15(root):
            return control.sync(root,names,concurrent=True) if action=='prepare' else control.chain(root,names)


def audit(root, *, output=None, kernel_overrides=None):
    root=Path(root);record=read(root/'AUTOSCALE_VALIDATION.json')
    output=Path(output) if output is not None else root
    output.mkdir(parents=True,exist_ok=True)
    expected_uids=[t['task_uid'] for t in read(root/'EPISODE_PLAN.json')['tasks']]
    expected=Counter(expected_uids)
    plan_bound=(len(expected_uids)==record['episodes'] and
        hashlib.sha256('\n'.join(expected_uids).encode()).hexdigest()==record.get('task_uid_sha256'))
    journal=root/'driver/journal.jsonl' if record['path']=='r17' else next(root.glob('runs/*/client/journal.jsonl'))
    accepted=Counter();errors=0
    for line in journal.read_text().splitlines():
        if not line:continue
        row=json.loads(line)
        if row.get('accepted') is not False and row.get('status') in ('done','failed'):
            accepted[row['task_uid']]+=1;errors+=bool(row.get('error'))
    if record['path']=='r17':
        campaign=read(root/'campaign.json')
        nodes=campaign.get('execution_worker_nodes') or campaign['worker_nodes']
        scopes=[root/'workers'/node for node,count in nodes.items() if count>0]
    else:scopes=[journal.parent/'autoscale']
    events=[];samples=[];evidence=[]
    for scope in scopes:
        for name in ('AUTOSCALE_DECISIONS.jsonl',):
            p=scope/name
            if p.exists():events += [json.loads(s) for s in p.read_text().splitlines() if s]
        p=scope/'AUTOSCALE_SAMPLES.jsonl'
        if p.exists():samples += [json.loads(s) for s in p.read_text().splitlines() if s]
        p=scope/'AUTOSCALE_KERNEL.json'
        if p.exists():
            original=read(p)
            replacement=(kernel_overrides or {}).get(str(scope))
            if replacement is not None and any(replacement.get(k)!=original.get(k) for k in ('started','ended')):
                raise ValueError('replacement kernel evidence must cover the original lane window')
            evidence.append(replacement if replacement is not None else original)
    exits=sum(bool(e.get('reason')=='generation_exit' and e.get('signals',{}).get('complete')) for e in events)
    minimum=[e.get('signals',{}).get('minimum_available',e.get('signals',{}).get('available')) for e in events+samples]
    minimum=[v for v in minimum if v is not None]
    if record['path']=='r17':
        receipt=read(root/'RECEIPT.json') if (root/'RECEIPT.json').exists() else {}
        closed=receipt.get('status')=='PASS' and receipt.get('all_owned_closed') is True
        rejection=root/'driver/rejected_worker_probes.json'
        no_probe_conflicts=not rejection.exists() or not read(rejection)
    else:
        closed=bool(list((root/'state').glob('*.DONE')))
        no_probe_conflicts=True  # stock legacy attestation failures prevent DONE
    checks=dict(original_uid_plan_bound=plan_bound,native_closed_success=closed,no_probe_conflicts=no_probe_conflicts,every_uid_once=bool(expected) and accepted==expected and all(v==1 for v in expected.values()),
        accepted_errors_zero=errors==0,recycled_generations=exits>=2,
        above_critical=bool(minimum) and min(minimum)>=10*1024**3,
        zero_oom=bool(evidence) and len(evidence)==len(scopes) and all(e.get('status')=='PASS' and e.get('coverage') is True and not e.get('oom_lines') for e in evidence),
        scaling_reasons=bool(events) and all(e.get('reason') for e in events))
    import csv
    curve=output/'AUTOSCALE_ACTIVE.csv'
    with curve.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('time','node','active','target','draining','delta','reason','minimum_available'))
        for row in sorted(events+samples,key=lambda r:r.get('time',0)):
            counts=row.get('counts',{});signals=row.get('signals',{})
            writer.writerow((row.get('time'),row.get('node'),counts.get('active'),counts.get('target'),
                counts.get('draining'),row.get('delta'),row.get('reason'),signals.get('minimum_available',signals.get('available'))))
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,accepted=sum(accepted.values()),
        expected=record['episodes'],accepted_errors=errors,recycled_generations=exits,
        active_curve=str(curve),global_switch='off',kernel_evidence=evidence)
    write(output/'AUTOSCALE_VALIDATION_RESULT.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','inspect','prepare','launch','audit'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--path',choices=['r17','r15']);p.add_argument('--episodes',type=int,default=256)
    a=p.parse_args(argv)
    if a.action=='build':result=build(a.source,a.run,a.path,a.episodes)
    elif a.action=='audit':result=audit(a.run)
    else:result=run(a.action,a.run)
    print(json.dumps(result,default=str,indent=2))
    return 1 if isinstance(result,dict) and result.get('status')=='FAIL' else 0
if __name__=='__main__':raise SystemExit(main())
