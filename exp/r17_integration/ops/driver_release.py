"""Separate, immutable orchestration pin; qualified model/worker releases stay sealed."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import tarfile

from .immutable_release import archive_manifest, identity, install, sha, verify
from .formal_process import write_json

OVERLAY=('exp/r17_integration/ops/formal_driver_fast.py','exp/r17_integration/ops/group_commit.py')
CACHE_OVERLAY=('exp/r17_integration/ops/cache_study.py','exp/r17_integration/ops/cache_resume.py',
               'exp/r17_integration/ops/cache_driver_legacy.py')
LIVE_OVERLAY=tuple('exp/r17_integration/ops/'+name+'.py' for name in
    ('formal_driver','live_fleets','live_fleet_driver','live_fleet_worker'))
# Every driver runtime, frozen ones included: the central scheduler's BatchClient and its conductor hook.
HOOK=('src/openpi/conductor/driver.py','src/openpi/conductor/worker.py','exp/r17_integration/qualification/driver.py')


def driver_files(files,mode):
    names={n for n in files if n.startswith(('src/openpi/conductor/','exp/r17_integration/conductor/'))}
    names.update(n for n in ('exp/r17_integration/ops/formal_driver.py','exp/r17_integration/ops/formal_eval.py',
        'exp/r17_integration/qualification/driver.py','exp/r17_integration/ops/cache_study.py',
        'exp/r17_integration/ops/cache_resume.py','exp/r17_integration/ops/cache_driver_legacy.py') if n in files)
    if mode=='group-commit':names.update(OVERLAY)
    names.update(n for n in LIVE_OVERLAY if n in files)
    from exp.offline_search.rounds.r17.scheduler.defaults import PACKAGE
    names.update(n for n in files if n.startswith(PACKAGE+'/'))
    return {n:files[n] for n in sorted(names)}


def pin(files,mode):
    return dict(mode=mode,module='exp.r17_integration.ops.formal_driver_fast' if mode=='group-commit' else
        'exp.r17_integration.ops.formal_driver',files=driver_files(files,mode))


def prepare(campaign,directory,checkout):
    directory,checkout=Path(directory),Path(checkout)
    mode=campaign.get('driver_mode','legacy')
    if mode not in ('legacy','group-commit'):raise ValueError('unknown driver mode')
    if mode=='group-commit' and (campaign.get('library') or campaign.get('native_parity_only')):
        raise ValueError('grouped driver supports pure inference only')
    archive=directory/'runtime.tar';base=archive_manifest(archive)
    baseline=None
    source=campaign.get('runtime_from') or campaign.get('resume_from')
    if source:
        old=json.loads((Path(source)/'campaign.json').read_text())
        if old.get('orchestration'):
            validate(old,source)
            baseline=old['orchestration']['selected']
    if baseline is None:
        baseline=pin(base['files'],'legacy')
        baseline['identity']=identity(baseline)
    overlay=((*OVERLAY,) if mode=='group-commit' else ())
    if campaign.get('cache_study'):overlay=(*overlay,*CACHE_OVERLAY)
    if campaign.get('live_fleets'):overlay=(*overlay,*LIVE_OVERLAY)
    from exp.offline_search.rounds.r17.scheduler.defaults import files as scheduler_files
    overlay=(*overlay,*scheduler_files(),*HOOK)
    selected_archive=directory/'driver_runtime.tar'
    with tarfile.open(archive) as original,tarfile.open(selected_archive,'w') as target:
        content={m.name:original.extractfile(m).read() for m in original.getmembers()}
        content.update({name:(checkout/name).read_bytes() for name in overlay})
        for name,data in sorted(content.items()):
            member=tarfile.TarInfo(name);member.size=len(data);member.mode=0o444
            target.addfile(member,io.BytesIO(data))
    selected=archive_manifest(selected_archive)
    delta={name for name in set(base['files'])|set(selected['files'])
           if base['files'].get(name)!=selected['files'].get(name)}
    root='/data/mw_finetune/qualification/formal_driver_'+identity(selected)
    install(selected_archive,root,sha(selected_archive),identity(selected))
    current=pin(selected['files'],mode);current['identity']=identity(current)
    document=dict(schema='r17.orchestration_pin.v1',baseline=baseline,selected=current,
        source_runtime_identity=campaign['runtime_identity'],driver_runtime=root,
        driver_runtime_identity=identity(selected),overlay_files=sorted(delta),
        model_code_unchanged=True,worker_payload_unchanged=True,
        durability=('evidence fsync, terminal journal fsync, then scheduler acceptance; '
                    'nonterminal diagnostics group committed; resume requires fresh closed-source segment'
                    if mode=='group-commit' else 'legacy per-row/per-dispatch synchronous writes'))
    campaign.update(driver_runtime=root,driver_module=current['module'],driver_mode=mode,orchestration=document)
    write_json(directory/'ORCHESTRATION.json',document)


def validate(campaign,directory):
    directory=Path(directory)
    # Rebuilt source templates (longseg_1017) carry the pin only in campaign.json.
    path=directory/'ORCHESTRATION.json'
    document=json.loads(path.read_text()) if path.exists() else campaign['orchestration']
    if not verify(document['driver_runtime'],'RUNTIME_RELEASE.json',document['driver_runtime_identity'])['present']:
        raise ValueError('orchestration release missing')


def planned_tasks(campaign,directory):
    """Endpoint rebinds must also use the recorded planner, never checkout code."""
    from . import formal_eval as launch
    validate(campaign,directory)
    code=('import json; from pathlib import Path; from dataclasses import asdict; '
          'from exp.r17_integration.ops.formal_driver import plan; '
          'c=json.loads(Path('+repr(str(Path(directory)/'campaign.json'))+').read_text()); '
          '_,_,tasks,_=plan(c); print(json.dumps([asdict(t) for t in tasks]))')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             PYTHONPATH=launch.pythonpath(campaign['driver_runtime']))
    return json.loads(launch.execute([launch.CONTROL_PYTHON,"-c",code],seconds=1800,
                                    env=env,cwd=campaign['driver_runtime']))


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def preserve_resume_segment(directory):
    """Retain a durable raw segment before the stock merger replaces its files."""
    directory=Path(directory);backup=directory/'durable_segment'
    backup.mkdir(exist_ok=False)
    pins={}
    for name in ('journal.jsonl','attempts.jsonl','per_step.jsonl','DRIVER_RECEIPT.json'):
        with (directory/'driver'/name).open('rb') as source,(backup/name).open('xb') as target:
            shutil.copyfileobj(source,target);target.flush();os.fsync(target.fileno())
        pins[name]=sha(backup/name)
    proof=dict(schema='r17.durable_resume_segment.v1',root=str(backup),files=pins)
    write_json(backup/'MANIFEST.json',proof)
    with (backup/'MANIFEST.json').open('rb') as stream:os.fsync(stream.fileno())
    sync_directory(backup);sync_directory(directory)
    return proof


def sync_resume_merge(directory):
    """Publish a composite only after its evidence, plan and receipt are durable."""
    directory=Path(directory)
    for name in ('driver/journal.jsonl','driver/attempts.jsonl','driver/per_step.jsonl',
                 'driver/DRIVER_RECEIPT.json','EPISODE_PLAN.json','RESUME_MERGE.json'):
        with (directory/name).open('rb') as stream:os.fsync(stream.fileno())
    sync_directory(directory/'driver');sync_directory(directory)
