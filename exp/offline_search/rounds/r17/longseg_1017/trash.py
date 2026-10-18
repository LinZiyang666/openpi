"""Manifest first, then same-filesystem renames of empirically invalid outcomes."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .audit import HERE,EVAL,PROTECTED,io_ready

PROJECTS=Path('/home/weiland/projects')
REPO=PROJECTS/'openpi'
TRASH=PROJECTS/'_trash_longseg_20261017'
BASES=(Path('/data/mw_finetune'),Path('/home/weiland/trace_runs'),PROJECTS,Path('/tmp'))
KEEP={'assets','study_inputs','control_inputs','.startup_inputs','runtime.tar','workers.tar',
      'driver_runtime.tar','evidence_runtime.tar','startup_assets.tar','assets.tar','study_inputs.tar',
      'campaign.json','STUDY.json','EPISODE_PLAN.json','FULL_EPISODE_PLAN.json','RUNTIME_RELEASE.json',
      'WORKER_DEPLOYMENT.json','LAUNCH_WORKER_DEPLOYMENT.json','PREPARED.json','LAUNCH.json',
      'ORCHESTRATION.json','EVIDENCE_ORCHESTRATION.json','CACHE_CONTROL.json','FFK_DEFINITION.json',
      'LOCAL_WORKER_ORCHESTRATION.json','MPS_CONTROL_ORCHESTRATION.json','OFFLINE_READY.json',
      'LOCAL_PLAN_SEAL.json','CACHE_RESUME.json','CONTINUATION_LOCAL_ONLY.json'}


def target(path):
    for base in BASES:
        if path.is_relative_to(base):return base/'_trash_longseg_20261017'/path.relative_to(base)
    raise ValueError('No authorized same-filesystem trash root: '+str(path))


def measure(path):
    # Do not walk large outcome trees: cold inode seeks can saturate this HDD
    # even when the byte rate and PSI look small. A rename preserves the whole
    # subtree atomically; shallow payload bytes are an explicitly lower bound.
    paths=[path]
    directory=path.is_dir() and not path.is_symlink()
    if directory:paths += list(path.iterdir())
    stats=[]
    for p in paths:
        if str(p).startswith('/data/'):
            io_ready();time.sleep(.1)
        stats.append(p.lstat())
    return dict(size_bytes=sum(s.st_size for s in stats if not __import__('stat').S_ISDIR(s.st_mode)),
                allocated_bytes=sum(s.st_blocks*512 for s in stats),entries=len(paths),
                recursive_size_exact=not directory,
                device=stats[0].st_dev,inode=stats[0].st_ino,mtime_ns=stats[0].st_mtime_ns)


def manifest():
    report=json.loads((HERE/'AUDIT.json').read_text());affected={r['run']:r for r in report['rows'] if r['affected']}
    assert not (set(affected)&PROTECTED)
    assert all(r['model'] not in ('pi05','cosmos') for r in affected.values())
    assert all(not r['changed_during_scan'] for r in affected.values()), 'An affected lane changed during the audit; do not move it'
    # Continuation selections derived from invalid outcomes are withdrawn too;
    # their definitions/archives are retained, never resumed against old SR.
    index=REPO/'exp/offline_search/rounds/r17/filter_failed/CONTINUATIONS.json'
    continued=json.loads(index.read_text())
    invalid_cont=[j for j in continued['jobs'] if Path(j['source']).name in affected or Path(j['run']).name in affected]
    sources=set(affected)|{Path(j['run']).name for j in invalid_cont}
    rows={};replacements={}
    def add(path,category,reason,stats=None):
        path=Path(path).absolute()
        if not path.exists() and not path.is_symlink():return
        if any(p==path or p in path.parents for p in rows):return
        if path.is_symlink() and path.is_dir():
            add(path.resolve(strict=True),category,reason+' (resolved outcome directory)',stats)
        rows[path]=dict(source_path=str(path),trash_path=str(target(path)),category=category,
                        reason=reason,cut_statistics=stats,**measure(path))
    for name in sorted(sources):
        io_ready()
        root=EVAL/name
        if not root.is_dir():continue
        c=json.loads((root/'campaign.json').read_text())
        if c['model'] in ('cosmos','pi05'):raise ValueError('Protected model '+name)
        stats=affected.get(name,dict(affected=True,reason='continuation of affected lane'))
        category='eval_'+c['model']+'_'+c['domain']
        for p in root.iterdir():
            if p.name in KEEP or p.name.endswith(('_DISK_ADMISSION.json','_MEMORY_ADMISSION.json','_GPU_ADMISSION.json')):continue
            if p.name in ('driver','mps','mps_control','evidence_transport'):
                add(p,category,'Outcome of midpoint-cut campaign',stats)
            elif p.name=='workers':
                add(p,category,'Worker outcome directory; archived worker release retained',stats)
            elif p.name=='servers':
                for f in p.rglob('*'):
                    if not f.is_file() or f.name=='ready.json' or 'ADMISSION' in f.name or 'calibrat' in f.name.lower():continue
                    if f.suffix in ('.log','.jsonl') or f.name.endswith(('.owner.json','.pid','.job.json')):
                        add(f,category,'Server/worker outcome; startup identities retained',stats)
            elif (p.suffix in ('.log','.jsonl') or p.name in ('servers.tar','SUMMARY.json','RECEIPT.json',
                    'BACKEND_EVIDENCE.json','EVIDENCE_FINALIZATION.json','SEALED_EVIDENCE_RECEIPT.json',
                    'GPU_RELEASED.json','BUNDLE_ACK.json','STUDY_BUNDLE_ACKS.json','PROXY_METADATA.json',
                    'WORKER_NODE_STATE.json') or p.name.startswith('WORKER_NODE_STATE.')):
                add(p,category,'Campaign outcome/analysis receipt',stats)
        time.sleep(1)
    def whole(path,reason):add(path,'derived',reason)
    knob=REPO/'exp/offline_search/rounds/r17/knob'
    ffk=REPO/'exp/offline_search/rounds/r17/filter_failed_knob'
    for model in ('groot','dp','fastwam'):
        for mode in ('full','reduced'):
            whole(knob/'mt50'/f'{model}_{mode}','MT50 descent derived from affected lane')
            whole(ffk/'results'/f'{model}_mt50_{mode}','Filtered MT50 descent derived from affected lane')
    for path in (knob/'full_system_reduced').glob('*'):
        if any(t in path.name for t in ('dp','fw')):whole(path,'Knob-off/IR table derived from affected LIBERO lane')
    for model in ('dp','fastwam'):
        for path in (ffk/'formal_a_1016/off_decisions').glob(model+'_*'):
            whole(path,'Off classification must use fixed-segment outcomes')
    # Mixed documents: move the complete original, then retain unaffected rows
    # verbatim as JSON values in a replacement at the original address.
    def invalid_row(r,context=''):
        if not isinstance(r,dict):return False
        if any(t in context for t in ('mt50_cells.json','CACHE_RESULTS.json','/cells.json')) and r.get('group') not in ('ch','chr','chg','chrg'):
            return False  # native-length and ungated cache-variant rows remain valid
        model=r.get('model','');model={'DP':'dp','GR00T':'groot','Fast-WAM':'fastwam'}.get(model,model)
        bench=r.get('benchmark',r.get('domain','mt50' if 'mt50' in context else 'libero'))
        return model in ('dp','fastwam') or model=='groot' and bench in ('mt50','metaworld','metaworld_mt50')
    def mixed(path):
        if not path.is_file():return
        doc=json.loads(path.read_text());context=str(path)
        def walk(obj):
            if isinstance(obj,list):return [walk(r) for r in obj if not invalid_row(r,context)]
            if isinstance(obj,dict):return {k:walk(v) for k,v in obj.items() if not invalid_row(v,context)}
            return obj
        kept=walk(doc)
        if kept!=doc:
            add(path,'mixed_derived','Withdraw affected rows; retain unrelated models/benchmarks')
            replacements[str(path)]=kept
    for path in (ffk/'formal_a_1016/RESULTS.json',knob/'full_system_ir.json',
                 REPO/'exp/offline_search/rounds/r17/cache/CACHE_RESULTS.json'):
        mixed(path)
    for root in (PROJECTS/'openpi_ext/scratch/r17_ops/livepage/build_1015',
                 PROJECTS/'openpi_ext/scratch/r17_ops/livepage/live_1015/blobs'):
        for name in ('mt50_knob.json','mt50_cells.json','knob.json'):
            mixed(root/name)
    live=PROJECTS/'openpi_ext/scratch/r17_ops/livepage/live_1015'
    for dataset in ('cells','knob','mt50_cells','mt50_knob'):
        entry=live/'datasets'/(dataset+'.json')
        if not entry.exists():continue
        ref=json.loads(entry.read_text()).get('source',{}).get('url','').split('/')[-1]
        blob=live/'blobs'/(ref+'.json')
        if not blob.exists():continue
        data=json.loads(blob.read_text())
        kept=[r for r in data if not invalid_row(r,'/'+dataset+'.json')]
        if data!=kept:
            add(blob,'mixed_derived','Published website data; remove only invalid rows')
            replacements[str(blob)]=kept
    cache_report=REPO/'exp/offline_search/rounds/r17/cache/CACHE_RESULTS.json'
    regenerate_cache=str(cache_report) in replacements
    if regenerate_cache:
        for extension in ('md','csv'):
            add(cache_report.with_suffix('.'+extension),'mixed_derived','Regenerate from retained cache-study rows')
        # These are cache-variant outcome summaries, not baseline/pure results.
        for filename in ('FASTWAM_K3_CELLS.md',):
            whole(REPO/'exp/offline_search/rounds/r17/cache/mt50'/filename,'Contains affected full-system cache-variant measurements')
    # Affected entries cannot remain active continuation tips. Unaffected
    # entries (including the Cosmos continuation) are kept as identical values.
    if invalid_cont:
        add(index,'continuation_registry','Remove only invalid continuation entries')
        replacements[str(index)]=dict(continued,jobs=[j for j in continued['jobs'] if j not in invalid_cont])
    entries=sorted(rows.values(),key=lambda r:r['source_path'])
    result=dict(schema='r17.longseg.withdrawal.v1',created_at=datetime.now(timezone.utc).isoformat(),
        affected_runs=sorted(affected),continuations_withdrawn=invalid_cont,entries=entries,
        replacements=replacements,regenerate_cache=regenerate_cache,protected_runs=sorted(PROTECTED),
        keep='Definitions, plans, lifecycle launch claims, frozen releases, inputs, ready/admission/calibration artifacts and pure references')
    TRASH.mkdir(parents=True,exist_ok=True)
    path=TRASH/'MANIFEST.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n')
    (TRASH/'DRY_RUN.tsv').write_text('source\ttrash\tbytes\tcategory\tcuts\n'+''.join(
        f'{r["source_path"]}\t{r["trash_path"]}\t{r["size_bytes"]}\t{r["category"]}\t{(r["cut_statistics"] or {}).get("cuts",0)}\n' for r in entries))
    print(json.dumps(dict(entries=len(entries),GiB=sum(r['size_bytes'] for r in entries)/2**30,manifest=str(path))))


def move():
    doc=json.loads((TRASH/'MANIFEST.json').read_text())
    for r in doc['entries']:
        if r['source_path'].startswith('/data/'):io_ready()
        if r['source_path'].startswith('/data/'):time.sleep(.1)
        source=Path(r['source_path']);dest=Path(r['trash_path'])
        if dest.exists() or dest.is_symlink():raise FileExistsError(dest)
        s=source.lstat()
        assert (s.st_dev,s.st_ino)==(r['device'],r['inode']),source
        dest.parent.mkdir(parents=True,exist_ok=True)
        assert dest.parent.stat().st_dev==s.st_dev,(source,dest)
    journal=TRASH/'MOVE_JOURNAL.jsonl'
    for r in doc['entries']:
        source=Path(r['source_path']);dest=Path(r['trash_path'])
        if str(source).startswith('/data/'):io_ready();time.sleep(.1)
        subprocess.run(['mv','--no-copy','--no-clobber','-T',str(source),str(dest)],check=True)
        with journal.open('a') as f:f.write(json.dumps(dict(source=str(source),trash=str(dest)))+'\n')
    for path,data in doc['replacements'].items():Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    if doc.get('regenerate_cache'):
        from exp.offline_search.rounds.r17.cache.analyze_cache import write_report,claim_table
        path=REPO/'exp/offline_search/rounds/r17/cache/CACHE_RESULTS.json'
        report=json.loads(path.read_text());report['claims']=claim_table(report['rows'])
        report['withdrawal_manifest']=str(TRASH/'MANIFEST.json');write_report(path.parent,report)
    counts=Counter();sizes=Counter()
    for r in doc['entries']:
        p=Path(r['trash_path']);s=p.lstat();assert s.st_ino==r['inode'] and s.st_dev==r['device']
        assert not Path(r['source_path']).exists() or r['source_path'] in doc['replacements'] or doc.get('regenerate_cache') and Path(r['source_path']).name in ('CACHE_RESULTS.csv','CACHE_RESULTS.md')
        counts[r['category']]+=1;sizes[r['category']]+=r['size_bytes']
    out=dict(status='PASS',entries=len(doc['entries']),bytes=sum(sizes.values()),
             categories={k:dict(count=counts[k],bytes=sizes[k],GiB=sizes[k]/2**30) for k in counts},
             protected_runs_untouched=True)
    (TRASH/'VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('manifest','move'))
    manifest() if p.parse_args().action=='manifest' else move()
