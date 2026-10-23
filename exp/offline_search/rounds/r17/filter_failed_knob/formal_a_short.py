"""CPU-only formal-A naming migration and path/startup proofs. Never stages."""
from __future__ import annotations
import argparse
import ast
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from unittest.mock import patch

from . import formal_a as f, formal_a_queue as q, formal_a_names as n
from .common import read,dump,sha

AUDIT=f.ROOT/'short_names_1017'
FAILED='ffk1015_cosmos_libero_reduced_formal_A_original_levels'


def snapshot():
    AUDIT.mkdir(parents=True,exist_ok=True)
    if (AUDIT/'QUEUE_BEFORE.json').exists():return read(AUDIT/'QUEUE_BEFORE.json')
    index=read(q.INDEX);dump(AUDIT/'QUEUE_BEFORE.json',index)
    commands={str(p):sha(p) for base in (f.HERE/'commands',f.ROOT/'commands') for p in base.glob('*.sh')}
    ops=Path('/home/weiland/projects/openpi_ext/scratch/r17_ops/ff')
    commands.update({str(p):sha(p) for p in [*ops.glob('run_*.sh'),ops/'chain_ffk.run.sh']})
    inputs={}
    for u in index['units']:
        for j in u['jobs']:
            root=Path(j['run'])
            for name in ('ROOT.json','arms.json','arms_in.json','eval500.json','group.json','EPISODE_PLAN.json'):
                if (root/name).exists():inputs[str(root/name)]=sha(root/name)
    dump(AUDIT/'BEFORE.json',dict(commands=commands,inputs=inputs))
    return index


def mapping(index):
    aliases={};roots={};units=[]
    for u in index['units']:
        old=u['name']
        short=('fa_pi_f_p10' if 'pure_l10' in old else 'fa_pi_f_ps') if u['stage']=='pure' else n.unit(u['model'],u['mode'],u['arm'],u['stage'])
        aliases[old]=short;aliases[old+'_worker_auto']=short
        pipeline='R17' if u['model'] in f.TEST_SOURCE else 'R15' if u['mode']=='full' else 'R16'
        runs=[]
        if pipeline=='R15' and u['stage']!='pure':
            for suite,size in q.cells(u['model']):
                cell=f'{u["model"]}_{"l10" if suite=="libero_10" else "spatial"}_{size}'
                old_run=old+'_'+cell;new_run=n.cell_name(short,cell);aliases[old_run]=new_run
                runs.append(dict(old=old_run,name=new_run,root=str(f.EVAL/new_run),cell=cell))
        else:runs=[dict(old=old,name=short,root=str(f.EVAL/short))]
        for run in runs:
            previous=(f.EVAL/run['old']) if pipeline=='R17' else f.ROOT/'queue'/old/run['old']
            run['old_root']=str(previous);roots[str(previous)]=run['root']
        for j in u['jobs']:
            short_run=aliases[j['name']];roots[j['run']]=str(f.EVAL/short_run)
        units.append(dict(old=old,name=short,pipeline=pipeline,gpu=u['gpu'],status=u['status'],runs=runs))
    retired='ffk1015_pi05_libero_full_formal_A_pure_l10'
    aliases[retired]='fa_pi_f_p10'
    roots[str(f.ROOT/'queue'/retired/retired)]=str(f.EVAL/'fa_pi_f_p10')
    return dict(schema='formal_A.short_names.v1',names=aliases,runs=roots,units=units,
        tokens=dict(models=n.MODELS,modes=n.MODES,arms=n.ARMS,stages=n.STAGES),
        pure_l10_policy='replacement_only_do_not_merge',legacy_arm_identifiers='retained on R15/R16 plans')


def clone_r17(old,short):
    """Relocate output only, preserving every YAML byte and every planned UID."""
    from .build import cpu_plan
    src=Path(old['run']);root=f.EVAL/short
    from exp.r17_integration.ops.immutable_release import nvme_dir
    nvme_dir(root)  # real dir on /nvme, /data path a symlink
    shutil.copytree(src/'assets',root/'assets')
    for file in ('runtime.tar','workers.tar','RUNTIME_RELEASE.json','STUDY.json','EPISODE_PLAN.json'):
        shutil.copyfile(src/file,root/file)
    source=src/'.startup_inputs/campaign.json'
    if not source.exists():
        saved=sorted(src.glob('.startup_inputs.failed_*/campaign.json'))
        if saved:source=saved[-1]
    c=read(source if source.exists() else src/'campaign.json')
    for key in ('orchestration','runtime_source_pins','server_campaign'):c.pop(key,None)
    # Serving assets retain their existing physical address, so YAML hashes,
    # bundle identities, task UIDs and all RNG inputs remain exactly unchanged.
    # Native prepare snapshots the identical local mirror to that address.
    c.update(run_name=short,local_assets=str(root/'assets'),server_run=str(root/'servers'),
             cache_study=str(root/'STUDY.json'),prepare_only=True)
    dump(root/'campaign.json',c)
    _,model,tasks,_=cpu_plan(c)
    plan=read(root/'EPISODE_PLAN.json')
    assert [asdict(t) for t in tasks]==plan['tasks']
    assert model==plan['model']
    job=dict(old,name=short,run=str(root),renamed_from=str(src))
    dump(root/'FFK_DEFINITION.json',job)
    dump(root/'OFFLINE_READY.json',dict(status='READY_FOR_COORDINATOR_STAGE',episodes=old['episodes'],
         native_runtime_unchanged=True,worker_archive_unchanged=True,plan_byte_identical=True))
    return job


def clone_vla(old,short):
    from exp.r17_integration.ops.immutable_release import nvme_dir
    src=Path(old['run']);root=nvme_dir(f.EVAL/short)  # real dir on /nvme, /data path a symlink
    if old['pipeline']=='R15':
        manifest=root/'eval500.json';manifest.write_bytes((src/'eval500.json').read_bytes())
        specs=copy.deepcopy(read(src/'arms_in.json'))
        for spec in specs:spec['manifest']=str(manifest)
        rows=read(src/'arms.json');cell=read(src/'ROOT.json')['cell']
        f.vf.p.emit(root,cell,specs,[r['r14_descent_level'] for r in rows],pool='A',final_root=root)
        assert [r['arm'] for r in rows]==[r['arm'] for r in read(root/'arms.json')]
        for spec,before in zip(specs,read(src/'arms_in.json')):
            assert {k:v for k,v in spec.items() if k!='manifest'}=={k:v for k,v in before.items() if k!='manifest'}
    else:
        from exp.offline_search.rounds.r16 import conductor as c,analyze as a
        group=read(src/'group.json');group['root']=str(root);group.pop('group_sha256')
        group['group_sha256']=a.sha(c.canonical(group));dump(root/'group.json',group)
        dump(root/'ROOT.json',dict(group_sha256=group['group_sha256'],definitions_sha256=group['definitions_sha256']))
        assert group['entries']==read(src/'group.json')['entries']
    return dict(old,name=short,run=str(root),worker_host_failover=True,renamed_from=str(src))


def migrate():
    before=snapshot()
    if (AUDIT/'MIGRATED.json').exists():raise ValueError('Already migrated')
    failed=f.EVAL/FAILED
    if (failed/'LAUNCH.json').exists() or (failed/'PREPARED.json').exists():
        raise ValueError('First reset the documented failed MPS startup to UNSTAGED')
    m=mapping(before);dump(n.MAP,m)
    active=[]
    for old in before['units']:
        u=copy.deepcopy(old);u['name']=m['names'][old['name']]
        u['renamed_from']=old['name'];u['dependencies']=[m['names'].get(d,d) for d in u.get('dependencies',[])]
        if u['model'] not in f.TEST_SOURCE:u['worker_host_failover']=True
        jobs=[]
        for j in old['jobs']:
            short=m['names'][j['name']]
            job=clone_r17(j,short) if j['pipeline']=='R17' else clone_vla(j,short)
            jobs.append(job);print('SHORT_DEFINED',short,job['episodes'],flush=True)
        u['jobs']=jobs;active.append(u);dump(f.ROOT/'queue'/u['name']/'UNIT.json',u)
    index=dict(before,units=active,short_names=str(n.MAP));dump(q.INDEX,index)
    overrides=read(f.ROOT/'PURE_A_OVERRIDES.json')
    for key,value in overrides.items():value['root']=m['runs'].get(value['root'],value['root'])
    dump(f.ROOT/'PURE_A_OVERRIDES.json',overrides)
    q.publish()
    for gpu,file in (('H100','H100_ORDER.txt'),('RTX 4090','4090_ORDER.txt')):
        shutil.copyfile(f.ROOT/'commands'/file,f.HERE/'commands'/('formal_A_short_'+file))
    pins=read(AUDIT/'BEFORE.json')
    dump(AUDIT/'MIGRATED.json',dict(units=len(active),native_runs=sum(len(u['jobs']) for u in active),
         old_commands_unchanged=len(pins['commands']),old_plans_unchanged=len(pins['inputs']),
         hashes_checked=False,staged=False,launched=False))


def paths():
    from exp.r17_integration.ops.plan import unix_socket_path
    code=('import json;from pathlib import Path;rows=[]\n'
          'for p in Path("/proc/driver/nvidia/gpus").glob("*/information"):\n'
          ' d={k.strip():v.strip() for k,v in (s.split(":",1) for s in p.read_text().splitlines() if ":" in s)}\n'
          ' rows.append(dict(uuid=d["GPU UUID"],minor=int(d["Device Minor"]),model=d["Model"]))\n'
          'print(json.dumps(rows))')
    observed={}
    for gpu,node in (('H100','h100'),('RTX 4090','local')):
        argv=['/usr/bin/python3','-B','-S','-c',code]
        if node!='local':argv=['tether','exec','--safe','--timeout','30s',node,'--',*argv]
        observed[gpu]=json.loads(subprocess.run(argv,capture_output=True,text=True,check=True,timeout=35).stdout)
        assert len(observed[gpu])==1 and observed[gpu][0]['minor']==0
    uuids={gpu:cards[0]['uuid'] for gpu,cards in observed.items()}
    dump(AUDIT/'GPU_IDENTITY.json',dict(observed=observed,read_only_procfs=True,gpu_queries=0))
    m=read(n.MAP);rows=[]
    before={u['name']:u for u in read(AUDIT/'QUEUE_BEFORE.json')['units']}
    for unit in m['units']:
        old=before[unit['old']]
        source=next(j for j in q.templates() if (j['model'],j['mode'],j['arm'])==
            (old['model'],old['mode'],old['arm'])) if old['stage']!='pure' else old['jobs'][0]
        active=next(u for u in read(q.INDEX)['units'] if u['name']==unit['name'])
        for run in unit['runs']:
            root=Path(run['root']);gpu=unit['gpu'];uuid=uuids[gpu]
            if unit['pipeline']=='R17':
                archive=Path(active['jobs'][0]['run'] if active['jobs'] else source['run'])/'runtime.tar'
                with tarfile.open(archive) as tar:body=tar.extractfile('exp/r17_integration/models/fastwam/ops/mps.py').read()
                function=next(x for x in ast.parse(body).body if isinstance(x,ast.FunctionDef) and x.name=='paths')
                scope=dict(Path=Path,hashlib=hashlib,os=os)
                exec(compile(ast.Module(body=[function],type_ignores=[]),'<frozen mps.paths>','exec'),scope)
                pipe=scope['paths'](root,uuid)[1]
                enabled=bool(source.get('mps'))
            else:
                from exp.r17_integration.models.fastwam.ops.mps import paths as native_paths
                pipe=native_paths(root,uuid)[1];enabled=False
            socket=unix_socket_path(root/'servers/replica_0','0'*64)
            rows.append(dict(unit=unit['name'],name=run['name'],gpu=gpu,gpu_uuid=uuid,
                gpu_hash=hashlib.sha256(uuid.encode()).hexdigest()[:6],mps_enabled=enabled,
                pipe=str(pipe),pipe_bytes=len(os.fsencode(pipe)),portable_socket=str(socket),
                socket_bytes=len(os.fsencode(socket)),status='PASS',materialized=root.exists()))
    derived=[]
    for unit in m['units']:
        if unit['pipeline']=='R15':
            for run in unit['runs']:
                cell=run.get('cell')
                if cell:
                    for level in ('off','ir45','ir40','ir32','ir25'):
                        arm=unit['old']+'_'+cell+'_'+level
                        derived.append('/scratch/zixuans8/openpi_trace_h100/os_cl/cfg/manifest_'+run['name']+'_'+arm+'_'+'0'*64+'.json')
    for u in read(q.INDEX)['units']:
        for j in u['jobs']:
            root=Path(j['run']);derived += [str(root),str(root/'servers/replica_0/ready.json')]
            if j['pipeline']=='R15':
                for row in read(root/'arms.json'):
                    derived.append('/data/oscl_h100/runs/'+root.name+'/runs/'+row['arm']+'/server_'+row['arm']+'_23230.owner.json')
                    for island in ('/scratch/zixuans8/openpi_trace','/scratch/zixuans8/openpi_trace_h100'):
                        derived += [island+'/os_cl/cfg/manifest_'+root.name+'_'+row['arm']+'_'+'0'*64+'.json',
                                    island+'/os_cl/cfg/matrix_'+row['arm']+'.yaml',
                                    island+'/os_cl/runs/'+root.name+'/'+row['arm']+'/journal.jsonl']
                    derived.append('/tmp/oscl_sb3_stage/collect_'+root.name+'_'+row['arm']+'.tar')
            if j['pipeline']=='R16':
                derived += [str(root/'per_step'/(e['row']['arm']+'.jsonl')) for e in read(root/'group.json')['entries']]
    maximum=max(len(os.fsencode(part)) for p in derived for part in Path(p).parts)
    assert maximum<=255 and max(len(os.fsencode(p)) for p in derived)<4096
    dump(AUDIT/'PATH_CHECKS.json',dict(rows=rows,max_pipe_bytes=max(r['pipe_bytes'] for r in rows),
         max_socket_bytes=max(r['socket_bytes'] for r in rows),max_derived_component_bytes=maximum,
         max_derived_path_bytes=max(len(os.fsencode(p)) for p in derived),
         limits=dict(mps_pipe=85,unix_socket=103,filename_component=255,absolute_path=4095)))
    print('PATH_PASS',len(rows),'max_pipe',max(r['pipe_bytes'] for r in rows),'max_socket',max(r['socket_bytes'] for r in rows))


def startup_deferred():
    """Verify the full immutable ladder template; dispatch remains WAITING_OFF."""
    from . import formal_a_analysis as a
    templates=q.templates();units=read(q.INDEX)['units'];m=read(n.MAP)
    for unit in units:
        if unit['status']!='WAITING_OFF':continue
        definition=next(u for u in m['units'] if u['name']==unit['name'])
        sources=[j for j in templates if (j['model'],j['mode'],j['arm'])==
                 (unit['model'],unit['mode'],unit['arm'])]
        for run in definition['runs']:
            source=(next(j for j in sources if read(Path(j['run'])/'ROOT.json')['cell']==run['cell'])
                    if definition['pipeline']=='R15' else sources[0])
            job=dict(source,name=run['name'])
            target=AUDIT/'deferred_cpu'/(job['name']+'.json')
            if target.exists() and read(target).get('startup')=='PASS':continue
            native_dump=a.dump
            def output(path,value):
                if Path(path).parent==f.ROOT/'cpu_checks':return dump(target,dict(value,
                    verification='immutable_full_ladder_template',future_run=run['root'],
                    future_dispatch='WAITING_OFF',template=job['run']))
                if Path(path)==f.ROOT/'CPU_CHECKS.json':return
                return native_dump(path,value)
            with patch.object(a,'active_jobs',return_value=[job]),patch.object(a,'dump',output):a.check()
    print('DEFERRED_STARTUP_PASS',len(list((AUDIT/'deferred_cpu').glob('*.json'))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('snapshot','migrate','paths','startup-deferred'))
    a=p.parse_args();globals()[a.action.replace('-','_')]()


if __name__=='__main__':main()
