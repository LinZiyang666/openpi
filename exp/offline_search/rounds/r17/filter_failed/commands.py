"""Write exact coordinator prepare/launch commands; execute none of them."""
from __future__ import annotations

from pathlib import Path
import shlex
import shutil
import yaml

from .audit_counts import HERE, REPO, read
from .build import TAG, write
from exp.r17_integration.ops.immutable_release import sha


def generate():
    document=read(HERE/'CAMPAIGNS.json');cells=read(HERE/'CELLS.json')
    directory=HERE/'commands';directory.mkdir(exist_ok=True)
    for cell in cells['cells']:
        if cell['source']!='R10':continue
        for source,artifact in cell['artifacts'].items():
            old=Path(artifact['path'])
            target=Path(cell['original_fit']).with_name(Path(cell['run_fit']).stem+'_'+TAG+'.pkl')
            if old!=target:
                shutil.copyfile(old,target)
                assert sha(old)==sha(target)
                artifact['path']=str(target)
                # Remove only our own newly created artifact from the original
                # run directory; original fits/campaign files are never edited.
                assert old.parent==Path(cell['run_fit']).parent and TAG in old.name
                old.unlink()
    write(HERE/'CELLS.json',cells)
    # Keep the reviewable studies on canonical original size-N descriptors.
    # Physical paths are excluded from recipe/bundle content identities.
    from exp.r17_integration.ops.cache_study import view_path, validate_manifest
    from exp.r17_integration.common.contracts import digest
    digests={}
    for study_path in sorted({j['study'] for j in document['jobs'] if 'study' in j}):
        study=read(study_path)
        for rows in study['models'].values():
            for row in rows:
                path=Path(study_path).parent/row['file']
                config=yaml.safe_load(path.read_text())
                config['recipe']['store_path']=str(view_path(row))
                path.write_text(yaml.safe_dump(config,sort_keys=False));row['sha256']=sha(path)
        study['digest']=digest({k:v for k,v in study.items() if k!='digest'})
        write(study_path,study);validate_manifest(study_path);digests[study_path]=study['digest']
    for job in document['jobs']:
        root=Path(job['run'])
        if job['domain']!='r10_libero':
            campaign=read(root/'campaign.json');source=read(Path(job['source'])/'campaign.json')
            if 'cache_worker_orchestration' in source:
                campaign['cache_worker_orchestration']=source['cache_worker_orchestration']
            write(root/'campaign.json',campaign)
            inputs=read(root/'assets/cache_inputs/manifest.json')
            for group in inputs['groups']:group['num_trials_per_task']=50
            inputs['source_study_digest']=digests[job['study']]
            inputs['digest']=digest({k:v for k,v in inputs.items() if k!='digest'})
            write(root/'assets/cache_inputs/manifest.json',inputs)
            seal=read(root/'assets/CACHE_STUDY.json')
            for group in seal['groups']:group['num_trials_per_task']=50
            seal.update(source_study_digest=digests[job['study']],inputs_digest=inputs['digest'])
            seal['digest']=digest({k:v for k,v in seal.items() if k!='digest'})
            write(root/'assets/CACHE_STUDY.json',seal)
            plan=read(root/'EPISODE_PLAN.json')
            for task in plan['tasks']:task['extra']['num_trials_per_task']=50
            write(root/'EPISODE_PLAN.json',plan)
            prepare=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
                     'PYTHONPATH=.:src',str(REPO/'.venv/bin/python'),'-m',
                     'exp.offline_search.rounds.r17.filter_failed.stage','--run',str(root)]
            launch=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONPATH=.:src',
                    str(REPO/'.venv/bin/python'),'-m','exp.offline_search.rounds.r17.filter_failed.launch','--run',str(root)]
        else:
            original=Path('/home/weiland/trace_runs/os_closed_loop')/f"r10_size_{job['model']}"
            old_arms={a['arm']:a for a in read(original/'arms.json')};arms=read(root/'arms.json')
            for arm in arms:
                old=old_arms[f"r10_{job['model']}_{arm['suite_short']}_{arm['kwargs']['size']}_A"]
                shutil.copyfile(old['yaml'],arm['yaml'])
                arm['yaml_sha256']=sha(arm['yaml']);arm['src_yaml']=old['yaml'];arm['src_sha256']=arm['yaml_sha256']
            write(root/'arms.json',arms)
            prefix=['env','SERVER_HOST=h100','WORKER_HOST='+next(iter(job['workers'])),'PORTS='+job['ports'],
                    'WPS='+str(job['wps']),'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
                    'CUDA_VISIBLE_DEVICES=','PYTHONPATH=.:src',str(REPO/'.venv/bin/python'),'-m',
                    'exp.offline_search.rounds.r17.filter_failed.legacy']
            prepare=[*prefix,'sync',str(root),*job['arms']]
            launch=[*prefix,'chain',str(root),*job['arms']]
        for phase,command in (('prepare',prepare),('launch',launch)):
            path=directory/f"{phase}_{job['name']}.sh"
            path.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\n'+shlex.join(command)+'\n')
            job[phase+'_command']='bash '+shlex.quote(str(path))
    write(HERE/'CAMPAIGNS.json',document)
    for phase in ('prepare','launch'):
        path=directory/(phase+'_all.sh')
        path.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+
                        '\n'.join(job[phase+'_command'] for job in document['jobs'])+'\n')
    (directory/'prepare_and_launch_all.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+
        '\n'.join(command for job in document['jobs'] for command in
                  (job['prepare_command'],job['launch_command']))+'\n')
    print(f"Wrote {len(document['jobs'])} coordinator prepare/launch command pairs.")


if __name__=='__main__':generate()
