"""Refresh current CPU inventories, driver plans and static/source receipts."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .assets import write_json


def collect(host):
    q=Path(__file__).resolve().parent
    rows=[]
    for path in sorted((q/'specs').glob('asset_*.json')):
        job=json.loads(path.read_text())
        if job['host']!=host or job.get('future_checkpoint'):continue
        assets=Path(job['assets'])
        receipt=json.loads((assets/'CPU_RECEIPT.json').read_text())
        spec=json.loads((assets/'model_spec.json').read_text())
        label=json.loads((assets/'SYNTHETIC.json').read_text())
        assert receipt['status']=='PASS' and receipt['actual_gpu_executed'] is False
        assert label['synthetic'] is True and label['real_teacher'] is False
        load=json.loads((assets/'load.json').read_text())
        if job['model']=='dp' and job['domain']=='metaworld':
            from .run_slot import validate_dp_export
            validate_dp_export(job,load)
        files=('model_manifest.json','model_spec.json','load.json','OBSERVATIONS.json','SYNTHETIC.json',
            'synthetic_teacher_store/READY.json','synthetic_fit/READY.json','synthetic_library.pkl.receipt.json')
        rows.append(dict(status='PASS',asset_id=job['asset_id'],host=host,assets=str(assets),
            checkpoint=job['checkpoint'],model_id=spec['model_id'],checkpoint_identity=spec['checkpoint_identity'],
            norm_identity=spec['norm_identity'],cpu_receipt=receipt,synthetic=True,
            file_hashes={name:hashlib.sha256((assets/name).read_bytes()).hexdigest() for name in files},
            job=job))
    return rows


def remote(command, *, timeout=60):
    result=subprocess.run(['tether','exec','--safe','--timeout',str(timeout)+'s','h100','--',*command],
        capture_output=True,text=True,timeout=timeout+5)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    return result.stdout


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true')
    p.add_argument('--host',choices=('local','h100'),default='local');p.add_argument('--output')
    a=p.parse_args()
    if a.collect:print(json.dumps(collect(a.host)));return
    if not a.output:p.error('--output required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    q=Path(__file__).resolve().parent;root=q.parents[2]
    runtime='/data/mw_finetune/qualification/runtime_q1'
    remote_env=['env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','PYTHONDONTWRITEBYTECODE=1',
        'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'PYTHONPATH='+runtime+':'+runtime+'/src:'+runtime+'/packages/openpi-client/src',
        '/home/exouser/openpi/.venv/bin/python']
    assets=collect('local')+json.loads(remote(remote_env+['-m',
        'exp.r17_integration.qualification.refresh_evidence','--collect','--host','h100']))
    write_json(q/'receipts/ASSET_SEALS.json',dict(status='PASS',assets=assets,cpu_only=True))
    plans=[]
    for row in assets:
        job=row['job'];key=job['asset_id'];run=out/key
        if job['domain']=='metaworld':
            command=['-m','exp.r17_integration.qualification.driver','--job',
                str(q/'specs'/('asset_'+key+'.json')),'--run',str(run),'--plan-only']
            if job['host']=='h100':
                remote_q=runtime+'/exp/r17_integration/qualification'
                command[3]=remote_q+'/specs/asset_'+key+'.json'
                result=remote(remote_env+command)
                manifest=json.loads(remote(remote_env+['-c',
                    'import pathlib;print(pathlib.Path('+repr(str(run/'episode_manifest.json'))+').read_text())']))
            else:
                result=subprocess.run([str(root/'.venv/bin/python'),*command],check=True,capture_output=True,text=True,timeout=30).stdout
                manifest=json.loads((run/'episode_manifest.json').read_text())
            assert manifest['expected_episodes']==2
            plans.append(dict(asset_id=key,status='PASS',run=str(run),host=job['host'],tasks=len(manifest['tasks'])))
        else:
            env=dict(os.environ,MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LIBGL_ALWAYS_SOFTWARE='1',
                LD_LIBRARY_PATH='/data/mw_finetune/envs/osmesa/usr/lib/x86_64-linux-gnu')
            result=subprocess.run(['/home/weiland/projects/openpi_ext/envs/libero_sim/bin/python',str(q/'libero_worker.py'),
                '--endpoint','unused:1','--run',str(run),'--model',job['model'],'--protocol',
                json.loads((Path(row['assets'])/'model_spec.json').read_text())['allowed_benchmark_protocols'][0]
                    if job['host']=='local' else json.loads((q/'specs/libero_benchmark_spec.json').read_text())['protocol_identity'],
                '--benchmark-spec',str(q/'specs/libero_benchmark_spec.json'),'--plan-only'],
                env=env,check=True,capture_output=True,text=True,timeout=30).stdout
            receipt=json.loads((run/'receipt.json').read_text());assert receipt['status']=='PASS' and not receipt['actual_simulator']
            plans.append(dict(asset_id=key,status='PASS',run=str(run),host='local_cpu_osmesa_plan',actual_simulator=False))
        (out/(key+'.log')).write_text(result)
    write_json(q/'receipts/DRIVER_PLANS.json',dict(status='PASS',plans=plans,cpu_only=True))
    py=list(q.glob('*.py'));shell=list(q.glob('slot_*.sh'))
    for path in py:ast.parse(path.read_text(),filename=str(path))
    for path in shell:subprocess.run(['bash','-n',str(path)],check=True,timeout=5)
    snapshot=json.loads((q/'runtime_code_snapshot.json').read_text())
    changed=[name for name,sha in snapshot['files'].items() if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha]
    write_json(q/'receipts/STATIC.json',dict(status='PASS',python_files=len(py),shells=len(shell),cpu_only=True))
    write_json(q/'receipts/SOURCE_SNAPSHOT.json',dict(status='PASS' if not changed else 'PARTIAL',files=len(snapshot['files']),changed=changed))
    print(json.dumps(dict(status='PASS' if not changed else 'PARTIAL',assets=len(assets),plans=len(plans),changed=changed)))


if __name__=='__main__':main()
