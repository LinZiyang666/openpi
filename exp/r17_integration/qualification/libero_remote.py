"""CPU OSMesa LIBERO island on weilandserver, also callable from H100 relay."""
from __future__ import annotations
import json
import os
from pathlib import Path
from .run_slot import cpu_environment,Partial


def run_libero(job,run,owned):
    proof=owned.spawn('ws_proof',[os.sys.executable,'-u','-m','exp.r17_integration.qualification.driver',
        '--job',str(run/'job.json'),'--run',str(run),'--ws-only'],cpu_environment(Path(job['runtime']),job))
    if owned.wait(proof,120)!=0: raise RuntimeError('actual two-client WS proof failed')
    protocol=json.loads((Path(job['assets'])/'model_spec.json').read_text())['allowed_benchmark_protocols'][0]
    root='/home/weiland/projects/openpi'
    argv=['env','CUDA_VISIBLE_DEVICES=','MUJOCO_GL=osmesa','PYOPENGL_PLATFORM=osmesa','LIBGL_ALWAYS_SOFTWARE=1',
        'LD_LIBRARY_PATH=/data/mw_finetune/envs/osmesa/usr/lib/x86_64-linux-gnu',
        'PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'PYTHONPATH='+root+':'+root+'/packages/openpi-client/src',
        '/home/weiland/projects/openpi_ext/envs/libero_sim/bin/python','-u','-m','exp.r17_integration.qualification.libero_worker',
        '--model',job['model'],'--protocol',protocol,'--endpoint',job['endpoint_host']+':'+str(job['server_port']),
        '--benchmark-spec',job['coordinator_benchmark_spec'],
        '--run',str(run/'libero')]
    if job['host']=='h100':
        owned.report['remote_owned']=dict(node='weilandserver',remote_run=str(run/'libero'),
            module='exp.r17_integration.qualification.libero_worker',python='/home/weiland/projects/openpi_ext/envs/libero_sim/bin/python')
        owned.report['remote_owned']['run']=owned.report['remote_owned'].pop('remote_run')
        argv=['tether','exec','--safe','--timeout','530s','weilandserver','--',*argv,'--transfer-stage']
    worker=owned.spawn('libero_worker',argv)
    if owned.wait(worker,510)!=0: raise Partial('LIBERO mechanics incomplete; original loop exception retained')
    if job['host']=='h100':
        lines=(run/'libero_worker.log').read_text().splitlines()
        transfers=[json.loads(line.split('Q_LIBERO_ARCHIVE ',1)[1]) for line in lines if line.startswith('Q_LIBERO_ARCHIVE ')]
        if len(transfers)!=1: raise Partial('LIBERO closed archive receipt absent')
        transfer=transfers[0]
        from .archive_stage import stage_identity
        if (transfer.get('stage_sha_verified') is not True or transfer.get('run_key')!=run.name
                or transfer.get('kind')!='libero'
                or stage_identity(transfer['archive'],run.name,'libero')!=transfer['archive_sha256']):
            raise Partial('LIBERO closed archive has no exact run-scoped SHA-verified stage')
        owned.report['libero_transfer']=dict(transfer,node='weilandserver',verified_pull=False,stage_removed=False)
        pull=owned.spawn('libero_pull',['tether','pull','weilandserver:'+transfer['archive'],str(run/'libero.tar'),'--timeout','35s'])
        if owned.wait(pull,125)!=0: raise Partial('LIBERO relay transfer incomplete')
        from exp.r17_integration.ops.sync_back import install_snapshot
        install_snapshot(run/'libero.tar',run/'libero',expected_sha256=transfer['archive_sha256'])
        from .run_slot import cleanup_transfer_stage
        cleanup_transfer_stage(job,run,owned,'weilandserver',transfer)
        owned.report['libero_transfer'].update(verified_pull=True,stage_removed=True)
