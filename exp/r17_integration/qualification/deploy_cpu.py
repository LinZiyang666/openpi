"""CPU/file-only P6 payload deployment to new Q-owned SHA destinations."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

from .assets import write_json


def runtime_archive(root, archive):
    prefixes = ['src/openpi', 'packages/openpi-client/src/openpi_client', 'exp/r17_integration/common',
        'exp/r17_integration/models', 'exp/r17_integration/benchmarks', 'exp/r17_integration/conductor',
        'exp/r17_integration/debug', 'exp/r17_integration/search', 'exp/r17_integration/ops',
        'exp/r17_integration/qualification', 'exp/mw_finetune', 'exp/metaworld',
        'exp/offline_search/debug', 'exp/offline_search/closed_loop', 'examples/libero']
    files = {root / 'exp/__init__.py'}
    for prefix in prefixes:
        files.update(p for p in (root / prefix).rglob('*.py') if not set(p.parts) & {'tests', 'scratch', '__pycache__','preflight','receipts'}
            and not p.name.startswith('test_'))
    files.update((root / 'exp/mw_finetune/dp').rglob('RECIPE.md'))
    files.update((root / 'exp/r17_integration/qualification/specs').glob('*.json'))
    files.add(root / 'exp/r17_integration/qualification/worker_deployment.json')
    with tarfile.open(archive, 'w') as tar:
        for path in sorted(files):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--h100-runtime', default='/data/mw_finetune/qualification/runtime_q1')
    p.add_argument('--runtime-only', action='store_true')
    p.add_argument('--local-only', action='store_true',
                   help='refresh local runtime and timan107 only; leave H100/timan108 untouched')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('hide CUDA for deployment')
    root, out = Path.cwd(), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    from exp.r17_integration.ops.build_client_bundle import worker_payload_files, build_client_bundle, sha256
    from exp.r17_integration.ops.deploy import tether_transfer
    report = dict(cpu_only=True, actual_gpu_executed=False, nodes={})
    def execute(argv):
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=75)
        if result.returncode:
            raise RuntimeError(str(argv[:5])+'\n'+result.stdout+'\n'+result.stderr)
        return result.stdout
    if not a.runtime_only:
        files = worker_payload_files(root, debug=True)
        files += ['exp/r17_integration/qualification/'+name+'.py' for name in ('remote_workers','archive_stage')]
        files += ['exp/r17_integration/ops/'+name+'.py' for name in ('build_client_bundle', 'sync_back', 'deploy')]
        payload = build_client_bundle(root, files, out / 'workers.tar')
        digest = payload['archive_sha256']
        report['payload'] = payload
        for node in (('timan107',) if a.local_only else ('timan107', 'timan108')):
            # Re-deployment can reuse a payload SHA, but tether push forbids
            # overwriting the earlier staged archive. Keep each run's stage new.
            stage = '/tmp/r17_Q/'+digest+'/'+out.name
            destination = '/scratch/zixuans8/mw_r17/qualification/payload_'+digest
            execute(['tether','exec','--safe','--timeout','30s',node,'--','mkdir','-p',stage])
            for local, remote in ((out/'workers.tar', stage+'/workers.tar'),
                                  (root/'exp/r17_integration/qualification/bootstrap_install.py', stage+'/install.py')):
                tether_transfer(['tether','push',str(local),node+':'+remote,'--timeout','60s'], lane='tether')
            installed = execute(['tether','exec','--safe','--timeout','60s',node,'--',
                '/scratch/zixuans8/mw_r17/envs/sim/bin/python',stage+'/install.py',stage+'/workers.tar',destination,digest])
            env = ['env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','PYTHONDONTWRITEBYTECODE=1',
                   'PYTHONPATH='+destination+':'+destination+'/src:'+destination+'/packages/openpi-client/src',
                   'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
            probe = execute(['tether','exec','--safe','--timeout','30s',node,'--',*env,
                '/scratch/zixuans8/mw_r17/envs/sim/bin/python','-m','exp.r17_integration.qualification.remote_workers',
                '--endpoint','unused:1','--driver-host','unused','--driver-port','1','--run',stage+'/unused','--import-probe'])
            report['nodes'][node] = dict(destination=destination, stage=stage, installed=json.loads(installed), probe=json.loads(probe))
        write_json(out / 'WORKER_DEPLOYMENT.json', report)
        deployment = root / 'exp/r17_integration/qualification/worker_deployment.json'
        if a.local_only and deployment.exists():
            previous = json.loads(deployment.read_text())
            previous['nodes'].update(report['nodes'])
            previous['local_refresh'] = str(out / 'WORKER_DEPLOYMENT.json')
            write_json(deployment, previous)
        else:
            write_json(deployment, report)
    archive = out / 'runtime.tar'
    files = runtime_archive(root, archive)
    # Hash the bytes actually archived, including any concurrent owner changes.
    with tarfile.open(archive) as tar:
        source_hashes={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()}
    snapshot=root/'exp/r17_integration/qualification/runtime_code_snapshot.json'
    write_json(snapshot,dict(schema='r17.Q.runtime_code.v1',files=source_hashes))
    with tarfile.open(archive,'a') as tar:
        tar.add(snapshot,arcname=str(snapshot.relative_to(root)),recursive=False)
    digest = sha256(archive)
    if a.local_only:
        installed = dict(runtime=str(root), sha256=digest, host='local', remote_runtime_untouched=True)
        write_json(out / 'RUNTIME_DEPLOYMENT.json', dict(cpu_only=True, files=len(files),
            archive_bytes=archive.stat().st_size, installed=installed, source_hashes=source_hashes))
        print(json.dumps(dict(status='PASS', output=str(out), runtime=installed)))
        return
    stage = '/tmp/r17_Q/runtime_'+digest+'/'+out.name
    execute(['tether','exec','--safe','--timeout','30s','h100','--','mkdir','-p',stage])
    tether_transfer(['tether','push',str(archive),'h100:'+stage+'/runtime.tar','--timeout','60s'], lane='tether')
    script = ('import hashlib,tarfile,pathlib,json; p=pathlib.Path('+repr(stage+'/runtime.tar')+'); '
        'assert hashlib.file_digest(p.open("rb"),"sha256").hexdigest()=='+repr(digest)+'; '
        'd=pathlib.Path('+repr(a.h100_runtime)+'); d.mkdir(parents=True,exist_ok=True); '
        't=tarfile.open(p); assert all(m.isfile() and not pathlib.Path(m.name).is_absolute() and ".." not in pathlib.Path(m.name).parts for m in t.getmembers()); '
        't.extractall(d); print(json.dumps({"runtime":str(d),"sha256":'+repr(digest)+'}))')
    installed = execute(['tether','exec','--safe','--timeout','60s','h100','--',
                         '/home/exouser/openpi/.venv/bin/python','-c',script])
    write_json(out / 'RUNTIME_DEPLOYMENT.json', dict(cpu_only=True, files=len(files), archive_bytes=archive.stat().st_size,
               installed=json.loads(installed), source_hashes=source_hashes))
    print(json.dumps(dict(status='PASS', output=str(out), runtime=json.loads(installed))))


if __name__ == '__main__':
    main()
