"""Real tether allowed-root transfers, with fake CPU workers and no simulator."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

from .archive_stage import RUN_ROOT,stage_archive,file_sha256
from .assets import write_json
from .relay import Broker

RUNTIME='/data/mw_finetune/qualification/runtime_q1'
ROOT='/home/weiland/projects/openpi'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    q=Path(__file__).resolve().parent
    deployment=json.loads((q/'worker_deployment.json').read_text())
    report=dict(status='RUNNING',cpu_only=True,gpu_executed=False,fake_workers=True,results=[])
    write_json(out/'receipt.json',report)
    def execute(argv,timeout=55):
        result=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        return result.stdout
    def environment(root):
        return ['env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','PYTHONDONTWRITEBYTECODE=1',
            'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
            'PYTHONPATH='+root+':'+root+'/src:'+root+'/packages/openpi-client/src']
    def cleanup_argv(node,t):
        root=deployment['nodes'][node]['destination'] if node.startswith('timan') else ROOT
        python='/scratch/zixuans8/mw_r17/envs/sim/bin/python' if node.startswith('timan') else ROOT+'/.venv/bin/python'
        return ['exec','--safe','--timeout','15s',node,'--',*environment(root),python,'-m',
            'exp.r17_integration.qualification.archive_stage','--cleanup',t['archive'],
            '--run-key',t['run_key'],'--kind',t['kind'],'--sha256',t['archive_sha256']]
    def remote_fixture(node,key):
        root=deployment['nodes'][node]['destination'];python='/scratch/zixuans8/mw_r17/envs/sim/bin/python'
        run='/scratch/zixuans8/mw_r17/qualification/preflight/'+key
        code=('import sys; from exp.r17_integration.qualification import remote_workers as r; '
            'r.worker_command=lambda a,p,i:("cpu_"+str(i),[sys.executable,"-c","import time; time.sleep(.05)"]); '
            'raise SystemExit(r.main(["--endpoint","unused:1","--driver-host","unused","--driver-port","1",'
            '"--run",'+repr(run)+',"--seconds","5"]))')
        stdout=execute(['tether','exec','--safe','--timeout','30s',node,'--',*environment(root),python,'-c',code],35)
        (out/(key+'.log')).write_text(stdout)
        t=json.loads(next(line.split('Q_REMOTE_ARCHIVE ',1)[1] for line in stdout.splitlines() if line.startswith('Q_REMOTE_ARCHIVE ')))
        remote=json.loads(next(line.split('Q_REMOTE_RECEIPT ',1)[1] for line in stdout.splitlines() if line.startswith('Q_REMOTE_RECEIPT ')))
        assert remote['status']=='PASS' and all(row['exit_code']==0 for row in remote['pids'])
        return t
    def add(row):
        report['results'].append(row);write_json(out/'receipt.json',report)
    try:
        from exp.r17_integration.ops.sync_back import install_snapshot,snapshot_run
        for node in ('timan107','timan108'):
            key='gpu_fix4_cpu_'+node+'_'+uuid.uuid4().hex[:12]
            t=remote_fixture(node,key);archive=out/(key+'.tar')
            execute(['tether','pull',node+':'+t['archive'],str(archive),'--timeout','35s'],40)
            assert file_sha256(archive)==t['archive_sha256']
            install_snapshot(archive,out/(key+'_installed'),expected_sha256=t['archive_sha256'])
            cleaned=json.loads(execute(['tether',*cleanup_argv(node,t)],20))
            assert cleaned['stage_removed'] is True
            add(dict(status='PASS',lane=node+' → coordinator',transfer=t,verified_p6=True,stage_removed=True,cpu_only=True))
        for node,kind in (('timan108','workers'),('weilandserver','libero')):
            key='gpu_fix4_cpu_relay_'+kind+'_'+uuid.uuid4().hex[:12]
            if kind=='workers':t=remote_fixture(node,key)
            else:
                source=out/key;source.mkdir()
                write_json(source/'manifest.json',dict(cpu_only=True,gpu_executed=False,synthetic=True))
                write_json(source/'receipt.json',dict(status='PASS',cpu_only=True,gpu_executed=False,synthetic=True))
                archive=out/(key+'.tar');native=snapshot_run(source,archive)
                t=dict(native,**stage_archive(archive,key,kind,native['archive_sha256']))
            run=RUN_ROOT/key;run.mkdir(parents=True,exist_ok=False)
            broker=Broker(run)
            try:
                args=['pull',node+':'+t['archive'],str(run/(kind+'.tar')),'--timeout','35s']
                # The request originates on H100 and uses the actual HTTP broker.
                code=('import json,pathlib,urllib.request; '
                    'from exp.r17_integration.ops.sync_back import install_snapshot; '
                    'url='+repr(broker.url)+'; token='+repr(broker.token)+'; '
                    'def_call=None\n'
                    'def call(args):\n'
                    ' r=urllib.request.Request(url,data=json.dumps(args).encode(),headers={"Authorization":"Bearer "+token}); '
                    'v=json.load(urllib.request.urlopen(r,timeout=100)); assert v["rc"]==0,v; return v\n'
                    'v=call('+repr(args)+'); '
                    'install_snapshot('+repr(str(run/(kind+'.tar')))+','+repr(str(run/'verified'))+',expected_sha256='+repr(t['archive_sha256'])+'); '
                    'c=call('+repr(cleanup_argv(node,t))+'); '
                    'receipt=json.loads(c["stdout"]); assert receipt["stage_removed"]; '
                    'print(json.dumps({"status":"PASS","cpu_only":True,"gpu_executed":False,"verified_p6":True,"producer_stage_removed":True,"h100_copy":json.loads(v["stdout"])}))')
                stdout=execute(['tether','exec','--safe','--timeout','140s','h100','--',*environment(RUNTIME),
                    '/home/exouser/openpi/.venv/bin/python','-c',code],145)
                (out/(key+'_h100.log')).write_text(stdout)
                receipt=json.loads(stdout)
                assert receipt['status']=='PASS' and receipt['h100_copy']['incoming_stage_removed']
                add(dict(status='PASS',lane='H100 → coordinator → '+node+' → coordinator → H100',
                    transfer=t,receipt=receipt,verified_p6=True,stage_removed=True,cpu_only=True,broker_run=str(run)))
            finally:broker.close()
        report['status']='PASS'
    except BaseException as exc:
        report.update(status='FAIL',error=type(exc).__name__+': '+str(exc))
    finally:
        write_json(out/'receipt.json',report);write_json(q/'receipts/TRANSFER_CPU.json',report)
    print(json.dumps(dict(status=report['status'],cases=len(report['results']),output=str(out),error=report.get('error'))))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
