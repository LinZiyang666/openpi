"""CPU proof: H100 → Q relay → actual deployed timan108 import probe."""
import json
import argparse
import os
from pathlib import Path
import subprocess
from .relay import Broker
from .assets import write_json


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='/data/mw_finetune/qualification/preflight/relay_cpu_v2')
    a=parser.parse_args()
    run=Path(a.run);run.mkdir(parents=True,exist_ok=False)
    broker=Broker(run)
    report=dict(status='FAIL',cpu_only=True,actual_gpu_executed=False)
    try:
        deployment=json.loads(Path('exp/r17_integration/qualification/worker_deployment.json').read_text())
        payload=deployment['nodes']['timan108']['destination']
        args=['exec','--safe','--timeout','25s','timan108','--','env','CUDA_VISIBLE_DEVICES=',
            'PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+payload+':'+payload+'/src:'+payload+'/packages/openpi-client/src',
            '/scratch/zixuans8/mw_r17/envs/sim/bin/python','-m','exp.r17_integration.qualification.remote_workers',
            '--endpoint','unused:1','--driver-host','unused','--driver-port','1','--run','/tmp/r17_Q/qualification_import','--import-probe']
        script=('import urllib.request,json; req=urllib.request.Request('+repr(broker.url)+',data='+repr(json.dumps(args).encode())+',headers={"Authorization":'+repr('Bearer '+broker.token)+'}); '
                'r=json.load(urllib.request.urlopen(req,timeout=40)); assert r["rc"]==0,r; print(r["stdout"])')
        result=subprocess.run(['tether','exec','--safe','--timeout','50s','h100','--',
            '/home/exouser/openpi/.venv/bin/python','-c',script],capture_output=True,text=True,timeout=55)
        assert result.returncode==0,result.stdout+result.stderr
        report.update(status='PASS',remote_probe=json.loads(result.stdout),route='h100 → ziyanglin.com:23158 → timan108')
    except BaseException as exc: report['error']=str(exc)
    finally: broker.close();write_json(Path('exp/r17_integration/qualification/preflight/relay_cpu.json'),report)
    print(json.dumps(report))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
