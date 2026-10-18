"""Dedicated A100: finished pi05 -> DP using the same frozen openpi Python island.

Default is a CPU plan. Reuses one environment; deletes only the completed pi05
study's weights/tokenizer/cache before copying the DP weights. Never two models'
weights at once. Public rsync is serial and capped, as in a100_stage.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from a100_stage import WORK,audit,cleanup,copy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pi-run',required=True);p.add_argument('--output-name',required=True)
    p.add_argument('--port',type=int,default=29800);p.add_argument('--run',action='store_true')
    a=p.parse_args();previous=Path(a.pi_run).resolve()
    if not previous.is_relative_to(WORK/'results') or Path(a.output_name).name!=a.output_name:
        raise ValueError('study-owned paths required')
    print(json.dumps(dict(previous=str(previous),next_model='dp',deployment='longseg_recal1018',
        preserve='one shared openpi env',copy_bytes_approx=920000000,cold_starts=2,run=a.run)),flush=True)
    if not a.run:return
    names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
    if 'A100' not in names or 'H100' in names:raise ValueError('dedicated A100 only')
    subprocess.run(['df','-h','-l'],check=True)
    current=json.loads((previous/'RUN.json').read_text())
    if current.get('status') not in ('COMPLETE','FAILED') or not current.get('mps_cleanup',{}).get('daemon_closed'):
        raise ValueError('finish and close the own pi05 test before any transfer/cleanup')
    if current['pin']['model']!='pi05':raise ValueError('pi05 receipt required')
    code='import importlib.metadata as m,sys,json; print(json.dumps(dict(python=list(sys.version_info[:2]),packages={k:m.version(k) for k in ["torch","transformers","numpy"]})))'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    island=json.loads(subprocess.check_output([str(WORK/'env/bin/python'),'-c',code],env=env,text=True))
    if island!=dict(python=[3,11],packages={'torch':'2.7.1','transformers':'4.53.2','numpy':'1.26.4'}):
        raise ValueError('shared serving island versions changed: '+str(island))
    audit('shared_openpi_island_verified',**island)
    cleanup('pi05',previous/'RUN.json',keep_environment=True)
    for module,dest in [('xr_tools','tools'),('xc_bundles','concurrent/bundles'),
        ('xc_references','concurrent/references'),('xr_rootfs','rootfs/data/mw_finetune'),
        ('xc_all_cache','rootfs/data/mw_finetune')]:copy(module,dest)
    for cold in range(2):
        target=WORK/'results'/(a.output_name+('_cold2' if cold else ''))
        argv=[str(WORK/'env/bin/python'),str(WORK/'tools/run_concurrent.py'),
            '--host','a100','--model','dp','--deployment','longseg_recal1018','--batch-telemetry',
            '--output',str(target),'--port',str(a.port),'--run']
        audit('start_concurrent',cold_start=cold,argv=argv)
        result=subprocess.run(argv)
        audit('concurrent_finished',model='dp',cold_start=cold,returncode=result.returncode)
        subprocess.run([str(WORK/'env/bin/python'),str(WORK/'tools/a100_summarize.py'),str(target),
            '--absolute-budget','0.000345707'],env=env,check=True)
        if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
