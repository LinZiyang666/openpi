"""A100-only, one-model transfer plan and owned-file cleanup. No GPU launch.

Run local /proc/pressure/io inspection immediately before --run. Every transfer
uses the authorized read-only public daemon, serially, capped at 16 MiB/s.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

WORK=Path('/home/exouser/xgpu_1017')
PLANS={
    'pi05':dict(python='python',version='3.11',modules=[
        ('xc_pi_data','rootfs/data'),('xc_pi_site_complete','env/lib/python3.11/site-packages'),
        ('xc_pi_patches','env/lib/python3.11/site-packages'),('xc_pi_tokenizer','tokenizer/big_vision'),
        ('xc_mt50_inputs','inputs_mt50_real3')],minimum_free_gib=16,
        aliases={'/data/ckpt':'rootfs/data/ckpt'}),
    'groot':dict(python='python',version='3.11',modules=[
        ('xc_gr_data','rootfs/data'),('xc_gr_site','env/lib/python3.11/site-packages'),
        ('xc_gr_vendor','vendor/gr00t_n15')],minimum_free_gib=15,
        aliases={'/home/exouser/gr00t_n15':'vendor/gr00t_n15'}),
    'fastwam':dict(python='python312',version='3.12',python_module='xc_python312',modules=[
        ('xc_fw_data','rootfs/data'),('xc_fw_site','env/lib/python3.12/site-packages')],minimum_free_gib=33,
        aliases={'/data/xwam':'rootfs/data/xwam'}),
    'cosmos':dict(python='python310',version='3.10',python_module='xc_python310',modules=[
        ('xc_co_data','rootfs/data'),('xc_co_vendor','rootfs/data/cosmos/cosmos-policy'),
        ('xc_co_site','env/lib/python3.10/site-packages'),('xc_co_extra','env/lib/python3.10/site-packages'),
        ('xc_cosmos_inputs','inputs_cosmos3')],minimum_free_gib=28,
        aliases={'/data/cosmos':'rootfs/data/cosmos','/data/cosmos/cosmos-policy/.venv':'env'}),
}
CLEAN={
    'pi05':['rootfs/data/ckpt/pi05_metaworld_rlinf','env','tokenizer','model_cache'],
    'groot':['rootfs/data/mw_finetune/groot/4090_mt50/checkpoint-60000','env','vendor/gr00t_n15','model_cache'],
    'fastwam':['rootfs/data/mw_finetune/models/fastwam_libero','rootfs/data/xwam','env','model_cache'],
    'cosmos':['rootfs/data/mw_finetune/models/cosmos_libero','rootfs/data/cosmos','env','model_cache'],
}


def audit(event,**kw):
    r=dict(time=time.time(),event=event,**kw)
    with (WORK/'A100_STAGE_AUDIT.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)


def command(argv):
    audit('command',argv=argv)
    subprocess.run(argv,check=True)


def copy(module,dest):
    path=WORK/dest;path.mkdir(parents=True,exist_ok=True)
    command(['rsync','-a','--stats','--bwlimit=16384',
        'rsync://ziyanglin.com:23177/'+module+'/',str(path)+'/'])


def cleanup(model,receipt,keep_environment=False):
    # Never delete arbitrary paths from a remote receipt.
    receipt=Path(receipt).resolve(strict=True)
    if not receipt.is_relative_to(WORK/'results'):raise ValueError('own study receipt required')
    report=json.loads(receipt.read_text())
    if report['status'] not in ('COMPLETE','FAILED') or report['pin']['model']!=model:
        raise ValueError('test must have finished for the selected model')
    if not report.get('mps_cleanup',{}).get('daemon_closed'):
        raise ValueError('owned GPU residents/MPS must be closed first')
    if keep_environment and model!='pi05':raise ValueError('only the shared pi05 -> DP serving island may be reused')
    for rel in CLEAN[model]:
        if keep_environment and rel=='env':
            audit('retain_shared_openpi_environment',after_test=str(receipt),next_model='dp')
            continue
        path=WORK/rel
        if not path.exists():continue
        if path.is_symlink() or not path.resolve().is_relative_to(WORK):
            raise ValueError('refuse deleting symlink or nonowned tree')
        size=int(subprocess.check_output(['du','-sb',str(path)]).split()[0])
        audit('delete_created_model_tree',model=model,path=str(path),bytes=size,after_test=str(receipt))
        shutil.rmtree(path)


def stage(model):
    cfg=PLANS[model]
    command(['df','-h','-l'])
    if (WORK/'env').exists():raise ValueError('finish current model, analyse, then explicitly cleanup its env')
    free=shutil.disk_usage(WORK).free
    if free<cfg['minimum_free_gib']*2**30:raise RuntimeError('insufficient disk; no model transfer started')
    if cfg.get('python_module'):copy(cfg['python_module'],cfg['python'])
    command([str(WORK/cfg['python']/'bin'/('python'+cfg['version'])),'-m','venv','--without-pip',str(WORK/'env')])
    for module,dest in cfg['modules']:copy(module,dest)
    for alias,relative in cfg['aliases'].items():
        target=WORK/relative;link=Path(alias)
        if link.is_symlink() and link.resolve()==target.resolve():continue
        if link.exists() or link.is_symlink():raise ValueError('canonical path belongs to someone else: '+alias)
        link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(target)
        audit('owned_alias',alias=alias,target=str(target))
    command(['du','-sb',str(WORK)])
    audit('stage_complete',model=model,free_bytes=shutil.disk_usage(WORK).free)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['stage','cleanup']);p.add_argument('--model',choices=list(CLEAN),required=True)
    p.add_argument('--receipt');p.add_argument('--run',action='store_true')
    a=p.parse_args()
    print(json.dumps(dict(mode=a.mode,model=a.model,plan=PLANS.get(a.model),cleanup_paths=CLEAN[a.model],run=a.run)))
    if a.run:
        # Fixed root plus GPU identity prevents accidental local or H100 cleanup.
        names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
        if not WORK.is_dir() or 'A100' not in names or 'H100' in names:raise RuntimeError('dedicated A100 only')
        if a.mode=='cleanup':cleanup(a.model,a.receipt)
        else:stage(a.model)
