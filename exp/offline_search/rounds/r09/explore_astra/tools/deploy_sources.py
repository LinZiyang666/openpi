"""Coordinator utility: install ONLY the two new server modules, never overwrite.

Default is a local print-only plan. --execute performs remote writes and must be
run with the assigned idle worker fleet. No deployment was performed in R9.
"""
import argparse
from pathlib import Path
from .common import HERE,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    root=Path('/home/weiland/trace_runs/os_closed_loop/r09_astra_confirmation')
    relative=Path('exp/offline_search/rounds/r09/explore_astra')
    files=[HERE/'methods.py',HERE/'inference.py']
    for p in files:print(sha(p),p,'->','/data/oscl_h100/openpi'/relative/p.name)
    if not args.execute:return
    from exp.offline_search.closed_loop.ops.h100 import control as c
    with c.fleet_lock(root):
        for p in files:
            digest=sha(p);stage=c.BASE/'runs'/root.name/'source_install'/f'{digest}_{p.name}'
            target=c.BASE/'openpi'/relative/p.name
            c.remote('h100',['mkdir','-p',stage.parent]);c.push('h100',p,stage)
            program='''import hashlib,os,sys
from pathlib import Path
source,target,digest=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
target.parent.mkdir(parents=True,exist_ok=True)
try: os.link(source,target)
except FileExistsError: pass
assert hashlib.sha256(target.read_bytes()).hexdigest()==digest, 'Refuse differing existing source'
print('SOURCE_READY',target,digest)
'''
            print(c.remote('h100',['python3','-c',program,stage,target,digest]))


if __name__=='__main__':main()
