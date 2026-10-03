"""Print/optionally install only this owned package after coordinator fleet idle.

This tool is handed back, NOT executed remotely during round2b. Refuse to
overwrite differing files. The round2 inference module is an explicit dependency.
"""
import argparse
from pathlib import Path
from .data import HERE,sha
from .prepare_confirmation import OUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    project=Path('/home/weiland/projects/openpi')
    files=[HERE/'__init__.py',HERE/'methods.py',HERE/'inference.py',
        HERE.parent/'round2/__init__.py',HERE.parent/'round2/inference.py']
    for f in files:print(sha(f),f,'->',Path('/data/oscl_h100/openpi')/f.relative_to(project))
    if not args.execute:return
    from exp.offline_search.closed_loop.ops.h100 import control as c
    with c.fleet_lock(OUT):
        for f in files:
            digest=sha(f);stage=c.BASE/'runs'/OUT.name/'source_install'/f'{digest}_{f.name}'
            target=c.BASE/'openpi'/f.relative_to(project)
            c.remote('h100',['mkdir','-p',stage.parent]);c.push('h100',f,stage)
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
