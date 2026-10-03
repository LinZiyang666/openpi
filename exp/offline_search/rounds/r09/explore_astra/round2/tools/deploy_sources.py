"""Print/install only new round2 serving modules; refuse differing remote files."""
import argparse
from pathlib import Path
from .data import HERE, sha
from .prepare_confirmation import OUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    relative=Path('exp/offline_search/rounds/r09/explore_astra/round2')
    files=[HERE/'__init__.py',HERE/'methods.py',HERE/'inference.py']
    for file in files: print(sha(file),file,'->',Path('/data/oscl_h100/openpi')/relative/file.name)
    if not args.execute:return
    from exp.offline_search.closed_loop.ops.h100 import control as c
    with c.fleet_lock(OUT):
        for file in files:
            digest=sha(file);stage=c.BASE/'runs'/OUT.name/'source_install'/f'{digest}_{file.name}'
            target=c.BASE/'openpi'/relative/file.name
            c.remote('h100',['mkdir','-p',stage.parent]);c.push('h100',file,stage)
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
