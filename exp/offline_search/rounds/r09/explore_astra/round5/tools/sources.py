"""Inventory the static Python import closure; optional coordinator-only install.

Default is local hash inventory. --execute is never part of local verification.
The coordinator install uses the existing fleet lock and refuses differing files.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path

from .safe import HERE, RUN, REPO, dump, sha

STARTS = [
 'exp.offline_search.rounds.r09.explore_astra.round5.tools.methods',
 'exp.offline_search.rounds.r09.explore_fable.round3.tools.methods',
]


def module_path(name):
    for p in (REPO/Path(*name.split('.')).with_suffix('.py'), REPO/Path(*name.split('.'))/'__init__.py'):
        if p.is_file():return p


def closure():
    todo=list(STARTS);seen={}
    while todo:
        name=todo.pop()
        if name in seen or not name.startswith('exp.offline_search'):continue
        p=module_path(name)
        if p is None:continue
        seen[name]=p
        package=name if p.name=='__init__.py' else name.rpartition('.')[0]
        for i in range(1,len(name.split('.'))):
            parent='.'.join(name.split('.')[:i])
            if module_path(parent):todo.append(parent)
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,ast.Import):todo.extend(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom):
                n=node.module or ''
                if node.level:
                    n=importlib.util.resolve_name('.'*node.level+n,package)
                todo.append(n)
                todo.extend(n+'.'+a.name for a in node.names if a.name!='*')
    return [dict(module=n,local=str(p),relative=str(p.relative_to(REPO)),
                 remote=str(Path('/data/oscl_h100/openpi')/p.relative_to(REPO)),sha256=sha(p))
            for n,p in sorted(seen.items())]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    files=closure()
    if not a.execute:
        dump(HERE/'H100_SOURCES.json',files)
        (HERE/'H100_SOURCES.sha256').write_text(''.join(f"{r['sha256']}  {r['relative']}\n" for r in files))
        print(f'{len(files)} source files inventoried; no network action')
        return
    frozen=json.loads((HERE/'H100_SOURCES.json').read_text())
    if files!=frozen:raise ValueError('source inventory differs from freeze')
    from exp.offline_search.closed_loop.ops.h100 import control as c
    with c.fleet_lock(RUN):
        for r in files:
            source=Path(r['local']);digest=r['sha256']
            stage=c.BASE/'runs'/RUN.name/'source_install'/f"{digest}_{source.name}"
            target=c.BASE/'openpi'/r['relative']
            c.remote('h100',['mkdir','-p',stage.parent]);c.push('h100',source,stage)
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
