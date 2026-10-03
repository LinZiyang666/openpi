"""Static source closure and SHA inventory only. No network/deployment operation."""
import ast
import importlib.util
from pathlib import Path
from .data import HERE, REPO, dump, sha


def module_path(name):
    for p in (REPO/Path(*name.split('.')).with_suffix('.py'), REPO/Path(*name.split('.'))/'__init__.py'):
        if p.is_file():
            return p


def closure():
    todo=['exp.offline_search.rounds.r09.explore_astra.round7.tools.methods',
          'exp.offline_search.rounds.r09.explore_fable.round3.tools.methods']
    seen={}
    while todo:
        name=todo.pop()
        if name in seen or not name.startswith('exp.offline_search'):
            continue
        p=module_path(name)
        if p is None:
            continue
        seen[name]=p
        package=name if p.name=='__init__.py' else name.rpartition('.')[0]
        for i in range(1,len(name.split('.'))):
            parent='.'.join(name.split('.')[:i])
            if module_path(parent): todo.append(parent)
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n,ast.Import): todo.extend(a.name for a in n.names)
            elif isinstance(n,ast.ImportFrom):
                mod=n.module or ''
                if n.level: mod=importlib.util.resolve_name('.'*n.level+mod,package)
                todo.append(mod)
                todo.extend(mod+'.'+a.name for a in n.names if a.name!='*')
    return [dict(module=n,local=str(p),relative=str(p.relative_to(REPO)),
                 remote=str(Path('/data/oscl_h100/openpi')/p.relative_to(REPO)),sha256=sha(p),
                 new_round7=p.is_relative_to(HERE)) for n,p in sorted(seen.items())]


def main():
    rows=closure()
    dump(HERE/'H100_SOURCES.json',rows)
    (HERE/'H100_SOURCES.sha256').write_text(''.join(f"{r['sha256']}  {r['relative']}\n" for r in rows))
    (HERE/'H100_NEW_SOURCES.sha256').write_text(''.join(f"{r['sha256']}  {r['relative']}\n" for r in rows if r['new_round7']))
    print(f'{len(rows)} source dependencies; {sum(r["new_round7"] for r in rows)} new files; no network action')


if __name__=='__main__': main()
