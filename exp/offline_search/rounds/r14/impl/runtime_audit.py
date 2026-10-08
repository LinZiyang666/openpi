"""Clean-interpreter serving closure, numerical payloads and per-host SHA lists."""
import json
import pickle
from pathlib import Path
import sys
from .common import *

def main():
    records=json.loads((HERE/'fits.json').read_text())
    payloads={}
    for rec in records:
        path=Path(rec['path']);assert sha(path)==rec['sha256']
        m=pickle.loads(path.read_bytes())['method']
        b=m.inner.base
        from exp.offline_search.rounds.r10.data import IndexedRows
        assert isinstance(b.act,IndexedRows)
        p=Path(b.act.path)
        if str(p) not in payloads:
            payloads[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size,kind='parent_library_actions')
        assert b.act._array is None
    forbidden=('exp.offline_search.rounds.r14.astra','exp.offline_search.rounds.r14.knob.fitting',
               'exp.offline_search.rounds.r10.recipe.fitting','exp.offline_search.rounds.r11.knob.fitting',
               'exp.offline_search.rounds.r11.astra','exp.offline_search.rounds.r11.opus')
    assert not any(n.startswith(forbidden) for n in sys.modules)
    sources={Path(m.__file__).resolve() for n,m in sys.modules.items() if n!='__main__'
        and (getattr(m,'__file__','') or '').endswith('.py')
        and Path(m.__file__).resolve().is_relative_to(REPO/'exp/offline_search')}
    oldlist=ROUNDS/'r11/knob/H100_SOURCES.sha256'
    sources|={REPO/line.split('  ',1)[1] for line in oldlist.read_text().splitlines()}
    sources|={REPO/f'exp/offline_search/closed_loop/{name}.py' for name in
              ('plugin','look_recording','worker_pool','rollout_seed','execution_worker')}
    sources.add(REPO/'exp/offline_search/closed_loop/ops/h100/control.py')
    # common is a build/audit helper, rather than a serving import. Include
    # the coordinator install/validation commands in their own manifest.
    runtime={p for p in sources if not p.is_relative_to(HERE)}
    hashes={str(p.relative_to(REPO)):sha(p) for p in sorted(runtime)}
    for host in ('h100','timan107','timan108'):
        atomic(HERE/(host.upper()+'_SOURCES.sha256'),''.join(f'{v}  {k}\n' for k,v in hashes.items()).encode())
    atomic(HERE/'RUNTIME_SOURCES.sha256',''.join(f'{v}  {k}\n' for k,v in hashes.items()).encode())
    control={str(p.relative_to(REPO)):sha(p) for p in sorted(HERE.glob('*.py'))}
    write_json(HERE/'runtime_closure.json',dict(PASS=True,artifacts=len(records),source_files=hashes,
        coordinator_source_files=control,payloads=payloads,
        fits={r['path']:dict(sha256=r['sha256'],bytes=r['bytes']) for r in records},
        no_fit_or_explorer_imports=True,serving_external_payload='only IndexedRows parent action npy',
        new_serving_files=['exp/offline_search/rounds/r14/knob/__init__.py',
                           'exp/offline_search/rounds/r14/knob/controller.py',
                           'exp/offline_search/rounds/r14/knob/recipe.py']))
    print('RUNTIME_IMPORT_PASS',len(records),len(hashes),len(payloads),flush=True)

if __name__=='__main__':main()
