"""Regenerate B held-out tables with references, writing only under knob/."""
from concurrent.futures import ThreadPoolExecutor
from .build import HERE, run_cpu, CELLS
from exp.offline_search.rounds.r10.data import write_json

def one(cell):
    model,suite,size=cell
    log=HERE/'validation'/f'rebuild_{model}_{suite}_{size}.log'
    run_cpu(['-m','exp.offline_search.rounds.r11.knob.build','rebuild','--model',model,'--suite',suite,'--r10-size',str(size)],log)
    assert f'REBUILD_EXACT {model}_{suite}_{size}' in log.read_text()
    print('REBUILD_EXACT',model,suite,size,flush=True)
    return dict(cell=f'{model}_{suite}_{size}',exact_arrays=True,log=str(log))

if __name__=='__main__':
    with ThreadPoolExecutor(4) as pool: records=list(pool.map(one,CELLS))
    write_json(HERE/'rebuild_audit.json',dict(PASS=True,cells=8,astra_nested_tables_exact=True,opus_anchor_tables_exact=True,records=records))
